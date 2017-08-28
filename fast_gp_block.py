from __future__ import division
import numpy as np
from numpy.linalg import solve
from scipy.sparse.linalg import LinearOperator, cg
from blcgrq import block_cg

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft, ifft
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import fft, ifft


regularizer = 1e-8


def sparse_w_cubic(u, x):
    """
    Cubic convolutional interpolation

    Implementation of Keys, R. G. (1981). Cubic convolution interpolation
    for digital image processing. Acoustics, Speech and Signal Processing,
    IEEE Transactions on, 29(6), 1153-1160.

    NOTE: inducing points u must be sorted.
    """

    idx = np.searchsorted(u, x).astype(np.int32)

    # Avoid indices out of bounds
    idx_lbound = 2
    idx_ubound = len(u) - 2
    idx[idx < idx_lbound] = idx_lbound
    idx[idx > idx_ubound] = idx_ubound

    idx0 = idx - 2
    idx1 = idx - 1
    idx2 = idx
    idx3 = idx + 1
    idx = np.hstack((idx0[:, np.newaxis],
                     idx1[:, np.newaxis],
                     idx2[:, np.newaxis],
                     idx3[:, np.newaxis]))
    unit = u[1] - u[0]
    s = np.fabs(x[:, np.newaxis] - u[idx]) / unit
    w = np.zeros_like(s)

    # Parameter for cubic interpolation

    cond1 = s <= 1
    s1 = s[cond1]
    w[cond1] = 1.5 * s1**3 - 2.5 * s1**2 + 1

    cond2 = np.logical_and(1 < s, s < 2)
    s2 = s[cond2]
    w[cond2] = -0.5 * s2**3 + 2.5 * s2**2 - 4 * s2 + 2

    # Normalize for boundary cases
    w = w / w.sum(axis=1)[:, np.newaxis]
    return idx, w


def sparse_w_linear(u, x):
    """
    Linear interpolation
    Assume u and x are sorted
           x[0] > u[0], x[-1] < u[-1]
    """
    idx1 = np.searchsorted(u, x).astype(np.int32)
    # TODO: check if idx < 0
    idx0 = idx1 - 1
    idx = np.hstack((idx0[:, np.newaxis],
                     idx1[:, np.newaxis]))
    w0 = x - u[idx0]
    w1 = u[idx1] - x
    w_sum = w0 + w1
    w = np.hstack(((w0 / w_sum)[:, np.newaxis],
                   (w1 / w_sum)[:, np.newaxis]))
    return idx, w


sparse_w = sparse_w_cubic
#sparse_w = sparse_w_linear


def idx_w_to_matrix(idx, w, len_u):
    n = idx.shape[0]
    w_mat = np.zeros((n, len_u))
    for i in xrange(n):
        w_mat[i, idx[i]] = w[i]
    return w_mat


def cov_fn(x, y, params):
    """
    Squared exponential kernel for 1-D inputs
    x: 1-D vector
    y: 1-D vector or a scalar
    """
    a, b = params
    if np.isscalar(y):
        K = a * np.exp(-b * (x - y)**2)
    else:
        K = a * np.exp(-b * (x[:, np.newaxis] - y)**2)
    return K


def covmat_col0(u, cov_params):
    return cov_fn(u, u[0], cov_params)


def interp_from_u(idx, w, y):
    """
    compute Wy
    W.shape: (n, u)
    y.shape: (u,)
    """
    return (y[idx] * w).sum(axis=1)


def interp_from_u_block(idx, w, y):
    """
    compute Wy
    W.shape: (n, u)
    y.shape: (u,) or (m, u)
    """
    return (y[:, idx] * w).sum(axis=-1)


try:
    from native_interp_to_u import interp_to_u, interp_to_u_block
except ImportError:
    def interp_to_u(idx, w, len_u, y):
        """
        compute W'y
        W.shape: (n, u)
        y.shape: (n,)
        """
        n = len(y)
        s = np.zeros(len_u)
        for i in xrange(n):
            s[idx[i]] += y[i] * w[i]
        return s


    def interp_to_u_block(idx, w, len_u, y):
        """
        y.shape: (m, n)
        """
        m, n = y.shape
        s = np.zeros((m, len_u))
        i = n - 1
        for i in xrange(n):
            s[:, idx[i]] += np.outer(y[:, i], w[i])
        return s


def toeplitz_matvec(col0, vec, n):
    """
    col0: first column of the Toeplitz matrix with shape (n,)
    vec: vector with shape (n,)
    """
    p = (ifft(fft(np.r_[col0, col0[-2:0:-1]]) *
              fft(np.r_[vec, np.zeros(n - 2)])).real)[:n]
    return p


def toeplitz_matvec_block(col0, vec, n):
    """
    col0: first column of the Toeplitz matrix with shape (n,)
    vec: vector with shape (m, n)
    """
    m, n = vec.shape
    padded_vec = np.zeros((m, n * 2 - 2))
    padded_vec[:, :n] = vec
    p = ifft(fft(np.r_[col0, col0[-2:0:-1]]) * fft(padded_vec)).real[:, :n]
    return p


def Ky(idx, w, u, len_u, indep_noise, y):
    """
    compute K(idx, w, u, sigma2) y
    """
    v1 = interp_to_u(idx, w, len_u, y)
    v2 = toeplitz_matvec(u, v1, len_u)
    v3 = interp_from_u(idx, w, v2)
    return v3 + indep_noise * y


def Ky_block(idx, w, u, len_u, indep_noise, y):
    """
    compute K(idx, w, u, sigma2) y
    """
    v1 = interp_to_u_block(idx, w, len_u, y)
    v2 = toeplitz_matvec_block(u, v1, len_u)
    v3 = interp_from_u_block(idx, w, v2)
    return v3 + indep_noise * y


def cond(idx, w, u, len_u, indep_noise):
    n = w.shape[0]
    I = np.eye(n)
    K = np.empty((n, n))
    for i in xrange(n):
        K[i] = Ky(idx, w, u, len_u, indep_noise, I[i])
    return np.linalg.cond(K)


def Kinv_y(idx, w, u, len_u, indep_noise, y):
    """
    compute K(idx, w, u, sigma2)^{-1} y
    """
    len_test = len(y)
    K = LinearOperator((len_test, len_test),
                       matvec=lambda x: Ky(idx, w, u, len_u, indep_noise, x))
    #print cg(K, y, maxiter=100, tol=1e-12)[1]
    #invK_y = cg(K, y, maxiter=20, tol=1e-12)[0]
    invK_y = cg(K, y, maxiter=50, tol=1e-10)[0]
    return invK_y


def Kinv_y_block(idx, w, u, len_u, indep_noise, y):
    """
    compute K(idx, w, u, sigma2)^{-1} y
    """
    def mat_vec(v):
        return Ky_block(idx, w, u, len_u, indep_noise, v.T).T
    invK_y = block_cg(mat_vec, y.T).T
    return invK_y


def post_cov_y(idx_train, w_train, idx_test, w_test,
               u, len_u, indep_noise, y):
    v1 = interp_to_u(idx_test, w_test, len_u, y[0])
    v2 = toeplitz_matvec(u, v1, len_u)
    v3 = interp_from_u(idx_train, w_train, v2)
    v4 = Kinv_y(idx_train, w_train, u, len_u, indep_noise, v3)
    v5 = interp_to_u(idx_train, w_train, len_u, v4)
    v6 = toeplitz_matvec(u, v5, len_u)
    f = interp_from_u(idx_test, w_test, v2 - v6)
    #f = f + regularizer * y
    return f[np.newaxis]


def post_cov_ys(idx_train, w_train, idx_test, w_test,
                u, len_u, indep_noise, y):
    v1 = interp_to_u_block(idx_test, w_test, len_u, y)
    v2 = toeplitz_matvec_block(u, v1, len_u)
    v3 = interp_from_u_block(idx_train, w_train, v2)
    v4 = Kinv_y_block(idx_train, w_train, u, len_u, indep_noise, v3)
    v5 = interp_to_u_block(idx_train, w_train, len_u, v4)
    v6 = toeplitz_matvec_block(u, v5, len_u)
    f = interp_from_u_block(idx_test, w_test, v2 - v6)
    #f = f + regularizer * y
    return f


def grad_cov_y(idx_train, w_train, idx_test, w_test,
               u, gp_params, indep_noise, y, grad_u, gz):
    """
    y.shape: (1, #test)
    grad_u.shape: (#u, #params)
    gz.shape: (1, #test)
    """
    gz0 = gz[0]
    n_params = grad_u.shape[1]
    len_u = len(u)

    grad_gp_params = np.zeros_like(gp_params)

    v1 = interp_to_u(idx_test, w_test, len_u, y[0])
    v3 = toeplitz_matvec(u, v1, len_u)
    v4_1 = interp_from_u(idx_train, w_train, v3)
    v4 = Kinv_y(idx_train, w_train, u, len_u, indep_noise, v4_1)
    v5_1 = interp_to_u(idx_train, w_train, len_u, v4)

    for p in xrange(n_params):
        grad_up = grad_u[:, p]

        v2 = toeplitz_matvec(grad_up, v1, len_u)
        v5 = toeplitz_matvec(grad_up, v5_1, len_u)
        v6_1 = interp_from_u(idx_train, w_train, v5 - v2)
        v6_2 = Kinv_y(idx_train, w_train, u, len_u, indep_noise, v6_1)
        v6_3 = interp_to_u(idx_train, w_train, len_u, v6_2)
        v6 = toeplitz_matvec(u, v6_3, len_u)
        g = interp_from_u(idx_test, w_test, v2 - v5 + v6)
        grad_gp_params[p] += g.dot(gz0)

    h1 = Kinv_y(idx_train, w_train, u, len_u, indep_noise, v4)
    h2 = interp_to_u(idx_train, w_train, len_u, h1)
    h3 = toeplitz_matvec(u, h2, len_u)
    h = interp_from_u(idx_test, w_test, h3)
    grad_noise = h.dot(gz0)

    grad_y = post_cov_y(idx_train, w_train, idx_test, w_test, u, len_u,
                        indep_noise, gz)

    return grad_gp_params, np.asarray(grad_noise), grad_y


def grad_cov_ys(idx_train, w_train, idx_test, w_test,
                u, gp_params, indep_noise, ys, grad_u, gz):
    """
    ys.shape: (#z, #test)
    grad_u.shape: (#u, #params)
    gz.shape: (#z, #test)
    """
    n_params = grad_u.shape[1]
    len_u = len(u)

    grad_gp_params = np.zeros_like(gp_params)
    grad_noise = 0

    v1 = interp_to_u_block(idx_test, w_test, len_u, ys)
    v3 = toeplitz_matvec_block(u, v1, len_u)
    v4_1 = interp_from_u_block(idx_train, w_train, v3)
    v4 = Kinv_y_block(idx_train, w_train, u, len_u, indep_noise, v4_1)
    v5_1 = interp_to_u_block(idx_train, w_train, len_u, v4)

    for p in xrange(n_params):
        grad_up = grad_u[:, p]

        v2 = toeplitz_matvec_block(grad_up, v1, len_u)
        v5 = toeplitz_matvec_block(grad_up, v5_1, len_u)
        v6_1 = interp_from_u_block(idx_train, w_train, v5 - v2)
        v6_2 = Kinv_y_block(idx_train, w_train, u, len_u, indep_noise, v6_1)
        v6_3 = interp_to_u_block(idx_train, w_train, len_u, v6_2)
        v6 = toeplitz_matvec_block(u, v6_3, len_u)
        g = interp_from_u_block(idx_test, w_test, v2 - v5 + v6)
        grad_gp_params[p] = (g * gz).sum()

    h1 = Kinv_y_block(idx_train, w_train, u, len_u, indep_noise, v4)
    h2 = interp_to_u_block(idx_train, w_train, len_u, h1)
    h3 = toeplitz_matvec_block(u, h2, len_u)
    h = interp_from_u_block(idx_test, w_test, h3)
    grad_noise = (h * gz).sum()

    grad_ys = post_cov_ys(idx_train, w_train, idx_test, w_test, u, len_u,
                          indep_noise, gz)

    return grad_gp_params, np.asarray(grad_noise), grad_ys


def post_mean(idx_train, w_train, idx_test, w_test,
              u, len_u, indep_noise, y):
    #print 'condition number', cond(idx_train, w_train, u, len_u, indep_noise)
    v1 = Kinv_y(idx_train, w_train, u, len_u, indep_noise, y)
    v2 = interp_to_u(idx_train, w_train, len_u, v1)
    v3 = toeplitz_matvec(u, v2, len_u)
    f = interp_from_u(idx_test, w_test, v3)
    return f


def grad_post_mean(idx_train, w_train, idx_test, w_test,
                   u, gp_params, indep_noise, y, grad_u, gz):
    # grad_u.shape: (#u, #gp_params)
    n_gp_params = len(gp_params)
    len_u = len(u)

    v1 = Kinv_y(idx_train, w_train, u, len_u, indep_noise, y)
    v2_1 = interp_to_u(idx_train, w_train, len_u, v1)

    grad_gp_params = np.empty_like(gp_params)
    for p in xrange(n_gp_params):
        v2 = toeplitz_matvec(grad_u[:, p], v2_1, len_u)
        v3_1 = interp_from_u(idx_train, w_train, v2)
        v3_2 = Kinv_y(idx_train, w_train, u, len_u, indep_noise, v3_1)
        v3_3 = interp_to_u(idx_train, w_train, len_u, v3_2)
        v3 = toeplitz_matvec(u, v3_3, len_u)
        g = interp_from_u(idx_test, w_test, v2 - v3)
        grad_gp_params[p] = g.dot(gz)

    h1 = Kinv_y(idx_train, w_train, u, len_u, indep_noise, v1)
    h2 = interp_to_u(idx_train, w_train, len_u, h1)
    h3 = toeplitz_matvec(u, h2, len_u)
    h = -interp_from_u(idx_test, w_test, h3)
    grad_noise = h.dot(gz)

    return grad_gp_params, np.asarray(grad_noise)


def post_mean_sor(x_train, x_test, x_inducing, params, indep_noise, y):
    Kzu = cov_fn(x_test, x_inducing, params)
    Kuu = cov_fn(x_inducing, x_inducing, params)
    Kux = cov_fn(x_inducing, x_train, params)
    mu = Kzu.dot(solve(Kux.dot(Kux.T) + indep_noise * Kuu, Kux.dot(y)))
    #print '-=' * 30
    #print 'Ky sor'
    #print (Kux.dot(Kux.T) + indep_noise * Kuu).dot(Kux.dot(y))
    #print '-=' * 30
    #print '#2'
    #print solve(Kux.dot(Kux.T) + indep_noise * Kuu, Kux.dot(y))
    return mu


def post_mean_exact(x_train, x_test, params, indep_noise, y):
    n = x_train.shape[0]
    Kzx = cov_fn(x_test, x_train, params)
    Kxx = cov_fn(x_train, x_train, params)
    mu = Kzx.dot(solve(Kxx + indep_noise * np.eye(n), y))
    return mu


def test_w():
    idx = np.array([[0, 1], [3, 4], [5, 6]])
    w = np.array([[.1, .9], [.3, .7], [.2, .8]])
    print idx
    print w
    y = np.array([5, 6, 7], dtype=np.float)
    #y = np.vstack((y, y))
    z = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float)
    z = np.vstack((z, z))
    #print y
    #print interp_to_u(idx, w, 8, y)
    print z
    print interp_from_u(idx, w, z)


def test_idx_w():
    u = np.arange(10) + 1
    x = np.array([1.2, 5.6, 8.7])
    idx, w = sparse_w(u, x)
    print 'u'
    print u
    print 'x'
    print x
    print 'idx'
    print idx
    print 'w'
    print w
    idx, w = sparse_w(u, u[1:])
    print 'idx'
    print idx
    print 'w'
    print w


def test_u():
    x = np.arange(6)
    cov_params = (1, .1)
    u = covmat_col0(x, cov_params)
    print u


def test_gp():
    np.random.seed(0)
    n_data = 10
    x = np.random.uniform(size=n_data)
    #x = np.float32(x)
    x = np.sort(x)
    a = .1
    b = 20
    c = .001
    mu = np.zeros(n_data)
    cov = a * np.exp(-b * (x[:, np.newaxis] - x)**2) + c * np.eye(n_data)
    y = np.random.multivariate_normal(mu, cov)
    #print x
    #print y
    x_min, x_max = x.min(), x.max()
    #len_u = 2048 + 1
    #len_u = 1024 + 1
    len_u = 50
    extra_u = 2
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)
    x_test = u[1:]
    #x_test = u
    #x_test = u[2:-1]
    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)

    ku = covmat_col0(u, (a, b))
    #print 'ku'
    #print ku
    #print (cov_fn(u, u, (a, b)) + np.eye(len_u))[:, 0]
    len_test = len(x_test)
    import time

    t1 = time.time()
    mu = post_mean(idx_train, w_train, idx_test, w_test, ku, len_u, c, y)
    var = post_cov_ys(idx_train, w_train, idx_test, w_test,
                      ku, len_u, c, np.eye(len_test))
    t2 = time.time()
    print t2 - t1
    var = np.diag(var)

    #print var
    #return

    import pylab as pl
    pl.figure()
    std2 = np.sqrt(var) * 2
    color = 'b'
    pl.fill_between(x_test, mu - std2, mu + std2, color=color,
                    edgecolor='none', alpha=.3)
    pl.plot(x_test, mu, '-', c=color)
    pl.plot(x, y, 'o', c=color)
    pl.show()


def main():
    #test_w()
    #test_idx_w()
    #test_u()
    test_gp()


if __name__ == '__main__':
    main()
