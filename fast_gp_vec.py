from __future__ import division
import numpy as np
from numpy.fft import fft, ifft
from numpy.linalg import solve
from scipy.sparse.linalg import LinearOperator, cg
import scipy


def sparse_w(u, x):
    """
    Linear interpolation
    Assume u and x are sorted
           x[0] > u[0], x[-1] < u[-1]
    """
    idx1 = np.searchsorted(u, x)
    idx0 = idx1 - 1
    idx = np.hstack((idx0[:, np.newaxis],
                     idx1[:, np.newaxis]))
    w0 = x - u[idx0]
    w1 = u[idx1] - x
    w_sum = w0 + w1
    w = np.hstack(((w0 / w_sum)[:, np.newaxis],
                   (w1 / w_sum)[:, np.newaxis]))
    return idx, w


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


class FastGP(object):
    def __init__(self, x_train, x_test, inducing_points):
        self.x_train = np.sort(x_train)
        self.x_test = np.sort(x_test)
        self.u = inducing_points
        # idx.shape: (n, 2) dtype=int
        # w.shape: (n, 2)
        idx_train, w_train = sparse_w(self.u, self.x_train)
        idx_test, w_test = sparse_w(self.u, self.x_test)


def interp_from_u(idx, w, y):
    """
    compute Wy
    W.shape: (n, u)
    y.shape: (u,) or (m, u)
    """
    return (y[..., idx] * w).sum(axis=-1)


try:
    from native_interp_to_u import interp_to_u
except ImportError:
    # FIXME: to handle stacked y
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


# FIXME: argument 'n' is not needed
def toeplitz_matvec(col0, vec, n):
    """
    col0: first column of the Toeplitz matrix with shape (n,)
    vec: vector with shape (m, n)
    """
    print 'vec'
    print vec.shape
    m, n = vec.shape
    padded_vec = np.zeros((m, n * 2 - 2))
    padded_vec[:, :n] = vec
    p = ifft(fft(np.r_[col0, col0[-2:0:-1]]) * fft(padded_vec)).real[..., :n]
    return p


def mean_Ky(idx, w, u, len_u, sigma2, y):
    """
    compute K(idx, w, u, sigma2) y
    """
    #print 'y'
    #print y
    if y.ndim == 1:
        y = y[np.newaxis]
    Kuu_y = toeplitz_matvec(u, y, len_u)
    #print Kuu_y
    Wx_y1 = interp_from_u(idx, w, Kuu_y)
    WxT_y2 = interp_to_u(idx, w, len_u, Wx_y1)
    Kuu_y3 = toeplitz_matvec(u, WxT_y2, len_u)
    K_y = sigma2 * Kuu_y + Kuu_y3
    return K_y


def cov_Ky(idx, w, u, len_u, sigma2, y):
    """
    compute K(idx, w, u, sigma2) y
    """
    #print 'y'
    #print y
    Kuu_y = toeplitz_matvec(u, y, len_u)
    #print Kuu_y
    Wx_y1 = interp_from_u(idx, w, Kuu_y)
    WxT_y2 = interp_to_u(idx, w, len_u, Wx_y1)
    Kuu_y3 = toeplitz_matvec(u, WxT_y2, len_u)
    K_y = Kuu_y + Kuu_y3 / sigma2
    return K_y


def mean_Kinv_y(idx, w, u, len_u, sigma2, y):
    """
    compute K(idx, w, u, sigma2)^{-1} y
    """

    K = LinearOperator((len_u, len_u),
                       matvec=lambda x: mean_Ky(idx, w, u, len_u, sigma2, x))
    invK_y = np.empty(y.shape)
    for i, each_y in enumerate(y):
        invK_y[i] = cg(K, each_y[np.newaxis], maxiter=20)[0]
    return invK_y


def cov_Kinv_y(idx, w, u, len_u, sigma2, y):
    """
    compute K(idx, w, u, sigma2)^{-1} y
    """

    K = LinearOperator((len_u, len_u),
                       matvec=lambda x: cov_Ky(idx, w, u, len_u, sigma2, x))
    invK_y = np.empty(y.shape)
    for i, each_y in enumerate(y):
        invK_y[i] = cg(K, each_y, maxiter=20)[0]
    return invK_y


def post_cov_y(idx_train, w_train, idx_test, w_test, u, len_u, sigma2, y):
    WsT_y = interp_to_u(idx_test, w_test, len_u, y)
    Kuu_y1 = toeplitz_matvec(u, WsT_y, len_u)
    Kinv_y2 = cov_Kinv_y(idx_train, w_train, u, len_u, sigma2, Kuu_y1)
    Kuu_y3 = toeplitz_matvec(u, Kinv_y2, len_u)
    Ws_y4 = interp_from_u(idx_test, w_test, Kuu_y3)
    return Ws_y4


def post_mean(idx_train, w_train, idx_test, w_test, u, len_u, sigma2, y):
    WxT_y = interp_to_u(idx_train, w_train, len_u, y)
    Kuu_y1 = toeplitz_matvec(u, WxT_y, len_u)
    Kinv_y2 = mean_Kinv_y(idx_train, w_train, u, len_u, sigma2, Kuu_y1)
    #print '#1'
    #print Kinv_y2
    Kuu_y3 = toeplitz_matvec(u, Kinv_y2, len_u)
    Ws_y4 = interp_from_u(idx_test, w_test, Kuu_y3)
    return Ws_y4


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
    y = y[np.newaxis]
    #y = np.vstack((y, y))
    z = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float)
    z = np.vstack((z, z, z, z))
    print y
    print interp_to_u(idx, w, 8, y)
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


def test_toeplitz_matvec():
    n = 6
    a = np.random.random(size=n)
    b = np.random.random(size=n)
    M = scipy.linalg.toeplitz(a)
    #b = np.vstack((b, b))
    print a
    print M
    print
    if b.ndim == 1:
        b = b[np.newaxis, :]
    print M.dot(b.T).T
    print toeplitz_matvec(a, b, n)


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
    len_u = 1024 + 1
    extra_u = 2
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)
    x_test = u[1:]
    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)

    ku = covmat_col0(u, (a, b))
    #print 'ku'
    #print ku
    #print (cov_fn(u, u, (a, b)) + np.eye(len_u))[:, 0]
    len_test = len(x_test)
    import time

    t1 = time.time()
    mu = post_mean(idx_train, w_train, idx_test, w_test, ku, len_u, c, y[np.newaxis])
    t2 = time.time()
    print t2 - t1
    #print '-' * 30
    #print mu
    t1 = time.time()
    mu_sor = post_mean_sor(x, x_test, u, (a, b), c, y)
    t2 = time.time()
    print t2 - t1
    #print mu_sor
    #print '-' * 30

    t1 = time.time()
    mu_exact = post_mean_exact(x, x_test, (a, b), c, y)
    t2 = time.time()
    print t2 - t1
    #print mu_exact

    return

    var = np.empty(len_test)
    for i in xrange(len_test):
        v = np.zeros(len_test)
        v[i] = 1
        var[i] = post_cov_y(idx_train, w_train, idx_test, w_test,
                            ku, len_u, c, v)[i]

    #print var

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
    test_toeplitz_matvec()
    #test_gp()


if __name__ == '__main__':
    main()
