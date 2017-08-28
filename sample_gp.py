import numpy as np
import theano
import theano.tensor as T
import theano.tests.unittest_tools
from gp_kernel import kernel, symbolic_kernel
from cov_vec_bprop import CovVec, PosteriorMean
from lanczos_sqrtm import lanczos
import scipy
import time


def main():
    from fast_gp import sparse_w
    np.random.seed(0)
    n_data = 10
    x = np.random.uniform(size=n_data)
    #x = np.float32(x)
    x = np.sort(x)
    a = .1
    b = 10
    c = .001
    mu = np.zeros(n_data)
    cov = a * np.exp(-b * (x[:, np.newaxis] - x)**2) + c * np.eye(n_data)
    y = np.random.multivariate_normal(mu, cov)
    #print x
    #print y
    x_min, x_max = x.min(), x.max()
    #len_u = 2048 + 1
    len_u = 1024 + 1
    #len_u = 128 + 1
    #len_u = 64
    extra_u = 2
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)
    x_test = u[1:]
    #x_test = np.linspace(x_min, x_max, 20)
    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)

    t_idx_train = T.imatrix()
    t_w_train = T.matrix()
    t_idx_test = T.imatrix()
    t_w_test = T.matrix()
    t_gp_params = T.vector()
    t_indep_noise = T.scalar()
    t_ys = T.matrix()
    t_y = T.vector()

    cov_vec = CovVec(u, kernel, symbolic_kernel)

    def linear_op(zs):
        return cov_vec(t_idx_train, t_w_train, t_idx_test, t_w_test,
                       t_gp_params, t_indep_noise, zs)

    n_lanczos_basis = 10
    batch_size = 10
    cov_zs = lanczos(linear_op, t_ys, n_lanczos_basis, batch_size)

    post_mean = PosteriorMean(u, kernel, symbolic_kernel)
    mu = post_mean(t_idx_train, t_w_train, t_idx_test, t_w_test,
                   t_gp_params, t_indep_noise, t_y)

    gp_samples = mu.dimshuffle('x', 0) + cov_zs

    gp_samples_fn = theano.function([
        t_idx_train, t_w_train, t_idx_test, t_w_test,
        t_gp_params, t_indep_noise, t_y, t_ys], gp_samples)

    len_test = len(x_test)
    y_test = np.random.normal(size=(batch_size, len_test))
    gdraws = gp_samples_fn(idx_train, w_train, idx_test, w_test,
                           (a, b), c, y, y_test)
    print gdraws.shape

    t_random_proj = T.matrix()
    val = (gp_samples * t_random_proj).sum()

    val_fn = theano.function([
        t_idx_train, t_w_train, t_idx_test, t_w_test,
        t_gp_params, t_indep_noise, t_y, t_ys, t_random_proj], val)

    grad_val_fn = theano.function([
        t_idx_train, t_w_train, t_idx_test, t_w_test,
        t_gp_params, t_indep_noise, t_y, t_ys, t_random_proj],
        theano.grad(val,
                    wrt=[t_gp_params, t_indep_noise],
                    consider_constant=[t_idx_train, t_w_train,
                                       t_idx_test, t_w_test,
                                       t_y, t_ys, t_random_proj]))

    grad_val_fn1 = theano.function([
        t_idx_train, t_w_train, t_idx_test, t_w_test,
        t_gp_params, t_indep_noise, t_y, t_ys, t_random_proj],
        theano.grad(val,
                    wrt=[t_random_proj],
                    consider_constant=[t_idx_train, t_w_train,
                                       t_idx_test, t_w_test,
                                       t_y, t_ys]))

    random_proj = np.random.rand(batch_size, len_test)
    t1 = time.time()
    for _ in xrange(10):
        grad_val_fn(idx_train, w_train, idx_test, w_test,
                (a, b), c, y, y_test, random_proj)
    t2 = time.time()
    print t2 - t1
    t1 = time.time()
    for _ in xrange(10):
        grad_val_fn1(idx_train, w_train, idx_test, w_test,
                (a, b), c, y, y_test, random_proj)
    t2 = time.time()
    print t2 - t1
    return

    n_test = 10
    for _ in xrange(n_test):
        random_proj = np.random.rand(batch_size, len_test)
        print 'test grad'
        print val_fn(idx_train, w_train, idx_test, w_test,
                     (a, b), c, y, y_test, random_proj)
        print grad_val_fn(idx_train, w_train, idx_test, w_test,
                          (a, b), c, y, y_test, random_proj)

        def val_fn1(x):
            a, b, c = x
            return val_fn(idx_train, w_train, idx_test, w_test,
                          (a, b), c, y, y_test, random_proj)

        def grad_val_fn1(x):
            a, b, c = x
            [a, b], c = grad_val_fn(idx_train, w_train, idx_test, w_test,
                                    (a, b), c, y, y_test, random_proj)
            return np.array([a, b, c])

        print scipy.optimize.check_grad(val_fn1, grad_val_fn1,
                                        np.array([a, b, c]))

    return
    import pylab as pl
    pl.figure()

    for each_sample in gdraws:
        pl.plot(x_test, each_sample, '-', c='b', alpha=.5)
    pl.plot(x, y, 'o', c='r')
    pl.show()


if __name__ == '__main__':
    main()
