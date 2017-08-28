from __future__ import division
import numpy as np
from fast_gp import sparse_w
from fast_gp import post_cov_ys, grad_cov_ys
from fast_gp_block import post_cov_ys as post_cov_ys_block
from fast_gp_block import grad_cov_ys as grad_cov_ys_block


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

    #return
    #mu = post_mean(idx_train, w_train, idx_test, w_test, ku, len_u, c, y)

    t1 = time.time()
    var1 = post_cov_ys(idx_train, w_train, idx_test, w_test,
                       ku, len_u, c, np.eye(len_test))
    t2 = time.time()
    print t2 - t1

    t1 = time.time()
    var2 = post_cov_ys_block(idx_train, w_train, idx_test, w_test,
                             ku, len_u, c, np.eye(len_test))
    t2 = time.time()
    print t2 - t1
    print np.allclose(var1, var2)

    grad_u = np.random.normal(size=(len_u, 2))
    gz = np.ones((len_test, len_test))

    t1 = time.time()
    g1 = grad_cov_ys(idx_train, w_train, idx_test, w_test,
                     ku, (a, b), c, np.eye(len_test), grad_u, gz)
    t2 = time.time()
    print t2 - t1

    t1 = time.time()
    g2 = grad_cov_ys_block(idx_train, w_train, idx_test, w_test,
                           ku, (a, b), c, np.eye(len_test), grad_u, gz)
    t2 = time.time()
    print t2 - t1
    for a, b in zip(g1, g2):
        print np.allclose(a, b)

    #print var
    #return


def main():
    #test_w()
    #test_idx_w()
    #test_u()
    test_gp()


if __name__ == '__main__':
    main()
