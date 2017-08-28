import numpy as np
import theano
import theano.tensor as T
import theano.tests.unittest_tools
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from gp_kernel import kernel, symbolic_kernel
from fast_gp import sparse_w
from cov_vec_bprop import CovVec, PosteriorMean
from lanczos_sqrtm import lanczos
import scipy
import time


class PosteriorGP(object):
    def __init__(self, inducing_pts, x_test,
                 kernel, symbolic_kernel, init_params=None, random_seed=101):
        self.rng = RandomStreams(seed=random_seed)
        self.len_test = len(x_test)
        self.cov_vec = CovVec(inducing_pts, kernel, symbolic_kernel)
        self.post_mean = PosteriorMean(inducing_pts, kernel, symbolic_kernel)

        # input symbolic variables
        self.t_idx_train = T.imatrix()
        self.t_w_train = T.matrix()
        self.t_idx_test = T.imatrix()
        self.t_w_test = T.matrix()
        self.t_y_train = T.vector()

        if init_params is None:
            #init_params = [np.log(np.array([2., 10.])), np.log(0.3)]
            init_params = [np.array([-0.7, 5]), -2.5]
        log_gp_params, log_indep_noise = init_params
        self.log_gp_params = theano.shared(log_gp_params)
        self.log_indep_noise = theano.shared(log_indep_noise)

        self.gp_params = T.exp(self.log_gp_params)
        self.indep_noise = T.exp(self.log_indep_noise)

        # collection of symbolic variables derived from data
        self.data_variables = [self.t_idx_train, self.t_w_train,
                               self.t_idx_test, self.t_w_test,
                               self.t_y_train]
        # GP hyperparameters and noise parameter
        self.params = [self.log_gp_params, self.log_indep_noise]

    def set_params(self, params):
        log_gp_params, log_indep_noise = params
        self.log_gp_params.set_value(log_gp_params)
        self.log_indep_noise.set_value(log_indep_noise)

    def mean(self):
        mu = self.post_mean(self.t_idx_train, self.t_w_train,
                            self.t_idx_test, self.t_w_test,
                            self.gp_params, self.indep_noise,
                            self.t_y_train)
        return mu

    def cov_rand_proj(self, n_sample=10, n_lanczos_basis=10):
        cov_vec = self.cov_vec
        if n_sample == 1:
            cov_vec.use_single_sample()

        def linear_op(zs):
            return cov_vec(self.t_idx_train, self.t_w_train,
                           self.t_idx_test, self.t_w_test,
                           self.gp_params, self.indep_noise, zs)

        eps = self.rng.normal(size=(n_sample, self.len_test))
        cov_zs = lanczos(linear_op, eps, n_lanczos_basis, n_sample)
        return cov_zs

    def cov_proj(self, eps, n_sample=10, n_lanczos_basis=10):
        cov_vec = self.cov_vec

        def linear_op(zs):
            return cov_vec(self.t_idx_train, self.t_w_train,
                           self.t_idx_test, self.t_w_test,
                           self.gp_params, self.indep_noise, zs)

        cov_zs = lanczos(linear_op, eps, n_lanczos_basis, n_sample)
        return cov_zs


def main():
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
    #len_u = 1024 + 1
    len_u = 50
    #len_u = 128 + 1
    #len_u = 64
    extra_u = 2
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)
    #x_test = u[1:]
    x_test = u[2:-1]
    #x_test = np.linspace(x_min, x_max, 20)

    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)
    post_gp = PosteriorGP(u, x_test, kernel, symbolic_kernel)

    post_gp.log_gp_params.set_value(np.log(np.array([a, b])))
    post_gp.log_indep_noise.set_value(np.log(c))

    batch_size = 10
    cov_zs = post_gp.cov_rand_proj(n_sample=batch_size, n_lanczos_basis=10)
    mu = post_gp.mean()
    gp_samples = mu.dimshuffle('x', 0) + cov_zs

    variables = post_gp.data_variables
    gp_samples_fn = theano.function(variables, gp_samples)

    len_test = len(x_test)
    #y_test = np.random.normal(size=(batch_size, len_test))
    gdraws = gp_samples_fn(idx_train, w_train, idx_test, w_test, y)
    print gdraws.shape

    import pylab as pl
    pl.figure()

    for each_sample in gdraws:
        pl.plot(x_test, each_sample, '-', c='b', alpha=.5)
    pl.plot(x, y, 'o', c='r')
    pl.show()


if __name__ == '__main__':
    main()
