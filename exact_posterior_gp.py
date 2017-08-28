import numpy as np
import theano
import theano.tensor as T
import theano.tests.unittest_tools
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from fast_gp import sparse_w
from theano_sqrtm import sqrtm as theano_sqrtm
import scipy
import time


def cov_mat(x1, x2, gp_params):
    x1_col = x1.dimshuffle(0, 'x')
    x2_row = x2.dimshuffle('x', 0)
    K = gp_params[0] * T.exp(-gp_params[1] * T.sqr(x1_col - x2_row))
    return K


class PosteriorGP(object):
    def __init__(self, x_test, init_params=None, random_seed=101):
        self.rng = RandomStreams(seed=random_seed)
        self.len_test = len(x_test)

        # input symbolic variables
        self.t_x_train = T.vector()
        self.t_x_test = T.vector()
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
        self.data_variables = [self.t_x_train, self.t_x_test, self.t_y_train]
        # GP hyperparameters and noise parameter
        self.params = [self.log_gp_params, self.log_indep_noise]

    def set_params(self, params):
        log_gp_params, log_indep_noise = params
        self.log_gp_params.set_value(log_gp_params)
        self.log_indep_noise.set_value(log_indep_noise)

    def mean(self):
        x = self.t_x_train
        y = self.t_y_train
        x_test = self.t_x_test
        gp_params = self.gp_params
        indep_noise = self.indep_noise

        Kxt = cov_mat(x, x_test, gp_params)
        Kxx = cov_mat(x, x, gp_params)
        Kxx = Kxx + indep_noise * T.identity_like(Kxx)
        KxtT_Kxxinv = Kxt.T.dot(T.nlinalg.matrix_inverse(Kxx))
        mu = KxtT_Kxxinv.dot(y)
        return mu

    def cov_rand_proj(self, n_sample=10):
        x = self.t_x_train
        x_test = self.t_x_test
        gp_params = self.gp_params
        indep_noise = self.indep_noise

        Ktt = cov_mat(x_test, x_test, gp_params)
        Kxt = cov_mat(x, x_test, gp_params)
        Kxx = cov_mat(x, x, gp_params)
        Kxx = Kxx + indep_noise * T.identity_like(Kxx)
        KxtT_Kxxinv = Kxt.T.dot(T.nlinalg.matrix_inverse(Kxx))
        K = Ktt - KxtT_Kxxinv.dot(Kxt)
        #K = K + 1e-10 * T.identity_like(K)
        R = theano_sqrtm(K)
        eps = self.rng.normal(size=(n_sample, self.len_test))
        return eps.dot(R.T)

    def cov_proj(self, eps, n_sample=10):
        x = self.t_x_train
        x_test = self.t_x_test
        gp_params = self.gp_params
        indep_noise = self.indep_noise

        Ktt = cov_mat(x_test, x_test, gp_params)
        Kxt = cov_mat(x, x_test, gp_params)
        Kxx = cov_mat(x, x, gp_params)
        Kxx = Kxx + indep_noise * T.identity_like(Kxx)
        KxtT_Kxxinv = Kxt.T.dot(T.nlinalg.matrix_inverse(Kxx))
        K = Ktt - KxtT_Kxxinv.dot(Kxt)
        #K = K + 1e-10 * T.identity_like(K)
        R = theano_sqrtm(K)
        return eps.dot(R.T)


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
    len_test = 50
    x_test = np.linspace(x_min, x_max, len_test)

    post_gp = PosteriorGP(x_test)

    post_gp.log_gp_params.set_value(np.log(np.array([a, b])))
    post_gp.log_indep_noise.set_value(np.log(c))

    batch_size = 10
    cov_zs = post_gp.cov_rand_proj(n_sample=batch_size)
    mu = post_gp.mean()
    gp_samples = mu.dimshuffle('x', 0) + cov_zs

    variables = post_gp.data_variables
    gp_samples_fn = theano.function(variables, gp_samples)

    len_test = len(x_test)
    #y_test = np.random.normal(size=(batch_size, len_test))
    gdraws = gp_samples_fn(x, x_test, y)
    print gdraws.shape

    import pylab as pl
    pl.figure()

    for each_sample in gdraws:
        pl.plot(x_test, each_sample, '-', c='b', alpha=.5)
    pl.plot(x, y, 'o', c='r')
    pl.show()


if __name__ == '__main__':
    main()
