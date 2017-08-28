import numpy as np
import theano.tensor as T
import theano
import lasagne
from lasagne.updates import adagrad, adadelta, nesterov_momentum
from scipy.optimize import fmin_l_bfgs_b
from itertools import izip
import sys
import time
from pickle_io import pickle_load, pickle_save
from logdet import logabsdet
#import gp


batch_update = True


def marginal_likelihood(x, y, n_epochs=100, lrate=.1):
    gp_params = np.array([2., 10.])
    indep_noise = 0.3
    #gp_params = np.array([.4966, 148.41])
    #indep_noise = .0821

    t_log_gp_params = theano.shared(np.log(gp_params))
    t_log_indep_noise = theano.shared(np.log(indep_noise))

    def print_params():
        print t_log_gp_params.get_value()
        print t_log_indep_noise.get_value()

    print_params()

    t_x = T.vector('x')
    t_y = T.vector('y')
    t_gp_params = T.exp(t_log_gp_params)
    t_indep_noise = T.exp(t_log_indep_noise)
    x_col = t_x.dimshuffle(0, 'x')
    x_row = t_x.dimshuffle('x', 0)
    K = t_gp_params[0]* T.exp(-t_gp_params[1] * T.sqr(x_col - x_row))
    K = K + t_indep_noise * T.identity_like(K)
    y_Kinv_y = t_y.dot(T.nlinalg.matrix_inverse(K)).dot(t_y)
    #logdetK = T.log(T.nlinalg.det(K))
    logdetK = logabsdet(K)
    marginal_ll = -0.5 * (y_Kinv_y + logdetK)
    loss = -marginal_ll

    if batch_update:
        loss_fn = theano.function([t_x, t_y], loss)
        grad_loss_fn = theano.function([t_x, t_y],
                theano.grad(loss, [t_log_gp_params, t_log_indep_noise]))

        n_params = len(gp_params) + 1
        def f_df(params):
            a, b, c = params
            t_log_gp_params.set_value([a, b])
            t_log_indep_noise.set_value(c)
            total_loss = 0
            grad = np.zeros(n_params)
            for each_x, each_y in izip(x, y):
                total_loss += loss_fn(each_x, each_y)
                grad += np.append(*grad_loss_fn(each_x, each_y))
            return total_loss, grad

        init_params = np.r_[gp_params, indep_noise]
        opt = fmin_l_bfgs_b(f_df, init_params,
                            factr=1e3, pgtol=1e-07, disp=1,
                           )[0]
        print opt
        opt_gp_params = opt[:-1]
        opt_indep_noise = opt[-1]
    else:
        # Stochastic optimization
        updates = adagrad(loss, [t_log_gp_params, t_log_indep_noise],
                          learning_rate=lrate)
        loss_fn = theano.function([t_x, t_y], loss, updates=updates)
        grad_loss_fn = theano.function([t_x, t_y],
                theano.grad(loss, [t_log_gp_params, t_log_indep_noise]))

        for i in xrange(n_epochs):
            count = 1
            trace = []
            sys.stderr.write('%4d  ' % i)
            for each_x, each_y in izip(x, y):
                val = loss_fn(each_x, each_y)
                trace.append(val)
                count += 1
                if count % 20 == 0:
                    sys.stderr.write('.')
            print
            print np.mean(trace), np.std(trace)
            print_params()
        opt_gp_params = t_log_gp_params.get_value()
        opt_indep_noise = t_log_indep_noise.get_value()
    return opt_gp_params, opt_indep_noise


def train_gp(data):
    np.random.seed(123)
    lasagne.random.set_rng(np.random.RandomState(seed=123))
    x_train, y_train, x_test, y_test, l_train, l_test = pickle_load(data)
    gp_params, indep_noise = marginal_likelihood(x_train, y_train)
    return gp_params, indep_noise


def main():
    for sparsity in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        #data = 'data/UWaveGestureLibraryAll-%d.pkl' % sparsity
        data = 'data/PhalangesOutlinesCorrect-%d.pkl' % sparsity
        print data
        t1 = time.time()
        gp_params, indep_noise = train_gp(data)
        t2 = time.time()
        print 'time', t2 - t1
        data_id = data.rsplit('/', 1)[-1]
        pickle_save('params/%s' % data_id, gp_params, indep_noise)


if __name__ == '__main__':
    main()
