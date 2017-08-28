from __future__ import division
import numpy as np
import theano.tensor as T
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.nonlinearities import softmax, rectify, tanh
from lasagne.layers import InputLayer, DenseLayer, Conv1DLayer, MaxPool1DLayer
from lasagne.layers import DimshuffleLayer, DropoutLayer
from lasagne.layers import LSTMLayer, SliceLayer, ConcatLayer
from lasagne.layers import get_all_param_values, set_all_param_values
from lasagne.objectives import categorical_crossentropy
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import adagrad, adadelta, nesterov_momentum
from itertools import izip
import time
import argparse
import sys
from pickle_io import pickle_load, pickle_save
from posterior_gp import PosteriorGP
from gp_kernel import kernel, symbolic_kernel
from fast_gp import sparse_w
from logger import Logger


class GPNet(object):
    def __init__(self,
                 n_classes,
                 inducing_pts,
                 t_test,
                 update_gp=True,
                 init_gp_params=None,   # kernel parameters & noise parameter
                 #n_inducing_pts=50,
                 #t_min=0,
                 #t_max=1,
                 n_lanczos_basis=10,
                 n_samples=1000,
                 n_epochs=100,
                 gamma=5,
                 regularize_weight=0,
                 optimizer=adadelta,
                 optimizer_kwargs={},
                 load_params=None,
                 random_seed=123):
        '''
        n_samples: number of Monte Carlo samples to estimate the expectation
        n_inducing_pts: number of inducing points
        '''
        lasagne.random.set_rng(np.random.RandomState(seed=random_seed))
        W = np.random.normal(0, 1 / gamma**2, size=(n_samples, len(t_test)))
        b = np.random.uniform(0, 2 * np.pi, size=n_samples)
        self.random_weight = theano.shared(W)
        self.random_offset = theano.shared(b)

        if load_params:
            (model_params,
             network_params,
             init_gp_params) = pickle_load(load_params)
            (self.n_classes,
             self.inducing_pts,
             self.idx_test, self.w_test,
             self.gp_output_len) = model_params
        else:
            self.n_classes = n_classes
            self.inducing_pts = inducing_pts
            self.idx_test, self.w_test = sparse_w(inducing_pts, t_test)
            self.gp_output_len = len(t_test)

        self.n_lanczos_basis = n_lanczos_basis
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.post_gp = PosteriorGP(inducing_pts, t_test,
                                   kernel, symbolic_kernel,
                                   init_params=init_gp_params)
        self.update_gp = update_gp
        self.regularize_weight = regularize_weight
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.compile_train_predict()

        if load_params:
            self.load_params(network_params)

    def compile_train_predict(self):
        # symbolic functions to compute marginal posterior GP
        input_vars = self.post_gp.data_variables
        gp_hyperparams = self.post_gp.params
        self.gp_hyperparams = gp_hyperparams

        self.network = self.extend_network()

        train_predict = lasagne.layers.get_output(self.network)

        # Compute the exepcted prediction
        #if stochastic_train and self.n_samples > 1:
        #    train_predict = train_predict.mean(axis=0, keepdims=True)

        label = T.ivector('label')

        loss = categorical_crossentropy(train_predict, label).mean()
        # For expected prediction
        #loss = categorical_crossentropy(train_predict, label).mean()
        if self.regularize_weight > 0:
            penalty = (self.regularize_weight *
                       regularize_network_params(self.network, l2))
            loss += penalty

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        update_params = params
        if self.update_gp:
            update_params += gp_hyperparams
        grad_loss = theano.grad(loss, update_params,
                                consider_constant=input_vars)
        updates = self.optimizer(grad_loss, update_params,
                                 **self.optimizer_kwargs)
        self.train_fn = theano.function(input_vars + [label],
                                        loss, updates=updates)

        # Set deterministic=True for dropout training if used.
        test_predict = lasagne.layers.get_output(self.network,
                                                 deterministic=True)
        if self.n_samples > 1:
            test_predict = test_predict.mean(axis=0, keepdims=True)

        self.predict_fn = theano.function(input_vars, test_predict)

    def cov_mat(self, x1, x2, exp_a, exp_b):
        x1_col = x1.dimshuffle(0, 'x')
        x2_row = x2.dimshuffle('x', 0)
        K = exp_a * T.exp(-exp_b * T.sqr(x1_col - x2_row))
        return K

    def extend_network(self):
        m = self.n_samples
        W = self.random_weight
        b = self.random_offset
        mu = self.post_gp.mean()
        #mu = mu.dimshuffle('x', 0)   # make a row out of 1d vector (N to 1xN)

        cov_zs = self.post_gp.cov_proj(W, n_sample=m,
                                       n_lanczos_basis=self.n_lanczos_basis)
        exp_term = T.exp(-0.5 * (cov_zs * W).sum(axis=1))
        cos_term = T.cos(W.dot(mu) + b)
        random_feature = np.sqrt(2 / m) * exp_term * cos_term

        network = InputLayer(shape=(1, m),
                             input_var=random_feature.dimshuffle('x', 0))
        l_output = DenseLayer(network, num_units=self.n_classes,
                              nonlinearity=softmax)
        return l_output

    def predict(self, x, y):
        test_prediction = self.predict_proba(x, y)
        return np.argmax(test_prediction, axis=1)

    def predict_proba(self, x, y):
        test_prediction = []
        for each_x, each_y in izip(x, y):
            idx, w = sparse_w(self.inducing_pts, each_x)
            test_prediction.append(
                    self.predict_fn(idx, w, self.idx_test, self.w_test,
                                    each_y))
        test_prediction = np.vstack(test_prediction)
        return test_prediction

    def evaluate_prediction(self, x_test, y_test, l_test):
        if x_test is None:
            return 0, 0
        predict_test_proba = self.predict_proba(x_test, y_test)
        predict_test = np.argmax(predict_test_proba, axis=1)
        # accuracy
        accuracy = np.mean(l_test == predict_test)
        # categorical crossentropy
        prob_hit = predict_test_proba[np.arange(len(l_test)), l_test]
        crossentropy = -np.log(prob_hit).mean()
        return accuracy, crossentropy

    def inspect_train(self, x_train, y_train, l_train,
                      x_valid, y_valid, l_valid, x_test, y_test, l_test,
                      save_params=None):
        idx_w = []
        for each_x in x_train:
            idx_w.append(sparse_w(self.inducing_pts, each_x))
        for epoch in xrange(self.n_epochs):
            history = []
            count = 1
            t1 = time.time()
            for (idx_train, w_train), each_y, each_label in izip(
                    idx_w, y_train, l_train):
                history.append(self.train_fn(idx_train, w_train,
                                             self.idx_test, self.w_test,
                                             each_y, [each_label]))
                count += 1
                if count % 20 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                #for v in self.gp_hyperparams:
                #    print v.get_value()
            #print '-' * 30
            t2 = time.time()
            print ' ', t2 - t1
            #print
            mean_loss = np.mean(history)

            accuracy_train, loss_train = self.evaluate_prediction(
                    x_train, y_train, l_train)
            accuracy_valid, loss_valid = self.evaluate_prediction(
                    x_valid, y_valid, l_valid)
            accuracy_test, loss_test = self.evaluate_prediction(
                    x_test, y_test, l_test)

            print '%4d - %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f' % (
                    epoch, mean_loss, accuracy_train, loss_train,
                    accuracy_valid, loss_valid, accuracy_test, loss_test)
            for v in self.gp_hyperparams:
                print v.get_value()

            if save_params:
                self.save_params(save_params + '-%03d' % epoch)

    def save_params(self, params_file):
        # TODO: save t_test instead of idx_test and w_test
        model_params = (self.n_classes, self.inducing_pts,
                        self.idx_test, self.w_test, self.gp_output_len)
        network_params = get_all_param_values(self.network)
        gp_params = [p.get_value() for p in self.post_gp.params]
        pickle_save(params_file, model_params, network_params, gp_params)

    def load_params(self, network_params):
        set_all_param_values(self.network, network_params)


def run_gpnet(data,
              task_name,   # for logging
              n_inducing_pts=256,
              n_network_inputs=1000,
              update_gp=True,
              init_gp_params=None,   # kernel parameters & noise parameter
              regularize_weight=0,
              n_lanczos_basis=3,
              n_samples=5,
              n_epochs=500,
              gamma=5,
              optimizer=adadelta,
              optimizer_kwargs={},
              subset_train=None,
              validation_set=.3,
              swap=False,
              save_params_epochs=None,
              save_params=None,
              load_params=None):
    np.random.seed(1)
    x_train, y_train, x_test, y_test, l_train, l_test = pickle_load(data)
    if swap:
        x_train, x_test = x_test, x_train
        y_train, y_test = y_test, y_train
        l_train, l_test = l_test, l_train

    if subset_train:
        if 0 < subset_train <= 1:
            n_train = int(len(l_train) * validation_set)
        elif subset_train > 1:
            n_train = int(subset_train)
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
        l_train = l_train[:n_train]

    x_valid, y_valid, l_valid = None, None, None
    if validation_set:
        total_train = len(l_train)
        if 0 < validation_set <= 1:
            n_valid = int(total_train * validation_set)
        elif validation_set > 1:
            n_valid = int(validation_set)

        n_train = total_train - n_valid
        x_train, x_valid = x_train[:n_train], x_train[n_train:]
        y_train, y_valid = y_train[:n_train], y_train[n_train:]
        l_train, l_valid = l_train[:n_train], l_train[n_train:]

    n_classes = len(set(l_train) | set(l_test))

    t_min, t_max = 0, 1
    extra_u = 2
    margin = (t_max - t_min) / (n_inducing_pts - extra_u * 2) * 2
    inducing_pts = np.linspace(t_min - margin, t_max + margin, n_inducing_pts)
    if n_network_inputs <= 0:
        t_test = inducing_pts[1:-1]
    else:
        t_test = np.linspace(t_min, t_max, n_network_inputs)

    gpnet = GPNet(n_classes,
                  inducing_pts=inducing_pts,
                  t_test=t_test,
                  update_gp=update_gp,
                  init_gp_params=init_gp_params,
                  regularize_weight=regularize_weight,
                  n_lanczos_basis=n_lanczos_basis,
                  n_samples=n_samples,
                  n_epochs=n_epochs,
                  gamma=gamma,
                  optimizer=optimizer,
                  optimizer_kwargs=optimizer_kwargs,
                  load_params=load_params)

    def print_parameters():
        print 'data:', data
        print 'n_train:', len(x_train)
        print 'n_valid:', len(x_valid) if x_valid else 0
        print 'n_test:', len(x_test)
        print 'n_classes:', n_classes
        print 'n_inducing_pts:', n_inducing_pts
        print 'n_net_inputs:', len(t_test)
        print 'n_lanczos_basis:', n_lanczos_basis
        print 'n_samples:', n_samples
        print 'n_epochs:', n_epochs
        print 'gamma:', gamma
        print 'optimizer:', optimizer.__name__
        print 'optimizer_kwargs:', optimizer_kwargs
        print 'regularize_weight:', regularize_weight
        print 'init_gp_params:', init_gp_params
        print 'update_gp:', update_gp
        print 'load:', load_params
        print 'save:', save_params
        print 'save_epochs:', save_params_epochs

    print_parameters()

    for v in gpnet.post_gp.params:
        print v.get_value()

    t1 = time.time()
    gpnet.inspect_train(x_train, y_train, l_train,
                        x_valid, y_valid, l_valid,
                        x_test, y_test, l_test,
                        save_params_epochs)
    t2 = time.time()

    if save_params:
        gpnet.save_params(save_params)

    print_parameters()
    print 'time:', t2 - t1
    print task_name

    return

    gpnet.train(x_train, y_train, l_train)

    predict_train = gpnet.predict(x_train, y_train)
    print np.mean(l_train == predict_train)

    predict_test = gpnet.predict(x_test, y_test)
    print np.mean(l_test == predict_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='data',
                        #default='data/UWaveGestureLibraryAll-10.pkl',
                        default='data/B-UWaveGestureLibraryAll-10.pkl',
                        help='data file')
    parser.add_argument('-e', dest='epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('-o', dest='optimizer', default='nesterov_momentum',
                        help='optimizer: nesterov_momentum, adagrad, adadelta')
    parser.add_argument('-r', dest='lrate', type=float, default=0.0008,
                        help='learning rate')
    parser.add_argument('-s', dest='subset', type=float, default=None,
                        help='portion or size of subset of data')
    parser.add_argument('-v', dest='validate', type=float, default=.3,
                        help='portion or size of validation set (override -s)')
    parser.add_argument('-p', dest='gp_params', default=None,
                        help='file of GP parameters')
    parser.add_argument('-k', dest='ind_pts', type=int, default=256,
                        help='number of inducing points')
    parser.add_argument('-i', dest='net_ins', type=int, default=0,
                        help='number of network inputs. '
                             '0: use inducing points')
    parser.add_argument('-b', dest='lanczos_basis', type=int, default=3,
                        help='number of Lanczos bases')
    parser.add_argument('-m', dest='samples', type=int, default=500,
                        help='number of Monte Carlo samples')
    parser.add_argument('-g', dest='reg', type=float, default=0,
                        help='regularization weight')
    parser.add_argument('-u', dest='fix_gp', action='store_true',
                        default=False, help='fix GP parameters')
    parser.add_argument('-l', dest='log', action='store_true',
                        default=False, help='log stdout')
    parser.add_argument('-x', dest='swap', action='store_true',
                        default=False, help='swap training/test set')
    parser.add_argument('-a', dest='gamma', type=float, default=5,
                        help='gamma')
    parser.add_argument('--saveall', dest='saveall', default=None,
                        help='save parameters at each epoch')
    parser.add_argument('--save', dest='save', default=None,
                        help='save parameters')
    parser.add_argument('--load', dest='load', default=None,
                        help='load parameters (override -p)')
    args = parser.parse_args()

    if args.log:
        sys.stdout = Logger()

    init_gp_params = None
    if args.gp_params:
        try:
            init_gp_params = pickle_load(args.gp_params)
        except:
            pass

    task_name = args.data.rsplit('/', 1)[-1][:-4]
    print task_name

    optimizer = nesterov_momentum
    if args.optimizer == 'adagrad':
        optimizer = adagrad
    elif args.optimizer == 'adadelta':
        optimizer = adadelta

    # candidate 1: 'Cricket_Z/09_1000_60_dat.pkl'
    run_gpnet(args.data,
              task_name,
              n_lanczos_basis=args.lanczos_basis,
              n_samples=args.samples,
              n_inducing_pts=args.ind_pts,
              n_network_inputs=args.net_ins,
              update_gp=(not args.fix_gp),
              init_gp_params=init_gp_params,
              subset_train=args.subset,
              validation_set=args.validate,
              n_epochs=args.epochs,
              regularize_weight=args.reg,
              gamma=args.gamma,
              optimizer=optimizer,
              optimizer_kwargs={'learning_rate': args.lrate},
              swap=args.swap,
              save_params_epochs=args.saveall,
              save_params=args.save,
              load_params=args.load)


if __name__ == '__main__':
    main()
