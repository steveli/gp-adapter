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
from exact_posterior_gp import PosteriorGP
#from posterior_gp import PosteriorGP
#from gp_kernel import kernel, symbolic_kernel
from fast_gp import sparse_w
from logger import Logger


def build_logistic_regression(input_layer):
    return input_layer


def build_cnn(input_layer):
    # Add a channel axis for convolutional nets
    network = DimshuffleLayer(input_layer, (0, 'x', 1))
    network = Conv1DLayer(network, num_filters=4, filter_size=5,
                          nonlinearity=rectify)
    network = MaxPool1DLayer(network, pool_size=2)
    network = Conv1DLayer(network, num_filters=4, filter_size=5,
                          nonlinearity=rectify)
    network = MaxPool1DLayer(network, pool_size=2)
    network = DropoutLayer(network, p=.5)
    network = DenseLayer(network, num_units=256, nonlinearity=rectify)
    return network


def build_mlp(input_layer):
    network = DenseLayer(input_layer, num_units=256, nonlinearity=rectify)
    network = DenseLayer(network, num_units=256, nonlinearity=rectify)
    return network


def sliding_window_input(input_layer):
    window_size = 5
    sub_input = []
    for i in xrange(window_size):
        indices = slice(window_size - i - 1, -i if i > 0 else None)
        network = DimshuffleLayer(SliceLayer(input_layer, indices, axis=-1),
                                  (0, 1, 'x'))
        sub_input.append(network)
    network = ConcatLayer(sub_input, -1)
    return network


def build_lstm(input_layer):
    #network = sliding_window_input(input_layer)
    network = DimshuffleLayer(input_layer, (0, 1, 'x'))

    n_hidden = 50
    grad_clipping = 20
    network = LSTMLayer(network, num_units=n_hidden,
                        grad_clipping=grad_clipping, nonlinearity=tanh)
    network = LSTMLayer(network, num_units=n_hidden,
                        grad_clipping=grad_clipping, nonlinearity=tanh)
    network = SliceLayer(network, indices=-1, axis=1)
    #network = DenseLayer(network, num_units=256, nonlinearity=rectify)
    return network


cls_network = {
    'logreg': build_logistic_regression,
    'cnn': build_cnn,
    'mlp': build_mlp,
    #'lstm': build_lstm,
}


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
                 net_arch='logreg',
                 stochastic_train=True,
                 stochastic_predict=False,
                 n_samples=10,
                 n_epochs=100,
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
        self.rng = RandomStreams(seed=random_seed)

        self.t_test = t_test

        if load_params:
            (model_params,
             network_params,
             init_gp_params) = pickle_load(load_params)
            (self.net_arch,
             self.n_classes,
             self.inducing_pts,
             self.idx_test, self.w_test,
             self.gp_output_len) = model_params
        else:
            self.net_arch = net_arch
            self.n_classes = n_classes
            self.inducing_pts = inducing_pts
            self.idx_test, self.w_test = sparse_w(inducing_pts, t_test)
            self.gp_output_len = len(t_test)

        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.post_gp = PosteriorGP(t_test, init_params=init_gp_params)
        self.update_gp = update_gp
        self.regularize_weight = regularize_weight
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        # Save stochastic train/predict flags for storing parameters
        self.stochastic_train = stochastic_train
        self.stochastic_predict = stochastic_predict
        self.compile_train_predict(stochastic_train, stochastic_predict)

        if load_params:
            self.load_params(network_params)

    def compile_train_predict(self, stochastic_train, stochastic_predict):
        # symbolic functions to compute marginal posterior GP
        input_vars = self.post_gp.data_variables
        gp_hyperparams = self.post_gp.params
        self.gp_hyperparams = gp_hyperparams

        mu = self.post_gp.mean()
        mu = mu.dimshuffle('x', 0)   # make a row out of 1d vector (N to 1xN)

        self.train_network = self.extend_network(mu, stochastic_train)

        train_predict = lasagne.layers.get_output(self.train_network)

        # Compute the exepcted prediction
        #if stochastic_train and self.n_samples > 1:
        #    train_predict = train_predict.mean(axis=0, keepdims=True)

        label = T.ivector('label')

        # For expected loss
        if stochastic_train:
            label_rep = label.repeat(self.n_samples)
        else:
            label_rep = label

        loss = categorical_crossentropy(train_predict, label_rep).mean()
        # For expected prediction
        #loss = categorical_crossentropy(train_predict, label).mean()
        if self.regularize_weight > 0:
            penalty = (self.regularize_weight *
                       regularize_network_params(self.train_network, l2))
            loss += penalty

        params = lasagne.layers.get_all_params(self.train_network,
                                               trainable=True)
        update_params = params
        if self.update_gp:
            update_params += gp_hyperparams
        grad_loss = theano.grad(loss, update_params,
                                consider_constant=input_vars)
        updates = self.optimizer(grad_loss, update_params,
                                 **self.optimizer_kwargs)
        self.train_fn = theano.function(input_vars + [label],
                                        loss, updates=updates)

        if stochastic_train == stochastic_predict:
            self.test_network = self.train_network
            self.copy_params = False
        else:
            self.test_network = self.extend_network(mu, stochastic_predict)
            self.copy_params = True

        # Set deterministic=True for dropout training if used.
        test_predict = lasagne.layers.get_output(self.test_network,
                                                 deterministic=True)
        if stochastic_predict and self.n_samples > 1:
            test_predict = test_predict.mean(axis=0, keepdims=True)

        self.predict_fn = theano.function(input_vars, test_predict)

    def cov_mat(self, x1, x2, exp_a, exp_b):
        x1_col = x1.dimshuffle(0, 'x')
        x2_row = x2.dimshuffle('x', 0)
        K = exp_a * T.exp(-exp_b * T.sqr(x1_col - x2_row))
        return K

    def extend_network(self, mu, draw_sample):
        if not draw_sample:
            batch_size = 1
            input_data = mu
        else:
            batch_size = self.n_samples
            cov_zs = self.post_gp.cov_rand_proj(n_sample=batch_size)
            input_data = mu + cov_zs

        input_layer = InputLayer(shape=(batch_size, self.gp_output_len),
                                 input_var=input_data)
        network_builder = cls_network[self.net_arch]
        network = network_builder(input_layer)
        #network = DropoutLayer(network, p=.5)
        l_output = DenseLayer(network, num_units=self.n_classes,
                              nonlinearity=softmax)
        return l_output

    def predict(self, x, y):
        test_prediction = self.predict_proba(x, y)
        return np.argmax(test_prediction, axis=1)

    def predict_proba(self, x, y):
        test_prediction = []
        for each_x, each_y in izip(x, y):
            test_prediction.append(
                    self.predict_fn(each_x, self.t_test, each_y))
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
        for epoch in xrange(self.n_epochs):
            history = []
            count = 1
            t1 = time.time()
            for each_x, each_y, each_label in izip(x_train, y_train, l_train):
                history.append(self.train_fn(each_x, self.t_test, each_y,
                                             [each_label]))
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

            if self.copy_params:
                all_params = get_all_param_values(self.train_network)
                set_all_param_values(self.test_network, all_params)

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
        model_params = (self.net_arch, self.n_classes, self.inducing_pts,
                        self.idx_test, self.w_test, self.gp_output_len)
        network_params = get_all_param_values(self.train_network)
        gp_params = [p.get_value() for p in self.post_gp.params]
        pickle_save(params_file, model_params, network_params, gp_params)

    def load_params(self, network_params):
        set_all_param_values(self.train_network, network_params)
        set_all_param_values(self.test_network, network_params)


def run_gpnet(data,
              task_name,   # for logging
              n_inducing_pts=256,
              n_network_inputs=1000,
              update_gp=True,
              init_gp_params=None,   # kernel parameters & noise parameter
              net_arch='logreg',
              regularize_weight=0,
              stochastic_train=True,
              stochastic_predict=False,
              n_samples=5,
              n_epochs=500,
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
                  net_arch=net_arch,
                  regularize_weight=regularize_weight,
                  stochastic_train=stochastic_train,
                  stochastic_predict=stochastic_predict,
                  n_samples=n_samples,
                  n_epochs=n_epochs,
                  optimizer=optimizer,
                  optimizer_kwargs=optimizer_kwargs,
                  load_params=load_params)

    def print_parameters():
        print '[exact GP inference]'
        print 'data:', data
        print 'n_train:', len(x_train)
        print 'n_valid:', len(x_valid) if x_valid else 0
        print 'n_test:', len(x_test)
        print 'n_classes:', n_classes
        print 'n_net_inputs:', len(t_test)
        print 'stochastic_train:', stochastic_train
        print 'stochastic_predict:', stochastic_predict
        print 'n_samples:', n_samples
        print 'n_epochs:', n_epochs
        print 'network:', gpnet.net_arch
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
    parser.add_argument('-d', dest='det_train', action='store_true',
                        default=False, help='deterministic train')
    parser.add_argument('-o', dest='optimizer', default='nesterov_momentum',
                        help='optimizer: nesterov_momentum, adagrad, adadelta')
    parser.add_argument('-n', dest='net_arch', default='logreg',
                        help='network architecture: ' + ', '.join(cls_network))
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
    parser.add_argument('-m', dest='samples', type=int, default=10,
                        help='number of Monte Carlo samples')
    parser.add_argument('-g', dest='reg', type=float, default=0,
                        help='regularization weight')
    parser.add_argument('-u', dest='fix_gp', action='store_true',
                        default=False, help='fix GP parameters')
    parser.add_argument('-l', dest='log', action='store_true',
                        default=False, help='log stdout')
    parser.add_argument('-x', dest='swap', action='store_true',
                        default=False, help='swap training/test set')
    parser.add_argument('--saveall', dest='saveall', default=None,
                        help='save parameters at each epoch')
    parser.add_argument('--save', dest='save', default=None,
                        help='save parameters')
    parser.add_argument('--load', dest='load', default=None,
                        help='load parameters (override -n and -p)')
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
              n_samples=args.samples,
              n_inducing_pts=args.ind_pts,
              n_network_inputs=args.net_ins,
              update_gp=(not args.fix_gp),
              init_gp_params=init_gp_params,
              subset_train=args.subset,
              validation_set=args.validate,
              n_epochs=args.epochs,
              stochastic_train=(not args.det_train),
              net_arch=args.net_arch,
              regularize_weight=args.reg,
              optimizer=optimizer,
              optimizer_kwargs={'learning_rate': args.lrate},
              swap=args.swap,
              save_params_epochs=args.saveall,
              save_params=args.save,
              load_params=args.load)


if __name__ == '__main__':
    main()
