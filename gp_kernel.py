import numpy as np
import theano.tensor as T


def symbolic_kernel(t_diff, gp_params):
    return gp_params[0] * T.exp(-gp_params[1] * T.sqr(t_diff))


def kernel(t_diff, gp_params):
    return gp_params[0] * np.exp(-gp_params[1] * np.square(t_diff))
