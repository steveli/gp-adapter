import numpy as np
import scipy.linalg
import theano
from theano.tensor import as_tensor_variable
import theano.tensor as T
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType
from fast_gp import post_cov_ys, post_mean, grad_cov_ys
import theano.tests.unittest_tools


def cov_mat(x, y, gp_params):
    a, b = gp_params
    return a * np.exp(-b * (x[:, np.newaxis] - y)**2)


class CovVec(Op):
    def __init__(self, inducing_points, kernel_numpy, kernel_jacobian):
        self.inducing_points = inducing_points
        self.kernel_numpy = kernel_numpy
        self.kernel_jacobian = kernel_jacobian

    def make_node(self, idx_train, w_train, idx_test, w_test,
                  gp_params, indep_noise, ys):
        idx_train = as_tensor_variable(idx_train)
        w_train = as_tensor_variable(w_train)
        idx_test = as_tensor_variable(idx_test)
        w_test = as_tensor_variable(w_test)
        gp_params = as_tensor_variable(gp_params)
        indep_noise = as_tensor_variable(indep_noise)
        ys = as_tensor_variable(ys)
        return Apply(self, [idx_train, w_train, idx_test, w_test,
                            gp_params, indep_noise, ys], [ys.type()])

    def perform(self, node, inputs, outputs):
        (idx_train, w_train, idx_test, w_test,
                gp_params, indep_noise, ys) = inputs
        z, = outputs
        t_diff = self.inducing_points - self.inducing_points[0]
        u = self.kernel_numpy(t_diff, gp_params)
        len_u = len(u)
        z[0] = post_cov_ys(idx_train, w_train, idx_test, w_test,
                           u, len_u, indep_noise, ys)

    def grad(self, inputs, g_outputs):
        (idx_train, w_train, idx_test, w_test,
                gp_params, indep_noise, ys) = inputs
        gz, = g_outputs
        ys_shape = ys.shape

        t_diff = self.inducing_points - self.inducing_points[0]
        # symbolic kernel computation
        u = gp_params[0] * T.exp(-gp_params[1] * t_diff**2)
        grad_u = theano.gradient.jacobian(u, gp_params)
        flat_ys = ys.flatten()
        grad_ys_params, grad_ys_noise = theano.gradient.jacobian(flat_ys,
                [gp_params, indep_noise], disconnected_inputs='ignore')

        grad_ys_noise = grad_ys_noise.reshape(ys_shape)
        v = CovVecGrad(self.inducing_points)(
                    idx_train, w_train, idx_test, w_test,
                    u, gp_params, indep_noise, ys, gz, grad_u,
                    grad_ys_params, grad_ys_noise)
        return ([DisconnectedType()(),      # idx_train
                 DisconnectedType()(),      # w_train
                 DisconnectedType()(),      # idx_test
                 DisconnectedType()()] +    # w_test
                v +
                [DisconnectedType()()])     # ys

    def connection_pattern(self, node):
        return [[False], [False], [False], [False], [True], [True], [False]]

    def infer_shape(self, node, shapes):
        return [shapes[-1]]


class CovVecGrad(Op):
    def __init__(self, inducing_points):
        self.inducing_points = inducing_points

    def make_node(self, idx_train, w_train, idx_test, w_test,
                  u, gp_params, indep_noise, ys, gz, grad_u,
                  grad_ys_params, grad_ys_noise):
        idx_train = as_tensor_variable(idx_train)
        w_train = as_tensor_variable(w_train)
        idx_test = as_tensor_variable(idx_test)
        w_test = as_tensor_variable(w_test)
        u = as_tensor_variable(u)
        gp_params = as_tensor_variable(gp_params)
        indep_noise = as_tensor_variable(indep_noise)
        ys = as_tensor_variable(ys)
        gz = as_tensor_variable(gz)
        grad_u = as_tensor_variable(grad_u)
        grad_ys_params = as_tensor_variable(grad_ys_params)
        grad_ys_noise = as_tensor_variable(grad_ys_noise)
        return Apply(self, [idx_train, w_train, idx_test, w_test,
                            u, gp_params, indep_noise, ys, gz,
                            grad_u, grad_ys_params, grad_ys_noise],
                     [gp_params.type(), indep_noise.type()])

    def perform(self, node, inputs, outputs):
        (idx_train, w_train, idx_test, w_test,
              u, gp_params, indep_noise, ys, gz, grad_u,
              grad_ys_params, grad_ys_noise) = inputs
        # gz.shape: (#z, #test)
        grad_z_gp_params = np.empty_like(gp_params)
        n_gp_params = len(gp_params)
        grad_gp_params, grad_noise = grad_cov_ys(
                idx_train, w_train, idx_test, w_test,
                u, gp_params, indep_noise, ys, grad_u,
                grad_ys_params, grad_ys_noise)

        for i in xrange(n_gp_params):
            grad_z_gp_params[i] = (gz * grad_gp_params[i]).sum()
        grad_z_noise = (gz * grad_noise).sum()

        outputs[0][0] = grad_z_gp_params
        outputs[1][0] = grad_z_noise

    def infer_shape(self, node, shapes):
        return [shapes[5], shapes[6]]


class PosteriorMean(Op):
    def __init__(self, inducing_points, kernel_numpy, kernel_jacobian):
        self.inducing_points = inducing_points
        self.kernel_numpy = kernel_numpy
        self.kernel_jacobian = kernel_jacobian

    def make_node(self, idx_train, w_train, idx_test, w_test,
                  gp_params, indep_noise, y):
        idx_train = as_tensor_variable(idx_train)
        w_train = as_tensor_variable(w_train)
        idx_test = as_tensor_variable(idx_test)
        w_test = as_tensor_variable(w_test)
        gp_params = as_tensor_variable(gp_params)
        indep_noise = as_tensor_variable(indep_noise)
        y = as_tensor_variable(y)
        return Apply(self, [idx_train, w_train, idx_test, w_test,
                            gp_params, indep_noise, y], [y.type()])

    def perform(self, node, inputs, outputs):
        (idx_train, w_train, idx_test, w_test,
                gp_params, indep_noise, y) = inputs
        z, = outputs
        t_diff = self.inducing_points - self.inducing_points[0]
        u = self.kernel_numpy(t_diff, gp_params)
        len_u = len(u)
        z[0] = post_mean(idx_train, w_train, idx_test, w_test,
                         u, len_u, indep_noise, y)

    def grad(self, inputs, g_outputs):
        (idx_train, w_train, idx_test, w_test,
                gp_params, indep_noise, ys) = inputs
        t_diff = self.inducing_points - self.inducing_points[0]
        u = self.kernel_numpy(t_diff, gp_params)
        gz, = g_outputs
        return None

    def infer_shape(self, node, shapes):
        return [shapes[-1]]


def test_gp():
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
    #len_u = 1024 + 1
    len_u = 64
    extra_u = 2
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)
    #x_test = u[1:]
    x_test = np.linspace(x_min, x_max, 20)
    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)
    len_test = len(x_test)
    y_test = np.random.uniform(size=(2, len_test))
    #y_test = np.random.uniform(size=(1, len_test))

    def kernel_numpy(t_diff, gp_params):
        a, b = gp_params
        return a * np.exp(-b * t_diff**2)

    cov_vec = CovVec(u, kernel_numpy, None)

    t_idx_train = theano.tensor.imatrix()
    t_w_train = theano.tensor.matrix()
    t_idx_test = theano.tensor.imatrix()
    t_w_test = theano.tensor.matrix()
    t_gp_params = theano.tensor.vector()
    t_indep_noise = theano.tensor.scalar()
    t_ys = theano.tensor.matrix()
    t_y = theano.tensor.vector()

    v = cov_vec(t_idx_train, t_w_train, t_idx_test, t_w_test,
                t_gp_params, t_indep_noise, t_ys)
    ys1 = t_ys - v
    v = cov_vec(t_idx_train, t_w_train, t_idx_test, t_w_test,
                t_gp_params, t_indep_noise, ys1)
    v_fn = theano.function([t_idx_train, t_w_train, t_idx_test, t_w_test,
                            t_gp_params, t_indep_noise, t_ys], v)

    def vf(t_gp_params, t_indep_noise):
        v = cov_vec(idx_train, w_train, idx_test, w_test,
                    t_gp_params, t_indep_noise, y_test)
        ys1 = -y_test + v
        v = cov_vec(idx_train, w_train, idx_test, w_test,
                    t_gp_params, t_indep_noise, ys1)
        ys1 = y_test - v
        v = cov_vec(idx_train, w_train, idx_test, w_test,
                    t_gp_params, t_indep_noise, ys1)
        return v

    print 'verify grad ##'
    theano.tests.unittest_tools.verify_grad(vf, [(a, b), c],
                                            n_tests=5, eps=1.0e-5,
                                            abs_tol=0.001, rel_tol=0.001)
    print '###'
    vsum = v.sum()

    vsum_fn = theano.function([t_idx_train, t_w_train,
                               t_idx_test, t_w_test,
                               t_gp_params, t_indep_noise, t_ys],
                               vsum)
    v_ = vsum_fn(idx_train, w_train, idx_test, w_test, (a, b), c, y_test)

    grad_vsum = theano.grad(vsum, [t_gp_params, t_indep_noise])
    grad_vsum_fn = theano.function([t_idx_train,
                                    t_w_train,
                                    t_idx_test,
                                    t_w_test,
                                    t_gp_params,
                                    t_indep_noise,
                                    t_ys
                                    ],
                                   grad_vsum,
                                   on_unused_input='ignore'
                                   )
    grad_v =  grad_vsum_fn(idx_train, w_train, idx_test, w_test, (a, b), c,
                           y_test)
    print 'grad v'
    print grad_v

    def sub_cov_vec(t_gp_params, t_indep_noise):
        return cov_vec(idx_train, w_train, idx_test, w_test,
                       t_gp_params, t_indep_noise, y_test)

    print 'verify grad'
    theano.tests.unittest_tools.verify_grad(sub_cov_vec, [(a, b), c],
                                            n_tests=5, eps=1.0e-5,
                                            abs_tol=0.001, rel_tol=0.001)

    v_fn = theano.function([t_idx_train, t_w_train, t_idx_test, t_w_test,
                            t_gp_params, t_indep_noise, t_ys], v)

    v_ = v_fn(idx_train, w_train, idx_test, w_test, (a, b), c, y_test)
    print 'v'
    print v_
    #return
    var = v_fn(idx_train, w_train, idx_test, w_test, (a, b), c,
               np.eye(len_test))
    var = np.diag(var)

    post_mean = PosteriorMean(u, kernel_numpy, None)
    pmean = post_mean(t_idx_train, t_w_train, t_idx_test, t_w_test,
                   t_gp_params, t_indep_noise, t_y)
    pmean_fn = theano.function([t_idx_train, t_w_train, t_idx_test, t_w_test,
                             t_gp_params, t_indep_noise, t_y], pmean)
    pmu = pmean_fn(idx_train, w_train, idx_test, w_test, (a, b), c, y)

    #print var

    import pylab as pl
    pl.figure()
    std2 = np.sqrt(var) * 2

    color = 'b'
    pl.fill_between(x_test, pmu - std2, pmu + std2, color=color,
                    edgecolor='none', alpha=.3)
    pl.plot(x_test, pmu, '-', c=color)
    pl.plot(x, y, 'o', c=color)
    pl.show()


def main():
    test_gp()
    return
    x = theano.tensor.matrix()
    sum_x = sqrtm(x).sum()
    sum_x_fn = theano.function([x], sum_x)

    n = 50
    L = np.random.uniform(-1, 1, size=(n, n + 500)) * .1
    cov = L.dot(L.T) + np.eye(n) * .5
    print sum_x_fn(cov)

    grad = theano.grad(sum_x, x)
    grad_fn = theano.function([x], grad)
    print grad_fn(cov)

    from lanczos_theano import reg_cov_mat
    for i in xrange(10):
        cov = reg_cov_mat(np.random.uniform(0, 1, size=n), 1, 8, .1)
        theano.tests.unittest_tools.verify_grad(sqrtm, [cov])


if __name__ == '__main__':
    main()
