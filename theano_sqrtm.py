import numpy as np
import scipy.linalg
import theano
from theano.tensor import as_tensor_variable
import theano.tests.unittest_tools
from theano.gof import Op, Apply


class MatrixSquareRoot(Op):
    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x, = inputs
        z, = outputs
        z[0] = scipy.linalg.sqrtm(x).real

    def grad(self, inputs, g_outputs):
        x, = inputs
        gz, = g_outputs
        return [MatrixSquareRootGrad()(self(x), gz)]

    def infer_shape(self, node, shapes):
        return shapes


sqrtm = MatrixSquareRoot()


class MatrixSquareRootGrad(Op):
    def make_node(self, sqrtx, gz):
        sqrtx = as_tensor_variable(sqrtx)
        gz = as_tensor_variable(gz)
        assert sqrtx.ndim == 2
        assert gz.ndim == 2
        return Apply(self, [sqrtx, gz], [sqrtx.type()])

    def perform(self, node, inputs, outputs):
        sqrtx, gz = inputs
        z, = outputs
        z[0] = scipy.linalg.solve_sylvester(sqrtx, sqrtx, gz)

    def infer_shape(self, node, shapes):
        return [shapes[0]]


def main():
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
