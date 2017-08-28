import theano
from theano.gof import Op, Apply
import numpy

matrix_inverse = theano.tensor.nlinalg.MatrixInverse()

class LogAbsDet(Op):

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, (x,), (z,)):
        try:
            _, logabsdet = numpy.linalg.slogdet(x)
            z[0] = numpy.asarray(logabsdet, dtype=x.dtype)
        except Exception:
            print('Failed to compute determinant of {}.'.format(x))
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * matrix_inverse(x).T]

    def __str__(self):
        return "LogAbsDet"


logabsdet = LogAbsDet()
