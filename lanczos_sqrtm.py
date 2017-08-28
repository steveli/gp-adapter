import numpy as np
from scipy.linalg import sqrtm as scipy_sqrtm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
import theano
from theano_sqrtm import sqrtm as theano_sqrtm


def lanczos(linear_op, z, m, batch_size):
    s = z.norm(2, axis=1)
    v = z / s.dimshuffle(0, 'x')

    alpha = []
    beta = []
    V = []
    V.append(v)
    v_curr = v
    b = None
    v_prev = None

    for j in xrange(m):
        if j == 0:
            r = linear_op(v_curr)
        else:
            r = linear_op(v_curr) - b.dimshuffle(0, 'x') * v_prev
        a = T.batched_dot(v_curr, r)
        r = r - a.dimshuffle(0, 'x') * v_curr
        b = r.norm(2, axis=1)
        v_prev = v_curr
        v_curr = r / b.dimshuffle(0, 'x')
        alpha.append(a)
        if j < m - 1:
            V.append(v_curr)
            beta.append(b)

    Az_list = []
    for idx in xrange(batch_size):
        alpha_diag = T.diag(T.stacklists([a_[idx] for a_ in alpha]))
        beta_diag = T.diag(T.stacklists([b_[idx] for b_ in beta] + [0]))
        M = alpha_diag + T.roll(beta_diag, 1, 0) + T.roll(beta_diag, 1, 1)
        V_matrix = T.stacklists([v_[idx] for v_ in V]).T
        approx_sqrt = s[idx] * V_matrix.dot(theano_sqrtm(M)[:, 0])
        Az_list.append(approx_sqrt)

    Azs = T.stacklists(Az_list)

    return Azs


def sqrtm(A):
    return scipy_sqrtm(A).real


def cov_mat(x1, x2, a, b):
    return a * np.exp(-b * (x1[:, np.newaxis] - x2)**2)


def reg_cov_mat(x, a, b, c):
    return cov_mat(x, x, a, b) + c * np.eye(x.shape[0])


def main():
    n = 1024
    #n = 10
    #L = np.random.uniform(-1, 1, size=(n, n + 500)) * .1
    #cov = L.dot(L.T) + np.eye(n) * .5
    cov = reg_cov_mat(np.random.uniform(0, 1, size=n), 1, 8, .1)
    m = 10
    batch_size = 10
    z = np.random.normal(size=(batch_size, n))
    #lanczos(cov, z, m, batch_size)

    A_symb = T.matrix()
    z_symb = T.matrix()

    def Ky(vec):
        return A_symb.dot(vec.T).T

    Azs = lanczos(Ky, z_symb, m, batch_size)

    Azs_fn = theano.function([A_symb, z_symb], Azs)
    Azs_approx = Azs_fn(cov, z)
    Azs_exact = sqrtm(cov).dot(z.T).T

    print np.linalg.norm(Azs_exact - Azs_approx)
    print np.fabs(Azs_exact - Azs_approx).max()


if __name__ == '__main__':
    main()

