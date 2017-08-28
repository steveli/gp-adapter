import numpy as np
from numpy.linalg import norm
from scipy.linalg import sqrtm
import theano.tensor as T
import theano
from theano_sqrtm import sqrtm as theano_sqrtm


def cov_mat(x1, x2, a, b):
    return a * np.exp(-b * (x1[:, np.newaxis] - x2)**2)


def reg_cov_mat(x, a, b, c):
    return cov_mat(x, x, a, b) + c * np.eye(x.shape[0])


def mat_fn(M):
    return sqrtm(M).real


def lanczos(A, v, m, z):
    n = A.shape[0]
    alpha = np.zeros(m)
    beta = np.zeros(m)
    V = np.empty((n, m))
    V[:, 0] = v
    v_prev = np.zeros(n)
    v_curr = v
    b = 0
    for j in xrange(m):
        v = A.dot(v_curr) - b * v_prev
        a = v_curr.dot(v)
        v = v - a * v_curr
        b = norm(v)
        print b
        v_prev = v_curr
        v_curr = v / b
        alpha[j] = a
        if j < m - 1:
            V[:, j + 1] = v_curr
            beta[j + 1] = b

    M = np.diag(alpha) + np.diag(beta[1:], 1) + np.diag(beta[1:], -1)
    b = norm(z)
    #print M
    print 'compute approx'
    Az_approx = b * V.dot(mat_fn(M)[:, 0])
    print 'done'
    Az = mat_fn(A).dot(z)
    print Az.min(), Az.max()
    print np.fabs(Az - Az_approx).max()
    print norm(Az - Az_approx)
    print Az[:5]
    print (Az.T.dot(Az) - Az_approx.T.dot(Az_approx)).max()
    print np.fabs(Az.T.dot(Az) - Az_approx.T.dot(Az_approx)).max()
    print norm(Az.T.dot(Az) - Az_approx.T.dot(Az_approx))
    return Az_approx


def lanczos_theano(A_, v_, m, z_):
    A = T.matrix()
    v = T.vector()
    z = T.vector()

    alpha = []
    beta = []
    V = []
    V.append(v)
    v_curr = v
    b = 0
    v_prev = None

    for j in xrange(m):
        if j == 0:
            r = A.dot(v_curr)
        else:
            r = A.dot(v_curr) - b * v_prev
        a = v_curr.dot(r)
        r = r - a * v_curr
        b = r.norm(2)
        v_prev = v_curr
        v_curr = r / b
        alpha.append(a)
        if j < m - 1:
            V.append(v_curr)
            beta.append(b)

    alpha_diag = T.diag(T.stacklists(alpha))
    beta_diag = T.diag(T.stacklists(beta + [0]))

    M = alpha_diag + T.roll(beta_diag, 1, 0) + T.roll(beta_diag, 1, 1)
    M_fn = theano.function([A, v], M)
    print 'M'
    print M_fn(A_, v_)
    b = z.norm(2)
    V_matrix = T.stacklists(V).T
    V_fn = theano.function([A, v], V_matrix)
    print 'V'
    print V_fn(A_, v_).shape
    approx_sqrt = b * V_matrix.dot(theano_sqrtm(M)[:, 0])
    #approx_sqrt_fn = theano.function([A, v, z], T.grad(approx_sqrt.sum(), [v]))
    approx_sqrt_fn = theano.function([A, v, z], approx_sqrt)
    Az_approx = approx_sqrt_fn(A_, v_, z_)
    return Az_approx


def main():
    n = 2000
    #L = np.random.uniform(-1, 1, size=(n, n + 500)) * .1
    #cov = L.dot(L.T) + np.eye(n) * .5
    cov = reg_cov_mat(np.random.uniform(0, 1, size=n), 1, 8, .1)
    m = 5
    z = np.random.uniform(-1, 1, size=n)
    a = lanczos(cov, z / norm(z), m, z)
    b = lanczos_theano(cov, z / norm(z), m, z)
    print np.linalg.norm(a - b)


if __name__ == '__main__':
    main()

