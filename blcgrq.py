import numpy as np
from scipy.linalg import qr


def block_cg(Ax_fn, B, tol=1e-10, maxiter=100):
    """
    Implementation of "Retooling the method of block conjugate gradients"
    by Augustin A Dubrulle.
    """
    n, m = B.shape
    overcomplete = False
    if n < m:
        overcomplete = True
        B, R, perm = qr(B, mode='economic', pivoting=True)
        m = n
    norm_B = np.linalg.norm(B)
    X = np.zeros_like(B)
    Q, C = qr(B, mode='economic')
    S = np.eye(m)
    P = np.zeros_like(B)

    for i in xrange(maxiter):
        P = Q + P.dot(S.T)
        AP = Ax_fn(P)
        T = np.linalg.inv(P.T.dot(AP))
        X = X + P.dot(T).dot(C)
        V = Q - AP.dot(T)
        Q, S = qr(V, mode='economic')
        C = S.dot(C)
        err = np.linalg.norm(C) / norm_B
        if err < tol:
            break

    if overcomplete:
        return X.dot(R)[:, np.argsort(perm)]
    return X


def main():
    #np.random.seed(11)
    n = 1000
    m = 2000
    A = np.random.normal(size=(n * 2, n))
    A = A.T.dot(A)
    B = np.random.normal(size=(n, m))
    print 'start'
    import time
    t1 = time.time()
    def Ax(x):
        return A.dot(x)
    U = block_cg(Ax, B)
    t2 = time.time()
    print t2 - t1
    t1 = time.time()
    V = np.linalg.solve(A, B)
    t2 = time.time()
    print t2 - t1
    #print U - V
    print np.allclose(U, V)


def test_dep():
    n = 10
    a0 = np.random.normal(size=n)
    a1 = np.random.normal(size=n)
    A = np.random.normal(size=(n, n))
    A = A.T.dot(A)
    B = np.vstack((a0, a1, a0 + a1, a0 - 2 * a1)).T
    def Ax(x):
        return A.dot(x)
    U = block_cg(Ax, B)
    V = np.linalg.solve(A, B)
    print U
    print V
    #print U - V
    print np.allclose(U, V)


if __name__ == '__main__':
    main()
    test_dep()
