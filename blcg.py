import numpy as np
from scipy.linalg import qr


def block_cg(Ax_fn, B, X0=None, eps=1e-12, maxiter=100):
    """
    Implementation of "A block conjugate gradient method applied to
    linear systems with multiple right-hand sides", Y.T. Feng, et al. (1995)
    """
    n = B.shape[0]
    if X0 is None:
        X0 = np.ones_like(B) * B.mean()
    B_orig, X0_orig = B, X0

    R = B - Ax_fn(X0)
    Q, M, perm = qr(R, pivoting=True, mode='economic')
    print 'perm', perm
    X0 = X0[:, perm]
    B = B[:, perm]

    if B.shape[1] > n:
        X0 = X0[:, :n]
        B = B[:, :n]

    sel_idx = np.fabs(np.diag(M)) > 1e-10
    print np.diag(M)
    dim = sel_idx.sum()
    print 'dim', dim
    if dim == 0:
        return X0_orig

    X0 = X0[:, sel_idx]
    B = B[:, sel_idx]
    Q = Q[:, sel_idx]
    M = M[sel_idx]

    #print np.diag(M)
    P = Q
    R = Q
    RR = R.T.dot(R)
    X = np.zeros_like(B)
    idx = np.arange(dim)
    X_all = np.empty_like(X)

    for i in xrange(maxiter):
        AP = Ax_fn(P)
        V = np.linalg.solve(P.T.dot(AP), RR)
        X = X + P.dot(V)
        R_new = R - AP.dot(V)

        # Check convergence
        res = np.linalg.norm(R_new, axis=0)
        print res
        conv_idx = res < eps
        if np.any(conv_idx):
            X_all[:, idx[conv_idx]] = X[:, conv_idx]
            if np.all(conv_idx):
                break
            keep_idx = np.logical_not(conv_idx)
            P = P[:, keep_idx]
            X = X[:, keep_idx]
            RR = RR[keep_idx][:, keep_idx]
            R_new = R_new[:, keep_idx]
            idx = idx[keep_idx]

        RR_new = R_new.T.dot(R_new)
        S = np.linalg.solve(RR, RR_new)
        P = R_new + P.dot(S)

        R = R_new
        RR = RR_new

    print i

    return X0_orig + X_all.dot(M)[:, np.argsort(perm)]


def reorder(Ax_fn, B, X0):
    n, m = B.shape
    R = B - Ax_fn(X0)
    Q, M = np.linalg.qr(R)
    print M.diagonal()
    idx = np.argsort(M.diagonal())[::-1]
    return B[:, idx], X0[:, idx]


def main():
    #np.random.seed(11)
    n = 1000
    m = 2000
    A = np.random.normal(size=(n * 2, n))
    A = A.T.dot(A)
    B = np.random.normal(size=(n, m))
    X0 = np.ones_like(B)
    #X0 = np.random.normal(size=B.shape)
    print 'start'
    import time
    t1 = time.time()
    def Ax(x):
        return A.dot(x)
    X0 = np.linalg.solve(A, B)
    U = block_cg(Ax, B, X0)
    t2 = time.time()
    print t2 - t1
    t1 = time.time()
    V = np.linalg.solve(A, B)
    t2 = time.time()
    print t2 - t1
    #print U - V
    print np.allclose(U, V)


if __name__ == '__main__':
    main()
