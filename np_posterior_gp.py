import numpy as np
from gp_kernel import kernel
from fast_gp import sparse_w, post_cov_ys, post_mean
from scipy.linalg import sqrtm
import time
from pickle_io import pickle_load


def lanczos(linear_op, z, m, batch_size):

    batch_size = z.shape[0]
    v = z / np.linalg.norm(z, axis=1)[:, np.newaxis]

    alpha = []
    beta = []
    V = []
    V.append(v)

    v_prev = None
    v_curr = v
    b = None

    for j in xrange(m):
        if j == 0:
            r = linear_op(v_curr)
        else:
            r = linear_op(v_curr) - b[:, np.newaxis] * v_prev

        a = (v_curr * r).sum(axis=1)
        r = r - a[:, np.newaxis] * v_curr
        b = np.linalg.norm(r, axis=1)
        v_prev = v_curr
        v_curr = r / b[:, np.newaxis]
        alpha.append(a)
        if j < m - 1:
            V.append(v_curr)
            beta.append(b)

    b = np.linalg.norm(z, axis=1)
    Az_list = []
    for idx in xrange(batch_size):
        alpha_diag = np.diag([a_[idx] for a_ in alpha])
        beta_diag = np.array([b_[idx] for b_ in beta])
        M = alpha_diag + np.diag(beta_diag, 1) + np.diag(beta_diag, -1)
        V_matrix = np.vstack([v_[idx] for v_ in V]).T
        Az_approx = b[idx] * V_matrix.dot(sqrtm(M)[:, 0])
        Az_list.append(Az_approx)

    return np.vstack(Az_list)


class PosteriorGP(object):
    def __init__(self, inducing_pts, t_test,
                 t_train, y_train, kernel, gp_params, indep_noise,
                 random_seed=101):
        self.rng = np.random.RandomState(random_seed)
        self.len_test = len(t_test)
        self.gp_params = gp_params
        self.indep_noise = indep_noise
        self.idx_train, self.w_train = sparse_w(inducing_pts, t_train)
        self.idx_test, self.w_test = sparse_w(inducing_pts, t_test)
        self.y_train = y_train
        t_diff = inducing_pts - inducing_pts[0]
        self.u = kernel(t_diff, gp_params)
        self.len_u = len(inducing_pts)
        self.gp_params = gp_params
        self.indep_noise = indep_noise

    def mean(self):
        mu = post_mean(self.idx_train, self.w_train,
                       self.idx_test, self.w_test,
                       self.u, len(self.u), self.indep_noise, self.y_train)
        return mu

    def cov_rand_proj(self, n_sample=10, n_lanczos_basis=10, eps=None):
        def linear_op(zs):
            return post_cov_ys(self.idx_train, self.w_train,
                               self.idx_test, self.w_test,
                               self.u, self.len_u, self.indep_noise, zs)
        if eps is None:
            eps = self.rng.normal(size=(n_sample, self.len_test))
        else:
            n_sample = eps.shape[0]
        cov_zs = lanczos(linear_op, eps, n_lanczos_basis, n_sample)
        return cov_zs


def main():
    np.random.seed(0)
    n_data = 10
    x = np.random.uniform(size=n_data)
    #x = np.float32(x)
    x = np.sort(x)
    a = 1
    b = 10
    c = .01
    mu = np.zeros(n_data)
    cov = a * np.exp(-b * (x[:, np.newaxis] - x)**2) + c * np.eye(n_data)
    y = np.random.multivariate_normal(mu, cov)

    #data = 'ECG200/11_1000_60_dat.pkl'
    #gp_parms, ts_train, ts_test, l_train, l_test = pickle_load(data)[:5]
    #x_train = np.array([each_ts[0] for each_ts in ts_train])
    #y_train = np.array([each_ts[1] for each_ts in ts_train])
    #eg_id = 0
    #x = x_train[eg_id]
    #y = y_train[eg_id]

    #print x
    #print y
    x_min, x_max = x.min(), x.max()
    #len_u = 2048 + 1
    len_u = 1024 + 1
    #len_u = 50
    #len_u = 128 + 1
    #len_u = 64
    extra_u = 2
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)
    x_test = u[1:]
    #x_test = np.linspace(x_min, x_max, 20)

    gp_params = np.array([a, b])
    indep_noise = c

    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)
    post_gp = PosteriorGP(u, x_test, x, y, kernel, gp_params, indep_noise)

    batch_size = 10
    cov_zs = post_gp.cov_rand_proj(n_sample=batch_size, n_lanczos_basis=10)
    mu = post_gp.mean()
    gp_samples = mu + cov_zs

    #return

    import pylab as pl
    pl.figure()

    for each_sample in gp_samples:
        pl.plot(x_test, each_sample, '-', c='b', alpha=.5)
    pl.plot(x, y, 'o', c='r')
    pl.show()


if __name__ == '__main__':
    main()
