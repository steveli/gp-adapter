import numpy as np
from gp_kernel import kernel, symbolic_kernel
from fast_gp import sparse_w, post_cov_ys, post_mean
from scipy.linalg import sqrtm
from theano_sqrtm import sqrtm as theano_sqrtm
import time
import theano
import theano.tensor as T
from theano.tensor.slinalg import cholesky
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pickle_io import pickle_load, pickle_save
from posterior_gp import PosteriorGP


rng = RandomStreams(seed=123)


def cov_mat(x1, x2, gp_params):
    x1_col = x1.dimshuffle(0, 'x')
    x2_row = x2.dimshuffle('x', 0)
    K = gp_params[0] * T.exp(-gp_params[1] * T.sqr(x1_col - x2_row))
    return K


def exact_post_mean(x, x_test, gp_params, indep_noise, y):
    Kxt = cov_mat(x, x_test, gp_params)
    Kxx = cov_mat(x, x, gp_params)
    Kxx = Kxx + indep_noise * T.identity_like(Kxx)
    KxtT_Kxxinv = Kxt.T.dot(T.nlinalg.matrix_inverse(Kxx))
    mu = KxtT_Kxxinv.dot(y)
    return mu


def exact_proj_cholesky(x, x_test, gp_params, indep_noise, batch_size):
    Ktt = cov_mat(x_test, x_test, gp_params)
    Kxt = cov_mat(x, x_test, gp_params)
    Kxx = cov_mat(x, x, gp_params)
    Kxx = Kxx + indep_noise * T.identity_like(Kxx)
    KxtT_Kxxinv = Kxt.T.dot(T.nlinalg.matrix_inverse(Kxx))
    K = Ktt - KxtT_Kxxinv.dot(Kxt)
    K = K + 1e-10 * T.identity_like(K)
    R = cholesky(K)
    eps = rng.normal(size=(batch_size, x_test.shape[0]))
    return R.dot(eps.T).T


def exact_proj_sqrtm(x, x_test, gp_params, indep_noise, batch_size):
    Ktt = cov_mat(x_test, x_test, gp_params)
    Kxt = cov_mat(x, x_test, gp_params)
    Kxx = cov_mat(x, x, gp_params)
    Kxx = Kxx + indep_noise * T.identity_like(Kxx)
    KxtT_Kxxinv = Kxt.T.dot(T.nlinalg.matrix_inverse(Kxx))
    K = Ktt - KxtT_Kxxinv.dot(Kxt)
    K = K + 1e-10 * T.identity_like(K)
    R = theano_sqrtm(K)
    eps = rng.normal(size=(batch_size, x_test.shape[0]))
    return R.dot(eps.T).T


def time_approx(x, y, x_test, gp_params, indep_noise, len_u,
                proj_1d, proj_2d, lanczos_basis=5, batch_size=1):
    extra_u = 2
    x_min, x_max = x.min(), x.max()
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)

    t_proj_1d = T.vector()
    t_proj_2d = T.matrix()

    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)
    post_gp = PosteriorGP(u, x_test, kernel, symbolic_kernel)
    post_gp.log_gp_params.set_value(np.log(gp_params))
    post_gp.log_indep_noise.set_value(np.log(indep_noise))

    mu = post_gp.mean()
    cov_zs = post_gp.cov_rand_proj(n_sample=batch_size,
                                   n_lanczos_basis=lanczos_basis)
    approx_gp_samples = mu.dimshuffle('x', 0) + cov_zs

    approx_fn = theano.function([post_gp.t_idx_train,
                                 post_gp.t_w_train,
                                 post_gp.t_idx_test,
                                 post_gp.t_w_test,
                                 post_gp.t_y_train], approx_gp_samples)

    approx_bp = theano.grad((approx_gp_samples * t_proj_2d).sum(),
                            post_gp.params)

    approx_bp_fn = theano.function([post_gp.t_idx_train,
                                    post_gp.t_w_train,
                                    post_gp.t_idx_test,
                                    post_gp.t_w_test,
                                    post_gp.t_y_train,
                                    t_proj_2d], approx_bp)

    approx_mu_fn = theano.function([post_gp.t_idx_train,
                                    post_gp.t_w_train,
                                    post_gp.t_idx_test,
                                    post_gp.t_w_test,
                                    post_gp.t_y_train], mu)

    approx_mu_bp_fn = theano.function([post_gp.t_idx_train,
                                       post_gp.t_w_train,
                                       post_gp.t_idx_test,
                                       post_gp.t_w_test,
                                       post_gp.t_y_train,
                                       t_proj_1d],
                                       theano.grad(mu.dot(t_proj_1d),
                                                   post_gp.params))

    t1 = time.time()
    approx_fn(idx_train, w_train, idx_test, w_test, y)
    t2 = time.time()
    print_label('approx sample')
    print t2 - t1
    sample_time = t2 - t1

    t1 = time.time()
    approx_bp_fn(idx_train, w_train, idx_test, w_test, y, proj_2d)
    t2 = time.time()
    print_label('approx sample grad')
    print t2 - t1
    grad_sample_time = t2 - t1

    t1 = time.time()
    approx_mu_fn(idx_train, w_train, idx_test, w_test, y)
    t2 = time.time()
    print_label('approx mean')
    print t2 - t1
    mean_time = t2 - t1

    t1 = time.time()
    approx_mu_bp_fn(idx_train, w_train, idx_test, w_test, y, proj_1d)
    t2 = time.time()
    print_label('approx mean grad')
    print t2 - t1
    grad_mean_time = t2 - t1

    return sample_time, grad_sample_time, mean_time, grad_mean_time


def time_exact(x, y, x_test, gp_params, indep_noise,
               proj_1d, proj_2d, batch_size=1):
    t_x = T.vector()
    t_x_test = T.vector()
    t_y = T.vector()
    t_proj_1d = T.vector()
    t_proj_2d = T.matrix()

    log_gp_params = theano.shared(np.log(gp_params))
    log_indep_noise = theano.shared(np.log(indep_noise))
    t_params = [log_gp_params, log_indep_noise]

    t_gp_params = T.exp(log_gp_params)
    t_indep_noise = T.exp(log_indep_noise)

    exact_mean = exact_post_mean(t_x, t_x_test,
                                 t_gp_params, t_indep_noise, t_y)

    exact_mean_fn = theano.function([t_x, t_x_test, t_y], exact_mean)
    exact_mean_bp_fn = theano.function([t_x, t_x_test, t_y, t_proj_1d],
            theano.grad(exact_mean.dot(t_proj_1d), t_params))

    t1 = time.time()
    exact_mean_fn(x, x_test, y)
    t2 = time.time()
    print_label('exact mean')
    print t2 - t1
    mean_time = t2 - t1

    t1 = time.time()
    exact_mean_bp_fn(x, x_test, y, proj_1d)
    t2 = time.time()
    print_label('exact mean grad')
    print t2 - t1
    grad_mean_time = t2 - t1

    cov_zs_cholesky = exact_proj_cholesky(t_x, t_x_test,
                                          t_gp_params, t_indep_noise,
                                          batch_size)
    gp_samples_cholesky = exact_mean.dimshuffle('x', 0) + cov_zs_cholesky

    cholesky_fn = theano.function([t_x, t_x_test, t_y],
                                  gp_samples_cholesky)

    cov_zs_sqrtm = exact_proj_sqrtm(t_x, t_x_test,
                                    t_gp_params, t_indep_noise,
                                    batch_size)
    gp_samples_sqrtm = exact_mean.dimshuffle('x', 0) + cov_zs_sqrtm

    sqrtm_fn = theano.function([t_x, t_x_test, t_y],
                               gp_samples_sqrtm)

    cholesky_bp = theano.grad((gp_samples_cholesky * t_proj_2d).sum(),
                              t_params)

    cholesky_bp_fn = theano.function([t_x, t_x_test, t_y, t_proj_2d],
                                     cholesky_bp)

    sqrtm_bp = theano.grad((gp_samples_sqrtm * t_proj_2d).sum(),
                           t_params)

    sqrtm_bp_fn = theano.function([t_x, t_x_test, t_y, t_proj_2d],
                                  sqrtm_bp)
    t1 = time.time()
    cholesky_fn(x, x_test, y)
    t2 = time.time()
    print_label('cholesky sample')
    print t2 - t1
    cholesky_sample_time = t2 - t1

    t1 = time.time()
    cholesky_bp_fn(x, x_test, y, proj_2d)
    t2 = time.time()
    print_label('cholesky sample grad')
    print t2 - t1
    grad_cholesky_sample_time = t2 - t1

    t1 = time.time()
    sqrtm_fn(x, x_test, y)
    t2 = time.time()
    print_label('sqrtm sample')
    print t2 - t1
    sqrtm_sample_time = t2 - t1

    t1 = time.time()
    sqrtm_bp_fn(x, x_test, y, proj_2d)
    t2 = time.time()
    print_label('sqrtm sample grad')
    print t2 - t1
    grad_sqrtm_sample_time = t2 - t1

    return (mean_time, grad_mean_time,
            cholesky_sample_time, grad_cholesky_sample_time,
            sqrtm_sample_time, grad_sqrtm_sample_time)


def print_label(s):
    print '%20s ' % s,


def gen_data(n_data=500, gp_params=(1, 10), indep_noise=.01):
    x = np.random.uniform(size=n_data)
    #x = np.float32(x)
    x = np.sort(x)
    a, b = gp_params
    c = indep_noise
    mu = np.zeros(n_data)
    cov = a * np.exp(-b * (x[:, np.newaxis] - x)**2) + c * np.eye(n_data)
    y = np.random.multivariate_normal(mu, cov)
    return x, y


def cmp_n_train_test(save_as=None, n_inducing_pts=512):
    np.random.seed(0)
    batch_size = 1

    results = []

    for n_data in xrange(500, 3000 + 1, 500):
        #n_data = 1000
        #n_data = 1000
        gp_params = np.array([.1, 100])
        indep_noise = .001
        x, y = gen_data(n_data, gp_params, indep_noise)

        x_min, x_max = x.min(), x.max()
        #len_u = 2048 + 1

        print '# data', n_data

        n_test = n_data
        x_test = np.linspace(x_min, x_max, n_test)

        proj_1d = np.random.uniform(-1, 1, size=len(x_test))
        proj_1d /= np.linalg.norm(proj_1d)

        proj_2d = np.random.uniform(-1, 1, size=(batch_size, len(x_test)))
        proj_2d /= np.linalg.norm(proj_2d, axis=1)[:, np.newaxis]

        t_exact = time_exact(x, y, x_test, gp_params, indep_noise,
                             proj_1d, proj_2d)

        # parameters for approximation algorithm
        len_u = n_inducing_pts

        proj_1d = np.random.uniform(-1, 1, size=len(x_test))
        proj_1d /= np.linalg.norm(proj_1d)

        proj_2d = np.random.uniform(-1, 1, size=(batch_size, len(x_test)))
        proj_2d /= np.linalg.norm(proj_2d, axis=1)[:, np.newaxis]

        t_approx = time_approx(x, y, x_test, gp_params, indep_noise, len_u,
                               proj_1d, proj_2d)
        print
        results.append(np.concatenate((t_exact, t_approx)))

    if save_as:
        pickle_save(save_as, np.vstack(results))


def cmp_n_test():
    np.random.seed(0)
    #n_data = 500
    n_data = 1000
    #n_data = 1000
    gp_params = np.array([.1, 100])
    indep_noise = .001
    x, y = gen_data(n_data, gp_params, indep_noise)

    x_min, x_max = x.min(), x.max()
    #len_u = 2048 + 1

    batch_size = 1

    for n_test in xrange(100, 1600, 100):
        print '# test', n_test

        x_test = np.linspace(x_min, x_max, n_test)

        proj_1d = np.random.uniform(-1, 1, size=len(x_test))
        proj_1d /= np.linalg.norm(proj_1d)

        proj_2d = np.random.uniform(-1, 1, size=(batch_size, len(x_test)))
        proj_2d /= np.linalg.norm(proj_2d, axis=1)[:, np.newaxis]

        time_exact(x, y, x_test, gp_params, indep_noise, proj_1d, proj_2d)

        # parameters for approximation algorithm
        len_u = 512

        proj_1d = np.random.uniform(-1, 1, size=len(x_test))
        proj_1d /= np.linalg.norm(proj_1d)

        proj_2d = np.random.uniform(-1, 1, size=(batch_size, len(x_test)))
        proj_2d /= np.linalg.norm(proj_2d, axis=1)[:, np.newaxis]

        time_approx(x, y, x_test, gp_params, indep_noise, len_u,
                    proj_1d, proj_2d)
        print


def cmp_n_inducing_points():
    np.random.seed(0)
    #n_data = 500
    n_data = 2000
    #n_data = 1000
    gp_params = np.array([.1, 100])
    indep_noise = .001
    x, y = gen_data(n_data, gp_params, indep_noise)

    x_min, x_max = x.min(), x.max()
    #len_u = 2048 + 1

    batch_size = 1

    x_test = np.linspace(x_min, x_max, 1000)

    proj_1d = np.random.uniform(-1, 1, size=len(x_test))
    proj_1d /= np.linalg.norm(proj_1d)

    proj_2d = np.random.uniform(-1, 1, size=(batch_size, len(x_test)))
    proj_2d /= np.linalg.norm(proj_2d, axis=1)[:, np.newaxis]

    time_exact(x, y, x_test, gp_params, indep_noise, proj_1d, proj_2d)

    for log_u in xrange(3, 13):
        # parameters for approximation algorithm
        len_u = 2**log_u + 1
        print log_u, 2**log_u

        proj_1d = np.random.uniform(-1, 1, size=len(x_test))
        proj_1d /= np.linalg.norm(proj_1d)

        proj_2d = np.random.uniform(-1, 1, size=(batch_size, len(x_test)))
        proj_2d /= np.linalg.norm(proj_2d, axis=1)[:, np.newaxis]

        time_approx(x, y, x_test, gp_params, indep_noise, len_u,
                    proj_1d, proj_2d)
        print

        #print '-' * 40
        #print (cov_zs**2).sum(axis=0)
        #print (cov_zs_exact**2).sum(axis=0)


if __name__ == '__main__':
    #cmp_n_inducing_points()
    #cmp_n_test()
    n_inducing_pts = 256
    for i in xrange(10):
        cmp_n_train_test('results/cmpsize-L5-%d-%02d.pkl' % (n_inducing_pts, i),
                         n_inducing_pts)
