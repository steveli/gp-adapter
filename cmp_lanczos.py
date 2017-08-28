import numpy as np
from gp_kernel import kernel
from fast_gp import sparse_w, post_cov_ys, post_mean
from scipy.linalg import sqrtm
import time
from pickle_io import pickle_load, pickle_save
from np_posterior_gp import PosteriorGP


def cov_mat(x1, x2, gp_params):
    a, b = gp_params
    return a * np.exp(-b * (x1[:, np.newaxis] - x2)**2)


def exact_proj_cholesky(x, y, x_test, gp_params, indep_noise, batch_size,
                        eps=None):
    Ktt = cov_mat(x_test, x_test, gp_params)
    Kxt = cov_mat(x, x_test, gp_params)
    Kxx = cov_mat(x, x, gp_params) + indep_noise * np.eye(len(x))
    KxtT_Kxxinv = np.linalg.solve(Kxx, Kxt).T
    K = Ktt - KxtT_Kxxinv.dot(Kxt) + 1e-12 * np.eye(len(x_test))
    R = np.linalg.cholesky(K)
    if eps is None:
        eps = np.random.normal(size=(batch_size, len(x_test)))
    return R.dot(eps.T).T


def exact_proj_sqrtm(x, y, x_test, gp_params, indep_noise, batch_size,
                        eps=None):
    Ktt = cov_mat(x_test, x_test, gp_params)
    Kxt = cov_mat(x, x_test, gp_params)
    Kxx = cov_mat(x, x, gp_params) + indep_noise * np.eye(len(x))
    KxtT_Kxxinv = np.linalg.solve(Kxx, Kxt).T
    K = Ktt - KxtT_Kxxinv.dot(Kxt) + 1e-12 * np.eye(len(x_test))
    R = sqrtm(K)
    if eps is None:
        eps = np.random.normal(size=(batch_size, len(x_test)))
    return R.dot(eps.T).T


def cov_sor(x, z, u, gp_params):
    Kxu = cov_mat(x, u, gp_params)
    Kuu = cov_mat(u, u, gp_params) #+ np.eye(len(u)) * 1e-12
    Kuz = cov_mat(u, z, gp_params)
    return Kxu.dot(np.linalg.solve(Kuu, Kuz))


def exact_gp_sample(x, y, x_test, gp_params, indep_noise, eps,
                    decomp='sqrtm', return_mean=False):
    Ktt = cov_mat(x_test, x_test, gp_params)
    Kxt = cov_mat(x, x_test, gp_params)
    Kxx = cov_mat(x, x, gp_params) + indep_noise * np.eye(len(x))
    #print 'condition number', np.linalg.cond(Kxx)
    KxtT_Kxxinv = np.linalg.solve(Kxx, Kxt).T
    K = Ktt - KxtT_Kxxinv.dot(Kxt) + 1e-12 * np.eye(len(x_test))

    if decomp == 'sqrtm':
        R = sqrtm(K).real
    elif decomp == 'cholesky':
        R = np.linalg.cholesky(K)
    else:
        raise NotImplementedError('Undefined decomposition method.')

    mu = KxtT_Kxxinv.dot(y)
    sample = mu + R.dot(eps.T).T
    if return_mean:
        return mu, sample
    return sample


def sor_gp_sample(x, y, x_test, u, gp_params, indep_noise, eps):
    Ktt = cov_sor(x_test, x_test, u, gp_params)
    Kxt = cov_sor(x, x_test, u, gp_params)
    Kxx = cov_sor(x, x, u, gp_params) + indep_noise * np.eye(len(x))
    KxtT_Kxxinv = np.linalg.solve(Kxx, Kxt).T
    K = Ktt - KxtT_Kxxinv.dot(Kxt) + 1e-12 * np.eye(len(x_test))
    R = sqrtm(K).real
    mu = KxtT_Kxxinv.dot(y)
    #print mu
    return mu + R.dot(eps.T).T


def approx_gp_sample(x, y, x_test, gp_params, indep_noise, eps,
                     len_u, lanczos_basis=10, return_mean=False):
    x_min, x_max = x.min(), x.max()
    extra_u = 2   # for better interpolation
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)

    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)
    post_gp = PosteriorGP(u, x_test, x, y, kernel, gp_params, indep_noise)
    cov_zs = post_gp.cov_rand_proj(n_lanczos_basis=lanczos_basis, eps=eps)
    mu = post_gp.mean()
    sample = mu + cov_zs
    if return_mean:
        return mu, sample
    return sample


def norm_diff():
    n_data = 1000
    x = np.random.uniform(size=n_data)
    #x = np.float32(x)
    x = np.sort(x)
    a = 1
    b = 1000.
    c = .01
    mu = np.zeros(n_data)
    cov = a * np.exp(-b * (x[:, np.newaxis] - x)**2) + c * np.eye(n_data)
    y = np.random.multivariate_normal(mu, cov)

    gp_params = np.array([a, b])
    indep_noise = c

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
    len_u = 4096
    #for t in xrange(3):

    ####################
    extra_u = 2
    margin = (x_max - x_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(x_min - margin, x_max + margin, len_u)
    x_test = u[1:]

    idx_train, w_train = sparse_w(u, x)
    idx_test, w_test = sparse_w(u, x_test)
    post_gp = PosteriorGP(u, x_test, x, y, kernel, gp_params, indep_noise)

    batch_size = 10
    eps = np.random.normal(size=(batch_size, len(x_test)))

    '''
    exact_samples = exact_gp_sample(x, y, x_test,
                                    gp_params, indep_noise, eps)

    for n_basis in xrange(2, 20 + 1):

        print n_basis
        cov_zs = post_gp.cov_rand_proj(n_sample=batch_size,
                                       n_lanczos_basis=n_basis, eps=eps)
        mu = post_gp.mean()
        gp_samples = mu + cov_zs
        print mu

        #sor_samples = sor_gp_sample(x, y, x_test, u,
        #                            gp_params, indep_noise, eps)
        #print np.linalg.norm(gp_samples - sor_samples)
        #print np.linalg.norm(sor_samples - exact_samples)

        print np.linalg.norm(gp_samples - exact_samples)
        print

    print '-' * 30
    '''

    x_test = np.linspace(x_min, x_max, 512)
    #x_test = np.linspace(x_min, x_max, 1024)
    eps = np.random.normal(size=(batch_size, len(x_test)))
    exact_mu, exact_samples = exact_gp_sample(x, y, x_test,
                                    gp_params, indep_noise, eps,
                                    return_mean=True)

    import pylab as pl
    pl.figure()
    pl.ion()
    pl.plot(x, y, '.-')
    pl.plot(x_test, exact_mu, '-')
    pl.pause(0.001)

    for log_len_u in xrange(3, 12):
        len_u = 2**log_len_u
        print len_u

        mu, gp_samples = approx_gp_sample(x, y, x_test,
                                      gp_params, indep_noise, eps, len_u,
                                      lanczos_basis=10, return_mean=True)
        pl.plot(x_test, mu, '-')
        pl.pause(0.001)

        print np.linalg.norm(gp_samples - exact_samples)
        print
    pl.ioff()
    pl.show()



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


def read_data(data='ECG200/11_1000_60_dat.pkl'):
    #data = 'ECG200/11_1000_60_dat.pkl'
    gp_parms, ts_train, ts_test, l_train, l_test = pickle_load(data)[:5]
    x_train = np.array([each_ts[0] for each_ts in ts_train])
    y_train = np.array([each_ts[1] for each_ts in ts_train])
    eg_id = 0
    x = x_train[eg_id]
    y = y_train[eg_id]
    return x, y


def cmp_lanczos_basis(n_data=500):
    #n_data = 500
    print '# data:', n_data
    gp_params = np.array([.1, 100])
    indep_noise = .001
    x, y = gen_data(n_data, gp_params, indep_noise)

    #import pylab as pl
    #pl.plot(x, y, '.-')
    #pl.show()
    #return

    n_test = n_data

    x_test = np.linspace(x.min(), x.max(), n_test)
    batch_size = 1
    eps = np.random.normal(size=(batch_size, len(x_test)))

    exact_samples = exact_gp_sample(x, y, x_test,
                                    gp_params, indep_noise, eps)

    len_u = 256
    errors = []
    #for n_basis in xrange(1, 51):
    for n_basis in xrange(1, 21):
        print '# basis:', n_basis
        approx_samples = approx_gp_sample(x, y, x_test, gp_params,
                                          indep_noise, eps, len_u,
                                          lanczos_basis=n_basis)
        error = (np.linalg.norm(approx_samples - exact_samples) /
                 np.linalg.norm(exact_samples))
        print error
        errors.append(error)
        #results.append(error)
    return np.array(errors)

    #results = []
    import pylab as pl
    pl.figure()
    pl.ion()
    #for log_len_u in xrange(3, 13):
    for log_len_u in [8]:
        len_u = 2**log_len_u
        print log_len_u, len_u

        errors = []
        #for n_basis in xrange(1, 51):
        for n_basis in xrange(1, 21):
            print '# basis:', n_basis
            approx_samples = approx_gp_sample(x, y, x_test, gp_params,
                                              indep_noise, eps, len_u,
                                              lanczos_basis=n_basis)
            error = np.linalg.norm(approx_samples - exact_samples)
            print error
            errors.append(error)
            #results.append(error)
        pl.semilogy(errors, '.-')
        pl.pause(.001)
        print
    pl.ioff()
    pl.show()

    #return np.array(results)


def cmp_n_inducing_points(n_data=500):
    #n_data = 500
    print '# data:', n_data
    gp_params = np.array([.1, 100])
    indep_noise = .001
    x, y = gen_data(n_data, gp_params, indep_noise)

    #import pylab as pl
    #pl.plot(x, y, '.-')
    #pl.show()
    #return

    n_test = n_data

    x_test = np.linspace(x.min(), x.max(), n_test)
    batch_size = 1
    eps = np.random.normal(size=(batch_size, len(x_test)))

    exact_samples = exact_gp_sample(x, y, x_test,
                                    gp_params, indep_noise, eps)

    results = []
    for log_len_u in xrange(3, 13):
        len_u = 2**log_len_u
        print log_len_u, len_u

        approx_samples = approx_gp_sample(x, y, x_test,
                                          gp_params, indep_noise, eps, len_u)

        error = np.linalg.norm(approx_samples - exact_samples)
        print error
        results.append(error)
        print

    return np.array(results)


def cmp_time():
    n_data = 500
    gp_params = np.array([1., 10.])
    indep_noise = .1

    x, y = gen_data(n_data, gp_params, indep_noise)
    #x, y = read_data('ECG200/11_1000_60_dat.pkl')

    #print x
    #print y
    for log_u in xrange(4, 13):
        len_u = 2**log_u + 1
        print log_u, 2**log_u

        x_test = np.linspace(x.min(), x.max(), len_u)
        batch_size = 1
        eps = np.random.normal(size=(batch_size, len(x_test)))

        t1 = time.time()
        approx_gp_sample(x, y, x_test, gp_params, indep_noise, eps, len_u)
        t2 = time.time()
        print t2 - t1

        t1 = time.time()
        exact_gp_sample(x, y, x_test, gp_params, indep_noise,
                        eps, decomp='cholesky')
        t2 = time.time()
        print t2 - t1

        t1 = time.time()
        exact_gp_sample(x, y, x_test, gp_params, indep_noise,
                        eps, decomp='sqrtm')
        t2 = time.time()
        print t2 - t1

        print
        #print '-' * 40
        #print (cov_zs**2).sum(axis=0)
        #print (cov_zs_exact**2).sum(axis=0)


if __name__ == '__main__':
    np.random.seed(0)
    #cmp_time()
    #errors = cmp_lanczos_basis(500)

    for i in xrange(10):
        results = []
        for n_data in xrange(500, 3000 + 1, 500):
            errors = cmp_lanczos_basis(n_data)
            results.append(errors)
        pickle_save('results/lanczos-relerr-%02d.pkl' % i, np.vstack(results))

    '''
    for i in xrange(10):
        results = []
        for n_data in xrange(200, 3000 + 1, 200):
            errors = cmp_n_inducing_points(n_data)
            results.append(errors)
        pickle_save('results/error-%02d.pkl' % i, np.vstack(results))
    '''

    #norm_diff()
