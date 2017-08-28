import numpy as np
cimport numpy as np
#cimport cython


#@cython.boundscheck(False)
def interp_to_u(np.ndarray[np.int32_t, ndim=2] idx,
                np.ndarray[np.float_t, ndim=2] w,
                int len_u,
                np.ndarray[np.float_t, ndim=1] y):
    cdef int n = idx.shape[0]
    cdef int m = idx.shape[1]
    cdef np.ndarray s = np.zeros(len_u, dtype=np.float)
    cdef unsigned int i, j
    cdef np.float_t yi

    for i in xrange(n):
        yi = y[i]
        for j in xrange(m):
            s[<unsigned int>idx[i, j]] += yi * w[i, j]
    return s


def interp_to_u_block(np.ndarray[np.int32_t, ndim=2] idx,
                      np.ndarray[np.float_t, ndim=2] w,
                      int len_u,
                      np.ndarray[np.float_t, ndim=2] y):
    cdef int n = idx.shape[0]
    cdef int m = idx.shape[1]
    cdef int len_y = y.shape[0]
    cdef np.ndarray s = np.zeros((len_y, len_u), dtype=np.float)
    cdef unsigned int i, j, k
    cdef np.float_t yi

    for k in xrange(len_y):
        for i in xrange(n):
            yi = y[k, i]
            for j in xrange(m):
                s[k, <unsigned int>idx[i, j]] += yi * w[i, j]
    return s

