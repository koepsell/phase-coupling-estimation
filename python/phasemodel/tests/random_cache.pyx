import numpy as np
cimport numpy as np
cimport cython

import random

cdef double *randcache
cdef unsigned int ncache = 0
cdef unsigned int cacheinit = 0

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
    void *realloc(void *ptr, size_t size)
    size_t strlen(char *s)
    char *strcpy(char *dest, char *src)
    
cdef rand():
    global randcache
    global cacheinit
    global ncache
    cdef unsigned int nmax = 10000
    cdef unsigned int i
    cdef np.ndarray[np.double_t] tmp
    if cacheinit == 0:
        cacheinit = 1
        randcache = <double *>malloc(nmax * sizeof(double))
    if ncache >= nmax: ncache = 0
    if ncache == 0:
        tmp = np.random.rand(nmax)
        for i in range(nmax):
            randcache[i] = tmp[i]
    ncache += 1
    return randcache[ncache-1]

cpdef test_random():
    cdef unsigned int i
    cdef unsigned int nmax = 10000
    cdef np.ndarray[np.double_t] rands = np.zeros(nmax)
    ran = random.random
    for i in range(nmax):
        rands[i] = ran()

cpdef test_random_cached():
    cdef unsigned int i
    cdef unsigned int nmax = 10000
    cdef np.ndarray[np.double_t] rands = np.zeros(nmax)
    for i in range(nmax): rands[i] = rand()
