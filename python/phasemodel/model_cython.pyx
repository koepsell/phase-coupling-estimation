import numpy as np
cimport numpy as np
cimport cython

cdef extern from "complexobject.h":

    struct Py_complex:
        double real
        double imag

    ctypedef class __builtin__.complex [object PyComplexObject]:
        cdef Py_complex cval


from scipy import sparse
from scipy.sparse.linalg import isolve,dsolve

from utils import tic, toc
from circstats import p2torus, p2dtorus


@cython.boundscheck(False)
def fill_model_matrix(np.ndarray[np.double_t, ndim=2] phi):
    cdef unsigned int d = phi.shape[0]
    cdef unsigned int nz = phi.shape[1]

    cdef unsigned int i,j,k,l,n
    cdef int ij = -1
    cdef int kl = -1
    cdef int ia = -1

    cdef unsigned int nij = d**2-d # number of coupling terms
    cdef unsigned int na = 4*d**3-10*d**2+6*d # upper bound for number of elements in sparse matrix

    cdef np.ndarray[np.complex128_t, ndim=3] z = np.concatenate(
        (np.exp(1j*phi),np.exp(-1j*phi))).reshape(2,d,nz)
    cdef np.ndarray[np.complex128_t, ndim=1] adata = np.zeros(na,np.complex128)
    cdef np.ndarray[np.int_t, ndim=1] arow = np.zeros(na,int)
    cdef np.ndarray[np.int_t, ndim=1] acol = np.zeros(na,int)
    cdef np.ndarray[np.complex128_t, ndim=1] b = np.zeros(nij,np.complex128)

    tic('cython')
    for i in range(d):
        for j in range(d):
            if i == j: continue
            ij += 1
            kl = -1
            # b = -2*conj(C)
            # b_ij = -2*w(i)*z(j)
            for n in range(nz): b[ij] = b[ij] - 2.*z[1,i,n]*z[0,j,n]

            for k in range(d):
                for l in range(d):
                    if k == l: continue
                    kl += 1
                    if (i!=k and i!=l and j!=k and j!=l): continue

                    # a = .5*m.T*C.T + .5*C.T*m.T - .5*conj(Q)*m*P - .5*conj(P)*m*Q
                    # a_ij_kl = .5*d_i_l*C_j_k + .5*d_k_j*C_l_i - .5*conj(Q)_i_k*P_l_j - .5*conj(P)_i_k*Q_l_j
                    #         = .5*d_i_l*z(j)*w(k) + .5*d_k_j*z(l)*w(i) - .5*d_i_k*w(i)*w(k)*z(l)*z(j) - .5*d_l_j*w(i)*w(k)*z(l)*z(j)
                    ia += 1
                    arow[ia] = ij
                    acol[ia] = kl
                    if i == k:
                        for n in range(nz): adata[ia] = adata[ia] - .5*z[1,i,n]*z[1,k,n]*z[0,l,n]*z[0,j,n]
                    if i == l:
                        for n in range(nz): adata[ia] = adata[ia] + .5*z[0,j,n]*z[1,k,n]

                    if j == k:
                        for n in range(nz): adata[ia] = adata[ia] + .5*z[1,i,n]*z[0,l,n]

                    if j == l:
                        for n in range(nz): adata[ia] = adata[ia] - .5*z[1,i,n]*z[1,k,n]*z[0,l,n]*z[0,j,n]

    toc('cython')
    return adata, arow, acol, b

@cython.boundscheck(False)
def fill_gen_model_matrix(np.ndarray[np.double_t, ndim=2] phi):
    cdef unsigned int d = phi.shape[0]
    cdef unsigned int ny = 2
    cdef unsigned int nx = phi.shape[1]
    cdef unsigned int i,j,k,l,n
    cdef unsigned int i0,i1,j0,j1,k0,k1,l0,l1
    cdef int ij = -1
    cdef int kl = -1
    cdef int ia = -1
    cdef double temp

    cdef unsigned int nij = 4*d**2 # number of coupling terms
    cdef unsigned int na = 32*d**3 - 16*d**2 # number of elements in large matrix multiplying mij

    cdef np.ndarray[np.double_t, ndim=3] x = p2torus(phi).reshape(d,2,nx)
    cdef np.ndarray[np.double_t, ndim=3] q = p2dtorus(phi).reshape(d,2,nx)
    cdef np.ndarray[np.double_t, ndim=1] adata = np.zeros(na,float)
    cdef np.ndarray[np.int_t, ndim=1] arow = np.zeros(na,int)
    cdef np.ndarray[np.int_t, ndim=1] acol = np.zeros(na,int)
    cdef np.ndarray[np.double_t, ndim=1] b = np.zeros(nij,float)

    tic('cython')
    for i0 in range(d):
        for i1 in range(2):
            for j0 in range(d):
                for j1 in range(2):
                    i = 2*i0+i1
                    j = 2*j0+j1
                    ij += 1

                    # b = (Q-C)
                    if i0 == j0:
                        for n in range(nx):
                            temp  = 2.
                            temp *= q[i0,i1,n]
                            temp *= q[j0,j1,n]
                            b[ij] -= temp
                            
                    for n in range(nx):
                        temp = 2.
                        temp *= x[i0,i1,n]
                        temp *= x[j0,j1,n]
                        b[ij] += temp
                    
                    kl = -1
                    for k0 in range(d):
                        for k1 in range(2):
                            for l0 in range(d):
                                for l1 in range(2):
                                    k = 2*k0+k1
                                    l = 2*l0+l1
                                    kl += 1
                                    if (i0 != k0 and j0 != l0): continue
                                    ia += 1
                                    # a = (np.tensordot(C[:,k,:],Q[l,:,:],axes=(1,1)) +
                                    #      np.tensordot(Q[:,k,:],C[l,:,:],axes=(1,1)))
                                    arow[ia] = ij;
                                    acol[ia] = kl;
                                    if j0 == l0:
                                        for n in range(nx):
                                            temp =  x[i0,i1,n]
                                            temp *= x[k0,k1,n]
                                            temp *= q[l0,l1,n]
                                            temp *= q[j0,j1,n]
                                            adata[ia] += temp

                                    if i0 == k0:
                                        for n in range(nx):
                                            temp =  q[i0,i1,n]
                                            temp *= q[k0,k1,n]
                                            temp *= x[l0,l1,n]
                                            temp *= x[j0,j1,n]
                                            adata[ia] += temp

    toc('cython')
    return adata, arow, acol, b
