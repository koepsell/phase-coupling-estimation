"""
This module contains functions to model univariate and multivariate
phase distributions and to fit them to data.

:Authors: Charles Cadieu <cadieu@berkeley.edu> and
          Kilian Koepsell <kilian@berkeley.edu>

:Reference: Cadieu CF, Koepsell K (2010) Phase coupling estimation from
            multivariate phase statistics. Neural Computation (in press).

:Copyright: 2008-2010, UC Berkeley
:License: BSD Style
"""

__all__ = ['fit_model', 'fit_gen_model']
from . import __use_weave__
from . import __use_cython__

import os, sys
import numpy as np
import scipy as sp

from utils import tic,toc,smod
from circstats import m_vec2mat

from scipy import sparse
from scipy.sparse.linalg import isolve,dsolve
from model_weave import fill_model_matrix, fill_gen_model_matrix

if __use_cython__:
    os.environ['C_INCLUDE_PATH']=np.get_include()
    import pyximport; pyximport.install()
    from model_cython import fill_model_matrix, fill_gen_model_matrix
elif __use_weave__:
    from model_weave import fill_model_matrix, fill_gen_model_matrix
else:
    raise ImportError("Either cython or weave required")

#
# pretty printing of matices/arrays
# printing precision (number of digits)
#
np.set_printoptions(linewidth=195,precision=4)


def fit_model(phi, eps=0.):
    assert phi.ndim == 2, 'data has to be two-dimensional'
    assert phi.shape[1] > phi.shape[0], 'data samples have to be in columns'    
    d, nsamples = phi.shape
    nij = d**2-d # number of coupling terms

    adata, arow, acol, b = fill_model_matrix(phi)
    a = sparse.coo_matrix((adata,(arow,acol)), (nij,nij))

    tic('matrix inversion')
    if eps > 0:
        a2 = np.dot(a.T,a) + eps*nsamples*sparse.eye(nij,nij,format='coo')
        b2 = np.dot(a.todense().T,np.atleast_2d(b).T)
        # this sparse multiplication is buggy !!!!, I can't get the shape of b2 to be = (b.size,)
        b3 = b2.copy().flatten().T
        b3.shape = (b3.size,)
        k_vec = dsolve.spsolve(a2.tocsr(),b3)
        k_mat = np.zeros((d,d),complex)
        k_mat.T[np.where(np.diag(np.ones(d))-1)] = k_vec.ravel()
    else:
        k_vec = dsolve.spsolve(a.tocsr(),b)
        k_mat = np.zeros((d,d),complex)
        k_mat.T[np.where(np.diag(np.ones(d))-1)] = k_vec
    toc('matrix inversion')
    return k_mat

def fit_model_biased(phi):
    return fit_model(np.vstack((np.zeros(phi.shape[1]),phi)))


def fit_gen_model(phi):
    assert phi.ndim == 2, 'data has to be two-dimensional'
    assert phi.shape[1] > phi.shape[0], 'data samples have to be in columns'
    d, nsamples = phi.shape
    nij = 4*d**2 # number of coupling terms

    adata, arow, acol, b = fill_gen_model_matrix(phi)
    a = sparse.coo_matrix((adata,(arow,acol)), (nij,nij))

    tic('matrix inversion')
    m_vec,flag = isolve.cg(a.tocsr(),b)
    # print 'exit flag = ', flag
    assert flag==0
    m = m_vec2mat(m_vec)
    toc('matrix inversion')
    return m

def fit_gen_model_biased(phi):
    return fit_gen_model(np.vstack((np.zeros(phi.shape[1]),phi)))



def sample_hmc(m,nsamples,burnin=1000,steps=10,step_sz=.2,diagnostics=False,persistence=0):
    from hmc2 import opt, hmc2
    from f_energy import f_phasedist, g_phasedist
    opts = opt(nsamples=nsamples,nomit=burnin,steps=steps,stepadj=step_sz,
               persistence=persistence,display=True)

    samps, E, diagn = hmc2(f=f_phasedist, x=np.zeros(m.shape[0]/2,float),
                           options=opts, gradf=g_phasedist, args=(m,),
                           return_energies=True, return_diagnostics=True)
    print opts
    print diagn
    if diagnostics:
        return smod(samps.T),E,diagn
    else:
        return smod(samps.T)


def density_from_coupling(phi,K):
    """Returns the (un-normalized) probablility density
       based on multivariate phase data and given coupling.
    """
    assert phi.ndim == 2, 'data has to be two-dimensional'
    assert phi.shape[1] > phi.shape[0], 'data samples have to be in columns'    
    x = np.exp(1j*phi)
    E = np.diag(np.dot(np.dot(x.T.conj(),K),x))
    return np.exp(-E)


def density_from_coupling_biased(phi,K):
    """Returns the (un-normalized) probablility density
       based on multivariate phase data and given coupling.
    """
    assert phi.ndim == 2, 'data has to be two-dimensional'
    assert phi.shape[1] > phi.shape[0], 'data samples have to be in columns'    
    d,nsamples = phi.shape
    phib = np.zeros((d+1,nsamples))
    phib[0,:] = 0
    phib[1:,:] = phi
    return density_from_coupling(phib,K)


def sample_hmc_mlabwrap(m,nsamples,burnin=1000,steps=10,step_sz=.2,diagnostics=False,persistence=0):
    from scikits.mlabwrap import mlab
    mlab.addpath('matlab')
    mlab.addpath('matlab/f_energy')

    opts = mlab.hmc2_opt()
    opts.nsamples = nsamples
    opts.nomit = burnin
    opts.steps = steps
    opts.stepadj = step_sz
    opts.persistence = persistence

    samps, E, diagn = mlab.hmc2('f_phasedist',np.zeros(m.shape[0]/2,float),opts,'g_phasedist',m,nout=3)
    print opts
    print diagn
    if diagnostics:
        return smod(samps.T),E,diagn
    else:
        return smod(samps.T)

def timing_benchmark(eval_dim=None,dims=[2, 4, 6, 8, 10],nsamps=10**4):
    ind = 0
    t = np.zeros(len(dims),float)
    for d in dims:
        print 'Benchmarking dim: ', d
        phi = 2*np.pi*np.random.rand(d,nsamps)

        tic('fit parameters')
        c_inv = fit_model(phi)
        t[ind] = toc('fit parameters')
        ind += 1
    pol = np.polyfit(dims[1:],t[1:],3)
    if eval_dim:
        print np.polyval(pol,eval_dim)
    else:
        return pol

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plotlib import plot_phasedist
    dim = 3

    M = np.random.randn(2*dim,2*dim)
    M += M.T.copy()
    for i in np.arange(M.shape[0]/2):
        s = M[2*i,2*i] + M[2*i+1,2*i+1]
        M[2*i,2*i]     -= s/2
        M[2*i+1,2*i+1] -= s/2

    tic('sampling')
    nsamples = 10**4
    burnin = 10**3
    lf_steps = 50
    step_sz = .15
    phi,E,diagn = sample_hmc(M,nsamples,burnin,lf_steps,step_sz,diagnostics=True)
    toc('sampling')

    tic('fiting')
    M_hat = fit_model(phi)
    Mneg_hat,Mpos_hat= m2kappa(M_hat)
    # anti-symmetrize diagonal elements for estimation matrix
    for i in np.arange(M_hat.shape[0]/2):
        s = M_hat[2*i,2*i] + M_hat[2*i+1,2*i+1]
        M_hat[2*i,2*i]     -= s/2
        M_hat[2*i+1,2*i+1] -= s/2
    toc('fiting')

    M_error = M - M_hat
    M_max = max(abs(M).max(),abs(M_hat).max())
    print 'M_error norm = ', (M_error**2).sum()

    tic('sampling')
    phi_hat = sample_hmc(M_hat,nsamples,burnin,lf_steps,step_sz)
    toc('sampling')

    plt.close('all')
    plt.figure()
    plt.subplot(131)
    plt.imshow(M,interpolation='nearest',vmin=-M_max,vmax=M_max)
    plt.axis('off')
    plt.title('true M')
    plt.subplot(132)
    plt.imshow(M_hat,interpolation='nearest',vmin=-M_max,vmax=M_max)
    plt.axis('off')
    plt.title('est. M')
    plt.subplot(133)
    plt.imshow(M_error,interpolation='nearest',vmin=-M_max,vmax=M_max)
    plt.axis('off')
    plt.title('error')
    plt.colorbar(shrink=.3)

    print
    plt.ioff()
    plot_phasedist(phi)
    plot_phasedist(phi_hat)
    plt.ion()
    plt.show()




    # from scipy import io
    # mdict = io.loadmat('testdata/three_phases_gen')
    # vars().update(mdict); M=M_true; M_hat=M_python; phi=data;
    #
    # save data for matlab
    #
    # mdict = dict(data=phi,M_true=M,M_python=M_hat)
    # io.savemat('testdata/three_phases_gen',mdict)



    # from scipy import io
    # mdict = np.load('testdata/three_phases_gen.npz')
    # for var in mdict.files:
    #     globals()[var] = mdict[var]
    #
    # save data for python
    #
    # mdict = dict(data=phi,M_true=M,M_python=M_hat)
    # np.savez('testdata/three_phases_gen',**mdict)

