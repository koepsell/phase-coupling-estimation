"""
This module contains useful functions for circular statistics
"""

__all__ = ['circular_mean', ' circular_correlation', 'circular_variance',
           'mises', 'mises_params', 'phasecorr', 'p2torus', 'torus2p',
           'p2dtorus', 'm_vec2mat', 'm2kappa', 'kappa2m']
           
import os,sys
import numpy as np

try:
    from rpy import r as R
    R.library("CircStats")
except:
    print "could not load R-project"

def circular_mean(phases):
    return np.mean(np.exp(1j*phases))

def circular_correlation(phases1,phases2):
    return circular_mean(phases2-phases1)

def circular_variance(phases):
    return 1-abs(circular_mean(phases))**2


def phasedist(phases):
    "Returns parameters of Von Mises distribution fitted to phase data"
    n = len(phases)
    (kappa,mu,p) = mises_params(circular_mean(phases),n)
    return (kappa,mu,p,n)

def mises(phi,kappa,mu):
    "Returns the Von Mises distribution with mean mu and concentration kappa"
    from scipy.special import i0
    return (1./(2.*np.pi*i0(kappa)))*np.exp(kappa*np.cos(phi-mu))

def mises_params(direction,n=1):
    from scipy.optimize import fmin
    from scipy.special import i0,i1
    "Computes parameters of Von Mises distribution from direction vector"
    def bess(x,r):
        return (i1(x)/i0(x)-r)**2
    try: # using R (faster by a factor of 10)
        kappa = R.A1inv(np.abs(direction))
    except:
        kappa = float(fmin(bess,np.array([1.]),(np.abs(direction),),disp=0));
    mu = np.angle(direction)
    z = float(n)*np.abs(direction)**2
    p = np.exp(-z)*(1.+(2.*z-z**2)/(4.*n)-
                 (24.*z-132.*z**2+76.*z**3-9.*z**4)/(288.*n**2))
    return kappa,mu,p

def phasecorr(phi,get_kappa=False,get_bias=False):
    d = phi.shape[0]
    cpos = np.zeros((d,d),'D')
    cneg = np.zeros((d,d),'D')
    for i in range(d-1):
        if get_bias:
            print "[%d]"%i,
            if get_kappa:
                (kappa,mu,p,n) = phasedist(phi[i,:])
                cneg[i,i] = kappa*np.exp(-1j*mu)
                (kappa,mu,p,n) = phasedist(2*phi[i,:])
                cpos[i,i] = kappa*np.exp(-1j*mu)
            else:
                cneg[i,i] = circular_mean(phi[i,:])
                cpos[i,i] = circular_mean(2*phi[i,:])
        for j in range(i+1,d):
            print "[%d,%d]"%(i,j),
            if get_kappa:
                (kappa,mu,p,n) = phasedist(phi[i,:]-phi[j,:])
                cneg[i,j] = kappa*np.exp(-1j*mu)
                cneg[j,i] = np.conj(cneg[i,j])
                (kappa,mu,p,n) = phasedist(phi[i,:]+phi[j,:])
                cpos[i,j] = cpos[j,i] = kappa*np.exp(-1j*mu)
            else:
                cneg[i,j] = circular_mean(-phi[i,:]+phi[j,:])
                cneg[j,i] = np.conj(cneg[i,j])
                cpos[i,j] = cpos[j,i] = circular_mean(phi[i,:]+phi[j,:])
    return cneg,cpos


def p2torus(phi):
    (dims, samps) = phi.shape
    ptorus = np.zeros((2*dims,samps))
    ptorus[::2,:] = np.cos(phi)
    ptorus[1::2,:] = np.sin(phi)
    return ptorus

def torus2p(x):
    return np.arctan2(x[1::2,:],x[::2,:])

def p2dtorus(phi):
    (dims, samps) = phi.shape
    dtorus = np.zeros((2*dims,samps))
    dtorus[::2,:] = -np.sin(phi)
    dtorus[1::2,:] = np.cos(phi)
    return dtorus

def m_vec2mat(m_vec):
    sz = np.sqrt(len(m_vec))
    m = m_vec.copy()
    m.shape = (sz,sz)
    return m

def m_mat2vec(m):
    return m.flatten()

def m2kappa(m):
    """
    convert real 2N x 2N coupling matrix into
    complex N x N coupling matrices kappa+ and kappa-
    """
    c = -m[::2,::2]
    s = -m[1::2,1::2]
    d = -m[1::2,::2]
    p = .5*(c-s) + .5j*(d.T+d)
    n = .5*(c+s) + .5j*(d.T-d)
    # p = .5*(d.T+d) + .5j*(c-s)
    # n = .5*(d.T-d) + .5j*(c+s)
    return n,p

def kappa2m(n,p=None):
    """
    convert complex N x N coupling matrices kappa+ and kappa-
    into real 2N x 2N coupling matrix
    """
    sz = n.shape[0]
    if p is None:
        c = n.real
        s = n.real
        d = -n.imag
        dt = n.imag
    else:
        c = n.real+p.real
        s = n.real-p.real
        d  = -n.imag+p.imag
        dt = n.imag+p.imag
        # c = n.imag+p.imag
        # s = n.imag-p.imag
        # d = n.real.T+p.real.T
    m = np.zeros((2*sz,2*sz),float)
    m[::2,::2] = -c
    m[1::2,1::2] = -s
    m[1::2,::2] = -d
    m[::2,1::2] = -dt
    return m
