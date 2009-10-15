"""
This script fits the multivariate phase distribution to data
and (hopefully) at some point also the Ising model to binary data.
"""

# load numpy, scipy, etc.
import numpy as np
import scipy as sp
from scipy import io,sparse,weave
from scipy.sparse.linalg import isolve,dsolve
import matplotlib.pyplot as plt
import sys

#
# load (and reload) modules
#
modules = ['utils','model','plotlib']
for name in modules:
    mod = __import__(name,globals(),locals(),[])
    # reload modules (useful during development)
    reload(mod)

from model import phasedist,circular_mean,circular_correlation
from plotlib import plot_phasedist3d
from utils import tic,toc,smod

#
# pretty printing of matices/arrays
# printing precision (number of digits)
#
np.set_printoptions(linewidth=195,precision=4)


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

def fit_model(phi):
    z = np.concatenate((np.exp(1j*phi),np.exp(-1j*phi)))
    d = phi.shape[0]
    nz = phi.shape[1]
    z.shape = (2,d,nz)
    nij = d**2-d # number of coupling terms
    na = 4*d**3-10*d**2+6*d # upper bound for number of elements in sparse matrix
    adata = np.zeros(na,complex)
    arow = np.zeros(na,int)
    acol = np.zeros(na,int)
    b = np.zeros(nij,complex)

    tic('weave')
    weave.inline(phasemodel_code_blitz, ['z','adata','arow','acol','b'],
                 type_converters=weave.converters.blitz)
    a = sparse.coo_matrix((adata,(arow,acol)), (nij,nij))
    toc('weave')

    tic('matrix inversion')
    k_vec = dsolve.spsolve(a.tocsr(),b)
    k_mat = np.zeros((d,d),complex)
    k_mat.T[np.where(np.diag(np.ones(d))-1)] = k_vec
    toc('matrix inversion')
    return k_mat

def fit_model_biased(phi):
    return fit_model(np.vstack((np.zeros(phi.shape[1]),phi)))


phasemodel_code_blitz = """
int d = Nz[1];
int nz = Nz[2];
int na = Nadata[0];
int nij = Nb[0];
int ij = -1;
int kl = -1;
int ia = -1;
std::cout << "starting complex weave code (difference coupling only)" << std::endl;
ij = -1;
ia = -1;
for (int i=0; i < d; i++)
{
    for (int j=0; j < d; j++)
    {
        if (i==j) continue;
        ij++;
        kl = -1;
        // b = -2*conj(C)
        // b_ij = -2*w(i)*z(j)
        for (int n=0; n < nz; n++) b(ij) -= 2.*z(1,i,n)*z(0,j,n);
        for (int k=0; k < d; k++)
        {
            for (int l=0; l < d; l++)
            {
                if (k==l) continue;
                kl++;
                if (i!=k and i!=l and j!=k and j!=l) continue;
                // a = .5*m.T*C.T + .5*C.T*m.T - .5*conj(Q)*m*P - .5*conj(P)*m*Q
                // a_ij_kl = .5*d_i_l*C_j_k + .5*d_k_j*C_l_i - .5*conj(Q)_i_k*P_l_j - .5*conj(P)_i_k*Q_l_j
                //         = .5*d_i_l*z(j)*w(k) + .5*d_k_j*z(l)*w(i) - .5*d_i_k*w(i)*w(k)*z(l)*z(j) - .5*d_l_j*w(i)*w(k)*z(l)*z(j)
                ia++;
                arow(ia) = ij;
                acol(ia) = kl;
                if (i==k) for (int n=0; n < nz; n++) adata(ia) -= .5*z(1,i,n)*z(1,k,n)*z(0,l,n)*z(0,j,n);
                if (i==l) for (int n=0; n < nz; n++) adata(ia) += .5*z(0,j,n)*z(1,k,n);
                if (j==k) for (int n=0; n < nz; n++) adata(ia) += .5*z(1,i,n)*z(0,l,n);
                if (j==l) for (int n=0; n < nz; n++) adata(ia) -= .5*z(1,i,n)*z(1,k,n)*z(0,l,n)*z(0,j,n);
            }
        }
    }
}
// std::cout << "number of sparse matrix elements:" << ia+1 <<std::endl;
"""

def fit_gen_model(phi):
    d,nsamples = phi.shape
    x = p2torus(phi)
    q = p2dtorus(phi)
    x.shape = (d,2,nsamples)
    q.shape = (d,2,nsamples)

    nij = 4*d**2 # number of coupling terms
    na = 32*d**3 - 16*d**2 # number of elements in large matrix multiplying mij
    adata = np.zeros(na,float)
    arow = np.zeros(na,int)
    acol = np.zeros(na,int)
    b = np.zeros(nij,float)

    tic('weave')
    weave.inline(gen_phasemodel_code, ['x','q','adata','arow','acol','b'])
    a = sparse.coo_matrix((adata,(arow,acol)), (nij,nij))
    toc('weave')

    tic('matrix inversion')
    m_vec,flag = isolve.cg(a.tocsr(),b)
    # print 'exit flag = ', flag
    assert flag==0
    m = m_vec2mat(m_vec)
    toc('matrix inversion')
    return m

def fit_gen_model_biased(phi):
    return fit_gen_model(np.vstack((np.zeros(phi.shape[1]),phi)))


gen_phasemodel_code = """
int d = Nx[0];
int ny = Nx[1];
int nx = Nx[2];
int na = Nadata[0];
int i,j,k,l;
int ij = -1;
int kl = -1;
int ia = -1;
ij = -1;
ia = -1;
double temp;
for (int i0=0; i0 < d; i0++) {
  for (int i1=0; i1 < 2; i1++) {
    for (int j0=0; j0 < d; j0++) {
      for (int j1=0; j1 < 2; j1++) {
        i = 2*i0+i1;
        j = 2*j0+j1;
        ij++;
        // b = (Q-C)
        if (i0==j0) {
          for (int n=0; n < nx; n++) {
            temp  = 2.;
            temp *= q[(i0*ny+i1)*nx+n];
            temp *= q[(j0*ny+j1)*nx+n];
            b[ij] -= temp;
          }
        }
        for (int n=0; n < nx; n++) {
          temp = 2.;
          temp *= x[(i0*ny+i1)*nx+n];
          temp *= x[(j0*ny+j1)*nx+n];
          b[ij] += temp;
        }
        kl = -1;
        for (int k0=0; k0 < d; k0++) {
          for (int k1=0; k1 < 2; k1++) {
            for (int l0=0; l0 < d; l0++) {
              for (int l1=0; l1 < 2; l1++) {
                k = 2*k0+k1;
                l = 2*l0+l1;
                kl++;
                if (i0!=k0 and j0!=l0) continue;
                ia++;
                //a = (np.tensordot(C[:,k,:],Q[l,:,:],axes=(1,1)) +
                //     np.tensordot(Q[:,k,:],C[l,:,:],axes=(1,1)))
                arow[ia] = ij;
                acol[ia] = kl;
                if (j0==l0) {
                  for (int n=0; n < nx; n++) {
                    temp =  x[(i0*ny+i1)*nx+n];
                    temp *= x[(k0*ny+k1)*nx+n];
                    temp *= q[(l0*ny+l1)*nx+n];
                    temp *= q[(j0*ny+j1)*nx+n];
                    adata[ia] += temp;
                  }
                }
                if (i0==k0) {
                  for (int n=0; n < nx; n++) {
                    temp =  q[(i0*ny+i1)*nx+n];
                    temp *= q[(k0*ny+k1)*nx+n];
                    temp *= x[(l0*ny+l1)*nx+n];
                    temp *= x[(j0*ny+j1)*nx+n];
                    adata[ia] += temp;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
// std::cout << "matrix elements needed: " << ia+1 << std::endl;
"""

def sample_hmc(m,nsamples,burnin=1000,steps=10,step_sz=.2,diagnostics=False,persistence=0):
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
    plot_phasedist3d(phi)
    plot_phasedist3d(phi_hat)
    plt.ion()
    plt.show()




    # mdict = io.loadmat('testdata/three_phases_gen')
    # vars().update(mdict); M=M_true; M_hat=M_python; phi=data;
    #
    # save data for matlab
    #
    # mdict = dict(data=phi,M_true=M,M_python=M_hat)
    # io.savemat('testdata/three_phases_gen',mdict)



    # mdict = np.load('testdata/three_phases_gen.npz')
    # for var in mdict.files:
    #     globals()[var] = mdict[var]
    #
    # save data for python
    #
    # mdict = dict(data=phi,M_true=M,M_python=M_hat)
    # np.savez('testdata/three_phases_gen',**mdict)

