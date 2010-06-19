# load numpy, scipy, etc.
import numpy as np
import scipy as sp
from scipy import weave

from utils import tic,toc,smod
from circstats import p2torus, p2dtorus

def fill_model_matrix(phi):
    z = np.concatenate((np.exp(1j*phi),np.exp(-1j*phi)))
    d, nsamples = phi.shape
    z.shape = (2,d,nsamples)
    nij = d**2-d # number of coupling terms
    na = 4*d**3-10*d**2+6*d # upper bound for number of elements in sparse matrix
    adata = np.zeros(na,complex)
    arow = np.zeros(na,int)
    acol = np.zeros(na,int)
    b = np.zeros(nij,complex)

    tic('weave')
    weave.inline(phasemodel_code_blitz, ['z','adata','arow','acol','b'],
                 type_converters=weave.converters.blitz)
    toc('weave')
    return adata, arow, acol, b

phasemodel_code_blitz = """
int d = Nz[1];
int nz = Nz[2];
int na = Nadata[0];
int nij = Nb[0];
int ij = -1;
int kl = -1;
int ia = -1;
std::cout << "starting complex weave code (difference coupling only)" << std::endl;
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

def fill_gen_model_matrix(phi):
    d, nsamples = phi.shape
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
    toc('weave')
    return adata, arow, acol, b

gen_phasemodel_code = """
int d = Nx[0];
int ny = Nx[1];
int nx = Nx[2];
int na = Nadata[0];
int i,j,k,l;
int ij = -1;
int kl = -1;
int ia = -1;
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


if __name__ == '__main__':
    import nose
    nose.run()
