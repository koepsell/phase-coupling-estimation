// file:        fill_matrix.c
// authors:     Charles Cadieu and Kilian Koepsell
// description: Computes linear system of equations for estimation
//              of the multivariate phase distribution.

// Copyright (c) 2008 The Regents of the University of California
// All Rights Reserved.
// 
// Created by Charles Cadieu and Kilian Koepsell (UC Berkeley)
// 
// Permission to use, copy, modify, and distribute this software and its
// documentation for educational, research and non-profit purposes,
// without fee, and without a written agreement is hereby granted,
// provided that the above copyright notice, this paragraph and the
// following three paragraphs appear in all copies.
// 
// This software program and documentation are copyrighted by The Regents
// of the University of California. The software program and
// documentation are supplied "as is", without any accompanying services
// from The Regents. The Regents does not warrant that the operation of
// the program will be uninterrupted or error-free. The end-user
// understands that the program was developed for research purposes and
// is advised not to rely exclusively on the program for any reason.
// 
// This software embodies a method for which a patent is pending.
// 
// IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
// FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
// INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF
// CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
// BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE
// MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

#include <math.h>
#include "mex.h"

#define Z_IN            prhs[0]         // input data
#define NIJ_IN          prhs[1]         // matrix size
#define NA_IN           prhs[2]         // sparse matrix elements

#define A_OUT           plhs[0]         // a matrix values
#define B_OUT           plhs[1]         // b vector values

/* complex multiplication */
#define mr(xr,xi,yr,yi) (double)(xr*yr-xi*yi)
#define mi(xr,xi,yr,yi) (double)(xr*yi+xi*yr)
#define pr(zr,zi,i1,i2) (double)(zr[i1]*zr[i2]-zi[i1]*zi[i2])
#define pi(zr,zi,i1,i2) (double)(zr[i1]*zi[i2]+zi[i1]*zr[i2])

/* complex multiplication, conjugate second argument */
#define pcr(zr,zi,i1,i2) (double)(zr[i1]*zr[i2]+zi[i1]*zi[i2])
#define pci(zr,zi,i1,i2) (double)(zi[i1]*zr[i2]-zr[i1]*zi[i2])

void fill_matrix_single(mwSize d, mwSize nz, mwSize nij, mwSize na, const mxArray *z_in, mxArray *a_out, mxArray *b_out);
void fill_matrix_double(mwSize d, mwSize nz, mwSize nij, mwSize na, const mxArray *z_in, mxArray *a_out, mxArray *b_out);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwSize  d, nz;
  mwSize nij, na;
  mwIndex ij = 0;
  mwIndex kl;
  mwIndex ia = 0;
  mwIndex i,j,k,l,nd;

  /* check for the proper number of arguments */
  if(nrhs != 3)
    mexErrMsgTxt("Three inputs required.");
  if(nlhs != 2)
    mexErrMsgTxt("Two output arguments required.");
    
  /* Get the number of dimensions in the input argument. */
  if (mxGetNumberOfDimensions(Z_IN) != 2)
    mexErrMsgTxt("Z must be 2d array.");
  
  /*Check that z is complex*/
  if (!mxIsComplex(Z_IN) )
    mexErrMsgTxt("Z must be complex.\n");
  
  /*Check that nij and na are scalars*/
  if (!mxIsDouble(NIJ_IN) || !mxIsDouble(NA_IN))
    mexErrMsgTxt("nij and na must be scalars.\n");

  d = mxGetM(Z_IN);
  nz = mxGetN(Z_IN);
  nij = (mwSize)mxGetScalar(NIJ_IN);
  na = (mwSize)mxGetScalar(NA_IN);
  // mexPrintf("d = %d, nz = %d\n",d,nz);
  // mexPrintf("nij = %d, na = %d\n",nij,na);
  // mexPrintf("z-dimensions: %d, %d, %d\n",shape[0],shape[1],shape[2]);
  
  // Allocate memory and assign output pointers
  A_OUT = mxCreateSparse(nij,nij,na,mxCOMPLEX);
  B_OUT = mxCreateDoubleMatrix(nij,1,mxCOMPLEX);

  /*Check if Z is single or double precision*/
  if (mxIsDouble(Z_IN)) fill_matrix_double(d, nz, nij, na, Z_IN, A_OUT, B_OUT);
  else if (mxIsSingle(Z_IN)) fill_matrix_single(d, nz, nij, na, Z_IN, A_OUT, B_OUT);
  else mexErrMsgTxt("Z has to be single or double precision.\n");

  return;
}


void fill_matrix_double(mwSize d, mwSize nz, mwSize nij, mwSize na, const mxArray *z_in, mxArray *a_out, mxArray *b_out)
{
  double *zr, *zi;    
  double xr, xi; 
  double yr, yi;
  mwIndex *arow, *acol;
  double *ar, *ai;    
  double *br, *bi;    
  mwIndex ij = 0;
  mwIndex kl;
  mwIndex ia = 0;
  mwIndex i,j,k,l,nd;

  // Get a pointer to the z data
  zr = mxGetPr(z_in);
  zi = mxGetPi(z_in);

  // Get a pointer to the data space in allocated memory
  ar = mxGetPr(a_out);
  ai = mxGetPi(a_out);
  arow = mxGetIr(a_out);
  acol = mxGetJc(a_out);
  br = mxGetPr(b_out);
  bi = mxGetPi(b_out);

  for (i=0; i < d; i++) {
    for (j=0; j < d; j++) {
      if (i==j) continue;
      kl = 0;
      if (ij >= nij) mexErrMsgTxt("vector b too small! This should not happen!");
      for (nd=0; nd < nz*d; nd+=d) {
          // b(ij) -= 2.*z(1,i,n)*z(0,j,n);
          br[ij] -= 2.*pcr(zr,zi,j+nd,i+nd);
          bi[ij] -= 2.*pci(zr,zi,j+nd,i+nd);
      }
      acol[ij] = ia;
      ij++;
      for (k=0; k < d; k++) {
        for (l=0; l < d; l++) {
          if (k==l) continue;
          if ((i!=k) && (i!=l) && (j!=k) && (j!=l)) {
            kl++;
            continue;
          }
          if (ia >= na) mexErrMsgTxt("sparse matrix too small! This should not happen!");
          arow[ia] = kl;
          // if (i==k) for (n=0; n < nz; n++) adata(ia) -= .5*z(1,i,n)*z(1,k,n)*z(0,l,n)*z(0,j,n);
          if (i==k) for (nd=0; nd < nz*d; nd+=d) {
              xr = pr(zr,zi,i+nd,k+nd);
              xi = -pi(zr,zi,i+nd,k+nd);
              yr = pr(zr,zi,l+nd,j+nd);
              yi = pi(zr,zi,l+nd,j+nd);
              ar[ia] -= .5*mr(xr,xi,yr,yi);
              ai[ia] -= .5*mi(xr,xi,yr,yi);
            }

          // if (i==l) for (n=0; n < nz; n++) adata(ia) += .5*z(0,j,n)*z(1,k,n);
          if (i==l) for (nd=0; nd < nz*d; nd+=d) {
              ar[ia] += .5*pcr(zr,zi,j+nd,k+nd);
              ai[ia] += .5*pci(zr,zi,j+nd,k+nd);
            }

          // if (j==k) for (n=0; n < nz; n++) adata(ia) += .5*z(1,i,n)*z(0,l,n);
          if (j==k) for (nd=0; nd < nz*d; nd+=d) {
              ar[ia] += .5*pcr(zr,zi,l+nd,i+nd);
              ai[ia] += .5*pci(zr,zi,l+nd,i+nd);
            }

          // if (j==l) for (n=0; n < nz; n++) adata(ia) -= .5*z(1,i,n)*z(1,k,n)*z(0,l,n)*z(0,j,n);
          if (j==l) for (nd=0; nd < nz*d; nd+=d) {
              xr = pr(zr,zi,i+nd,k+nd);
              xi = -pi(zr,zi,i+nd,k+nd);
              yr = pr(zr,zi,l+nd,j+nd);
              yi = pi(zr,zi,l+nd,j+nd);
              ar[ia] -= .5*mr(xr,xi,yr,yi);
              ai[ia] -= .5*mi(xr,xi,yr,yi);           
            }
          kl++;
          ia++;
        }
      }
    }
  }
  acol[nij] = ia;
  return;
}


void fill_matrix_single(mwSize d, mwSize nz, mwSize nij, mwSize na, const mxArray *z_in, mxArray *a_out, mxArray *b_out)
{
  float *zr, *zi;    
  double xr, xi; 
  double yr, yi;
  mwIndex *arow, *acol;
  double *ar, *ai;    
  double *br, *bi;    
  mwIndex ij = 0;
  mwIndex kl;
  mwIndex ia = 0;
  mwIndex i,j,k,l,nd;

  // Get a pointer to the z data
  zr = (float*)mxGetPr(z_in);
  zi = (float*)mxGetPi(z_in);

  // Get a pointer to the data space in allocated memory
  ar = mxGetPr(a_out);
  ai = mxGetPi(a_out);
  arow = mxGetIr(a_out);
  acol = mxGetJc(a_out);
  br = mxGetPr(b_out);
  bi = mxGetPi(b_out);

  for (i=0; i < d; i++) {
    for (j=0; j < d; j++) {
      if (i==j) continue;
      kl = 0;
      if (ij >= nij) mexErrMsgTxt("vector b too small! This should not happen!");
      for (nd=0; nd < nz*d; nd+=d) {
          // b(ij) -= 2.*z(1,i,n)*z(0,j,n);
          br[ij] -= 2.*pcr(zr,zi,j+nd,i+nd);
          bi[ij] -= 2.*pci(zr,zi,j+nd,i+nd);
      }
      acol[ij] = ia;
      ij++;
      for (k=0; k < d; k++) {
        for (l=0; l < d; l++) {
          if (k==l) continue;
          if ((i!=k) && (i!=l) && (j!=k) && (j!=l)) {
            kl++;
            continue;
          }
          if (ia >= na) mexErrMsgTxt("sparse matrix too small! This should not happen!");
          arow[ia] = kl;
          // if (i==k) for (n=0; n < nz; n++) adata(ia) -= .5*z(1,i,n)*z(1,k,n)*z(0,l,n)*z(0,j,n);
          if (i==k) for (nd=0; nd < nz*d; nd+=d) {
              xr = pr(zr,zi,i+nd,k+nd);
              xi = -pi(zr,zi,i+nd,k+nd);
              yr = pr(zr,zi,l+nd,j+nd);
              yi = pi(zr,zi,l+nd,j+nd);
              ar[ia] -= .5*mr(xr,xi,yr,yi);
              ai[ia] -= .5*mi(xr,xi,yr,yi);
            }

          // if (i==l) for (n=0; n < nz; n++) adata(ia) += .5*z(0,j,n)*z(1,k,n);
          if (i==l) for (nd=0; nd < nz*d; nd+=d) {
              ar[ia] += .5*pcr(zr,zi,j+nd,k+nd);
              ai[ia] += .5*pci(zr,zi,j+nd,k+nd);
            }

          // if (j==k) for (n=0; n < nz; n++) adata(ia) += .5*z(1,i,n)*z(0,l,n);
          if (j==k) for (nd=0; nd < nz*d; nd+=d) {
              ar[ia] += .5*pcr(zr,zi,l+nd,i+nd);
              ai[ia] += .5*pci(zr,zi,l+nd,i+nd);
            }

          // if (j==l) for (n=0; n < nz; n++) adata(ia) -= .5*z(1,i,n)*z(1,k,n)*z(0,l,n)*z(0,j,n);
          if (j==l) for (nd=0; nd < nz*d; nd+=d) {
              xr = pr(zr,zi,i+nd,k+nd);
              xi = -pi(zr,zi,i+nd,k+nd);
              yr = pr(zr,zi,l+nd,j+nd);
              yi = pi(zr,zi,l+nd,j+nd);
              ar[ia] -= .5*mr(xr,xi,yr,yi);
              ai[ia] -= .5*mi(xr,xi,yr,yi);           
            }
          kl++;
          ia++;
        }
      }
    }
  }
  acol[nij] = ia;
  return;
}

