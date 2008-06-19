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

#define Z_IN		prhs[0]		// input data
#define NIJ_IN		prhs[1]		// matrix size
#define NA_IN		prhs[2]		// sparse matrix elements

#define A_OUT		plhs[0]		// a matrix values
#define B_OUT		plhs[1]		// b vector values

/* complex multiplication, real part */
double pr(double* zr, double* zi, int ind1, int ind2)
{
  return zr[ind1]*zr[ind2]-zi[ind1]*zi[ind2];
}

/* complex multiplication, imaginary part */
double pi(double* zr, double* zi, int ind1, int ind2)
{
  return zr[ind1]*zi[ind2]+zi[ind1]*zr[ind2];
}

/* complex multiplication, conjugate second argument, real part */
double pcr(double* zr, double* zi, int ind1, int ind2)
{
  return zr[ind1]*zr[ind2]+zi[ind1]*zi[ind2];
}

/* complex multiplication, conjugate second argument, imaginary part */
double pci(double* zr, double* zi, int ind1, int ind2)
{
  return -zr[ind1]*zi[ind2]+zi[ind1]*zr[ind2];
}

/* complex multiplication, real part */
double mr(double xr, double xi, double yr, double yi)
{
  return xr*yr-xi*yi;
}

/* complex multiplication, imaginary part */
double mi(double xr, double xi, double yr, double yi)
{
  return xr*yi+xi*yr;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *zr, *zi;    
  double xr, xi; 
  double yr, yi;    
  mwSize  d, nz;
  mwSize nij, na;
  const mwSize *shape;
  mwIndex *arow, *acol;
  double *ar, *ai;    
  double *br, *bi;    
  mwIndex ij = 0;
  mwIndex kl;
  mwIndex ia = 0;
  mwIndex i,j,k,l,n;

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

  shape = mxGetDimensions(Z_IN);
  d = shape[0];
  nz = shape[1];
  nij = (mwSize)mxGetScalar(NIJ_IN);
  na = (mwSize)mxGetScalar(NA_IN);
  // mexPrintf("d = %d, nz = %d\n",d,nz);
  // mexPrintf("nij = %d, na = %d\n",nij,na);
  // mexPrintf("z-dimensions: %d, %d, %d\n",shape[0],shape[1],shape[2]);
  

  // Allocate memory and assign output pointers
  A_OUT = mxCreateSparse(nij,nij,na,mxCOMPLEX);
  B_OUT = mxCreateDoubleMatrix(nij,1,mxCOMPLEX);

  // Get a pointer to the z data
  zr = mxGetPr(Z_IN);
  zi = mxGetPi(Z_IN);

  // Get a pointer to the data space in allocated memory
  ar = mxGetPr(A_OUT);
  ai = mxGetPi(A_OUT);
  arow = mxGetIr(A_OUT);
  acol = mxGetJc(A_OUT);
  br = mxGetPr(B_OUT);
  bi = mxGetPi(B_OUT);
    
  for (i=0; i < d; i++) {
    for (j=0; j < d; j++) {
      if (i==j) continue;
      kl = 0;
      if (ij >= nij) mexErrMsgTxt("vector b too small! This should not happen!");
      for (n=0; n < nz; n++) {
	  // b(ij) -= 2.*z(1,i,n)*z(0,j,n);
	  br[ij] -= 2.*pcr(zr,zi,j+n*d,i+n*d);
	  bi[ij] -= 2.*pci(zr,zi,j+n*d,i+n*d);
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
	  if (i==k) for (n=0; n < nz; n++) {
	      xr = pr(zr,zi,i+n*d,k+n*d);
	      xi = -pi(zr,zi,i+n*d,k+n*d);
	      yr = pr(zr,zi,l+n*d,j+n*d);
	      yi = pi(zr,zi,l+n*d,j+n*d);
	      ar[ia] -= .5*mr(xr,xi,yr,yi);
	      ai[ia] -= .5*mi(xr,xi,yr,yi);
	    }

	  // if (i==l) for (n=0; n < nz; n++) adata(ia) += .5*z(0,j,n)*z(1,k,n);
	  if (i==l) for (n=0; n < nz; n++) {
	      ar[ia] += .5*pcr(zr,zi,j+n*d,k+n*d);
	      ai[ia] += .5*pci(zr,zi,j+n*d,k+n*d);
	    }

	  // if (j==k) for (n=0; n < nz; n++) adata(ia) += .5*z(1,i,n)*z(0,l,n);
	  if (j==k) for (n=0; n < nz; n++) {
	      ar[ia] += .5*pcr(zr,zi,l+n*d,i+n*d);
	      ai[ia] += .5*pci(zr,zi,l+n*d,i+n*d);
	    }

	  // if (j==l) for (n=0; n < nz; n++) adata(ia) -= .5*z(1,i,n)*z(1,k,n)*z(0,l,n)*z(0,j,n);
	  if (j==l) for (n=0; n < nz; n++) {
	      xr = pr(zr,zi,i+n*d,k+n*d);
	      xi = -pi(zr,zi,i+n*d,k+n*d);
	      yr = pr(zr,zi,l+n*d,j+n*d);
	      yi = pi(zr,zi,l+n*d,j+n*d);
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

