#include <math.h>
#include "mex.h"

#define Z_IN		prhs[0]		// input data
#define NIJ_IN		prhs[1]		// matrix size
#define NA_IN		prhs[2]		// sparse matrix elements

#define A_OUT		plhs[0]		// a matrix values
#define B_OUT		plhs[1]		// b vector values

/* complex multiplication, real part */
double mr(double* zr, double* zi, int zind, double* wr, double* wi, int wind)
{
  return zr[zind]*wr[wind]-zi[zind]*wi[wind];
}

/* complex multiplication, imaginary part */
double mi(double* zr, double* zi, int zind, double* wr, double* wi, int wind)
{
  return zr[zind]*wi[wind]+zi[zind]*wr[wind];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *zr, *zi;    
  double *wr, *wi;
  mxArray *x,*y;
  double *xr, *xi; 
  double *yr, *yi;    
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
  if (mxGetNumberOfDimensions(Z_IN) != 3)
    mexErrMsgTxt("Z must be 3d array.");
  
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
  
  // Allocate memory for intermediate values
  x = mxCreateDoubleMatrix(1,1,mxCOMPLEX);
  y = mxCreateDoubleMatrix(1,1,mxCOMPLEX);
  xr = mxGetPr(x);
  xi = mxGetPi(x);
  yr = mxGetPr(y);
  yi = mxGetPi(y);

  // Allocate memory and assign output pointers
  A_OUT = mxCreateSparse(nij,nij,na,mxCOMPLEX);
  B_OUT = mxCreateDoubleMatrix(nij,1,mxCOMPLEX);

  // Get a pointer to the z data
  zr = mxGetPr(Z_IN);
  zi = mxGetPi(Z_IN);
  wr = zr + d*nz;
  wi = zi + d*nz;

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
	  br[ij] -= 2.*mr(wr,wi,i+n*d,zr,zi,j+n*d);
	  bi[ij] -= 2.*mi(wr,wi,i+n*d,zr,zi,j+n*d);
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
	      xr[0] = mr(wr,wi,i+n*d,wr,wi,k+n*d);
	      xi[0] = mi(wr,wi,i+n*d,wr,wi,k+n*d);
	      yr[0] = mr(zr,zi,l+n*d,zr,zi,j+n*d);
	      yi[0] = mi(zr,zi,l+n*d,zr,zi,j+n*d);
	      ar[ia] -= .5*mr(xr,xi,0,yr,yi,0);
	      ai[ia] -= .5*mi(xr,xi,0,yr,yi,0);
	    }

	  // if (i==l) for (n=0; n < nz; n++) adata(ia) += .5*z(0,j,n)*z(1,k,n);
	  if (i==l) for (n=0; n < nz; n++) {
	      ar[ia] += .5*mr(zr,zi,j+n*d,wr,wi,k+n*d);
	      ai[ia] += .5*mi(zr,zi,j+n*d,wr,wi,k+n*d);
	    }

	  // if (j==k) for (n=0; n < nz; n++) adata(ia) += .5*z(1,i,n)*z(0,l,n);
	  if (j==k) for (n=0; n < nz; n++) {
	      ar[ia] += .5*mr(wr,wi,i+n*d,zr,zi,l+n*d);
	      ai[ia] += .5*mi(wr,wi,i+n*d,zr,zi,l+n*d);
	    }

	  // if (j==l) for (n=0; n < nz; n++) adata(ia) -= .5*z(1,i,n)*z(1,k,n)*z(0,l,n)*z(0,j,n);
	  if (j==l) for (n=0; n < nz; n++) {
	      xr[0] = mr(wr,wi,i+n*d,wr,wi,k+n*d);
	      xi[0] = mi(wr,wi,i+n*d,wr,wi,k+n*d);
	      yr[0] = mr(zr,zi,l+n*d,zr,zi,j+n*d);
	      yi[0] = mi(zr,zi,l+n*d,zr,zi,j+n*d);
	      ar[ia] -= .5*mr(xr,xi,0,yr,yi,0);
	      ai[ia] -= .5*mi(xr,xi,0,yr,yi,0);	      
	    }
	  kl++;
	  ia++;
	}
      }
    }
  }
  acol[nij] = ia;
  mxDestroyArray(x);
  mxDestroyArray(y);
  return;
}

