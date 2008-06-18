#include <math.h>
#include "mex.h"

#define Z_IN		prhs[0]		// input data
#define NIJ_IN		prhs[1]		// matrix size
#define NA_IN		prhs[2]		// sparse matrix elements

#define A_OUT		plhs[0]		// a matrix values
#define B_OUT		plhs[1]		// b vector values

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	
	// Allocate memory and assign output pointers
	plhs[0] = mxCreateSparse(nij,nij,na,mxComplex)
	plhs[1] = mxCreateDoubleMatrix(na,na,mxComplex)
	
	// Get a pointer to the data space in allocated memory
	a = mxGetPr(plhs[0])
	b = mxGetPr(plhs[0])
	
}