"""
Tests the accuracy of the algorithm and the code implementation.

:Authors: Charles Cadieu <cadieu@berkeley.edu> and
          Kilian Koepsell <kilian@berkeley.edu>

:Reference: Cadieu CF, Koepsell K (2010) Phase coupling estimation from
            multivariate phase statistics. Neural Computation (in press).

:Copyright: 2008-2010, UC Berkeley
:License: BSD Style
"""

# make sure phasemodel package is in path
import sys,os
cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,os.path.join(cwd,"..",".."))

import numpy as np
import phasemodel

def test_phasemodel():
    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files:
        globals()[var] = mdict[var]

    # fit test data
    K_fit = phasemodel.model.fit_model(data);

    print K_true
    print K_fit

    K_error = (abs(K_true-K_fit)).mean()
    code_error = (abs(K_python-K_fit)).mean()

    print """

    mean-absolute-difference = %6.8f; expect: 0.01730561
    print 'difference from python code = %6.8f; expect: 0.0

    """%(K_error,code_error)

    np.testing.assert_almost_equal(code_error,0)

def test_reg_phasemodel():
    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files:
        globals()[var] = mdict[var]

    subsampled_data = data[:,::100].copy()
    # fit test data
    eps = 18.
    K_fit = phasemodel.model.fit_model(subsampled_data,eps);

    print K_true
    print K_fit

    K_error = (abs(K_true-K_fit)).mean()

    print 'eps = ', eps
    print """

    mean-absolute-difference = %6.8f; expect: 0.15093053

    """%(K_error)

    np.testing.assert_almost_equal(K_error, 0.15093053)

def test_compare_weave_cython():
    from phasemodel.model_weave import fill_model_matrix
    from phasemodel.model_cython import fill_model_matrix as cfill_model_matrix

    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files:
        globals()[var] = mdict[var]

    adata, arow, acol, b = fill_model_matrix(data)
    cadata, carow, cacol, cb = cfill_model_matrix(data)
    np.testing.assert_equal(adata,cadata)
    np.testing.assert_equal(arow,carow)
    np.testing.assert_equal(acol,cacol)
    np.testing.assert_equal(b,cb)    

if __name__ == '__main__':
    import nose
    nose.runmodule()

    # test_phasemodel()
    # test_reg_phasemodel()
