"""
Tests the accuracy of the algorithm and the code implementation.

:Authors: Charles Cadieu <cadieu@berkeley.edu> and
          Kilian Koepsell <kilian@berkeley.edu>

:Reference: C. Cadieu and K. Koepsell, A Multivaraite Phase Distribution and its
            Estimation, NIPS, 2009 (in submission).

:Copyright: 2008, UC Berkeley
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
    K_fit = phasemodel.fit_model(data);

    print K_true
    print K_fit

    K_error = (abs(K_true-K_fit)).mean()
    code_error = (abs(K_python-K_fit)).mean()

    print """

    mean-absolute-difference = %6.8f; expect: 0.01730561
    print 'difference from python code = %6.8f; expect: 0.0

    """%(K_error,code_error)

    np.testing.assert_almost_equal(code_error,0)

if __name__ == '__main__':
    test_phasemodel()
