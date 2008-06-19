"""
Tests the accuracy of the algorithm and the code implementation.

:Authors: Charles Cadieu <cadieu@berkeley.edu> and
          Kilian Koepsell <kilian@berkeley.edu>

:Copyright: 2008, UC Berkeley
:License: BSD Style
"""

import numpy as np
import phasemodel

def test_gen_phasemodel():
    # load test data
    mdict = np.load('testdata/three_phases_gen.npz')
    for var in mdict.files:
        globals()[var] = mdict[var]

    # fit test data
    M_fit = phasemodel.fit_gen_model(data);

    print M_true
    print M_fit

    M_error = (abs(M_true-M_fit)).mean()
    code_error = (abs(M_python-M_fit)).mean()

    print """

    mean-absolute-difference = %6.8f; expect: 0.04363234
    print 'difference from python code = %6.8f; expect: 0.0

    """%(M_error,code_error)

    np.testing.assert_almost_equal(code_error,0)

if __name__ == '__main__':
    test_gen_phasemodel()
