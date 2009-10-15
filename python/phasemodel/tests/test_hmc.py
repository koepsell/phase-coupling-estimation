"""
Test script for hybrid montecarlo sampling.
"""

import numpy as np
import phasemodel
import utils
import hmc2

import os
os.environ['C_INCLUDE_PATH']=np.get_include()
import pyximport; pyximport.install()
import f_energy as en

def test_hmc():
    # load test data
    mdict = np.load('testdata/three_phases.npz')
    for var in mdict.files:
        globals()[var] = mdict[var]

    sz = K_true.shape[0] 

    # convert coupling from complex 3x3 to real 6x6 matrix
    M = phasemodel.kappa2m(K_true);

    # some settings
    opts = hmc2.opt(
        nsamples = 10**4,
        nomit = 10**3,
        steps = 50,
        stepadj = .15,
        persistence = False)

    # generate test data
    utils.tic()
    samps = hmc2.hmc2(en.f_phasedist,np.zeros(sz),opts,en.g_phasedist,M)
    data = utils.smod(samps.T)
    utils.toc()

    # fit test data
    K_fit = phasemodel.fit_model(data);

    print K_true
    print K_fit

    K_error = (abs(K_true-K_fit)).mean()

    print """

    mean-absolute-difference = %6.8f; expect: 0.1

    """%(K_error)


if __name__ == '__main__':
    test_hmc()

