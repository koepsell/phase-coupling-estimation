"""
Test script for hybrid montecarlo sampling.
"""

# make sure phasemodel package is in path
import sys,os
cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,os.path.join(cwd,"..",".."))

import numpy as np
import phasemodel
import phasemodel.utils as utils
import phasemodel.hmc2 as hmc2
import phasemodel.f_energy as en

def test_hmc():
    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files:
        globals()[var] = mdict[var]

    sz = K_true.shape[0] 

    # convert coupling from complex 3x3 to real 6x6 matrix
    M = phasemodel.kappa2m(K_true);

    # some settings
    opts = hmc2.opt(
        nsamples = 10**3,
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

