"""
Test script for hybrid montecarlo sampling.
"""

# make sure phasemodel package is in path
import sys,os
cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(cwd)[0])[0])

import numpy as np
import phasemodel
import phasemodel.utils as utils
import phasemodel.hmc2 as hmc2
import phasemodel.f_energy as en

def test_hmc_gen():
    # load test data
    datadir = os.path.join(os.path.split(phasemodel.__file__)[0],'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_gen_v2.npz'))
    for var in mdict.files:
        globals()[var] = mdict[var]

    sz = M_true.shape[0]/2

    # some settings
    opts = hmc2.opt(
        nsamples = 10**3,
        nomit = 10**3,
        steps = 50,
        stepadj = .15,
        persistence = False)

    # generate test data
    utils.tic()
    samps = hmc2.hmc2(en.f_phasedist,np.zeros(sz),opts,en.g_phasedist,M_true)
    data = utils.smod(samps.T)
    utils.toc()

    # fit test data
    M_fit = phasemodel.fit_gen_model(data);

    print M_true
    print M_fit

    M_error = (abs(M_true-M_fit)).mean()

    print """

    mean-absolute-difference = %6.8f; expect: 0.2

    """%(M_error)


if __name__ == '__main__':
    test_hmc_gen()

