"""
Test script for hybrid montecarlo sampling.
"""

import numpy as np
import phasemodel
import utils
import hmc2
# import f_energy as en

import os
os.environ['C_INCLUDE_PATH']=np.get_include()
import pyximport; pyximport.install()
import f_energy as en

def test_hmc_biased():
    # generate random coupling
    dim = 5
    
    K_true = np.random.randn(dim+1,dim+1)+1j*np.random.randn(dim+1,dim+1)
    K_true[np.diag(np.ones(dim+1,bool))] = 0
    K_true = .5*(K_true+np.conj(K_true.T))

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
    samps = hmc2.hmc2(en.f_phasedist_biased,np.zeros(dim),opts,en.g_phasedist_biased,M)
    data = utils.smod(samps.T)
    utils.toc()

    # fit test data
    K_fit = phasemodel.fit_model_biased(data);

    print K_true
    print K_fit

    K_error = (abs(K_true-K_fit)).mean()

    print """

    mean-absolute-difference = %6.8f; expect: 0.1

    """%(K_error)


if __name__ == '__main__':
    test_hmc_biased()

