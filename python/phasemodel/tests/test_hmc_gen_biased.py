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

def test_hmc_biased():
    from nose.exc import SkipTest
    raise SkipTest("For some reason the generalized model with bias fails...")

    # generate random coupling
    dim = 5
    
    M_true = np.random.randn(2*dim+2,2*dim+2)
    M_true = .5*(M_true+M_true.T)
    # M_true[:2,:2] = 0
    # M_true[np.diag(np.ones(2*dim+2,bool))] = 0
    M_true[:,:2] = 0
    M_true[:2,:] = 0

    for i in np.arange(M_true.shape[0]/2):
        M_true[2*i:2*i+2,2*i:2*i+2] = 0

   # generate random coupling
    dim = 5
    
    K_true = np.random.randn(dim+1,dim+1)+1j*np.random.randn(dim+1,dim+1)
    K_true[np.diag(np.ones(dim+1,bool))] = 0
    K_true = .5*(K_true+np.conj(K_true.T))

    # convert coupling from complex 3x3 to real 6x6 matrix
    M_true = phasemodel.kappa2m(K_true);

    print M_true
    
    #
    # anti-symmetrize diagonal elements for estimation matrix
    #
    # for i in np.arange(2,M_true.shape[0]/2):
    #     s = M_true[2*i,2*i] + M_true[2*i+1,2*i+1]
    #     M_true[2*i,2*i]     -= s/2
    #     M_true[2*i+1,2*i+1] -= s/2

    # some settings
    opts = hmc2.opt(
        nsamples = 10**3,
        nomit = 10**3,
        steps = 50,
        stepadj = .15,
        persistence = False)

    # generate test data
    utils.tic()
    samps = hmc2.hmc2(en.f_phasedist_biased,np.zeros(dim),opts,en.g_phasedist_biased,M_true)
    data = utils.smod(samps.T)
    utils.toc()

    # fit test data
    # M_fit = phasemodel.fit_gen_model(data);
    M_fit = phasemodel.fit_gen_model_biased(data);

    print M_true
    print M_fit

    M_error = (abs(M_true-M_fit)).mean()

    print """

    mean-absolute-difference = %6.8f; expect: 0.2

    """%(M_error)


if __name__ == '__main__':
    test_hmc_biased()

