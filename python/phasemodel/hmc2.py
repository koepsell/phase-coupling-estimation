"""Hybrid Monte Carlo Sampling

   This package is a straight-forward port of the functions hmc2.m and
   hmc2_opt.m from the MCMCstuff matlab toolbox written by Aki Vehari
   <http://www.lce.hut.fi/research/mm/mcmcstuff/>.
   
   The code is originally based on the functions hmc.m from the netlab toolbox
   written by Ian T Nabney <http://www.ncrg.aston.ac.uk/netlab/index.php>.

   The portion of algorithm involving "windows" is derived from the C code for
   this function included in the Software for Flexible Bayesian Modeling
   written by Radford Neal <http://www.cs.toronto.edu/~radford/fbm.software.html>.
   
       Copyright (c) 1996-1998 Ian T Nabney
       Copyright (c) 1998-2000 Aki Vehtari
       Copyright (c) 2008-2009 Kilian Koepsell

   % This software is distributed under the GNU General Public 
   % License (version 2 or later); please refer to the file 
   % License.txt, included with the software, for details.
"""

import numpy as np
import os

# Global variable to store state of momentum variables: set by set_state
# Used to initialize variable if set
HMC_MOM = None

class opt(object):

    @property
    def display(self):
        return self._display

    @property
    def checkgrad(self):
        return self._checkgrad

    @property
    def steps(self):
        return self._steps

    @property
    def nsamples(self):
        return self._nsamples

    @property
    def nomit(self):
        return self._nomit

    @property
    def persistence(self):
        return self._persistence

    @property
    def decay(self):
        return self._decay

    @property
    def stepadj(self):
        return self._stepadj

    @property
    def stepsf(self):
        return self._stepsf

    @property
    def window(self):
        return self._window

    def __init__(self,**kargs):
        self._display = kargs.get('display',False)
        self._checkgrad = kargs.get('checkgrad',False)
        self._steps = kargs.get('steps',1)
        self._nsamples = kargs.get('nsamples',1)
        self._nomit = kargs.get('nomit', 0)
        self._persistence = kargs.get('persistence', False)
        self._decay = kargs.get('decay', 0.9)
        self._stepadj = kargs.get('stepadj', 0.2)
        self._stepsf = kargs.get('stepsf', None)
        self._window = kargs.get('window', 1)

        assert self.steps >= 1, 'step size has to be >= 1'
        assert self.nsamples >= 1, 'nsamples has to be >= 1'
        assert self.nomit >= 0, 'nomit has to be >= 0'
        assert self.decay >= 0, 'decay has to be >= 0'
        assert self.decay <= 1, 'decay has to be <= 1'
        assert self.window >= 0, 'window has to be >= 0'
        if self.window > self.steps: self._window = self.steps

def test_default_opt():
    import nose
    myopt = opt()
    for key,val in options.items():
        yield nose.tools.assert_equal, getattr(myopt,key), val

def test_set_opt():
    import nose
    options = dict(display = True,
                   checkgrad = True,
                   steps = 2,
                   nsamples = 100,
                   nomit = 1,
                   persistence = True,
                   decay = 0.8,
                   stepadj = 0.1,
                   stepsf = None,
                   window = 2)
    myopt = opt(**options)
    for key,val in options.items():
        yield nose.tools.assert_equal, getattr(myopt,key), val

def check_grad(func, grad, x0, *args):
    """from scipy.optimize
    """
    _epsilon = np.sqrt(np.finfo(float).eps)
    def approx_fprime(xk,f,epsilon,*args):
        f0 = f(*((xk,)+args))
        grad = np.zeros((len(xk),), float)
        ei = np.zeros((len(xk),), float)
        for k in range(len(xk)):
            ei[k] = epsilon
            grad[k] = (f(*((xk+ei,)+args)) - f0)/epsilon
            ei[k] = 0.0
        return grad
    
    return np.sqrt(np.sum((grad(x0,*args)-approx_fprime(x0,func,_epsilon,*args))**2))


def test_checkgrad():
    import nose
    import f_energy as en
    sz = 10
    x0 = np.zeros(sz)
    M = np.random.rand(2*sz,2*sz)
    M += M.T.copy()
    opts = opt(checkgrad=True)
    samps = hmc2(en.f_phasedist, x0, opts, en.g_phasedist, M)
    error = check_grad(en.f_phasedist, en.g_phasedist, x0, M)
    nose.tools.assert_almost_equal(error,0,5)


def hmc2(f, x, options, gradf, *args, **kargs):
    """
    SAMPLES = HMC2(F, X, OPTIONS, GRADF)

    Description
      SAMPLES = HMC2(F, X, OPTIONS, GRADF) uses a  hybrid Monte Carlo
      algorithm to sample from the distribution P ~ EXP(-F), where F is the
      first argument to HMC2. The Markov chain starts at the point X, and
      the function GRADF is the gradient of the `energy' function F.

      HMC2(F, X, OPTIONS, GRADF, P1, P2, ...) allows additional arguments to
      be passed to F() and GRADF().

      [SAMPLES, ENERGIES, DIAGN] = HMC2(F, X, OPTIONS, GRADF) also returns a
      log of the energy values (i.e. negative log probabilities) for the
      samples in ENERGIES and DIAGN, a structure containing diagnostic
      information (position, momentum and acceptance threshold) for each
      step of the chain in DIAGN.POS, DIAGN.MOM and DIAGN.ACC respectively.
      All candidate states (including rejected ones) are stored in
      DIAGN.POS. The DIAGN structure contains fields: 

      pos
       the position vectors of the dynamic process
      mom
       the momentum vectors of the dynamic process
      acc
       the acceptance thresholds
      rej
       the number of rejections
      stp
       the step size vectors
    """
    global HMC_MOM

    # Reference to structures is much slower, so...
    opt_nsamples = options.nsamples
    opt_nomit = options.nomit
    opt_window = options.window
    opt_steps = options.steps
    opt_display = options.display
    opt_persistence = options.persistence

    if opt_persistence:
        alpha = options.decay
        salpha = np.sqrt(1-alpha**2);
    else:
        alpha = salpha = 0.

    # TODO: not implemented yet. Haven't figured out how this is supposed to work...
    if options.stepsf is not None:
        # Stepsizes, varargin gives the opt.stepsf arguments (net, x ,y)
        # where x is input data and y is a target data.
        # epsilon = feval(opt.stepsf,varargin{:}).*opt.stepadj;
        raise NotImplementedError
    else:
        epsilon = options.stepadj

    nparams = len(x)

    # Check the gradient evaluation.
    if options.checkgrad:
        # Check gradients
        error = check_grad(f, gradf, x, *args)
        print "Energy gradient error: %f"%error

    # Initialize matrix of returned samples
    samples = np.zeros((opt_nsamples, nparams))

    # Check all keyword arguments
    known_keyargs = ['return_energies','return_diagnostics']
    for key in kargs.keys():
        assert key in known_keyargs, 'unknown option %s'%key

    # Return energies?
    return_energies = kargs.get('return_energies',False)
    if return_energies:
        energies = np.zeros(opt_nsamples)
    else:
        energies = np.zeros(0)

    # Return diagnostics?
    return_diagnostics = kargs.get('return_diagnostics',False)
    if return_diagnostics:
        diagn_pos = np.zeros(opt_nsamples, nparams)
        diagn_mom = np.zeros(opt_nsamples, nparams)
        diagn_acc = np.zeros(opt_nsamples)
    else:
        diagn_pos = np.zeros((0,0))
        diagn_mom = np.zeros((0,0))
        diagn_acc = np.zeros(0)

    if not opt_persistence or HMC_MOM is None or nparams != len(HMC_MOM):
        # Initialise momenta at random
        p = np.random.randn(nparams)
    else:
        # Initialise momenta from stored state
        p = HMC_MOM
        
    # Main loop.
    all_args = [f,
                x,
                gradf,
                args,
                p,
                samples,
                energies,
                diagn_pos,
                diagn_mom,
                diagn_acc,
                opt_nsamples,
                opt_nomit,
                opt_window,
                opt_steps,
                opt_display,
                opt_persistence,
                return_energies,
                return_diagnostics,
                alpha,
                salpha,
                epsilon]

    try:
        os.environ['C_INCLUDE_PATH']=np.get_include()
        import pyximport; pyximport.install()
        from hmc2c import hmc_main_loop as c_hmc_main_loop
        print "Using compiled code"
        c_hmc_main_loop(*all_args)
    except:
        print "Using pure python code"
        hmc_main_loop(*all_args)

    if opt_display:
        print '\nFraction of samples rejected:  %g\n'%(nreject/float(opt_nsamples))

    # Store diagnostics
    if return_diagnostics:
        diagn = dict()
        diagn['pos'] = diagn_pos   # positions matrix
        diagn['mom'] = diagn_mom   # momentum matrix
        diagn['acc'] = diagn_acc   # acceptance treshold matrix
        diagn['rej'] = nreject/float(opt_nsamples)   # rejection rate
        diagn['stps'] = epsilon    # stepsize vector

    # Store final momentum value in global so that it can be retrieved later
    if opt_persistence:
        HMC_MOM = p
    else:
        HMC_MOM = None

    if return_energies or return_diagnostics:
        out = (samples,)
    else:
        return samples
    
    if return_energies: out += (energies,)
    if return_diagnostics: out += (diagn,)
    return out

def hmc_main_loop(f, x, gradf, args, p, samples,
                  energies, diagn_pos, diagn_mom, diagn_acc,
                  opt_nsamples, opt_nomit, opt_window, opt_steps, opt_display,
                  opt_persistence, return_energies, return_diagnostics,
                  alpha, salpha, epsilon):
    nparams = len(x)
    nreject = 0              # number of rejected samples
    window_offset = 0        # window offset initialised to zero
    k = -opt_nomit       # nomit samples are omitted, so we store
    
    # Evaluate starting energy.
    E = f(x, *args)

    while k < opt_nsamples:  # samples from k >= 0
        # Store starting position and momenta
        xold = x
        pold = p
        # Recalculate Hamiltonian as momenta have changed
        Eold = E
        # Hold = E + 0.5*(p*p')
        Hold = E + 0.5*(p**2).sum()

        # Decide on window offset, if windowed HMC is used
        if opt_window > 1:
            # window_offset=fix(opt_window*rand(1));
            window_offset = int(opt_window*np.random.rand())

        have_rej = 0
        have_acc = 0
        n = window_offset
        direction = -1 # the default value for direction 
                       # assumes that windowing is used

        while direction == -1 or n != opt_steps:
            # if windowing is not used or we have allready taken
            # window_offset steps backwards...
            if direction == -1 and n==0:
                # Restore, next state should be original start state.
                if window_offset > 0:
                    x = xold
                    p = pold
                    n = window_offset

                # set direction for forward steps
                E = Eold
                H = Hold
                direction = 1
                stps = direction
            else:
                if n*direction+1<opt_window or n > (opt_steps-opt_window):
                    # State in the accept and/or reject window.
                    stps = direction
                else:
                    # State not in the accept and/or reject window. 
                    stps = opt_steps-2*(opt_window-1)

                # First half-step of leapfrog.
                # p = p - direction*0.5*epsilon.*feval(gradf, x, varargin{:});
                p = p - direction*0.5*epsilon*gradf(x, *args)
                x = x + direction*epsilon*p
                
                # Full leapfrog steps.
                # for m = 1:(abs(stps)-1):
                for m in xrange(abs(stps)-1):
                    # p = p - direction*epsilon.*feval(gradf, x, varargin{:});
                    p = p - direction*epsilon*gradf(x, *args)
                    x = x + direction*epsilon*p

                # Final half-step of leapfrog.
                # p = p - direction*0.5*epsilon.*feval(gradf, x, varargin{:});
                p = p - direction*0.5*epsilon*gradf(x, *args)

                # E = feval(f, x, varargin{:});
                E = f(x, *args)
                # H = E + 0.5*(p*p');
                H = E + 0.5*(p**2).sum()

                n += stps

            if opt_window != opt_steps+1 and n < opt_window:
                # Account for state in reject window.  Reject window can be
                # ignored if windows consist of the entire trajectory.
                if not have_rej:
                    rej_free_energy = H
                else:
                    rej_free_energy = -addlogs(-rej_free_energy, -H)

                if not have_rej or np.random.rand() < np.exp(rej_free_energy-H):
                    E_rej = E
                    x_rej = x
                    p_rej = p
                    have_rej = 1

            if n > (opt_steps-opt_window):
                # Account for state in the accept window.
                if not have_acc:
                    acc_free_energy = H
                else:
                    acc_free_energy = -addlogs(-acc_free_energy, -H)

                if not have_acc or  np.random.rand() < np.exp(acc_free_energy-H):
                    E_acc = E
                    x_acc = x
                    p_acc = p
                    have_acc = 1
  
        # Acceptance threshold.
        a = np.exp(rej_free_energy - acc_free_energy)

        if return_diagnostics and k >= 0:
            diagn_pos[k,:] = x_acc
            diagn_mom[k,:] = p_acc
            diagn_acc[k,:] = a

        if opt_display:
            print 'New position is\n',x

        # Take new state from the appropriate window.
        if a > np.random.rand():
            # Accept 
            E = E_acc
            x = x_acc
            p = -p_acc # Reverse momenta
            if opt_display:
                print 'Finished step %4d  Threshold: %g\n'%(k,a)
        else:
            # Reject
            if k >= 0:
                nreject = nreject + 1

            E = E_rej
            x = x_rej
            p = p_rej
            if opt_display:
                print '  Sample rejected %4d.  Threshold: %g\n'%(k,a)

        if k >= 0:
            # Store sample
            samples[k,:] = x;
            if return_energies:
                # Store energy
                energies[k] = E

        # Set momenta for next iteration
        if opt_persistence:
            # Reverse momenta
            p = -p
            # Adjust momenta by a small random amount
            p = alpha*p + salpha*np.random.randn(nparams)
        else:
            # Replace all momenta
            p = np.random.randn(nparams)

        k += 1


def get_state():
    """Return complete state of sampler (including momentum)

            Description
            get_state() returns a state structure that contains the state of
            the internal random number generators and the momentum of the
            dynamic process. These are contained in fields randstate mom
            respectively.
            The momentum state is only used for a persistent momentum update.
    """
    global HMC_MOM
    return dict(randstate = np.random.get_state(),
                mom = HMC_MOM)


def set_state(state):
    """Set complete state of sampler (including momentum).
        
            Description
            set_state(state) resets the state to a given state.
            If state is a dictionary returned by get_state() then it resets the
            generator to exactly the same state.
    """
    global HMC_MOM
    assert type(state) == dict, 'state has to be a state dictionary'
    assert state.has_key('randstate'), 'state does not contain randstate'
    assert state.has_key('mom'), 'state does not contain momentum'
    np.random.set_state(state['randstate'])
    HMC_MOM = state['mom']


def test_set_randstate():
    import nose
    state = get_state()
    rand = np.random.rand()
    set_state(state)
    nose.tools.assert_equal(np.random.rand(),rand)


def test_set_momentum():
    global HMC_MOM
    HMC_MOM = np.ones(3)
    state = get_state()
    HMC_MOM = np.zeros(3)
    set_state(state)
    np.testing.assert_array_equal(np.ones(3),HMC_MOM)


def addlogs(a,b):
    """Add numbers represented by their logarithms.
    
            Description
            Add numbers represented by their logarithms.
            Computes log(exp(a)+exp(b)) in such a fashion that it 
            works even when a and b have large magnitude.
    """
    
    if a>b:
        return a + np.log(1+np.exp(b-a))
    else:
        return b + np.log(1+np.exp(a-b))


def test_addlogs():
    import nose
    a,b = np.random.randn(2)
    nose.tools.assert_almost_equal(addlogs(a,b),np.log(np.exp(a)+np.exp(b)))


if __name__ == '__main__':
    import nose
    # nose.runmodule(exit=False,argv=['nose','-s','--pdb-failures'])
    nose.runmodule(exit=False,argv=['nose','-s'])
