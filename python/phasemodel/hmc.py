"""hmc2.py Hybrid Monte Carlo Sampling

   This code is ported from the matlab functions hmc.m which is part
   of the netlab toolbox: http://www.ncrg.aston.ac.uk/netlab/index.php
       
       Copyright (c) Ian T Nabney (1996-2001)
       Copyright (c) Kilian Koepsell (2009)
"""

#-----------------------------------------------------------------------------
# Public interface
#-----------------------------------------------------------------------------
__all__ = ['hmc',
           'set_state',
           'get_state']
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import numpy as np
import os

#-----------------------------------------------------------------------------
# Module globals
#-----------------------------------------------------------------------------

# Global variable to store state of momentum variables: set by set_state
# Used to initialise variable if set
HMC_MOM = None

#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------

def check_grad(func, grad, x0, *args):
    """Checks the gradient grad against the numerical gradient of f

    from scipy.optimize
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
    samps = hmc(en.f_phasedist, x0, opts, en.g_phasedist, M, checkgrad=True)
    error = check_grad(en.f_phasedist, en.g_phasedist, x0, M)
    nose.tools.assert_almost_equal(error,0,5)


def get_state():
    """Returns the state of the sampler.

    Returns a state dictionary that contains the state of the internal random
    number generator and the momentum of the dynamic process.  These are
    contained in fields randstate and mom respectively. The momentum state is
    only used for a persistent momentum update.
    """
    global HMC_MOM
    return dict(randstate = np.random.get_state(),
                mom = HMC_MOM)


def set_state(state):
    """Resets the state of the sampler.
        
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


#-----------------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------------

def hmc(f, x, options, gradf, *args, **kargs):
    r"""Hybrid Monte Carlo sampling.

    Uses a hybrid Monte Carlo algorithm to sample from the distribution
    P ~ exp(-f), where F is the first argument to hmc. The Markov chain starts
    at the point x, and the function gradf is the gradient of the `energy'
    function f.

    Also returns a log of the energy values (i.e. negative log probabilities)
    for the samples in ENERGIES and DIAGN, a structure containing diagnostic
    information (position, momentum and acceptance threshold) for each step of
    the chain in DIAGN.POS, DIAGN.MOM and DIAGN.ACC respectively.  All
    candidate states (including rejected ones) are stored in DIAGN.POS.


    Parameters
    ----------

    f : function
      energy function

    x : array, 1-dimensional

    display: int
      is set to 1 to display the energy values and rejection threshold at each
      step of the Markov chain. If the value is 2, then the position vectors
      at each step are also displayed.

    persistence: bool
      is set to True if momentum persistence is used; default False,
      for complete replacement of momentum variables.

    steps: int
      defines the trajectory length (i.e. the number of leap-frog
      steps at each iteration).  Minimum value 1.

    check_grad: bool
      is set to True to check the user defined gradient function.

    nsamples: int
      is the number of samples retained from the Markov chain; default 100.

    nomit: int
      is the number of samples omitted from the start of the chain; default 0.

    OPTIONS(17) defines the momentum used when a persistent update of
    (leap-frog) momentum is used.  This is bounded to the interval [0, 1).

    OPTIONS(18) is the step size used in leap-frogs; default 1/trajectory
    length.


    Returns
    -------

    (samples, {energies, pos, mom, acc})

    samples: array, 2-dimensional, shape (n,d)
      n samples from the d-dimensional probability distribution
    
    energies: array, 2-dimensional, shape (n,d)
      energies corresponding to samples.
      (only returned when retall=True)

    pos: array, 2-dimensional, shape (n,d)
      position vectors of the dynamic process.
      (only returned when retall=True)

    mom: array, 2-dimensional, shape (n,d)
      momentum vectors of the dynamic process.
      (only returned when retall=True)

    acc: array, 1-dimensional, shape (n)
      acceptance thresholds.


    """

    global HMC_MOM


    options = dict(display = 0,
                   checkgrad = False,
                   steps = 1,
                   nsamples = 100,
                   nomit = 0,
                   persistence = False,
                   decay = 0.9,
                   stepadj = 0.2,
                   stepsf = None,
                   window = 1)

    for key in kargs.keys():
        assert key in options.keys(), 'unknown option %s'%key

    options.update(kargs)

"""
display = options(1);
if (round(options(5) == 1))
  persistence = 1;
  % Set alpha to lie in [0, 1)
  alpha = max(0, options(17));
  alpha = min(1, alpha);
  salpha = sqrt(1-alpha*alpha);
else
  persistence = 0;
end
L = max(1, options(7)); % At least one step in leap-frogging
if options(14) > 0
  nsamples = options(14);
else
  nsamples = 100;       % Default
end
if options(15) >= 0
  nomit = options(15);
else
  nomit = 0;
end
if options(18) > 0
  step_size = options(18);      % Step size.
else
  step_size = 1/L;              % Default  
end




x = x(:)';              % Force x to be a row vector
nparams = length(x);

% Set up strings for evaluating potential function and its gradient.
f = fcnchk(f, length(varargin));
gradf = fcnchk(gradf, length(varargin));

% Check the gradient evaluation.
if (options(9))
  % Check gradients
  feval('gradchek', x, f, gradf, varargin{:});
end

samples = zeros(nsamples, nparams);     % Matrix of returned samples.
if nargout >= 2
  en_save = 1;
  energies = zeros(nsamples, 1);
else
  en_save = 0;
end
if nargout >= 3
  diagnostics = 1;
  diagn_pos = zeros(nsamples, nparams);
  diagn_mom = zeros(nsamples, nparams);
  diagn_acc = zeros(nsamples, 1);
else
  diagnostics = 0;
end

n = - nomit + 1;
Eold = feval(f, x, varargin{:});        % Evaluate starting energy.
nreject = 0;
if (~persistence | isempty(HMC_MOM))
  p = randn(1, nparams);                % Initialise momenta at random
else
  p = HMC_MOM;                          % Initialise momenta from stored state
end
lambda = 1;

% Main loop.
while n <= nsamples

  xold = x;                 % Store starting position.
  pold = p;                 % Store starting momenta
  Hold = Eold + 0.5*(p*p'); % Recalculate Hamiltonian as momenta have changed

  if ~persistence
    % Choose a direction at random
    if (rand < 0.5)
      lambda = -1;
    else
      lambda = 1;
    end
  end
  % Perturb step length.
  epsilon = lambda*step_size*(1.0 + 0.1*randn(1));

  % First half-step of leapfrog.
  p = p - 0.5*epsilon*feval(gradf, x, varargin{:});
  x = x + epsilon*p;
  
  % Full leapfrog steps.
  for m = 1 : L - 1
    p = p - epsilon*feval(gradf, x, varargin{:});
    x = x + epsilon*p;
  end

  % Final half-step of leapfrog.
  p = p - 0.5*epsilon*feval(gradf, x, varargin{:});

  % Now apply Metropolis algorithm.
  Enew = feval(f, x, varargin{:});      % Evaluate new energy.
  p = -p;                               % Negate momentum
  Hnew = Enew + 0.5*p*p';               % Evaluate new Hamiltonian.
  a = exp(Hold - Hnew);                 % Acceptance threshold.
  if (diagnostics & n > 0)
    diagn_pos(n,:) = x;
    diagn_mom(n,:) = p;
    diagn_acc(n,:) = a;
  end
  if (display > 1)
    fprintf(1, 'New position is\n');
    disp(x);
  end

  if a > rand(1)                        % Accept the new state.
    Eold = Enew;                        % Update energy
    if (display > 0)
      fprintf(1, 'Finished step %4d  Threshold: %g\n', n, a);
    end
  else                                  % Reject the new state.
    if n > 0 
      nreject = nreject + 1;
    end
    x = xold;                           % Reset position 
    p = pold;                           % Reset momenta
    if (display > 0)
      fprintf(1, '  Sample rejected %4d.  Threshold: %g\n', n, a);
    end
  end
  if n > 0
    samples(n,:) = x;                   % Store sample.
    if en_save 
      energies(n) = Eold;               % Store energy.
    end
  end

  % Set momenta for next iteration
  if persistence
    p = -p;
    % Adjust momenta by a small random amount.
    p = alpha.*p + salpha.*randn(1, nparams);
  else
    p = randn(1, nparams);      % Replace all momenta.
  end

  n = n + 1;
end

if (display > 0)
  fprintf(1, '\nFraction of samples rejected:  %g\n', ...
    nreject/(nsamples));
end
if diagnostics
  diagn.pos = diagn_pos;
  diagn.mom = diagn_mom;
  diagn.acc = diagn_acc;
end
% Store final momentum value in global so that it can be retrieved later
HMC_MOM = p;
return
"""
