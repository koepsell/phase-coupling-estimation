import numpy as np

try:
    from rpy import r as R
    R.library("CircStats")
except:
    print "could not load R-project"


def circular_mean(phases):
    return np.mean(np.exp(1j*phases))

def circular_correlation(phases1,phases2):
    return circular_mean(phases2-phases1)

def circular_variance(phases):
    return 1-abs(circular_mean(phases))**2


def phasedist(phases):
    "Returns parameters of Von Mises distribution fitted to phase data"
    n = len(phases)
    (kappa,mu,p) = mises_params(circular_mean(phases),n)
    return (kappa,mu,p,n)

def mises(phi,kappa,mu):
    "Returns the Von Mises distribution with mean mu and concentration kappa"
    from scipy.special import i0
    return (1./(2.*np.pi*i0(kappa)))*np.exp(kappa*np.cos(phi-mu))

def mises_params(direction,n=1):
    from scipy.optimize import fmin
    from scipy.special import i0,i1
    "Computes parameters of Von Mises distribution from direction vector"
    def bess(x,r):
        return (i1(x)/i0(x)-r)**2
    try: # using R (faster by a factor of 10)
        kappa = R.A1inv(np.abs(direction))
    except:
        kappa = float(fmin(bess,np.array([1.]),(np.abs(direction),),disp=0));
    mu = np.angle(direction)
    z = float(n)*np.abs(direction)**2
    p = np.exp(-z)*(1.+(2.*z-z**2)/(4.*n)-
                 (24.*z-132.*z**2+76.*z**3-9.*z**4)/(288.*n**2))
    return kappa,mu,p

def phasecorr(phi,get_kappa=False):
    d = phi.shape[0]
    cpos = np.zeros((d,d),'D')
    cneg = np.zeros((d,d),'D')
    for i in range(d-1):
        for j in range(i+1,d):
            if get_kappa:
                (kappa,mu,p,n) = phasedist(phi[i,:]-phi[j,:])
                cneg[i,j] = kappa*np.exp(-1j*mu)
                cneg[j,i] = np.conj(cneg[i,j])
                (kappa,mu,p,n) = phasedist(phi[i,:]+phi[j,:])
                cpos[i,j] = cpos[j,i] = kappa*np.exp(-1j*mu)
            else:
                cneg[i,j] = circular_mean(-phi[i,:]+phi[j,:])
                cneg[j,i] = np.conj(cneg[i,j])
                cpos[i,j] = cpos[j,i] = circular_mean(phi[i,:]+phi[j,:])
    return cneg,cpos

