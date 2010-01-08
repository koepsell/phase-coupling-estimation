"""Energy functions for phase model
"""

import numpy as np


#-----------------------------------------------------------------------------
# Energy functions
#-----------------------------------------------------------------------------

def f_phasedist(theta,M):
    """Energy function for phase model
    """
    x = np.zeros(2*len(theta))
    x[::2] = np.cos(theta)
    x[1::2] = np.sin(theta)
    return -.5*np.dot(np.dot(x,M),x)

def f_phasedist_biased(theta,M):
    """Energy function for phase model with bias term
    """
    x = np.zeros(2*len(theta)+2)
    x[0] = 1
    x[2::2] = np.cos(theta)
    x[3::2] = np.sin(theta)
    return -.5*np.dot(np.dot(x,M),x)


#-----------------------------------------------------------------------------
# Energy gradients
#-----------------------------------------------------------------------------

def g_phasedist(theta,M):
    """Energy gradient for phase model
    """
    x = np.zeros(2*len(theta))
    x[::2] = np.cos(theta)
    x[1::2] = np.sin(theta)
    xdot = np.zeros((2*len(theta),len(theta)))
    xdot[::2,:] = np.diag(-np.sin(theta))
    xdot[1::2,:] = np.diag(np.cos(theta)) 
    return -np.dot(np.dot(x,M),xdot)

def g_phasedist_biased(theta,M):
    """Energy gradient for phase model with bias term
    """
    x = np.zeros(2*len(theta)+2)
    x[0] = 1
    x[2::2] = np.cos(theta)
    x[3::2] = np.sin(theta)
    xdot = np.zeros((2*len(theta)+2,len(theta)))
    xdot[2::2,:] = np.diag(-np.sin(theta))
    xdot[3::2,:] = np.diag(np.cos(theta)) 
    return -np.dot(np.dot(x,M),xdot)

#-----------------------------------------------------------------------------
# Score functions
#-----------------------------------------------------------------------------

def s_phasedist(theta,M):
    """Score function for phase model

    4 S = x* M M x + Re( x* M Q conj( M x) ) + 8 E(x,M)
        = x* M M x + Re( x* M Q conj( M x) ) + 4 x* M x
    """
    x = np.zeros(2*len(theta))
    x[::2] = np.cos(theta)
    x[1::2] = np.sin(theta)
    xm = np.dot(x,M)
    mx = np.dot(M,x)
    return -.5*np.dot(np.dot(x,M),x)
