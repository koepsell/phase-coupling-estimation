"""
This module contains utilities
"""
import numpy as np
import os,sys
import time

if sys.platform == 'win32':
    now = time.clock
else:
    now = time.time

def flush():
    sys.stdout.flush()

def smod(x,p=2*np.pi):
    "Returns x mod p, symmetrically centered at zero"
    return np.mod(x+p/2,p)-p/2.

_timer = now()
def tic(timer='timer'):
    print 'tic (%s) ... '%timer
    globals()['_'+timer] = now()
    flush()

def toc(timer='timer'):
    _timer = globals().get('_'+timer,'_timer')
    dt = now()-_timer
    if dt < 1: print "toc (%s) ... %.1f ms"% (timer,1000.*dt)
    else: print "toc (%s) ... %.1f s"% (timer,dt)
    flush()
    return dt

