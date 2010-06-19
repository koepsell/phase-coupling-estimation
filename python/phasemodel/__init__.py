"""
Phasemodel Package: XXX: put some short description here

The module has several sub-modules: 

- ``model``: contains the main fitting routines

- ``utils``: cantain various utilities

- ``plotlib``: contains various plotting routines

- ``f_energy``: contains various energy functions

All of the sub-modules will be imported as part of ``__init__``, so that users
have all of these things at their fingertips.
"""

__docformat__ = 'restructuredtext'

from version import version as __version__
__status__   = 'alpha'
__url__     = 'http://redwood.berkeley.edu'

__use_cython__ = True
if __use_cython__:
    try:
        import os
        import numpy as np
        os.environ['C_INCLUDE_PATH']=np.get_include()
        import pyximport; pyximport.install()
    except:
        print "WARNING: Could not load pyximport"
        __use_cython__ = False

__use_weave__ = True
if __use_weave__:
    try:
        from scipy import weave
        weave.inline('std::cout << "weave works!" << std::endl;')
    except:
        print "WARNING: Coul not load weave"
        __use_weave__ = False

__use_R__ = True
if __use_R__:
    try:
        from rpy import r as R
        R.library("CircStats")
    except:
        print "WARNING: Could not load R-project"
        __use_R__ = False

#
# load (and reload) modules
#
modules = ['model','utils','circstats','plotlib','f_energy']
for name in modules:
    mod = __import__(name,globals(),locals(),[])
    # reload modules (useful during development)
    reload(mod)

# from model import *
# from circstats import *

# These lines commented out because "Tester" not in all versions of Numpy
#
#from numpy.testing import Tester
#test = Tester().test
#bench = Tester().bench
