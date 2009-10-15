"""
Phasemodel Package: XXX: put some short description here

The module has several sub-modules: 

- ``core``: contains the main fitting routines

- ``model``: contains modeling code (copied from neuropy, should probably factored away)

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
        print "Warning: Could not find pyximport"
        __use_cython__ = False

import core, model, utils, plotlib, f_energy

from core import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
