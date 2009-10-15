"""
Test script for caching random number generation (for speedup)
"""

# make sure phasemodel package is in path
import sys,os
cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.append(os.path.split(os.path.split(cwd)[0])[0])

import numpy as np
from phasemodel import utils

import os
os.environ['C_INCLUDE_PATH']=np.get_include()
import pyximport; pyximport.install()
import random_cache as rand

utils.tic()
rand.test_random()
utils.toc()

utils.tic()
rand.test_random_cached()
utils.toc()
