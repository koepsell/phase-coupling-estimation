"""
Test script for caching random number generation (for speedup)
"""

# make sure phasemodel package is in path
import sys,os
cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,os.path.join(cwd,"..",".."))

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
