"""
Test script for caching random number generation (for speedup)
"""

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
