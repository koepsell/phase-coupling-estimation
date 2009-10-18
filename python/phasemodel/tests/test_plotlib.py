# make sure phasemodel package is in path
import sys,os
cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,os.path.join(cwd,"..",".."))

import numpy as np
import phasemodel
import matplotlib.pyplot as plt
import nose


@nose.tools.nottest
def test_plot_phasedist():
    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files:
        globals()[var] = mdict[var]

    phasemodel.plotlib.plot_phasedist_nd(data)


@nose.tools.nottest
def test_plot_joint_phasedist():
    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files:
        globals()[var] = mdict[var]

    phasemodel.plotlib.plot_joint_phasedist_nd(data)


if __name__ == "__main__":
    test_plot_phasedist()
    test_plot_joint_phasedist()
    plt.show()
