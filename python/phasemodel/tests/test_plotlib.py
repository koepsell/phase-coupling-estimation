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
    for var in mdict.files: globals()[var] = mdict[var]

    phasemodel.plotlib.plot_phasedist(data)


@nose.tools.nottest
def test_plot_joint_phasedist():
    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files: globals()[var] = mdict[var]

    phasemodel.plotlib.plot_joint_phasedist(data)


@nose.tools.nottest
def test_plot_graph():
    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files: globals()[var] = mdict[var]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    phasemodel.plotlib.plot_graph(np.abs(K_true),start_angle=.5*np.pi,stop_angle=1.5*np.pi,endpoint=True,ax=ax)
    ax = fig.add_subplot(122)
    phasemodel.plotlib.plot_graph(np.abs(K_true),start_angle=0,stop_angle=2*np.pi,endpoint=False,ax=ax)

@nose.tools.nottest
def test_plot_matrix():
    # load test data
    datadir = os.path.join(os.path.dirname(phasemodel.__file__),'tests','testdata')
    mdict = np.load(os.path.join(datadir,'three_phases_v2.npz'))
    for var in mdict.files: globals()[var] = mdict[var]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    phasemodel.plotlib.plot_matrix(np.abs(K_true),ax=ax)

    # color bar
    from matplotlib import mpl
    fig.subplots_adjust(bottom=.25)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
    norm = mpl.colors.Normalize(vmin=0, vmax=np.abs(K_true).max())
    mpl.colorbar.ColorbarBase(ax, cmap=plt.cm.Reds, norm=norm, orientation='horizontal')

    
if __name__ == "__main__":
    test_plot_graph()
    test_plot_phasedist()
    test_plot_joint_phasedist()
    test_plot_matrix()
    plt.show()
