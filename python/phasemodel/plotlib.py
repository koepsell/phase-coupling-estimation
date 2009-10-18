import numpy as np
import matplotlib.pyplot as plt
import model
import utils


def plot_phasedist_1d(phases,nbins=37,plot_fit=True,fig=None,ax=None,linewidth=2,**kargs):
    if fig is None and ax is None: fig=plt.figure()
    if ax is None: ax = fig.add_subplot(111)
    
    # default plot arguments
    plotargs = dict(color='b',alpha=.75)
    plotargs.update(kargs)

    phases = utils.smod(phases,2*np.pi)
    (N, bins, patches) = ax.hist(phases,np.linspace(-np.pi,np.pi,nbins+1),normed=1,**kargs)
    plt.setp(patches, 'facecolor', plotargs['color'], 'alpha', plotargs['alpha'])
    (kappa,mu,p,n) = model.phasedist(phases)

    if plot_fit:
        plt.plot (np.linspace(-np.pi,np.pi,100),model.mises(np.linspace(-np.pi,np.pi,100),
                                                     kappa,mu),'k',linewidth=linewidth)
    ax.set_xlim([-np.pi,np.pi])
    ax.set_xlabel('phase (rad)')
    ax.set_title ('[kappa=%4.2f, mu=%3.1f]' % (kappa,180.*mu/np.pi))
    print 'kappa = %4.2f, mu = %1.2f, N = %d' % (kappa,mu,n)
    return kappa,mu


def plot_phasedist_nd(phases,fig=None,**kargs):
    if fig is None: fig=plt.figure()
    ylim = (0,1.5)

    dim = phases.shape[0]
    for row in xrange(dim):
        for col in xrange(dim):
            ax = fig.add_subplot(dim,dim,row*dim+col+1)
            if row == col:
                plot_phasedist_1d(phases[row], ax=ax, **kargs)
                # ax.set_title(r'$\phi_%d$'%row)
                ax.text(-2.8,1.4,r'$\phi_%d$'%row,va='top')
            elif row > col:
                plot_phasedist_1d(phases[row]-phases[col], ax=ax, **kargs)
                # ax.set_title(r'$\phi_%d-\phi_%d$'%(row,col))
                ax.text(-2.8,1.4,r'$\phi_%d-\phi_%d$'%(row,col),va='top')
            elif row < col:
                plot_phasedist_1d(phases[row]+phases[col], ax=ax, **kargs)
                # ax.set_title(r'$\phi_%d+\phi_%d$'%(row,col))
                ax.text(-2.8,1.4,r'$\phi_%d+\phi_%d$'%(row,col),va='top')

            if row < dim-1:
                ax.set_xticklabels('')
                ax.set_xlabel('')
            ax.set_yticklabels('')
            ax.set_ylim(ylim)
            ax.set_title('')


def plot_joint_phasedist_2d(p1,p2,nbins=37,fig=None,ax=None,vmax=0.15,**kargs):
    if fig is None and ax is None: fig=plt.figure()
    if ax is None: ax = fig.add_subplot(111)
    assert p1.ndim == 1, 'phase p1 has to be one-dimensional'
    assert p2.ndim == 1, 'phase p2 has to be one-dimensional'

    nsamples = len(p1)
    extent = [-np.pi,np.pi,-np.pi,np.pi]
    bins = np.linspace(-np.pi,np.pi,nbins+1)
    phasehist = np.histogram2d(p1,p2,bins=bins)
    ax.imshow(float(nbins)*phasehist[0]/float(nsamples),interpolation='nearest',extent=extent,vmax=vmax,origin='lower',cmap=plt.cm.hot)



def plot_joint_phasedist_nd(phases,fig=None,**kargs):
    if fig is None: fig=plt.figure()

    dim = phases.shape[0]
    for row in xrange(1,dim):
        for col in xrange(dim-1):
            if row > col:
                ax = fig.add_subplot(dim-1,dim-1,(row-1)*(dim-1)+col+1)
                ax.text(-2.8,1.4,r'$\phi_%d+\phi_%d$'%(row,col),va='top')
                plot_joint_phasedist_2d(phases[row],phases[col], ax=ax, **kargs)
                if col == 0: ax.set_ylabel(r'$\phi_%d$'%row)
                if row == dim-1: ax.set_xlabel(r'$\phi_%d$'%col)
                ax.set_title(r'')


