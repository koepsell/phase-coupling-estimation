import numpy as np
import matplotlib.pyplot as plt
import model
reload(model)
import utils
reload(utils)

def plot_phasedist(phases,bins=37,plot_fit=True,linewidth=2,**args):

    # default plot arguments
    plotargs = dict(figure=plt.gcf(),color='b',alpha=.75)
    plotargs.update(args)

    # get current axis
    fig = plotargs['figure']
    if hasattr(fig,'next'): #select next subplot
        fig.next()
    ax = fig.gca()

    phases = utils.smod(phases,2*np.pi)
    (N, bins, patches) = plt.hist(phases,np.linspace(-np.pi,np.pi,bins+1),normed=1,**args)
    plt.setp(patches, 'facecolor', plotargs['color'], 'alpha', plotargs['alpha'])
    (kappa,mu,p,n) = model.phasedist(phases)

    if plot_fit:
        plt.plot (np.linspace(-np.pi,np.pi,100),model.mises(np.linspace(-np.pi,np.pi,100),
                                                     kappa,mu),'k',linewidth=linewidth)
    plt.xlim([-np.pi,np.pi])
    plt.xlabel('phase (rad)')
    plt.title ('[kappa=%4.2f, mu=%3.1f, N=%d]' % (kappa,180.*mu/np.pi,n))
    print 'kappa = %4.2f, mu = %1.2f, N = %d, p = %10.10f /100' % (
        kappa,mu,n,100.*p)
    return (kappa,mu,p,n)



def plot_phasedist2d(p1,p2):
    """
    plot 2-d distributions
    """
    # plot marginal distributions (individual phase histograms)
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.plot_phasedist(th1)
    plt.title('p1 + p2')
    plt.subplot(2,2,2)
    plt.plot_phasedist(th2)
    plt.title('p1 - p2')
    plt.subplot(2,2,3)
    plot_phasedist(p1)
    plt.title('p1')
    plt.subplot(2,2,4)
    plot_phasedist(p2)
    plt.title('p2')

    # plot 2d distribution
    fig = plt.figure()
    nbins = 37
    extent = [-np.pi,np.pi,-np.pi,np.pi]
    # bins = np.arange(-np.pi,np.pi,2*np.pi/(nbins+1))
    bins = np.linspace(-np.pi,np.pi,nbins+1)
    phasehist = np.histogram2d(p1,p2,bins=bins)
    plt.imshow(phasehist[0]/float(size),interpolation='nearest',extent=extent)
    plt.colorbar()
    # plot(p1,p2,'k.')
    # axis([-np.pi,np.pi,-np.pi,np.pi])

def plot_phasedist3d(phi):
    """
    plot 2-d phased distributions
    """
    vmax = .15

    # plot marginal distributions in alpha and a
    fig = plt.figure()
    plt.subplot(3,3,1)
    plot_phasedist(phi[0,:])
    plt.title('p1')
    plt.subplot(3,3,2)
    plot_phasedist(phi[1,:])
    plt.title('p2')
    plt.subplot(3,3,3)
    try:
        plot_phasedist(phi[2,:])
    except:
        pass
    plt.title('p3')
    plt.subplot(3,3,4)
    plot_phasedist(phi[0,:]-phi[1,:])
    plt.title('p1-p2')
    plt.subplot(3,3,5)
    try:
        plot_phasedist(phi[0,:]-phi[2,:])
    except:
        pass
    plt.title('p1-p3')
    plt.subplot(3,3,6)
    try:
        plot_phasedist(phi[1,:]-phi[2,:])
    except:
        pass
    plt.title('p2-p3')
    plt.subplot(3,3,7)
    plot_phasedist(phi[0,:]+phi[1,:])
    plt.title('p1+p2')
    plt.subplot(3,3,8)
    try:
        plot_phasedist(phi[0,:]+phi[2,:])
    except:
        pass
    plt.title('p1+p3')
    plt.subplot(3,3,9)
    try:
        plot_phasedist(phi[1,:]+phi[2,:])
    except:
        pass
    plt.title('p2+p3')

    # plot 2d distribution
    nbins = 37
    extent = [-np.pi,np.pi,-np.pi,np.pi]
    # bins = np.arange(-np.pi,np.pi,2*np.pi/(nbins+1))
    bins = np.linspace(-np.pi,np.pi,nbins+1)
    phasehist12 = np.histogram2d(phi[0,:],phi[1,:],bins=bins,normed=True)
    try:
        phasehist13 = np.histogram2d(phi[0,:],phi[2,:],bins=bins,normed=True)
        phasehist23 = np.histogram2d(phi[1,:],phi[2,:],bins=bins,normed=True)
    except:
        pass

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(phasehist12[0],interpolation='nearest',extent=extent,vmax=vmax,origin='lower',cmap=plt.cm.hot)
    plt.title('p1 vs. p2')
    plt.subplot(1,3,2)
    try:
        plt.imshow(phasehist13[0],interpolation='nearest',extent=extent,vmax=vmax,origin='lower',cmap=plt.cm.hot)
    except:
        pass
    plt.title('p1 vs. p3')
    plt.subplot(1,3,3)
    try:
        plt.imshow(phasehist23[0],interpolation='nearest',extent=extent,vmax=vmax,origin='lower',cmap=plt.cm.hot)
    except:
        pass
    plt.title('p2 vs. p3')

