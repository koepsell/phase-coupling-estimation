import numpy as np
import matplotlib.pyplot as plt
import model
import utils

__all__ = ['plot_phasedist','plot_joint_phasdist','plot_graph','plot_matrix']

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


def plot_phasedist(phases,fig=None,**kargs):
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
    ax.set_xticks([-np.pi,0,np.pi])
    ax.set_yticks([-np.pi,0,np.pi])
    ax.set_xticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
    ax.set_yticklabels([r'$-\pi$',r'$0$',r'$\pi$'])


def plot_joint_phasedist(phases,fig=None,**kargs):
    if fig is None: fig=plt.figure()

    dim = phases.shape[0]
    for row in xrange(1,dim):
        for col in xrange(dim-1):
            if row > col:
                ax = fig.add_subplot(dim-1,dim-1,(row-1)*(dim-1)+col+1)
                plot_joint_phasedist_2d(phases[row],phases[col], ax=ax, **kargs)
                if col == 0: ax.set_ylabel(r'$\phi_%d$'%row)
                if row == dim-1: ax.set_xlabel(r'$\phi_%d$'%col)
                if row < dim-1:
                    ax.set_xticklabels('')
                    ax.set_xlabel('')
                if col:
                    ax.set_yticklabels('')
                    ax.set_ylabel('')
                ax.set_title(r'')


def circular_layout(G, start_angle=0, stop_angle=2.0*np.pi, endpoint=False):
    """circular graph layout

    This function is similar to the circular layout function of networkx,
    but allows to layout graph nodes on a circle segment.
    """
    import networkx as nx
    # t = np.arange(start_angle, stop_angle, (stop_angle-start_angle)/len(G), dtype=np.float32)
    t = np.linspace(start_angle, stop_angle, len(G), endpoint=endpoint)
    pos = np.transpose(np.array([np.cos(t),np.sin(t)]))
    return dict(zip(G,pos))


def plot_graph(weights, labels=None, pos=None, fig=None, ax=None,
               start_angle=.5*np.pi, stop_angle=1.5*np.pi, endpoint=True, **kargs):
    """Plot graph using networkx
    """
    try:
        import networkx as nx
    except ImportError:
        print "Warning: can't import networkx"
        return

    if fig is None and ax is None: fig = plt.figure(1)
    if ax is None: ax = fig.add_subplot(111)
    assert weights.ndim == 2, 'weight matrix has to be 2-dimensional'
    assert weights.shape[0] == weights.shape[1], 'weight matrix has to be square'
    dim = weights.shape[0]
    if labels is None: labels = np.arange(dim)
    
    G=nx.Graph()
    G.add_nodes_from(labels)

    for i in xrange(dim):
        for j in xrange(i+1,dim):
            G.add_edge(labels[i],labels[j],weight=weights[i,j])

    if pos is None: pos = circular_layout(labels, start_angle=start_angle, stop_angle=stop_angle, endpoint=endpoint)

    draw_args = dict(edge_cmap=plt.cm.Reds, font_size=10, width=4)
    draw_args.update(kargs)
    colors = [e[2]['weight'] for e in G.edges(data=True)]
    nx.draw(G,pos,ax=ax,edge_color=colors,**draw_args)
    ax.axis('equal')


def plot_matrix(weights, labels=None, fig=None, ax=None, **kargs):
    """Plot weight matrix
    """
    if fig is None and ax is None: fig = plt.figure(1)
    if ax is None: ax = fig.add_subplot(111)
    assert weights.ndim == 2, 'weight matrix has to be 2-dimensional'
    assert weights.shape[0] == weights.shape[1], 'weight matrix has to be square'
    dim = weights.shape[0]
    if labels is None: labels = np.arange(dim)

    vmax = kargs.get('vmax',weights.max())
    plot_args = dict(interpolation='nearest', vmin=0, vmax=vmax, cmap=plt.cm.Reds, origin='lower')
    plot_args.update(**kargs)
    ax.imshow(weights,**plot_args)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xticks(xrange(dim))
    ax.set_xticklabels(labels)
    ax.set_yticks(xrange(dim))
    ax.set_yticklabels(labels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
