import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

def plot_contour(F, task=1, traj=None, xl=11, plotbar=False, name="tmp"): 
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)

    cmap = cm.get_cmap('viridis')

    yy = -8.3552
    if task == 0:
        Yv = Ys.mean(1)
        plt.plot(-8.5, 7.5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot(-8.5, -5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot( 9, 9, marker='o', markersize=10, zorder=5, color='k')
        plt.plot([-7, 7], [yy, yy], linewidth=8.0, zorder=0, color='gray')
        plt.plot(0, yy, marker='*', markersize=15, zorder=5, color='k')
    elif task == 1:
        Yv = Ys[:,0]
        plt.plot(7, yy, marker='*', markersize=15, zorder=5, color='k')
    else:
        Yv = Ys[:,1]
        plt.plot(-7, yy, marker='*', markersize=15, zorder=5, color='k')

    c = plt.contour(X, Y, Yv.view(n,n), cmap=cm.viridis, linewidths=4.0)

    if traj is not None:
        for tt in traj:
            l = tt.shape[0]
            color_list = np.zeros((l,3))
            color_list[:,0] = 1.
            color_list[:,1] = np.linspace(0, 1, l)
            #color_list[:,2] = 1-np.linspace(0, 1, l)
            ax.scatter(tt[:,0], tt[:,1], color=color_list, s=6, zorder=10)

    if plotbar:
        cbar = fig.colorbar(c, ticks=[-15, -10, -5, 0, 5])
        cbar.ax.tick_params(labelsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.yticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=100)
    plt.close()