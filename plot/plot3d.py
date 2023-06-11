import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot3d(F, xl=11):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.mean(1).view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=cm.viridis)
    # print(Ys.mean(1).min(), Ys.mean(1).max())

    ax.set_zticks([-16, -8, 0, 8])
    ax.set_zlim(-20, 10)

    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax.view_init(25)
    plt.tight_layout()
    plt.savefig(f"3d-obj.png", dpi=1000)