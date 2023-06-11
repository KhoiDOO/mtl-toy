import matplotlib.pyplot as plt
import numpy as np

from .smooth import smooth

def plot_loss(F, trajs, name="tmp"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormaps = {
        "sgd": "tab:blue",
        "pcgrad": "tab:orange",
        "mgd": "tab:purple",
        "cagrad": "tab:red",
    }
    maps = {
        "sgd" : "Adam",
        "pcgrad" : "PCGrad",
        "mgd" : "MGDA",
        "cagrad" : "RGD (ours)",
    }
    for method in ["sgd", "mgd", "pcgrad", "cagrad"]:
        traj = trajs[method][::100]
        Ys = F.batch_forward(traj)
        x = np.arange(traj.shape[0])
        #y = torch.cummin(Ys.mean(1), 0)[0]
        y = Ys.mean(1)

        ax.plot(x, smooth(list(y)),
                color=colormaps[method],
                linestyle='-',
                label=maps[method], linewidth=4.)

    plt.xticks([0, 200, 400, 600, 800, 1000],
               ["0", "20K", "40K", "60K", "80K", "100K"],
               fontsize=15)

    plt.yticks(fontsize=15)
    ax.grid()
    plt.legend(fontsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=100)
    plt.close()