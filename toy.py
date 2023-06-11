from tqdm import tqdm
import torch

from metrics import *
from problems import Toy

### Define the problem ###
F = Toy()

maps = {
    "sgd": mean_grad,
    "cagrad": cagrad,
    "mgd": mgd,
    "pcgrad": pcgrad,
}

### Start experiments ###

def run_all():
    all_traj = {}

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([-8.5, -5.]),
        torch.Tensor([9.,   9.]),
    ]

    for i, init in enumerate(inits):
        for m in tqdm(["sgd", "mgd", "pcgrad", "cagrad"]):
            all_traj[m] = None
            traj = []
            solver = maps[m]
            x = init.clone()
            x.requires_grad = True

            n_iter = 100000
            opt = torch.optim.Adam([x], lr=0.001)

            for it in range(n_iter):
                traj.append(x.detach().numpy().copy())

                f, grads = F(x, True)
                if m== "cagrad":
                    g = solver(grads, c=0.5)
                else:
                    g = solver(grads)
                opt.zero_grad()
                x.grad = g
                opt.step()

            all_traj[m] = torch.tensor(traj)
        torch.save(all_traj, f"toy{i}.pt")

if __name__ == "__main__":
    run_all()