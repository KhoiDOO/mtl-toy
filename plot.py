import os

import torch
from torch import nn

from tqdm import tqdm
from problems import Toy

from plot import *

F = Toy()

def plot_results(img_dir:str):
    plot3d(F)
    plot_contour(F, 1, name="toy_task_1")
    plot_contour(F, 2, name="toy_task_2")
    t1 = torch.load(f"toy0.pt")
    t2 = torch.load(f"toy1.pt")
    t3 = torch.load(f"toy2.pt")

    length = t1["sgd"].shape[0]

    for method in ["sgd", "mgd", "pcgrad", "cagrad"]:
        
        save_dir = img_dir + f"/{method}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        ranges = list(range(10, length, 1000))
        ranges.append(length-1)
        for t in tqdm(ranges):
            plot_contour(F,
                         task=0, # task == 0 meeas plot for both tasks
                         traj=[t1[method][:t],t2[method][:t],t3[method][:t]],
                         plotbar=(method == "cagrad"),
                         name=f"./imgs/toy_{method}_{t}")

if __name__ == "__main__":
    
    img_dir = os.getcwd() + "/imgs"
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    
    plot_results(img_dir=img_dir)