import torch,torch.nn as nn,torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot2d_percent_on_axis(ax,data:torch.Tensor): # data[0]是n,1是c
    value = data
    # if "weight" in k:
    #     value = v.T # oc,ic -> ic,oc 因为想看一个通道内的分布是不是好量化的
    # else:
    #     value = v.flatten(0,-2) # (l,c)
    value = value.float().cuda()
    pmax = torch.amax(value,dim=0).cpu().float().numpy()
    p9999 = torch.quantile(value,0.9999,dim=0).cpu().numpy()
    p99 = torch.quantile(value,0.99,dim=0).cpu().numpy()
    p75 = torch.quantile(value,0.75,dim=0).cpu().numpy()
    p25 = torch.quantile(value,0.25,dim=0).cpu().numpy()
    p01 = torch.quantile(value,0.01,dim=0).cpu().numpy()
    p0001 = torch.quantile(value,0.0001,dim=0).cpu().numpy()
    pmin = torch.amin(value,dim=0).cpu().numpy()
    x_label_ids = np.arange(len(pmin))
    ax.plot(x_label_ids,p75,color='orange',label='25/75 Percentile',linewidth=0.3)
    ax.plot(x_label_ids,p25,color='orange',linewidth=0.3)
    ax.plot(x_label_ids,p99,color='purple',label='1/99 Percentile',linewidth=0.3)
    ax.plot(x_label_ids,p01,color='purple',linewidth=0.3)
    ax.plot(x_label_ids,p9999,color='red',label='1/9999 Percentile',linewidth=0.3)
    ax.plot(x_label_ids,p0001,color='red',linewidth=0.3)
    ax.plot(x_label_ids,pmin,color='blue',label='Min/Max',linewidth=0.3)
    ax.plot(x_label_ids,pmax,color='blue',linewidth=0.3)
    # ax.set_title(k)
    if not ax.get_xlabel():
        ax.set_xlabel("hidden dimension index")
        ax.set_ylabel("value")
    ax.legend(loc="upper right")