import numpy as np,torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Union,List
def plot3d_surface_on_axis(ax:Axes3D,data:Union[torch.Tensor,np.ndarray],labels:List[str]=None):
    """
    labels: [label1,labels] 按照顺序排列
    """
    if isinstance(data,torch.Tensor):
        data = data.detach().cpu().numpy()
    assert data.ndim == 2, "must"
    x = np.arange(data.shape[1]) # x 是低的
    y = np.arange(data.shape[0]) # y 是高的
    x,y = np.meshgrid(x,y) # x会重复的更快, y会重复的更慢
    z = data
    ax.plot_surface(x,y,z,cmap="viridis")
    
    if labels is not None:
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[0])
    if not ax.get_xlabel():
        ax.set_xlabel('X')
        ax.set_ylabel('Y')


def plot3d_on_axis(ax:Axes3D,data:Union[torch.Tensor,np.ndarray],labels:List[str]=None):
    """
    labels: [label1,labels] 按照顺序排列
    """
    if isinstance(data,torch.Tensor):
        data = data.detach().cpu().numpy()
    assert data.ndim == 2, "must"
    # 创建 x, y, z 轴的数据
    m,n = data.shape
    x, y = np.meshgrid(np.arange(n), np.arange(m))
    x = x.ravel()  # 转成一维，bar3d 需要 1D 坐标
    y = y.ravel()
    z = np.zeros_like(x)  # 柱子起始高度为0
    dx = dy = 0.8  # 每个柱子的宽度
    dz = data.ravel()  # 将高度值展平为一维
    # 使用颜色映射创建颜色数组
    norm = plt.Normalize(dz.min(), dz.max())
    colors = cm.viridis(norm(dz))
    # 绘制3D柱状图
    # ax.bar3d(x, y, z, dx, dy, dz, color=colors, edgecolor="black") 
    ax.bar3d(x, y, z, dx, dy, dz, color=colors) 
    if labels is not None:
        ax.set_ylabel(labels[0])
        ax.set_xlabel(labels[1])
    if not ax.get_xlabel():
        ax.set_xlabel('X')
        ax.set_ylabel('Y')