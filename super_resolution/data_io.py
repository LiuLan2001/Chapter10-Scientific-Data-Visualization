import numpy as np
import torch
from torch.utils.data import TensorDataset
#读取数据
def read_volume(path, shape):
    return np.fromfile(path, dtype='<f4').reshape(shape[::-1])

def save_volume(path, volume):
    np.asarray(volume, dtype='<f4').tofile(path)

# 在 [-1, 1) 内生成坐标，使整数倍加密后的网格保留原采样位置。
def make_coords(shape):
    axes = [2 * torch.arange(n, dtype=torch.float32) / n - 1 for n in shape]
    z, y, x = torch.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    return torch.stack((x, y, z), dim=-1).reshape(-1, 3)

#配置训练数据
def load_training_data(path, shape):
    #读入训练数据
    volume = read_volume(path, shape)
    #归一化数据到 [-1, 1] 范围
    offset = float(volume.min())
    scale = float(volume.max()) - offset
    values = torch.from_numpy(2 * (volume - offset) / scale - 1).reshape(-1, 1)
    #创建一个 TensorDataset，其中包含坐标和归一化后的体素值
    dataset = TensorDataset(make_coords(shape), values)
    return dataset, offset, scale
