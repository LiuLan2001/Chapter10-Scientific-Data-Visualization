import numpy as np
import torch
from torch import nn

from data_io import make_coords

#训练隐式神经表示
def train(model, loader, epochs, lr, device):
    #定义优化器、学习率调度器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    model.train()
    #训练循环
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for coords, values in loader:
            coords, values = coords.to(device), values.to(device)
            optimizer.zero_grad()
            loss = criterion(model(coords), values)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(coords)
        #更新学习率调度器
        scheduler.step()
        print(f'Epoch {epoch}/{epochs}, MSE = {total_loss / len(loader.dataset):.8f}')

#训练完成后，推理超分辨率数据
@torch.no_grad()
def reconstruct(model, shape, offset, scale, batch_size, device):
    model.eval()
    predictions = []
    for coords in make_coords(shape).split(batch_size):
        predictions.append(model(coords.to(device)).cpu().numpy())
    normalized = np.concatenate(predictions).reshape(shape[::-1])
    return ((normalized + 1) * scale / 2 + offset).astype(np.float32)
