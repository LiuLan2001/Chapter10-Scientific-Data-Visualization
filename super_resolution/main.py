from pathlib import Path

from data_io import load_training_data, save_volume

import torch
from torch.utils.data import DataLoader

from model import Siren
from train import train, reconstruct


ROOT = Path(__file__).resolve().parent
LR_PATH = ROOT / 'dataset/low.raw'
LR_SHAPE = (120, 180, 30)
OUTPUT_SHAPE = (480, 720, 120)  # 各方向的目标采样数为输入的4倍
EPOCHS = 200
BATCH_SIZE = 8192
INFERENCE_BATCH_SIZE = 65536
LEARNING_RATE = 1e-4
DEVICE = 'cuda'
OUT_DIR = ROOT / 'result'

#训练入口
def main():
    torch.manual_seed(0)
    dataset, offset, scale = load_training_data(LR_PATH, LR_SHAPE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = Siren().to(DEVICE)
    train(model, loader, EPOCHS, LEARNING_RATE, DEVICE)
    #训练完成后，保存模型
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT_DIR / 'model.pth'
    torch.save({
        'model': model.state_dict(),
        'offset': offset,
        'scale': scale,
        'lr_shape': LR_SHAPE,
    }, checkpoint)
    print(f'Saved: {checkpoint}')
    #推理超分辨率数据
    prediction = reconstruct(model, OUTPUT_SHAPE, offset, scale, INFERENCE_BATCH_SIZE, DEVICE)
    out_path = OUT_DIR / 'reconstruction.raw'
    save_volume(out_path, prediction)
    print(f'Saved: {out_path}, shape: {OUTPUT_SHAPE}')


if __name__ == '__main__':
    main()
