from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PREDICTION_PATH = ROOT / 'result/reconstruction.raw'
GT_PATH = ROOT / 'dataset/high.raw'


def psnr(prediction, gt):
    mse = np.mean((prediction.astype(np.float64) - gt.astype(np.float64)) ** 2)
    data_range = float(gt.max()) - float(gt.min())
    return float(10 * np.log10(data_range ** 2 / mse))


def main():
    prediction = np.fromfile(PREDICTION_PATH, dtype='<f4')
    gt = np.fromfile(GT_PATH, dtype='<f4')
    print(f'PSNR = {psnr(prediction, gt):.4f} dB')


if __name__ == '__main__':
    main()
