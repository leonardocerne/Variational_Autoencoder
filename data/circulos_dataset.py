import torch
from torch.utils.data import Dataset
import numpy as np

class CirculosDataset(Dataset):
    def __init__(self, npz_file, normalize=True):
        """
        npz_file: caminho do arquivo .npz que contém 'imagens'
        normalize: se True, divide por 2 para mapear [-2,2] -> [-1,1]
        """
        data = np.load(npz_file)
        self.imgs = data["imagens"].astype(np.float32)  # (N, H, W)
        self.normalize = normalize

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # (H, W)

        if self.normalize:
            img = img / 2.0 

        img = np.expand_dims(img, 0)  # vira (1, H, W)

        return torch.tensor(img, dtype=torch.float32)
