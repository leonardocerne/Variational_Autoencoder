import torch
from torch.utils.data import Dataset
import numpy as np

class ParamDataset(Dataset):
    def __init__(self, npz_path, mu_path, logvar_path):
        data = np.load(npz_path)
        self.params = data["parametros"]
        self.mu = torch.load(mu_path).float()
        self.logvar = torch.load(logvar_path).float()
        self.params = torch.tensor(self.params, dtype=torch.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        ks, kc, r = self.params[idx]
        ks_n = (ks + 2) / 4
        kc_n = (kc + 2) / 4
        r_n  = (r - 3.0) / (45.25 - 3.0)

        params_norm = torch.tensor([ks_n, kc_n, r_n], dtype=torch.float32)

        return params_norm, self.mu[idx], self.logvar[idx]