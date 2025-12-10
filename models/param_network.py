import torch
from torch import nn

class ParamToZ(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.camadas = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.to_mu = nn.Linear(128, latent_dim)
        self.to_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.camadas(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar
