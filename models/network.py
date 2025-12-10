import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from config import *
import torch.fft


class VanillaVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=LATENT_DIM, hidden_dims: List[int]=None, img_size=IMG_SIZE):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        # --- Encoder ---
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Determina dinamicamente o tamanho do flatten
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            enc_out = self.encoder(dummy)
            self._enc_shape = enc_out.shape[1:]  # (C, H, W)
            self._enc_out_dim = enc_out.numel()

        #self.fc_enc_1 = nn.Linear(self._enc_out_dim, 512)
        #self.fc_enc_2 = nn.Linear(512, 256)
        self.fc_enc = nn.Linear(self._enc_out_dim, 256)

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        # --- Decoder ---

        #self.fc_dec1 = nn.Linear(latent_dim, 256)
        #self.fc_dec2 = nn.Linear(256, 512)
        self.fc_dec = nn.Linear(latent_dim, 256)
        self.decoder_input = nn.Linear(256, self._enc_out_dim)

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    #nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               1,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Tanh()
            #nn.Sigmoid()
        )

    # --- Métodos ---
    def encode(self, x):
        enc_out = self.encoder(x)
        enc_flat = torch.flatten(enc_out, start_dim=1)
        f = F.leaky_relu(self.fc_enc(enc_flat))
        #f1 = F.leaky_relu(self.fc_enc_1(enc_flat))
        #f2 = F.leaky_relu(self.fc_enc_2(f1))
        mu = self.fc_mu(f)
        log_var = self.fc_var(f)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        #result = F.leaky_relu(self.fc_dec1(z))
        #result = F.leaky_relu(self.fc_dec2(result))
        result = F.leaky_relu(self.fc_dec(z))
        #result = F.leaky_relu(self.decoder_input(result))
        result = self.decoder_input(result)
        result = result.view(-1, *self._enc_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = F.interpolate(result, size=(self.img_size, self.img_size))
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, kl_weight = KL_WEIGHT,  M_N=1.0, epoch=None):
        recons_loss = F.mse_loss(recons, input)
        #recons_loss = F.binary_cross_entropy(recons, input, reduction='mean')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        #beta = min(1.0, epoch / 1000)
        lambda_rec = LAMBDA_REC
        loss = lambda_rec * recons_loss + kl_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}