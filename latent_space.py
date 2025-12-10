import torch
from torch.utils.data import DataLoader
from config import *
from data.circulos_dataset import CirculosDataset
from models.network import VanillaVAE

# Salvando espaço latente
dataset = CirculosDataset(DATASET_PATH, normalize=True)  # normalização já aplicada
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

vae = VanillaVAE(
    in_channels=1,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    img_size=IMG_SIZE
)
vae.load_state_dict(torch.load("first_test_vae/model_results/vae_3_params.pth", map_location=DEVICE))
vae.to(DEVICE)
vae.eval()

all_mu = []
all_logvar = []
all_z = []

with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=64, shuffle=False):
        batch = batch.to(DEVICE)

        mu, log_var = vae.encode(batch)
        z = vae.reparameterize(mu, log_var)

        all_mu.append(mu.cpu())
        all_logvar.append(log_var.cpu())
        all_z.append(z.cpu())

all_mu = torch.cat(all_mu)
all_logvar = torch.cat(all_logvar)
all_z = torch.cat(all_z)

print("Tamanhos:")
print(all_mu.shape, all_logvar.shape, all_z.shape)

torch.save(all_z, "first_test_vae/latent_space/z.pt")
torch.save(all_mu, "first_test_vae/latent_space/mu.pt")
torch.save(all_logvar, "first_test_vae/latent_space/logvar.pt")