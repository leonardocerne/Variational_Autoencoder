import torch
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from models.network import VanillaVAE
from data.circulos_dataset import CirculosDataset
from config import IMG_SIZE, LATENT_DIM, HIDDEN_DIMS, DATASET_PATH, DEVICE, BATCH_SIZE

# -----------------------------
# Dataset e DataLoader
# -----------------------------
dataset = CirculosDataset(DATASET_PATH, normalize=True)  # normalização já aplicada
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Carrega o modelo treinado
# -----------------------------
vae = VanillaVAE(
    in_channels=1,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    img_size=IMG_SIZE
)
vae.load_state_dict(torch.load("first_test_vae/model_results/vae_3_params.pth", map_location=DEVICE))
vae.to(DEVICE)
vae.eval()

# -----------------------------
# Pega um batch aleatório
# -----------------------------
batch = next(iter(dataloader)).to(DEVICE)

with torch.no_grad():
    recons, _, _, _ = vae(batch)

# -----------------------------
# TESTE: inspeciona formato e valores da saída
# -----------------------------
print("Batch de entrada:", batch.shape, "min =", batch.min().item(), "max =", batch.max().item())
print("Saída do VAE:", recons.shape, "min =", recons.min().item(), "max =", recons.max().item())

# -----------------------------
# Ajusta ranges
# Entradas [-2,2] → [-1,1] para plot
# Saídas do VAE assumidas em [-1,1] devido a Tanh, convertemos para [0,1]
# -----------------------------
orig = batch.cpu().numpy()
recons = recons.cpu().numpy()
orig = (orig - orig.min()) / (orig.max() - orig.min())
recons = (recons - recons.min()) / (recons.max() - recons.min())

# -----------------------------
# Seleciona n imagens aleatórias do batch
# -----------------------------
n = min(5, batch.size(0))
indices = random.sample(range(batch.size(0)), n)
orig_subset = orig[indices]
recons_subset = recons[indices]

# -----------------------------
# Plota as imagens
# -----------------------------
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original
    plt.subplot(2, n, i + 1)
    plt.imshow(orig_subset[i, 0], cmap='coolwarm', vmin=0, vmax=1)
    plt.axis('off')
    if i == 0:
        plt.ylabel("Original")

    # Reconstruída
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(recons_subset[i, 0], cmap='coolwarm', vmin=0, vmax=1)
    plt.axis('off')
    if i == 0:
        plt.ylabel("Reconstruída")

plt.tight_layout()
plt.show()
