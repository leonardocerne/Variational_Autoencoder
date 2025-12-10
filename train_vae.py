import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.network import VanillaVAE
from data.circulos_dataset import CirculosDataset
from config import *
import matplotlib.pyplot as plt
import random
import numpy as np

def show_reconstructions(orig, recon, n=4):
    """
    orig, recon: tensores [B, 1, H, W]
    n: quantas imagens mostrar
    """
    orig = orig.cpu().detach()
    recon = recon.cpu().detach()
    
    batch_size = orig.shape[0]
    indices = random.sample(range(batch_size), min(n, batch_size))
    
    for i in indices:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow((orig[i,0]+1)/2, cmap='coolwarm', vmin=0.0, vmax=1.0)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow((recon[i,0]+1)/2, cmap='coolwarm', vmin=0.0, vmax=1.0)
        plt.title('Reconstrução')
        plt.axis('off')
        plt.show()

dataset = CirculosDataset("dataset_autoencoder_bin/dataset.npz", normalize=True)

train_size = int(0.8 * (len(dataset)))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Modelo
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
vae = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM, img_size=IMG_SIZE)
vae.to(device)

# Otimizador
optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
#optimizer = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
epochs = NUM_EPOCHS

losses_history = []
val_losses = []


# Loop de treino
for epoch in range(NUM_EPOCHS):
    vae.train()
    epoch_loss = 0.0
    total_recon = 0
    total_kl = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        recons, input, mu, log_var = vae(batch)
        losses = vae.loss_function(recons, input, mu, log_var, epoch=epoch)
        loss = losses['loss']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += loss.item()
        total_recon += losses['Reconstruction_Loss'].item()
        total_kl += losses['KLD'].item()

    avg_loss = epoch_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    losses_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f} - Recon Loss: {avg_recon:.4f} - KL Loss: {avg_kl:.4f}")
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in test_loader:
            val_batch = val_batch.to(device)
            recons, input, mu, log_var = vae(val_batch)
            losses = vae.loss_function(recons, input, mu, log_var, epoch=epoch)
            val_loss += losses['loss'].item()
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    print(f"Val Loss: {val_loss:.4f}")


sample_batch = next(iter(test_loader)).to(device)
recons, input, mu, log_var = vae(sample_batch)
show_reconstructions(sample_batch, recons, n=4)
# ===== Avaliação no conjunto de teste =====
vae.eval()
with torch.no_grad():
    test_batch = next(iter(test_loader)).to(device)
    recons, input, mu, log_var = vae(test_batch)
    test_losses = vae.loss_function(recons, input, mu, log_var)
    print(f"\nLoss no conjunto de teste: {test_losses['loss']:.4f}")

# Plot da curva de perdas
plt.figure(figsize=(8,5))
plt.plot(range(1, NUM_EPOCHS+1), losses_history, marker='o')
plt.title("Curva de Perdas do VAE")
plt.xlabel("Época")
plt.ylabel("Loss média")
plt.grid(True)
plt.show()

# Salva modelo
torch.save(vae.state_dict(), "first_test_vae/model_results/vae_3_params.pth")


