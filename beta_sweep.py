import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.network import VanillaVAE
from data.circulos_dataset import CirculosDataset
from config import *
import matplotlib.pyplot as plt
import pandas as pd
import time
import gc
import os

# Lista de betas [0.0001, 0.0002, ... , 0.1]
#b1 = np.arange(0.0001, 0.0011, 0.0001)
#b2 = np.arange(0.001, 0.0101, 0.001)
#b3 = np.arange(0.01, 0.1001, 0.02)
#betas = sorted(set(np.concatenate([b1, b2, b3])))
#betas = [round(float(b), 5) for b in betas]
#betas = list(dict.fromkeys(betas))
#betas.append(0.1)

betas = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.05, 0.1]
dataset = CirculosDataset("dataset_autoencoder_bin/dataset.npz", normalize = True)

train_size = int(0.8 * (len(dataset)))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

epochs = NUM_EPOCHS

results = []

save_path = "first_test_vae/results/beta_results_2.csv"

if os.path.exists(save_path):
    df = pd.read_csv(save_path)
    done_betas = set(df[df["beta"] != "CONFIG"]["beta"].astype(float))
    print(f"Retomando experimento. Betas já concluídos: {done_betas}")
else:
    df = pd.DataFrame()
    done_betas = set()
    config_row = pd.DataFrame([{
        "beta": "CONFIG",
        "train_loss": f"IMG_SIZE={IMG_SIZE}",
        "val_loss": f"LATENT_DIM={LATENT_DIM}",
        "recon_loss": f"HIDDEN_DIMS={HIDDEN_DIMS}",
        "kl_loss": f"BATCH_SIZE={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}, LR={LEARNING_RATE}, DEVICE={DEVICE}"
    }])
    df = pd.concat([config_row], ignore_index=True)
    df.to_csv(save_path, index=False)
    print("Novo arquivo de resultados criado.")

for beta in betas:
    if beta in done_betas:
        print(f"Pulando beta={beta} (já treinado)")
        continue
    print(f"Iniciando treinamento com beta:{beta}\n")
    vae = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM, img_size=IMG_SIZE).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    # Loop de treinamento
    for epoch in range(epochs):
        vae.train()
        epoch_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        for batch in train_loader:
            batch = batch.to(device)
            recons, input, mu, log_var = vae(batch)
            losses = vae.loss_function(recons, input, mu, log_var, kl_weight = beta)
            loss = losses['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += losses['loss'].item()
            total_recon += losses['Reconstruction_Loss'].item()
            total_kl += losses['KLD'].item()
        train_loss = epoch_loss / len(train_loader)
        recon_loss = total_recon / len(train_loader)
        kl_loss = total_kl / len(train_loader)
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in test_loader:
                val_batch = val_batch.to(device)
                recons, input, mu, log_var = vae(val_batch)
                losses = vae.loss_function(recons, input, mu, log_var, kl_weight=beta)
                val_loss += losses['loss'].item()
        val_loss /= len(test_loader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Recon: {recon_loss:.4f} | KL: {kl_loss:.4f}")
    
    # --- Salva resultado no CSV imediatamente ---
    new_row = pd.DataFrame([{
        "beta": round(beta,5),
        "train_loss": round(train_loss,4),
        "val_loss": round(val_loss,4),
        "recon_loss": round(recon_loss,4),
        "kl_loss": round(kl_loss,4)
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(save_path, index=False)
    print(f"Resultados de beta={beta} salvos.")

    betas_para_salvar = [0.0001, 0.001, 0.01, 0.1]
    if round(beta, 5) in betas_para_salvar:
        idx = betas_para_salvar.index(round(beta, 5))
        model_path = f"first_test_vae/model_results/beta{idx}.pth"
        torch.save(vae.state_dict(), model_path)
        print(f"Modelo salvo em {model_path}")

    # --- Limpeza de memória ---
    del vae, optimizer, losses, recons, input, mu, log_var
    gc.collect()
    torch.cuda.empty_cache()


print("\nTodos os betas concluídos e salvos com sucesso.")