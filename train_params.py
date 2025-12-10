import torch
from torch.utils.data import DataLoader
from data.parametros_dataset import ParamDataset
from models.param_network import ParamToZ
from config import LATENT_DIM, DEVICE, BATCH_SIZE

npz_path = "dataset_autoencoder_bin/dataset.npz"
mu_path = "first_test_vae/latent_space/mu.pt"
logvar_path = "first_test_vae/latent_space/logvar.pt"

dataset = ParamDataset(npz_path, mu_path, logvar_path)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ParamToZ(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

EPOCHS = 60

for epoch in range(EPOCHS):
    total_loss = 0

    for params, mu_target, logvar_target in loader:
        params = params.to(DEVICE)
        mu_target = mu_target.to(DEVICE)
        logvar_target = logvar_target.to(DEVICE)

        mu_pred, logvar_pred = model(params)
        loss_mu = criterion(mu_pred, mu_target)
        loss_logvar = criterion(logvar_pred, logvar_target)
        loss = loss_mu + 0.1 * loss_logvar

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[{epoch+1}/{EPOCHS}] loss = {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "first_test_vae/model_results/param_to_z.pth")
