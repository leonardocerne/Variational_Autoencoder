import torch
import numpy as np
import matplotlib.pyplot as plt
from config import DEVICE, LATENT_DIM, HIDDEN_DIMS, IMG_SIZE
from models.network import VanillaVAE
from models.param_network import ParamToZ
from generating_data import gerar_imagem_tensor


KS_MIN, KS_MAX = -2.0, 2.0
KC_MIN, KC_MAX = -2.0, 2.0
R_MIN,  R_MAX  =  3.0, 45.25

def normalizar_params(ks, kc, r):
    ks_n = (ks - KS_MIN) / (KS_MAX - KS_MIN)
    kc_n = (kc - KC_MIN) / (KC_MAX - KC_MIN)
    r_n  = (r  - R_MIN ) / (R_MAX  - R_MIN )
    return torch.tensor([[ks_n, kc_n, r_n]], dtype=torch.float32, device=DEVICE)


def carregar_modelos(vae_path, param2z_path):
    vae = VanillaVAE(
        in_channels=1,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        img_size=IMG_SIZE
    ).to(DEVICE)

    vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
    vae.eval()

    param2z = ParamToZ(latent_dim=LATENT_DIM).to(DEVICE)
    param2z.load_state_dict(torch.load(param2z_path, map_location=DEVICE))
    param2z.eval()

    return vae, param2z


def params_to_image(ks, kc, r, vae, param2z, debug=False):

    params = normalizar_params(ks, kc, r)

    with torch.no_grad():
        mu_pred, logvar_pred = param2z(params)
        std = torch.exp(0.5 * logvar_pred)
        eps = torch.randn_like(std)
        z = mu_pred + eps * std
        recon = vae.decode(z)

    if debug:
        z_np = z.detach().cpu().numpy()[0]
        print(f"z_pred mean={z_np.mean():.4e}, std={z_np.std():.4e}, "
              f"min={z_np.min():.4e}, max={z_np.max():.4e}")

    img = recon.squeeze().detach().cpu().numpy()
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)

    return img


if __name__ == "__main__":
    vae_path = "first_test_vae/model_results/vae_3_params.pth"
    param2z_path = "first_test_vae/model_results/param_to_z.pth"

    print("Carregando modelos…")
    vae, param2z = carregar_modelos(vae_path, param2z_path)
    print("Modelos carregados.")

    ks = -1.5
    kc = 0.5
    r  = 25

    print(f"Gerando imagem para ks={ks}, kc={kc}, r={r}…")
    img = params_to_image(ks, kc, r, vae, param2z, debug=True)
    img2 = gerar_imagem_tensor(ks, kc, r)
    img2 = (img2 + 2.0) / 4.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img2, cmap="coolwarm", vmin=0, vmax=1)
    axes[0].set_title(f"Tensor — ks={ks}, kc={kc}, r={r}")
    axes[0].axis("off")
    axes[1].imshow(img, cmap="coolwarm", vmin=0, vmax=1)
    axes[1].set_title("Params -> Latent Space -> Decoder")
    axes[1].axis("off")
    plt.tight_layout()
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0,1)),
        ax=axes.ravel().tolist()
    )
    plt.show()
    fig.savefig("first_test_vae/results/realxparam_network2.png", dpi=300, bbox_inches="tight")
    print("Plot salvo como comparacao.png")