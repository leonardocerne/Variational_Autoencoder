import numpy as np
import os
import csv

def gerar_imagem_tensor(ks, kc, r, L=128, n=128):
    """Gera um tensor (n x n) com círculos nos 4 cantos e no centro."""
    x1 = np.arange(0, n)
    y1 = np.arange(0, n)
    X, Y = np.meshgrid(x1, y1)
    img = np.full((n, n), ks)

    # círculos nos cantos
    corners = [(0,0), (0,L-1), (L-1,0), (L-1,L-1)]
    for cx, cy in corners:
        mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        img[mask] = kc

    # círculo central
    center = (L/2, L/2)
    mask_c = (X - center[0])**2 + (Y - center[1])**2 <= r**2
    img[mask_c] = kc

    return img


def gerar_dataset_binario_unico(qtd=1000, n=1024, pasta_saida="dataset_autoencoder_bin",
                                arquivo_saida="dataset.npz", contraste_min=4, r_min_abs=3):
    """Gera dataset completo com contraste mínimo e raio mínimo."""
    os.makedirs(pasta_saida, exist_ok=True)

    csv_path = os.path.join(pasta_saida, "parametros.csv")
    npz_path = os.path.join(pasta_saida, arquivo_saida)

    L = n
    r_max = L / (2 * np.sqrt(2))

    log_vals = np.log10(np.logspace(-2, 2, num=qtd*2))
    np.random.shuffle(log_vals)
    ks_vals = log_vals[:qtd]
    kc_vals = log_vals[qtd:]

    imagens = []
    parametros = []  # será salvo também dentro do npz

    for i in range(qtd):
        ks = ks_vals[i]
        kc = kc_vals[i]

        # raio mínimo garantido
        r = np.random.uniform(r_min_abs, r_max)

        nome = f"img_{i:04d}"

        img = gerar_imagem_tensor(ks, kc, r, L=L, n=n)
        imagens.append(img)

        parametros.append([ks, kc, r])

    imagens = np.array(imagens, dtype=np.float32)

   
    np.savez_compressed(npz_path,
                        imagens=imagens,
                        parametros=np.array(parametros, dtype=float))

    
    import csv
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["nome", "ks", "kc", "r", "L"])
        for ks, kc, r in parametros:
            writer.writerow([f"{ks:.6f}", f"{kc:.6f}", f"{r:.2f}"])

    print(f"{qtd} tensores salvos em '{npz_path}'")
    print(f"Parâmetros salvos em '{csv_path}'")
    print(f"Raio: mínimo = {r_min_abs}px, máximo = {r_max:.2f}px")
    print(f"Shape do tensor final: {imagens.shape}")


# Exemplo de uso
gerar_dataset_binario_unico(qtd=10000, n=128)

