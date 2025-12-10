# config.py
IMG_SIZE = 128     # tamanho da imagem (LxL)
LATENT_DIM = 3  # dimensão do espaço latente
HIDDEN_DIMS = [8, 16, 32, 64]  # canais de cada camada
BATCH_SIZE = 128
DATASET_PATH = "dataset_autoencoder_bin/dataset.npz"
DEVICE = "cpu"  # ou "cpu"
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
LAMBDA_REC = 1
KL_WEIGHT = 0.005