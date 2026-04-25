# Variational Autoencoder para Simulações Físicas Parametrizáveis

Este repositório contém o desenvolvimento de um **Variational Autoencoder (VAE)** aplicado à modelagem de simulações físicas parametrizáveis, como parte de um projeto de Iniciação Científica.

## Descrição

O projeto tem como objetivo aprender representações latentes de dados provenientes de simulações físicas, permitindo a geração e reconstrução de amostras a partir de diferentes configurações de parâmetros.

A abordagem utiliza um VAE para capturar a estrutura subjacente dos dados, possibilitando explorar o espaço latente e gerar novas simulações de forma controlada.

## Funcionalidades

* Treinamento de modelo VAE
* Geração de dados sintéticos a partir de parâmetros
* Reconstrução de amostras
* Exploração e visualização do espaço latente
* Ajuste de hiperparâmetros (beta-VAE)

## Estrutura do Projeto

* `models/` – Definição da arquitetura do VAE
* `data/` – Dados utilizados para treinamento
* `results/` – Saídas gerais do modelo
* `model_results/` – Resultados específicos de experimentos
* `latent_space/` – Visualizações e análises do espaço latente

### Scripts principais

* `train_vae.py` – Treinamento do modelo
* `train_params.py` – Treinamento da rede auxiliar
* `beta_sweep.py` – Experimentos com diferentes valores de beta
* `generate_from_params.py` – Geração de amostras condicionadas para rede auxiliar
* `generating_data.py` – Preparação/geração de dados
* `latent_space.py` – Análise do espaço latente
* `visualize_recons.py` – Visualização de reconstruções
* `config.py` – Configurações gerais do projeto

## Tecnologias utilizadas

* Python
* Bibliotecas de Deep Learning (PyTorch)
* Bibliotecas para manipulação de dados (NumPy)
* Bibliotecas de visualização (Matplotlib)
* Biblioteca de simulações físicas (CHFEM)

## Como executar

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/variational-autoencoder.git
   ```

2. Acesse o diretório:

   ```bash
   cd variational-autoencoder
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute o treinamento:

   ```bash
   python train_vae.py
   ```

## Objetivo

Investigar o uso de modelos generativos, especialmente VAEs, na compressão e geração de dados de simulações físicas, permitindo explorar o espaço de parâmetros de forma eficiente.

## Observações

Este projeto possui caráter acadêmico e faz parte de uma Iniciação Científica, com foco em aprendizado de representações e modelagem generativa aplicada a problemas físicos.
