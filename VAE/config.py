# File: config.py

CONFIG = {
    # Data Parameters
    "data_path": "/Users/tristanaretz/Desktop/Master_Bioinformatik/WiSe_24_25/cmsg/project/yeast-embeddings/data/protein_emb.parquet",

    # Model Architecture
    "latent_dim": 1024,  # Dimensionality of latent space
    "hidden_layers": [1024],  # Encoder/Decoder hidden layers
    "activation_function": "ReLU",  # Activation function: ReLU, LeakyReLU, etc.
    "dropout_rate": 0.2,  # Dropout rate for regularization
    "batch_norm": True,  # Whether to use batch normalization

    # Training Parameters
    "learning_rate": 0.002,
    "batch_size": 64,
    "num_epochs": 10,

    # Miscellaneous
    "seed": 42,  # For reproducibility
}