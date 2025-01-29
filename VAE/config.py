# File: config.py

CONFIG = {
    # Data Parameters
    "data_path": "/vol/storage/shared/tristan/yeast/yeast-embeddings/data/yeast_emb_full.parquet",

    # Model Architecture
    "latent_dim": 256,  # Dimensionality of latent space
    "hidden_layers":[2048,1024,512],  # Encoder/Decoder hidden layers
    "activation_function": "SELU",  # Activation function: ReLU, LeakyReLU, etc.
    "dropout_rate": 0.4,  # Dropout rate for regularization
    "batch_norm": False,  # Whether to use batch normalization

    # Training Parameters
    "learning_rate": 1e-3,
    "momentum": 0.9,
    "weight_decay": 1e-3,
    "batch_size": 512,
    "num_epochs": 200,

    # Miscellaneous
    "seed": 42,  # For reproducibility
}
