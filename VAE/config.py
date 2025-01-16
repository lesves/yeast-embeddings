# File: config.py

CONFIG = {
    # Data Parameters
    "data_path": "/vol/storage/shared/tristan/yeast/yeast-embeddings/data/yeast_emb_full.parquet",

    # Model Architecture
    "latent_dim": 512,  # Dimensionality of latent space
    "hidden_layers":[2048,1024],  # Encoder/Decoder hidden layers
    "activation_function": "LeakyReLU",  # Activation function: ReLU, LeakyReLU, etc.
    "dropout_rate": 0.2,  # Dropout rate for regularization
    "batch_norm": True,  # Whether to use batch normalization

    # Training Parameters
    "learning_rate": 0.0001,
    "batch_size": 64,
    "num_epochs": 100,

    # Miscellaneous
    "seed": 42,  # For reproducibility
}
