# File: config.py

CONFIG = {
    # Data Parameters
    "data_path": "/vol/storage/shared/tristan/yeast/yeast-embeddings/data/yeast_emb_full.parquet",
    "smf_path": "/vol/storage/shared/tristan/yeast/yeast-embeddings/data/smf.csv",

    # Model Architecture
    "latent_dim": 256,  # Dimensionality of latent space
    "hidden_layers":[2048,1024,512],  # Encoder/Decoder hidden layers
    "activation_function": "ELU",  # Activation function: ReLU, LeakyReLU, etc.
    "dropout_rate": 0.5,  # Dropout rate for regularization
    "batch_norm": False,  # Whether to use batch normalization

    # Training Parameters
    "learning_rate": 1e-5,
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "num_epochs": 500,

    # Miscellaneous
    "seed": 42,  # For reproducibility
}
