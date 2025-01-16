# Test the VAE with new data.

import torch
from model.vae import VAE
from data.data_loader import load_data
from model.loss import vae_loss
from config import CONFIG

def test_vae(test_loader):
    # Load model
    vae = VAE(
        input_dim=test_loader.dataset.tensors[0].shape[1],
        latent_dim=CONFIG["latent_dim"],
        hidden_layers=CONFIG["hidden_layers"],
        activation_function=CONFIG["activation_function"],
        dropout_rate=CONFIG["dropout_rate"],
        batch_norm=CONFIG["batch_norm"],
    )

    # Load trained weights
    vae.load_state_dict(torch.load("vae_model.pth", weights_only=True))
    vae.eval()

    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            recon, mu, logvar = vae(x)
            test_loss += vae_loss(recon, x, mu, logvar).item()

    print(f"Test Loss: {test_loss / len(test_loader.dataset):.4f}")

# Load data and get test loader
_, _, test_loader, _ = load_data(CONFIG["data_path"], CONFIG["batch_size"])

# Call the test function
test_vae(test_loader)

