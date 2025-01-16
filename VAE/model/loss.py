import torch
import torch.nn.functional as F

# Define the loss function for the VAE.

def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    # Reconstruction loss (MSE or BCE)
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    # KL Divergence loss
    kld_loss = -0.5 * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld_loss
