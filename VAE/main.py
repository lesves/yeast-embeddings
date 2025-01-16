# Train the VAE.

import torch
import torch.optim as optim
from model.vae import VAE
from model.loss import vae_loss
from data.data_loader import load_data
from config import CONFIG

# Set random seed
torch.manual_seed(CONFIG["seed"])

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and split data
train_loader, val_loader, test_loader, input_dim = load_data(CONFIG["data_path"], CONFIG["batch_size"])

# Initialize the model
vae = VAE(
    input_dim=input_dim,
    latent_dim=CONFIG["latent_dim"],
    hidden_layers=CONFIG["hidden_layers"],
    activation_function=CONFIG["activation_function"],
    dropout_rate=CONFIG["dropout_rate"],
    batch_norm=CONFIG["batch_norm"],
).to(device)

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=CONFIG["learning_rate"])

# Training loop
vae.train()
for epoch in range(CONFIG["num_epochs"]):
    epoch_loss = 0
    for batch in train_loader:
        x = batch[0].to(device)  # Input data
        optimizer.zero_grad()

        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validate the model
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            recon, mu, logvar = vae(x)
            val_loss += vae_loss(recon, x, mu, logvar).item()

    print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Train Loss: {epoch_loss / len(train_loader.dataset):.4f}, Val Loss: {val_loss / len(val_loader.dataset):.4f}")

# Save the trained model
torch.save(vae.state_dict(), "vae_model.pth")
