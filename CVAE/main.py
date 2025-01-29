import torch
import torch.optim as optim
from model.cvae import CVAE  # Import the updated model
from model.loss import vae_loss
from data.data_loader import load_data
from config import CONFIG
import pandas as pd

# Set random seed
torch.manual_seed(CONFIG["seed"])

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and split data
train_loader, val_loader, test_loader, input_dim, cond_dim = load_data(
    file_path=CONFIG["data_path"],
    smf_path=CONFIG["smf_path"],
    batch_size=CONFIG["batch_size"],
    conditioning=True
)

# Initialize the model
vae = CVAE(
    input_dim=input_dim,
    latent_dim=CONFIG["latent_dim"],
    hidden_layers=CONFIG["hidden_layers"],
    activation_function=CONFIG["activation_function"],
    dropout_rate=CONFIG["dropout_rate"],
    batch_norm=CONFIG["batch_norm"],
    cond_dim=cond_dim,  # Conditioning dimension
).to(device)

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

vae.apply(weights_init)

# Get the dataset from the DataLoader
train_dataset = train_loader.dataset

# Access the tensors inside the dataset
train_embed_tensor, train_cond_tensor = train_dataset.tensors

# Print the shape of the entire dataset
print("Train embeddings shape:", train_embed_tensor.shape)
print("Train conditioning variable shape:", train_cond_tensor.shape)

# Training loop
vae.train()
for epoch in range(CONFIG["num_epochs"]):
    epoch_loss = 0
    beta = min(1, epoch / (CONFIG["num_epochs"] // 2))  # Gradually increase beta

    for batch in train_loader:
        x, c = batch[0].to(device), batch[1].to(device)  # Input data and conditioning variable
        optimizer.zero_grad()

        recon, mu, logvar = vae(x, c)  # Pass both x and c
        loss = vae_loss(recon, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validate the model
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, c = batch[0].to(device), batch[1].to(device)
            recon, mu, logvar = vae(x, c)
            val_loss += vae_loss(recon, x, mu, logvar).item()

    print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Train Loss: {epoch_loss / len(train_loader.dataset):.4f}, Val Loss: {val_loss / len(val_loader.dataset):.4f}")

# Save the trained model
torch.save(vae.state_dict(), "cvae_model.pth")
