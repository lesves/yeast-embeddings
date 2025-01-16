# Extract and save embeddings from the trained VAE model.

import torch
import pandas as pd
from model.vae import VAE
from data.data_loader import load_data
from config import CONFIG

def extract_and_save_embeddings(model_path, data_path, output_path):
    """
    Extract embeddings from the trained VAE model and save them.

    Parameters:
        model_path (str): Path to the trained VAE model file.
        data_path (str): Path to the input data file.
        output_path (str): Path to save the extracted embeddings.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    _, _, test_loader, input_dim = load_data(data_path, batch_size=CONFIG["batch_size"])

    # Initialize the model
    vae = VAE(
        input_dim=input_dim,
        latent_dim=CONFIG["latent_dim"],
        hidden_layers=CONFIG["hidden_layers"],
        activation_function=CONFIG["activation_function"],
        dropout_rate=CONFIG["dropout_rate"],
        batch_norm=CONFIG["batch_norm"],
    ).to(device)

    # Load trained model weights
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()

    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            mu, _ = vae.encode(x)  # Use the mean (mu) as the embedding
            embeddings.append(mu.cpu().numpy())

    # Concatenate all embeddings
    embeddings = torch.cat([torch.tensor(embed) for embed in embeddings], dim=0).numpy()

    # Save embeddings to a file
    print(f"Saving embeddings to {output_path}...")
    pd.DataFrame(embeddings).to_parquet(output_path, index=False)
    print("Embeddings saved successfully!")

if __name__ == "__main__":
    model_path = "vae_model.pth"  # Path to the trained model
    data_path = CONFIG["data_path"]  # Path to the input data
    output_path = "vae_embeddings.parquet"  # Path to save the embeddings

    extract_and_save_embeddings(model_path, data_path, output_path)
