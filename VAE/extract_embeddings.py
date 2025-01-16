# Extract and save embeddings from the trained VAE model.

import torch
import pandas as pd
import numpy as np
from model.vae import VAE
from config import CONFIG
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def extract_and_save_embeddings(model_path, data_path, output_path, recon_output_path):
    """
    Extract embeddings from the trained VAE model and save them. Also, diagnose the quality of embeddings.

    Parameters:
        model_path (str): Path to the trained VAE model file.
        data_path (str): Path to the input data file.
        output_path (str): Path to save the extracted embeddings.
        recon_output_path (str): Path to save the reconstructed embeddings.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load full dataset
    print("Loading full dataset...")
    data = pd.read_parquet(data_path)
    data = data.set_index('gene_id').values
   
   # Normalize the data
   # scaler = StandardScaler()
    #data = scaler.fit_transform(data)

    # Create DataLoader for the full dataset
    full_loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=CONFIG["batch_size"], shuffle=False)

    # Initialize the model
    vae = VAE(
        input_dim=data.shape[1],
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
    reconstructed_data = []
    with torch.no_grad():
        for batch in full_loader:
            x = batch[0].to(device)
            mu, _ = vae.encode(x)  # Use the mean (mu) as the embedding
            recon, _, _ = vae(x)  # Reconstructed output

            embeddings.append(mu.cpu().numpy())
            reconstructed_data.append(recon.cpu().numpy())

    # Concatenate all embeddings and reconstructed data
    embeddings = np.vstack(embeddings)
    reconstructed_data = np.vstack(reconstructed_data)

    # Save embeddings to a file
    print(f"Saving embeddings to {output_path}...")
    pd.DataFrame(embeddings).to_parquet(output_path, index=False)
    print("Embeddings saved successfully!")

    # Save reconstructed data to a file
    print(f"Saving reconstructed embeddings to {recon_output_path}...")
    pd.DataFrame(reconstructed_data).to_parquet(recon_output_path, index=False)
    print("Reconstructed embeddings saved successfully!")

    # Diagnose the embeddings
    print("Diagnosing embeddings...")

    # Compute reconstruction error for a few samples
    sample_indices = np.random.choice(len(data), size=5, replace=False)
    for idx in sample_indices:
        original = data[idx]
        reconstructed = reconstructed_data[idx]

        # Print sample comparison
        print(f"Sample {idx}:")
        print("Original (first 5 features):", original[:5])
        print("Reconstructed (first 5 features):", reconstructed[:5])
        print("Reconstruction Error (MSE):", np.mean((original - reconstructed) ** 2))
        print()

    # Analyze latent space
    print("Analyzing latent space...")
    print("Latent space mean (first 5 embeddings):")
    print(embeddings[:5])
    print("Latent space variance (first 5 dimensions):", np.var(embeddings, axis=0)[:5])

if __name__ == "__main__":
    model_path = "vae_model.pth"  # Path to the trained model
    data_path = CONFIG["data_path"]  # Path to the input data
    output_path = "vae_embeddings.parquet"  # Path to save the embeddings
    recon_output_path = "vae_reconstructed.parquet"  # Path to save the reconstructed embeddings

    extract_and_save_embeddings(model_path, data_path, output_path, recon_output_path)

