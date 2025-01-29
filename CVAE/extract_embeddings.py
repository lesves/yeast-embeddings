import torch
import pandas as pd
import numpy as np
from model.cvae import CVAE
from config import CONFIG
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def extract_and_save_embeddings(model_path, data_path, smf_path, output_path, recon_output_path):
    """
    Extract embeddings from the trained CVAE model and save them. Handles conditioning during embedding extraction.

    Parameters:
        model_path (str): Path to the trained CVAE model file.
        data_path (str): Path to the input data file (e.g., embeddings).
        smf_path (str): Path to the SMF conditioning file.
        output_path (str): Path to save the extracted embeddings.
        recon_output_path (str): Path to save the reconstructed embeddings.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the primary dataset
    dnalm_df = pd.read_parquet(data_path)
    
    # Load the SMF conditioning file
    smf_filter = pd.read_csv(smf_path, index_col=0)

    merged_df = smf_filter.merge(dnalm_df, on='gene_id').dropna()
    embeddings = merged_df.iloc[:, 5:]
    conditioning_var = merged_df['smf_30']

    # Extract embeddings and conditioning variable
    embeddings = embeddings.values
    conditioning_var = conditioning_var.values.reshape(-1, 1)

    # Normalize data
    #scaler_embeddings = StandardScaler()
    #embeddings_scaled = scaler_embeddings.fit_transform(embeddings)

    scaler_conditioning = StandardScaler()
    conditioning_scaled = scaler_conditioning.fit_transform(conditioning_var)

    # Create DataLoader for the full dataset
    full_loader = DataLoader(
        TensorDataset(
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(conditioning_scaled, dtype=torch.float32)
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    # Initialize the CVAE model
    vae = CVAE(
        input_dim=embeddings.shape[1],
        latent_dim=CONFIG["latent_dim"],
        hidden_layers=CONFIG["hidden_layers"],
        activation_function=CONFIG["activation_function"],
        dropout_rate=CONFIG["dropout_rate"],
        batch_norm=CONFIG["batch_norm"],
        cond_dim=conditioning_var.shape[1],  # Dimension of conditioning variable
    ).to(device)

    # Load trained model weights
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()

    # Extract embeddings and reconstructed data
    embeddings_list = []
    reconstructed_data_list = []

    with torch.no_grad():
        for batch in full_loader:
            x, c = batch[0].to(device), batch[1].to(device)  # Input data and conditioning variable
            mu, _ = vae.encode(x, c)  # Use the mean (mu) as the embedding
            recon, _, _ = vae(x, c)  # Reconstructed output

            embeddings_list.append(mu.cpu().numpy())
            reconstructed_data_list.append(recon.cpu().numpy())

    # Concatenate all embeddings and reconstructed data
    embeddings = np.vstack(embeddings_list)
    reconstructed_data = np.vstack(reconstructed_data_list)

    # Save embeddings to a file
    print(f"Saving embeddings to {output_path}...")
    pd.DataFrame(embeddings, index=merged_df.index).to_parquet(output_path, index=True)
    print("Embeddings saved successfully!")

    # Save reconstructed data to a file
    print(f"Saving reconstructed embeddings to {recon_output_path}...")
    pd.DataFrame(reconstructed_data, index=merged_df.index).to_parquet(recon_output_path, index=True)
    print("Reconstructed embeddings saved successfully!")

    pd.DataFrame(merged_df, index=merged_df.index).to_parquet("smf30_filtered_emb.parquet", index=True)

if __name__ == "__main__":
    model_path = "cvae_model.pth"  # Path to the trained model
    data_path = CONFIG["data_path"]  # Path to the input data
    smf_path = CONFIG["smf_path"]  # Path to the SMF conditioning file
    output_path = "cvae_embeddings.parquet"  # Path to save the embeddings
    recon_output_path = "cvae_reconstructed.parquet"  # Path to save the reconstructed embeddings

    extract_and_save_embeddings(model_path, data_path, smf_path, output_path, recon_output_path)

