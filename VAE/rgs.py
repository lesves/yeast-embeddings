import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import random
from model.vae import VAE
from model.loss import vae_loss
from config import CONFIG
from torch.utils.data import DataLoader, TensorDataset
from data.data_loader import load_data

def train_vae_and_extract_embeddings(params, data_path, model_save_path, embeddings_save_path):
    """
    Train a VAE with given parameters and extract latent embeddings for the whole dataset.

    Parameters:
        params (dict): Dictionary of hyperparameters.
        data_path (str): Path to the input data file.
        model_save_path (str): Path to save the trained model.
        embeddings_save_path (str): Path to save the extracted embeddings.

    Returns:
        None
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    optimizer = torch.optim.Adam(vae.parameters(), lr=CONFIG["learning_rate"])

    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    vae.apply(weights_init)

    # Training loop
    vae.train()
    for epoch in range(CONFIG["num_epochs"]):
        epoch_loss = 0
        beta = min(1, epoch / (CONFIG["num_epochs"] // 2))  # Gradually increase beta

        for batch in train_loader:
            x = batch[0].to(device)  # Input data
            optimizer.zero_grad()

            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        # Load and split dath {epoch + 1}/{params['num_epochs']}, Loss: {epoch_loss:.4f}")
                
    # Save the trained model
    torch.save(vae.state_dict(), model_save_path)

    # Extract latent embeddings for the whole dataset
    vae.eval()

    data = pd.read_parquet(data_path)
    data = data.set_index('gene_id').values

    full_loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=CONFIG["batch_size"], shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch in full_loader:
            x = batch[0].to(device)
            mu, _ = vae.encode(x)
            embeddings.append(mu.cpu().numpy())
    embeddings = np.vstack(embeddings)

    # Save embeddings
    pd.DataFrame(embeddings, index=pd.read_parquet(data_path).index).to_parquet(embeddings_save_path)

def calculate_r2(full_emb_path, vae_emb_path, smf_filter_path):
    """
    Calculate R2 score using latent embeddings.

    Parameters:
        full_emb_path (str): Path to the full embedding file.
        vae_emb_path (str): Path to the extracted VAE embeddings file.
        smf_filter_path (str): Path to the SMF filter file.

    Returns:
        float: R2 score
    """
    full_emb = pd.read_parquet(full_emb_path).set_index('gene_id')
    vae_emb = pd.read_parquet(vae_emb_path).set_index(full_emb.index)
    smf_filter = pd.read_csv(smf_filter_path, index_col=0)

    A = smf_filter.merge(vae_emb, on='gene_id').dropna()
    X = A.iloc[:, 5:]
    y = A['smf_30']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    lm = LinearRegression().fit(X_train, y_train)
    r2 = lm.score(X_test, y_test)
    return r2


def random_grid_search(num_trials, data_path, full_emb_path, smf_filter_path):
    """
    Perform random grid search for VAE hyperparameters.

    Parameters:
        num_trials (int): Number of random trials.
        data_path (str): Path to the input data file.
        full_emb_path (str): Path to the full embedding file.
        smf_filter_path (str): Path to the SMF filter file.

    Returns:
        None
    """
    param_space = {
        "learning_rate": [1e-4, 1e-5, 1e-6],
        "hidden_layers": [[2048, 1024, 512], [2048, 1024, 512, 256], [2048, 1024, 512, 256, 128]],
        "latent_dim": [512, 256, 128],
        "batch_norm": [True, False],
        "dropout_rate": [0.0, 0.2, 0.5],
        "num_epochs": [50, 100, 200, 400],
        "batch_size": [64, 128, 256, 512],
        "activation_function": ["ReLU", "LeakyReLU", "ELU", "Tanh"]
    }

    best_r2 = float('-inf')
    best_params = None

    for trial in range(num_trials):
        # Sample random parameters
        params = {key: random.choice(values) for key, values in param_space.items()}
        print(f"Trial {trial + 1}/{num_trials}: Testing parameters: {params}")

        # Train VAE and extract embeddings
        train_vae_and_extract_embeddings(
            params,
            data_path,
            model_save_path=f"vae_model_trial_{trial}.pth",
            embeddings_save_path=f"vae_embeddings_trial_{trial}.parquet"
        )

        # Calculate R2 score
        r2 = calculate_r2(full_emb_path, f"vae_embeddings_trial_{trial}.parquet", smf_filter_path)
        print(f"R2 score for trial {trial + 1}: {r2}")

        # Update best parameters
        if r2 > best_r2:
            best_r2 = r2
            best_params = params

    print(f"Best R2 score: {best_r2}")
    print(f"Best parameters: {best_params}")


# Run the random grid search
random_grid_search(
    num_trials=20,  # Number of trials
    data_path="../data/yeast_emb_full.parquet",
    full_emb_path="../data/yeast_emb_full.parquet",
    smf_filter_path="../data/smf.csv"
)
