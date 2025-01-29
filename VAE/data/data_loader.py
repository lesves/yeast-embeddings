# Load the data and prepare the DataLoader.

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path, batch_size):
    # Load data
    dnalm_df = pd.read_parquet(file_path)
    dnalm_df = dnalm_df.set_index('gene_id')
    print("Input dimensions: ", dnalm_df.shape)

    # Normalize data
    #scaler = StandardScaler()
    #dnalm_scaled = scaler.fit_transform(dnalm_df)

    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(dnalm_df.values, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train_df, dtype=torch.float32)
    val_tensor = torch.tensor(val_df, dtype=torch.float32)
    test_tensor = torch.tensor(test_df, dtype=torch.float32)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dnalm_df.shape[1]
