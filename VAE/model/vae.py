# Define the VAE model.

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers, activation_function, dropout_rate, batch_norm):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim

        # Activation function
        activation_fn = getattr(nn, activation_function)()

        # Build Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(activation_fn)
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu_layer = nn.Linear(in_dim, latent_dim)  # Mean of latent distribution
        self.logvar_layer = nn.Linear(in_dim, latent_dim)  # Log-variance of latent distribution

        # Build Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(activation_fn)
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        decoder_layers.append(nn.Linear(in_dim, input_dim))  # Reconstruction layer
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
