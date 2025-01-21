import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):  # Rename to CVAE to highlight it is conditional
    def __init__(self, input_dim, latent_dim, hidden_layers, activation_function, dropout_rate, batch_norm, cond_dim):
        super(CVAE, self).__init__()
        
        self.latent_dim = latent_dim

        # Activation function
        activation_fn = getattr(nn, activation_function)()

        # Build Encoder
        encoder_layers = []
        in_dim = input_dim + cond_dim  # Include conditioning dimension
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
        in_dim = latent_dim + cond_dim  # Include conditioning dimension
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

    def encode(self, x, c):
        x_cond = torch.cat([x, c], dim=1)  # Concatenate input and conditioning variable
        hidden = self.encoder(x_cond)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z_cond = torch.cat([z, c], dim=1)  # Concatenate latent vector and conditioning variable
        return self.decoder(z_cond)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

