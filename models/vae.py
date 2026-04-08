import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class MathVAE(nn.Module):
    def __init__(self, latent_dim=128, model_name='all-MiniLM-L6-v2'):
        super(MathVAE, self).__init__()
        
        # 1. The Language Backbone (Frozen for simplicity)
        self.embedding_model = SentenceTransformer(model_name)
        input_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # 2. Encoder: Maps sentence embedding to Mu and Sigma
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # 3. Decoder: Reconstructs the original sentence embedding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode_text(self, sentences):
        """Helper to go from raw strings to Mu and LogVar"""
        with torch.no_grad():
            # Get 384-dim embeddings from SentenceTransformer
            embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
        
        h = self.encoder(embeddings)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, sentences):
        mu, logvar = self.encode_text(sentences)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Standard VAE Loss: Reconstruction (MSE) + KL Divergence.
    Beta controls the 'smoothness' of the manifold.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize KLD by batch size
    kld_loss /= recon_x.size(0)
    
    return recon_loss + (beta * kld_loss)