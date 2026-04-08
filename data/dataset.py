import torch
from torch.utils.data import Dataset
import re

class LatentStepDataset(Dataset):
    def __init__(self, paragraphs, vae, max_steps=16, device="cpu"):
        """
        Args:
            paragraphs (list): List of strings (paragraphs of math reasoning).
            vae (MathVAE): Your trained VAE model.
            max_steps (int): Fixed sequence length for the diffusion model.
        """
        self.paragraphs = paragraphs
        self.vae = vae.to(device)
        self.vae.eval()
        self.max_steps = max_steps
        self.device = device
        self.latent_dim = vae.fc_mu.out_features

    def _split_into_thoughts(self, text):
        # Simple split by sentences. You can make this more complex later.
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return [s for s in sentences if len(s) > 2]

    def __len__(self):
        return len(self.paragraphs)

    @torch.no_grad()
    def __getitem__(self, idx):
        text = self.paragraphs[idx]
        thoughts = self._split_into_thoughts(text)
        
        # 1. Encode each thought into the VAE latent space (Mu)
        # We use Mu (mean) as the "stable" semantic coordinate for diffusion
        mu, _ = self.vae.encode_text(thoughts) 
        
        # mu shape: (num_thoughts, latent_dim)
        seq_len = mu.shape[0]
        
        # 2. Handle sequence length (Padding or Truncating)
        # We initialize a zero buffer (our "fixed thought slots")
        latent_trajectory = torch.zeros((self.max_steps, self.latent_dim))
        
        if seq_len > self.max_steps:
            # Truncate if too long
            latent_trajectory = mu[:self.max_steps]
            mask = torch.ones(self.max_steps)
        else:
            # Pad if too short
            latent_trajectory[:seq_len] = mu
            # Create a mask so the model knows which slots are real thoughts
            mask = torch.zeros(self.max_steps)
            mask[:seq_len] = 1.0

        return {
            "latent_trajectory": latent_trajectory, # (max_steps, latent_dim)
            "mask": mask,                           # (max_steps)
            "actual_len": min(seq_len, self.max_steps)
        }

def collate_fn(batch):
    """Utility to batch the trajectories correctly"""
    return {
        "latent_trajectory": torch.stack([x["latent_trajectory"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
    }