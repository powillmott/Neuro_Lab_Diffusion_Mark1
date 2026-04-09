import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import re

class LatentStepDataset(Dataset):
    def __init__(self, vae, split="train", max_steps=16, device="cpu"):
        """
        Fetches GSM8K data internally and uses the VAE to prepare latent trajectories.
        """
        print(f"Loading GSM8K {split} split for Diffusion training...")
        self.raw_data = load_dataset("gsm8k", "main")[split]
        
        self.vae = vae.to(device)
        self.vae.eval()
        self.max_steps = max_steps
        self.device = device
        self.latent_dim = vae.fc_mu.out_features

    def _split_into_thoughts(self, text):
        # GSM8K logic often separates reasoning from the answer with '####'
        reasoning_text = text.split('####')[0].strip()
        sentences = re.split(r'(?<=[.!?]) +', reasoning_text)
        return [s for s in sentences if len(s) > 2]

    def __len__(self):
        return len(self.raw_data)

    @torch.no_grad()
    def __getitem__(self, idx):
        # Access the text from the Hugging Face dataset object
        example_text = self.raw_data[idx]['answer']
        thoughts = self._split_into_thoughts(example_text)
        
        # Use the VAE to turn sentences into latent coordinates (Mu)
        mu, _ = self.vae.encode_text(thoughts) 
        mu = mu.detach().clone().cpu() # Move to CPU for storage in DataLoader
        
        # Create the trajectory buffer (max_steps x latent_dim)
        latent_trajectory = torch.zeros((self.max_steps, self.latent_dim))
        
        # Fill the buffer
        seq_len = min(mu.shape[0], self.max_steps)
        latent_trajectory[:seq_len] = mu[:seq_len]
        
        # Create a mask (1.0 for real thoughts, 0.0 for padding)
        mask = torch.zeros(self.max_steps)
        mask[:seq_len] = 1.0

        return {
            "latent_trajectory": latent_trajectory,
            "mask": mask
        }

def collate_fn(batch):
    return {
        "latent_trajectory": torch.stack([x["latent_trajectory"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
    }