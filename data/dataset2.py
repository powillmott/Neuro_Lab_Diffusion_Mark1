import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
import random
import re

class LatentBridgeDataset(Dataset):
    def __init__(self, vae, split="train", device="cpu"):
        print(f"Loading Hendrycks MATH subsets for Bridge training...")
        
        # We want the model to be an expert in all math areas
        subsets = ["algebra", "geometry", "intermediate_algebra", "number_theory", "prealgebra"]
        
        # Combine all subsets into one big list
        self.raw_data = []
        for sub in subsets:
            print(f"  - Fetching {sub}...")
            ds = load_dataset("EleutherAI/hendrycks_math", sub, split=split)
            self.raw_data.extend(list(ds))
        
        self.vae = vae.to(device)
        self.vae.eval()
        self.device = device
    
    def _split_into_thoughts(self, text):
        # Handle the LaTeX-heavy 'solution' key
        text = text.split('####')[0].strip()
        # Robust math-aware sentence splitting
        sentences = re.split(r'(?<!\d)\.(?!\d) +|(?<=[.!?]) +', text)
        return [s for s in sentences if len(s) > 5]

    def __len__(self):
        return len(self.raw_data)

    @torch.no_grad()
    def __getitem__(self, idx):
        # MATH uses 'solution' as the key
        example_text = self.raw_data[idx]['solution']
        thoughts = self._split_into_thoughts(example_text)
        
        if len(thoughts) < 3:
            # Recursively try another if the solution is too short
            return self.__getitem__(random.randint(0, len(self.raw_data)-1))

        # Sandwich logic: i < j < k
        i = random.randint(0, len(thoughts) - 3)
        k = random.randint(i + 2, len(thoughts) - 1)
        j = random.randint(i + 1, k - 1)

        # Encode thoughts and move back to CPU for batching
        mu, _ = self.vae.encode_text([thoughts[i], thoughts[j], thoughts[k]])
        mu = mu.cpu()

        rel_pos = torch.tensor([(j - i) / (k - i)], dtype=torch.float32)

        return {
            "z_start": mu[0], 
            "z_target": mu[1], 
            "z_end": mu[2], 
            "rel_pos": rel_pos
        }

def bridge_collate_fn(batch):
    return {
        "z_start": torch.stack([x["z_start"] for x in batch]),
        "z_target": torch.stack([x["z_target"] for x in batch]),
        "z_end": torch.stack([x["z_end"] for x in batch]),
        "rel_pos": torch.stack([x["rel_pos"] for x in batch]),
    }