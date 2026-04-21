from datasets import load_dataset
import re
import torch
from torch.utils.data import random_split

from datasets import load_dataset
import re

# def get_math_reasoning_data(dataset_name="EleutherAI/hendrycks_math", split="train"):
#     print(f"Fetching {dataset_name} ({split})...")
    
#     # Hendrycks MATH has subsets (algebra, counting_and_probability, etc.)
#     # We will load a few core ones to get a broad 'high school' manifold
#     subsets = ["algebra", "geometry", "intermediate_algebra", "number_theory"]
#     all_steps = []
    
#     for sub in subsets:
#         print(f"  - Extracting {sub}...")
#         ds = load_dataset(dataset_name, sub, split=split)
        
#         for example in ds:
#             # Hendrycks MATH uses 'solution' as the key
#             text = example['solution'].strip()
#             # Remove the LaTeX boxed final answer if it exists
#             text = text.split('####')[0] 
            
#             # Split into individual thoughts/sentences
#             steps = re.split(r'(?<!\d)\.(?!\d) +|(?<=[.!?]) +', text)
#             all_steps.extend([s for s in steps if len(s) > 8])
            
#     print(f"Total high-school steps extracted: {len(all_steps)}")
#     return all_steps

def get_math_reasoning_data(dataset_name="EleutherAI/hendrycks_math", val_ratio=0.1, seed=42):
    """
    Training function: Calls the metadata function to ensure exact split alignment,
    then flattens the problems into individual steps.
    """

    data_dict = get_math_data_with_metadata(dataset_name, val_ratio, seed)
    
    train_steps = []
    test_steps = []

    def flatten_examples(examples):
        steps_list = []
        for ex in examples:
            text = ex['solution'].split('####')[0].strip()
            steps = re.split(r'(?<!\d)\.(?!\d) +|(?<=[.!?]) +', text)
            steps_list.extend([s for s in steps if len(s) > 8])
        return steps_list

    print("Flattening steps for training...")
    for sub in data_dict["train"]:
        train_steps.extend(flatten_examples(data_dict["train"][sub]))
        test_steps.extend(flatten_examples(data_dict["test"][sub]))
        
    print(f"Total flattened steps: {len(train_steps)} Train | {len(test_steps)} Test/Val")
    return train_steps, test_steps

# def get_math_data_with_metadata(dataset_name="EleutherAI/hendrycks_math", split="train"):
#     """Used for Diagnostics: returns a dict {category: [list of examples with metadata]}."""
#     subsets = ["algebra", "geometry", "intermediate_algebra", "number_theory", "prealgebra"]
#     data_dict = {}
    
#     for sub in subsets:
#         # Load the full HF dataset object
#         ds = load_dataset(dataset_name, sub, split=split)
#         # Convert to a list of dicts to ensure it's iterable and contains metadata
#         data_dict[sub] = [ex for ex in ds]
        
#     return data_dict

def get_math_data_with_metadata(dataset_name="EleutherAI/hendrycks_math", val_ratio=0.1, seed=42):
    """
    Core function: Splits the dataset at the PROBLEM level to prevent leakage.
    Returns: {"train": {category: [examples]}, "test": {category: [examples]}}
    """
    print(f"Fetching {dataset_name} (seed={seed}, val_ratio={val_ratio})...")
    subsets = ["algebra", "geometry", "intermediate_algebra", "number_theory", "prealgebra"]
    data_dict = {"train": {}, "test": {}}
    
    # Lock the seed for reproducible splits
    generator = torch.Generator().manual_seed(seed)
    
    for sub in subsets:
        print(f"  - Loading and partitioning {sub}...")
        ds = load_dataset(dataset_name, sub, split="train")
        
        # Create deterministic shuffled indices
        indices = torch.randperm(len(ds), generator=generator).tolist()
        split_idx = int((1 - val_ratio) * len(ds))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Use .select() for speed, cast to list for easy downstream iteration
        data_dict["train"][sub] = list(ds.select(train_indices))
        data_dict["test"][sub] = list(ds.select(test_indices))
        
    return data_dict