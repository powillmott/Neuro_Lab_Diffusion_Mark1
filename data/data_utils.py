from datasets import load_dataset
import re

def get_math_reasoning_data(dataset_name="gsm8k", config="main", split="train"):
    print(f"Fetching {dataset_name}...")
    # This downloads the data to ~/.cache/huggingface/datasets
    ds = load_dataset(dataset_name, config, split=split)
    
    all_steps = []
    for example in ds:
        reasoning_only = example['answer'].split('####')[0].strip()
        steps = re.split(r'(?<=[.!?]) +', reasoning_only)
        all_steps.extend([s for s in steps if len(s) > 5])
        
    return all_steps