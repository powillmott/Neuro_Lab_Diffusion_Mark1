from datasets import load_dataset
import re

from datasets import load_dataset
import re

def get_math_reasoning_data(dataset_name="EleutherAI/hendrycks_math", split="train"):
    print(f"Fetching {dataset_name} ({split})...")
    
    # Hendrycks MATH has subsets (algebra, counting_and_probability, etc.)
    # We will load a few core ones to get a broad 'high school' manifold
    subsets = ["algebra", "geometry", "intermediate_algebra", "number_theory"]
    all_steps = []
    
    for sub in subsets:
        print(f"  - Extracting {sub}...")
        ds = load_dataset(dataset_name, sub, split=split)
        
        for example in ds:
            # Hendrycks MATH uses 'solution' as the key
            text = example['solution'].strip()
            # Remove the LaTeX boxed final answer if it exists
            text = text.split('####')[0] 
            
            # Split into individual thoughts/sentences
            steps = re.split(r'(?<!\d)\.(?!\d) +|(?<=[.!?]) +', text)
            all_steps.extend([s for s in steps if len(s) > 8])
            
    print(f"Total high-school steps extracted: {len(all_steps)}")
    return all_steps