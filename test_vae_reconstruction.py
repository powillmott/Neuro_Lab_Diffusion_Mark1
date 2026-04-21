import torch
import math
import pandas as pd
from models.vae import MathVAE
from data.data_utils import get_math_data_with_metadata

def generate_true_performance_matrix(model_path="weights/vae_highschool_weights.pt", partition="test"):
    device = "cuda"
    vae = MathVAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()

    # Get the dictionary, pull out ONLY the requested partition ("train" or "test")
    full_data = get_math_data_with_metadata()
    data_by_subset = full_data[partition] 
    
    data_records = []
    
    print(f"\nComputing metrics for {partition.upper()} set...")
    with torch.no_grad():
        for sub, examples in data_by_subset.items():
            for example in examples:
                level_label = str(example['level']) 
                sample = example['solution'].split('####')[0]
                
                if len(sample) < 10: continue
                    
                recon, mu, _ = vae([sample])
                orig_emb = vae.embedding_model.encode(sample)
                
                mse = torch.nn.functional.mse_loss(recon, torch.tensor(orig_emb).to(device).view(1, -1))
                
                data_records.append({
                    "Category": sub, 
                    "Level": level_label, 
                    "SumSqErr": mse.item(), 
                    "Count": 1
                })

    df = pd.DataFrame(data_records)
    if df.empty:
        print(f"No data collected for {partition} partition!")
        return

    # True Marginalization Logic
    cell_grouped = df.groupby(['Category', 'Level'])[['SumSqErr', 'Count']].sum()
    cell_grouped['RMSE'] = (cell_grouped['SumSqErr'] / cell_grouped['Count']).apply(math.sqrt)
    
    cat_grouped = df.groupby('Category')[['SumSqErr', 'Count']].sum()
    cat_grouped['Total_RMSE'] = (cat_grouped['SumSqErr'] / cat_grouped['Count']).apply(math.sqrt)
    
    lvl_grouped = df.groupby('Level')[['SumSqErr', 'Count']].sum()
    lvl_grouped['Total_RMSE'] = (lvl_grouped['SumSqErr'] / lvl_grouped['Count']).apply(math.sqrt)

    pivot = cell_grouped.reset_index().pivot(index="Category", columns="Level", values="RMSE")

    print(f"\n=== True RMSE Matrix: {partition.upper()} ===")
    print(pivot.round(4))
    print(f"\n--- Marginalized per Category ({partition.upper()}) ---")
    print(cat_grouped['Total_RMSE'].round(4))
    print(f"\n--- Marginalized per Level ({partition.upper()}) ---")
    print(lvl_grouped['Total_RMSE'].round(4))
    print("=" * 40)

if __name__ == "__main__":
    # Run the matrix on the Training data to establish the baseline fit
    generate_true_performance_matrix(partition="train")
    
    # Run the matrix on the Test data to check for overfitting/generalization
    generate_true_performance_matrix(partition="test")