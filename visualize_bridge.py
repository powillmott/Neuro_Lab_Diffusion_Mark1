import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from models.vae import MathVAE
from models.bridge_dit import BridgeDiT
from data.dataset2 import LatentBridgeDataset, bridge_collate_fn

# Use a pretty style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def generate_flow_bridge_path(model, z_start, z_end, rel_pos, num_steps=50):
    """Generates the full sequence of latents along the bridge (Euler method)."""
    model.eval()
    device = z_start.device
    
    # Record the entire path, starting exactly at z_start
    path = [z_start.squeeze(0).cpu().numpy()]
    z_current = z_start.clone()
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((1, 1), t_val, device=device)
            t_embed = (t * 1000).squeeze(1)
            
            # Predict velocity
            velocity = model(z_current, t_embed, z_start, z_end, rel_pos)
            
            # Take an Euler step
            z_current = z_current + (velocity * dt)
            path.append(z_current.squeeze(0).cpu().numpy())
            
    # Add z_end as the final point in the path (at t=1.0)
    path.append(z_end.squeeze(0).cpu().numpy())
    return np.array(path)

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading models and data...")
    # 1. VAE (Needed to encode a large batch for PCA fitting)
    vae = MathVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("weights/vae_highschool_weights.pt", map_location=device))
    vae.eval()

    # 2. Flow Matching DiT
    model = BridgeDiT(latent_dim=latent_dim).to(device)
    # Load your BEST Flow Matching weights here
    model.load_state_dict(torch.load("weights/bridge_dit_flow_final.pt", map_location=device))
    model.eval()

    # 3. Dataset (We need a good chunk of data to fit PCA)
    full_ds = LatentBridgeDataset(vae=vae, device=device)
    # Take 500 samples to define the "shape" of the manifold
    pca_subset = Subset(full_ds, range(min(500, len(full_ds))))
    dl = DataLoader(pca_subset, batch_size=len(pca_subset), collate_fn=bridge_collate_fn)
    batch = next(iter(dl))
    
    # Extract 'target' latents (z_t) as representative points on the manifold
    z_t_all = batch["z_target"].cpu().numpy() 

    # --- PART 1: FIT PCA ON THE MANIFOLD ---
    print("Fitting PCA to visualize 128D latent space in 2D...")
    pca = PCA(n_components=2)
    pca.fit(z_t_all)
    z_t_2d = pca.transform(z_t_all)

    # Visualization 1: The VAE Manifold Structure
    plt.figure(figsize=(10, 8))
    plt.scatter(z_t_2d[:, 0], z_t_2d[:, 1], alpha=0.5, s=15, c='lightgray', label='VAE Manifold Points (z_t)')
    plt.title("PCA Projection of 128D VAE Latent Space")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plot_path = output_dir / "vae_manifold.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Manifold visualization to {plot_path}")
    plt.close()

    # --- PART 2: VISUALIZE THE TRAVERSAL ---
    print("\nGenerating a single test bridge traversal...")
    # Pick one example from the batch (e.g., index 0)
    z_s_test = batch["z_start"][0:1].to(device)
    z_e_test = batch["z_end"][0:1].to(device)
    # We set rel_pos to 0.5 just as a dummy context (the full generation covers 0 to 1)
    rel_pos_test = torch.full((1, 1), 0.5, device=device)
    
    # Generate 50 steps along the bridge
    bridge_path_128d = generate_flow_bridge_path(model, z_s_test, z_e_test, rel_pos_test, num_steps=50)
    
    # Transform the generated 128D path into the SAME 2D PCA space
    bridge_path_2d = pca.transform(bridge_path_128d)

    # Visualization 2: The Traversal
    plt.figure(figsize=(12, 10))
    # 1. Plot the background manifold (faded)
    plt.scatter(z_t_2d[:, 0], z_t_2d[:, 1], alpha=0.2, s=10, c='lightgray', label='Manifold Background')
    
    # 2. Plot the Start and End Points (Anchors)
    z_s_2d = bridge_path_2d[0]
    z_e_2d = bridge_path_2d[-1]
    plt.scatter(z_s_2d[0], z_s_2d[1], c='green', s=150, marker='X', label='Start (z_start)', zorder=10)
    plt.scatter(z_e_2d[0], z_e_2d[1], c='red', s=150, marker='X', label='End (z_end)', zorder=10)

    # 3. Plot the Continuous Traversal Path
    plt.plot(bridge_path_2d[:, 0], bridge_path_2d[:, 1], c='blue', linewidth=2.5, alpha=0.8, label='Generated Bridge Flow')
    
    # 4. (Optional) Add arrows to show direction of the flow
    # Plot an arrow every 10 steps
    for i in range(0, len(bridge_path_2d) - 1, 10):
        plt.arrow(bridge_path_2d[i, 0], bridge_path_2d[i, 1], 
                  bridge_path_2d[i+1, 0] - bridge_path_2d[i, 0], 
                  bridge_path_2d[i+1, 1] - bridge_path_2d[i, 1],
                  head_width=0.1, head_length=0.15, fc='blue', ec='blue', alpha=0.8, zorder=5)

    plt.title("Flow Matching Bridge Traversing the VAE Latent Manifold")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(loc='upper right')
    
    # Zoom in slightly around the bridge if needed
    x_buffer = (bridge_path_2d[:, 0].max() - bridge_path_2d[:, 0].min()) * 0.2
    y_buffer = (bridge_path_2d[:, 1].max() - bridge_path_2d[:, 1].min()) * 0.2
    plt.xlim(bridge_path_2d[:, 0].min() - x_buffer, bridge_path_2d[:, 0].max() + x_buffer)
    plt.ylim(bridge_path_2d[:, 1].min() - y_buffer, bridge_path_2d[:, 1].max() + y_buffer)

    final_plot_path = output_dir / "bridge_traversal.png"
    plt.savefig(final_plot_path, dpi=300)
    print(f"Saved Traversal visualization to {final_plot_path}")
    print("\nVisualizations complete. Check the 'visualizations/' directory.")

if __name__ == "__main__":
    visualize()