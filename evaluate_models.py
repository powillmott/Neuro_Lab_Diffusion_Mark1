import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import time
import math

from models.bridge_dit import BridgeDiT
from models.vae import MathVAE
from data.dataset2 import LatentBridgeDataset, bridge_collate_fn
# Assume you have your old DiffusionEngine available for Model 1
from models.diffusion import DiffusionEngine 

def generate_flow_bridge(model, z_start, z_end, rel_pos, num_steps=20):
    """Generates the intermediate step using the Euler ODE solver (Mark 2)."""
    device = z_start.device
    batch_size = z_start.shape[0]
    
    # Start at pure noise
    z_current = torch.randn_like(z_start)
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t_val = i * dt
        # Shape t to match batch size
        t = torch.full((batch_size, 1), t_val, device=device)
        t_embed = (t * 1000).squeeze(1) # Match training scale
        
        velocity = model(z_current, t_embed, z_start, z_end, rel_pos)
        z_current = z_current + (velocity * dt)
        
    return z_current

def generate_noise_bridge(model, engine, z_start, z_end, rel_pos):
    """Generates the intermediate step using the DDPM Scheduler (Mark 1)."""
    # Note: Replace this inner loop with your engine's actual inference/sample method 
    # if you have one built into DiffusionEngine.
    device = z_start.device
    batch_size = z_start.shape[0]
    
    z_current = torch.randn_like(z_start)
    
    # Standard reverse diffusion loop
    for i in reversed(range(engine.timesteps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
        # Predict the noise
        predicted_noise = model(z_current, t, z_start, z_end, rel_pos)
        
        # Remove the noise (Simplified DDPM step, use engine.step if available)
        alpha_t = engine.alphas[t][:, None]
        alpha_hat_t = engine.alphas_cumprod[t][:, None]
        beta_t = engine.betas[t][:, None]
        
        if i > 0:
            noise = torch.randn_like(z_current)
        else:
            noise = torch.zeros_like(z_current)
            
        z_current = (1 / torch.sqrt(alpha_t)) * (z_current - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        
    return z_current

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128

    vae = MathVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("weights/vae_highschool_weights.pt", map_location=device))
    vae.eval()
    
    # 1. Load Data (Just taking 100 samples for a quick benchmark)
    full_ds = LatentBridgeDataset(vae=vae, device=device)
    val_ds = Subset(full_ds, range(len(full_ds) - 100, len(full_ds)))
    val_loader = DataLoader(val_ds, batch_size=100, shuffle=False, collate_fn=bridge_collate_fn)
    batch = next(iter(val_loader))
    
    z_s = batch["z_start"].to(device)
    z_t_true = batch["z_target"].to(device)
    z_e = batch["z_end"].to(device)
    rel = batch["rel_pos"].to(device)

    # 2. Load Model 1 (Noise / Mark 1)
    print("--- Evaluating Model 1 (Noise Prediction) ---")
    model_noise = BridgeDiT(latent_dim=latent_dim).to(device)
    model_noise.load_state_dict(torch.load("weights/bridge_dit_final.pt", map_location=device))
    model_noise.eval()
    engine = DiffusionEngine(timesteps=1000, device=device)
    
    start_time = time.time()
    with torch.no_grad():
        z_pred_noise = generate_noise_bridge(model_noise, engine, z_s, z_e, rel)
    noise_time = time.time() - start_time
    
    # 3. Load Model 2 (Flow Matching / Mark 2)
    print("\n--- Evaluating Model 2 (Flow Matching) ---")
    model_flow = BridgeDiT(latent_dim=latent_dim).to(device)
    model_flow.load_state_dict(torch.load("weights/bridge_dit_flow_final.pt", map_location=device))
    model_flow.eval()
    
    start_time = time.time()
    with torch.no_grad():
        # Using 20 steps for Flow Matching
        z_pred_flow = generate_flow_bridge(model_flow, z_s, z_e, rel, num_steps=20)
    flow_time = time.time() - start_time

    # 4. Calculate Metrics
    print("\n================ BENCHMARK RESULTS ================")
    
    # Speed
    print(f"Inference Time (100 samples):")
    print(f"  Model 1 (Noise - 1000 steps): {noise_time:.2f} seconds")
    print(f"  Model 2 (Flow  - 20 steps):   {flow_time:.2f} seconds")
    print(f"  Speedup: {noise_time / flow_time:.2f}x faster")
    
    # Cosine Similarity (How close is the direction?)
    # 1.0 is perfect match, -1.0 is opposite direction
    cos_sim_noise = F.cosine_similarity(z_pred_noise, z_t_true, dim=-1).mean().item()
    cos_sim_flow = F.cosine_similarity(z_pred_flow, z_t_true, dim=-1).mean().item()
    
    print(f"\nCosine Similarity (Target=1.000):")
    print(f"  Model 1 (Noise): {cos_sim_noise:.4f}")
    print(f"  Model 2 (Flow):  {cos_sim_flow:.4f}")
    
    # L2 Distance / MSE (How close is the magnitude?)
    # Lower is better
    mse_noise = F.mse_loss(z_pred_noise, z_t_true).item()
    mse_flow = F.mse_loss(z_pred_flow, z_t_true).item()
    
    print(f"\nL2 Distance / MSE (Target=0.000):")
    print(f"  Model 1 (Noise): {mse_noise:.4f}")
    print(f"  Model 2 (Flow):  {mse_flow:.4f}")
    print("===================================================")

if __name__ == "__main__":
    evaluate()