import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import math
from pathlib import Path
from models.vae import MathVAE
from models.bridge_dit import BridgeDiT  # Use the new model
from models.diffusion import DiffusionEngine
from data.dataset2 import LatentBridgeDataset, bridge_collate_fn

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128
    batch_size = 128
    epochs = 50
    lr = 1e-4
    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. VAE
    vae = MathVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("weights/vae_highschool_weights.pt", map_location=device))
    vae.eval()

    # 2. Bridge Dataset
    full_ds = LatentBridgeDataset(vae=vae, device=device)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=bridge_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=bridge_collate_fn)
    # 3. Bridge DiT
    model = BridgeDiT(latent_dim=latent_dim).to(device)
    trainable, total = model.get_model_stats()
    print(f"Model Parameters: {trainable:,} trainable, {total:,} total")
    engine = DiffusionEngine(timesteps=1000, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    
    for epoch in range(epochs):
        model.train()
        train_mse_sum = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            z_s, z_t, z_e, rel = batch["z_start"].to(device), batch["z_target"].to(device), batch["z_end"].to(device), batch["rel_pos"].to(device)
            
            t = torch.randint(0, engine.timesteps, (z_t.shape[0],), device=device)
            x_noisy, noise_target = engine.add_noise(z_t, t)
            
            pred = model(x_noisy, t, z_s, z_e, rel)
            loss = torch.nn.functional.mse_loss(pred, noise_target)
            
            loss.backward()
            optimizer.step()
            train_mse_sum += loss.item()

        # Validation Step
        model.eval()
        val_mse_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                z_s, z_t, z_e, rel = batch["z_start"].to(device), batch["z_target"].to(device), batch["z_end"].to(device), batch["rel_pos"].to(device)
                t = torch.randint(0, engine.timesteps, (z_t.shape[0],), device=device)
                x_noisy, noise_target = engine.add_noise(z_t, t)
                pred = model(x_noisy, t, z_s, z_e, rel)
                val_mse_sum += torch.nn.functional.mse_loss(pred, noise_target).item()

        # Calculate final averages and take the square root for RMSE
        train_rmse = math.sqrt(train_mse_sum / len(train_loader))
        val_rmse = math.sqrt(val_mse_sum / len(val_loader))

        print(f"Epoch {epoch+1} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

        # Save checkpoint every 10 epochs (overwrites existing file names in weights/)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = weights_dir / f"bridge_dit_ep{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    final_path = weights_dir / "bridge_dit_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Weights saved to {final_path}")

if __name__ == "__main__":
    train()