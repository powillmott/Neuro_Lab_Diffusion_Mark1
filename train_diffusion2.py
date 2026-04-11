import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vae import MathVAE
from models.bridge_dit import BridgeDiT  # Use the new model
from models.diffusion import DiffusionEngine
from data.dataset2 import LatentBridgeDataset, bridge_collate_fn

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128
    batch_size = 128 # We can use a bigger batch since we're only doing 1 step at a time
    epochs = 50
    lr = 1e-4

    # 1. VAE
    vae = MathVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("vae_weights.pt", map_location=device))
    vae.eval()

    # 2. Bridge Dataset
    dataset = LatentBridgeDataset(vae=vae, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=bridge_collate_fn)

    # 3. Bridge DiT
    model = BridgeDiT(latent_dim=latent_dim).to(device)
    engine = DiffusionEngine(timesteps=1000, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print(f"Starting Bridge Diffusion training on {device}...")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            z_start = batch["z_start"].to(device)
            z_target = batch["z_target"].to(device)
            z_end = batch["z_end"].to(device)
            rel_pos = batch["rel_pos"].to(device)

            # Standard Diffusion noise logic
            t = torch.randint(0, engine.timesteps, (z_target.shape[0],), device=device)
            x_noisy, noise_target = engine.add_noise(z_target, t)

            # The key change: model takes start, end, and position as context
            predicted_noise = model(x_noisy, t, z_start, z_end, rel_pos)

            loss = torch.nn.functional.mse_loss(predicted_noise, noise_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(dataloader):.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"bridge_dit_ep{epoch+1}.pt")

    torch.save(model.state_dict(), "bridge_dit_final.pt")
    print("Training complete!")

if __name__ == "__main__":
    train()