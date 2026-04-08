import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from vae import MathVAE
from dit_1d import DiT1D
from diffusion import DiffusionEngine
from dataset import LatentStepDataset, collate_fn

def train():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128
    max_steps = 16
    batch_size = 64
    epochs = 50
    lr = 1e-4

    # 1. Load the Pre-trained VAE (The Manifold)
    vae = MathVAE(latent_dim=latent_dim).to(device)
    try:
        vae.load_state_dict(torch.load("vae_weights.pt", map_location=device))
        print("Loaded VAE weights successfully.")
    except FileNotFoundError:
        print("Error: vae_weights.pt not found. Train the VAE first!")
        return
    vae.eval() # VAE stays frozen during diffusion training

    # 2. Setup Dataset (Using the structure we built)
    # For now, pass a dummy list or the HF dataset setup from dataset.py
    # math_paragraphs = ["Paragraph 1...", "Paragraph 2..."] 
    dataset = LatentStepDataset(vae=vae, max_steps=max_steps, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 3. Initialize the Brain (DiT) and the Engine (Diffusion)
    model = DiT1D(input_dim=latent_dim, max_steps=max_steps).to(device)
    engine = DiffusionEngine(timesteps=1000, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    print(f"Starting Diffusion training on {device}...")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            # x_0 is the clean latent trajectory from the VAE
            x_0 = batch["latent_trajectory"].to(device)
            mask = batch["mask"].to(device)

            # Sample a random timestep for each item in the batch
            t = torch.randint(0, engine.timesteps, (x_0.shape[0],), device=device)

            # Add noise to the trajectory according to the cosine schedule
            x_noisy, noise_target = engine.add_noise(x_0, t)

            # The model tries to predict the noise added
            # We pass the mask so the model can focus on real 'thoughts'
            predicted_noise = model(x_noisy, t)

            # Calculate Loss (MSE between actual noise and predicted noise)
            # We apply the mask so we don't penalize noise on padding slots
            loss = torch.mean((predicted_noise - noise_target)**2 * mask.unsqueeze(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(dataloader):.6f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"dit_checkpoint_ep{epoch+1}.pt")

    # Final Save
    torch.save(model.state_dict(), "dit_final.pt")
    print("Diffusion training complete. Model saved as dit_final.pt")

if __name__ == "__main__":
    train()