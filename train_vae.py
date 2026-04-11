import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Importing from your subfolders
from data.data_utils import get_math_reasoning_data
from models.vae import MathVAE, vae_loss

# Optimized for L40S
torch.set_float32_matmul_precision('high')

class MathTextDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def train():
    # --- Hyperparameters ---
    latent_dim = 128
    batch_size = 128  # Increased for L40S memory
    epochs = 30
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Real Data from Hugging Face ---
    # Using the function in your data/data_utils.py
    math_data = get_math_reasoning_data(dataset_name="EleutherAI/hendrycks_math")
    
    dataset = MathTextDataset(math_data)
    # Optimized DataLoader for server hardware
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # --- Initialize Model ---
    model = MathVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on {len(math_data)} high-school level math steps.")
    print(f"Starting VAE training on {device}...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_sentences in dataloader:
            optimizer.zero_grad()

            # 1. Get the "ground truth" BERT embeddings
            with torch.no_grad():
                target_embeddings = model.embedding_model.encode(
                    batch_sentences, convert_to_tensor=True
                ).detach().clone().to(device)

            # 2. Forward pass through VAE
            recon_batch, mu, logvar = model(batch_sentences)

            # 3. Calculate Loss (using beta=0.1 for a smoother manifold)
            loss = vae_loss(recon_batch, target_embeddings, mu, logvar, beta=0.1)
            
            # 4. Backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    # --- Save the Weights ---
    torch.save(model.state_dict(), "vae_weights.pt")
    print("Training complete. Weights saved to vae_weights.pt")

if __name__ == "__main__":
    train()