import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from vae import MathVAE, vae_loss

# 1. Simple Dataset to handle math sentences
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
    batch_size = 32
    epochs = 20
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dummy Data (Replace with your math corpus later) ---
    math_data = [
        "Let x be equal to five.",
        "Add three to both sides of the equation.",
        "The derivative of x squared is two x.",
        "Simplify the expression by grouping terms.",
        "Solve for the variable y."
    ] * 100 # Artificial inflation for testing
    
    dataset = MathTextDataset(math_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Initialize Model ---
    model = MathVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
                ).to(device)

            # 2. Forward pass through VAE
            recon_batch, mu, logvar = model(batch_sentences)

            # 3. Calculate Loss
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