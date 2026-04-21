import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Importing from your subfolders
from data.data_utils import get_math_reasoning_data
from models.vae import MathVAE, vae_loss
import math

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
    train_steps, val_steps = get_math_reasoning_data()

    train_ds = MathTextDataset(train_steps)
    val_ds = MathTextDataset(val_steps)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # --- Initialize Model ---
    model = MathVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on {len(train_ds)} steps, Validating on {len(val_ds)} steps.")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_sentences in train_loader:
            optimizer.zero_grad()
            
            # Ground truth from SentenceTransformer
            with torch.no_grad():
                target = model.embedding_model.encode(batch_sentences, convert_to_tensor=True).detach().clone().to(device)

            recon_batch, mu, logvar = model(batch_sentences)
            loss = vae_loss(recon_batch, target, mu, logvar, beta=0.1)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Step
        model.eval()
        val_mse_sum = 0
        with torch.no_grad():
            for batch_sentences in val_loader:
                target = model.embedding_model.encode(batch_sentences, convert_to_tensor=True).detach().clone().to(device)
                recon_batch, mu, logvar = model(batch_sentences)
                val_mse_sum += vae_loss(recon_batch, target, mu, logvar, beta=0.1).item()
        val_rmse = math.sqrt(val_mse_sum / len(val_loader))
        train_rmse = math.sqrt(train_loss / len(train_loader))

        print(f"Epoch {epoch+1} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

    torch.save(model.state_dict(), "weights/vae_highschool_weights.pt")
    print("Training complete. Weights saved.")

if __name__ == "__main__":
    train()