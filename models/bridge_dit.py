# Core tensor operations and GPU support.
import torch
# Neural network layers and module utilities.
import torch.nn as nn
# Mathematical helpers used for sinusoidal embeddings.
import math


# Apply adaptive affine modulation to normalized activations.
def modulation(x, shift, scale):
    # Keep identity behavior when scale is zero by using (1 + scale).
    return x * (1 + scale) + shift


# One conditioned transformer-style block used in the bridge diffusion model.
class DiTBlock(nn.Module):
    # Create attention + MLP sublayers with AdaLN modulation parameters.
    def __init__(self, hidden_size, n_head):
        # Initialize nn.Module internals.
        super().__init__()
        # First normalization before the attention branch.
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Self-attention over the sequence dimension (sequence length is 1 here).
        self.attn = nn.MultiheadAttention(hidden_size, n_head, batch_first=True)
        # Second normalization before the MLP branch.
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Two-layer feed-forward network for residual transformation.
        self.mlp = nn.Sequential(
            # Expand hidden width for richer nonlinear mixing.
            nn.Linear(hidden_size, 4 * hidden_size),
            # Apply smooth nonlinearity.
            nn.GELU(),
            # Project back to the model hidden size.
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
        # Convert combined context (time + anchors + position) into
        # shifts/scales/gates for both attention and MLP branches.
        self.adaLN_modulation = nn.Sequential(
            # Nonlinear activation before generating modulation values.
            nn.SiLU(),
            # Produce 6 vectors: shift/scale/gate for MSA and MLP.
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    # Run one conditioned block update on the hidden state.
    def forward(self, x, c):
        # x is shaped (batch, seq_len=1, hidden_size).
        # c is conditioning shaped (batch, hidden_size) or (batch, 1, hidden_size).
        
        # Squeeze optional singleton sequence dimension from context.
        if c.dim() == 3:
            c = c.squeeze(1)

        # Generate all AdaLN parameters, then split into six chunks.
        mod = self.adaLN_modulation(c).chunk(6, dim=1)
        # Unpack shift/scale/gate tensors for attention and MLP paths.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        
        # Add sequence dimension so tensors broadcast with x.
        shift_msa = shift_msa.unsqueeze(1)
        # Add sequence dimension for attention scale.
        scale_msa = scale_msa.unsqueeze(1)
        # Add sequence dimension for attention residual gate.
        gate_msa = gate_msa.unsqueeze(1)
        # Add sequence dimension for MLP shift.
        shift_mlp = shift_mlp.unsqueeze(1)
        # Add sequence dimension for MLP scale.
        scale_mlp = scale_mlp.unsqueeze(1)
        # Add sequence dimension for MLP residual gate.
        gate_mlp = gate_mlp.unsqueeze(1)

        # Normalize features before attention.
        norm_x = self.norm1(x)
        # Inject context via adaptive shift/scale modulation.
        modulated_x = modulation(norm_x, shift_msa, scale_msa)
        
        # Compute attention residual on the conditioned token.
        res_attn, _ = self.attn(modulated_x, modulated_x, modulated_x)
        # Add gated attention residual to hidden state.
        x = x + gate_msa * res_attn
        
        # Normalize updated state before the MLP branch.
        norm_x2 = self.norm2(x)
        # Apply MLP-specific adaptive modulation.
        modulated_x2 = modulation(norm_x2, shift_mlp, scale_mlp)
        # Compute MLP residual transformation.
        res_mlp = self.mlp(modulated_x2)
        # Add gated MLP residual to hidden state.
        x = x + gate_mlp * res_mlp
        
        # Return block output with same shape as input x.
        return x


# Diffusion transformer that predicts the target latent noise from bridge context.
class BridgeDiT(nn.Module):
    # Build all embedding, conditioning, transformer, and output layers.
    def __init__(self, latent_dim=128, hidden_size=512, depth=8, n_head=8):
        # Initialize nn.Module internals.
        super().__init__()
        # Store hidden size for consistent reshaping and embeddings.
        self.hidden_size = hidden_size

        # Project noisy latent input into model hidden space.
        self.x_embedder = nn.Linear(latent_dim, hidden_size)
        # Project concatenated start/end anchor latents into conditioning space.
        self.anchor_projector = nn.Linear(latent_dim * 2, hidden_size)
        
        # Learned projection stack for sinusoidal timestep embeddings.
        self.t_embedder = nn.Sequential(
            # First linear transform on timestep embedding.
            nn.Linear(hidden_size, hidden_size),
            # Nonlinearity for richer conditioning.
            nn.SiLU(),
            # Second linear transform to final timestep context.
            nn.Linear(hidden_size, hidden_size)
        )

        # Project scalar relative position into hidden conditioning space.
        self.pos_projector = nn.Linear(1, hidden_size)

        # Create the stack of conditioned transformer blocks.
        self.blocks = nn.ModuleList([
            # Instantiate one DiT block per depth level.
            DiTBlock(hidden_size, n_head) for _ in range(depth)
        ])

        # Map final hidden representation back to latent dimension.
        self.final_layer = nn.Linear(hidden_size, latent_dim)

    # Predict noise for the target latent using noisy latent + bridge context.
    def forward(self, x_noisy, t, z_start, z_end, rel_pos):
        # Embed noisy latent and create sequence dimension expected by attention.
        x = self.x_embedder(x_noisy).view(-1, 1, self.hidden_size)

        # Concatenate start and end anchors along feature dimension.
        anchors = torch.cat([z_start, z_end], dim=-1)
        # Project anchor pair into conditioning vector.
        c_anchors = self.anchor_projector(anchors)
        
        # Build timestep conditioning from sinusoidal embedding.
        t_emb = self.t_embedder(self.timestep_embedding(t, self.hidden_size))
        
        # Ensure relative position is shaped (batch, 1) before projection.
        p_emb = self.pos_projector(rel_pos.view(-1, 1)) 
        
        # Combine all context sources into one conditioning vector.
        c = c_anchors + t_emb + p_emb
        # Ensure conditioning is 2D (batch, hidden_size).
        c = c.view(-1, self.hidden_size)

        # Apply each conditioned DiT block in sequence.
        for block in self.blocks:
            # Update hidden state with attention and MLP residuals.
            x = block(x, c)

        # Project back to latent noise; keeps current output width assumption of 128.
        return self.final_layer(x).view(-1, 128)

    # Create sinusoidal timestep embedding used by diffusion models.
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        # Reserve half channels for cosine and half for sine.
        half = dim // 2
        # Compute exponentially spaced frequencies.
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        # Multiply each timestep by all frequencies to get phase angles.
        args = timesteps[:, None].float() * freqs[None]
        # Concatenate cosine and sine values into one embedding.
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # If dim is odd, append one zero channel so output matches requested dim.
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # Return final timestep embedding tensor.
        return embedding

    def get_model_stats(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total