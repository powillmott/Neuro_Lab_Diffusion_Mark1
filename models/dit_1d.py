import torch
import torch.nn as nn
import math

def modulation(x, shift, scale):
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_head, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Pointwise MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
        # Modulation layers for Diffusion Timestep
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c is the embedding of the current diffusion timestep
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention Layer
        x = x + gate_msa.unsqueeze(1) * self.attn(modulation(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)), 
                                                  modulation(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)), 
                                                  modulation(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)))[0]
        
        # MLP Layer
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulation(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)))
        return x

class DiT1D(nn.Module):
    def __init__(self, input_dim=128, hidden_size=512, depth=8, n_head=8, max_steps=16):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Map VAE latents to Transformer hidden size
        self.x_embeder = nn.Linear(input_dim, hidden_size)
        
        # 2. Positional Embeddings (The "Shape" of the movement)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_steps, hidden_size))
        
        # 3. Diffusion Timestep Embedder (Time in the diffusion process)
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 4. The Transformer Layers
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_head) for _ in range(depth)
        ])
        
        # 5. Output Layer
        self.final_layer = nn.Linear(hidden_size, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed with standard sine/cosine formula
        pos_embed = torch.zeros(16, self.hidden_size)
        position = torch.arange(0, 16).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2) * -(math.log(10000.0) / self.hidden_size))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

    def forward(self, x, t):
        """
        x: (Batch, Seq_Len, Input_Dim) - Noisy latent trajectory
        t: (Batch,) - Diffusion timesteps
        """
        # Embed input and add position
        x = self.x_embeder(x) + self.pos_embed 
        
        # Embed diffusion time (using a simple sinusoidal embedding helper is usually better, 
        # but for simplicity we'll assume t is already embedded or use a linear layer)
        t_freq = self.timestep_embedding(t, self.hidden_size)
        c = self.t_embedder(t_freq)
        
        for block in self.blocks:
            x = block(x, c)
            
        return self.final_layer(x)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding