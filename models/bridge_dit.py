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
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
        # This takes the combined context (Time + Anchors + Pos) 
        # and turns it into specific shifts/scales for this layer
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x is (B, 1, H)
        # c is (B, H)
        
        # Ensure c is 2D (B, H)
        if c.dim() == 3:
            c = c.squeeze(1)

        mod = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        
        # Explicitly make these (B, 1, H)
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)

        # Attention
        norm_x = self.norm1(x) # (B, 1, H)
        modulated_x = modulation(norm_x, shift_msa, scale_msa)
        
        # attn wants (B, S, H). modulated_x is (B, 1, H)
        res_attn, _ = self.attn(modulated_x, modulated_x, modulated_x)
        x = x + gate_msa * res_attn
        
        # MLP
        norm_x2 = self.norm2(x)
        modulated_x2 = modulation(norm_x2, shift_mlp, scale_mlp)
        res_mlp = self.mlp(modulated_x2)
        x = x + gate_mlp * res_mlp
        
        return x

class BridgeDiT(nn.Module):
    def __init__(self, latent_dim=128, hidden_size=512, depth=8, n_head=8):
        super().__init__()
        self.hidden_size = hidden_size

        self.x_embedder = nn.Linear(latent_dim, hidden_size)
        self.anchor_projector = nn.Linear(latent_dim * 2, hidden_size)
        
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.pos_projector = nn.Linear(1, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_head) for _ in range(depth)
        ])

        self.final_layer = nn.Linear(hidden_size, latent_dim)


    # In BridgeDiT.forward:
    def forward(self, x_noisy, t, z_start, z_end, rel_pos):
        x = self.x_embedder(x_noisy).view(-1, 1, self.hidden_size)

        anchors = torch.cat([z_start, z_end], dim=-1)
        c_anchors = self.anchor_projector(anchors)
        
        t_emb = self.t_embedder(self.timestep_embedding(t, self.hidden_size))
        
        # Ensure rel_pos is (B, 1) before projecting
        p_emb = self.pos_projector(rel_pos.view(-1, 1)) 
        
        c = c_anchors + t_emb + p_emb
        c = c.view(-1, self.hidden_size)

        for block in self.blocks:
            x = block(x, c)

        return self.final_layer(x).view(-1, 128)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding