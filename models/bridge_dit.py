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
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)

    def forward(self, x, c):
        if c.dim() == 3:
            c = c.squeeze(1)

        mod = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)

        norm_x = self.norm1(x)
        modulated_x = modulation(norm_x, shift_msa, scale_msa)
        
        res_attn, _ = self.attn(modulated_x, modulated_x, modulated_x)
        x = x + gate_msa * res_attn
        
        norm_x2 = self.norm2(x)
        modulated_x2 = modulation(norm_x2, shift_mlp, scale_mlp)
        res_mlp = self.mlp(modulated_x2)
        x = x + gate_mlp * res_mlp
        
        return x


class BridgeDiT(nn.Module):
    def __init__(self, latent_dim=128, hidden_size=512, depth=8, n_head=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

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

    def forward(self, x_noisy, t, z_start, z_end, rel_pos):
        x = self.x_embedder(x_noisy).view(-1, 1, self.hidden_size)

        anchors = torch.cat([z_start, z_end], dim=-1)
        c_anchors = self.anchor_projector(anchors)
        
        t_emb = self.t_embedder(self.timestep_embedding(t, self.hidden_size))
        
        p_emb = self.pos_projector(rel_pos.view(-1, 1))
        
        c = c_anchors + t_emb + p_emb
        c = c.view(-1, self.hidden_size)

        for block in self.blocks:
            x = block(x, c)

        return self.final_layer(x).view(-1, self.latent_dim)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def get_model_stats(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total