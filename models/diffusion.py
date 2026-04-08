import torch
import torch.nn as nn
import math

class DiffusionEngine:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        
        # 1. Cosine Schedule: alphas_cumprod (alpha_bar)
        # f(t) = cos(((t/T + s) / (1 + s)) * pi/2)^2
        def f(t, T, s=0.008):
            return torch.cos(((t / T + s) / (1 + s)) * math.pi / 2)**2

        t = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = f(t, timesteps) / f(torch.tensor([0.0]), timesteps)
        
        # 2. Derive betas from alphas_cumprod
        # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999).to(device)
        
        # 3. Precompute essential values
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def add_noise(self, x_start, t):
        # Forward Process (same logic, now with cosine-derived values)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        noise = torch.randn_like(x_start)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise

    @torch.no_grad()
    def sample(self, model, shape, mask=None):
        model.eval()
        x = torch.randn(shape).to(self.device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # Reverse step formula
            coeff = (1 / torch.sqrt(alpha))
            noise_coeff = (beta / torch.sqrt(1 - alpha_cumprod))
            x = coeff * (x - noise_coeff * predicted_noise) + torch.sqrt(beta) * noise
            
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                
        return x