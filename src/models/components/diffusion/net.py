import torch
from torch import nn
from src.models.components.diffusion.gaussian_diffusion import GaussianDiffusion
from src.utils.state_dict import load_from_dir


class VQGAELatentDiffusion(nn.Module):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        pretrain_ckpt: str,
        pretrain_model: nn.Module,
        distribution_match: bool = True,
        **kwargs
    ):
        super().__init__()

        self.vae_encoder = load_from_dir(pretrain_model, pretrain_ckpt)
        for params in self.vae_encoder.parameters():
            params.requires_grad = False

        self.diffusion = diffusion_model

        self.distribution_match = distribution_match

        print(f"Use distribution match: {self.distribution_match} ({type(self.distribution_match)})")
        self.loss = nn.L1Loss(reduction="mean")

    def normalize_latent(self, latent):
        return (latent - torch.mean(latent)) / torch.std(latent)
    
    def customize_latent(self, latent, mu, std):
        return std * latent + mu
    
    def forward(self, edge_mtx: torch.Tensor, condition: torch.Tensor, label: torch.Tensor, **kwargs):
        # Diffusion
        latent = self.vae_encoder.encode_feat(edge_mtx)

        # normalize latent distribution to standard gaussian
        if self.distribution_match:
            latent = self.normalize_latent(latent)

        loss = self.diffusion.log_prob(latent, torch.max(label, dim=1)[1])

        return loss, None, edge_mtx
    
    def reconstruct(self, latent: torch.Tensor):
        latent = self.vae_encoder.quantize(latent)
        _, reconstruct = self.vae_encoder._decoder(latent)

        b, n, _ = reconstruct.shape
        indices = torch.arange(n, device=reconstruct.device)
        reconstruct[:, indices, indices] = 1

        return latent, reconstruct
    
    @torch.no_grad()
    def sample_wi_seed(self, edge_mtx, condition, label: torch.Tensor, guidance=None, **kwargs):
        # encoder latent space
        latent = self.vae_encoder.encode_feat(edge_mtx)

        # normalize latent space to gaussian distribution
        if self.distribution_match:
            mu, std = torch.mean(latent), torch.std(latent)
            latent = self.normalize_latent(latent)

        # generation perform in gaussian distribution space
        latent = self.diffusion.sample_wi_seed(latent, torch.max(label, dim=1)[1], guidance)

        # scale latent to its original distribution
        if self.distribution_match:
            latent = self.customize_latent(latent, mu, std)

        # reconstruct
        latent, recons = self.reconstruct(latent)

        return latent, recons