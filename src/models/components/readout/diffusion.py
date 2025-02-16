import torch
from torch import nn
from typing import List, Literal
from src.utils.state_dict import load_from_dir
from src.models.components.diffusion import VQGAELatentDiffusion


class InferencePredictor(nn.Module):
    def __init__(
        self,
        diffusion_ckpt: str,
        diffusion_model: VQGAELatentDiffusion,
        condition_mode: Literal['identical', 'opposite', 'union'],
        label_key: str = "DX_GROUP",
        sample_num: int = 3,
        guidance_level: List[float] = [0.5],
        *kwargs
    ) -> None:
        super().__init__()

        assert condition_mode in ['identical', 'opposite', 'union']

        self.diffusion_model = load_from_dir(diffusion_model, diffusion_ckpt, skip_prefix=['net'])
        for params in self.diffusion_model.parameters():
            params.requires_grad = False
        # self.diffusion_model.diffusion.guidance_level = guidance_level
        self.guidance_level = guidance_level

        self.label_key = label_key
        self.condition_mode = condition_mode
        self.sample_num = sample_num

        print(f"Condition Mode: {self.condition_mode}")

    def get_flattened_edge(self, edge_mtx: torch.Tensor):
        # Flatten the edge matrix
        b, n, _ = edge_mtx.shape
        mask = torch.tril(torch.ones(n, n), diagonal=-1).bool()
        edge = edge_mtx[:, mask].reshape(b, -1)

        return edge
    
    def forward(
        self,
        edge_mtx: torch.Tensor,
        condition: torch.Tensor,
        condition_key: List[List[str]],
        label: torch.Tensor,
        **kwargs
    ):
        """
        codition: (batch size, condition size)
        """
        b, n, n = edge_mtx.shape

        # Inference for original latent
        h = self.diffusion_model.vae_encoder.encode_feat(edge_mtx)

        """With Label Editing"""
        samples = [edge_mtx]
        labels = [label]
        latents = [h]

        for _ in range(self.sample_num):
            for gd_level in self.guidance_level:
                if self.condition_mode in ["identical", "union"]:
                    # sample: (batch size, node num, node num)
                    h, sample = self.diffusion_model.sample_wi_seed(edge_mtx, label, label, gd_level)

                    # Record predicted samples
                    samples.append(sample)
                    labels.append(label)
                    latents.append(h)

                if self.condition_mode in ["opposite", "union"]:
                    _label = 1 - label
                    # sample: (batch size, node num, node num)
                    h, sample = self.diffusion_model.sample_wi_seed(edge_mtx, _label, _label, gd_level)

                    # Record predicted samples
                    samples.append(sample)
                    labels.append(_label)
                    latents.append(h)

        return label, torch.stack(samples, dim=1), torch.stack(labels, dim=1), torch.stack(latents, dim=1)