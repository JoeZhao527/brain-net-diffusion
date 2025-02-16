from functools import partial
from inspect import isfunction

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        beta_end=0.1,
        diff_steps=100,
        loss_type="l2",
        betas=None,
        beta_schedule="linear",
        guidance_level=0.4,
        contrastive_emb: bool = True
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.__scale = None
        self.guidance_level = guidance_level
        self.contrastive_emb = contrastive_emb

        print(f"Use contrastive embedding: {self.contrastive_emb} ({type(self.contrastive_emb)})")

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        noise, _ = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=noise
        )
        
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )

        noise = noise_like(x.shape, device, repeat_noise)
        # noise = noise_sampler(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # print(f"noise: {torch.mean(model_mean).item():.3f}, {torch.mean(model_log_variance).item():.3f}, {torch.mean(noise).item():.3f}")
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )
        return img
    
    @torch.no_grad()
    def p_sample_loop_wi_seed(self, seed: torch.Tensor, cond, guidance=None):
        device = self.betas.device

        img = seed
        b = seed.shape[0]

        guidance_level = self.guidance_level if guidance == None else guidance
        print(f"guidance: {guidance_level}")
        # add noise to seed
        t0 = int(self.num_timesteps * (1 - guidance_level))
        time = torch.full((b,), t0 + 1, device=device, dtype=torch.long)

        if t0 == self.num_timesteps:
            # img = noise_sampler(seed, direction=False)
            img = torch.randn(seed.shape, device=seed.device)
        else:
            img = self.q_sample(img, time)

        # Start denosing from gauidance started step
        # print(f"seed mean: {torch.mean(seed).item()}")
        # print(f"noise mean: {torch.mean(img).item()}")
        for i in reversed(range(0, t0)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )
            # print(f"noise mean at {i}: {torch.mean(img).item()}")

        return img

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None):
        shape = sample_shape
        x_hat = self.p_sample_loop(shape, cond)  # TODO reshape x_hat to (B,T,-1)

        if self.scale is not None:
            x_hat *= self.scale
        return x_hat
    
    @torch.no_grad()
    def sample_wi_seed(self, x: torch.Tensor, cond=None, guidance=None):
        x_hat = self.p_sample_loop_wi_seed(x, cond, guidance)  # TODO reshape x_hat to (B,T,-1)

        if self.scale is not None:
            x_hat *= self.scale
        return x_hat

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # noise = default(noise, noise_sampler(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _contrastive_loss(self, latents, cond, temperature=0.1, scale=0.01):
        batch_size = latents.size(0)
        # Normalize the latents
        latents = F.normalize(latents, p=2, dim=1)
        
        # Compute the pairwise similarity matrix
        sim_mtx = torch.matmul(latents, latents.T) / temperature
        
        # Mask to remove self-similarity scores
        mask = torch.eye(batch_size, dtype=torch.bool, device=latents.device)
        sim_mtx.masked_fill_(mask, -float('inf'))

        # Create positive pair mask
        cond = cond.unsqueeze(0)
        pos_mask = (cond == cond.T) ^ mask

        # Compute log_softmax over similarity matrix
        log_prob = - F.log_softmax(sim_mtx, dim=1)

        # Compute the loss
        positive_log_prob = log_prob[pos_mask].sum() / batch_size

        return scale * positive_log_prob
    
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon, cond_loss = self.denoise_fn(x_noisy, t, cond=cond)

        # nn.Embedding(num_classes, hidden_size)
        emb_dist = self.denoise_fn.get_label_emb_loss()

        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        # print(f"{emb_dist.item():.3f}, {emb_size.item():.3f}, {loss.item():.3f}")
        # loss = emb_dist + emb_size + loss
        loss_dict = {
            "recon_loss": loss,
            "ccl_loss": cond_loss + emb_dist,
        }

        if self.contrastive_emb:
            loss_dict["loss"] = loss + cond_loss + emb_dist
        else:
            loss_dict["loss"] = loss
        
        return loss_dict

    def log_prob(self, x, cond, *args, **kwargs):
        if self.scale is not None:
            x /= self.scale

        time = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        loss = self.p_losses(x, cond, time, *args, **kwargs)

        return loss