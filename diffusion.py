import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(values, t, x_shape):
    """Gather schedule coefficients for a batch of timesteps and reshape for broadcasting."""
    out = values.gather(0, t).float()
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM forward process, noise-prediction loss, and ancestral sampler.

    Training uses the simplified objective from Ho et al.: predict the noise
    added at a uniformly sampled timestep. Images are expected in [-1, 1].
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # Posterior q(x_{t-1} | x_t, x_0) variance from the DDPM paper.
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x0, t, noise=None):
        """Diffuse x0 to timestep t: x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def predict_x0_from_noise(self, x_t, t, noise):
        return (
            x_t - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * noise
        ) / _extract(self.sqrt_alphas_cumprod, t, x_t.shape)

    def q_posterior_mean(self, x0, x_t, t):
        return (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

    def p_losses(self, model, x0):
        """Simplified noise-prediction MSE, with t drawn uniformly from 0..T-1."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        predicted = model(x_t, t)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t_index):
        b = x_t.shape[0]
        t = torch.full((b,), t_index, device=x_t.device, dtype=torch.long)
        predicted_noise = model(x_t, t)
        x0 = self.predict_x0_from_noise(x_t, t, predicted_noise).clamp(-1.0, 1.0)
        mean = self.q_posterior_mean(x0, x_t, t)
        if t_index == 0:
            return mean
        variance = _extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, model, shape, device=None):
        """Draw samples by running the reverse chain from N(0, I) down to t = 0."""
        if device is None:
            device = next(model.parameters()).device
        image = torch.randn(shape, device=device)
        for t_index in reversed(range(self.timesteps)):
            image = self.p_sample(model, image, t_index)
        return image
