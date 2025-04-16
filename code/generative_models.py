
from typing import Literal
from torch.distributions import Normal, kl_divergence
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, out_features, activation: Literal["relu", "sigmoid", "identity"]):
        super().__init__()
        self.mean = nn.Linear(in_features, out_features)
        self.activation = activation
        self.reset_parameters(in_features, out_features)
     
    
    def _forward(self, x, weight, bias):
      x = F.linear(x, weight, bias)
      match self.activation:
        case "relu":
            return F.relu(x)
        case "sigmoid":
            return F.sigmoid(x)
        case _:
            return x
            
    def forward(self, x):
        weight = self.mean.weight
        bias = self.mean.bias
        return self._forward(x, weight, bias)

    def reset_parameters(self, in_features, out_features, constant=1.0):
        scale = constant * np.sqrt(6.0 / (in_features + out_features))
        nn.init.uniform_(self.mean.weight, -scale, scale)
        nn.init.zeros_(self.mean.bias)


class MeanFieldMlp(Mlp):
    def __init__(self, in_features, out_features, activation: Literal["relu", "sigmoid", "identity"]):
        super(Mlp, self).__init__()
        self.mean = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        self.activation = activation
        self.prior_mean = nn.Linear(in_features, out_features)
        self.prior_log_sigma = nn.Linear(in_features, out_features)
        self.reset_parameters(in_features, out_features)
        self.initialize_prior()

    def forward(self, x: torch.Tensor):
      weight_dist = Normal(self.mean.weight, torch.exp(self.log_sigma.weight))
      bias_dist = Normal(self.mean.bias, torch.exp(self.log_sigma.bias))

      weight = weight_dist.rsample()
      bias = bias_dist.rsample()

      return super(MeanFieldMlp, self)._forward(x, weight, bias)

    def reset_parameters(self, in_features, out_features, constant=1.0):
      scale = constant * np.sqrt(6.0 / (in_features + out_features))
      nn.init.uniform_(self.mean.weight, -scale, scale)
      nn.init.zeros_(self.mean.bias)
      # initialize logvars
      nn.init.constant_(self.log_sigma.weight, -6.0)
      nn.init.constant_(self.log_sigma.bias, -6.0)

    def initialize_prior(self):
      nn.init.constant_(self.prior_mean.weight, 0.0)
      nn.init.constant_(self.prior_mean.bias, 0.0)
      nn.init.constant_(self.prior_log_sigma.weight, 0.0)
      nn.init.constant_(self.prior_log_sigma.bias, 0.0)
      self.prior_mean.weight.requires_grad = False
      self.prior_mean.bias.requires_grad = False
      self.prior_log_sigma.weight.requires_grad = False
      self.prior_log_sigma.bias.requires_grad = False

    def advance_prior(self):
      self.prior_mean.weight.copy_(self.mean.weight.detach().clone())
      self.prior_mean.bias.copy_(self.mean.bias.detach().clone())
      self.prior_log_sigma.weight.copy_(self.log_sigma.weight.detach().clone())
      self.prior_log_sigma.bias.copy_(self.log_sigma.bias.detach().clone())
      # log sigma correction from the paper
      with torch.no_grad():
        self.log_sigma.weight.fill_(-6.0)
        self.log_sigma.bias.fill_(-6.0)
      # ensure we do not optimize the prior
      self.prior_mean.weight.requires_grad = False
      self.prior_mean.bias.requires_grad = False
      self.prior_log_sigma.weight.requires_grad = False
      self.prior_log_sigma.bias.requires_grad = False

    def kl(self):
      # weight dists
      variational_weight = Normal(
        loc=self.mean.weight,
        scale=torch.exp(self.log_sigma.weight)
      )
      prior_weight = Normal(
        loc=self.prior_mean.weight,
        scale=torch.exp(self.prior_log_sigma.weight)
      )
      # bias dists
      variational_bias= Normal(
        loc=self.mean.bias,
        scale=torch.exp(self.log_sigma.bias)
      )
      prior_bias = Normal(
        loc=self.prior_mean.bias,
        scale=torch.exp(self.prior_log_sigma.bias)
      )
      # mean field
      return (
        kl_divergence(variational_weight, prior_weight).sum() + 
        kl_divergence(variational_bias, prior_bias).sum()   
      )

class NaiveDecoderTail(nn.Sequential):
    def __init__(self, hidden_dim: int, output_dim: int):
        super(NaiveDecoderTail, self).__init__(
            Mlp(hidden_dim, hidden_dim, "relu"),
            Mlp(hidden_dim, output_dim, "sigmoid")
        )

class MeanFieldDecoderTail(nn.Sequential):
    def __init__(self, hidden_dim: int, output_dim: int):
      super(MeanFieldDecoderTail, self).__init__(
          MeanFieldMlp(hidden_dim, hidden_dim, "relu"),
          MeanFieldMlp(hidden_dim, output_dim, "sigmoid")
      )
  
    def kl(self):
      kl = 0.0
      for mlp in self:
        kl += mlp.kl()
      return kl
  
    def advance_prior(self):
      for mlp in self:
        mlp.advance_prior()

class Encoder(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
      super(Encoder, self).__init__(
          Mlp(input_dim, hidden_dim, "relu"),
          Mlp(hidden_dim, hidden_dim, "relu"),
          Mlp(hidden_dim, hidden_dim, "relu"),
          Mlp(hidden_dim, 2*latent_dim, "identity"),
      )
    def forward(self, x: torch.Tensor) -> Normal:
      [mean, log_sigma] = super(Encoder, self).forward(x).chunk(2, dim=-1)
      return Normal(
        loc=mean,
        scale=torch.exp(log_sigma)
      )

NUM_TRAIN_MONTE_CARLO_SAMPLES = 10

class VAE(nn.Module):
  def __init__(
      self,
      num_tasks: int,
      input_dim: int,
      latent_dim: int,
      hidden_dim: int,
      decoder_tail: NaiveDecoderTail | MeanFieldDecoderTail,
  ):
    super(VAE, self).__init__()
    self.encoders = nn.ModuleList(
      Encoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
      )
      for _ in range(num_tasks)
    )
    self.decoder_heads = nn.ModuleList(
      nn.Sequential(
          Mlp(in_features=latent_dim, out_features=hidden_dim, activation="relu"),
          Mlp(in_features=hidden_dim, out_features=hidden_dim, activation="relu")
      ) for _ in range(num_tasks)
    )
    self.decoder_tail = decoder_tail
    self.latent_dim = latent_dim

  def decode(self, z, head_idx: int):
    h = self.decoder_heads[head_idx](z)
    x = self.decoder_tail(h)
    return x
  
  def sample(self, num_samples: int, head_idx: int, device: torch.device):
    p_z = Normal(
        torch.zeros((self.latent_dim, ), device=device), 
        torch.ones((self.latent_dim, ), device=device)
    )
    z = p_z.rsample((num_samples, )) # [num_samples, latent_dim]
    x = self.decode(z, head_idx)
    return x

  @torch.no_grad()
  def sample_grid(self, num_samples: int, head_idx: int, device):
    from torchvision.transforms.functional import to_pil_image
    from torchvision.utils import make_grid
    
    x = (
      self.sample(num_samples, head_idx, device)
        .view((-1, 1, 28, 28))
        .cpu()
    )
    return to_pil_image(
      make_grid(x, nrow=int(np.sqrt(num_samples)))
    )

  def ll(self, x: torch.Tensor, q: Normal, head_idx:int, num_mc_samples=NUM_TRAIN_MONTE_CARLO_SAMPLES):
      log_likelihood = 0.0
      for _ in range(num_mc_samples):
          log_likelihood += -F.binary_cross_entropy(
            input=self.decode(q.rsample(), head_idx),
            target=x,
            reduction="none"
          ).sum(dim=-1) / num_mc_samples
      return log_likelihood # [b, ]
  
  def elbo(self, x, head_idx: int, num_mc_samples: int = NUM_TRAIN_MONTE_CARLO_SAMPLES):
      q_z_given_x = self.encoders[head_idx](x)
      p_z = Normal(
        torch.zeros_like(q_z_given_x.loc),
        torch.ones_like(q_z_given_x.scale)
      )
      kl = kl_divergence(q_z_given_x, p_z).sum(dim=-1) # [b, ]
      ll = self.ll(x, q_z_given_x, head_idx, num_mc_samples)  # [b, ] Monte Carlo
      return (ll - kl) # [b, ]
    
  def forward(self, x, head_idx: int, num_mc_samples: int = NUM_TRAIN_MONTE_CARLO_SAMPLES):
    return self.elbo(x, head_idx, num_mc_samples)
  
  def learnable_parameters(self):
    return {
      name: param
      for (name, param) in self.named_parameters()
      if param.requires_grad
    }.items()

  def shared_learnable_parameters(self):
    return {
      name: param
      for (name, param) in self.decoder_tail()
      if param.requires_grad
    }.items()