from typing import Literal, Optional, Tuple
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from einops import rearrange, repeat

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import mse_loss_onehot

class DiscriminativeNaive(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        output_dim: int,
        hidden_dim: int,
        num_hidden: int,
        num_heads: int,
        objective: Literal["classification", "regression"] = "classification"
    ):
        super(DiscriminativeNaive, self).__init__()
        self.backbone = nn.ModuleList([
             nn.Linear(
                in_features=input_dim if hidden_idx == 0 else hidden_dim,
                out_features=hidden_dim
             )
             for hidden_idx in range(num_hidden)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(
                in_features=hidden_dim,
                out_features=output_dim
            )
            for _ in range(num_heads)
        ])
        self.reset_parameters()
        self.num_heads = len(self.heads)
        self.output_dim = output_dim
        self.objective : Literal["classification", "regression"] = objective
    
    # to be compatible with evals
    def predict(self, x, head_idx: int, num_samples: int):
       return self.forward(x, head_idx)
    
    def forward(self, x, head_idx: int):
        x = x
        for layer in self.backbone:
          x = layer(x)
          x = F.relu(x)
        x = self.heads[head_idx](x)
        return x

    def reset_parameters(self):
      for layer in (self.backbone + self.heads):
        torch.nn.init.normal_(layer.weight, mean=0, std=0.1) # init from https://arxiv.org/pdf/1905.02099
        torch.nn.init.normal_(layer.bias, mean=0, std=0.1) # init from https://arxiv.org/pdf/1905.02099
  
    def learnable_parameters(self):
      return {
        param_name: param
        for (param_name, param) in self.named_parameters()
        if param.requires_grad
      }.items()


class DiscriminativeMeanField(nn.Module):
    def __init__(
      self,
      input_dim: int,
      output_dim: int,
      num_layers: int,
      hidden_dim: int,
      num_heads: int,
      init_prior_mean: float = 0.0,
      init_prior_var: float = 1.0,
      objective: Literal["classification", "regression"] = "classification"
    ):
        super(DiscriminativeMeanField, self).__init__()
        # Variational approximation
        self.input_proj_weights_params = nn.Parameter(
            torch.empty((input_dim, 2 * hidden_dim))
        )
        self.input_proj_biases_params = nn.Parameter(
            torch.empty((2 * hidden_dim, ))
        )
        self.backbone_weights_params = nn.Parameter(
            torch.empty((num_layers - 1, hidden_dim, 2 * hidden_dim))
        )
        self.backbone_biases_params = nn.Parameter(
            torch.empty((num_layers - 1, 2 * hidden_dim))
        )
        self.head_weights_params = nn.Parameter(
            torch.empty((num_heads, hidden_dim, 2 * output_dim))
        )
        self.head_biases_params = nn.Parameter(
            torch.empty((num_heads, 2 * output_dim))
        )
        # Prior
        self.register_buffer(
            "prior_input_proj_weights_params",
            torch.empty((input_dim, 2 * hidden_dim))
        )
        self.register_buffer(
            "prior_input_proj_biases_params",
            torch.empty((2 * hidden_dim, ))
        )
        self.register_buffer(
            "prior_backbone_weights_params",
            torch.empty((num_layers - 1, hidden_dim, 2 * hidden_dim))
        )
        self.register_buffer(
            "prior_backbone_biases_params",
            torch.empty((num_layers - 1, 2 * hidden_dim))
        )
        self.register_buffer(
            "prior_head_weights_params",
            torch.empty((num_heads, hidden_dim, 2 * output_dim))
        )
        self.register_buffer(
            "prior_head_biases_params",
            torch.empty((num_heads, 2 * output_dim))
        )
        self.num_heads = num_heads
        self.output_dim = output_dim
        # Initialize variational
        self._init_variational_params("input_proj_weights_params")
        self._init_variational_params("input_proj_biases_params")
        self._init_variational_params("backbone_weights_params")
        self._init_variational_params("backbone_biases_params")
        self._init_variational_params("head_weights_params")
        self._init_variational_params("head_biases_params")
        # Initialize prior
        self._init_prior_params("prior_input_proj_weights_params", init_prior_mean, init_prior_var)
        self._init_prior_params("prior_input_proj_biases_params", init_prior_mean, init_prior_var)
        self._init_prior_params("prior_backbone_weights_params", init_prior_mean, init_prior_var)
        self._init_prior_params("prior_backbone_biases_params", init_prior_mean, init_prior_var)
        self._init_prior_params("prior_head_weights_params", init_prior_mean, init_prior_var)
        self._init_prior_params("prior_head_biases_params", init_prior_mean, init_prior_var)
        self.objective: Literal["classification", "regression"]  = objective

    def _init_variational_params(self, params: str):
        [means, logvars] = getattr(self, params).chunk(2, dim=-1)
        means = means.clone()
        logvars = logvars.clone()
        torch.nn.init.normal_(means, mean=0, std=0.1) # init from https://arxiv.org/pdf/1905.02099
        torch.nn.init.constant_(logvars, math.log(1e-3)) # init from https://arxiv.org/pdf/1905.02099
        getattr(self, params).data.copy_(
            torch.cat([means, logvars], dim=-1)
        )

    def _init_prior_params(self, params: str, mean: float, var: float):
        [means, logvars] = getattr(self, params).chunk(2, dim=-1)
        means = means.clone()
        logvars = logvars.clone()
        torch.nn.init.constant_(means, val=mean)
        torch.nn.init.constant_(logvars, val=math.log(math.sqrt(var)))
        getattr(self, params).copy_(
            torch.cat([means, logvars], dim=-1)
        )

    def _reparameterize(self, means: torch.Tensor, logvars: torch.Tensor):
      stds = torch.exp(0.5*logvars)
      return means + stds * torch.randn_like(means)
    
    def _get_mean_field_param_dist(
        self,
        in_proj_weights_params,
        in_proj_biases_params,
        backbone_weights_params,
        backbone_biases_params,
        head_weights_params,
        head_biases_params,
        head_idx: Optional[int] = None,
        head_only: bool = False
    ) -> Normal:
      [in_proj_weight_means, in_proj_weight_logvars] = (
          in_proj_weights_params.chunk(2, dim=-1)
      )
      [in_proj_bias_means, in_proj_bias_logvars] = (
          in_proj_biases_params.chunk(2, dim=-1)
      )
      [backbone_weight_means, backbone_weight_logvars] = (
          backbone_weights_params.chunk(2, dim=-1)
      )
      [backbone_bias_means, backbone_bias_logvars] = (
          backbone_biases_params.chunk(2, dim=-1)
      )
      [head_weight_means, head_weight_logvars] = (
          head_weights_params.chunk(2, dim=-1)
      )
      [head_bias_means, head_bias_logvars] = (
          head_biases_params.chunk(2, dim=-1)
      )

      if head_only:
        assert head_idx is not None
        return Normal(
          loc=torch.cat(
              [
                head_weight_means[head_idx].flatten(),
                head_bias_means[head_idx].flatten()
              ], 
              dim=0
          ),
          scale=torch.exp(
            0.5 * torch.cat(
              [
                head_weight_logvars[head_idx].flatten(),
                head_bias_logvars[head_idx].flatten()
              ], 
              dim=0
            )
          )
        )
         
      if head_idx is not None:
        return Normal(
          loc=torch.cat(
              [
                in_proj_weight_means.flatten(),
                in_proj_bias_means.flatten(),
                backbone_weight_means.flatten(),
                backbone_bias_means.flatten(),
                head_weight_means[head_idx].flatten(),
                head_bias_means[head_idx].flatten()
              ], 
              dim=0
          ),
          scale=torch.exp(
            0.5 * torch.cat(
              [
                in_proj_weight_logvars.flatten(),
                in_proj_bias_logvars.flatten(),
                backbone_weight_logvars.flatten(),
                backbone_bias_logvars.flatten(),
                head_weight_logvars[head_idx].flatten(),
                head_bias_logvars[head_idx].flatten()
              ], 
              dim=0
            )
          )
        )
      else:
        return Normal(
          loc=torch.cat(
              [
                in_proj_weight_means.flatten(),
                in_proj_bias_means.flatten(),
                backbone_weight_means.flatten(),
                backbone_bias_means.flatten()
              ], 
              dim=0
          ),
          scale=torch.exp(
            0.5 * torch.cat(
              [
                in_proj_weight_logvars.flatten(),
                in_proj_bias_logvars.flatten(),
                backbone_weight_logvars.flatten(),
                backbone_bias_logvars.flatten()
              ], 
              dim=0
            )
          )
        )
    
    def advance_prior(self, head_idx: int):
      getattr(self, "prior_input_proj_weights_params").copy_(
        getattr(self, "input_proj_weights_params").clone().detach()
      )
      getattr(self, "prior_input_proj_biases_params").copy_(
        getattr(self, "input_proj_biases_params").clone().detach()
      )
      getattr(self, "prior_backbone_weights_params").copy_(
        getattr(self, "backbone_weights_params").clone().detach()
      )
      getattr(self, "prior_backbone_biases_params").copy_(
        getattr(self, "backbone_biases_params").clone().detach()
      )
      # Carefully reset head prior
      head_weight_params = getattr(self, "head_weights_params")[head_idx]
      prior_heads_weight_params = getattr(self, "prior_head_weights_params")
      prior_heads_weight_params[head_idx] = head_weight_params.clone().detach()
      getattr(self, "prior_head_weights_params").copy_(
        prior_heads_weight_params.clone().detach()
      )
      head_biases_params = getattr(self, "head_biases_params")[head_idx]
      prior_heads_biases_params = getattr(self, "prior_head_biases_params")
      prior_heads_biases_params[head_idx] = head_biases_params.clone().detach()
      getattr(self, "prior_head_biases_params").copy_(
        prior_heads_biases_params.clone().detach()
      )
  

    def kl(self, head_idx: Optional[int] = None, head_only: bool = False):
      variational = self._get_mean_field_param_dist(
        in_proj_weights_params=getattr(self, "input_proj_weights_params"),
        in_proj_biases_params=getattr(self, "input_proj_biases_params"),
        backbone_weights_params=getattr(self, "backbone_weights_params"),
        backbone_biases_params=getattr(self, "backbone_biases_params"),
        head_weights_params=getattr(self, "head_weights_params"),
        head_biases_params=getattr(self, "head_biases_params"),
        head_idx=head_idx,
        head_only=head_only
      )
      prior = self._get_mean_field_param_dist(
        in_proj_weights_params=getattr(self, "prior_input_proj_weights_params"),
        in_proj_biases_params=getattr(self, "prior_input_proj_biases_params"),
        backbone_weights_params=getattr(self, "prior_backbone_weights_params"),
        backbone_biases_params=getattr(self, "prior_backbone_biases_params"),
        head_weights_params=getattr(self, "prior_head_weights_params"),
        head_biases_params=getattr(self, "prior_head_biases_params"),
        head_idx=head_idx,
        head_only=head_only
      )
      assert variational.loc.shape == prior.loc.shape
      assert variational.scale.shape == prior.scale.shape
      return kl_divergence(variational, prior).sum(dim=-1)
    
    def _forward(
        self,
        x: torch.Tensor,
        head_idx: int,
        in_proj_params: Tuple[torch.Tensor],
        backbone_params: Tuple[torch.Tensor],
        head_params: Tuple[torch.Tensor],
    ):  
        [in_weights, in_biases] = in_proj_params
        [backbone_weights, backbone_biases] = backbone_params
        [head_weights, head_biases] = head_params
        x = F.linear(x, in_weights.T, in_biases)
        x = F.relu(x)
        for (weight, bias) in zip(backbone_weights, backbone_biases):
          x = F.linear(x, weight.T, bias)
          x = F.relu(x)
        [head_weight, head_bias] = [head_weights[head_idx], head_biases[head_idx]]
        x = F.linear(x, head_weight.T, head_bias)
        return x
    
    def predict(
      self, 
      x: torch.Tensor,
      head_idx: int,
      num_monte_carlo_samples: int,
    ):
      outputs = []
      for _ in range(num_monte_carlo_samples):
          outputs.append(
            self.forward(x, head_idx, deterministic=False).unsqueeze(1)
          )
      outputs = torch.cat(outputs, dim=1) # [b, k, d]
      match self.objective:
        case "classification":
          preds = F.softmax(outputs, dim=-1).mean(dim=1) # [b, d]
          return preds
        case "regression":
          return outputs.mean(dim=1)
    
    def mle_step(self, x: torch.Tensor, y: torch.Tensor, head_idx: int):
      logits = self.forward(x, head_idx, deterministic=True)
      match self.objective:
        case "classification":
          return F.cross_entropy(logits, y, reduction="none") # [b, ]
        case "regression":
          return mse_loss_onehot(logits, y, reduction="none", num_classes=self.output_dim).sum(dim=-1) # [b, ]
    
    def vcl_coreset_task_step(
      self, 
      x: torch.Tensor,
      y: torch.Tensor,
      task_idx: int,
      kl_weight: float,
      num_monte_carlo_samples: int
    ):
      k = num_monte_carlo_samples
      outputs = []
      head_idx = 0 if self.num_heads == 1 else task_idx
      for _ in range(num_monte_carlo_samples):
          outputs.append(
            self.forward(x, head_idx, deterministic=False).unsqueeze(1)
          )
      outputs = torch.cat(outputs, dim=1) # [b, k, d]
      outputs = rearrange(outputs, "b k d -> (b k) d")
      labels = repeat(y, "b -> (b k)", k=k)

      match self.objective:
        case "classification":
          return {
            "nll": rearrange(
              F.cross_entropy(outputs, labels, reduction="none"),
              "(b k) -> b k",
              k=k
            ).mean(dim=-1),
            "kl": kl_weight * self.kl(head_idx, head_only=True)     # [1, ]
          }
        case _:
          return {
            "nll": rearrange(
              mse_loss_onehot(outputs, labels, reduction="none", num_classes=self.output_dim).sum(dim=-1), # [(b k) d] -> (b k)
              "(b k) -> b k",
              k=k
            ).mean(dim=-1),
            "kl": kl_weight * self.kl(head_idx, head_only=True)     # [1, ]
          }
    
    def vcl_step(
      self,
      x: torch.Tensor,
      y: torch.Tensor,
      head_idx: int,
      kl_weight: float,
      num_monte_carlo_samples: int
    ):
      assert not self.prior_input_proj_weights_params.requires_grad, "prior_input_proj_weights_params requires grad!"
      assert not self.prior_input_proj_biases_params.requires_grad, "prior_input_proj_biases_params requires grad!"
      assert not self.prior_backbone_weights_params.requires_grad, "prior_backbone_weights_params requires grad!"
      assert not self.prior_backbone_biases_params.requires_grad, "prior_backbone_biases_params requires grad!"
      assert not self.prior_head_weights_params.requires_grad, "prior_head_weights_params requires grad!"
      assert not self.prior_head_biases_params.requires_grad, "prior_head_biases_params requires grad!"
      k = num_monte_carlo_samples
      
      outputs = []
      for _ in range(num_monte_carlo_samples):
          outputs.append(
            self.forward(x, head_idx, deterministic=False).unsqueeze(1)
          )
  
      outputs = torch.cat(outputs, dim=1) # [b, k, d]
      outputs = rearrange(outputs, "b k d -> (b k) d")
      labels = repeat(y, "b -> (b k)", k=k)

      match self.objective:
        case "classification":
          nll = rearrange(
            F.cross_entropy(outputs, labels, reduction="none"),
            "(b k) -> b k",
            k=k
          ).mean(dim=-1) # [b]
        case "regression":
          nll = rearrange(
            mse_loss_onehot(outputs, labels, reduction="none", num_classes=self.output_dim).sum(dim=-1), # (b k) d -> (b k)
            "(b k) -> b k",
            k=k
          ).mean(dim=-1) # [b]
      
      kl = self.kl(head_idx) # [1]      
      return nll.mean() + kl_weight * kl

    def forward(self, x: torch.Tensor, head_idx: int, deterministic: bool = False):
      [in_proj_weight_means, in_proj_weight_logvars] = (
          getattr(self, "input_proj_weights_params").chunk(2, dim=-1)
      )
      [in_proj_bias_means, in_proj_bias_logvars] = (
          getattr(self, "input_proj_biases_params").chunk(2, dim=-1)
      )
      [backbone_weight_means, backbone_weight_logvars] = (
          getattr(self, "backbone_weights_params").chunk(2, dim=-1)
      )
      [backbone_bias_means, backbone_bias_logvars] = (
          getattr(self, "backbone_biases_params").chunk(2, dim=-1)
      )
      [head_weight_means, head_weight_logvars] = (
          getattr(self, "head_weights_params").chunk(2, dim=-1)
      )
      [head_bias_means, head_bias_logvars] = (
          getattr(self, "head_biases_params").chunk(2, dim=-1)
      )
      if deterministic:
        return self._forward(
          x,
          head_idx,
          in_proj_params=[in_proj_weight_means, in_proj_bias_means],
          backbone_params=[backbone_weight_means, backbone_bias_means],
          head_params=[head_weight_means, head_bias_means]
        )
      else:
        return self._forward(
          x,
          head_idx,
          in_proj_params=[
              self._reparameterize(in_proj_weight_means, in_proj_weight_logvars),
              self._reparameterize(in_proj_bias_means, in_proj_bias_logvars),
          ],
          backbone_params=[
              self._reparameterize(backbone_weight_means, backbone_weight_logvars),
              self._reparameterize(backbone_bias_means, backbone_bias_logvars),
          ],
          head_params=[
              self._reparameterize(head_weight_means, head_weight_logvars),
              self._reparameterize(head_bias_means, head_bias_logvars),
          ]
        )