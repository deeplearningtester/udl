
import heapq
from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from generative_models import VAE
from losses import mse_loss_onehot
from models import DiscriminativeMeanField, DiscriminativeNaive
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tasks import Task, subsample_dataset
from utils import evaluate_many_tasks, save
from enum import StrEnum
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from typing import Dict

class CoresetMethod(StrEnum):
    RANDOM = "random"
    K_CENTER = "k_center"

class CoresetSplit(NamedTuple):
    coreset: Dataset
    non_coreset: Dataset

class EWC:
    def __init__(
        self,
        model: DiscriminativeNaive,
        ewc_lambda: float
    ):
        self.model = model
        # Dictionary to store Fisher information matrices for each task
        self.fisher_infos: Dict[int, Dict[str, torch.Tensor]] = {}
        # Dictionary to store optimal parameters for each task
        self.optimal_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self.ewc_lambda = ewc_lambda

    def ll(self, x: torch.Tensor, task_idx: int, y: Optional[torch.Tensor]):
      if isinstance(self.model, DiscriminativeNaive):
        head_idx = 0 if self.model.num_heads == 1 else task_idx
        outputs = self.model(x, head_idx)
        b, _ = x.shape
        match self.model.objective:
          case "classification":
            log_probs = torch.log_softmax(outputs, dim=-1)
            log_likelihood = log_probs[
              torch.arange(0, b, device=x.device).to(torch.long),
              y
            ] # [b, ]
          case "regression":
            log_likelihood = -mse_loss_onehot(outputs, y, reduction="none", num_classes=self.model.output_dim).sum(dim=-1) # [b, ]
        return log_likelihood
      else:
        return self.model.elbo(x, task_idx)
        
    def compute_task_fisher_info(
        self, 
        task_dataloader: DataLoader, 
        task_idx: int, 
        device: torch.device
    ):
        self.model.eval()
        self.model = self.model.to(device)
        fisher_info = {n: torch.zeros_like(p) for n, p in self.model.learnable_parameters()}
        self.optimal_params[task_idx] = {n: p.clone().detach() for n, p in self.model.learnable_parameters()}
        sample_count = 0
        for x, y in task_dataloader:
          x = x.to(device)
          y = y.to(device)
          b, _ = x.shape
          sample_count += b
          self.model.zero_grad()
          log_likelihood = self.ll(x, task_idx, y)
          log_likelihood.backward()
          for name, param in self.model.learnable_parameters():
              if param.grad is not None:
                  fisher_info[name] += param.grad.pow(2).detach()
        # Normalize by the number of samples (Monte Carlo estimate)
        for name in fisher_info:
            fisher_info[name] /= max(1, sample_count)
        self.fisher_infos[task_idx] = fisher_info
        self.model.train()

    def ewc_step(self, x: torch.Tensor, y: torch.Tensor, task_idx: int):
      if isinstance(self.model, DiscriminativeNaive):
        match self.model.objective:
          case "classification":
            return (
              F.cross_entropy(self.model(x, task_idx), y) + 
              self.ewc_penalty()
            )
          case "regression":
            return (
              mse_loss_onehot(
                self.model(x, task_idx),
                y,
                reduction="none",
                num_classes=self.model.output_dim
              ).sum(dim=-1).mean() + self.ewc_penalty()
            )
      else:
        elbo = self.model.elbo(x, task_idx)
        return -elbo + self.ewc_penalty()

    def ewc_penalty(self):
        if not self.fisher_infos:
            return 0.0
        penalty = 0.0
        for task_idx, task_fisher in self.fisher_infos.items():
            task_optimal_params = self.optimal_params[task_idx]
            if isinstance(self.model, DiscriminativeNaive):
              for param_name, param in self.model.learnable_parameters():
                  if param_name not in task_fisher or param_name not in task_optimal_params:
                      continue
                  penalty += (task_fisher[param_name] * (param - task_optimal_params[param_name]).pow(2)).sum()
            else:
              for param_name, param in self.model.shared_learnable_parameters():
                  if param_name not in task_fisher or param_name not in task_optimal_params:
                      continue
                  penalty += (task_fisher[param_name] * (param - task_optimal_params[param_name]).pow(2)).sum()

        return 0.5 * self.ewc_lambda * penalty

class SynapticIntelligence:
    def __init__(self, model: nn.Module, lr: float, si_lambda=0.01, device=None):
        self.model = model
        self.si_lambda = si_lambda
        self.device = device or next(model.parameters()).device
        self._init_si_buffers()
        self.damping = 0.1
        self.lr = lr

    def _init_si_buffers(self):
        # Store previous parameter values (theta*)
        self.previous_weights = {
          n: torch.zeros_like(p, device=self.device, requires_grad=False)
          for n, p in self.model.named_parameters() if p.requires_grad
        }
        # Accumulate omega (importance measure during training)
        self.omega = {
          n: torch.zeros_like(p, device=self.device, requires_grad=False)
          for n, p in self.model.named_parameters() if p.requires_grad
        }
        # Buffer to accumulate gradients × parameter change
        self.w = {
           n: torch.zeros_like(p, device=self.device, requires_grad=False)
           for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.grad_cache = {}
        self.weight_cache = {}

    def cache_grads(self):
        # Call after backward() to cache current gradients
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.grad_cache[n] = p.grad.detach().clone()
                self.weight_cache[n] = p.detach().clone()

    def penalty(self):
      penalty = 0.0

      for n, p in self.model.named_parameters():
        penalty += torch.sum(self.omega[n] * (self.previous_weights[n] - p)**2)

      return self.si_lambda * penalty
                
    def update_w(self):
        # Call after optimizer step: accumulate gradient * parameter change
        for n, p in self.model.named_parameters():
            #self.w[n] += (self.lr * p.grad * self.grad_cache[n]).detach().clone()
            self.w[n] += (self.grad_cache[n] * (self.weight_cache[n] - p)).detach().clone()

        self.grad_cache.clear()

    def reset_w(self):
      for n, p in self.model.named_parameters():
        self.w[n].zero_()
        self.previous_weights[n] = p.detach().clone()

    def update_omega(self):
      # Call at the end of a task: update parameter importance weights
      for n, p in self.model.named_parameters():
        delta = p.detach() - self.previous_weights[n]
        self.omega[n] += self.w[n] / (self.damping + delta ** 2)


def get_dataloader(data, batch_size: int, shuffle: bool = False, use_multiple_workers: bool = False):
    NUM_WORKERS = 8
    PREFETCH_FACTOR=2
    PIN_MEMORY=True
    PERSISTENT_WORKERS=True
    if not use_multiple_workers:
      return DataLoader(
        data,
          batch_size=batch_size,
          shuffle=shuffle,
          pin_memory=PIN_MEMORY
      )
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY
    )

def get_k_center_coreset_indices(dataset: Dataset, coreset_size: int):
    xs = np.array([x for x, _ in iter(dataset)])

    # Compute initial distances from the first point
    dists = np.linalg.norm(xs - xs[0], axis=1)
    # Ensure self-distance is zero
    dists[0] = 0
    # Start from the first data point
    coreset_indices = [0]

    heap = [(-d, i) for i, d in enumerate(dists)]
    heapq.heapify(heap)

    # Greedily choose the next point
    for _ in tqdm(range(coreset_size - 1), desc="Selecting coreset points"):
        while True:
            _, next_point_idx = heapq.heappop(heap)
            if next_point_idx not in coreset_indices:
                break
        # Add to the coreset
        coreset_indices.append(next_point_idx)
        # Compute pairwise distances between all points and the newly selected point
        new_distances = np.linalg.norm(xs - xs[next_point_idx], axis=1)
        # Only update distances for points NOT in the coreset
        mask = np.ones(len(xs), dtype=bool)  # Create a mask for all points
        mask[coreset_indices] = False  # Mark coreset points as False (don't update them)
        dists[mask] = np.minimum(dists[mask], new_distances[mask])
        # Rebuild the heap with updated distances
        heap = [(-dists[j], j) for j in range(len(xs)) if j not in coreset_indices]
        heapq.heapify(heap)

    return coreset_indices

def get_coreset(method: CoresetMethod, size: int, dataset: Dataset) -> CoresetSplit:
    match method:
        case CoresetMethod.RANDOM:
          [coreset, non_coreset] = random_split(
            dataset,
            [size, len(dataset) - size]
          )
        case CoresetMethod.K_CENTER:
            task_coreset_indices = get_k_center_coreset_indices(dataset, size)
            task_non_coreset_indices = list(
                set(range(len(dataset))).difference(set(task_coreset_indices))
            )
            [coreset, non_coreset] = (
                Subset(dataset, task_coreset_indices),
                Subset(dataset, task_non_coreset_indices),
            )
    return CoresetSplit(
        coreset=coreset,
        non_coreset=non_coreset
    )

def mle(
    model: DiscriminativeMeanField,
    data,
    epochs,
    batch_size,
    device,
    lr,
    experiment_dir: Path = Path(".")
):
    print("[mle] ...")

    optimizer = Adam(model.parameters(), lr=lr)
    loader = get_dataloader(data, batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for (x, y) in loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            loss = model.mle_step(x.to(device), y.to(device), head_idx=0) # [b, ]
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        print(f"epoch {epoch+1} average loss: {avg_loss:.6f}")

    save(model, optimizer, experiment_dir / f"mle.pth")

def train_vcl(
  model: DiscriminativeMeanField,
  optimizer: Adam,
  dataset: Dataset,
  task_idx: int,
  epochs: int,
  batch_size: int,
  num_train_monte_carlo_samples: int,
  device: torch.device,
):
  loader = get_dataloader(
      dataset,
      batch_size,
      shuffle=True
  )
  kl_weight = 1.0 / len(dataset)
  model.train()

  for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    for (x, y) in loader:
        optimizer.zero_grad()
        loss = model.vcl_step(
            x.to(device),
            y.to(device),
            head_idx=(0 if model.num_heads == 1 else task_idx),
            kl_weight=kl_weight,
            num_monte_carlo_samples=num_train_monte_carlo_samples
        )
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    avg_loss = epoch_loss / num_batches
    print(f"epoch {epoch+1} average loss: {avg_loss:.6f}")

def train_coreset_balanced(
    model: DiscriminativeMeanField,
    optimizer: Adam,
    coreset: List[Dataset],
    epochs: int,
    batch_size: int,
    num_train_monte_carlo_samples: int,
    device: torch.device,
):  
    print("training on coreset (task balanced)")

    num_tasks = len(coreset)
    mini_batch_size = batch_size // num_tasks

    task_coreset_lengths = [
        len(task_coreset)
        for task_coreset in coreset
    ]
    task_coreset_dataloaders = [
        get_dataloader(
            task_coreset,
            mini_batch_size,
            shuffle=True
        )
        for task_coreset in coreset
    ]
    kl_weight = 1.0 / sum(task_coreset_lengths)
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        task_coreset_iterators = [
            iter(loader)
            for loader in task_coreset_dataloaders
        ]
        # Randomize order of tasks
        task_order = np.arange(len(task_coreset_iterators))
        np.random.shuffle(task_order)
        # Iterate through all task dataloaders
        while True:
            try:
                optimizer.zero_grad()
                nll: List[torch.Tensor] = []
                kl: List[torch.Tensor] = [
                  kl_weight * model.kl(head_idx=None) # backbone KL
                ]
                for task_idx in task_order:
                    x, y = next(task_coreset_iterators[task_idx])
                    x = x.to(device)
                    y = y.to(device)
                    losses = model.vcl_coreset_task_step(
                        x,
                        y,
                        task_idx,
                        kl_weight,
                        num_train_monte_carlo_samples
                    )
                    nll.append(losses["nll"])
                    # ensure that kl is computed once per head
                    if model.num_heads == 1:
                        if task_idx == 0:
                            kl.append(losses["kl"])
                    else:
                      kl.append(losses["kl"])
                # collect all losses
                nll = torch.cat(nll, dim=0) # [b, ]
                kl = torch.tensor(kl)
                # sum backbone kl with all task head kls
                loss = nll.mean() + kl.sum()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            except StopIteration:
                break
        avg_loss = epoch_loss / num_batches
        print(f"epoch {epoch+1} average loss: {avg_loss:.6f}")

def train_coreset_inorder(
    model: DiscriminativeMeanField,
    optimizer: Adam,
    coreset: List[Dataset],
    epochs: int,
    batch_size: int,
    num_train_monte_carlo_samples: int,
    device: torch.device,
):  
    print("training on coreset (task inorder)")
    task_coreset_lengths = [
        len(task_coreset)
        for task_coreset in coreset
    ]
    task_coreset_dataloaders = [
        get_dataloader(
            task_coreset,
            batch_size,
            shuffle=True
        )
        for task_coreset in coreset
    ]
    kl_weight = 1.0 / sum(task_coreset_lengths)
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        # Randomize order of tasks
        task_order = np.arange(len(task_coreset_dataloaders))
        np.random.shuffle(task_order)
        # Process tasks in order
        for task_idx in task_order:
            for (x, y) in task_coreset_dataloaders[task_idx]:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                losses = model.vcl_coreset_task_step(
                    x,
                    y,
                    task_idx,
                    kl_weight,
                    num_train_monte_carlo_samples
                )
                nll = losses["nll"].mean()
                head_kl = losses["kl"] # Already weighted
                backbone_kl = kl_weight * model.kl(head_idx=None)
                total_kl = backbone_kl + head_kl
                loss = nll + total_kl
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"epoch {epoch+1} average loss: {avg_loss:.6f}")

def coreset_only(
  model: DiscriminativeMeanField,
  tasks: List[Task],
  coreset_method: CoresetMethod,
  coreset_size: int,
  epochs: int,
  lr: float,
  num_train_monte_carlo_samples: int,
  num_test_monte_carlo_samples: int,
  device: torch.device,
  experiment_dir: Path,
  coreset_balanced: bool
) -> List[np.array]:
    history: List[np.array] = []
    coreset: List[Dataset] = []

    for task_idx in range(len(tasks)):
        # get the task
        task_coreset, _ = get_coreset(
            coreset_method,
            coreset_size,
            tasks[task_idx].train
        )
        # update coreset
        if not coreset:
            coreset = [task_coreset]
        else:
            coreset.append(task_coreset)

        current_coreset_size = sum(len(dataset) for dataset in coreset)
        optimizer = Adam(
          model.parameters(),
          lr=lr
        )
        model.train()
        # train 
        if coreset_balanced:
            train_coreset_balanced(
                model,
                optimizer,
                coreset,
                epochs,
                current_coreset_size,
                num_train_monte_carlo_samples,
                device
            )
        else: 
            train_coreset_inorder(
                model,
                optimizer,
                coreset,
                epochs,
                current_coreset_size,
                num_train_monte_carlo_samples,
                device
            )        
        # advance prior
        if model.num_heads == 1:
          model.advance_prior(0)
        else:
          for head in range(task_idx + 1):
            model.advance_prior(head)
        # save the model after the task
        save(model, optimizer, experiment_dir / f"task_{task_idx}.pth")
        # evaluate on all observed tasks so far
        result = evaluate_many_tasks(
            [t for t in range(task_idx + 1)],
            model,
            [
              get_dataloader(tasks[t].test, batch_size=current_coreset_size)
              for t in range(task_idx + 1)
            ],
            device,
            num_test_monte_carlo_samples
        )
        print(result, result.mean())
        print("----")
        history.append(result)

    return history

def vcl(
  model: DiscriminativeMeanField,
  tasks: List[Task],
  epochs: int,
  batch_size: int,
  lr: float,
  num_train_monte_carlo_samples: int,
  num_test_monte_carlo_samples: int,
  device: torch.device,
  experiment_dir: Path
) -> List[np.array]:
    history = []

    for task_idx in range(len(tasks)):
        optimizer = Adam(
            model.parameters(),
            lr=lr
        )
        # train on task
        print(f"[vcl] [task={task_idx}]...")
        train_vcl(
            model,
            optimizer,
            tasks[task_idx].train,
            task_idx,
            epochs,
            batch_size,
            num_train_monte_carlo_samples,
            device
        )
        # advance prior
        model.advance_prior(
          head_idx=(0 if model.num_heads == 1 else task_idx)
        )
        save(model, optimizer, experiment_dir / f"task_{task_idx}.pth")
        result = evaluate_many_tasks(
            [t for t in range(task_idx + 1)],
            model,
            [
              get_dataloader(tasks[t].test, batch_size=batch_size)
              for t in range(task_idx + 1)
            ],
            device,
            num_test_monte_carlo_samples
        )
        print(result, result.mean())
        print("----")
        history.append(result)

    return history

def vcl_with_coreset(
  model: DiscriminativeMeanField,
  tasks: List[Task],
  coreset_method: CoresetMethod,
  coreset_size: int,
  epochs: int,
  batch_size: int,
  lr: float,
  num_train_monte_carlo_samples: int,
  num_test_monte_carlo_samples: int,
  device: torch.device,
  experiment_dir: Path,
  coreset_balanced: bool = False
):
    from copy import deepcopy
    history: List[np.array] = []
    coreset: List[Dataset] = []
    for task_idx in range(len(tasks)):
        task_coreset, task_non_coreset = get_coreset(
            coreset_method,
            coreset_size,
            tasks[task_idx].train
        )
        if not coreset:
            coreset = [task_coreset]
        else:
            coreset.append(task_coreset)
        optimizer = Adam(model.parameters(), lr=lr)
        # train on this task
        print(f"[vcl+coreset({coreset_method}, {coreset_size})] [task={task_idx}] training on non-coreset ...")
        train_vcl(
            model,
            optimizer,
            task_non_coreset,
            task_idx,
            epochs,
            batch_size,
            num_train_monte_carlo_samples,
            device,
        )
        save(model, optimizer, experiment_dir / f"task_{task_idx}_without_coreset.pth")
        # advance prior
        head = 0 if (model.num_heads == 1) else task_idx
        model.advance_prior(head)
        # train on coreset
        print(f"[vcl+coreset({coreset_method}, {coreset_size})] training on coreset ...")
        coreset_model = deepcopy(model)
        coreset_optimizer = Adam(coreset_model.parameters(), lr=lr)
        coreset_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        if coreset_balanced:
            train_coreset_balanced(
                model,
                optimizer,
                coreset,
                epochs,
                batch_size,
                num_train_monte_carlo_samples,
                device
            )
        else: 
            train_coreset_inorder(
                model,
                optimizer,
                coreset,
                epochs,
                batch_size,
                num_train_monte_carlo_samples,
                device
            )                        
        save(coreset_model, optimizer, experiment_dir / f"task_{task_idx}_with_coreset.pth")
        result = evaluate_many_tasks(
            [t for t in range(task_idx + 1)],
            model,
            [
              get_dataloader(tasks[t].test, batch_size=batch_size)
              for t in range(task_idx + 1)
            ],
            device,
            num_test_monte_carlo_samples
        )
        print(result, result.mean())
        print("----")
        history.append(result)
        
    return history

def train_si(
  si: SynapticIntelligence,
  optimizer: Adam,
  dataset: Dataset,
  task_idx: int,
  epochs: int,
  batch_size: int,
  device: torch.device,
):
  train_loader = get_dataloader(dataset, batch_size, True)

  si.model.to(device)
  si.model.train()

  for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    for (x, y) in train_loader:
        head_idx = 0 if si.model.num_heads == 1 else task_idx
        optimizer.zero_grad()
        match si.model.objective:
          case "classification":
            task_loss = F.cross_entropy(
              si.model(x.to(device), head_idx),
              y.to(device)
            )
          case "regression":
            task_loss = mse_loss_onehot(
              si.model(x.to(device), head_idx),
              y.to(device)
            ).sum(dim=-1).mean()
        task_loss.backward()
        si.cache_grads()
        optimizer.zero_grad()
        match si.model.objective:
          case "classification":
            task_loss = F.cross_entropy(
              si.model(x.to(device), head_idx),
              y.to(device)
            )
          case "regression":
            task_loss = mse_loss_onehot(
              si.model(x.to(device), head_idx),
              y.to(device)
            ).sum(dim=-1).mean()
        total_loss = task_loss + si.penalty()
        total_loss.backward()
        optimizer.step()
        si.update_w()
        epoch_loss += total_loss.item()
        num_batches += 1
    avg_loss = epoch_loss / num_batches
    print(f"epoch {epoch+1} average loss: {avg_loss:.6f}")

def si(
  model: DiscriminativeNaive,
  tasks: List[Task],
  epochs: int,
  batch_size: int,
  lr: float,
  num_test_monte_carlo_samples: int,
  device: torch.device,
  experiment_dir: Path,
  si_lambda: float,
  seed: int
) -> List[np.array]:
    history = []
    model = model.to(device)
    si = SynapticIntelligence(model, lr, si_lambda)
    for task_idx in range(len(tasks)):
        optimizer = Adam(
            model.parameters(),
            lr=lr
        )
        print(f"[si] [lambda={si_lambda}] [task={task_idx}]...")
        train_si(
          si,
          optimizer,
          tasks[task_idx].train,
          task_idx,
          epochs,
          batch_size,
          device
        )
        si.update_omega()
        si.reset_w()
        print(f"[si] [task={task_idx}] importance...")
        save(model, optimizer, experiment_dir / f"task_{task_idx}.pth")
        result = evaluate_many_tasks(
            [t for t in range(task_idx + 1)],
            model,
            [
              get_dataloader(tasks[t].test, batch_size=batch_size)
              for t in range(task_idx + 1)
            ],
            device,
            num_test_monte_carlo_samples
        )
        print(result, result.mean())
        print("----")
        history.append(result)
    return history

def train_ewc(
  ewc: EWC,
  optimizer: Adam,
  dataset: Dataset,
  task_idx: int,
  epochs: int,
  batch_size: int,
  device: torch.device,
):
  train_loader = get_dataloader(dataset, batch_size, True)

  ewc.model.to(device)
  ewc.model.train()

  for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    for (x, y) in train_loader:
        optimizer.zero_grad()
        head_idx = 0 if ewc.model.num_heads == 1 else task_idx
        loss = ewc.ewc_step(x.to(device), y.to(device), head_idx)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    avg_loss = epoch_loss / num_batches
    print(f"epoch {epoch+1} average loss: {avg_loss:.6f}")

def ewc(
  model: DiscriminativeNaive,
  tasks: List[Task],
  epochs: int,
  batch_size: int,
  lr: float,
  num_ewc_samples: int,
  num_test_monte_carlo_samples: int,
  device: torch.device,
  experiment_dir: Path,
  ewc_lambda: float,
  seed: int
) -> List[np.array]:
    history = []
    ewc = EWC(model, ewc_lambda)
    for task_idx in range(len(tasks)):
        optimizer = Adam(
            model.parameters(),
            lr=lr
        )
        print(f"[ewc] [lambda={ewc_lambda}] [task={task_idx}]...")
        task_train = tasks[task_idx].train
        ewc_fit = subsample_dataset(task_train, num_ewc_samples, seed)
        ewc_loader = get_dataloader(ewc_fit, 1, True)
        train_ewc(
          ewc,
          optimizer,
          task_train,
          task_idx,
          epochs,
          batch_size,
          device,
        )
        print(f"[ewc] [samples={num_ewc_samples}] [task={task_idx}]...")
        ewc.compute_task_fisher_info(
          ewc_loader,
          task_idx,
          device
        )
        save(model, optimizer, experiment_dir / f"task_{task_idx}.pth")
        result = evaluate_many_tasks(
            [t for t in range(task_idx + 1)],
            model,
            [
              get_dataloader(tasks[t].test, batch_size=batch_size)
              for t in range(task_idx + 1)
            ],
            device,
            num_test_monte_carlo_samples
        )
        print(result, result.mean())
        print("----")
        history.append(result)
    return history


# ----------------- VAE -----------------

def train_vae_naive(
    model: VAE,
    optimizer: Adam,
    num_epochs: int,
    batch_size: int,
    task_idx: int,
    task: Task,
    experiment_dir: Path,
    device: torch.device,
    log_every_steps: int = 20,
):
    task_dir = experiment_dir / f"task={task_idx}"
    task_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader = get_dataloader(task.train, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        print("starting epoch " + str(epoch))
        running_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            elbo = model(x.to(device), head_idx=task_idx) # [b, ]
            loss = -elbo.mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % log_every_steps == 0:
                avg_loss = running_loss / log_every_steps
                print(f"[naive] [epoch {epoch + 1}, step {batch_idx + 1}] loss: {avg_loss:.5f}")
                running_loss = 0.0
        model.sample_grid(100, task_idx, device).save(str(task_dir/ f"epoch_{epoch}.png"))

def train_vae_vcl(
    model: VAE,
    optimizer: Adam,
    num_epochs: int,
    batch_size: int,
    task_idx: int,
    task: Task,
    experiment_dir: Path,
    device: torch.device,
    log_every_steps: int = 20,
):
    task_dir = experiment_dir / f"task={task_idx}"
    task_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader = get_dataloader(task.train, batch_size=batch_size, shuffle=True)
    kl_weight = 1.0 / len(task.train)

    for epoch in range(num_epochs):
        print("starting epoch " + str(epoch))
        running_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            elbo = model(x.to(device), head_idx=task_idx)
            kl = kl_weight * model.decoder_tail.kl()
            loss = -elbo.mean() + kl
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % log_every_steps == 0:
                avg_loss = running_loss / log_every_steps
                print(f"[vcl] [epoch {epoch + 1}, step {batch_idx + 1}] loss: {avg_loss:.5f}")
                running_loss = 0.0

        model.sample_grid(100, task_idx, device).save(str(task_dir/ f"epoch_{epoch}.png"))
        print(model.decoder_tail.kl())

    print("--")
    print("kl before advance", model.decoder_tail.kl())
    model.decoder_tail.advance_prior()
    print("kl after advance", model.decoder_tail.kl())
    print("--")

def train_vae_ewc(
    ewc: EWC,
    optimizer: Adam,
    num_epochs: int,
    batch_size: int,
    task_idx: int,
    task: Task,
    experiment_dir: Path,
    device: torch.device,
    log_every_steps: int = 20,
):
    task_dir = experiment_dir / f"task={task_idx}"
    task_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader = get_dataloader(task.train, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        print("starting epoch " + str(epoch))
        running_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = (
              ewc.ewc_step(x.to(device), None, task_idx)
                .mean()
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % log_every_steps == 0:
                avg_loss = running_loss / log_every_steps
                print(f"[ewc] [epoch {epoch + 1}, step {batch_idx + 1}] loss: {avg_loss:.5f}")
                running_loss = 0.0

        ewc.model.sample_grid(100, task_idx, device).save(str(task_dir/ f"epoch_{epoch}.png"))

def vae_ewc(
  model: VAE,
  tasks: List[Task],
  num_epochs: int,
  batch_size: int,
  lr: float,
  ewc_lambda: float,
  ewc_num_samples: int,
  experiment_dir: Path,
  device: torch.device,
  seed: int
):
  for (task_idx, task) in enumerate(tasks):
    task_dir = experiment_dir / f"task={task_idx}"
    task_dir.mkdir(exist_ok=True, parents=True)

    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)

    ewc = EWC(model, ewc_lambda)
    task_train = tasks[task_idx].train
    ewc_fit = subsample_dataset(task_train, ewc_num_samples, seed)
    ewc_loader = get_dataloader(ewc_fit, 1, True)

    print(f"[task={task_idx}] size : {len(task.train)}")

    train_vae_ewc(
        ewc,
        optimizer,
        num_epochs,
        batch_size,
        task_idx,
        task,
        experiment_dir,
        device,
    )

    print(f"[ewc] [samples={ewc_num_samples}] [task={task_idx}]...")
    ewc.compute_task_fisher_info(
      ewc_loader,
      task_idx,
      device
    )

    torch.save(
      model.state_dict(),
      str(experiment_dir / f"after_{task_idx}.pth")
    )

    samples: List[torch.Tensor] = []

    for head_idx in range(task_idx + 1):
      samples.append(
          model.sample(1, head_idx, device).view((1, 1, 28, 28))
      )

    samples = torch.cat(samples, dim=0)
    samples_row = to_pil_image(make_grid(samples, nrow=samples.shape[0]))
    samples_row.save(str(experiment_dir/ f"after_{task_idx}.png"))
    samples_row.show()    

def vae_vcl(
  model: VAE,
  tasks: List[Task],
  num_epochs: int,
  batch_size: int,
  lr: float,
  experiment_dir: Path,
  device: torch.device
):
  for (task_idx, task) in enumerate(tasks):
    task_dir = experiment_dir / f"task={task_idx}"
    task_dir.mkdir(exist_ok=True, parents=True)

    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)

    print(f"[task={task_idx}] size : {len(task.train)}")

    train_vae_vcl(
        model,
        optimizer,
        num_epochs,
        batch_size,
        task_idx,
        task,
        experiment_dir,
        device
    )

    torch.save(
      model.state_dict(),
      str(experiment_dir / f"after_{task_idx}.pth")
    )

    samples: List[torch.Tensor] = []

    for head_idx in range(task_idx + 1):
      samples.append(
          model.sample(1, head_idx, device).view((1, 1, 28, 28))
      )

    samples = torch.cat(samples, dim=0)
    samples_row = to_pil_image(make_grid(samples, nrow=samples.shape[0]))
    samples_row.save(str(experiment_dir/ f"after_{task_idx}.png"))
    samples_row.show()    

def vae_naive(
  model: VAE,
  tasks: List[Task],
  num_epochs: int,
  batch_size: int,
  lr: float,
  experiment_dir: Path,
  device: torch.device
):
  for (task_idx, task) in enumerate(tasks):
    task_dir = experiment_dir / f"task={task_idx}"
    task_dir.mkdir(exist_ok=True, parents=True)

    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)

    print(f"[task={task_idx}] size : {len(task.train)}")

    train_vae_naive(
        model,
        optimizer,
        num_epochs,
        batch_size,
        task_idx,
        task,
        experiment_dir,
        device
    )

    torch.save(
      model.state_dict(),
      str(experiment_dir / f"after_{task_idx}.pth")
    )

    samples: List[torch.Tensor] = []

    for head_idx in range(task_idx + 1):
      samples.append(
          model.sample(1, head_idx, device).view((1, 1, 28, 28))
      )

    samples = torch.cat(samples, dim=0)
    samples_row = to_pil_image(make_grid(samples, nrow=samples.shape[0]))
    samples_row.save(str(experiment_dir/ f"after_{task_idx}.png"))
    samples_row.show()