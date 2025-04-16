from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from typing import List

from losses import mse_loss_onehot


def save(
    nn: nn.Module,
    optim: Optional[Adam],
    path: Path 
):
    if optim is not None:
      torch.save({
        "model_state_dict": nn.state_dict(),
        "optim_state_dict": optim.state_dict()
      }, path)
    else:
      torch.save({
        "model_state_dict": nn.state_dict(),
      }, path)

def load(
    nn: nn.Module,
    optim: Optional[Adam],
    path: Path
):
    ckpt = torch.load(path, weights_only=True)
    nn.load_state_dict(ckpt["model_state_dict"])
    if optim is not None:
        optim.load_state_dict(ckpt["optim_state_dict"])
    return nn


def evaluate_single_task(
    nn: nn.Module,
    task: int,
    test_dataloader: DataLoader,
    device: torch.device,
    num_monte_carlo_samples: int
):
    metrics: List[torch.Tensor] = []
    for (x, y) in test_dataloader:
        batch_size, *_ = x.shape
        x = x.view((batch_size, -1))
        x = x.to(device)
        y = y.to(device)
        head = 0 if nn.num_heads == 1 else task
        outputs = nn.predict(x, head, num_monte_carlo_samples)
        match nn.objective:
          case "classification":
              preds = torch.argmax(outputs, -1)
              metrics.append((preds == y).float())
          case "regression":
              metrics.append(
                  mse_loss_onehot(
                    outputs,
                    y,
                    reduction="none",
                    num_classes=nn.output_dim
                  ).sum(dim=-1)
              )
    match nn.objective:
      case "classification":
        return torch.cat(metrics, dim=0).mean().item()
      case "regression":
        return torch.sqrt(torch.cat(metrics, dim=0).mean()).item()
     

def evaluate_many_tasks(
    tasks: List[int],
    nn: nn.Module,
    test_dataloaders: List[DataLoader],
    device: torch.device,
    num_monte_carlo_samples: int,
):
    task_metrics = [   
        evaluate_single_task(
            nn,
            task,
            test_dataloader,
            device,
            num_monte_carlo_samples
        )
        for (task, test_dataloader) in zip(tasks, test_dataloaders)
    ]
    return np.array(task_metrics)

def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)