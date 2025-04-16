from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from PIL import Image

def deterministic_permutation(size: int, seed: int) -> torch.Tensor:
    return torch.randperm(
        size,
        generator=torch.Generator().manual_seed(seed)
    )

class SplitMNIST(MNIST):
    def __init__(
        self,
        task: int = 0,
        transform=None,
        target_transform=None,
        *args,
        **kwargs,
    ):
        super(SplitMNIST, self).__init__(*args, **{
            **kwargs,
            **{
                "transform": None,
                "target_transform": None,
                "download": True
            }
        })
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.zeros = [0, 2, 4, 6, 8]
        self.ones = [1, 3, 5, 7, 9]
        self.setup_task()

    def setup_task(self):
        zero = self.zeros[self.task]
        one = self.ones[self.task]

        x = self.data
        y = self.targets

        zeros = x[y == zero]
        ones = x[y == one]

        num_zeros, *_ = zeros.shape
        num_ones, *_ = ones.shape

        images = torch.cat([zeros, ones])
        labels = torch.cat(
            [
                torch.zeros((num_zeros, ), device=x.device),
                torch.ones((num_ones, ), device=x.device)
            ]
        ).to(torch.long)
        
        permutation = deterministic_permutation(
            num_zeros + num_ones,
            seed=42
        )

        self.images = images[permutation]
        self.labels = labels[permutation]

    def __getitem__(self, index):
        x, y = (
            self.images[index].numpy(),
            self.labels[index].numpy()
        )
        x = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.images)

class SplitNotMNIST(Dataset):
    def __init__(self, root, transform=None, task: int = 0):
        """
        Args:
            root (str): Path to the dataset folder containing subfolders A-J.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.image_paths = []
        self.labels = []

        self.zeros = ["A", "B", "C", "D", "E"]
        self.ones = ["F", "G", "H", "I", "J"]

        zero = self.zeros[task]
        one = self.ones[task]

        zeros = list(
            sorted(
                (Path(root) / zero).iterdir()
            )
        )
        ones = list(
            sorted(
                (Path(root) / one).iterdir()
            )
        )

        num_zeros = len(zeros)
        num_ones = len(ones)

        self.images = zeros + ones
        self.labels = torch.cat(
            [
                torch.zeros((num_zeros, )),
                torch.ones((num_ones, ))
            ]
        ).to(torch.long)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        p, y = (
            self.images[index],
            self.labels[index].numpy()
        )
        x = Image.open(p).convert("L")
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.images)


class SingleLabelMNIST(MNIST):
    def __init__(
        self,
        root: str,
        label: int,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, train, transform, target_transform, download)
        if not 0 <= label <= 9:
            raise ValueError(f"Label must be between 0 and 9, got {label}")
        self.label = label
        # Create a mask for samples with the specified label
        label_mask = self.targets == label
        # Filter data and targets to only include the specified label
        self.data = self.data[label_mask]
        self.targets = self.targets[label_mask]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)
    
    def __len__(self) -> int:
        return len(self.data)

class SingleLabelNotMNIST(Dataset):
    def __init__(self, root, transform=None, task: int = 0):
        self.root_dir = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.image_paths = []
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        label = self.labels[task]

        self.images =list(
            sorted(
                (Path(root) / label).iterdir()
            )
        )
        self.labels =  task * torch.ones((len(self.images), )).to(torch.long)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        p, y = (
            self.images[index],
            self.labels[index].numpy()
        )
        x = Image.open(p).convert("L")
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.images)
    