from typing import List, NamedTuple
import torch
from torchvision.datasets import MNIST

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from transforms import Flatten, Normalize, Permute
from datasets import *

class Task(NamedTuple):
    train: Dataset
    test: Dataset


def get_permuted_mnist(
  num_tasks:int
):
    transforms = []
    for _ in range(num_tasks):
        permutation = torch.randperm(28*28)
        transforms.append(
          Compose([
              Flatten(),
              Normalize(),
              Permute(permutation)
          ])
        )
    return [
        Task(
            train=MNIST(
              root="/scratch/shared/beegfs/user/datasets",
              train=True,
              transform=transform
            ),
            test=MNIST(
              root="/scratch/shared/beegfs/user/datasets",
              train=False,
              transform=transform
            )
        )
        for transform in transforms
    ]

def get_split_mnist(
  num_tasks:int
):
    transform = Compose([
      Flatten(),
      Normalize(),
    ])
    return [
        Task(
            train=SplitMNIST(
              task=task,
              root="/scratch/shared/beegfs/user/datasets",
              train=True,
              transform=transform
            ),
            test=SplitMNIST(
              task=task,
              root="/scratch/shared/beegfs/user/datasets",
              train=False,
              transform=transform
            ),
        )
        for task in range(num_tasks)
    ]

def subsample_dataset(dataset: Dataset, num_samples: int, seed:int = 42) -> Dataset:
    from torch.utils.data import Subset
    assert num_samples <= len(dataset), "num_samples exceeds dataset size"
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:num_samples].tolist()
    return Subset(dataset, indices)

def get_split_not_mnist(
  num_tasks: int
):
    # TRAIN_PATH = "/scratch/shared/beegfs/user/datasets/datasets/jwjohnson314/notmnist/versions/2/notMNIST_large/notMNIST_large"
    # TEST_PATH = "/scratch/shared/beegfs/user/datasets/datasets/jwjohnson314/notmnist/versions/2/notMNIST_small/notMNIST_small"
    # transform = Compose([
    #   Flatten(),
    #   Scale(),
    # ])
    # return [
    #     Task(
    #         train=SplitNotMNIST(
    #           task=task,
    #           root=TRAIN_PATH,
    #           transform=transform
    #         ),
    #         test=SplitNotMNIST(
    #           task=task,
    #           root=TEST_PATH,
    #           transform=transform
    #         ),
    #     )
    #     for task in range(num_tasks)
    # ]
    from torch.utils.data import random_split

    DATASET_SPLIT_SEED=42
    NUM_TASKS = 5
    ROOT = "/scratch/shared/beegfs/user/datasets/datasets/jwjohnson314/notmnist/versions/2/notMNIST_large/notMNIST_large"

    transform = Compose([
        Flatten(),
        Normalize(),
    ])

    total_dataset_size = 0 
    for task in range(NUM_TASKS):
        dataset = SplitNotMNIST(
            root=ROOT,
            transform=transform,
            task=task
        )
        total_dataset_size += len(dataset)

    tasks: List[Task] = []
    for task in range(NUM_TASKS):
        dataset = SplitNotMNIST(
            root=ROOT,
            transform=transform,
            task=task
        )
        dataset = subsample_dataset(
            dataset,
            min(
                len(dataset),
                total_dataset_size // num_tasks
            )
        )
        # MNIST ratio
        train_ratio = 6 / (6 + 1)
        test_ratio = 1 - train_ratio
        [train, test] = random_split(
            dataset,
            [train_ratio, test_ratio],
            generator=torch.Generator().manual_seed(DATASET_SPLIT_SEED)
        )
        tasks.append(
          Task(
            train=train,
            test=test,
          )
        )

    return tasks

def get_single_label_mnist(num_tasks: int):
    transform = Compose([
        Flatten(),
        Normalize(),
    ])
    return [
        Task(
          train= SingleLabelMNIST(
            root="/scratch/shared/beegfs/user/datasets",
            label=task,
            train=True,
            transform=transform,
            download=False
          ),
          test= SingleLabelMNIST(
            root="/scratch/shared/beegfs/user/datasets",
            label=task,
            train=False,
            transform=transform,
            download=False
          )
        )
        for task in range(num_tasks)
    ]

def get_single_label_not_mnist(
  num_tasks: int
):
    TRAIN_ROOT = "/scratch/shared/beegfs/user/datasets/datasets/jwjohnson314/notmnist/versions/2/notMNIST_large/notMNIST_large"
    TEST_ROOT = "/scratch/shared/beegfs/user/datasets/datasets/jwjohnson314/notmnist/versions/2/notMNIST_small/notMNIST_small"
    TARGET_DATASET_SIZE = 400000 

    transform = Compose([
        Flatten(),
        Normalize(),
    ])

    tasks: List[Task] = []

    for task in range(num_tasks):
        task_train_dataset = SingleLabelNotMNIST(
            root=TRAIN_ROOT,
            transform=transform,
            task=task
        )
        task_train_dataset = subsample_dataset(
            task_train_dataset,
            min(
                len(task_train_dataset),
                TARGET_DATASET_SIZE // num_tasks
            )
        )
        task_test_dataset = SingleLabelNotMNIST(
            root=TEST_ROOT,
            transform=transform,
            task=task
        )
        tasks.append(
          Task(
            train=task_train_dataset,
            test=task_test_dataset,
          )
        )

    return tasks