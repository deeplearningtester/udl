import argparse
import pathlib
import sys
from enum import StrEnum
from pathlib import Path
from typing import Optional

import torch
from alg import (CoresetMethod, coreset_only, ewc, mle, vae_ewc, vae_naive,
                 vae_vcl, vcl, vcl_with_coreset)
from generative_models import VAE, MeanFieldDecoderTail, NaiveDecoderTail
from models import DiscriminativeMeanField, DiscriminativeNaive
from tasks import (get_permuted_mnist, get_single_label_mnist,
                   get_single_label_not_mnist, get_split_mnist,
                   get_split_not_mnist)
from torch.utils.data import DataLoader
from utils import evaluate_single_task, seed_everything


class Model(StrEnum):
    DISCRIMINATIVE_NAIVE = "discriminative_naive"
    DISCRIMINATIVE_MEAN_FIELD= "discriminative_mean_field" 
    GENERATIVE_NAIVE = "generative_naive"
    GENERATIVE_MEAN_FIELD = "generative_mean_field"

class Method(StrEnum):
    NAIVE = "naive"
    VCL = "vcl"
    VCL_RANDOM_CORESET = "vcl+random_coreset"
    VCL_K_CENTER_CORESET = "vcl+k_center_coreset"
    RANDOM_CORESET = "random_coreset"
    K_CENTER_CORESET = "k_center_coreset"
    EWC = "ewc"
    SI = "si"

class Benchmark(StrEnum):
    MNIST = "mnist"
    NOT_MNIST = "not_mnist"
    PERMUTED_MNIST = "permuted_mnist"
    SPLIT_MNIST = "split_mnist"
    SPLIT_NOT_MNIST = "split_not_mnist"

def discriminative_experiment(
    model: Model,
    method: Method,
    benchmark: Benchmark, 
    experiment_path: Path,
    device: torch.device,
    seed: int,
    coreset_size: Optional[int] = None,
    coreset_balanced: bool = False,
    lambd: Optional[float] = None,
    required_extension: bool = False
):
  if benchmark == Benchmark.PERMUTED_MNIST:
    # model config
    num_classes = 10
    num_hidden_layers = 2
    num_tasks = 10
    num_heads = 1
    hidden_dim = 100
    input_dim = 28 * 28
    output_dim = num_classes
    # training config
    num_epochs = 100
    batch_size = 256
    lr = 0.001
    num_train_monte_carlo_samples = 10
    # test config
    num_test_monte_carlo_samples = 100
    if method == Method.EWC:
      batch_size = 200
      num_epochs = 20
      num_ewc_samples = 600
  if benchmark == Benchmark.SPLIT_MNIST or benchmark == Benchmark.SPLIT_NOT_MNIST: 
    # model config
    num_classes = 2
    num_tasks = 5
    num_heads = 5
    if benchmark == Benchmark.SPLIT_MNIST:
      num_hidden_layers = 2
      hidden_dim = 256
    else: 
      num_hidden_layers = 4
      hidden_dim = 150
    input_dim = 28 * 28
    output_dim = num_classes
    # training config
    num_epochs = 120
    batch_size = 60000 # batch size is training set size
    lr = 0.001
    num_train_monte_carlo_samples = 10
    # test config
    num_test_monte_carlo_samples = 100
    if method == Method.EWC:
      num_ewc_samples = 200

  seed_everything(seed)
  if "coreset" in method:
    assert coreset_size is not None
    method_name = f"{method}_coreset_size={coreset_size}_coreset_balanced={coreset_balanced}"
  elif "ewc" in method:
    assert lambd is not None
    method_name = f"{method}_lambda={lambd}"
  elif "si" in method:
    assert lambd is not None
    method_name = f"{method}_lambda={lambd}"
  else:
    method_name = method
  experiment_path = experiment_path / benchmark / model / method_name / f"seed={seed}"
  experiment_path.mkdir(exist_ok=True, parents=True)
  match benchmark:
    case Benchmark.PERMUTED_MNIST:
      tasks = get_permuted_mnist(num_tasks)
    case Benchmark.SPLIT_MNIST:
      tasks = get_split_mnist(num_tasks)
    case Benchmark.SPLIT_NOT_MNIST:
      tasks = get_split_not_mnist(num_tasks)
  match model:
    case Model.DISCRIMINATIVE_MEAN_FIELD:
      model = DiscriminativeMeanField(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        objective=(
          "regression" if required_extension else "classification"
        )
      )
    case Model.DISCRIMINATIVE_NAIVE:
      model = DiscriminativeNaive(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden_layers,
        num_heads=num_heads,
        objective=(
          "regression" if required_extension else "classification"
        )
      )
  # print(required_extension)
  # print(model.objective)
  model = model.to(device)
  should_init_mle = (
    method != Method.RANDOM_CORESET and 
    method != Method.K_CENTER_CORESET and 
    method != Method.EWC and
    method != Method.SI
  )
  if should_init_mle:
    # MLE init
    mle(
      model,
      tasks[0].train,
      num_epochs,
      batch_size,
      device,
      lr,
      experiment_path
    )
    metric = evaluate_single_task(
        model,
        task=0,
        test_dataloader=DataLoader(
            dataset=tasks[0].test,
            batch_size=batch_size,
            shuffle=False
        ),
        device=device,
        num_monte_carlo_samples=num_test_monte_carlo_samples
    )
    if required_extension:
      print(f"[mle] [task=0] rmse: {metric}")
    else:
      print(f"[mle] [task=0] acc: {metric}")
  match method:
    case Method.VCL:
      history = vcl(
          model,
          tasks,
          num_epochs,
          batch_size,
          lr,
          num_train_monte_carlo_samples,
          num_test_monte_carlo_samples,
          device,
          experiment_path
      )
    case Method.VCL_RANDOM_CORESET:
      assert coreset_size is not None
      history = vcl_with_coreset(
          model,
          tasks,
          CoresetMethod.RANDOM,
          coreset_size,
          num_epochs,
          batch_size,
          lr,
          num_train_monte_carlo_samples,
          num_test_monte_carlo_samples,
          device,
          experiment_path,
          coreset_balanced
      )
    case Method.VCL_K_CENTER_CORESET:
      assert coreset_size is not None
      history = vcl_with_coreset(
          model,
          tasks,
          CoresetMethod.K_CENTER,
          coreset_size,
          num_epochs,
          batch_size,
          lr,
          num_train_monte_carlo_samples,
          num_test_monte_carlo_samples,
          device,
          experiment_path,
          coreset_balanced
      )
    case Method.RANDOM_CORESET:
      assert coreset_size is not None
      history = coreset_only(
          model,
          tasks,
          CoresetMethod.RANDOM,
          coreset_size,
          num_epochs,
          lr,
          num_train_monte_carlo_samples,
          num_test_monte_carlo_samples,
          device,
          experiment_path,
          coreset_balanced
      )  
    case Method.K_CENTER_CORESET:
      assert coreset_size is not None
      history = coreset_only(
          model,
          tasks,
          CoresetMethod.K_CENTER,
          coreset_size,
          num_epochs,
          lr,
          num_train_monte_carlo_samples,
          num_test_monte_carlo_samples,
          device,
          experiment_path,
          coreset_balanced
      )
    case Method.EWC:
      history = ewc(
        model,
        tasks,
        num_epochs,
        batch_size,
        lr,
        num_ewc_samples,
        num_test_monte_carlo_samples,
        device,
        experiment_path,
        lambd,
        seed
      )
    case Method.SI:
      history = []
        
  torch.save(history, experiment_path / f"history.pth")

def generative_experiment(
    model: Model,
    method: Method,
    benchmark: Benchmark, 
    experiment_path: Path,
    device: torch.device,
    seed: int,
    coreset_size: Optional[int] = None,
    coreset_balanced: bool = False,
    lambd: Optional[float] = None,
    batch_size: int = 64,
):
  if benchmark == Benchmark.MNIST:
    # model config
    num_tasks = 10
    hidden_dim = 500
    input_dim = 28 * 28
    latent_dim = 50
    num_epochs = 200
    lr = 0.0001

  if benchmark == Benchmark.NOT_MNIST:
    # model config
    num_tasks = 10
    hidden_dim = 500
    input_dim = 28 * 28
    latent_dim = 50
    num_epochs = 400
    lr = 0.0001    

  seed_everything(seed)
  if "coreset" in method:
    assert coreset_size is not None
    method_name = f"{method}_coreset_size={coreset_size}_coreset_balanced={coreset_balanced}"
  elif "ewc" in method:
    assert lambd is not None
    method_name = f"{method}_lambda={lambd}"
  elif "si" in method:
    assert lambd is not None
    method_name = f"{method}_lambda={lambd}"
  else:
    method_name = method

  experiment_path = experiment_path / benchmark / model / method_name / f"seed={seed}" / f"batch_size={batch_size}"
  experiment_path.mkdir(exist_ok=True, parents=True)
  match benchmark:
    case Benchmark.MNIST:
      tasks = get_single_label_mnist(num_tasks)
    case Benchmark.NOT_MNIST:
      tasks = get_single_label_not_mnist(num_tasks)

  match method:
    case Method.NAIVE:
      decoder_tail = NaiveDecoderTail(
        hidden_dim=hidden_dim,
        output_dim=input_dim
      )
    case Method.EWC:
      decoder_tail = NaiveDecoderTail(
        hidden_dim=hidden_dim,
        output_dim=input_dim
      )
    case Method.VCL:
      decoder_tail = MeanFieldDecoderTail(
        hidden_dim=hidden_dim,
        output_dim=input_dim
      )

  model = VAE(
    num_tasks=num_tasks,
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    decoder_tail=decoder_tail
  )

  print("model:")
  print(model)
  print("-------------------")

  model = model.to(device)
  match method:
    case Method.NAIVE:
      vae_naive(
        model,
        tasks,
        num_epochs,
        batch_size,
        lr,
        experiment_path,
        device
      )
    case Method.VCL:
      vae_vcl(
        model,
        tasks,
        num_epochs,
        batch_size,
        lr,
        experiment_path,
        device
      )
    case Method.EWC:
      vae_ewc(
        model,
        tasks,
        num_epochs,
        batch_size,
        lr,
        lambd,
        batch_size,
        experiment_path,
        device,
        seed
      )

def create_parser():
    parser = argparse.ArgumentParser(description='Run experiments with various models, methods, and benchmarks.')
    # Add arguments for model, method, and benchmark
    parser.add_argument('--model', 
                        type=Model, 
                        choices=list(Model), 
                        default=Model.DISCRIMINATIVE_MEAN_FIELD,
                        help='Model to use for the experiment')
    parser.add_argument('--method', 
                        type=Method, 
                        choices=list(Method), 
                        default=Method.VCL,
                        help='Method to use for the experiment')
    parser.add_argument('--benchmark', 
                        type=Benchmark, 
                        choices=list(Benchmark), 
                        default=Benchmark.PERMUTED_MNIST,
                        help='Benchmark dataset to use')
    # Add experiment directory argument parsed as a path
    parser.add_argument('--experiment-path',
                        type=pathlib.Path,
                        default=pathlib.Path('./experiments'),
                        help='Directory to store experiment results')
    # Add device argument for GPU index
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='GPU device index to use (default: 0)')
    # Add optional coreset size parameter
    parser.add_argument('--coreset-size',
                        type=int,
                        default=None,
                        help='Size of the coreset (default: None)')
    # Add optional coreset-balanced parameter
    parser.add_argument('--coreset-balanced',
                        action='store_true',
                        help='Enable balanced coreset selection (default: False)')
    parser.add_argument('--seed',
                        type=int,
                        default=17,
                        help='Seed')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch Size',
                        required=False)
    parser.add_argument('--lambd',
                      type=lambda x: float(x) if x is not None else None,
                      default=None,
                      help='Optional lambda parameter (default: None)')
    parser.add_argument('--required_extension',
                        action='store_true',
                        help='Enable required extension (default: False)')
    return parser

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    parser = create_parser()
    args = parser.parse_args()
    
    print(f"Selected model: {args.model}")
    print(f"Selected method: {args.method}")
    print(f"Coreset_size: {args.coreset_size}")
    print(f"Coreset balanced: {args.coreset_balanced}")
    print(f"Selected benchmark: {args.benchmark}")
    print(f"Directory: {args.experiment_path}")
    print(f"Seed: {args.seed}")
    print(f"Lambda: {args.lambd}")
    print(f"Batch Size: {args.batch_size}")

    device = torch.device(f"cuda:{args.device}")

    if "discriminative" in args.model:
      discriminative_experiment(
        args.model,
        args.method,
        args.benchmark,
        args.experiment_path,
        device,
        args.seed,
        args.coreset_size,
        args.coreset_balanced,
        args.lambd,
        args.required_extension
      )
    else:
      generative_experiment(
        args.model,
        args.method,
        args.benchmark,
        args.experiment_path,
        device,
        args.seed,
        args.coreset_size,
        args.coreset_balanced,
        args.lambd,
        args.batch_size
      )