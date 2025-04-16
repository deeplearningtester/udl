import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="SI")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["permuted_mnist", "split_mnist"],
        required=True,
        help="Benchmark to use: 'permuted_mnist' or 'split_mnist'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed"
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        required=True,
        help="Path to the experiment folder"
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="classification or regression objective"
    )
    return parser.parse_args()