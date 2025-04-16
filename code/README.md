# Variational Continual Learning

Due to submission file size limitations, we were not able to release the checkpoints (11GB in total). 
However, we release the reproduction scripts. Our experiments are adapted for the use on SLURM. 

## Setup
We use conda for environment management. To reproduce VCL and EWC results, it is necessary to create an environment using the provided environment.yml configuration in the project root.

## VCL + EWC

To reproduce our results, run the appropriate SLURM script.

**PermutedMNIST**
- sbatch submit_vcl_permuted_mnist.sh
- sbatch submit_vcl_permuted_mnist_required_extension.sh
- sbatch submit_ewc_permuted_mnist.sh
- sbatch submit_ewc_permuted_mnist_required_extension.sh

**SplitMNIST**
- sbatch submit_vcl_split_mnist.sh
- sbatch submit_vcl_split_mnist_required_extension.sh
- sbatch submit_ewc_split_mnist.sh
- sbatch submit_ewc_split_mnist_required_extension.sh

**MNIST**
- sbatch submit_vae_naive_mnist.sh
- sbatch submit_vae_vcl_mnist.sh

## SI

To reproduce SI results, it is necessary to create an environment using the provided si/environment.yml.

To reproduce classification on **PermutedMNIST** and **SplitMNIST**, run `bash si/reproduce_local.sh`.

To reproduce regression on **PermutedMNIST** and **SplitMNIST**, run `bash si/reproduce_local_required_extension.sh`.
