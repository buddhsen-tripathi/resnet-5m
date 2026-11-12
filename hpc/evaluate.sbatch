#!/bin/bash
# Flip-averaged CIFAR-10 test accuracy of the given checkpoints and their ensemble.
#   sbatch --account=$ACCOUNT --partition=$PARTITION hpc/evaluate.sbatch checkpoints/a.pt checkpoints/b.pt
#SBATCH --job-name=resnet-5m-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=00:20:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

.venv/bin/python evaluate.py "$@"
