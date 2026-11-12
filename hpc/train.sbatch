#!/bin/bash
# Run train.py on one GPU with the repo's .venv (built by hpc/setup.sh). Any arguments are passed
# to train.py; always give --out. --resume is added, so a timed-out or preempted job continues
# when the same command is submitted again.
#
#   sbatch --account=$ACCOUNT --partition=$PARTITION hpc/train.sbatch --out checkpoints/model.pt
#
#SBATCH --job-name=resnet-5m
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --requeue

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
.venv/bin/python train.py --resume --workers "$(( SLURM_CPUS_PER_TASK - 1 ))" "$@"
