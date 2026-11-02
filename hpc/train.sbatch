#!/bin/bash
# Run train.py on one GPU. Any arguments are passed to train.py; always give --out.
# --resume makes requeues and timed-out jobs continue: resubmit the same command to carry on.
#
#   sbatch --account=$ACCOUNT --partition=$PARTITION hpc/train.sbatch --out checkpoints/model.pt
#
#SBATCH --job-name=resnet-5m
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --requeue

SIF=${SIF:-/scratch/$USER/resnet-5m-env/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif}
OVERLAY=${OVERLAY:-/scratch/$USER/resnet-5m-env/overlay-15GB-500K.ext3}

module purge
singularity exec --nv --bind /scratch --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c \
    'source /ext3/env.sh; python train.py "$@"' _ --resume --workers "$SLURM_CPUS_PER_TASK" "$@"
