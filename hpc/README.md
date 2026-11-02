# Running on NYU HPC

Do the setup once, then submit jobs from the repo root on `/scratch`.

## Setup

Run this in a shell on a node where containers are available, following the HPC guide up to the Singularity step. Put the overlay and `.sif` in `/scratch/$USER/resnet-5m-env/`, or set `OVERLAY` and `SIF` to where they are.

Inside the container, with the overlay mounted `:rw`, install Miniconda to `/ext3/miniconda3`, create `/ext3/env.sh`, then:

```bash
source /ext3/env.sh
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

From the repo root, download CIFAR-10 once so parallel jobs don't race:

```bash
python -c "from torchvision import datasets; datasets.CIFAR10('data', download=True)"
```

## Jobs

Use your own course or project allocation:

```bash
export ACCOUNT=<your slurm account> PARTITION=<a GPU partition>
sbatch --account=$ACCOUNT --partition=$PARTITION hpc/train.sbatch --epochs 1 --out checkpoints/smoke.pt   # check first
hpc/round2_teachers.sh
```

A job stopped by its time limit is continued by submitting the same command again.
