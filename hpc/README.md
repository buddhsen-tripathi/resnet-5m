# Running on NYU HPC

Jobs run on the Google Cloud burst partitions of NYU HPC Cloud Bursting. Burst compute nodes have their own `/home` and `/scratch`, separate from Torch's, plus internet access, so everything is set up on the burst side: a uv virtual environment, CIFAR-10 and the teacher checkpoints.

## Connect

For Fall 2026, cloud bursting is reached only through the HPC Cloud Bursting Open OnDemand (OOD) site (NYU VPN off campus). Do not use the Torch cluster's own GPUs for coursework.

1. In OOD, start a **Jupyter** session on a CPU partition (`interactive` or `n2c48m24`) with account `cs_gy_6923-2026fa`. Don't put the notebook itself on a GPU: GPU instances idle for 20 minutes are shut down, and the hours count against the 300-hour allocation.
2. Open a terminal from Jupyter (New → Terminal). It runs on a burst compute node, which sees the burst `/scratch` and can submit batch jobs.

```bash
export ACCOUNT=cs_gy_6923-2026fa PARTITION=c12m85-a100-1   # or g2-standard-12 for one L4
```

## Setup (once)

Put the repo at `/scratch/$USER/resnet-5m`, either by cloning it in the Jupyter terminal or by uploading a tarball through the OOD file browser and extracting it there:

```bash
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone https://github.com/buddhsen-tripathi/resnet-5m.git && cd resnet-5m
bash hpc/setup.sh
```

`hpc/setup.sh` installs uv and a CUDA 12.6 PyTorch build into `.venv` (set `TORCH_INDEX` for another CUDA version), downloads CIFAR-10 and the ten round 9 teachers from the GitHub releases, and runs the unit tests. It skips anything already done.

Before a long run, check the GPU path, memory and epoch time:

```bash
SMOKE=1 bash hpc/round10.sh     # one epoch of the round 10 recipe; read slurm-smoke-<jobid>.out
```

## Jobs

Submit from the Jupyter terminal in the repo directory, and check progress with `squeue --me`. Batch jobs keep running after the Jupyter session ends.

`hpc/train.sbatch` runs `train.py` on one GPU with the given arguments and always adds `--resume`. A job stopped by its time limit or preempted is continued by submitting the same command again. It requests 12 CPUs, which matches the A100 and single-L4 machine types.

| Script | Purpose |
|---|---|
| `hpc/round10.sh` | Round 10: `R10_t12` (R9 recipe, 12 teachers) and `R10_shift` (plus resampling and brightness/contrast augmentation), each a chain of `SEGMENTS` (default 2) 4-hour jobs, then a flip-TTA evaluation job |
| `hpc/round2_teachers.sh` | Rebuild the five round 2 teachers |
| `hpc/evaluate.sbatch` | CIFAR-10 test accuracy with flip TTA for the given checkpoints and their ensemble |

Download checkpoints and `slurm-*.out` logs through the OOD file browser. Copying from Torch goes through `dtn.torch.hpc.nyu.edu` and needs two-factor authentication.
