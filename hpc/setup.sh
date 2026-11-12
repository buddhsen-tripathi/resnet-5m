#!/bin/bash
# One-time setup on an NYU burst (Google Cloud) compute node, run from the repo root on the
# burst /scratch. Burst nodes have internet access but cannot see Torch's /scratch, so this
# builds everything there: a uv virtual environment with CUDA PyTorch, CIFAR-10, and the ten
# round 9 teachers from the GitHub releases. Safe to rerun: finished steps are skipped.
# In an Open OnDemand Jupyter terminal:  cd /scratch/$USER/resnet-5m && bash hpc/setup.sh
set -euo pipefail
export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}
export UV_INSTALL_DIR=${UV_INSTALL_DIR:-/scratch/$USER/.local/bin}
export PATH="$UV_INSTALL_DIR:$PATH"
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}  # CUDA 12.6 runs on drivers >= 525
RELEASES=https://github.com/buddhsen-tripathi/resnet-5m/releases/download

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
.venv/bin/python -c "import torch, torchvision" 2>/dev/null || \
  uv pip install --python .venv/bin/python numpy torch torchvision --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple
.venv/bin/python -c "import torch, torchvision; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)"

mkdir -p teachers
for spec in v0.1/A_nose_ta_mix v0.1/C_mix v0.1/C_ta v0.1/C_ta_mix v0.1/C_ta_mix_noerase v0.1/KD_C_aug \
            v0.2/KD5_full v0.2/KD5_full_cutout v0.2/S2_highres v0.2/S4_deep; do
  file=teachers/$(basename "$spec").pt
  [ -s "$file" ] || curl -fsSL -o "$file" "$RELEASES/$spec.pt"
done
echo "teachers: $(ls teachers | wc -l) (plus weights/R8_base_300.pt and weights/R9_combined.pt)"

.venv/bin/python -c "from torchvision import datasets; datasets.CIFAR10('data', train=True, download=True); datasets.CIFAR10('data', train=False, download=True)"
.venv/bin/python -m unittest -q tests.test_training 2>&1 | tail -3
echo "Setup done. Next: the one-epoch GPU smoke test in hpc/README.md."
