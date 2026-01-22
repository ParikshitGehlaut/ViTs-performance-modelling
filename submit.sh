#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=vits_%j.out
#SBATCH --job-name=vits

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

module purge
module load anaconda/anaconda-24.1.1
source /export/apps/anaconda/anaconda-24.1.1/etc/profile.d/conda.sh

# Use the full path to the conda env
CONDA_ENV="$HOME/.conda/envs/vits-neusight"
export PATH="$CONDA_ENV/bin:$PATH"

echo "Using python: $(which python)"
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

cd /export/home/nikhil/parikshit/research/ViTs-performance-modelling

# Run inference with environment Python
CUDA_VISIBLE_DEVICES=1 bash src/models/Paligemma/launch_inference.sh

echo "Job finished at $(date)"
