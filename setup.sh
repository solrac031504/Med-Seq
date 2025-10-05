#!/bin/bash
#SBATCH --job-name=gspo_setup
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=Evaluation/setup_%J.out
#SBATCH --error=Evaluation/setup_%J.err
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# ============================================================
#  Setup environment for GSPO_MED
# ============================================================

echo "=== Starting environment setup for GSPO_MED ==="

# Load environment
source ~/.bashrc
conda activate base

# Ensure working directory
cd /lustre/fs1/home/ca362088/Med_GSPO || exit
mkdir -p Evaluation

# CUDA configuration
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA environment set to: $CUDA_HOME"

# ============================================================
#  Python package setup
# ============================================================

echo "Installing core dependencies..."

# Upgrade pip to latest
pip install --upgrade pip setuptools wheel

# Core libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face ecosystem
pip install transformers==4.44.2 accelerate==0.34.2 datasets==2.21.0
pip install peft==0.11.1 trl==0.9.6
pip install evaluate wandb tensorboardx

# Vision + utilities
pip install pillow opencv-python tqdm pyyaml
pip install qwen_vl_utils
pip install flash-attn --no-build-isolation

# vLLM optional support
pip install vllm==0.7.2

# ============================================================
#  Development install (if src/ exists)
# ============================================================

if [ -d "src" ]; then
    echo "Installing project source in editable mode..."
    cd src
    pip install -e .
    cd ..
fi

# ============================================================
#  Verification
# ============================================================

echo "Verifying installation..."
python3 -c "import torch, transformers, trl; print('✅ Torch:', torch.__version__, '| ✅ Transformers:', transformers.__version__, '| ✅ TRL:', trl.__version__)"
nvidia-smi

echo "=== Environment setup complete! ==="
