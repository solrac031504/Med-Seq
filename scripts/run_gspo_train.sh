#!/bin/bash
#SBATCH --job-name=gspo_qwen2vl
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=Evaluation/gspo_train_%J.out
#SBATCH --error=Evaluation/gspo_train_%J.err
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# ============================================================
#  Environment Setup
# ============================================================

echo "Activating conda environment..."
source ~/.bashrc
conda activate base

echo "Setting CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "Current CUDA version:"
nvcc --version || echo "nvcc not found, but proceeding..."

# ============================================================
#  Training Parameters
# ============================================================

PROJECT_ROOT="/lustre/fs1/home/ca362088/Med_GSPO"
CONFIG_PATH="$PROJECT_ROOT/configs/gspo_config.yaml"
SCRIPT_PATH="$PROJECT_ROOT/src/gspo_vqa_nothink.py"

mkdir -p "$PROJECT_ROOT/Evaluation"
mkdir -p "$PROJECT_ROOT/outputs_gspo_vqa"

echo "Running GSPO Training on Qwen/Qwen2-VL-2B..."
echo "Config: $CONFIG_PATH"
echo "Script: $SCRIPT_PATH"

# ============================================================
#  Run the GSPO training
# ============================================================

python3 "$SCRIPT_PATH" \
    --config "$CONFIG_PATH" \
    --model_name_or_path "Qwen/Qwen2-VL-2B" \
    --train_data_dir "$PROJECT_ROOT/Splits/modality/train" \
    --image_root "$PROJECT_ROOT/Images" \
    --output_dir "$PROJECT_ROOT/outputs_gspo_vqa" \
    --num_generations 2 \
    --max_steps 500 \
    --learning_rate 1e-5 \
    --bf16 True \
    --gradient_accumulation_steps 8 \
    --report_to none \
    --seed 42

echo "Training complete. Check logs in $PROJECT_ROOT/Evaluation."
