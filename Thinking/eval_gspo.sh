#!/bin/bash
#SBATCH --job-name=gspo_eval
#SBATCH --mem=32G
#SBATCH --constraint=gpu32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/gspo_eval.%J.out
#SBATCH --error=logs/gspo_eval.%J.err
#SBATCH --gres=gpu:1

echo "Starting GSPO evaluation job on GPU node..."

# Activate virtual environment
source /home/in642270/.venvs/gspo/bin/activate

# Move to project directory
cd /lustre/fs1/home/in642270/GSPO_MED/

# Define paths (pass checkpoint directly)
CHECKPOINT_PATH="/lustre/fs1/home/in642270/GSPO_MED/checkpoints/best_checkpoint"
TEST_JSON="/lustre/fs1/home/in642270/GSPO_MED/Splits/question_type/test/Other Biological Attributes_test.json"
IMAGE_ROOT="/lustre/fs1/home/in642270/GSPO_MED/Images"
OUTPUT_PATH="/lustre/fs1/home/in642270/GSPO_MED/outputs/eval_results.json"

# Run the evaluation script
python Evaluation/test_qwen2vl_vqa_think.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --test_json "$TEST_JSON" \
  --image_root "$IMAGE_ROOT" \
  --output_path "$OUTPUT_PATH" \
  --batch_size 1

echo "Evaluation job completed!"