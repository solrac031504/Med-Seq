#!/bin/bash
#SBATCH --job-name=gspo_eval
#SBATCH --mem=32G
#SBATCH --constraint=gpu32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/gspo_eval.%J.out
#SBATCH --error=logs/gspo_eval.%J.err
#SBATCH --gres=gpu:1

echo "Starting GSPO evaluation job on GPU node..."

# Activate virtual environment
source /home/in642270/.venvs/gspo/bin/activate

# Move to project directory
cd /lustre/fs1/home/in642270/GSPO_MED/

# -----------------------------
# Base paths
# -----------------------------
BASE_DIR="/lustre/fs1/home/in642270/GSPO_MED"
CHECKPOINT_BASE="$BASE_DIR/all_best_checkpoints"
OUTPUT_DIR="$BASE_DIR/outputs"
MODALITY_TEST="$BASE_DIR/Splits/modality/test"
QUESTION_TYPE_TEST="$BASE_DIR/Splits/question_type/test"
IMAGE_ROOT="$BASE_DIR/Images"
EVAL_SCRIPT="$BASE_DIR/Evaluation/test_qwen2vl_vqa_nothink.py"

# -----------------------------
# Short names for checkpoints
# -----------------------------
declare -A CKPT_SHORT
CKPT_SHORT["best_checkpoint_CT"]="CT"
CKPT_SHORT["best_checkpoint_Microscopy"]="Microscopy"
CKPT_SHORT["best_checkpoint_Other Biological Attributes"]="OtherBio"
CKPT_SHORT["best_checkpointAnatomy Identification"]="AnatomyID"

# -----------------------------
# 1?? Run modality test evaluations
# -----------------------------
for CKPT_NAME in "best_checkpoint_CT" "best_checkpoint_Microscopy"; do
  CKPT_PATH="$CHECKPOINT_BASE/$CKPT_NAME"
  SHORT_TRAIN=${CKPT_SHORT[$CKPT_NAME]}
  echo ">>> Evaluating checkpoint: $CKPT_NAME ($SHORT_TRAIN)"
  
  for JSON_FILE in "$MODALITY_TEST"/*.json; do
    JSON_NAME=$(basename "$JSON_FILE" .json)
    
    # Clean test name (remove brackets and spaces)
    SHORT_TEST=$(echo "$JSON_NAME" | sed 's/(.*)//; s/ //g; s/_test//')
    
    OUTPUT_PATH="$OUTPUT_DIR/train_on_${SHORT_TRAIN}_tested_on_${SHORT_TEST}.json"
    echo "Running $SHORT_TRAIN on $SHORT_TEST ..."
    
    python "$EVAL_SCRIPT" \
      --checkpoint_path "$CKPT_PATH" \
      --test_json "$JSON_FILE" \
      --image_root "$IMAGE_ROOT" \
      --output_path "$OUTPUT_PATH" \
      --batch_size 1
  done
done

# -----------------------------
# 2?? Run question-type test evaluations
# -----------------------------
for CKPT_NAME in "best_checkpoint_Other Biological Attributes" "best_checkpointAnatomy Identification"; do
  CKPT_PATH="$CHECKPOINT_BASE/$CKPT_NAME"
  SHORT_TRAIN=${CKPT_SHORT[$CKPT_NAME]}
  echo ">>> Evaluating checkpoint: $CKPT_NAME ($SHORT_TRAIN)"
  
  for JSON_FILE in "$QUESTION_TYPE_TEST"/*.json; do
    JSON_NAME=$(basename "$JSON_FILE" .json)
    
    SHORT_TEST=$(echo "$JSON_NAME" | sed 's/_test//; s/ /_/g')
    
    OUTPUT_PATH="$OUTPUT_DIR/train_on_${SHORT_TRAIN}_tested_on_${SHORT_TEST}.json"
    echo "Running $SHORT_TRAIN on $SHORT_TEST ..."
    
    python "$EVAL_SCRIPT" \
      --checkpoint_path "$CKPT_PATH" \
      --test_json "$JSON_FILE" \
      --image_root "$IMAGE_ROOT" \
      --output_path "$OUTPUT_PATH" \
      --batch_size 1
  done
done

echo "=== All evaluations completed successfully! ==="