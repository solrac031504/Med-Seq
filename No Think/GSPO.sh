#!/bin/bash
#SBATCH --job-name=gspo_train
#SBATCH --mem=32G
#SBATCH --constraint=gpu32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/gspo_train.%J.out
#SBATCH --error=logs/gspo_train.%J.err
#SBATCH --gres=gpu:1


echo "Starting GSPO training job on GPU node..."

# Activate your virtual environment
source /home/in642270/.venvs/gspo/bin/activate

# Move to project directory
cd /lustre/fs1/home/in642270/GSPO_MED/

# Run the GSPO training script (using the config file)
python main.py \
  --config gspo_config.yaml

echo "Training job completed!"

cp -r /lustre/fs1/home/in642270/GSPO_MED/outputs/* /lustre/fs1/home/in642270/GSPO_MED/checkpoints/
mkdir -p /lustre/fs1/home/in642270/GSPO_MED/checkpoints/best_checkpoint
mv /lustre/fs1/home/in642270/GSPO_MED/checkpoints/*.* /lustre/fs1/home/in642270/GSPO_MED/checkpoints/best_checkpoint/
rm -rf /lustre/fs1/home/in642270/GSPO_MED/outputs/*
