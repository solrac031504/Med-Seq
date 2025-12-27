import os
import yaml
import torch
import argparse
from gspo_trainer import GSPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Run GSPO training.")
    parser.add_argument("--config", type=str, default="configs/gspo_config.yaml", help="Path to config YAML file.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Run mode: train or eval.")
    args = parser.parse_args()

    # -----------------------
    # Environment setup
    # -----------------------
    print("\n==============================")
    print(f"Starting GSPO job in {args.mode.upper()} mode...")
    print("==============================\n")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_data_dir = config.get("train_data_dir")
    image_root = config.get("image_root")

    print(f"[CONFIG] Using config: {args.config}")
    print(f"[DATA] Training data directory: {train_data_dir}")
    print(f"[DATA] Image root directory: {image_root}")
    print(f"[DEVICE] Detected: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CUDA setup
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # -----------------------
    # Initialize trainer
    # -----------------------
    trainer = GSPOTrainer(config=config, debug=False)

    # -----------------------
    # Mode: train / eval
    # -----------------------
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        print("[EVAL] Starting evaluation using best checkpoint...")
        acc = trainer._evaluate()
        print(f"[EVAL] Final accuracy: {acc:.2f}%")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("\n==============================")
    print(f"{args.mode.capitalize()} job completed successfully!")
    print("==============================\n")


if __name__ == "__main__":
    main()
