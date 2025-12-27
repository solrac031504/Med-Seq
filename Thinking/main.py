import os
import re
import yaml
import torch
import argparse
from gspo_trainer import GSPOTrainer


def extract_final_answer(text: str) -> str:
    """
    Extracts only the final answer letter from model output.
    Example:
        <think>reasoning</think><answer>B</answer> -> 'B'
    """
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", str(text), re.IGNORECASE)
    return match.group(1).upper().strip() if match else ""


def main():
    parser = argparse.ArgumentParser(description="Run GSPO training.")
    parser.add_argument("--config", type=str, default="configs/gspo_config.yaml", help="Path to config YAML file.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Run mode: train or eval.")
    parser.add_argument("--show_thinking", action="store_true", help="Show full model reasoning (<think>...</think>) during eval.")
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
    trainer = GSPOTrainer(config=config, debug=True)

    # -----------------------
    # Mode: train / eval
    # -----------------------
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        print("[EVAL] Starting evaluation using best checkpoint...")
    
        # Run model generation for test cases (this assumes your trainer has eval dataset)
        results = []
        for i, sample in enumerate(trainer.eval_dataset):
            inputs = {k: v.unsqueeze(0).to(trainer.device) if torch.is_tensor(v) else v for k, v in sample.items()}
            with torch.no_grad():
                outputs = trainer.model.generate(**inputs, max_new_tokens=256)
            decoded = trainer.processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
            # Choose what to display
            if args.show_thinking:
                print(f"\n[Sample {i}] Full Output:\n{decoded}")
            else:
                print(f"\n[Sample {i}] Final Answer: {extract_final_answer(decoded)}")
    
            results.append(decoded)
            if i >= 2:  # show first few examples
                break
    
        print("\n[EVAL] Completed reasoning preview.")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("\n==============================")
    print(f"{args.mode.capitalize()} job completed successfully!")
    print("==============================\n")


if __name__ == "__main__":
    main()
