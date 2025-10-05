# gspo_vqa_nothink.py
# Main entry point for GSPO training on Qwen/Qwen2-VL-2B using "no-think" strategy

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from trainer.gspo_trainer import GSPOConfig, Qwen2VLGSPOTrainer, QwenVLChatCollator


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class MedVQADataset(Dataset):
    """
    Dataset that reads all JSONs from Splits/modality/train/
    Each JSON entry: {"image": <filename>, "problem": <question>, "solution": <optional>}
    """
    def __init__(self, split_dir: str, image_root: str):
        self.samples = []
        self.image_root = image_root

        # Collect all JSONs from modality/train
        json_files = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".json")
        ]

        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
                # Some of these JSONs may contain lists or dicts depending on dataset
                if isinstance(data, dict):
                    data = data.get("data", []) or data.get("annotations", [])
                for ex in data:
                    img_name = ex.get("image", None) or ex.get("img_path", None)
                    q = ex.get("question", None) or ex.get("problem", None)
                    if img_name and q:
                        img_path = os.path.join(self.image_root, img_name)
                        self.samples.append({
                            "image": img_path,
                            "problem": q,
                            "solution": ex.get("solution", "")
                        })
            except Exception as e:
                print(f"Skipping {jf}: {e}")

        print(f"Loaded {len(self.samples)} samples from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    # Paths
    base_dir = "/lustre/fs1/home/in642270/GSPO_MED"
    image_root = os.path.join(base_dir, "Images")
    split_dir = os.path.join(base_dir, "Splits/modality/train")

    # Model
    model_name = "Qwen/Qwen2-VL-2B"
    print(f"Loading model and processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto"
    )

    # Dataset
    dataset = MedVQADataset(split_dir=split_dir, image_root=image_root)

    # DataLoader
    collator = QwenVLChatCollator(processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator)

    # Config
    cfg = GSPOConfig(
        max_new_tokens=64,
        num_generations=2,       # group size K
        learning_rate=1e-5,
        max_steps=500,           # adjust depending on data size
        log_every=10,
        save_every=100,
        output_dir=os.path.join(base_dir, "outputs_gspo_vqa"),
    )

    # Trainer
    trainer = Qwen2VLGSPOTrainer(
        model=model,
        processor=processor,
        train_loader=loader,
        cfg=cfg,
        reward_fns=[],   # uses default reward (<answer>...</answer> format)
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
