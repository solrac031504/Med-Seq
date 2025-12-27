import os
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from trl import GRPOConfig, GRPOTrainer
from torch.utils.data import Dataset
from reward_functions import think_answer_reward


# ------------------------------
# Prompt (think + answer)
# ------------------------------
ANSWER_ONLY_TEMPLATE = (
    "{Question}\n"
    "First, think step by step inside <think>...</think>.\n"
    "Then provide ONLY the single-letter choice (A, B, C, D, ...) inside <answer>...</answer>.\n"
    "Rules:\n"
    " - Put your reasoning only inside <think> ... </think>.\n"
    " - End with exactly one <answer>A|B|C|D</answer>.\n"
    " - No extra text outside these tags."
)

# ------------------------------
# Dataset
# ------------------------------
class GSPODataset(Dataset):
    def __init__(self, json_path, image_root, processor, image_size=(384, 384)):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.processor = processor
        self.image_size = image_size

    def _make_conversation(self, example):
        """Builds multimodal conversation input."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {
                        "type": "text",
                        "text": ANSWER_ONLY_TEMPLATE.format(Question=example["problem"]),
                    },
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        return text_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image_path = sample["image"]
        if not image_path.startswith(self.image_root):
            image_path = os.path.join(self.image_root, sample["image"])
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        text_prompt = self._make_conversation(sample)

        inputs = self.processor(
            text=text_prompt, images=image, return_tensors="pt"
        )

        # Attach metadata for reward and debugging
        inputs["solution"] = sample["solution"]
        inputs["problem"] = sample["problem"]
        inputs["prompt"] = text_prompt

        # Remove batch dimension
        return {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }


# ------------------------------
# GSPO Trainer
# ------------------------------
class GSPOTrainer:
    def __init__(self, config: dict, debug: bool = False):
        self.debug = debug
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEVICE] Using device: {self.device}")

        # Directories
        self.checkpoint_dir = os.path.abspath(config.get("checkpoint_dir", "checkpoints"))
        self.output_dir = os.path.abspath(config.get("output_dir", "outputs"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # ------------------------------
        # Model & Processor (Qwen 2.5 VL)
        # ------------------------------
        model_name = config.get("model_name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct")
        print(f"[MODEL] Loading model: {model_name}")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # ? use dtype
            device_map="auto",
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)  # ? stable output

        # ------------------------------
        # Data
        # ------------------------------
        self.train_data_dir = config.get("train_data_dir")
        self.eval_data_dir = config.get("eval_data_dir")
        self.image_root = config.get("image_root")

        if not self.train_data_dir or not os.path.exists(self.train_data_dir):
            raise FileNotFoundError(f"Training data not found: {self.train_data_dir}")
        if not self.eval_data_dir or not os.path.exists(self.eval_data_dir):
            raise FileNotFoundError(f"Eval data not found: {self.eval_data_dir}")

        print(f"[DATA] Training: {self.train_data_dir}")
        print(f"[DATA] Evaluation: {self.eval_data_dir}")

        self.train_dataset = GSPODataset(
            json_path=self.train_data_dir,
            image_root=self.image_root,
            processor=self.processor,
        )
        self.eval_dataset = GSPODataset(
            json_path=self.eval_data_dir,
            image_root=self.image_root,
            processor=self.processor,
        )

        # ------------------------------
        # GSPO Configuration
        # ------------------------------
        self.grpo_config = GRPOConfig(
            learning_rate=float(config.get("learning_rate", 2e-5)),
            generation_batch_size=config.get("batch_size", 1),
            num_generations=config.get("num_generations", 2),
            epsilon=config.get("clip_range", 0.2),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            num_train_epochs=config.get("num_epochs", 2),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
            importance_sampling_level="sequence",  # GSPO sequence-level optimization
            logging_steps=config.get("logging_steps", 10),
            output_dir=self.output_dir,
            report_to="none",
        )
        
        # Manually attach kl_penalty for self-play stability reference
        self.grpo_config.kl_penalty = config.get("kl_penalty", 0.1)
        print(f"[TRAINING CONFIG] Using KL penalty: {self.grpo_config.kl_penalty}")

        # ------------------------------
        # Initialize GRPO Trainer (GSPO-style)
        # ------------------------------
        print("[TRAINER] Initializing GSPO (GRPO-based) trainer...")
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            reward_funcs=lambda *a, **kw: think_answer_reward(*a, **kw, debug=self.debug),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.grpo_config,
        )
    # ------------------------------
    # Training
    # ------------------------------
    def train(self):
        print("[TRAINING] Starting GSPO training...")
        result = self.trainer.train()
    
        # Save final model
        self.trainer.save_model(self.output_dir)
        print(f"[TRAINING] Completed. Model saved to {self.output_dir}")
    
        # Optionally log completion only
        print("[INFO] Skipping automatic evaluation. Model ready for external testing.")
        return result