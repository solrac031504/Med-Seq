"""
This implementation is from https://github.com/vivekvar-dl/GSPO-DeepSeek-R1-Distill-Qwen-1.5B/blob/main/gspo/trainer.py
"""

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import logging

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)

@dataclass
class GSPOConfig:
    """Configuration for GSPO training"""
    # Core GSPO parameters
    left_clip_range: float = 3e-4
    right_clip_range: float = 4e-4
    group_size: int = 2  # Reduced default for memory
    
    # Training parameters - memory optimized defaults
    learning_rate: float = 1e-6
    batch_size: int = 1  # Very small for memory
    mini_batch_size: int = 1
    max_length: int = 256  # Reduced sequence length
    gradient_accumulation_steps: int = 4  # Accumulate gradients
    
    # Memory optimization
    use_8bit_optimizer: bool = True
    use_gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3  # Aggressive clipping
    
    # Reward normalization
    reward_normalization: bool = True
    advantage_normalization: bool = True
    
    # Logging
    log_frequency: int = 10
    eval_frequency: int = 100

class Qwen2VLGSPOTrainer:
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: AutoTokenizer,
        config: GSPOConfig,
        device: str = "cuda"
    ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Initialize step counter and logging FIRST
        self.step = 0
        logging.basicConfig(level=logging.INFO)  # Changed back to INFO for cleaner output
        self.logger = logging.getLogger(__name__)
        self.logger.info("GSPO Trainer initialized for verification")
        
        # Ensure model is on correct device and in training mode
        self.model.to(self.device)
        self.model.train()
        
        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        # Store old model for importance ratio computation
        self.old_model = None
        self.update_old_model()
        
        # Use memory-efficient optimizer
        if config.use_8bit_optimizer and BNB_AVAILABLE:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(), 
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0
            )
            print("✓ Using 8-bit AdamW optimizer")
        else:
            # Fallback to regular AdamW with memory optimizations
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=config.learning_rate,
                betas=(0.9, 0.95),  # More stable betas
                eps=1e-8,
                weight_decay=0.0,
                foreach=False  # Disable foreach for memory
            )
            print("✓ Using regular AdamW optimizer")
        
        # Gradient accumulation
        self.accumulation_steps = 0