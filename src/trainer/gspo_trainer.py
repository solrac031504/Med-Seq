# gspo_trainer.py
# Minimal GSPO trainer for Qwen2-VL with "no-think" output
# Single-GPU, PyTorch + Transformers only (no Deepspeed / Accelerate dependencies)

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class GSPOConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0           # 0.0 = greedy
    top_p: float = 1.0
    do_sample: bool = False
    num_generations: int = 2           # K: generations per prompt (group size)
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    train_batch_size: int = 1          # number of prompts per step
    grad_accum_steps: int = 1
    max_steps: int = 1000
    log_every: int = 10
    save_every: int = 200
    output_dir: str = "./output"
    bfloat16: bool = True
    attn_impl: str = "sdpa"            # "sdpa" or "flash_attention_2"
    device: str = "cuda"
    max_pixels: int = 401_408          # 384x384 = 147,456; leave headroom
    min_pixels: int = 3_136


# ---------------------------------------------------------------------
# Utility: simple collator for Qwen2-VL chat format
# Each item in batch: {"image": <path_or_url>, "problem": <text>}
# ---------------------------------------------------------------------
class QwenVLChatCollator:
    def __init__(self, processor: AutoProcessor, max_pixels: int = 401_408):
        self.processor = processor
        self.max_pixels = max_pixels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Converts a list of samples into model inputs.
        Produces only the prompt (no labels), since we do RL on generated continuation.
        """
        # Build messages for each sample
        msgs = []
        image_inputs = []
        for ex in batch:
            image_ref = ex["image"]
            # Processor expects "file://" for local paths, but plain path also works in recent versions
            if isinstance(image_ref, str) and not image_ref.startswith("http") and not image_ref.startswith("file://"):
                image_ref = f"file://{image_ref}"

            msgs.append(
                [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_ref},
                        {"type": "text",
                         "text": f"{ex['problem']} "
                                 f"Output ONLY the choice letter in <answer>...</answer> "
                                 f"with no extra text."}
                    ],
                }]
            )
            image_inputs.append(image_ref)

        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in msgs
        ]

        # Processor returns tensors with input_ids/attention_mask and vision inputs
        inputs = self.processor(
            text=texts,
            images=[None] * len(batch),   # images already in messages
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        return inputs


# ---------------------------------------------------------------------
# Reward protocol
# A reward function receives:
#   prompts_text: List[str]
#   completions_text: List[str]
#   batch: original batch list (dicts) so it can access "solution" etc.
# Returns: List[float] rewards (one per completion)
# ---------------------------------------------------------------------
RewardFn = Callable[[Sequence[str], Sequence[str], Sequence[Dict[str, Any]]], List[float]]


def default_format_reward(prompts: Sequence[str],
                          completions: Sequence[str],
                          batch: Sequence[Dict[str, Any]]) -> List[float]:
    """Reward = 1 if completion has an <answer>...</answer> tag; else 0."""
    import re
    r = []
    pat = re.compile(r"\s*<answer>\s*([A-Za-z0-9]+)\s*</answer>\s*$")
    for c in completions:
        r.append(1.0 if pat.fullmatch(c.strip()) else 0.0)
    return r


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class Qwen2VLGSPOTrainer:
    """
    A minimal GSPO trainer:
      - For each prompt in the batch, generate K completions (group size).
      - Compute rewards r_i for each completion in the group.
      - Compute group baseline b = mean(r_i).
      - Compute policy loss = - sum_i (r_i - b) * logpi_i
        where logpi_i = log P_\theta(completion_i | prompt).
      - Backprop & optimize.
    """

    def __init__(
        self,
        model: Qwen2VLForConditionalGeneration,
        processor: AutoProcessor,
        train_loader: DataLoader,
        cfg: GSPOConfig,
        reward_fns: Optional[List[RewardFn]] = None,
    ):
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.cfg = cfg
        self.reward_fns = reward_fns if reward_fns and len(reward_fns) > 0 else [default_format_reward]

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.model.to(self.cfg.device)

        if self.cfg.bfloat16:
            self.model = self.model.to(dtype=torch.bfloat16)

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

        self.global_step = 0

    @torch.no_grad()
    def _generate(self, model_inputs: Dict[str, torch.Tensor], k: int) -> Dict[str, torch.Tensor]:
        """
        Generate K completions per prompt by repeating inputs along batch dim.
        Returns dict with:
          - input_ids (prompt tokens, left-padded),
          - generated_ids (prompt + completion),
          - generated_text (decoded),
        """
        device = self.cfg.device
        inp = {k0: v.to(device) for k0, v in model_inputs.items()}
        bsz = inp["input_ids"].size(0)

        # Repeat each prompt K times for grouped generations
        rep_inp = {k0: v.repeat_interleave(k, dim=0) for k0, v in inp.items()}

        gen_out = self.model.generate(
            **rep_inp,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature if self.cfg.do_sample else None,
            top_p=self.cfg.top_p if self.cfg.do_sample else None,
            use_cache=True,
        )

        # Split into prompt part and newly generated part
        input_lens = rep_inp["input_ids"].shape[1]
        completions = []
        for row in gen_out:
            comp_ids = row[input_lens:]
            completions.append(comp_ids)

        # Decode, grouped back per original item
        decoded = self.processor.batch_decode(gen_out, skip_special_tokens=True)
        decoded_prompts = self.processor.batch_decode(rep_inp["input_ids"], skip_special_tokens=True)

        # Extract only the completion text by trimming the prompt prefix
        completions_text = []
        for dp, full in zip(decoded_prompts, decoded):
            comp_txt = full[len(dp):].strip()
            completions_text.append(comp_txt)

        return {
            "input_ids": rep_inp["input_ids"],
            "generated_ids": gen_out,
            "completions_text": completions_text,
            "k": k,
            "bsz": bsz,
        }

    def _logprob_of_generated(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes log p_\theta(completion | prompt) token-wise and sums over completion tokens.
        Assumes generated_ids = [prompt || completion], input_ids = [prompt].
        """
        device = self.cfg.device
        with torch.no_grad():
            # get model logits for full sequence
            out = self.model(generated_ids.to(device), use_cache=False)
            logits = out.logits  # [B, T, V]
            log_probs = torch.log_softmax(logits, dim=-1)

        prompt_len = input_ids.shape[1]
        # completion tokens positions and their previous tokens as inputs
        comp_token_ids = generated_ids[:, prompt_len:]  # [B, T_comp]
        prev_logits = log_probs[:, prompt_len - 1:-1, :]  # shift so that prob of token t uses logits at t-1
        # Handle edge case when no generated tokens (shouldn't happen with max_new_tokens>0)
        if comp_token_ids.numel() == 0:
            return torch.zeros(generated_ids.size(0), device=device)

        # Gather log probs for the actual generated tokens
        lp = prev_logits.gather(-1, comp_token_ids.unsqueeze(-1)).squeeze(-1)  # [B, T_comp]
        # Sum log-probs over completion tokens
        return lp.sum(dim=1)  # [B]

    def train(self):
        self.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # bfloat16 used; AMP scaler unnecessary

        for epoch in range(10**9):  # effectively until max_steps
            for batch in self.train_loader:
                if self.global_step >= self.cfg.max_steps:
                    return

                # 1) Generate K completions per prompt
                gen = self._generate(batch, k=self.cfg.num_generations)

                # 2) Compute rewards per completion
                # Rebuild grouped size
                k = gen["k"]
                bsz = gen["bsz"]

                # Decode prompts for reward functions (text form)
                prompts_text = self.processor.batch_decode(gen["input_ids"], skip_special_tokens=True)

                # We need the original batch items replicated K times to align with completions
                # Build a flat list of originals aligned with completions_text
                originals: List[Dict[str, Any]] = []
                for ex in range(bsz):
                    # The dataloader gave us tensors; but batch came from collator, so no raw fields.
                    # For rewards that need ground truth, pass empty dict (or modify your dataloader to keep metadata).
                    originals.extend([{} for _ in range(k)])

                # Run all reward fns and sum
                rewards_sum = torch.zeros(bsz * k, device=self.cfg.device)
                offset = 0
                # Build prompts_text repeated to match completions
                rep_prompts_text = []
                for ex in range(bsz):
                    rep_prompts_text.extend([prompts_text[ex]] * k)

                for rfn in self.reward_fns:
                    r_vals = rfn(rep_prompts_text, gen["completions_text"], originals)
                    rewards_sum += torch.tensor(r_vals, device=self.cfg.device, dtype=torch.float32)

                # 3) Compute group baselines and advantages
                rewards_sum = rewards_sum.view(bsz, k)
                baseline = rewards_sum.mean(dim=1, keepdim=True)     # [bsz, 1]
                advantages = (rewards_sum - baseline).reshape(-1)    # [bsz*k]

                # 4) Compute log p_\theta(completion | prompt)
                # (needs gradients; recompute forward without no_grad)
                self.model.zero_grad(set_to_none=True)
                outputs = self.model(gen["generated_ids"].to(self.cfg.device), use_cache=False)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1)
                prompt_len = gen["input_ids"].shape[1]
                comp_token_ids = gen["generated_ids"][:, prompt_len:]
                # Shifted logits to align with next token
                prev_logits = log_probs[:, prompt_len - 1:-1, :]
                # Sometimes completion length is 0 if EOS comes immediately
                if comp_token_ids.numel() == 0:
                    # skip step (rare)
                    continue
                token_logp = prev_logits.gather(-1, comp_token_ids.unsqueeze(-1)).squeeze(-1)  # [B, Tcomp]
                seq_logp = token_logp.sum(dim=1)  # [B=bsz*k]

                # 5) GSPO objective: maximize sum_i (A_i * logpi_i)
                # => loss = - mean over group items
                loss = -(advantages.detach() * seq_logp).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

                self.global_step += 1

                if self.global_step % self.cfg.log_every == 0:
                    with torch.no_grad():
                        r_mean = rewards_sum.mean().item()
                        r_std = rewards_sum.std().item()
                    print(f"[step {self.global_step}] loss={loss.item():.4f} "
                          f"reward_mean={r_mean:.3f} reward_std={r_std:.3f}")

                if self.global_step % self.cfg.save_every == 0:
                    save_dir = os.path.join(self.cfg.output_dir, f"step_{self.global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    self.model.save_pretrained(save_dir)
                    self.processor.save_pretrained(save_dir)
                    print(f"Saved checkpoint to {save_dir}")
