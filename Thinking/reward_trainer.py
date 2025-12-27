import re


def accuracy_reward(completions, solution, **kwargs):
    """
    Accuracy-based reward function for GSPO.
    Compares model completions against ground truth answers (A-D)
    extracted from <answer>...</answer> tags.
    """
    def extract_answer(text):
        if not text:
            return ""
        text_str = str(text)
        match = re.search(r"<answer>\s*([A-D])\s*</answer>", text_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"Answer:\s*([A-D])", text_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"\b([A-D])\b", text_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"([A-D])", text_str)
        if match:
            return match.group(1).upper()
        return text_str.strip().upper()

    rewards = []
    for content, sol in zip(completions, solution):
        student_answer = extract_answer(content)
        ground_truth = extract_answer(sol)
        reward = 1.0 if student_answer == ground_truth else 0.0
        rewards.append(reward)
        if kwargs.get("debug", False) or len(rewards) <= 2:
            print("[REWARD DEBUG]")
            print(f"  Model Output: {content}")
            print(f"  Ground Truth: {sol}")
            print(f"  Parsed Output: {student_answer}")
            print(f"  Parsed GT: {ground_truth}")
            print(f"  Reward: {reward}")
    return rewards



# -------------------------------------------------------
# Think + Answer composite reward for reasoning control
# -------------------------------------------------------
def think_answer_reward(completions, solution, **kwargs):
    """
    Composite reward for 'think then answer-only' outputs.

    Reward components:
      ? Accuracy: 1.0 if <answer> matches ground truth, else 0.0
      ? Format bonus: +0.1 for well-formed single <answer> block
      ?? Leak penalty: -0.2 if text outside <think>/<answer> tags
      ?? Think bonus: +0.05 if <think> is used
      ?? Length penalty: gradual penalty for overly long <think> blocks
    """
    def extract(tag, text):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def extract_answer(text):
        m = re.search(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
        return m.group(1).upper() if m else ""

    rewards = []
    debug = kwargs.get("debug", False)  # ? controlled from main.py

    for idx, (output, sol) in enumerate(zip(completions, solution)):
        text = str(output) if output else ""
        match_gt = re.search(r"<answer>\s*([A-D])\s*</answer>", str(sol), re.IGNORECASE)
        gt = match_gt.group(1).upper() if match_gt else str(sol).strip().upper()


        think_content = extract("think", text)
        ans = extract_answer(text)
        acc = 1.0 if ans == gt else 0.0

        # ---- Format bonus ----
        has_single_answer = bool(re.fullmatch(r".*<answer>\s*[A-D]\s*</answer>.*", text, re.IGNORECASE | re.DOTALL))
        fmt_bonus = 0.1 if has_single_answer else -0.1

        # ---- Leak penalty ----
        allowed = ""
        if think_content:
            allowed += f"<think>{think_content}</think>"
        if ans:
            allowed += f"<answer>{ans}</answer>"
        clean_all = re.sub(r"\s+", "", text)
        clean_allowed = re.sub(r"\s+", "", allowed)
        leaked = len(clean_all) > len(clean_allowed)
        leak_pen = -0.2 if leaked else 0.0

        # ---- Think incentives ----
        think_bonus = 0.05 if think_content else 0.0
        over = max(0, len(think_content) - 1200)
        think_pen = -min(0.2, over / 8000.0)

        reward = acc + fmt_bonus + leak_pen + think_bonus + think_pen
        rewards.append(reward)

        # ? Print limited live reward logs (flush ensures they appear in SLURM stdout)
        if debug and (idx < 3 or idx % 50 == 0):
            print("\n[REWARD DEBUG]", flush=True)
            print(f"Step {idx} | GT: {gt} | Pred: {ans} | Acc: {acc}", flush=True)
            print(f"FormatBonus: {fmt_bonus}, Leak: {leak_pen}, ThinkBonus: {think_bonus}, ThinkPen: {think_pen}", flush=True)
            print(f"Reward Total: {reward:.3f}", flush=True)

    return rewards

