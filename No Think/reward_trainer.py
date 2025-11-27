import re

def accuracy_reward(completions, solution, **kwargs):
    """
    Accuracy-based reward function for GSPO.
    Compares model completions against ground truth answers (A-D)
    extracted from <answer>...</answer> tags.

    Args:
        completions (list[str]): model outputs (decoded text)
        solution (list[str]): list of correct answers (e.g. ["A", "C", ...])
        **kwargs: unused extra arguments passed by trainer

    Returns:
        list[float]: rewards (1.0 for correct, 0.0 for incorrect)
    """

    def extract_answer(text):
        """Extracts a single-letter answer (A-D) from text."""
        if not text:
            return ""
        text_str = str(text)

        # Pattern 1: <answer>X</answer>
        match = re.search(r"<answer>\s*([A-D])\s*</answer>", text_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 2: "Answer: X"
        match = re.search(r"Answer:\s*([A-D])", text_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 3: standalone capital letter
        match = re.search(r"\b([A-D])\b", text_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Fallback: any single capital letter A–D anywhere
        match = re.search(r"([A-D])", text_str)
        if match:
            return match.group(1).upper()

        # Default fallback: return cleaned text
        return text_str.strip().upper()

    rewards = []
    for content, sol in zip(completions, solution):
        student_answer = extract_answer(content)
        ground_truth = extract_answer(sol)

        reward = 1.0 if student_answer == ground_truth else 0.0
        rewards.append(reward)

        # Optional debug logging
        if kwargs.get("debug", False) or len(rewards) <= 2:
            print("[REWARD DEBUG]")
            print(f"  Model Output: {content}")
            print(f"  Ground Truth: {sol}")
            print(f"  Parsed Output: {student_answer}")
            print(f"  Parsed GT: {ground_truth}")
            print(f"  Reward: {reward}")

    return rewards
