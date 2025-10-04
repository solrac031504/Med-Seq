import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from PIL import Image
import difflib

# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--prompt_path", type=str, required=True)
args = parser.parse_args()

MODEL_PATH = args.model_path
BSZ = args.batch_size
OUTPUT_PATH = args.output_path
PROMPT_PATH = args.prompt_path

PROJECT_ROOT = "/content/drive/MyDrive/CAP_6614_current_topics_in_ML/Final_Project/Med-R1"

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load dataset
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Stricter prompt ---
QUESTION_TEMPLATE = (
    "{Question}\n"
    "Respond with ONLY the single correct option letter (A, B, C, D, ...) "
    "inside <answer>...</answer>. Do not include explanation or option text."
)

# Build messages with absolute image paths
messages = []
for item in data:
    rel_path = item['image']
    image_path = rel_path if os.path.isabs(rel_path) else os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=item['problem'])}
        ]
    }]
    messages.append(message)

def safe_process_vision_info(msgs):
    image_inputs, video_inputs = process_vision_info(msgs)
    fixed_images = []
    for img in image_inputs:
        if isinstance(img, str):
            fixed_images.append(Image.open(img).convert("RGB"))
        else:
            fixed_images.append(img)
    return fixed_images, video_inputs

# --- Improved extractor ---
def extract_option_answer(output_str, question_text=None):
    output_str = (output_str or "").strip()

    # Clean malformed tags like "</answer> </answer>"
    output_str = re.sub(r'</answer>\s*</answer>', '</answer>', output_str)

    # 1. Try <answer>...</answer>
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_str, re.IGNORECASE)
    candidate = match.group(1).strip() if match else output_str.strip()

    # 2. Direct letter
    if candidate.upper() in ["A", "B", "C", "D", "E"]:
        return candidate.upper()

    # 3. Yes/No → map if options exist
    if candidate.lower() in ["yes", "no"] and question_text:
        options = re.findall(r'([A-E])\)\s*([^,;\n]+)', question_text)
        for letter, text in options:
            if candidate.lower() == text.strip().lower():
                return letter.upper()
        # fallback: assume Yes=A, No=B
        return "A" if candidate.lower() == "yes" else "B"

    # 4. Fuzzy match descriptive answers to options
    if question_text:
        options = re.findall(r'([A-E])\)\s*([^,;\n]+)', question_text)
        for letter, text in options:
            if text.strip().lower() in candidate.lower():
                return letter.upper()
        # fuzzy ratio match
        for letter, text in options:
            ratio = difflib.SequenceMatcher(None, candidate.lower(), text.strip().lower()).ratio()
            if ratio > 0.6:  # adjustable threshold
                return letter.upper()

    # 5. Loose fallback (find A-E inside)
    match = re.search(r'\b([A-E])\)', candidate)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-E])\b', candidate)
    if match:
        return match.group(1).upper()

    return None

# Run inference
all_outputs = []
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    image_inputs, video_inputs = safe_process_vision_info(batch_messages)

    if video_inputs:
        inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    else:
        inputs = processor(text=text, images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    all_outputs.extend(batch_output_text)
    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

# Evaluation
final_output = []
correct_number = 0

for input_example, model_output in zip(data, all_outputs):
    raw_output = model_output
    ground_truth = input_example['solution']
    model_answer = extract_option_answer(raw_output, input_example['problem'])

    gt_match = re.search(r'<answer>\s*(\w+)\s*</answer>', ground_truth)
    ground_truth_letter = gt_match.group(1) if gt_match else ground_truth

    print(f"raw_output: {raw_output} | extracted: {model_answer} | ground_truth: {ground_truth_letter}")

    result = {
        'question': input_example,
        'ground_truth': ground_truth_letter,
        'raw_output': raw_output,
        'extracted_answer': model_answer
    }
    final_output.append(result)

    if model_answer is not None and model_answer == ground_truth_letter:
        correct_number += 1

accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

with open(OUTPUT_PATH, "w") as f:
    json.dump({'accuracy': accuracy, 'results': final_output}, f, indent=2)

print(f"Results saved to {OUTPUT_PATH}")
