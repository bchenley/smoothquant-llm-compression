# Author: Brandon Henley
# generate_scales.py
# ----------------------------
# This script performs post-training calibration to extract activation scales
# from a Hugging Face model using a small number of text samples. The output
# is a dictionary of max activation values per layer, saved as JSON.
#
# These activation scales are used by SmoothQuant to enable accurate INT8 quantization.
#
# Usage Example:
#   python generate_scales.py --model bert-base-uncased --output act_scales/bert-base-uncased.json

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import torch
import json

def load_model_and_tokenizer(model_name):
    """
    Load a Hugging Face model and tokenizer from the given model name or path.
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_calibration_inputs(tokenizer, dataset_name="amazon_polarity", num_samples=128, max_length=128):
    """
    Load a small number of samples from the training set and tokenize them.

    Args:
        tokenizer: Pre-loaded Hugging Face tokenizer
        dataset_name: Dataset identifier from Hugging Face
        num_samples: Number of calibration examples to load
        max_length: Maximum token length to truncate/pad to

    Returns:
        Tokenized inputs for the model
    """
    print("Loading calibration dataset...")
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    texts = dataset["content"] if "content" in dataset.features else dataset["text"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return inputs

def extract_activation_scales(model, inputs):
    """
    Register forward hooks on Linear layers to record max activation per layer.

    Args:
        model: Pre-trained Hugging Face model
        inputs: Tokenized input batch

    Returns:
        Dictionary of activation scales {layer_name: max_activation_value}
    """
    print("Extracting activation scales from model...")
    act_scales = {}
    hooks = []

    def register_hooks():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                def get_hook(name):
                    def hook(module, inp, out):
                        scale = out.abs().max().item()
                        act_scales[name] = scale
                    return hook
                hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    register_hooks()
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    print(f"Extracted {len(act_scales)} activation scale entries.")
    return act_scales

def save_scales(scales, output_path):
    """
    Save the activation scales as a JSON file.

    Args:
        scales: Dictionary of layer activation max values
        output_path: Destination path for the JSON file
    """
    print(f"Saving activation scales to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scales, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Extract simplified activation scales for SmoothQuant-compatible models")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name from Hugging Face")
    parser.add_argument("--output", type=str, default="act_scales/bert-base-uncased.json", help="Where to save act scales")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of samples for calibration")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    inputs = get_calibration_inputs(tokenizer, num_samples=args.num_samples, max_length=args.max_length)
    scales = extract_activation_scales(model, inputs)
    save_scales(scales, args.output)

if __name__ == "__main__":
    main()