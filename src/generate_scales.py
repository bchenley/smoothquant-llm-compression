import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import torch
import json
from collections import defaultdict

def load_model_and_tokenizer(model_name):
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_calibration_inputs(tokenizer, dataset_name="amazon_polarity", num_samples=128, max_length=128):
    print("Loading calibration dataset...")
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    texts = dataset["content"] if "content" in dataset.features else dataset["text"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return inputs


def extract_activation_scales(model, inputs):
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