# Author: Brandon Henley
# quantize.py
# ----------------------------
# This script applies SmoothQuant post-training quantization to a Hugging Face model.
# It uses precomputed activation scales to transform the model for INT8-compatible deployment.
# The quantized model is saved in Hugging Face format for future use.
#
# Usage Example:
#   python quantize.py --model bert-base-uncased --act_scales_path act_scales/bert-base-uncased.json --alpha 0.5 --output_dir results/quantized

import argparse
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from smoothquant.smooth import smooth_lm
import torch
import json

def load_model_and_tokenizer(model_name):
    """
    Load a Hugging Face model and tokenizer from the given model name or path.
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def load_act_scales(scales_path):
    """
    Load activation scale values from a precomputed JSON file.

    Args:
        scales_path: Path to the JSON file containing activation scales

    Returns:
        Dictionary of {layer_name: scale}
    """
    print(f"Loading activation scales from: {scales_path}")
    with open(scales_path, "r") as f:
        act_scales = json.load(f)
    return act_scales

def apply_smoothquant(model, act_scales, alpha):
    """
    Apply the SmoothQuant transformation to the model using activation scales.

    Args:
        model: Hugging Face model
        act_scales: Dictionary of activation scales
        alpha: Smoothing factor (0=all activations, 1=all weights)

    Returns:
        Modified (quantized) model
    """
    print(f"Applying SmoothQuant with alpha={alpha}...")
    model.eval()
    with torch.no_grad():
        smooth_lm(model, act_scales, alpha=alpha)
    return model

def save_model(model, tokenizer, save_path):
    """
    Save the quantized model and tokenizer in Hugging Face format.

    Args:
        model: Quantized model
        tokenizer: Tokenizer used for inference
        save_path: Output directory path
    """
    print(f"Saving quantized model to {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def main():
    parser = argparse.ArgumentParser(description="Apply SmoothQuant to a Hugging Face model using precomputed activation scales.")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model name or path to load from Hugging Face")
    parser.add_argument("--act_scales_path", type=str, required=True,
                        help="Path to activation scales JSON file")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant scaling factor between activations and weights")
    parser.add_argument("--output_dir", type=str, default="quantized_models",
                        help="Directory to save the quantized model")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    act_scales = load_act_scales(args.act_scales_path)
    quant_model = apply_smoothquant(model, act_scales, args.alpha)

    save_dir = Path(args.output_dir) / args.model.replace("/", "-")
    save_model(quant_model, tokenizer, save_dir)

if __name__ == "__main__":
    main()