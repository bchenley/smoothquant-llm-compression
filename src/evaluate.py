# Author: Brandon Henley
# evaluate.py
# ----------------------------
# This script evaluates the classification accuracy of a Hugging Face-compatible model
# (quantized or baseline) using a test split of the Amazon Polarity dataset.
#
# Usage Example:
#   python evaluate.py --model_path results/quantized/bert-base-uncased --num_samples 1000

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np

def evaluate(model_path, batch_size=16, max_length=128, num_samples=1000):
    """
    Evaluate the accuracy of a model on a subset of the Amazon Polarity test set.

    Args:
        model_path: Path to directory containing model weights and tokenizer
        batch_size: Number of samples per forward pass
        max_length: Max token length (truncation)
        num_samples: Total number of test samples to evaluate

    Returns:
        Accuracy (float) over the sampled dataset
    """
    print(f"Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Load subset of test data
    dataset = load_dataset("amazon_polarity", split=f"test[:{num_samples}]")
    texts = dataset["content"] if "content" in dataset.features else dataset["text"]
    labels = dataset["label"]

    correct = 0
    total = 0

    # Iterate over batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # Tokenize and run inference
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        # Update accuracy counters
        correct += (predictions == np.array(batch_labels)).sum()
        total += len(batch_labels)

    acc = correct / total
    print(f"Accuracy on {total} samples: {acc:.4f}")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized or baseline model accuracy")
    parser.add_argument("--model_path", type=str, required=True, help="Path to quantized model directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of test samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()

    evaluate(args.model_path, batch_size=args.batch_size, num_samples=args.num_samples)

if __name__ == "__main__":
    main()