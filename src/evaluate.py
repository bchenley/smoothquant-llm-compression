import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

def evaluate(model_path, batch_size=16, max_length=128, num_samples=1000):
    print(f"Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    dataset = load_dataset("amazon_polarity", split=f"test[:{num_samples}]")
    texts = dataset["content"] if "content" in dataset.features else dataset["text"]
    labels = dataset["label"]

    correct = 0
    total = 0

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        correct += (predictions == np.array(batch_labels)).sum()
        total += len(batch_labels)

    acc = correct / total
    print(f"Accuracy on {total} samples: {acc:.4f}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized or baseline model accuracy")
    parser.add_argument("--model_path", type=str, required=True, help="Path to quantized model directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of test samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    evaluate(args.model_path, batch_size=args.batch_size, num_samples=args.num_samples)


if __name__ == "__main__":
    main()