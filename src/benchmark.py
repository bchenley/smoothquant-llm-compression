# Author: Brandon Henley
# benchmark.py
# ----------------------------
# This script benchmarks the inference latency of a Hugging Face-compatible model.
# It loads a small batch of text samples from the Amazon Polarity dataset, performs
# multiple forward passes, and reports average and per-run latency.
#
# Usage Example:
#   python benchmark.py --model_path results/quantized/bert-base-uncased --num_samples 8 --runs 10

import argparse
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from statistics import mean

def measure_latency(model, tokenizer, texts, max_length=128, runs=10):
    """
    Measures average inference latency over multiple runs.

    Args:
        model: The preloaded Hugging Face model.
        tokenizer: Corresponding tokenizer.
        texts: List of input strings.
        max_length: Max token length (for truncation).
        runs: Number of repetitions for timing.

    Returns:
        Tuple: (average latency in ms, list of individual latencies)
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    model.eval()
    latencies = []

    with torch.no_grad():
        # Warmup to stabilize GPU/CPU performance
        for _ in range(2):
            _ = model(**inputs)

        # Timed runs
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(**inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # milliseconds

    return mean(latencies), latencies

def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference speed")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory (e.g. from quantize.py output)")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of test samples to use")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed inference runs")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load small slice of the Amazon Polarity test set
    dataset = load_dataset("amazon_polarity", split=f"test[:{args.num_samples}]")
    texts = dataset["content"] if "content" in dataset.features else dataset["text"]

    # Run benchmark
    avg_latency, latencies = measure_latency(model, tokenizer, texts, runs=args.runs)

    # Print timing results
    print(f"\nInference latency over {args.runs} runs (batch size={args.num_samples}):")
    print(f"  Average: {avg_latency:.2f} ms")
    print(f"  Individual runs: {[round(l, 2) for l in latencies]}")

if __name__ == "__main__":
    main()