import argparse
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
from statistics import mean


def measure_latency(model, tokenizer, texts, max_length=128, runs=10):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    model.eval()
    latencies = []
    with torch.no_grad():
        # warmup
        for _ in range(2):
            _ = model(**inputs)
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(**inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # milliseconds
    return mean(latencies), latencies


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference speed")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of examples for benchmarking")
    parser.add_argument("--runs", type=int, default=10, help="Number of times to run inference")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    dataset = load_dataset("amazon_polarity", split="test[:{}]".format(args.num_samples))
    texts = dataset["content"] if "content" in dataset.features else dataset["text"]

    avg_latency, latencies = measure_latency(model, tokenizer, texts, runs=args.runs)

    print(f"\nInference latency over {args.runs} runs (batch size={args.num_samples}):")
    print(f"  Average: {avg_latency:.2f} ms")
    print(f"  Individual runs: {[round(l, 2) for l in latencies]}")


if __name__ == "__main__":
    main()
