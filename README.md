# SmoothQuant: Efficient Post-Training Quantization for BERT
Post-training quantization of LLMs using SmoothQuant for faster, cheaper inference

## Problem

How can we reduce the inference cost of large language models without sacrificing accuracy?

Modern LLMs are powerful—but resource-intensive. We explore **SmoothQuant**, a post-training quantization method, to compress and accelerate BERT-style models using only calibration data.

---

## Data & Model

- **Dataset:** [`amazon_polarity`](https://huggingface.co/datasets/amazon_polarity) (~3M reviews)
- **Model:** `bert-base-uncased` via Hugging Face
- **Task:** Sentiment classification (binary)

---

## Technology Used

- **SmoothQuant** (MIT-Han Lab): INT8 quantization for weights + activations
- **Hugging Face Transformers/Datasets**
- **PyTorch 2.6**, **Python 3.10**
- **Calibrated on 128 samples**, quantized in-place using max activation capture

---

## Pipeline

1. `generate_scales.py` — Extracts per-layer activation scales from a small calibration batch
2. `quantize.py` — Applies `smooth_lm(...)` using captured scales + user-controlled `alpha`
3. `evaluate.py` — Reports post-quantization classification accuracy
4. `benchmark.py` — Times inference over multiple runs

---

## Results

### Accuracy (128-calibration, 1000 test samples)
- `bert-base-uncased`: **0.500**
- Quantized version: **0.500** (structure-only demo, untrained head)

### Inference Latency (Batch Size = 8)

| Model             | Avg Latency | Speedup |
|------------------|-------------|---------|
| `bert-base-uncased` | 212.21 ms   | —       |
| **Quantized**     | **182.56 ms** | ~14% faster |

---

## Usage

Generate activation scales:
```bash
python src/generate_scales.py \
  --model bert-base-uncased \
  --output act_scales/bert-base-uncased.json \
  --num_samples 128 \
  --max_length 128

Quantize:
```bash
python src/quantize.py \
  --model bert-base-uncased \
  --act_scales_path act_scales/bert-base-uncased.json \
  --alpha 0.5 \
  --output_dir results/quantized  

Evaluate:
```bash
python src/evaluate.py \
  --model_path results/quantized/bert-base-uncased

Benchmark:
``bash
python src/benchmark.py \
  --model_path results/quantized/bert-base-uncased


Repo Structure:
├── src/
│   ├── generate_scales.py
│   ├── quantize.py
│   ├── evaluate.py
│   └── benchmark.py
├── act_scales/
├── results/quantized/
└── environment.yml

Citation:
Xiao et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs

Built by Brandon Henley | CSCI E-104 Final Project
