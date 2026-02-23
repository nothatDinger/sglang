# SWE-bench with SGLang

[SWE-bench](https://www.swebench.com) is a benchmark for evaluating LLMs on real-world GitHub issue resolution tasks. This directory provides tooling to use SGLang as the inference backend for generating patches on SWE-bench instances.

## Overview

The evaluation pipeline has two stages:
1. **Inference** (this directory): Use SGLang to generate patches for each issue.
2. **Evaluation** (SWE-bench harness): Apply patches and run tests to measure the resolved rate. This step requires Docker.

## Prerequisites

```bash
pip install sglang[all] datasets openai
```

To run the evaluation stage, install the SWE-bench harness:
```bash
pip install swebench
```

## Step 1: Start the SGLang Server

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-72B-Instruct --port 30000
```

Any chat model supported by SGLang can be used. Models with strong coding ability (e.g., Qwen2.5-Coder, DeepSeek-Coder-V2) tend to perform better.

## Step 2: Generate Patches

Run the inference script to generate patches for all SWE-bench Lite instances:

```bash
python bench_sglang.py --port 30000 --output-file predictions.json
```

This generates a `predictions.json` file in the SWE-bench submission format.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 30000 | SGLang server port |
| `--host` | 0.0.0.0 | SGLang server host |
| `--base-url` | None | Full server base URL (overrides host/port) |
| `--model` | auto | Model name (auto-detected from server if not set) |
| `--dataset` | `princeton-nlp/SWE-bench_Lite` | HuggingFace dataset name |
| `--split` | `test` | Dataset split |
| `--num-instances` | all | Number of instances to evaluate |
| `--parallel` | 16 | Number of parallel requests |
| `--max-tokens` | 4096 | Max tokens for model output |
| `--temperature` | 0.0 | Sampling temperature |
| `--output-file` | `predictions.json` | Output predictions file |

### Example: Quick Test on 10 Instances

```bash
python bench_sglang.py --port 30000 --num-instances 10 --output-file predictions_10.json
```

### Example: Full SWE-bench (not Lite)

```bash
python bench_sglang.py --port 30000 \
    --dataset princeton-nlp/SWE-bench \
    --output-file predictions_full.json \
    --parallel 32
```

## Step 3: Evaluate Predictions

After generating predictions, use the SWE-bench harness to evaluate them. This step requires Docker.

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions.json \
    --max_workers 4 \
    --run_id my_run
```

The harness will:
1. Set up a Docker container for each instance
2. Apply the generated patch
3. Run the test suite
4. Report the **resolved rate** (percentage of issues fully fixed)

## Predictions Format

The `predictions.json` file follows the SWE-bench submission format:

```json
[
  {
    "instance_id": "astropy__astropy-12907",
    "model_patch": "diff --git a/astropy/...",
    "model_name_or_path": "Qwen/Qwen2.5-72B-Instruct"
  },
  ...
]
```

## Tips for Better Results

- **Larger models** generally produce better patches. Models with ≥ 32B parameters perform significantly better.
- **Coding-specific models** (e.g., `Qwen2.5-Coder-32B-Instruct`, `deepseek-ai/DeepSeek-Coder-V2-Instruct`) tend to outperform general models of similar size.
- **Higher `--max-tokens`** (e.g., 8192) can help for complex issues that require larger patches.
- Use **`--temperature 0.0`** for deterministic results, or a small positive value (e.g., 0.2) when running multiple trials.

## References

- [SWE-bench paper](https://arxiv.org/abs/2310.06770)
- [SWE-bench Lite](https://www.swebench.com/#lite) – a more tractable subset of 300 instances
- [SWE-bench GitHub](https://github.com/princeton-nlp/SWE-bench)
- [SGLang documentation](https://sglang.readthedocs.io)
