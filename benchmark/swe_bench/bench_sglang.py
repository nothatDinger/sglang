"""
SWE-bench inference using SGLang.

This script uses SGLang as the inference backend to generate patches for
SWE-bench instances. The output is saved in the SWE-bench submission format
for evaluation with the official SWE-bench evaluation harness.

Usage:
    # Start SGLang server first:
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-72B-Instruct --port 30000

    # Run inference:
    python bench_sglang.py --port 30000 --output-file predictions.json

    # Evaluate with SWE-bench harness (requires Docker):
    python -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_Lite \
        --predictions_path predictions.json \
        --max_workers 4 \
        --run_id my_run
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

MAX_RETRIES = 3

SYSTEM_MESSAGE = """You are an expert software engineer. You will be given a GitHub issue and you need to write a patch to fix the issue.

Your patch must be in the unified diff format (the standard output of `git diff`). Make sure:
1. The patch applies cleanly to the codebase
2. The patch fixes the described issue
3. The patch does not break existing functionality

Output ONLY the patch in unified diff format, wrapped in a ```diff code block. Do not include any explanations before or after the patch."""

USER_TEMPLATE = """Repository: {repo}

Issue:
{problem_statement}

Please provide a patch in unified diff format to fix this issue."""

USER_TEMPLATE_WITH_HINTS = """Repository: {repo}

Issue:
{problem_statement}

Hints:
{hints_text}

Please provide a patch in unified diff format to fix this issue."""


def extract_patch(model_output: str) -> str:
    """Extract the unified diff patch from model output."""
    # Try to extract from ```diff code block
    pattern = re.compile(r"```diff\n(.*?)```", re.DOTALL)
    matches = pattern.findall(model_output)
    if matches:
        return matches[0].strip()

    # Try to extract from ``` code block
    pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
    matches = pattern.findall(model_output)
    if matches:
        candidate = matches[0].strip()
        if candidate.startswith("diff ") or candidate.startswith("--- "):
            return candidate

    # Try to find raw diff content
    lines = model_output.split("\n")
    diff_lines = []
    in_diff = False
    for line in lines:
        if line.startswith("diff --git") or line.startswith("--- a/"):
            in_diff = True
        if in_diff:
            diff_lines.append(line)

    if diff_lines:
        return "\n".join(diff_lines).strip()

    return ""


def generate_patch(
    client: OpenAI,
    model: str,
    instance: dict,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate a patch for a single SWE-bench instance."""
    hints = instance.get("hints_text", "").strip()
    if hints:
        user_content = USER_TEMPLATE_WITH_HINTS.format(
            repo=instance["repo"],
            problem_statement=instance["problem_statement"],
            hints_text=hints,
        )
    else:
        user_content = USER_TEMPLATE.format(
            repo=instance["repo"],
            problem_statement=instance["problem_statement"],
        )

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]

    trial = 0
    while trial < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            backoff = 2**trial
            print(f"Request failed, retrying in {backoff}s: {e}")
            time.sleep(backoff)
            trial += 1

    return ""


def main(args):
    # Set up OpenAI client pointing to SGLang server
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    base_url = (
        f"{args.base_url}/v1"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1"
    )
    client = OpenAI(base_url=base_url)

    # Resolve model name
    model = args.model
    if model is None:
        model = client.models.list().data[0].id
    print(f"Using model: {model}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    instances = list(dataset)
    if args.num_instances:
        instances = instances[: args.num_instances]
    print(f"Loaded {len(instances)} instances")

    # Generate patches in parallel
    predictions = []
    tic = time.perf_counter()

    def process_instance(instance):
        raw_output = generate_patch(
            client, model, instance, args.max_tokens, args.temperature
        )
        patch = extract_patch(raw_output)
        return {
            "instance_id": instance["instance_id"],
            "model_patch": patch,
            "model_name_or_path": model,
        }

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(process_instance, inst): inst for inst in instances
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            predictions.append(future.result())

    latency = time.perf_counter() - tic

    # Sort by instance_id for reproducibility
    predictions.sort(key=lambda x: x["instance_id"])

    # Save predictions
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved {len(predictions)} predictions to {args.output_file}")

    # Statistics
    non_empty = sum(1 for p in predictions if p["model_patch"].strip())
    print(f"Non-empty patches: {non_empty}/{len(predictions)}")
    print(f"Total latency: {latency:.1f}s")
    print(f"Average latency per instance: {latency / len(predictions):.1f}s")

    # Save benchmark result
    with open(args.result_file, "a") as fout:
        value = {
            "task": "swe_bench",
            "backend": "sglang",
            "dataset": args.dataset,
            "model": model,
            "latency": round(latency, 3),
            "num_instances": len(predictions),
            "non_empty_patches": non_empty,
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SWE-bench inference using SGLang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base URL (overrides --host and --port)",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name/path. If not set, auto-detected from server.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Number of instances to evaluate (default: all)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="Number of parallel requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for model output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions.json",
        help="Output file for predictions (SWE-bench submission format)",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="result.jsonl",
        help="File to append benchmark result",
    )
    args = parser.parse_args()
    main(args)
