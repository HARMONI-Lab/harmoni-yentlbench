import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from yentlbench.dataset_prep import expand as prep_expand
from yentlbench.merge_runs import main as merge_main
from yentlbench.benchmark_stats import main as stats_main
from yentlbench.attention_pipeline.pipeline import main as attention_main

def check_artifact(path: str, stage_name: str):
    if not os.path.exists(path):
        print(f"Error: Missing required artifact from '{stage_name}' stage: {path}")
        sys.exit(1)

def run_prepare(args):
    # dataset_prep.py is a scripted process, not a function. 
    # However, we can wrap the logic.
    import yentlbench.dataset_prep as dp
    # dataset_prep.py runs logic at top level; for a proper entrypoint, 
    # we would want to wrap it in a function. Since the user said 
    # "nothing about the logic changes", we will call it as a script 
    # or manually trigger the parts we can.
    # Actually, dataset_prep.py was written as a script. 
    # I'll import the logic and call whatever is available or use bash.
    # Better: since I can't change logic, I'll use os.system or subprocess
    # but for 'prepare', it doesn't have args.
    import subprocess
    subprocess.run(["python3", "-m", "yentlbench.dataset_prep"])

def run_run(args):
    from yentlbench.local_runner.ollama_runner import OllamaRunner, OllamaNotRunningError
    
    if args.list_models:
        runner = OllamaRunner(model_name="", host=args.host)
        try:
            models = runner.list_models()
            print("Available Ollama models:")
            for m in models:
                print(f"  - {m}")
        except OllamaNotRunningError as e:
            print(f"Error: {e}")
        sys.exit(0)
        
    if not args.model or not args.variants:
        print("Error: --model and --variants are required unless --list-models is used.")
        sys.exit(1)
        
    dataset_path = args.data
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run 'prepare' first.")
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    # Filter rows with NaN in acuity
    df = df[df["acuity"].notna()]
    vignettes = df.to_dict(orient="records")
    
    runner = OllamaRunner(model_name=args.model, host=args.host)
    try:
        for variant in args.variants:
            runner.run(vignettes, variant, run_number=args.run_id, output_dir=args.results_dir, dry_run=args.dry_run)
    except OllamaNotRunningError as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_merge(args):
    # merge_runs.py expects argv: [--results-dir, ..., --output, ...]
    argv = ["--results-dir", args.results_dir, "--output", args.output]
    if args.include_metrics:
        argv.append("--include-metrics")
    if args.verbose:
        argv.append("--verbose")
    merge_main(argv)

def run_analyze(args):
    # benchmark_stats.py
    check_artifact(args.input, "merge")
    
    stats_argv = ["--input", args.input, "--output", args.output_stats]
    if args.verbose:
        stats_argv.append("--verbose")
    stats_main(stats_argv)
    
    # attention_pipeline/pipeline.py
    attention_argv = ["--input", args.input, "--output-dir", args.output_attention]
    if args.verbose:
        attention_argv.append("--verbose")
    attention_main(attention_argv)

def main():
    parser = argparse.ArgumentParser(prog="yentlbench", description="YentlBench Workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Prepare
    prep_p = subparsers.add_parser("prepare", help="Prepare dataset and expand quintets")

    # Run
    run_p = subparsers.add_parser("run", help="Run local LLM evaluations")
    run_p.add_argument("--model", required=False, help="Ollama model name (e.g. llama3:8b)")
    run_p.add_argument("--variants", nargs="+", default=["nb_ambiguous", "female", "male", "nb_label_only"], help="Target variants to run (default: all four)")
    run_p.add_argument("--host", default="http://localhost:11434", help="Ollama host url")
    run_p.add_argument("--run-id", type=int, default=1, help="Run ID to append to filename (e.g., 1)")
    run_p.add_argument("--results-dir", default="results", help="Directory to save .run.json files")
    run_p.add_argument("--data", default="eval/dataset_quintets.csv", help="Path to dataset_quintets.csv")
    run_p.add_argument("--list-models", action="store_true", help="List available Ollama models")
    run_p.add_argument("--dry-run", action="store_true", help="Print run info without calling Ollama endpoints")

    # Merge
    merge_p = subparsers.add_parser("merge", help="Merge run results")
    merge_p.add_argument("--results-dir", default="results", help="Directory with .run.json files")
    merge_p.add_argument("--output", default="eval/merged_evaluations.csv", help="Output CSV path")
    merge_p.add_argument("--include-metrics", action="store_true", help="Include metrics")
    merge_p.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Analyze
    analyze_p = subparsers.add_parser("analyze", help="Compute benchmark stats and attention analysis")
    analyze_p.add_argument("--input", default="eval/merged_evaluations.csv", help="Input merged CSV")
    analyze_p.add_argument("--output-stats", default="eval/benchmark_stats.csv", help="Stats output CSV")
    analyze_p.add_argument("--output-attention", default="eval/attention", help="Attention output dir")
    analyze_p.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "run":
        run_run(args)
    elif args.command == "merge":
        run_merge(args)
    elif args.command == "analyze":
        run_analyze(args)

if __name__ == "__main__":
    main()
