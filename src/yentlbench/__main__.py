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
    print("Local LLM runner is not yet implemented (Milestone 2).")
    sys.exit(1)

def run_merge(args):
    # merge_runs.py expects argv: [prog, --results-dir, ..., --output, ...]
    argv = ["yentlbench", "--results-dir", args.results_dir, "--output", args.output]
    if args.include_metrics:
        argv.append("--include-metrics")
    if args.verbose:
        argv.append("--verbose")
    merge_main(argv)

def run_analyze(args):
    # benchmark_stats.py
    check_artifact(args.input, "merge")
    
    stats_argv = ["yentlbench", "--input", args.input, "--output", args.output_stats, "--verbose"]
    if args.verbose:
        stats_argv.append("--verbose")
    stats_main(stats_argv)
    
    # attention_pipeline/pipeline.py
    attention_argv = ["yentlbench", "--input", args.input, "--output-dir", args.output_attention]
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
