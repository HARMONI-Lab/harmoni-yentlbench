#!/usr/bin/env python3
"""
ESI Triage Evaluation Run Merger

Parses, normalizes, and joins multiple evaluation run results from a batch
ESI (Emergency Severity Index) triage scoring system.

Usage:
    python merge_runs.py --results-dir results --output results/merged_evaluations.csv
    python merge_runs.py --include-metrics --verbose
"""

import json
import hashlib
import logging
import argparse
import re
import os
import glob
import sys
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from functools import reduce

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class SubrunResult:
    """Structured representation of a single subrun evaluation result."""

    run_id: Optional[str] = None
    state: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    prompt: Optional[str] = None
    prompt_hash: Optional[str] = None
    actual_score: Optional[float] = None
    predicted_score: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
REQUIRED_TOP_LEVEL_KEYS = {"subruns"}


def hash_prompt(text: str) -> str:
    """Create a short deterministic ID from prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_prompt(text: str) -> str:
    """Normalize a prompt by stripping and cutting before 'Chief complaint:'."""
    text = text.strip()
    cc_idx = text.find("Chief complaint:")
    if cc_idx != -1:
        text = text[cc_idx:]
    return text


def validate_run_file(data: Dict[str, Any], file_path: str) -> bool:
    """Return True if *data* contains all required top-level keys."""
    missing = REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    if missing:
        logger.error("%s is missing required keys: %s", file_path, missing)
        return False
    return True


def extract_run_label(filename: str) -> str:
    """Extract a composite label from *filename* capturing both the
    demographic variant and the model name.

    Example filenames:
        batch_esi_triage_scorer_female-run_id_Run_1_openai_gpt-5.4-2026-03-05.run.json
        batch_esi_triage_scorer_nb_ambiguous-run_id_Run_1_openai_gpt-5.4-mini-2026-03-17.run.json

    Returns labels like:
        female__openai_gpt-5.4-2026-03-05
        nb_ambiguous__openai_gpt-5.4-mini-2026-03-17
    """
    variant_match = re.search(r"scorer_(.+?)-run_id", filename)
    variant = variant_match.group(1) if variant_match else "unknown_variant"

    model_match = re.search(r"Run_\d+_(.+?)\.run\.json$", filename)
    model = model_match.group(1) if model_match else "unknown_model"

    return f"{variant}__{model}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _extract_prompt(conversations: List[Dict]) -> Optional[str]:
    """Safely walk the nested conversation structure and return the prompt."""
    if not conversations:
        return None
    first_conv = conversations[0]

    requests = first_conv.get("requests")
    if not requests:
        return None

    contents = requests[0].get("contents")
    if not contents:
        return None

    parts = contents[0].get("parts")
    if not parts:
        return None

    text = parts[0].get("text")
    if not text:
        return None

    return normalize_prompt(text)


def _extract_metrics(conversations: List[Dict]) -> Dict[str, Any]:
    """Return a dict of token / latency metrics (may be empty)."""
    if not conversations:
        return {}
    metrics = conversations[0].get("metrics")
    if not metrics:
        return {}
    return {
        "input_tokens": metrics.get("inputTokens"),
        "output_tokens": metrics.get("outputTokens"),
        "total_latency_ms": metrics.get("totalBackendLatencyMs"),
    }


def _extract_results(subrun: Dict) -> Dict[str, Any]:
    """Return actual_score and predicted_score from subrun results."""
    results = subrun.get("results", [])
    if not results:
        return {}
    dict_res = results[0].get("dictResult")
    if not dict_res:
        return {}
    return {
        "actual_score": dict_res.get("actual_score"),
        "predicted_score": dict_res.get("predicted_score"),
    }


def parse_run_results(
    file_path: str,
    include_metrics: bool = False,
) -> pd.DataFrame:
    """Parse a single run JSON file and return a tidy DataFrame."""

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not validate_run_file(data, file_path):
        logger.warning("Skipping invalid file: %s", file_path)
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    subruns = data.get("subruns", [])

    for idx, subrun in enumerate(subruns):
        conversations = subrun.get("conversations", [])
        prompt_text = _extract_prompt(conversations)

        if prompt_text is None:
            logger.warning(
                "%s: subrun %d (pyRunId=%s) has no extractable prompt — skipping",
                file_path,
                idx,
                subrun.get("pyRunId"),
            )
            continue

        result = SubrunResult(
            run_id=subrun.get("pyRunId"),
            state=subrun.get("state"),
            start_time=subrun.get("startTime"),
            end_time=subrun.get("endTime"),
            prompt=prompt_text,
            prompt_hash=hash_prompt(prompt_text),
        )

        # Evaluation scores
        scores = _extract_results(subrun)
        result.actual_score = scores.get("actual_score")
        result.predicted_score = scores.get("predicted_score")

        # Optional metrics
        if include_metrics:
            metrics = _extract_metrics(conversations)
            result.input_tokens = metrics.get("input_tokens")
            result.output_tokens = metrics.get("output_tokens")
            result.total_latency_ms = metrics.get("total_latency_ms")

        rows.append(asdict(result))

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("%s produced an empty DataFrame", file_path)
        return df

    missing_prompts = df["prompt"].isna().sum()
    if missing_prompts > 0:
        logger.warning(
            "%s: %d rows still missing prompts after parsing — dropping them",
            file_path,
            missing_prompts,
        )
        df = df.dropna(subset=["prompt"])

    return df


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

JOIN_KEYS = ["prompt_hash", "prompt"]
RUN_META_COLS = ["run_id", "state", "start_time", "end_time"]


def prepare_for_merge(
    df: pd.DataFrame,
    suffix: str,
    include_metrics: bool = False,
) -> pd.DataFrame:
    """Drop run metadata and rename value columns with *suffix*."""

    drop_cols = [c for c in RUN_META_COLS if c in df.columns]
    if not include_metrics:
        drop_cols += [
            c
            for c in ("input_tokens", "output_tokens", "total_latency_ms")
            if c in df.columns
        ]

    df_clean = df.drop(columns=drop_cols)

    keep_as_is = set(JOIN_KEYS)

    rename_map = {
        col: f"{col}__{suffix}" for col in df_clean.columns if col not in keep_as_is
    }
    df_clean = df_clean.rename(columns=rename_map)
    return df_clean


def validate_actual_scores(merged_df: pd.DataFrame) -> None:
    """Warn if actual_score columns from different runs disagree."""
    actual_cols = sorted(
        [c for c in merged_df.columns if c.startswith("actual_score")]
    )
    if len(actual_cols) <= 1:
        return

    base_col = actual_cols[0]
    base = merged_df[base_col]
    for col in actual_cols[1:]:
        mismatches = (base != merged_df[col]) & base.notna() & merged_df[col].notna()
        n = mismatches.sum()
        if n > 0:
            logger.warning(
                "actual_score mismatch between '%s' and '%s' in %d rows",
                base_col,
                col,
                n,
            )


def merge_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge a list of DataFrames on JOIN_KEYS."""

    if len(dataframes) == 1:
        merged = dataframes[0].copy()
    else:
        # Set JOIN_KEYS as index to perform highly optimized outer join via concat
        dfs_indexed = [df.set_index(JOIN_KEYS) for df in dataframes]
        merged = pd.concat(dfs_indexed, axis=1).reset_index()

    # Consolidate actual_score columns
    actual_cols = sorted(
        [c for c in merged.columns if c.startswith("actual_score")]
    )
    if actual_cols:
        validate_actual_scores(merged)
        merged["actual_score"] = merged[actual_cols].bfill(axis=1).iloc[:, 0]
        cols_to_drop = [c for c in actual_cols if c != "actual_score"]
        merged = merged.drop(columns=cols_to_drop)

    # Reorder columns nicely
    leading = ["prompt_hash", "prompt", "actual_score"]
    rest = sorted([c for c in merged.columns if c not in leading])
    merged = merged[[c for c in leading if c in merged.columns] + rest]

    return merged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(merged_df: pd.DataFrame) -> None:
    """Print per-run accuracy and inter-run agreement based on
    predicted_score vs actual_score."""

    pred_cols = sorted(
        [c for c in merged_df.columns if c.startswith("predicted_score")]
    )
    if not pred_cols:
        logger.info("No predicted_score columns found — skipping summary.")
        return

    if "actual_score" not in merged_df.columns:
        logger.info("No actual_score column found — skipping summary.")
        return

    # Compute accuracy for each run by comparing predicted to actual
    print("\n===== Per-Run Accuracy =====")
    for col in pred_cols:
        label = col.replace("predicted_score__", "")
        correct_mask = merged_df[col] == merged_df["actual_score"]
        n_total = merged_df[col].notna().sum()
        n_correct = int((correct_mask & merged_df[col].notna()).sum())
        acc = n_correct / n_total if n_total else 0
        print(f"  {label}: {acc:.2%}  ({n_correct}/{n_total})")

    # Agreement: all runs agree on the predicted score for the same case
    if len(pred_cols) > 1:
        valid = merged_df[pred_cols].dropna()
        if not valid.empty:
            agreement = valid.nunique(axis=1) == 1
            print(
                f"\n  Inter-run agreement rate: {agreement.mean():.2%}  "
                f"({agreement.sum()}/{len(agreement)} cases)"
            )

    # Preview
    preview_cols = ["prompt_hash", "actual_score"] + pred_cols
    preview_cols = [c for c in preview_cols if c in merged_df.columns]
    print(
        f"\nPreview of joined predictions:\n"
        f"{merged_df[preview_cols].head(10).to_string(index=False)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ESI triage evaluation run results.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing *.run.json files (default: results)",
    )
    parser.add_argument(
        "--output",
        default="results/merged_evaluations.csv",
        help="Output CSV path (default: results/merged_evaluations.csv)",
    )
    parser.add_argument(
        "--include-metrics",
        action="store_true",
        help="Include token count and latency metrics in the output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    pattern = os.path.join(args.results_dir, "*.run.json")
    file_paths = sorted(glob.glob(pattern))

    if not file_paths:
        logger.error("No *.run.json files found in '%s'", args.results_dir)
        sys.exit(1)

    logger.info("Found %d run file(s) in '%s'", len(file_paths), args.results_dir)

    dataframes: List[pd.DataFrame] = []
    seen_suffixes: Dict[str, str] = {}

    for file_path in file_paths:
        logger.info("Parsing: %s", file_path)
        df = parse_run_results(file_path, include_metrics=args.include_metrics)
        if df.empty:
            logger.warning("Skipping empty result from %s", file_path)
            continue

        filename = os.path.basename(file_path)
        suffix = extract_run_label(filename)

        if suffix in seen_suffixes:
            logger.error(
                "Suffix collision! '%s' maps to both:\n  %s\n  %s\n"
                "Rename your files or update extract_run_label().",
                suffix,
                seen_suffixes[suffix],
                file_path,
            )
            sys.exit(1)
        seen_suffixes[suffix] = file_path

        logger.debug("Run label: %s  (%d rows)", suffix, len(df))

        df_clean = prepare_for_merge(
            df, suffix, include_metrics=args.include_metrics
        )
        dataframes.append(df_clean)

    if not dataframes:
        logger.error("No valid data extracted from any file. Exiting.")
        sys.exit(1)

    merged_df = merge_dataframes(dataframes)

    print(f"\nSuccessfully joined {len(dataframes)} table(s).")
    print(f"Joined table shape: {merged_df.shape}")
    print(f"Columns: {merged_df.columns.tolist()}")

    print_summary(merged_df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged_df.to_csv(args.output, index=False)
    print(f"\nSaved full merged results to '{args.output}'")


if __name__ == "__main__":
    main()