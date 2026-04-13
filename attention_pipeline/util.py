"""Data loading, group discovery, and shared helpers."""

import logging
import os
import re
import sys
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from config import COMPLAINT_CATEGORIES

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_merged_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from '%s'", len(df), input_path)

    if "actual_score" not in df.columns:
        logger.error("'actual_score' column not found")
        sys.exit(1)

    pred_cols = sorted(
        [c for c in df.columns if c.startswith("predicted_score__")]
    )
    if not pred_cols:
        logger.error("No predicted_score__ columns found")
        sys.exit(1)

    return df


def parse_run_label(run_label: str) -> Tuple[Optional[str], Optional[str]]:
    if "__" not in run_label:
        return None, None
    parts = run_label.split("__", 1)
    return parts[0], parts[1]


def discover_groups(pred_cols: List[str]) -> Dict[str, Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = {}
    for col in pred_cols:
        run_label = col.replace("predicted_score__", "")
        variant, model = parse_run_label(run_label)
        if variant is None or model is None:
            continue
        if model not in groups:
            groups[model] = {}
        groups[model][variant] = col
    return groups


def get_valid_data(
    df: pd.DataFrame,
    variant_cols: Dict[str, str],
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    valid_mask = df["actual_score"].notna()
    for col in variant_cols.values():
        valid_mask &= df[col].notna()
    df_valid = df[valid_mask].copy()
    y_true = df_valid["actual_score"].astype(int).values
    predictions = {
        v: df_valid[col].astype(int).values for v, col in variant_cols.items()
    }
    return df_valid, y_true, predictions


def get_prompt_arrays(
    df_valid: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df_valid)
    prompt_hashes = (
        df_valid["prompt_hash"].values
        if "prompt_hash" in df_valid.columns
        else np.arange(n).astype(str)
    )
    prompts = (
        df_valid["prompt"].values
        if "prompt" in df_valid.columns
        else np.array([""] * n)
    )
    return prompt_hashes, prompts


def categorize_complaint(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for category, pattern in COMPLAINT_CATEGORIES.items():
        if re.search(pattern, prompt_lower):
            return category
    return "other"
    return "other"