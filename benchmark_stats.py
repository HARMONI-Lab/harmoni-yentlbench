#!/usr/bin/env python3
"""
ESI Triage Benchmark Statistics

Computes per-run benchmark statistics from the merged evaluation CSV
produced by merge_runs.py.

Usage:
    python benchmark_stats.py --input results/merged_evaluations.csv
    python benchmark_stats.py --input results/merged_evaluations.csv --output results/benchmark_stats.csv --verbose
"""

import argparse
import logging
import os
import sys
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    log_loss,
)

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
# Metric computation
# ---------------------------------------------------------------------------

# ESI levels (the expected label space)
ESI_LEVELS = [1, 2, 3, 4, 5]


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """Return numerator/denominator or None if denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def compute_distribution_stats(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, Any]:
    """Distribution and coverage statistics."""
    stats: Dict[str, Any] = {}

    stats["n_total"] = len(y_true)
    stats["n_missing_pred"] = int(y_pred.isna().sum())
    stats["n_evaluated"] = int(y_pred.notna().sum())
    stats["coverage"] = _safe_divide(stats["n_evaluated"], stats["n_total"])

    # Predicted label distribution
    pred_counts = y_pred.value_counts(normalize=True).sort_index()
    true_counts = y_true.value_counts(normalize=True).sort_index()
    for level in ESI_LEVELS:
        stats[f"true_pct_esi_{level}"] = true_counts.get(level, 0.0)
        stats[f"pred_pct_esi_{level}"] = pred_counts.get(level, 0.0)

    return stats


def compute_classification_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Core classification metrics."""
    stats: Dict[str, Any] = {}

    # --- Overall accuracy ---
    stats["accuracy"] = accuracy_score(y_true, y_pred)
    stats["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # --- Weighted, macro, micro averages ---
    for average in ("macro", "weighted", "micro"):
        stats[f"precision_{average}"] = precision_score(
            y_true, y_pred, labels=ESI_LEVELS, average=average, zero_division=0
        )
        stats[f"recall_{average}"] = recall_score(
            y_true, y_pred, labels=ESI_LEVELS, average=average, zero_division=0
        )
        stats[f"f1_{average}"] = f1_score(
            y_true, y_pred, labels=ESI_LEVELS, average=average, zero_division=0
        )

    # --- Per-class metrics ---
    for level in ESI_LEVELS:
        y_true_bin = (y_true == level).astype(int)
        y_pred_bin = (y_pred == level).astype(int)

        support = int(y_true_bin.sum())
        stats[f"support_esi_{level}"] = support

        if support > 0:
            stats[f"precision_esi_{level}"] = precision_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            stats[f"recall_esi_{level}"] = recall_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            stats[f"f1_esi_{level}"] = f1_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
        else:
            stats[f"precision_esi_{level}"] = None
            stats[f"recall_esi_{level}"] = None
            stats[f"f1_esi_{level}"] = None

    # --- Cohen's Kappa (inter-rater agreement vs chance) ---
    stats["cohen_kappa"] = cohen_kappa_score(y_true, y_pred, labels=ESI_LEVELS)

    # --- Weighted (quadratic) Kappa — penalizes distant misclassifications ---
    stats["cohen_kappa_quadratic"] = cohen_kappa_score(
        y_true, y_pred, labels=ESI_LEVELS, weights="quadratic"
    )

    # --- Matthews Correlation Coefficient ---
    stats["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)

    return stats


def compute_ordinal_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Ordinal / regression-style metrics that respect the ESI level ordering."""
    stats: Dict[str, Any] = {}

    errors = y_pred - y_true

    # --- Error magnitude ---
    stats["mean_absolute_error"] = mean_absolute_error(y_true, y_pred)
    stats["root_mean_squared_error"] = np.sqrt(mean_squared_error(y_true, y_pred))
    stats["max_absolute_error"] = int(np.max(np.abs(errors)))
    stats["median_absolute_error"] = float(np.median(np.abs(errors)))

    # --- Directional bias ---
    stats["mean_error"] = float(np.mean(errors))  # positive = over-triage tendency
    stats["std_error"] = float(np.std(errors, ddof=1)) if len(errors) > 1 else None

    # --- Within-1-level accuracy (common ESI benchmark metric) ---
    within_1 = np.abs(errors) <= 1
    stats["accuracy_within_1"] = float(np.mean(within_1))

    # --- Exact + adjacent accuracy (within 0 or 1 level) ---
    stats["exact_match_rate"] = float(np.mean(errors == 0))

    # --- Over-triage vs under-triage rates ---
    # Over-triage: predicted MORE urgent (lower ESI number) than actual
    # Under-triage: predicted LESS urgent (higher ESI number) than actual
    stats["over_triage_rate"] = float(np.mean(errors < 0))
    stats["under_triage_rate"] = float(np.mean(errors > 0))
    stats["over_triage_mean_magnitude"] = (
        float(np.mean(np.abs(errors[errors < 0]))) if np.any(errors < 0) else 0.0
    )
    stats["under_triage_mean_magnitude"] = (
        float(np.mean(errors[errors > 0])) if np.any(errors > 0) else 0.0
    )

    # --- Spearman rank correlation ---
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        spearman_r, spearman_p = scipy_stats.spearmanr(y_true, y_pred)
        stats["spearman_r"] = spearman_r
        stats["spearman_p"] = spearman_p
    else:
        stats["spearman_r"] = None
        stats["spearman_p"] = None

    # --- Kendall's Tau (ordinal association) ---
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        tau, tau_p = scipy_stats.kendalltau(y_true, y_pred)
        stats["kendall_tau"] = tau
        stats["kendall_tau_p"] = tau_p
    else:
        stats["kendall_tau"] = None
        stats["kendall_tau_p"] = None

    return stats


def compute_confusion_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Flatten the confusion matrix into named entries."""
    stats: Dict[str, Any] = {}

    cm = confusion_matrix(y_true, y_pred, labels=ESI_LEVELS)

    for i, true_level in enumerate(ESI_LEVELS):
        for j, pred_level in enumerate(ESI_LEVELS):
            stats[f"cm_true{true_level}_pred{pred_level}"] = int(cm[i, j])

    return stats


def compute_clinical_safety_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Clinical safety metrics specific to ESI triage.

    Critical misclassifications where patient safety is at risk:
    - Critical under-triage: ESI 1-2 patients classified as ESI 4-5
    - Severe under-triage:   ESI 1-2 patients classified as ESI 3+
    - Critical over-triage:  ESI 4-5 patients classified as ESI 1
    """
    stats: Dict[str, Any] = {}

    errors = y_pred - y_true

    # High-acuity patients (ESI 1-2)
    high_acuity_mask = np.isin(y_true, [1, 2])
    n_high_acuity = int(high_acuity_mask.sum())
    stats["n_high_acuity_cases"] = n_high_acuity

    if n_high_acuity > 0:
        high_acuity_pred = y_pred[high_acuity_mask]
        high_acuity_true = y_true[high_acuity_mask]
        high_acuity_errors = errors[high_acuity_mask]

        # Accuracy on high-acuity
        stats["high_acuity_accuracy"] = float(
            np.mean(high_acuity_pred == high_acuity_true)
        )

        # Severe under-triage: ESI 1-2 → ESI 3+
        severe_under = high_acuity_pred >= 3
        stats["severe_under_triage_count"] = int(severe_under.sum())
        stats["severe_under_triage_rate"] = float(np.mean(severe_under))

        # Critical under-triage: ESI 1-2 → ESI 4-5
        critical_under = high_acuity_pred >= 4
        stats["critical_under_triage_count"] = int(critical_under.sum())
        stats["critical_under_triage_rate"] = float(np.mean(critical_under))
    else:
        stats["high_acuity_accuracy"] = None
        stats["severe_under_triage_count"] = 0
        stats["severe_under_triage_rate"] = None
        stats["critical_under_triage_count"] = 0
        stats["critical_under_triage_rate"] = None

    # Low-acuity patients (ESI 4-5)
    low_acuity_mask = np.isin(y_true, [4, 5])
    n_low_acuity = int(low_acuity_mask.sum())
    stats["n_low_acuity_cases"] = n_low_acuity

    if n_low_acuity > 0:
        low_acuity_pred = y_pred[low_acuity_mask]
        low_acuity_true = y_true[low_acuity_mask]

        stats["low_acuity_accuracy"] = float(
            np.mean(low_acuity_pred == low_acuity_true)
        )

        # Critical over-triage: ESI 4-5 → ESI 1
        critical_over = low_acuity_pred == 1
        stats["critical_over_triage_count"] = int(critical_over.sum())
        stats["critical_over_triage_rate"] = float(np.mean(critical_over))
    else:
        stats["low_acuity_accuracy"] = None
        stats["critical_over_triage_count"] = 0
        stats["critical_over_triage_rate"] = None

    # ESI 1 sensitivity (most critical: do we catch all resuscitation cases?)
    esi1_mask = y_true == 1
    n_esi1 = int(esi1_mask.sum())
    stats["n_esi1_cases"] = n_esi1
    if n_esi1 > 0:
        stats["esi1_sensitivity"] = float(np.mean(y_pred[esi1_mask] == 1))
    else:
        stats["esi1_sensitivity"] = None

    return stats


def compute_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for a given metric function."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"ci_lower": None, "ci_upper": None, "ci_std": None}

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "ci_lower": float(np.percentile(scores, 100 * alpha)),
        "ci_upper": float(np.percentile(scores, 100 * (1 - alpha))),
        "ci_std": float(np.std(scores)),
    }


def compute_run_stats(
    y_true_full: pd.Series,
    y_pred_full: pd.Series,
    run_label: str,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Compute all benchmark statistics for a single run."""

    stats: Dict[str, Any] = {"run": run_label}

    # --- Distribution stats (before dropping NaNs) ---
    stats.update(compute_distribution_stats(y_true_full, y_pred_full))

    # --- Drop NaN predictions for classification metrics ---
    valid_mask = y_pred_full.notna() & y_true_full.notna()
    y_true = y_true_full[valid_mask].astype(int).values
    y_pred = y_pred_full[valid_mask].astype(int).values

    if len(y_true) == 0:
        logger.warning("Run '%s' has no valid predictions — skipping metrics.", run_label)
        return stats

    # --- Classification ---
    stats.update(compute_classification_stats(y_true, y_pred))

    # --- Ordinal ---
    stats.update(compute_ordinal_stats(y_true, y_pred))

    # --- Clinical safety ---
    stats.update(compute_clinical_safety_stats(y_true, y_pred))

    # --- Confusion matrix ---
    stats.update(compute_confusion_stats(y_true, y_pred))

    # --- Bootstrap CI for key metrics ---
    ci_metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "cohen_kappa": lambda yt, yp: cohen_kappa_score(yt, yp, labels=ESI_LEVELS),
        "f1_macro": lambda yt, yp: f1_score(
            yt, yp, labels=ESI_LEVELS, average="macro", zero_division=0
        ),
        "mae": mean_absolute_error,
    }
    for metric_name, metric_fn in ci_metrics.items():
        ci_result = compute_confidence_interval(
            y_true, y_pred, metric_fn, n_bootstrap=n_bootstrap
        )
        stats[f"{metric_name}_ci_lower"] = ci_result["ci_lower"]
        stats[f"{metric_name}_ci_upper"] = ci_result["ci_upper"]
        stats[f"{metric_name}_ci_std"] = ci_result["ci_std"]

    return stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def load_merged_data(input_path: str) -> pd.DataFrame:
    """Load the merged CSV and validate expected columns."""
    if not os.path.exists(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path)

    if "actual_score" not in df.columns:
        logger.error("'actual_score' column not found in %s", input_path)
        sys.exit(1)

    pred_cols = [c for c in df.columns if c.startswith("predicted_score__")]
    if not pred_cols:
        logger.error("No 'predicted_score__*' columns found in %s", input_path)
        sys.exit(1)

    logger.info(
        "Loaded %d rows, %d prediction columns from '%s'",
        len(df),
        len(pred_cols),
        input_path,
    )
    return df


def build_stats_table(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute stats for every predicted_score__ column and return a DataFrame."""

    pred_cols = sorted([c for c in df.columns if c.startswith("predicted_score__")])
    all_stats: List[Dict[str, Any]] = []

    for col in pred_cols:
        run_label = col.replace("predicted_score__", "")
        logger.info("Computing stats for: %s", run_label)

        run_stats = compute_run_stats(
            y_true_full=df["actual_score"],
            y_pred_full=df[col],
            run_label=run_label,
            n_bootstrap=n_bootstrap,
        )
        all_stats.append(run_stats)

    stats_df = pd.DataFrame(all_stats)

    # Move 'run' to be the first column
    cols = ["run"] + [c for c in stats_df.columns if c != "run"]
    stats_df = stats_df[cols]

    return stats_df


def print_stats_summary(stats_df: pd.DataFrame) -> None:
    """Print a compact summary to the console."""

    highlight_cols = [
        "run",
        "n_evaluated",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "cohen_kappa",
        "cohen_kappa_quadratic",
        "matthews_corrcoef",
        "mean_absolute_error",
        "accuracy_within_1",
        "over_triage_rate",
        "under_triage_rate",
        "severe_under_triage_rate",
        "critical_under_triage_rate",
        "esi1_sensitivity",
    ]
    available = [c for c in highlight_cols if c in stats_df.columns]

    print("\n===== Benchmark Summary =====\n")

    # Transpose for readability when there are many runs
    summary = stats_df[available].set_index("run").T

    # Format floats
    formatted = summary.map(
        lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    )
    print(formatted.to_string())

    # Highlight best performer for key metrics
    print("\n===== Best Performers =====\n")
    higher_is_better = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "cohen_kappa",
        "cohen_kappa_quadratic",
        "matthews_corrcoef",
        "accuracy_within_1",
        "esi1_sensitivity",
    ]
    lower_is_better = [
        "mean_absolute_error",
        "over_triage_rate",
        "under_triage_rate",
        "severe_under_triage_rate",
        "critical_under_triage_rate",
    ]

    ranked = stats_df[available].set_index("run")
    for metric in higher_is_better:
        if metric in ranked.columns:
            best_run = ranked[metric].idxmax()
            best_val = ranked[metric].max()
            print(f"  {metric:>35s}: {best_run}  ({best_val:.4f})")

    for metric in lower_is_better:
        if metric in ranked.columns:
            best_run = ranked[metric].idxmin()
            best_val = ranked[metric].min()
            print(f"  {metric:>35s}: {best_run}  ({best_val:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute benchmark statistics from merged ESI evaluation results.",
    )
    parser.add_argument(
        "--input",
        default="results/merged_evaluations.csv",
        help="Path to merged evaluations CSV (default: results/merged_evaluations.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_stats.csv",
        help="Output CSV for the stats table (default: results/benchmark_stats.csv)",
    )
    parser.add_argument(
        "--output-full",
        default="results/benchmark_stats_full.csv",
        help="Output CSV with ALL metrics including confusion matrix cells "
             "(default: results/benchmark_stats_full.csv)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals (default: 1000)",
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

    df = load_merged_data(args.input)
    stats_df = build_stats_table(df, n_bootstrap=args.n_bootstrap)

    print_stats_summary(stats_df)

    # --- Save compact version (no confusion matrix cells) ---
    cm_cols = [c for c in stats_df.columns if c.startswith("cm_")]
    compact_df = stats_df.drop(columns=cm_cols)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    compact_df.to_csv(args.output, index=False)
    print(f"\nSaved compact stats to '{args.output}'")

    # --- Save full version (everything) ---
    os.makedirs(os.path.dirname(args.output_full) or ".", exist_ok=True)
    stats_df.to_csv(args.output_full, index=False)
    print(f"Saved full stats (incl. confusion matrix) to '{args.output_full}'")


if __name__ == "__main__":
    main()