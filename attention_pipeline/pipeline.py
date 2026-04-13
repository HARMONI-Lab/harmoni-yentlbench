#!/usr/bin/env python3
"""
ESI Triage Gender Attention Analysis — Pipeline Orchestrator

Runs all 10 analyses for each model discovered in the merged evaluations CSV,
prints per-model and cross-model reports, and saves all outputs.

Usage:
    python pipeline.py --input eval/merged_evaluations.csv
    python pipeline.py --input eval/merged_evaluations.csv --output-dir eval/attention --verbose
"""

import argparse
import logging
import os
import sys
import warnings
from typing import Optional, List, Dict, Any

from config import BASELINE_VARIANT
from util import (
    setup_logging,
    load_merged_data,
    discover_groups,
    get_valid_data,
    get_prompt_arrays,
)
from analyze_baseline import analyze_baseline_deviation, decompose_sex_info_effect, compute_transition_matrices, analyze_transition_risk
from analyze_sensitivity import analyze_information_leakage, compute_sensitivity_scores
from analyze_vulnerability import (
    compute_vulnerability_by_esi,
    compute_vulnerability_by_clinical_category,
    analyze_decision_boundaries,
    analyze_consistency_by_difficulty,
)
from analyze_pairwise import build_case_detail_table, analyze_all_pairs
from analyze_significance import analyze_omnibus_significance
from report import (
    print_model_report,
    build_cross_model_summary,
    print_cross_model_summary,
)
from save import save_model_results, save_cross_model_results
from visuals import generate_visuals, generate_cross_model_visuals

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-model orchestrator
# ---------------------------------------------------------------------------
def analyze_model(
    df,
    model: str,
    variant_cols: Dict[str, str],
) -> Dict[str, Any]:
    """Run all 11 analyses for a single model.

    Returns a dict with all results keyed by analysis name,
    or an empty dict if no valid data is available.
    """
    df_valid, y_true, predictions = get_valid_data(df, variant_cols)
    n_valid = len(df_valid)

    if n_valid == 0:
        logger.warning("No valid cases for model '%s'", model)
        return {}

    prompt_hashes, prompts = get_prompt_arrays(df_valid)

    logger.info(
        "Analyzing '%s': %d cases, variants: %s",
        model,
        n_valid,
        sorted(predictions.keys()),
    )

    result: Dict[str, Any] = {"model": model, "n_cases": n_valid}

    # 1. Baseline deviation
    result["baseline_deviation"] = analyze_baseline_deviation(y_true, predictions)

    # 2. Effect decomposition
    result["effect_decomposition"] = decompose_sex_info_effect(y_true, predictions)

    # 3. Transition matrices + risk classification
    result["transition_matrices"] = compute_transition_matrices(predictions)
    result["dangerous_transitions"] = analyze_transition_risk(
        result["transition_matrices"]
    )

    # 4. Information leakage
    result["information_leakage"] = analyze_information_leakage(y_true, predictions)

    # 5. Sensitivity scoring
    result["sensitivity"] = compute_sensitivity_scores(predictions)

    # 6. Vulnerability by ESI level and clinical category
    result["vulnerability_by_esi"] = compute_vulnerability_by_esi(
        y_true, predictions
    )
    result["vulnerability_by_category"] = compute_vulnerability_by_clinical_category(
        y_true, predictions, prompts
    )

    # 7. Decision boundary analysis
    result["boundary_analysis"] = analyze_decision_boundaries(y_true, predictions)

    # 8. Consistency by difficulty
    result["consistency_by_difficulty"] = analyze_consistency_by_difficulty(
        y_true, predictions
    )

    # 9. Case-level detail
    result["case_detail"] = build_case_detail_table(
        y_true, predictions, prompt_hashes, prompts
    )

    # 10. Pairwise comparisons
    result["pairwise"] = analyze_all_pairs(y_true, predictions)

    # 11. Omnibus statistical significance
    result["statistical_significance"] = analyze_omnibus_significance(
        y_true, predictions
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sex/gender attention patterns in ESI triage models.",
    )
    parser.add_argument(
        "--input",
        default="eval/merged_evaluations.csv",
        help="Path to merged evaluations CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/attention",
        help="Output directory (default: eval/attention)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    # ── Load and validate data ────────────────────────────────────────
    df = load_merged_data(args.input)

    pred_cols = sorted(
        [c for c in df.columns if c.startswith("predicted_score__")]
    )
    groups = discover_groups(pred_cols)

    if not groups:
        logger.error("No model groups discovered. Check column naming.")
        sys.exit(1)

    logger.info("Discovered %d model(s):", len(groups))
    for model, variants in sorted(groups.items()):
        logger.info("  %s: %s", model, sorted(variants.keys()))
        if BASELINE_VARIANT not in variants:
            logger.warning(
                "  ⚠️  Model '%s' missing baseline variant '%s'",
                model,
                BASELINE_VARIANT,
            )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Per-model analysis loop ───────────────────────────────────────
    all_results: List[Dict[str, Any]] = []

    for model in sorted(groups.keys()):
        variant_cols = groups[model]

        if len(variant_cols) < 2:
            logger.warning("Model '%s' has < 2 variants — skipping", model)
            continue

        # Run all analyses
        result = analyze_model(df, model, variant_cols)
        if not result:
            continue

        all_results.append(result)

        # Print per-model report to console
        print_model_report(result)

        # Save per-model outputs to disk
        model_dir = os.path.join(
            args.output_dir,
            model.replace("/", "_").replace("\\", "_"),
        )
        save_model_results(result, model_dir)
        generate_visuals(result, model_dir)

    # ── Cross-model summary ───────────────────────────────────────────
    if not all_results:
        logger.error("No models produced results. Exiting.")
        sys.exit(1)

    print_cross_model_summary(all_results)
    save_cross_model_results(all_results, args.output_dir)
    generate_cross_model_visuals(all_results, args.output_dir)

if __name__ == "__main__":
    main()