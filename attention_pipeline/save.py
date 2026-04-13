"""All file output / saving functions."""

import os
import logging
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_model_results(
    result: Dict[str, Any],
    model_dir: str,
) -> None:
    """Save all per-model output files."""
    os.makedirs(model_dir, exist_ok=True)

    # Transition matrices
    for variant, matrix in result.get("transition_matrices", {}).items():
        matrix.to_csv(os.path.join(model_dir, f"transition_{variant}.csv"))

    # Dangerous transitions
    dangerous = result.get("dangerous_transitions", [])
    if dangerous:
        pd.DataFrame(dangerous).to_csv(
            os.path.join(model_dir, "dangerous_transitions.csv"), index=False
        )

    # Vulnerability by ESI
    vuln = result.get("vulnerability_by_esi", pd.DataFrame())
    if not vuln.empty:
        vuln.to_csv(os.path.join(model_dir, "vulnerability_by_esi.csv"), index=False)

    # Vulnerability by clinical category
    cat = result.get("vulnerability_by_category", pd.DataFrame())
    if not cat.empty:
        cat.to_csv(os.path.join(model_dir, "vulnerability_by_category.csv"), index=False)

    # Boundary analysis
    boundary = result.get("boundary_analysis", pd.DataFrame())
    if not boundary.empty:
        boundary.to_csv(os.path.join(model_dir, "boundary_crossings.csv"), index=False)

    # Consistency by difficulty
    diff = result.get("consistency_by_difficulty", pd.DataFrame())
    if not diff.empty:
        diff.to_csv(os.path.join(model_dir, "consistency_by_difficulty.csv"), index=False)

    # Pairwise comparisons
    pairwise_data = result.get("pairwise", [])
    pairwise = pd.DataFrame(pairwise_data) if isinstance(pairwise_data, list) else pairwise_data
    if not pairwise.empty:
        pairwise.to_csv(os.path.join(model_dir, "pairwise_comparisons.csv"), index=False)

    # Case-level detail
    case_df = result.get("case_detail", pd.DataFrame())
    if not case_df.empty:
        case_df.to_csv(os.path.join(model_dir, "case_detail_all.csv"), index=False)
        disagree = case_df[case_df["prediction_range"] > 0]
        if not disagree.empty:
            disagree.to_csv(
                os.path.join(model_dir, "case_detail_disagreements.csv"), index=False
            )

    # Flat model summary
    model_summary = {"model": result["model"], "n_cases": result["n_cases"]}
    model_summary.update(result.get("baseline_deviation", {}))
    model_summary.update(
        {f"decomp_{k}": v for k, v in result.get("effect_decomposition", {}).items()}
    )
    model_summary.update(
        {f"sens_{k}": v for k, v in result.get("sensitivity", {}).items()}
    )
    model_summary.update(
        {f"info_{k}": v for k, v in result.get("information_leakage", {}).items()}
    )
    model_summary.update(
        {f"stat_{k}": v for k, v in result.get("statistical_significance", {}).items() if not k.startswith("error")}
    )
    pd.DataFrame([model_summary]).to_csv(
        os.path.join(model_dir, "model_attention_summary.csv"), index=False
    )

    logger.info("  Saved outputs to '%s'", model_dir)


def save_cross_model_results(
    all_results: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Save cross-model summary and combined tables."""
    from report import build_cross_model_summary

    # Cross-model summary
    summary_df = build_cross_model_summary(all_results)
    summary_path = os.path.join(output_dir, "cross_model_attention_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved cross-model summary to '{summary_path}'")

    # Combined pairwise table
    all_pairwise = []
    for result in all_results:
        pw_data = result.get("pairwise", [])
        pw = pd.DataFrame(pw_data) if isinstance(pw_data, list) else pw_data
        if not pw.empty:
            pw = pw.copy()
            pw["model"] = result["model"]
            all_pairwise.append(pw)
    if all_pairwise:
        combined_pw = pd.concat(all_pairwise, ignore_index=True)
        pw_path = os.path.join(output_dir, "all_models_pairwise.csv")
        combined_pw.to_csv(pw_path, index=False)
        print(f"Saved combined pairwise comparisons to '{pw_path}'")

    # Combined dangerous transitions
    all_dangerous = []
    for result in all_results:
        for t in result.get("dangerous_transitions", []):
            t_copy = dict(t)
            t_copy["model"] = result["model"]
            all_dangerous.append(t_copy)
    if all_dangerous:
        dt_df = pd.DataFrame(all_dangerous)
        dt_path = os.path.join(output_dir, "all_models_dangerous_transitions.csv")
        dt_df.to_csv(dt_path, index=False)
        print(f"Saved combined dangerous transitions to '{dt_path}'")

    print(f"\nAll attention analysis outputs saved to '{output_dir}/'")