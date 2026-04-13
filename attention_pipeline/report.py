"""All console reporting / printing functions."""

from typing import List, Dict, Any

import numpy as np
import pandas as pd

from config import LABELED_VARIANTS, VARIANT_DESCRIPTIONS


def _get_risk_level(from_esi: int, shift: int) -> str:
    abs_shift = abs(shift)
    if abs_shift >= 3:
        return "CRITICAL"
    elif abs_shift == 2:
        return "HIGH"
    elif abs_shift == 1 and from_esi <= 2 and shift > 0:
        return "HIGH"
    elif abs_shift == 1:
        return "MODERATE"
    return "LOW"

def _get_dominant_effect(ed: Dict[str, Any]):
    mags = {}
    if "L1_presence_abs_mean_effect" in ed:
        mags["presence"] = ed["L1_presence_abs_mean_effect"]
    if "L2_binary_vs_nb_abs_mean" in ed:
        mags["binary_vs_nb"] = ed["L2_binary_vs_nb_abs_mean"]
    if "L3_female_vs_male_abs_mean" in ed:
        mags["female_vs_male"] = ed["L3_female_vs_male_abs_mean"]
    if "L4_nb_token_abs_mean" in ed:
        mags["nb_token"] = ed["L4_nb_token_abs_mean"]
    
    if mags:
        dominant = max(mags, key=mags.get)
        return dominant, mags[dominant]
    return None, 0.0

def print_model_report(result: Dict[str, Any]) -> None:
    model = result.get("model", "unknown")
    n = result.get("n_cases", 0)

    print(f"\n{'='*72}")
    print(f"  ATTENTION ANALYSIS: {model}")
    print(f"{'='*72}")
    print(f"  Cases analyzed: {n}")

    # Sensitivity
    sens = result.get("sensitivity", {})
    print(f"\n  📊 Perturbation Sensitivity Score: {sens.get('perturbation_sensitivity_score', 0):.4f}")
    print(f"     Cases fully consistent:           {sens.get('pct_fully_consistent', 0):.2%}")
    print(f"     Cases with any disagreement:      {sens.get('pct_any_disagreement', 0):.2%}")
    print(f"     Cases with range ≥ 2 ESI levels:  {sens.get('pct_range_gte_2', 0):.2%}")
    print(f"     Clinical boundary crossing (2↔3): {sens.get('pct_clinical_boundary_crossings', 0):.2%}")
    print(f"     Cases where sex changes baseline: {sens.get('pct_any_sex_label_changes_baseline', 0):.2%}")

    # Baseline deviation
    bd = result.get("baseline_deviation", {})
    if bd:
        print(f"\n  🎯 Baseline Deviation (vs no-sex-info baseline)")
        print(f"     Baseline (no sex) accuracy:       {bd.get('baseline_accuracy', 0):.4f}")
        for v in LABELED_VARIANTS:
            dr = bd.get(f"deviation_rate__{v}")
            if dr is None:
                continue
            md = bd.get(f"mean_deviation__{v}", 0)
            acc = bd.get(f"accuracy__{v}", 0)
            ad = bd.get(f"accuracy_delta_from_baseline__{v}", 0)
            helped = bd.get(f"sex_info_helped__{v}", 0)
            hurt = bd.get(f"sex_info_hurt__{v}", 0)
            wp = bd.get(f"wilcoxon_deviation_p__{v}")
            desc = VARIANT_DESCRIPTIONS.get(v, v)
            sig = ""
            if wp is not None and wp < 0.05:
                sig = " ⚠️ SIG"
            elif wp is not None and wp < 0.1:
                sig = " (marginal)"
            print(f"\n     {desc}:")
            print(f"       Predictions changed:     {dr:.2%}")
            print(f"       Mean deviation:          {md:+.4f} (+ = less urgent){sig}")
            print(f"       Accuracy:                {acc:.4f} (Δ = {ad:+.4f} from baseline)")
            print(f"       Sex info helped/hurt:    +{helped} / -{hurt} (net: {helped - hurt:+d})")

    # Effect decomposition
    ed = result.get("effect_decomposition", {})
    if ed:
        print(f"\n  🔬 Sex Information Effect Decomposition")

        l1 = ed.get("L1_presence_abs_mean_effect")
        if l1 is not None:
            helpful = "YES ✓" if ed.get("L1_accuracy_delta", 0) > 0 else "NO ✗"
            print(f"     Layer 1 — PRESENCE (any sex label vs none):")
            print(f"       Mean effect magnitude:   {l1:.4f}")
            print(f"       Accuracy Δ:              {ed.get('L1_accuracy_delta', 0):+.4f}")
            print(f"       Sex info helpful?        {helpful}")

        l2 = ed.get("L2_binary_vs_nb_abs_mean")
        if l2 is not None:
            print(f"     Layer 2 — CATEGORY (binary M/F vs non-binary):")
            print(f"       Mean effect magnitude:   {l2:.4f}")
            print(f"       Accuracy Δ:              {ed.get('L2_accuracy_delta', 0):+.4f}")

        l3 = ed.get("L3_female_vs_male_abs_mean")
        if l3 is not None:
            sig = " ⚠️" if (ed.get("L3_wilcoxon_p") or 1) < 0.05 else ""
            print(f"     Layer 3 — GENDER VALUE (female vs male):{sig}")
            print(f"       Mean effect magnitude:   {l3:.4f}")
            print(f"       Predictions differ:      {ed.get('L3_female_vs_male_pct_differs', 0):.2%}")
            print(f"       Female less urgent:      {ed.get('L3_pct_female_less_urgent', 0):.2%}")
            print(f"       Female more urgent:      {ed.get('L3_pct_female_more_urgent', 0):.2%}")
            print(f"       Accuracy Δ (F - M):      {ed.get('L3_accuracy_delta', 0):+.4f}")

        l4 = ed.get("L4_nb_token_abs_mean")
        if l4 is not None:
            ignores = (
                "YES → treats as no info"
                if ed.get("L4_nb_token_pct_changed", 1.0) < 0.02
                else "NO → has NB-specific behavior"
            )
            print(f"     Layer 4 — NON-BINARY TOKEN (NB vs no sex info):")
            print(f"       Mean effect magnitude:   {l4:.4f}")
            print(f"       Predictions changed:     {ed.get('L4_nb_token_pct_changed', 0):.2%}")
            print(f"       Model ignores NB token?  {ignores}")

        dominant, dominant_mag = _get_dominant_effect(ed)
        if dominant:
            print(f"\n     Dominant effect: {dominant} "
                  f"(magnitude: {dominant_mag:.4f})")

    # Information leakage
    info = result.get("information_leakage", {})
    if info:
        nmi = info.get("nmi_variant_prediction", 0)
        chi2_p = info.get("chi2_variant_prediction_p", 1)
        cv = info.get("cramers_v_variant_prediction", 0)
        sig = " ⚠️" if chi2_p < 0.05 else ""
        print(f"\n  🔍 Information Leakage")
        print(f"     NMI(sex variant, prediction):  {nmi:.6f}")
        print(f"     χ² test p-value:               {chi2_p:.6f}{sig}")
        print(f"     Cramér's V:                    {cv:.6f}")
        nmi_dev = info.get("nmi_sex_label_deviation")
        if nmi_dev is not None:
            print(f"     NMI(sex label, deviation):     {nmi_dev:.6f}")

    # Statistical significance
    stat = result.get("statistical_significance", {})
    if stat and "error" not in stat:
        print(f"\n  ⚖️  Omnibus Statistical Significance (Fairness)")
        print(f"     Variants tested:               {stat.get('variants_tested', 0)}")
        print(f"     Cochran's Q (Accuracy):        p = {stat.get('cochran_q_p', 1):.6f} (FDR adj p = {stat.get('cochran_q_fdr_p', 1):.6f}, stat={stat.get('cochran_q_stat', 0):.2f})")
        print(f"     Friedman Test (Scores):        p = {stat.get('friedman_p', 1):.6f} (FDR adj p = {stat.get('friedman_fdr_p', 1):.6f}, stat={stat.get('friedman_stat', 0):.2f})")
        if stat.get('cochran_q_fdr_p', 1) < 0.05 or stat.get('friedman_fdr_p', 1) < 0.05:
            print("     ⚠️  SIGNIFICANT difference found across variants!")

    # Vulnerability by ESI
    vuln = result.get("vulnerability_by_esi", pd.DataFrame())
    if not vuln.empty:
        print(f"\n  🎯 Vulnerability by ESI Level")
        for _, row in vuln.iterrows():
            print(f"     ESI {int(row['esi_level'])}: "
                  f"{row['pct_any_disagreement']:.1%} disagree, "
                  f"acc range = {row['accuracy_range']:.4f}")

    # Boundary crossings
    boundary = result.get("boundary_analysis", pd.DataFrame())
    if not boundary.empty:
        active = boundary[boundary["total_crosses"] > 0]
        if not active.empty:
            print(f"\n  🚧 Decision Boundary Crossings (baseline → sex-labeled)")
            for _, row in active.iterrows():
                b_str = f"ESI {row['lower_esi']}↔{row['upper_esi']}"
                print(f"     {b_str} [{row['sex_label']}]: "
                      f"{row['total_crosses']} crossings "
                      f"({row['cross_rate']:.1%} of near-boundary)")

    # Dangerous transitions
    dangerous = result.get("dangerous_transitions", [])
    high_risk = []
    for t in dangerous:
        risk = _get_risk_level(t["baseline_esi"], t["shift"])
        if risk in ("CRITICAL", "HIGH"):
            t_mapped = dict(t)
            t_mapped["risk_level"] = risk
            t_mapped["direction"] = "under-triage" if t["shift"] > 0 else "over-triage"
            high_risk.append(t_mapped)
            
    if high_risk:
        print(f"\n  ⚠️  High-Risk Sex-Induced Transitions")
        for t in sorted(high_risk, key=lambda x: -x["count"]):
            print(f"     [{t['risk_level']:>8s}] {t['sex_label']}: "
                  f"ESI {t['baseline_esi']}→{t['shifted_esi']} "
                  f"({t['direction']}) × {t['count']}")

    # Clinical category
    cat = result.get("vulnerability_by_category", pd.DataFrame())
    if not cat.empty:
        print(f"\n  🏥 Vulnerability by Clinical Category (top 10)")
        for _, row in cat.head(10).iterrows():
            print(f"     {row['clinical_category']:>20s}: "
                  f"n={int(row['n_cases']):>3d}, "
                  f"disagree={row['pct_any_disagreement']:.1%}, "
                  f"acc range={row['accuracy_range']:.4f}")

    # Difficulty
    diff = result.get("consistency_by_difficulty", pd.DataFrame())
    if not diff.empty:
        print(f"\n  📈 Sex Sensitivity by Case Difficulty (baseline error)")
        for _, row in diff.iterrows():
            err = row['baseline_error']
            diff_label = "baseline_correct" if err == 0 else f"baseline_off_by_{int(err)}"
            print(f"     {diff_label:>25s}: "
                  f"n={int(row['n_cases']):>4d}, "
                  f"disagree={row['pct_any_disagreement']:.1%}")


def build_cross_model_summary(
    all_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for result in all_results:
        if not result:
            continue
        sens = result.get("sensitivity", {})
        info = result.get("information_leakage", {})
        bd = result.get("baseline_deviation", {})
        ed = result.get("effect_decomposition", {})
        stat = result.get("statistical_significance", {})

        row = {
            "model": result["model"],
            "n_cases": result["n_cases"],
            "sensitivity_score": sens.get("perturbation_sensitivity_score"),
            "pct_any_disagreement": sens.get("pct_any_disagreement"),
            "pct_sex_changes_baseline": sens.get("pct_any_sex_label_changes_baseline"),
            "pct_range_gte_2": sens.get("pct_range_gte_2"),
            "baseline_accuracy": bd.get("baseline_accuracy"),
            "L1_presence_effect": ed.get("L1_presence_abs_mean_effect"),
            "L1_accuracy_delta": ed.get("L1_accuracy_delta"),
            "L3_female_vs_male_effect": ed.get("L3_female_vs_male_abs_mean"),
            "L3_female_vs_male_pct_diff": ed.get("L3_female_vs_male_pct_differs"),
            "L3_accuracy_delta_f_vs_m": ed.get("L3_accuracy_delta"),
            "L3_wilcoxon_p": ed.get("L3_wilcoxon_p"),
            "L4_nb_token_pct_changed": ed.get("L4_nb_token_pct_changed"),
            "L4_model_ignores_nb": ed.get("L4_nb_token_pct_changed", 1.0) < 0.02,
            "dominant_effect": _get_dominant_effect(ed)[0],
            "nmi_variant_prediction": info.get("nmi_variant_prediction"),
            "chi2_p": info.get("chi2_variant_prediction_p"),
            "cramers_v": info.get("cramers_v_variant_prediction"),
            "cochran_q_p": stat.get("cochran_q_p"),
            "cochran_q_fdr_p": stat.get("cochran_q_fdr_p"),
            "friedman_p": stat.get("friedman_p"),
            "friedman_fdr_p": stat.get("friedman_fdr_p"),
        }
        for v in LABELED_VARIANTS:
            row[f"accuracy_{v}"] = bd.get(f"accuracy__{v}")
            row[f"deviation_rate_{v}"] = bd.get(f"deviation_rate__{v}")
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("sensitivity_score")
    return df


def print_cross_model_summary(all_results: List[Dict[str, Any]]) -> None:
    summary_df = build_cross_model_summary(all_results)
    print(f"\n{'='*72}")
    print("  CROSS-MODEL ATTENTION RANKING")
    print(f"{'='*72}")
    print("  (Lower sensitivity score = more sex-invariant = better)\n")

    cols = [
        "model", "sensitivity_score", "pct_sex_changes_baseline",
        "baseline_accuracy", "L3_female_vs_male_pct_diff",
        "L3_accuracy_delta_f_vs_m", "cochran_q_p", "dominant_effect",
    ]
    available = [c for c in cols if c in summary_df.columns]
    formatted = summary_df[available].copy()
    for col in formatted.columns:
        if col in ("model", "dominant_effect"):
            continue
        if formatted[col].dtype in (np.float64, np.float32):
            formatted[col] = formatted[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
    print(formatted.to_string(index=False))

    print("\n  🏆 Models ranked by sex-invariance (best first):")
    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        sig = ""
        if row.get("L3_wilcoxon_p") is not None and row["L3_wilcoxon_p"] < 0.05:
            sig = " ⚠️ gender bias detected"
        elif row.get("cochran_q_p") is not None and row["cochran_q_p"] < 0.05:
            sig = " ⚠️ sig variant diff"
        print(f"    {i}. {row['model']}: "
              f"sensitivity={row.get('sensitivity_score', 0):.4f}, "
              f"baseline_acc={row.get('baseline_accuracy', 0):.4f}{sig}")