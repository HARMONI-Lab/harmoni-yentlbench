"""Analysis 9: Case-level detail. Analysis 10: Pairwise comparisons."""

from itertools import combinations
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import accuracy_score

from config import BASELINE_VARIANT, LABELED_VARIANTS


def build_case_detail_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    prompt_hashes: np.ndarray,
    prompts: np.ndarray,
) -> pd.DataFrame:
    variant_names = sorted(predictions.keys())
    pm = np.column_stack([predictions[v].astype(float) for v in variant_names])
    pr = np.ptp(pm, axis=1)
    pv = np.var(pm, axis=1)
    has_bl = BASELINE_VARIANT in predictions

    rows = []
    for i in range(len(y_true)):
        row = {
            "prompt_hash": prompt_hashes[i],
            "actual_score": y_true[i],
            "prediction_range": int(pr[i]),
            "prediction_variance": float(pv[i]),
        }
        for v in variant_names:
            row[f"pred_{v}"] = int(predictions[v][i])
        if has_bl:
            bl = int(predictions[BASELINE_VARIANT][i])
            for v in LABELED_VARIANTS:
                if v in predictions:
                    row[f"deviation_{v}"] = int(predictions[v][i]) - bl
        prompt_text = str(prompts[i])
        cc_end = prompt_text.find("\n")
        row["chief_complaint_snippet"] = (
            prompt_text[:cc_end] if cc_end > 0 else prompt_text[:120]
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("prediction_range", ascending=False)
    return df


def analyze_all_pairs(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows = []
    variant_names = sorted(predictions.keys())

    for v_a, v_b in combinations(variant_names, 2):
        pa = predictions[v_a]
        pb = predictions[v_b]
        diff = pb.astype(float) - pa.astype(float)

        acc_a = accuracy_score(y_true, pa)
        acc_b = accuracy_score(y_true, pb)

        ca = (pa == y_true).astype(int)
        cb = (pb == y_true).astype(int)
        b_val = int(np.sum((ca == 1) & (cb == 0)))
        c_val = int(np.sum((ca == 0) & (cb == 1)))
        n_disc = b_val + c_val

        if n_disc > 0:
            if n_disc < 25:
                mn_p = float(scipy_stats.binomtest(b_val, n_disc, 0.5).pvalue)
            else:
                chi2 = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
                mn_p = float(1 - scipy_stats.chi2.cdf(chi2, 1))
        else:
            mn_p = 1.0

        cohens_h = float(
            2 * np.arcsin(np.sqrt(acc_a)) - 2 * np.arcsin(np.sqrt(acc_b))
        )

        # Cramer's V for predictions
        ct = pd.crosstab(pa, pb)
        if ct.size > 0:
            chi2_ct, _, _, _ = scipy_stats.chi2_contingency(ct)
            n_val = len(pa)
            k_val = min(ct.shape) - 1
            cramers_v = float(np.sqrt(chi2_ct / (n_val * k_val))) if k_val > 0 and n_val > 0 else 0.0
        else:
            cramers_v = 0.0

        rows.append({
            "variant_a": v_a,
            "variant_b": v_b,
            "agreement_rate": float(np.mean(pa == pb)),
            "mean_signed_diff": float(np.mean(diff)),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "pct_differ": float(np.mean(diff != 0)),
            "accuracy_a": acc_a,
            "accuracy_b": acc_b,
            "accuracy_delta": acc_b - acc_a,
            "mcnemar_p": mn_p,
            "n_discordant": n_disc,
            "cohens_h": cohens_h,
            "cramers_v": cramers_v,
        })

    return rows