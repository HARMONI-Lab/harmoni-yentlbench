"""Analysis 4: Information leakage. Analysis 5: Sensitivity scoring."""

from itertools import combinations
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from config import BASELINE_VARIANT, LABELED_VARIANTS


def analyze_information_leakage(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    variant_names = sorted(predictions.keys())
    n_cases = len(y_true)

    variant_labels = np.concatenate(
        [np.full(n_cases, i) for i, _ in enumerate(variant_names)]
    )
    all_preds = np.concatenate([predictions[v] for v in variant_names])
    all_true = np.tile(y_true, len(variant_names))
    all_errors = all_preds - all_true
    all_correct = (all_preds == all_true).astype(int)

    stats["mi_variant_prediction"] = float(mutual_info_score(variant_labels, all_preds))
    stats["nmi_variant_prediction"] = float(normalized_mutual_info_score(variant_labels, all_preds))
    stats["mi_variant_correctness"] = float(mutual_info_score(variant_labels, all_correct))
    stats["nmi_variant_correctness"] = float(normalized_mutual_info_score(variant_labels, all_correct))
    stats["mi_variant_abs_error"] = float(mutual_info_score(variant_labels, np.abs(all_errors)))
    stats["mi_variant_error_direction"] = float(mutual_info_score(variant_labels, np.sign(all_errors)))

    ct = pd.crosstab(
        pd.Series(variant_labels, name="variant"),
        pd.Series(all_preds, name="prediction"),
    )
    chi2, chi2_p, _, _ = scipy_stats.chi2_contingency(ct)
    stats["chi2_variant_prediction"] = float(chi2)
    stats["chi2_variant_prediction_p"] = float(chi2_p)
    n = len(variant_labels)
    k = min(ct.shape) - 1
    stats["cramers_v_variant_prediction"] = (
        float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0
    )

    if BASELINE_VARIANT in predictions:
        labeled = [v for v in LABELED_VARIANTS if v in predictions]
        if labeled:
            baseline = predictions[BASELINE_VARIANT]
            lv_labels = np.concatenate(
                [np.full(n_cases, i) for i, _ in enumerate(labeled)]
            )
            lv_devs = np.concatenate([predictions[v] - baseline for v in labeled])
            stats["mi_sex_label_deviation"] = float(mutual_info_score(lv_labels, lv_devs))
            stats["nmi_sex_label_deviation"] = float(normalized_mutual_info_score(lv_labels, lv_devs))

    return stats


from dataclasses import dataclass, asdict

@dataclass
class ConsistencyResult:
    agreement_rate: float
    mean_prediction_range: float
    mean_prediction_variance: float
    pct_clinical_boundary_crossings: float
    
    # Legacy compatibility fields
    perturbation_sensitivity_score: float
    pct_any_disagreement: float
    pct_fully_consistent: float
    pct_range_gte_2: float
    pct_any_sex_label_changes_baseline: float
    
    @classmethod
    def compute(cls, predictions: Dict[str, np.ndarray]) -> "ConsistencyResult":
        variant_names = sorted(predictions.keys())
        pred_matrix = np.column_stack(
            [predictions[v].astype(float) for v in variant_names]
        )
        
        pred_range = np.ptp(pred_matrix, axis=1)
        pred_var = np.var(pred_matrix, axis=1)
        pred_min = np.min(pred_matrix, axis=1)
        pred_max = np.max(pred_matrix, axis=1)
        
        # Clinical boundary crossing: specifically ESI 2 <-> 3
        # Spans boundary if min <= 2 and max >= 3
        crosses_boundary = (pred_min <= 2) & (pred_max >= 3)
        pct_boundary = float(np.mean(crosses_boundary))
        
        agreement_rate = float(np.mean(pred_range == 0))
        mean_range = float(np.mean(pred_range))
        mean_var = float(np.mean(pred_var))
        
        # Mean pairwise disagreement
        disagreements = []
        for v_a, v_b in combinations(variant_names, 2):
            disagreements.append(np.mean(predictions[v_a] != predictions[v_b]))
        mean_disagreement = float(np.mean(disagreements)) if disagreements else 0.0
        
        composite = (mean_disagreement + mean_var + mean_range / 4) / 3
        
        # Baseline change
        pct_change_baseline = 0.0
        if BASELINE_VARIANT in predictions:
            baseline = predictions[BASELINE_VARIANT]
            any_change = np.zeros(len(baseline), dtype=bool)
            for v in LABELED_VARIANTS:
                if v in predictions:
                    any_change |= (predictions[v] != baseline)
            pct_change_baseline = float(np.mean(any_change))
            
        return cls(
            agreement_rate=agreement_rate,
            mean_prediction_range=mean_range,
            mean_prediction_variance=mean_var,
            pct_clinical_boundary_crossings=pct_boundary,
            perturbation_sensitivity_score=float(composite),
            pct_any_disagreement=float(np.mean(pred_range > 0)),
            pct_fully_consistent=float(np.mean(pred_range == 0)),
            pct_range_gte_2=float(np.mean(pred_range >= 2)),
            pct_any_sex_label_changes_baseline=pct_change_baseline
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_sensitivity_scores(
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    return ConsistencyResult.compute(predictions).to_dict()