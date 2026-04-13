"""Analysis 1: Baseline Deviation. Analysis 2: Effect Decomposition."""

import logging
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
)

from config import (
    ESI_LEVELS,
    BASELINE_VARIANT,
    LABELED_VARIANTS,
    BINARY_VARIANTS,
    ALL_VARIANTS,
    VARIANT_NO_SEX,
    VARIANT_FEMALE,
    VARIANT_MALE,
    VARIANT_NONBINARY,
)

logger = logging.getLogger(__name__)


def analyze_baseline_deviation(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    if BASELINE_VARIANT not in predictions:
        logger.warning("Baseline variant '%s' not found", BASELINE_VARIANT)
        return stats

    baseline = predictions[BASELINE_VARIANT]
    baseline_acc = accuracy_score(y_true, baseline)
    stats["baseline_accuracy"] = baseline_acc
    stats["baseline_balanced_accuracy"] = balanced_accuracy_score(y_true, baseline)
    stats["baseline_mae"] = float(mean_absolute_error(y_true, baseline))

    for variant in LABELED_VARIANTS:
        if variant not in predictions:
            continue

        pred = predictions[variant]
        vl = variant
        deviation = pred.astype(float) - baseline.astype(float)

        stats[f"deviation_rate__{vl}"] = float(np.mean(deviation != 0))
        stats[f"mean_deviation__{vl}"] = float(np.mean(deviation))
        stats[f"mean_abs_deviation__{vl}"] = float(np.mean(np.abs(deviation)))
        stats[f"std_deviation__{vl}"] = float(np.std(deviation))
        stats[f"max_abs_deviation__{vl}"] = int(np.max(np.abs(deviation)))

        stats[f"pct_more_urgent_than_baseline__{vl}"] = float(np.mean(deviation < 0))
        stats[f"pct_same_as_baseline__{vl}"] = float(np.mean(deviation == 0))
        stats[f"pct_less_urgent_than_baseline__{vl}"] = float(np.mean(deviation > 0))

        variant_acc = accuracy_score(y_true, pred)
        stats[f"accuracy__{vl}"] = variant_acc
        stats[f"accuracy_delta_from_baseline__{vl}"] = variant_acc - baseline_acc

        baseline_correct = baseline == y_true
        variant_correct = pred == y_true
        helped = int(np.sum(~baseline_correct & variant_correct))
        hurt = int(np.sum(baseline_correct & ~variant_correct))

        stats[f"sex_info_helped__{vl}"] = helped
        stats[f"sex_info_hurt__{vl}"] = hurt
        stats[f"sex_info_net_effect__{vl}"] = helped - hurt
        stats[f"both_correct__{vl}"] = int(np.sum(baseline_correct & variant_correct))
        stats[f"both_wrong__{vl}"] = int(np.sum(~baseline_correct & ~variant_correct))

        nonzero = deviation[deviation != 0]
        if len(nonzero) >= 10:
            try:
                _, w_p = scipy_stats.wilcoxon(nonzero)
                stats[f"wilcoxon_deviation_p__{vl}"] = float(w_p)
            except Exception:
                stats[f"wilcoxon_deviation_p__{vl}"] = None
        else:
            stats[f"wilcoxon_deviation_p__{vl}"] = None

        n_pos = int(np.sum(deviation > 0))
        n_neg = int(np.sum(deviation < 0))
        n_changed = n_pos + n_neg
        if n_changed > 0:
            stats[f"sign_test_p__{vl}"] = float(
                scipy_stats.binomtest(n_pos, n_changed, 0.5).pvalue
            )
            stats[f"direction_ratio__{vl}"] = n_pos / n_changed
        else:
            stats[f"sign_test_p__{vl}"] = 1.0
            stats[f"direction_ratio__{vl}"] = 0.5

        for level in ESI_LEVELS:
            mask = y_true == level
            if mask.sum() > 0:
                ld = deviation[mask]
                stats[f"deviation_rate_esi{level}__{vl}"] = float(np.mean(ld != 0))
                stats[f"mean_deviation_esi{level}__{vl}"] = float(np.mean(ld))

    return stats


def decompose_sex_info_effect(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    has = {v: v in predictions for v in ALL_VARIANTS}

    if not has[VARIANT_NO_SEX]:
        return stats

    baseline = predictions[VARIANT_NO_SEX].astype(float)
    baseline_acc = accuracy_score(y_true, predictions[VARIANT_NO_SEX])

    # LAYER 1: Presence
    available_labeled = [v for v in LABELED_VARIANTS if has[v]]
    if available_labeled:
        labeled_mean = np.mean(
            [predictions[v].astype(float) for v in available_labeled], axis=0
        )
        presence_effect = labeled_mean - baseline
        stats["L1_presence_mean_effect"] = float(np.mean(presence_effect))
        stats["L1_presence_abs_mean_effect"] = float(np.mean(np.abs(presence_effect)))
        stats["L1_presence_pct_changed"] = float(np.mean(np.abs(presence_effect) > 0.01))

        nonzero = presence_effect[np.abs(presence_effect) > 0.01]
        if len(nonzero) >= 10:
            try:
                _, p = scipy_stats.wilcoxon(nonzero)
                stats["L1_wilcoxon_p"] = float(p)
            except Exception:
                stats["L1_wilcoxon_p"] = None
        else:
            stats["L1_wilcoxon_p"] = None

        labeled_accs = [accuracy_score(y_true, predictions[v]) for v in available_labeled]
        stats["L1_accuracy_with_sex_info"] = float(np.mean(labeled_accs))
        stats["L1_accuracy_without_sex_info"] = baseline_acc
        stats["L1_accuracy_delta"] = float(np.mean(labeled_accs) - baseline_acc)

    # LAYER 2: Binary vs non-binary
    available_binary = [v for v in BINARY_VARIANTS if has[v]]
    if available_binary and has[VARIANT_NONBINARY]:
        binary_mean = np.mean(
            [predictions[v].astype(float) for v in available_binary], axis=0
        )
        nb_pred = predictions[VARIANT_NONBINARY].astype(float)
        cat_effect = binary_mean - nb_pred
        stats["L2_binary_vs_nb_mean_effect"] = float(np.mean(cat_effect))
        stats["L2_binary_vs_nb_abs_mean"] = float(np.mean(np.abs(cat_effect)))
        stats["L2_binary_vs_nb_pct_differs"] = float(np.mean(np.abs(cat_effect) > 0.01))

        nonzero = cat_effect[np.abs(cat_effect) > 0.01]
        if len(nonzero) >= 10:
            try:
                _, p = scipy_stats.wilcoxon(nonzero)
                stats["L2_wilcoxon_p"] = float(p)
            except Exception:
                stats["L2_wilcoxon_p"] = None
        else:
            stats["L2_wilcoxon_p"] = None

        binary_acc = np.mean([accuracy_score(y_true, predictions[v]) for v in available_binary])
        nb_acc = accuracy_score(y_true, predictions[VARIANT_NONBINARY])
        stats["L2_accuracy_binary"] = float(binary_acc)
        stats["L2_accuracy_nonbinary"] = float(nb_acc)
        stats["L2_accuracy_delta"] = float(binary_acc - nb_acc)

    # LAYER 3: Female vs Male
    if has[VARIANT_FEMALE] and has[VARIANT_MALE]:
        f_pred = predictions[VARIANT_FEMALE].astype(float)
        m_pred = predictions[VARIANT_MALE].astype(float)
        gender_diff = f_pred - m_pred

        stats["L3_female_vs_male_mean"] = float(np.mean(gender_diff))
        stats["L3_female_vs_male_abs_mean"] = float(np.mean(np.abs(gender_diff)))
        stats["L3_female_vs_male_pct_differs"] = float(np.mean(gender_diff != 0))
        stats["L3_pct_female_less_urgent"] = float(np.mean(gender_diff > 0))
        stats["L3_pct_female_more_urgent"] = float(np.mean(gender_diff < 0))

        nonzero = gender_diff[gender_diff != 0]
        if len(nonzero) >= 10:
            try:
                _, p = scipy_stats.wilcoxon(nonzero)
                stats["L3_wilcoxon_p"] = float(p)
            except Exception:
                stats["L3_wilcoxon_p"] = None
        else:
            stats["L3_wilcoxon_p"] = None

        f_acc = accuracy_score(y_true, predictions[VARIANT_FEMALE])
        m_acc = accuracy_score(y_true, predictions[VARIANT_MALE])
        stats["L3_accuracy_female"] = float(f_acc)
        stats["L3_accuracy_male"] = float(m_acc)
        stats["L3_accuracy_delta"] = float(f_acc - m_acc)

        for level in ESI_LEVELS:
            mask = y_true == level
            if mask.sum() > 0:
                ld = gender_diff[mask]
                stats[f"L3_female_vs_male_esi{level}_mean"] = float(np.mean(ld))
                stats[f"L3_female_vs_male_esi{level}_pct_diff"] = float(np.mean(ld != 0))

    # LAYER 4: Non-binary token
    if has[VARIANT_NONBINARY]:
        nb_pred = predictions[VARIANT_NONBINARY].astype(float)
        nb_effect = nb_pred - baseline

        stats["L4_nb_token_mean_effect"] = float(np.mean(nb_effect))
        stats["L4_nb_token_abs_mean"] = float(np.mean(np.abs(nb_effect)))
        stats["L4_nb_token_pct_changed"] = float(np.mean(nb_effect != 0))

        nonzero = nb_effect[nb_effect != 0]
        if len(nonzero) >= 10:
            try:
                _, p = scipy_stats.wilcoxon(nonzero)
                stats["L4_wilcoxon_p"] = float(p)
            except Exception:
                stats["L4_wilcoxon_p"] = None
        else:
            stats["L4_wilcoxon_p"] = None

        nb_acc = accuracy_score(y_true, predictions[VARIANT_NONBINARY])
        stats["L4_accuracy_nonbinary"] = float(nb_acc)
        stats["L4_accuracy_no_sex"] = baseline_acc
        stats["L4_accuracy_delta"] = float(nb_acc - baseline_acc)

        baseline_correct = predictions[VARIANT_NO_SEX] == y_true
        nb_correct = predictions[VARIANT_NONBINARY] == y_true
        stats["L4_nb_token_helped"] = int(np.sum(~baseline_correct & nb_correct))
        stats["L4_nb_token_hurt"] = int(np.sum(baseline_correct & ~nb_correct))

    # Effect magnitude comparison
    magnitudes = {}
    if "L1_presence_abs_mean_effect" in stats:
        magnitudes["presence"] = stats["L1_presence_abs_mean_effect"]
    if "L2_binary_vs_nb_abs_mean" in stats:
        magnitudes["binary_vs_nb"] = stats["L2_binary_vs_nb_abs_mean"]
    if "L3_female_vs_male_abs_mean" in stats:
        magnitudes["female_vs_male"] = stats["L3_female_vs_male_abs_mean"]
    if "L4_nb_token_abs_mean" in stats:
        magnitudes["nb_token"] = stats["L4_nb_token_abs_mean"]

    if magnitudes:
        for name, mag in magnitudes.items():
            stats[f"effect_magnitude__{name}"] = mag

    return stats

def compute_transition_matrices(
    predictions: Dict[str, np.ndarray],
) -> Dict[str, pd.DataFrame]:
    matrices = {}
    if BASELINE_VARIANT not in predictions:
        return matrices

    baseline = predictions[BASELINE_VARIANT]

    for variant in LABELED_VARIANTS:
        if variant not in predictions:
            continue
        pred = predictions[variant]
        matrix = pd.DataFrame(0, index=ESI_LEVELS, columns=ESI_LEVELS, dtype=int)
        matrix.index.name = "baseline_pred"
        matrix.columns.name = f"{variant}_pred"
        for b, v in zip(baseline, pred):
            matrix.loc[b, v] += 1
        matrices[variant] = matrix

    return matrices


def analyze_transition_risk(
    matrices: Dict[str, pd.DataFrame],
) -> List[Dict[str, Any]]:
    transitions = []
    for variant, matrix in matrices.items():
        total = matrix.values.sum()
        if total == 0:
            continue
        for from_esi in ESI_LEVELS:
            for to_esi in ESI_LEVELS:
                if from_esi == to_esi:
                    continue
                count = matrix.loc[from_esi, to_esi]
                if count == 0:
                    continue
                shift = to_esi - from_esi
                row_total = matrix.loc[from_esi].sum()
                abs_shift = abs(shift)

                transitions.append({
                    "sex_label": variant,
                    "baseline_esi": from_esi,
                    "shifted_esi": to_esi,
                    "shift": shift,
                    "count": count,
                    "pct_of_baseline_esi": float(count / row_total) if row_total > 0 else 0.0,
                    "pct_of_total": float(count / total),
                })
    return transitions