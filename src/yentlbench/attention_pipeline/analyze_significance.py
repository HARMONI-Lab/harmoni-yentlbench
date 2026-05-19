"""Analysis 11: Omnibus Statistical Significance."""

from typing import Dict, Any

import numpy as np
from scipy import stats as scipy_stats


def _cochrans_q(data: np.ndarray):
    """
    Computes Cochran's Q test for a binary matrix (cases x variants).
    """
    data = np.asarray(data)
    k = data.shape[1]
    
    col_sums = data.sum(axis=0)
    row_sums = data.sum(axis=1)
    
    numerator = (k - 1) * (k * np.sum(col_sums**2) - (np.sum(col_sums))**2)
    denominator = k * np.sum(row_sums) - np.sum(row_sums**2)
    
    if denominator == 0:
        return 0.0, 1.0
        
    q_stat = numerator / denominator
    p_value = scipy_stats.chi2.sf(q_stat, df=k-1)
    
    return float(q_stat), float(p_value)


def analyze_omnibus_significance(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Computes omnibus statistical tests across all variants.
    - Cochran's Q: for binary outcome (correct/incorrect)
    - Friedman Test: for ordinal outcome (ESI scores 1-5)
    """
    variant_names = sorted(predictions.keys())
    
    # We need at least 3 variants for a meaningful omnibus test
    if len(variant_names) < 3:
        return {
            "cochran_q_stat": None,
            "cochran_q_p": None,
            "friedman_stat": None,
            "friedman_p": None,
            "error": "Omnibus tests require at least 3 variants."
        }

    # 1. Cochran's Q (Accuracy / Correctness)
    # Matrix shape: (n_cases, n_variants)
    correctness_matrix = np.column_stack([
        (predictions[v] == y_true).astype(int) 
        for v in variant_names
    ])
    
    q_stat, q_p = _cochrans_q(correctness_matrix)
    
    # 2. Friedman Test (Predicted ESI Scores)
    # Each variant provides an array of scores
    pred_arrays = [predictions[v] for v in variant_names]
    
    try:
        f_stat, f_p = scipy_stats.friedmanchisquare(*pred_arrays)
        f_stat, f_p = float(f_stat), float(f_p)
    except ValueError:
        # Can happen if all values are identical
        f_stat, f_p = 0.0, 1.0

    # 3. FDR Correction (Benjamini-Hochberg)
    # We apply FDR correction across the p-values we have computed for this model.
    # In a full pipeline with multiple models, a global FDR might be applied later,
    # but here we compute the adjusted p-values for the omnibus tests within the model.
    p_values = [p for p in [q_p, f_p] if p is not None]
    
    try:
        from scipy.stats import false_discovery_control
        fdr_p_values = false_discovery_control(p_values)
        cochran_q_fdr_p = float(fdr_p_values[0]) if len(fdr_p_values) > 0 else None
        friedman_fdr_p = float(fdr_p_values[1]) if len(fdr_p_values) > 1 else None
    except ImportError:
        # Fallback if false_discovery_control is not available
        cochran_q_fdr_p = q_p
        friedman_fdr_p = f_p

    return {
        "cochran_q_stat": q_stat,
        "cochran_q_p": q_p,
        "cochran_q_fdr_p": cochran_q_fdr_p,
        "friedman_stat": f_stat,
        "friedman_p": f_p,
        "friedman_fdr_p": friedman_fdr_p,
        "variants_tested": len(variant_names)
    }
