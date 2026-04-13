from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class ConsistencyResult:
    agreement_rate: float
    mean_prediction_range: float
    mean_prediction_variance: float
    pct_clinical_boundary_crossings: float
    
    # Legacy compatibility fields
    perturbation_sensitivity_score: float
    pct_any_disagreement: float
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
        
        # Compute the legacy composite score
        # Using disagreement rate (1 - agreement_rate) to match old metric
        mean_disagreement = 1.0 - agreement_rate
        composite = (mean_disagreement + mean_var + mean_range / 4) / 3
        
        # Baseline change
        from config import BASELINE_VARIANT, LABELED_VARIANTS
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
            pct_any_disagreement=mean_disagreement,
            pct_range_gte_2=float(np.mean(pred_range >= 2)),
            pct_any_sex_label_changes_baseline=pct_change_baseline
        )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
