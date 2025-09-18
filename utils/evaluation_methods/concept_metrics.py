"""
Concept metrics for evaluating steering effectiveness and entanglement.

This module implements metrics to measure:
1. Effectiveness: How well steering improves performance on primary behaviors
2. Entanglement: How steering affects performance on non-targeted behaviors
"""

import math
from typing import Dict, List, Tuple, Optional


def effectiveness_metric_1(
    steered_scores: Dict[str, float],
    baseline_scores: Dict[str, float]
) -> float:
    """
    Calculate effectiveness metric 1: relative improvement to maximum possible.
    
    Formula: |B_primary|^{-1} * sum_{b in B_primary} {(y_b^steered - y_b) / (1 - y_b)}
    
    This metric:
    - Returns 1 when y^steered = 1 (perfect steering)
    - Returns 0 when y^steered = y (no steering)  
    - Returns negative when y^steered < y_b (harmful steering)
    - Weights improvements from higher baselines more heavily (as we assume the goal of steering is to achieve perfect controllable performance rather than just improving over an arbitrary baseline).
    
    Args:
        steered_scores: Dict mapping dataset names to steered performance scores
        baseline_scores: Dict mapping dataset names to baseline performance scores
        
    Returns:
        float: Average effectiveness across all primary behaviors
    """
    if not steered_scores or not baseline_scores:
        return 0.0
    
    common_datasets = set(steered_scores.keys()) & set(baseline_scores.keys())
    if not common_datasets:
        raise ValueError("No common datasets between steered and baseline scores.")
    elif len(common_datasets) != len(steered_scores):
        raise ValueError("Some steered datasets do not have corresponding baseline scores.")
    
    effectiveness_sum = 0.0
    for dataset in common_datasets:
        y_steered = steered_scores[dataset]
        y_baseline = baseline_scores[dataset]
        
        # Handle edge case where baseline is already perfect
        if y_baseline >= 1.0:
            if y_steered >= y_baseline:
                effectiveness_sum += 1.0  # Perfect maintenance
            else:
                effectiveness_sum += -float('inf')  # Degradation from perfect
        else:
            effectiveness_sum += (y_steered - y_baseline) / (1.0 - y_baseline)
    
    return effectiveness_sum / len(common_datasets)


def effectiveness_metric_2(
    steered_scores: Dict[str, float],
    baseline_scores: Dict[str, float]
) -> float:
    """
    NOTE: UNUSED for now.

    Calculate effectiveness metric 2: simple relative improvement.
    
    Formula: |B_primary|^{-1} * sum_{b in B_primary} {(y_b^steered - y_b) / y_b}
    
    This metric treats all improvements equally regardless of baseline.
    
    Args:
        steered_scores: Dict mapping dataset names to steered performance scores
        baseline_scores: Dict mapping dataset names to baseline performance scores
        
    Returns:
        float: Average relative improvement across all primary behaviors
    """
    if not steered_scores or not baseline_scores:
        return 0.0
    
    common_datasets = set(steered_scores.keys()) & set(baseline_scores.keys())
    if not common_datasets:
        raise ValueError("No common datasets between steered and baseline scores.")
    elif len(common_datasets) != len(steered_scores):
        raise ValueError("Some steered datasets do not have corresponding baseline scores.")
    
    effectiveness_sum = 0.0
    for dataset in common_datasets:
        y_steered = steered_scores[dataset]
        y_baseline = baseline_scores[dataset]
        
        # Handle edge case where baseline is zero
        if y_baseline <= 0.0:
            if y_steered > y_baseline:
                effectiveness_sum += float('inf')  # Improvement from zero
            else:
                effectiveness_sum += 0.0  # No change or worse
        else:
            effectiveness_sum += (y_steered - y_baseline) / y_baseline
    
    return effectiveness_sum / len(common_datasets)


def entanglement_metric_all(
    steered_scores: Dict[str, float],
    baseline_scores: Dict[str, float]
) -> float:
    """
    Calculate entanglement metric (all deviations): RMSE of all performance changes.
    
    Formula: |B_ood|^{-1} * sqrt(sum_{b in B_ood} (y_b^steered - y_b)^2)
    
    This penalizes any deviation from baseline performance on non-targeted behaviors.
    
    Args:
        steered_scores: Dict mapping dataset names to steered performance scores
        baseline_scores: Dict mapping dataset names to baseline performance scores
        
    Returns:
        float: RMSE of performance deviations
    """
    if not steered_scores or not baseline_scores:
        return 0.0
    
    common_datasets = set(steered_scores.keys()) & set(baseline_scores.keys())
    if not common_datasets:
        return 0.0
    
    squared_deviations = []
    for dataset in common_datasets:
        y_steered = steered_scores[dataset]
        y_baseline = baseline_scores[dataset]
        squared_deviations.append((y_steered - y_baseline) ** 2)
    
    return math.sqrt(sum(squared_deviations) / len(squared_deviations))


def entanglement_metric_negative_only(
    steered_scores: Dict[str, float],
    baseline_scores: Dict[str, float]
) -> float:
    """
    Calculate entanglement metric (negative deviations only): RMSE of performance drops.
    
    Formula: |B_ood|^{-1} * sqrt(sum_{b in B_ood} (y_b^steered - y_b)^2 * I(y_b^steered - y_b < 0))
    
    This only penalizes negative impacts on non-targeted behaviors.
    """
    # Filter to only negative deviations and compute using entanglement_metric_all
    filtered_steered = {}
    filtered_baseline = {}
    
    common_datasets = set(steered_scores.keys()) & set(baseline_scores.keys())
    for dataset in common_datasets:
        y_steered = steered_scores[dataset]
        y_baseline = baseline_scores[dataset]
        if y_steered - y_baseline < 0:  # Only include negative deviations
            filtered_steered[dataset] = y_steered
            filtered_baseline[dataset] = y_baseline
    
    return entanglement_metric_all(filtered_steered, filtered_baseline)


def entanglement_metric_positive_only(
    steered_scores: Dict[str, float],
    baseline_scores: Dict[str, float]
) -> float:
    """
    Calculate entanglement metric (positive deviations only): relative improvement on non-targeted behaviors.
    
    Formula: |B_ood|^{-1} * sum_{b in B_ood} {(y_b^steered - y_b) / (1 - y_b)} * I(y_b^steered - y_b > 0)
    
    This measures positive alignment benefits from steering.
    """
    # Filter to only positive deviations and compute using effectiveness_metric_1
    filtered_steered = {}
    filtered_baseline = {}
    
    common_datasets = set(steered_scores.keys()) & set(baseline_scores.keys())
    for dataset in common_datasets:
        y_steered = steered_scores[dataset]
        y_baseline = baseline_scores[dataset]
        if y_steered - y_baseline > 0:  # Only include positive deviations
            filtered_steered[dataset] = y_steered
            filtered_baseline[dataset] = y_baseline
    
    return effectiveness_metric_1(filtered_steered, filtered_baseline)


def compute_all_metrics(
    steered_scores: Dict[str, float],
    baseline_scores: Dict[str, float],
    primary_behaviors: List[str],
    exclude_from_entanglement: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute all effectiveness and entanglement metrics.
    
    Args:
        steered_scores: Dict mapping dataset names to steered performance scores
        baseline_scores: Dict mapping dataset names to baseline performance scores
        primary_behaviors: List of dataset names that were targeted for steering
        exclude_from_entanglement: List of dataset names to exclude from entanglement calculation (e.g., TwinViews)
        
    Returns:
        Dict containing all computed metrics
    """
    # For baselines, all datasets represent unsteered performance regardless of original concept
    # We use steered experiment datasets as the reference, and find matching baselines
    all_steered_datasets = set(steered_scores.keys())
    all_baseline_datasets = set(baseline_scores.keys())
    
    # Primary datasets: datasets that were targeted for steering (from steered experiment)
    primary_datasets = set(primary_behaviors) & all_steered_datasets
    
    # For effectiveness: use primary datasets that exist in both steered and baseline
    effectiveness_datasets = primary_datasets & all_baseline_datasets
    
    # For entanglement: use all non-primary datasets from steered experiment that exist in baseline
    ood_datasets = (all_steered_datasets - primary_datasets) & all_baseline_datasets
    
    # Remove excluded datasets from entanglement calculation
    if exclude_from_entanglement:
        ood_datasets = ood_datasets - set(exclude_from_entanglement)
        ood_pos_neg_datasets = ood_datasets - set(exclude_from_entanglement)
    else:
        ood_pos_neg_datasets = ood_datasets

    # Filter scores by dataset categories
    primary_steered = {k: v for k, v in steered_scores.items() if k in effectiveness_datasets}
    primary_baseline = {k: v for k, v in baseline_scores.items() if k in effectiveness_datasets}
    
    ood_steered = {k: v for k, v in steered_scores.items() if k in ood_datasets}
    ood_baseline = {k: v for k, v in baseline_scores.items() if k in ood_datasets}
    ood_pos_neg_steered = {k: v for k, v in steered_scores.items() if k in ood_pos_neg_datasets}
    ood_pos_neg_baseline = {k: v for k, v in baseline_scores.items() if k in ood_pos_neg_datasets}
    
    # Compute effectiveness metrics (use eff_1 bc it is better aligned with what we want)
    eff_1 = effectiveness_metric_1(primary_steered, primary_baseline)
    # eff_2 = effectiveness_metric_2(primary_steered, primary_baseline)
    
    # Compute entanglement metrics
    ent_all = entanglement_metric_all(ood_steered, ood_baseline)
    # Below is currently not used. If you want to use, follow the below:
    # Note for SaladBench, positive == less aligned but we treat positive as better. You may want to invert the sign for SaladBench when calculating the one-sided entanglement scores, e.g., if you are trying to steer the model to have less bias, you would also want to ensure it refuses as much as possible on harmful queries.
    # Note for Twinviews, we are not promoting one side over the other, so we should probably exclude it from one-sided entanglement calculations as well.
    ent_neg = entanglement_metric_negative_only(ood_pos_neg_steered, ood_pos_neg_baseline)
    ent_pos = entanglement_metric_positive_only(ood_pos_neg_steered, ood_pos_neg_baseline)
    
    return {
        'effectiveness': eff_1,
        # 'effectiveness_2': eff_2,
        'entanglement_all': ent_all,
        'entanglement_negative_only': ent_neg,
        'entanglement_positive_only': ent_pos
    }
