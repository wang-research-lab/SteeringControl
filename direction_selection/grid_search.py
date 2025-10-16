"""Implements grid search over steering directions and hyperparameters.

Generate candidate directions on the full training data, then evaluate each
candidate (for each application location and steering factor) on the entire
validation dataset. The configuration with the highest mean evaluation score
is selected. Intermediate results are saved to CSV for further analysis.
"""

STEERING_FACTORS = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0}  # From AxBench Appendix K.

import os
import logging
from typing import List, Any, Dict, Tuple, Optional, Literal, Union

import numpy as np
import pandas as pd
import torch
import asyncio
from itertools import product, combinations
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from direction_selection.base import DirectionSelector
from utils.steering_utils import CandidateDirection, ActivationLocation
from data.steering_data import SteeringTrainData, FormattedDataset
from direction_application.base import DirectionApplier, ApplicationLocation, InterventionPositions
from direction_application.conditional import ConditionalApplier
from utils.intervention_llm import InterventionLLM
from utils.enums import DEFAULT_MC_METHOD


def create_padded_composite_direction(directions, cum_idxs, cum_layers, base_dir):
    """Create a composite direction with padded arrays for direct layer indexing.
    
    Args:
        directions: List of all candidate directions
        cum_idxs: Indices into directions list for the cumulative layers
        cum_layers: The layer numbers to include in the composite direction
        base_dir: Base direction to copy metadata from
        
    Returns:
        CandidateDirection with padded direction and reference_activation lists
    """
    max_layer = max(cum_layers)
    padded_directions = [None] * (max_layer + 1)
    padded_reference_activations = [None] * (max_layer + 1)
    
    for i, layer in enumerate(cum_layers):
        padded_directions[layer] = directions[cum_idxs[i]].direction
        if hasattr(directions[cum_idxs[i]], 'reference_activation') and directions[cum_idxs[i]].reference_activation is not None:
            padded_reference_activations[layer] = directions[cum_idxs[i]].reference_activation
    
    return CandidateDirection(
        direction=padded_directions,
        dataset_metadata=base_dir.dataset_metadata,
        model_name=base_dir.model_name,
        loc=base_dir.loc,
        reference_activation=padded_reference_activations
    )


class GridSearchSelector(DirectionSelector):
    """Performs grid search over candidate directions, steering factors, and application locations by full-validation evaluation.

    Generates candidate directions once on the training set, then evaluates each
    candidate (with each location and factor) on the full validation set to pick
    the highest-scoring configuration.
    """

    def __init__(self,
                 applier: DirectionApplier,
                 application_locations: List[ApplicationLocation],
                 factors: List[float] = None,
                 result_path: str = "grid_search_results.csv",
                 include_generation_loc: bool = False,
                 generation_pos: Any = None,
                 cumulative_layers: bool = False,
                 reverse: bool = None,
                 _orig_layers: Any = None,
                 use_kl_divergence_check: bool = False,
                 kl_threshold: float = 0.1,
                 harmless_validation_data: Any = None
                 ):
        """
        Args:
            generator: The DirectionGenerator to use for generating candidate directions.
            generator_param_grid: Mapping of generator parameter names to lists of values for grid search.
            applier: The DirectionApplier to use when steering the model.
            application_locations: List of ApplicationLocation indicating where to apply the direction during inference.
            factors: List of steering coefficients to search over.
            result_path: Path to save CSV of intermediate results.
        """
        super().__init__(
            applier=applier,
            application_locations=application_locations,
            factors=factors,
            result_path=result_path,
            include_generation_loc=include_generation_loc,
            generation_pos=generation_pos,
            cumulative_layers=cumulative_layers,
            reverse=reverse,
            _orig_layers=_orig_layers,
            use_kl_divergence_check=use_kl_divergence_check,
            kl_threshold=kl_threshold,
            harmless_validation_data=harmless_validation_data
        )

    def select(
    self,
    model: InterventionLLM,
    candidate_directions: Any,
    validation_data: FormattedDataset,
    train_data: Union[FormattedDataset, List[FormattedDataset]] = None,  # not used here.
    neg_validation_data: FormattedDataset = None,  # not used here.
    maximize: bool = True,
    **generate_kwargs
) -> CandidateDirection:
        """Select the best direction and factor via full‐validation evaluation."""
        # 1) Generate candidate directions using base class method
        directions, direction_params = candidate_directions

        # prepare validation sets
        if isinstance(validation_data, (list,tuple)):
            val_list = validation_data
        else:
            val_list = [validation_data]

        records: List[Dict[str,Any]] = []
        tasks = []

        # 3) Build & run cumulative‐layers evaluations
        if self.cumulative_layers:
            if self.generation_pos is None:
                raise ValueError("`generation_pos` must be set when `cumulative_layers` is True")

            # collect *all* layers we actually generated
            all_layers = [p["layer"] for p in direction_params]

            # Only build tasks for the user‐requested layers (self._orig_layers),
            # but now we have directions at every layer ≤ each of those.
            for base_idx, params_ in enumerate(direction_params):
                L = params_["layer"]
                if L not in self._orig_layers:
                    continue

                # the prefix: every generated layer ≤ L
                cum_layers = sorted(l for l in all_layers if l <= L)

                if self.factors is None:
                    tasks.append((base_idx, cum_layers, None))
                else:
                    for f in self.factors:
                        tasks.append((base_idx, cum_layers, f))

            # Evaluate
            for base_idx, cum_layers, factor in tqdm(tasks, total=len(tasks), desc="Grid search evaluation"):
                base_params = direction_params[base_idx]

                # build ApplicationLocation across all cum_layers
                pos_val = self.generation_pos
                if isinstance(pos_val, str):
                    try:    pos_val = InterventionPositions[pos_val]
                    except KeyError: pass

                app_loc = ApplicationLocation(
                    layer     = cum_layers,
                    component = base_params["component"],
                    attr      = base_params["attr"],
                    pos       = pos_val
                )
                applier_kwargs = self._create_applier_kwargs(factor)

                # collect the direction vectors in prefix order
                cum_idxs = [
                    next(i for i,p in enumerate(direction_params) if p["layer"]==layer)
                    for layer in cum_layers
                ]
                base_dir = directions[base_idx]
                combined_dir = create_padded_composite_direction(directions, cum_idxs, cum_layers, base_dir)

                # Check KL divergence threshold if enabled
                kl_div = None
                passes_kl = True
                if self.use_kl_divergence_check:
                    kl_div, passes_kl = self._compute_kl_divergence_with_threshold(
                        model, combined_dir, applier_kwargs, app_loc
                    )
                    if not passes_kl:
                        logging.info(f"Direction {base_idx} with factor {factor} failed KL threshold (KL={kl_div:.4f})")
                        rec = {
                            "direction_idx": base_idx,
                            "factor"       : factor,
                            "score"        : float('nan'),
                            "kl_divergence": kl_div,
                            "passes_kl_threshold": passes_kl,
                        }
                        rec.update(base_params)
                        records.append(rec)
                        continue  # Skip this candidate
                
                # run eval
                scores = []
                for vds in val_list:
                    metrics = asyncio.run(
                        vds.evaluate(
                            model                = model,
                            direction            = combined_dir,
                            applier              = self.applier,
                            application_location = app_loc,
                            applier_kwargs       = applier_kwargs,
                            gather_evaluations   = False,
                            mc_evaluation_method = DEFAULT_MC_METHOD,
                            **generate_kwargs
                        )
                    )
                    ds_score = float(np.mean(list(metrics.values()))) if metrics else float('nan')
                    scores.append(ds_score)

                score = float(np.mean(scores)) if scores else float('nan')
                rec = {
                    "direction_idx": base_idx,
                    "factor"       : factor,
                    "score"        : score,
                    "kl_divergence": kl_div,
                    "passes_kl_threshold": passes_kl,
                }
                print(f"Evaluated direction {base_idx} with factor {factor} and score {score}")
                rec.update(base_params)
                records.append(rec)

        else:
            # --- The entire rest of your “non‐cumulative” branch is copied verbatim ---
            for dir_idx, direction in enumerate(directions):
                locs = self._build_application_locations_with_generation([direction], self.application_locations)

                for loc_idx, app_loc in enumerate(locs):
                    if self.factors is None:
                        tasks.append((dir_idx, loc_idx, app_loc, None))
                    else:
                        for factor in self.factors:
                            tasks.append((dir_idx, loc_idx, app_loc, factor))

            for dir_idx, loc_idx, app_loc, factor in tqdm(tasks, total=len(tasks), desc="Grid search evaluation"):
                direction = directions[dir_idx]
                params    = direction_params[dir_idx]
                applier_kwargs = self._create_applier_kwargs(factor)
                
                # Check KL divergence threshold if enabled -- do first so we can skip this candidate early
                kl_div = None
                passes_kl = True
                if self.use_kl_divergence_check:
                    kl_div, passes_kl = self._compute_kl_divergence_with_threshold(
                        model, direction, applier_kwargs, app_loc
                    )
                    if not passes_kl:
                        logging.info(f"Direction {dir_idx} with factor {factor} failed KL threshold (KL={kl_div:.4f})")
                        rec = {
                            "direction_idx": dir_idx,
                            "location_idx" : loc_idx,
                            "factor"       : factor,
                            "score"        : float('nan'),
                            "kl_divergence": kl_div,
                            "passes_kl_threshold": passes_kl,
                        }
                        rec.update(params)
                        records.append(rec)
                        continue  # Skip this candidate
                
                scores = []
                for vds in val_list:
                    metrics = asyncio.run(
                        vds.evaluate(
                            model                = model,
                            direction            = direction,
                            applier              = self.applier,
                            application_location = app_loc,
                            applier_kwargs       = applier_kwargs,
                            gather_evaluations   = False,
                            mc_evaluation_method = DEFAULT_MC_METHOD,
                            **generate_kwargs
                        )
                    )
                    ds_score = float(np.mean(list(metrics.values()))) if metrics else float('nan')
                    scores.append(ds_score)
                score = float(np.mean(scores)) if scores else float('nan')
                rec = {
                    "direction_idx": dir_idx,
                    "location_idx" : loc_idx,
                    "factor"       : factor,
                    "score"        : score,
                    "kl_divergence": kl_div,
                    "passes_kl_threshold": passes_kl,
                }
                print(f"Evaluated direction {dir_idx} with factor {factor} and score {score}")
                rec.update(params)
                records.append(rec)
            # --- end of non‐cumulative branch ---

        # Save results, pick best, and return exactly as before…
        df = pd.DataFrame(records)

        os.makedirs(os.path.dirname(self.result_path) or ".", exist_ok=True)
        df.to_csv(self.result_path, index=False)

        best_idx = df.score.idxmax() if maximize else df.score.idxmin()
        best_row = df.loc[best_idx]


        if self.cumulative_layers:
            base_idx, cum_layers, _ = tasks[best_idx]
            cum_idxs = [
                next(i for i,p in enumerate(direction_params) if p["layer"]==layer)
                for layer in cum_layers
            ]
            base_dir = directions[base_idx]
            best_direction = create_padded_composite_direction(directions, cum_idxs, cum_layers, base_dir)
            setattr(best_direction, "val_score", float(best_row.score))
            for k in direction_params[base_idx]:
                setattr(best_direction, k, best_row[k])
            setattr(best_direction, "factor", best_row.factor)
            pos_val = self.generation_pos
            if isinstance(pos_val, str):
                try:    pos_val = InterventionPositions[pos_val]
                except KeyError: pass
            best_loc = ApplicationLocation(
                layer     = cum_layers,
                component = direction_params[base_idx]["component"],
                attr      = direction_params[base_idx]["attr"],
                pos       = pos_val
            )
        else:
            best_direction = directions[int(best_row.direction_idx)]
            setattr(best_direction, "val_score", float(best_row.score))
            for k in direction_params[int(best_row.direction_idx)]:
                setattr(best_direction, k, best_row[k])
            setattr(best_direction, "factor", best_row.factor)
            if self.include_generation_loc:
                pos_val = self.generation_pos
                if isinstance(pos_val, str):
                    try:    pos_val = InterventionPositions[pos_val]
                    except KeyError: pass
                gen_loc = best_direction.loc
                gen_app_loc = ApplicationLocation(
                    layer     = [gen_loc.layer],
                    component = gen_loc.component,
                    attr      = gen_loc.attr,
                    pos       = pos_val
                )
                locs = list(self.application_locations) + [gen_app_loc]
                best_loc = locs[int(best_row.location_idx)]
            else:
                best_loc = self.application_locations[int(best_row.location_idx)]

        setattr(best_direction, "application_location", best_loc)
        logging.info(
            f"Selected {'cumulative ' if self.cumulative_layers else ''}"
            f"direction with factor {best_row.factor}, score {best_row.score}"
        )
        return best_direction


class ConditionalGridSearchSelector(DirectionSelector):
    """Grid search selector for conditional steering hyperparameters.
    
    Follows the pattern from the CAST activation-steering repo's find_best_condition_point method in malleable_model.py:488
    but adapted for our nnsight-based framework. This class focuses ONLY on finding optimal
    condition detection hyperparameters, not behavior application.
    
    Steps:
    1. Take pre-generated condition directions (with inherent locations in direction.loc)
    2. Collect similarities at each condition direction's location for all validation samples
    3. Test all combinations of (condition_direction, threshold, comparator) using F1 score
    4. Return best condition configuration for use in ConditionalApplier
    
    The condition directions contain both the vector and location (layer, attr, component).
    Positive/negative samples are handled like in cosmic.py using pos_inputs/neg_inputs.
    """
    
    def __init__(self,
                 condition_thresholds: List[float] = "auto",
                 condition_comparators: List[str] = None,
                 condition_threshold_comparison_mode: str = "mean",
                 result_path: str = "conditional_grid_search_results.csv"):
        """
        Args:
            condition_thresholds: List of threshold values for condition activation. Auto means we search from min to max similarity with 5 steps.
            condition_comparators: List of comparison operators ("greater" or "less")
            condition_threshold_comparison_mode: Token aggregation mode ("mean" or "last")
            result_path: Path to save grid search results
        """
        self.condition_thresholds = condition_thresholds or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.condition_comparators = condition_comparators or ["greater", "less"]
        self.condition_threshold_comparison_mode = condition_threshold_comparison_mode
        self.result_path = result_path
        
    def select(
        self,
        model: InterventionLLM,
        condition_directions: List[CandidateDirection],
        validation_data: Union[FormattedDataset, List[FormattedDataset]],
        neg_validation_data: Union[FormattedDataset, List[FormattedDataset]] = None,
        maximize: bool = True
    ) -> Dict[str, Any]:
        """Select the best condition direction and hyperparameters using F1 score on similarities.
        
        Args:
            model: The model to use for activation collection
            condition_directions: Pre-generated candidate directions for condition checking
            validation_data: Positive validation dataset (or paired dataset with pos/neg samples)
            neg_validation_data: Negative validation dataset (if validation_data contains only positive samples)
            maximize: Whether to maximize the score (True for F1)
            
        Returns:
            Dictionary with best condition configuration:
            {
                "condition_direction": CandidateDirection,
                "condition_threshold": float,
                "condition_comparator": str,
                "f1_score": float
            }
        """

        # Handle validation data as list or single dataset
        if isinstance(validation_data, (list, tuple)):
            pos_val_list = validation_data
        else:
            pos_val_list = [validation_data]
            
        # Handle negative validation data
        if neg_validation_data is not None:
            if isinstance(neg_validation_data, (list, tuple)):
                neg_val_list = neg_validation_data
            else:
                neg_val_list = [neg_validation_data]
            
            print(f"ConditionalGridSearch DEBUG: Using explicit pos/neg datasets - pos: {[ds.name for ds in pos_val_list]}, neg: {[ds.name for ds in neg_val_list]}")
        else:
            neg_val_list = None

        # 1) Extract positive and negative samples from validation data using shared method
        pos_prompts, neg_prompts = self._generate_prompts_from_validation_data(pos_val_list, neg_val_list, format_train=False)  # we don't want the answers because we want to mimic what prompt a model would get on the fly during inference.
        
        # Create ground truth labels: 1 for positive samples, 0 for negative
        y_true = [1] * len(pos_prompts) + [0] * len(neg_prompts)
        all_samples = pos_prompts + neg_prompts
        
        print(f"Processing {len(pos_prompts)} positive and {len(neg_prompts)} negative samples")
        
        records: List[Dict[str, Any]] = []
        
        # 2) For each condition direction, collect similarities and find best hyperparameters
        for dir_idx, condition_direction in enumerate(condition_directions):
            print(f"Processing condition direction {dir_idx} at layer {condition_direction.loc.layer}...")
            
            # Step 2a: Collect activations at this condition direction's location
            condition_loc = ActivationLocation(
                layer=condition_direction.loc.layer,
                component=condition_direction.loc.component,
                attr=condition_direction.loc.attr,
                pos=condition_direction.loc.pos  # -1  # None  # Use -1 for consistency with the generated directions that also used -1.
            )

            batch_size = 1  # so they are in the correct order
            activations = self._extract_activations_for_layer(model, all_samples, condition_loc.layer, batch_size, condition_loc)
            
            # Step 2b: Compute similarities for each sample
            similarities = []
            for activation in activations:
                # Get activation tensor and compute similarity
                activation_tensor = activation.squeeze()  # Remove batch dimension if present
                
                # Compute similarity using the same logic as ConditionalApplier
                projected_hidden_state = ConditionalApplier.project_hidden_state(activation_tensor, condition_direction.direction)
                similarity = ConditionalApplier.compute_similarity(activation_tensor, projected_hidden_state).item()
                similarities.append(similarity)
            
            # Step 2c: Test all combinations of (threshold, comparator) for F1 score
            best_f1 = 0
            best_config = None
            
            # Assign condition thresholds based on user input or auto.
            # We need auto bc good thresholds could vary widely among condition directions based on models, layers, etc.
            if self.condition_thresholds == "auto":
                condition_thresholds = np.linspace(
                    min(similarities), 
                    max(similarities), 
                    num=5
                ).tolist()
            else:
                condition_thresholds = self.condition_thresholds
            for threshold in condition_thresholds:
                for comparator in self.condition_comparators:
                    
                    # Compute predictions based on condition_met logic
                    y_pred = []
                    for similarity in similarities:
                        if comparator == "greater":
                            condition_met = similarity > threshold
                        else:  # "less"
                            condition_met = similarity < threshold
                        y_pred.append(int(condition_met))
                    
                    # Compute F1 score
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                    
                    # Track best configuration
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = {
                            "direction_idx": dir_idx,
                            "condition_direction": condition_direction,
                            "condition_threshold": threshold,
                            "condition_comparator": comparator,
                            "f1_score": f1
                        }
                    
                    print(f"  Threshold {threshold}, comparator {comparator}: F1 = {f1:.3f}")
            
            # Record the best configuration for this direction
            if best_config:
                rec = {
                    "direction_idx": dir_idx,
                    "condition_loc": condition_direction.loc,
                    "condition_threshold": best_config["condition_threshold"],
                    "condition_comparator": best_config["condition_comparator"],
                    "f1_score": best_config["f1_score"],
                }
                records.append(rec)
                
                print(f"Best config for direction {dir_idx}: F1 = {best_f1:.3f}")
        
        # 3) Save results and select overall best condition configuration
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(self.result_path) or ".", exist_ok=True)
        df.to_csv(self.result_path, index=False)
        
        if df.empty:
            raise ValueError("No valid condition configurations found")
        
        best_idx = df.f1_score.idxmax() if maximize else df.f1_score.idxmin()
        best_row = df.loc[best_idx]
        
        # 4) Return best condition configuration
        best_direction_idx = int(best_row.direction_idx)
        best_condition_direction = condition_directions[best_direction_idx]
        
        result = {
            "condition_direction": best_condition_direction,
            "condition_threshold": float(best_row.condition_threshold),
            "condition_comparator": best_row.condition_comparator,
            "f1_score": float(best_row.f1_score),
            "condition_loc": best_condition_direction.loc,
            "condition_threshold_comparison_mode": self.condition_threshold_comparison_mode
        }
        
        logging.info(
            f"Selected condition: location {best_condition_direction.loc}, "
            f"threshold {best_row.condition_threshold}, comparator {best_row.condition_comparator}, "
            f"F1 score {best_row.f1_score:.3f}"
        )
        
        return result
