"""COSMIC (Cosine Similarity Metrics for Inversion of Concepts) direction selection.

Implements the COSMIC method for automated direction selection using cosine similarity
metrics on model activations. This method selects layers based on low cosine similarity
between positive and negative prompts, then evaluates candidate directions using
concept inversion scoring.

Based on the paper: "COSMIC: Generalized Refusal Direction Identification in LLM Activations"
"""

import logging
from typing import List, Any, Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import pandas as pd
import os
import random
random.seed(42)

from direction_selection.base import DirectionSelector, kl_div_fn
from utils.steering_utils import CandidateDirection, ActivationLocation, ActivationResults
from data.steering_data import SteeringTrainData, FormattedDataset
from direction_application.base import DirectionApplier, ApplicationLocation, InterventionType
from utils.intervention_llm import InterventionLLM


class CosmicSelector(DirectionSelector):
    """COSMIC direction selector using cosine similarity metrics for automated direction selection.
    
    This method:
    1. Generates answer-conditioned prompts from validation data
    2. Selects layers with lowest cosine similarity between positive/negative activations
    3. Uses concept inversion scoring to evaluate candidate directions
    4. Applies KL divergence filtering to ensure direction quality
    """
    
    def __init__(self,
                 applier: DirectionApplier,
                 application_locations: List[ApplicationLocation],
                 factors: List[float] = None,
                 result_path: str = "cosmic_results.csv",
                 include_generation_loc: bool = False,
                 generation_pos: Any = None,
                 cumulative_layers: bool = False,
                 reverse: bool = None,
                 _orig_layers: Any = None,
                 layer_selection_pct: float = 0.1,
                 kl_threshold: float = 0.1,
                 harmless_validation_data: Any = None,
                 positive_applier: DirectionApplier = None,
                 negative_applier: DirectionApplier = None,
                 positive_application_locations: List[ApplicationLocation] = None,
                 negative_application_locations: List[ApplicationLocation] = None):
        """
        Args:
            applier: The DirectionApplier to use when applying interventions (fallback if positive/negative not specified).
            application_locations: List of ApplicationLocation indicating where to apply the direction during inference (fallback if positive/negative not specified).
            factors: List of steering coefficients to search over.
            result_path: Path to save CSV of intermediate results.
            include_generation_loc: Optional: include application location equal to each direction's generation loc
            generation_pos: Default position to use for generated-loc application (e.g., POST_INSTRUCTION)
            cumulative_layers: Whether to apply steering cumulatively across all layers up to the target layer
            reverse: Whether we need to switch the direction of the intervention based on if we want to amplify the generated direction or reduce it
            layer_selection_pct: Percentage of layers to select based on lowest cosine similarity.
            kl_threshold: Maximum KL divergence threshold for direction filtering.
            positive_applier: Optional: specific applier for positive interventions (overrides applier).
            negative_applier: Optional: specific applier for negative interventions (overrides applier).
            positive_application_locations: Optional: specific application locations for positive interventions (overrides application_locations).
            negative_application_locations: Optional: specific application locations for negative interventions (overrides application_locations).
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
            use_kl_divergence_check=True,  # COSMIC always uses KL divergence checking
            kl_threshold=kl_threshold,
            harmless_validation_data=harmless_validation_data,
            layer_selection_pct=layer_selection_pct
        )
        
        
        # Set up appliers for positive and negative interventions
        # POSITIVE defaults to the main applier (for final inference)
        self.positive_applier = positive_applier if positive_applier is not None else applier
        self.negative_applier = negative_applier if negative_applier is not None else applier
        
        # Set up application locations for positive and negative interventions  
        # POSITIVE defaults to the main application_locations (for final inference)
        self.positive_application_locations = positive_application_locations if positive_application_locations is not None else application_locations
        self.negative_application_locations = negative_application_locations if negative_application_locations is not None else application_locations
        # Validate that we have single application locations (COSMIC doesn't grid search over locations)
        if self.positive_application_locations and len(self.positive_application_locations) > 1:
            raise ValueError("COSMIC doesn't support multiple positive_application_locations. Use only one location.")
        if self.negative_application_locations and len(self.negative_application_locations) > 1:
            raise ValueError("COSMIC doesn't support multiple negative_application_locations. Use only one location.")
        if self.application_locations and len(self.application_locations) > 1:
            raise ValueError("COSMIC doesn't support multiple application_locations. Use only one location.")
    
    def _resolve_application_location_for_direction(self, direction: CandidateDirection, intervention_type: str = 'positive') -> ApplicationLocation:
        """Resolve the correct application location for a direction and intervention type.
        
        Args:
            direction: The candidate direction.
            intervention_type: Either 'positive' or 'negative'.
            
        Returns:
            The resolved ApplicationLocation to use.
        """
        from direction_application.base import ApplicationLocation, InterventionPositions
        # Determine which application location to use based on intervention type
        if intervention_type == 'positive' and len(self.positive_application_locations) > 0:
            app_loc = self.positive_application_locations[0]  # Use the single configured location
        elif intervention_type == 'negative' and len(self.negative_application_locations) > 0:
            # Handle special case of generation location marker
            neg_loc = self.negative_application_locations[0]
            if isinstance(neg_loc, dict) and neg_loc.get('use_generation_loc', False):
                # Create application location from the direction's generation location
                generation_pos = neg_loc.get('generation_pos', 'OUTPUT_ONLY')
                # Convert string to enum if needed
                if isinstance(generation_pos, str):
                    try:
                        pos_enum = InterventionPositions[generation_pos]
                    except KeyError:
                        pos_enum = generation_pos
                else:
                    pos_enum = generation_pos
                
                app_loc = ApplicationLocation(
                    layer=[direction.loc.layer],
                    component=direction.loc.component,
                    attr=direction.loc.attr,
                    pos=pos_enum
                )
            else:
                app_loc = self.negative_application_locations[0]  # Use the single configured location
        elif len(self.application_locations) > 0:
            app_loc = self.application_locations[0]  # Use the single default application location
        else:
            # Check if we should build from generation location using include_generation_loc
            if self.include_generation_loc and self.generation_pos is not None:
                pos_val = self.generation_pos
                if isinstance(pos_val, str):
                    try:
                        pos_val = InterventionPositions[pos_val]
                    except KeyError:
                        pass
                
                app_loc = ApplicationLocation(
                    layer=[direction.loc.layer],
                    component=direction.loc.component,
                    attr=direction.loc.attr,
                    pos=pos_val
                )
            else:
                raise ValueError(f"No application location available for {intervention_type} intervention")
        
        return app_loc
        
    def select(self,
               model: InterventionLLM,
               candidate_directions: Any,
               validation_data: Union[FormattedDataset, List[FormattedDataset]],
               train_data: Union[FormattedDataset, List[FormattedDataset]] = None,
               neg_validation_data: Union[FormattedDataset, List[FormattedDataset]] = None,
               **kwargs) -> CandidateDirection:
        """Select the best direction using COSMIC method.
        
        Args:
            model: The model to evaluate directions on.
            candidate_directions: Tuple of (directions, direction_params).
            validation_data: Positive validation data for evaluation (single dataset or list of datasets).
            train_data: Training data for layer selection (always has both positive and negative).
            neg_validation_data: Optional negative validation data. If provided, validation_data is treated as positive.
            
        Returns:
            The selected CandidateDirection with highest COSMIC score.
        """
        logging.info("Starting COSMIC direction selection...")
        
        # Extract batch_size from kwargs (defaults to 1 for backward compatibility)
        batch_size = kwargs.get('batch_size', 1)
        
        # Step 1: Generate candidate directions using base class method
        candidate_directions, direction_params = candidate_directions
        
        # Handle validation data as list or single dataset
        if isinstance(validation_data, (list, tuple)):
            pos_val_list = validation_data
        else:
            pos_val_list = [validation_data]
            
        # Handle training data for layer selection (similar to validation data handling)
        if train_data is not None:
            pos_train_list = [td.pos_inputs for td in train_data]
            neg_train_list = [td.neg_inputs for td in train_data]
            
        # Step 2: Prepare prompts from validation data
        if neg_validation_data is not None:
            # Explicit positive and negative datasets provided
            if isinstance(neg_validation_data, (list, tuple)):
                neg_val_list = neg_validation_data
            else:
                neg_val_list = [neg_validation_data]
            
        else:
            neg_val_list = None
            
        # Step 3: Select layers with lowest cosine similarity using TRAINING data (like official COSMIC)
        # This analyzes ALL layers, not just candidate layers
        pos_train_prompts, neg_train_prompts = self._generate_prompts_from_validation_data(pos_train_list, neg_train_list)
        selected_layers, layer_similarity_records = self._select_layers_by_cosine_similarity(
            model, pos_train_prompts, neg_train_prompts, batch_size
        )
        
        # Step 4: Evaluate candidate directions using concept inversion scoring (using VALIDATION data)
        pos_val_prompts, neg_val_prompts = self._generate_prompts_from_validation_data(pos_val_list, neg_val_list)
        best_direction, best_factor, evaluation_records = self._evaluate_candidates_with_cosmic(
            model, candidate_directions, pos_val_prompts, neg_val_prompts, selected_layers, pos_val_list, neg_val_list, direction_params, batch_size
        )
        
        # Step 5: Save results to CSV
        self._save_results_to_csv(layer_similarity_records, evaluation_records)
        
        # Step 6: Set application location and factor on best direction
        # Use POSITIVE application location since positive = final inference applier
        best_direction.application_location = self._resolve_application_location_for_direction(best_direction, 'positive')
        
        # Set the best factor found during evaluation
        best_direction.factor = best_factor
        
        return best_direction
    
    def _extract_base_prompt(self, prompt_data) -> str:
        """Extract base prompt string from formatted prompt data.
        
        Args:
            prompt_data: Either a string or list of chat messages.
            
        Returns:
            Base prompt as a string.
        """
        if isinstance(prompt_data, str):
            return prompt_data
        elif isinstance(prompt_data, list):
            # Chat format - extract content from messages
            content_parts = []
            for message in prompt_data:
                if isinstance(message, dict) and "content" in message:
                    content_parts.append(message["content"])
            return " ".join(content_parts)
        else:
            return str(prompt_data)
    
    def _select_layers_by_cosine_similarity(self, 
                                          model: InterventionLLM,
                                          pos_prompts: List[List[Dict[str, str]]],
                                          neg_prompts: List[List[Dict[str, str]]],
                                          batch_size: int = 16) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Select layers with lowest cosine similarity between pos/neg activations.
        
        Args:
            model: The model to extract activations from.
            pos_prompts: List of positive prompts.
            neg_prompts: List of negative prompts.
            batch_size: Batch size for activation collection.
            
        Returns:
            Tuple of (selected layer indices, layer similarity records for CSV).
        """
        logging.info("Selecting layers based on cosine similarity...")
        
        # COSMIC analyzes ALL layers, not just candidates - this ensures fair comparison
        # Use all model layers like the official COSMIC implementation
        all_layers = list(range(model.num_layers))
        unique_candidate_layers = all_layers
        layer_similarities = []
        layer_records = []
        
        for layer_idx in tqdm(unique_candidate_layers, desc="Computing layer similarities"):
            try:
                # Extract activations for this layer (last token only)
                pos_activations = self._extract_activations_for_layer(model, pos_prompts, layer_idx, batch_size)
                neg_activations = self._extract_activations_for_layer(model, neg_prompts, layer_idx, batch_size)
                
                # Check if we have valid activations
                if not pos_activations or not neg_activations:
                    logging.warning(f"No activations found for layer {layer_idx}")
                    continue
                
                # Compute mean activations
                pos_mean = torch.stack(pos_activations).mean(dim=0)
                neg_mean = torch.stack(neg_activations).mean(dim=0)
                
                # Compute cosine similarity between the entire vectors
                similarity = F.cosine_similarity(pos_mean, neg_mean, dim=-1).item()
                layer_similarities.append((layer_idx, similarity))
                
                # Record for CSV
                layer_records.append({
                    "layer": layer_idx,
                    "cosine_similarity": similarity,
                    "num_pos_prompts": len(pos_activations),
                    "num_neg_prompts": len(neg_activations)
                })
                
                logging.debug(f"Layer {layer_idx}: cosine similarity = {similarity:.4f}")
                
            except Exception as e:
                logging.warning(f"Error processing layer {layer_idx}: {e}")
                continue
        
        if not layer_similarities:
            raise ValueError("No valid layer similarities computed")
        
        # Sort by similarity and select lowest percentile
        layer_similarities.sort(key=lambda x: x[1])
        num_layers_to_select = max(1, int(len(unique_candidate_layers) * self.layer_selection_pct))
        selected_layers = [layer_idx for layer_idx, _ in layer_similarities[:num_layers_to_select]]
        selected_layers.sort()  # Ensure lower to higher order
        
        logging.info(f"Selected {len(selected_layers)} layers with lowest similarities: {selected_layers}")
        logging.info(f"Similarity range: {layer_similarities[0][1]:.4f} to {layer_similarities[-1][1]:.4f}")
        print(f"HOOKS COSMIC DEBUG: Selected {len(selected_layers)} layers out of {len(unique_candidate_layers)} candidate layers")
        print(f"HOOKS COSMIC DEBUG: Selected layers: {selected_layers}")
        print(f"HOOKS COSMIC DEBUG: All layer similarities: {[(layer, f'{sim:.4f}') for layer, sim in layer_similarities]}")
        
        # Store selected layers for comparison with official COSMIC
        self._selected_evaluation_layers = selected_layers
        
        # Mark selected layers in records
        for record in layer_records:
            record["selected"] = record["layer"] in selected_layers
        
        return selected_layers, layer_records
     
    def _evaluate_candidates_with_cosmic(self,
                                       model: InterventionLLM,
                                       candidate_directions: List[CandidateDirection],
                                       pos_prompts: List[str],
                                       neg_prompts: List[str],
                                       selected_layers: List[int],
                                       pos_validation_datasets: List[FormattedDataset],
                                       neg_validation_datasets: List[FormattedDataset] = None,
                                       direction_params: List[Dict[str, Any]] = None,
                                       batch_size: int = 1) -> Tuple[CandidateDirection, float, List[Dict[str, Any]]]:
        """Evaluate candidate directions using COSMIC scoring mechanism.
        
        Args:
            model: The model to evaluate on.
            candidate_directions: List of candidate directions to evaluate.
            pos_prompts: List of positive prompts (aggregated across all datasets).
            neg_prompts: List of negative prompts (aggregated across all datasets).
            selected_layers: List of selected layer indices.
            pos_validation_datasets: List of positive validation datasets.
            neg_validation_datasets: Optional list of negative validation datasets.
            direction_params: Direction parameters for evaluation.
            batch_size: Batch size for evaluation.
            
        Returns:
            Tuple of (best candidate direction, best factor, evaluation records for CSV).
        """
        logging.info(f"Evaluating {len(candidate_directions)} candidates using COSMIC...")
        
        best_direction = None
        best_score = -float('inf')
        best_factor = 1.0
        evaluation_records = []
        
        # Use the same factors as would be used in grid search
        factors_to_test = self.factors if self.factors is not None else [1.0]
        
        for dir_idx, direction in enumerate(tqdm(candidate_directions, desc="Evaluating candidates")):
            print(f"\n=== COSMIC DEBUG: Starting candidate {dir_idx+1}/{len(candidate_directions)} ===")
            
            # Test each factor for this direction
            for factor in factors_to_test:
                # Compute COSMIC scores for this direction and factor across all validation datasets
                cosmic_scores = []
                    
                if neg_validation_datasets is not None:
                    # We have explicit positive and negative datasets
                    # Compute COSMIC score using the aggregated pos/neg prompts from all datasets
                    try:
                        cosmic_score, last_debug = self._compute_cosmic_score(
                            model, direction, pos_prompts, neg_prompts, selected_layers, factor, batch_size, pos_validation_datasets
                        )
                        cosmic_scores.append(cosmic_score)
                    except Exception as e:
                        raise ValueError(f"Error computing COSMIC score with explicit pos/neg datasets: {e}")
                else:
                    # Original behavior - generate answer-conditioned prompts for each dataset
                    for vds in pos_validation_datasets:
                        try:
                            # Generate prompts for this specific dataset
                            vds_pos_prompts, vds_neg_prompts = self._generate_answer_conditioned_prompts([vds])
                                
                            # Compute COSMIC score for this dataset with the current factor
                            cosmic_score, last_debug = self._compute_cosmic_score(
                                model, direction, vds_pos_prompts, vds_neg_prompts, selected_layers, factor, batch_size, [vds]
                            )
                            cosmic_scores.append(cosmic_score)
                                
                        except Exception as e:
                            raise ValueError(f"Error computing COSMIC score for dataset {vds.name}: {e}")
                    
                # Aggregate COSMIC scores across all validation datasets
                total_cosmic_score = sum(cosmic_scores)
                    
                # Use designated application location for KL divergence checking
                # The positive applier's application location represents the final intervention we'll use
                app_loc = self._resolve_application_location_for_direction(direction, 'positive')
                kl_div, passes_kl = self._compute_kl_divergence_with_threshold(model, direction, self._create_applier_kwargs(factor), app_loc)
                    
                # Record evaluation results
                record = {
                    "direction_idx": dir_idx,
                    "factor": factor,
                    "total_cosmic_score": total_cosmic_score,
                    "individual_cosmic_scores": cosmic_scores,
                    "kl_divergence": kl_div,
                    "passes_kl_threshold": passes_kl,
                    "layer": direction.loc.layer,
                    "component": direction.loc.component,
                    "attr": direction.loc.attr,
                    "pos": direction.loc.pos,
                    "refuse_score": last_debug.get("refuse_score") if isinstance(last_debug, dict) else None,
                    "comply_score": last_debug.get("comply_score") if isinstance(last_debug, dict) else None
                }
                # Add direction parameters if available
                if dir_idx < len(direction_params):
                    record.update(direction_params[dir_idx])
                    
                evaluation_records.append(record)

                if passes_kl:
                    if total_cosmic_score > best_score:
                        best_score = total_cosmic_score
                        best_direction = direction
                        best_factor = factor
                        best_direction.val_score = total_cosmic_score
                        best_direction.cosmic_scores = cosmic_scores
                else:
                    logging.debug(f"Direction {direction.loc} with factor {factor} failed KL threshold")
                    
        if best_direction is None:
            raise ValueError("No valid directions found after COSMIC evaluation")
        
        logging.info(f"Selected direction with total COSMIC score: {best_direction.val_score} (individual scores: {best_direction.cosmic_scores}) and factor: {best_factor}")
        return best_direction, best_factor, evaluation_records
    
    def _is_refusal_concept(self, validation_datasets: List[FormattedDataset]) -> bool:
        """Determine if the validation datasets represent refusal concepts.
        
        Args:
            validation_datasets: List of validation datasets to check.
            
        Returns:
            True if this is a refusal concept, False otherwise.
        """
        from utils.enums import Concept
        
        if not validation_datasets:
            return True  # Default to refusal behavior
            
        # Check the concept of the first dataset (assuming all datasets in a run have the same concept)
        first_dataset = validation_datasets[0]
        concept = getattr(first_dataset, 'concept', None)
        
        # Refusal concepts: REFUSAL and similar
        refusal_concepts = {Concept.REFUSAL}
        
        if concept in refusal_concepts:
            return True
        
        # Also check by dataset name patterns for additional detection
        dataset_name = getattr(first_dataset, 'name', '').lower()
        if any(name in dataset_name for name in ['advbench', 'jailbreak', 'salad']):
            return True
            
        return False

    def _compute_cosmic_score(self,
                            model: InterventionLLM,
                            direction: CandidateDirection,
                            pos_prompts: List[str],
                            neg_prompts: List[str],
                            selected_layers: List[int],
                            factor: float = 1.0,
                            batch_size: int = 1,
                            validation_datasets: List[FormattedDataset] = None) -> Tuple[float, Dict[str, Any]]:
        """Compute COSMIC score for a candidate direction.
        
        The COSMIC score is based on cosine similarity between:
        1. Positive prompts with negative intervention vs negative prompts (refuse score)
        2. Negative prompts with positive intervention vs positive prompts (comply score)
        
        Focus on last token activations only from the selected layers.
        
        Args:
            model: The model to evaluate on.
            direction: The candidate direction to evaluate.
            pos_prompts: List of positive prompts.
            neg_prompts: List of negative prompts.
            selected_layers: List of selected layer indices.
            factor: The factor to use for interventions.
            
        Returns:
            The COSMIC score for this direction.
        """
        # Note: We apply intervention at direction.loc.layer but extract from selected_layers
        # Transformer propagation means intervention affects subsequent layers
        
        # Extract baseline activations from selected layers (last token only)
        pos_baseline = self._extract_mean_activations_from_selected_layers(model, pos_prompts, selected_layers, batch_size)
        neg_baseline = self._extract_mean_activations_from_selected_layers(model, neg_prompts, selected_layers, batch_size)
        
        # Determine if this is a refusal concept to decide applier assignment
        is_refusal = self._is_refusal_concept(validation_datasets) if validation_datasets else True
        
        if is_refusal:
            # REFUSAL CONCEPT: Use current logic (DirectionalAblation on pos/harmful, ActAdd on neg/harmless)
            # Apply DirectionalAblation to harmful prompts (to match official COSMIC refusal score)
            # Use positive_applier (DirectionalAblation) on pos_prompts (harmful)
            pos_reverse = (self.positive_applier.default_intervention_type == InterventionType.NEGATIVE)
            pos_with_intervention = self._extract_activations_with_intervention_from_selected_layers(
                model, pos_prompts, direction, selected_layers, coefficient=factor, reverse=pos_reverse, applier=self.positive_applier, intervention_type='positive', batch_size=batch_size
            )
            
            # Apply ActAdd to harmless prompts (to match official COSMIC steering score)  
            # Use negative_applier (ActAdd) on neg_prompts (harmless)
            neg_reverse = (self.negative_applier.default_intervention_type == InterventionType.POSITIVE)
            neg_with_intervention = self._extract_activations_with_intervention_from_selected_layers(
                model, neg_prompts, direction, selected_layers, coefficient=factor, reverse=neg_reverse, applier=self.negative_applier, intervention_type='negative', batch_size=batch_size
            )
        else:
            # NON-REFUSAL CONCEPT (e.g., BIAS): Reverse the applier assignments
            # Apply DirectionalAblation to neg_prompts (biased prompts)
            # Use positive_applier (DirectionalAblation) on neg_prompts
            neg_reverse = (self.positive_applier.default_intervention_type == InterventionType.NEGATIVE)
            neg_with_intervention = self._extract_activations_with_intervention_from_selected_layers(
                model, neg_prompts, direction, selected_layers, coefficient=factor, reverse=neg_reverse, applier=self.positive_applier, intervention_type='positive', batch_size=batch_size
            )
            
            # Apply ActAdd to pos_prompts (unbiased prompts)
            # Use negative_applier (ActAdd) on pos_prompts  
            pos_reverse = (self.negative_applier.default_intervention_type == InterventionType.POSITIVE)
            pos_with_intervention = self._extract_activations_with_intervention_from_selected_layers(
                model, pos_prompts, direction, selected_layers, coefficient=factor, reverse=pos_reverse, applier=self.negative_applier, intervention_type='negative', batch_size=batch_size
            )
        
        # Compute cosine similarities based on concept type
        if is_refusal:
            # REFUSAL CONCEPT: Current logic
            # Comply score: Harmful prompts + DirectionalAblation should look like harmless prompts
            # This matches the working controlled test logic
            refuse_score = F.cosine_similarity(
                pos_with_intervention, 
                neg_baseline.to(pos_with_intervention.device),
                dim=0  # Cosine similarity along the feature dimension
            ).item()
            
            # Refuse score: Harmless prompts + ActAdd should look like harmless prompts 
            # This matches the working controlled test logic
            comply_score = F.cosine_similarity(
                neg_with_intervention,
                pos_baseline.to(neg_with_intervention.device),
                dim=0  # Cosine similarity along the feature dimension
            ).item()
        else:
            # NON-REFUSAL CONCEPT (e.g., BIAS): Reverse the scoring logic
            # Bias reduction score: Biased prompts + DirectionalAblation should look like unbiased prompts
            refuse_score = F.cosine_similarity(
                neg_with_intervention,
                pos_baseline.to(neg_with_intervention.device).squeeze(0),
                dim=0  # Cosine similarity along the feature dimension
            ).item()
            
            # Steering score: Unbiased prompts + ActAdd should look like biased prompts
            comply_score = F.cosine_similarity(
                pos_with_intervention,
                neg_baseline.to(pos_with_intervention.device).squeeze(0),
                dim=0  # Cosine similarity along the feature dimension
            ).item()
        
        # Final COSMIC score
        cosmic_score = refuse_score + comply_score
        
        # Debug print to compare with official COSMIC
        concept_type = "REFUSAL" if is_refusal else "NON-REFUSAL"
        print(f"HOOKS COSMIC SCORES ({concept_type}): Refusal={refuse_score:.6f}, Steering={comply_score:.6f}, Total={cosmic_score:.6f}")
        
        debug = {
            'refuse_score': refuse_score,
            'comply_score': comply_score,
            'pos_baseline': pos_baseline.detach().cpu().numpy().tolist() if hasattr(pos_baseline, 'detach') else None,
            'neg_baseline': neg_baseline.detach().cpu().numpy().tolist() if hasattr(neg_baseline, 'detach') else None,
            'pos_with_intervention': (pos_with_intervention.detach().cpu().numpy().tolist() if 'pos_with_intervention' in locals() else None),
            'neg_with_intervention': (neg_with_intervention.detach().cpu().numpy().tolist() if 'neg_with_intervention' in locals() else None)
        }
        return cosmic_score, debug
    
    def _extract_mean_activations_from_selected_layers(self, 
                                                     model: InterventionLLM, 
                                                     prompts: List[str], 
                                                     layers: List[int],
                                                     batch_size: int = 16) -> torch.Tensor:
        """Extract mean activations from selected layers (last token only).
        
        Args:
            model: The model to extract activations from.
            prompts: List of prompts to process.
            layers: List of selected layer indices.
            batch_size: Batch size for activation collection.
            
        Returns:
            Mean activation vector computed across all prompts and layers (matching official COSMIC).
        """
        # Collect activations in format matching official COSMIC: [num_prompts, num_layers, hidden_size]
        raw_outputs = []
        
        for layer_idx in layers:
            layer_activations = self._extract_activations_for_layer(model, prompts, layer_idx, batch_size)
            if layer_activations:
                # Stack activations from all prompts for this layer: [num_prompts, hidden_size]
                layer_stack = torch.stack(layer_activations)
                # Ensure we have the right shape by squeezing any extra dimensions
                if layer_stack.dim() > 2:
                    layer_stack = layer_stack.squeeze()
                raw_outputs.append(layer_stack)
        
        if not raw_outputs:
            raise ValueError("No activations extracted from selected layers")
        
        # Stack to get shape [num_layers, num_prompts, hidden_size]
        raw_outputs = torch.stack(raw_outputs)
        # Transpose to get shape [num_prompts, num_layers, hidden_size] to match official COSMIC
        raw_outputs = raw_outputs.transpose(0, 1)
        
        # Match official COSMIC: mean across both prompts (dim=0) and layers (dim=1)
        mean_output_directions = raw_outputs.mean(dim=(0, 1))  # Shape: [hidden_size]
        
        # Normalize like official COSMIC
        norm = torch.norm(mean_output_directions, dim=-1, keepdim=True) + 1e-8
        normalized_vector = mean_output_directions / norm
        
        return normalized_vector
    
    def _extract_activations_with_intervention_from_selected_layers(self,
                                                                  model: InterventionLLM,
                                                                  prompts: List[str],
                                                                  direction: CandidateDirection,
                                                                  selected_layers: List[int],
                                                                  coefficient: float,
                                                                  reverse: bool=False,
                                                                  applier: DirectionApplier = None,
                                                                  intervention_type: str = 'positive',
                                                                  batch_size: int = 1) -> torch.Tensor:
        """Extract activations with intervention applied, focusing on selected layers.
        
        Uses the enhanced model.generate() method with activation extraction to get
        activations from selected layers during the steered forward pass.
        
        Args:
            model: The model to extract activations from.
            prompts: List of prompts to process.
            direction: The direction to apply as intervention.
            selected_layers: List of selected layer indices to extract from.
            coefficient: Intervention coefficient.
            reverse: Whether to reverse the direction (for negative intervention).
            applier: The applier to use for the intervention (defaults to self.applier).
            
        Returns:
            Mean activations with intervention applied from selected layers.
        """
        # Use the provided applier or fall back to self.applier
        intervention_applier = applier if applier is not None else self.applier
        
        all_activations = []
        
        # Use the centralized application location resolution logic
        app_loc = self._resolve_application_location_for_direction(direction, intervention_type)
        
        # IMPORTANT: For COSMIC scoring we must compare vectors built from the
        # hidden state outputs at the evaluated layers and the last token only.
        # The official COSMIC implementation uses outputs.hidden_states[layer][:, -1, :]
        # when building both the baseline vector and the intervention vector.
        # Therefore, regardless of how the direction was generated (e.g., at
        # attr="input"), we extract activations here at attr="output", component=None,
        # and pos=-1 to exactly mirror COSMIC.
        extraction_locations = []
        for layer_idx in selected_layers:
            extract_loc = ActivationLocation(
                layer=layer_idx,
                attr="output",  # Always use output to match official COSMIC (bc we use output for layers to evaluate as well). direction.loc.attr else.
                component=direction.loc.component,
                pos=direction.loc.pos
            )
            extraction_locations.append(extract_loc)
        
        # Process prompts in batches with intervention applied using enhanced model.generate()
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
                
            # try:
            if True:
                # Use enhanced generate() method with activation extraction (handles both single and batch)
                _, _, extracted_activations_batch, _ = model.generate(
                    prompt=batch_prompts if len(batch_prompts) > 1 else batch_prompts[0],
                    direction=direction,
                    applier=intervention_applier,
                    application_location=app_loc,
                    applier_kwargs=self._create_applier_kwargs(coefficient, reverse=reverse),
                    max_new_tokens=1,  # Minimal generation just to trigger steered forward pass
                    temperature=0.0,
                    top_p=1,
                    extract_activations=True,
                    extraction_locations=extraction_locations,
                    smart_generation_prompt=True  # Enable smart generation prompt to avoid tokenization issues
                )
                
                # Ensure extracted_activations_batch is a list for consistent processing
                if not isinstance(extracted_activations_batch, list):
                    extracted_activations_batch = [extracted_activations_batch]
                    
                # Process each prompt's extracted activations
                for extracted_activations in extracted_activations_batch:
                    # Store activations by layer for this prompt
                    prompt_layer_activations = []
                    for extract_loc in extraction_locations:
                        loc_str = model.get_location_str(extract_loc)
                        # Use string representation as key to match what's stored in extracted_activations
                        key = str(loc_str) if isinstance(loc_str, list) else loc_str
                        if key in extracted_activations and extracted_activations[key] is not None:
                            assert extracted_activations[key].dim() == 1, f"Expected 1D activation tensor, got {extracted_activations[key].shape}"
                            prompt_layer_activations.append(extracted_activations[key])  # Shape: [hidden_size]
                        
                    if prompt_layer_activations:
                        # Stack to get [num_layers, hidden_size] for this prompt
                        all_activations.append(torch.stack(prompt_layer_activations))
                            
        # Compute mean activation using official COSMIC approach
        if not all_activations:
            raise ValueError("No activations extracted")
            
        # Stack to get [num_prompts, num_layers, hidden_size] to match official COSMIC format
        raw_outputs = torch.stack(all_activations)
        
        # Mean across both prompts (dim=0) and layers (dim=1) like official COSMIC
        mean_output_directions = raw_outputs.mean(dim=(0, 1))  # Shape: [hidden_size]
        
        # Normalize like official COSMIC
        norm = torch.norm(mean_output_directions, dim=-1, keepdim=True) + 1e-8
        normalized_vector = mean_output_directions / norm
        
        return normalized_vector
    
    def _save_results_to_csv(self, layer_similarity_records: List[Dict[str, Any]], evaluation_records: List[Dict[str, Any]]):
        """Save layer similarity and evaluation results to CSV files.
        
        Args:
            layer_similarity_records: Records from layer cosine similarity analysis.
            evaluation_records: Records from COSMIC direction evaluation.
        """
        # Create results directory
        os.makedirs(os.path.dirname(self.result_path) or ".", exist_ok=True)
        
        # Save layer similarity results
        if layer_similarity_records:
            layer_df = pd.DataFrame(layer_similarity_records)
            layer_csv_path = self.result_path.replace('.csv', '_layer_similarities.csv')
            layer_df.to_csv(layer_csv_path, index=False)
            logging.info(f"Saved layer similarity results to {layer_csv_path}")
        
        # Save evaluation results
        if evaluation_records:
            # Flatten individual cosmic scores for CSV
            flattened_records = []
            for record in evaluation_records:
                base_record = {k: v for k, v in record.items() if k != 'individual_cosmic_scores'}
                if 'individual_cosmic_scores' in record:
                    for i, score in enumerate(record['individual_cosmic_scores']):
                        base_record[f'cosmic_score_dataset_{i}'] = score
                flattened_records.append(base_record)
            
            eval_df = pd.DataFrame(flattened_records)
            eval_csv_path = self.result_path.replace('.csv', '_evaluations.csv')
            eval_df.to_csv(eval_csv_path, index=False)
            logging.info(f"Saved evaluation results to {eval_csv_path}")
        
        # Save combined summary (similar to grid_search.py format)
        summary_records = []
        for record in evaluation_records:
            summary_record = {
                "direction_idx": record["direction_idx"],
                "factor": record["factor"],  # Include factor like grid_search.py
                "score": record["total_cosmic_score"],  # Use 'score' column name like grid_search.py
                "passes_kl_threshold": record["passes_kl_threshold"],
                "layer": record["layer"],
                "component": record.get("component"),
                "attr": record.get("attr"),
                "pos": record.get("pos")
            }
            # Add any additional direction parameters (like grid_search.py does with rec.update(params))
            for key, value in record.items():
                if key not in summary_record and key != "individual_cosmic_scores":
                    summary_record[key] = value
            
            summary_records.append(summary_record)
        
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_df.to_csv(self.result_path, index=False)
            logging.info(f"Saved summary results to {self.result_path}")
