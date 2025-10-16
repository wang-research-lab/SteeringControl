"""Abstract base class for direction generation component. Use InterventionLLM's `collect_activations` method to collect activations for the given location."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict
import torch
import logging
from itertools import product
from tqdm.auto import tqdm
from utils.steering_utils import ActivationLocation, ActivationResults, CandidateDirection
from utils.intervention_llm import InterventionLLM
from data.steering_data import SteeringTrainData

class DirectionGenerator(ABC):
    """Abstract base class for generating candidate directions in latent space."""

    @abstractmethod
    def generate_one_direction(self, model: InterventionLLM, activations: ActivationResults, **kwargs) -> CandidateDirection:
        """Generate candidate directions.
        Args:
            model: The model to steer.
            activations: The activation results from the model used to formulate the directions.
        Returns:
            CandidateDirections: A data structure containing the generated directions.
        """
        pass

    def generate_reference_activation(self, activations: ActivationResults) -> torch.Tensor:
        """Generate reference activation for ACE (Affine Concept Editing).
        
        Default implementation uses the mean of negative activations (harmless_mean).
        Subclasses can override this method to implement more sophisticated reference generation.
        
        Args:
            activations: The activation results containing pos/neg activations.
        Returns:
            torch.Tensor: Reference activation for affine steering.
        """
        return torch.cat(activations.neg, dim=0).mean(dim=0)


    def get_unit_norm(self, direction: torch.Tensor) -> torch.Tensor:
        """Normalize the direction to unit norm following [AxBench](https://github.com/stanfordnlp/axbench/blob/main/axbench/utils/model_utils.py#L63).
        Args:
            direction: The candidate direction to normalize.
        Returns:
            torch.Tensor: The normalized direction.
        """
        assert direction.ndim == 1, f"Direction must be a vector, got shape {direction.shape}"
        norm = torch.norm(direction)
        eps = torch.finfo(direction.dtype).eps
        return direction / (norm + eps)


    def save_direction(
        self,
        model: InterventionLLM,
        activations: ActivationResults,
        direction: CandidateDirection,
        **kwargs
    ) -> None:
        """Save the generated direction to the model's storage.
        We may want this class to always have a directory variable to save in.
        Args:
            model: The model to steer.
            activations: The activation results from the model used to formulate the directions.
            direction: The candidate direction to save.
        """
        # This method can be overridden by subclasses if they want to implement custom saving logic.
        pass

def generate_candidate_directions(model: Any, generator: Any, generator_param_grid: Any, train_data: Any, batch_size: int = 16, cumulative_layers: bool = False, **kwargs) -> List[Any]:
    """Generate candidate directions using the generator and param grid.
        
    Args:
        model: The model to generate directions for.
        train_data: Training data for direction generation.
        batch_size: Batch size for activation collection.
        **kwargs: Additional parameters passed through.
            
    Returns:
        List of candidate directions.
    """
    from utils.steering_utils import ActivationLocation
        
    # Build the param grid
    param_names = list(generator_param_grid.keys())
    param_values = [generator_param_grid[n] for n in param_names]
        
    # If we're doing cumulative layers, expand "layer" so we generate
    # *every* layer up to the max in the user‐provided grid.
    orig_layers = None
    if cumulative_layers:
        if 'layer' not in param_names:
            raise ValueError("`layer` must be in generator_param_grid when cumulative_layers=True")
        li = param_names.index('layer')
        orig_layers = list(param_values[li])                # copy of [6,8,...]
        min_layer = min(orig_layers)  # where to start the cumulative layers from
        max_layer = max(orig_layers)
        param_values[li] = list(range(min_layer, max_layer + 1))       # now [min_layer, min_layer+1,...,max_layer]. Not starting at 0 because we assume the concept starts somewhere at ~25% at the minimum, so may cause problems if we start earlier.

    # prepare product
    total_combos = 1
    for vals in param_values:
        total_combos *= len(vals)

    directions: List[Any] = []
    direction_params: List[Dict[str,Any]] = []

    # Generate one direction per combo
    for combo in tqdm(product(*param_values), total=total_combos, desc="Generating directions"):
        params = dict(zip(param_names, combo))
        loc_kwargs   = {k: params[k] for k in ('layer','component','attr','pos') if k in params}
        gen_kwargs   = {k:v for k,v in params.items() if k not in loc_kwargs}
        loc          = ActivationLocation(**loc_kwargs)
        activations  = model.collect_activations(loc, train_data, batch_size=batch_size)
        direction    = generator.generate_one_direction(model, activations, **gen_kwargs)
        directions.append(direction)
        direction_params.append(params)

    logging.info(f"Generated {len(directions)} candidate directions")
        
    return directions, direction_params, orig_layers
