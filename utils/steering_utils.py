from typing import Any, Dict, List, Optional, Union, Tuple, Literal
from dataclasses import dataclass
import torch
import numpy as np
import os

@dataclass
class ActivationLocation:
    """Location of activations to extract in direction generation step.
    
    Assuming we have two components: attention, MLP. We can get either the input or output of each component. We also can specify a component as None to get input/output of the specified layer.

    """
    layer: int
    attr: Literal["input", "output"]
    """Whether to get the input or output of the component."""
    component: Optional[Literal["attn", "mlp"]] = None
    """Which component of the model to target."""
    pos: Optional[Union[int, slice, Tuple[Any, ...]]] = None
    """Position of token in the sequence, if applicable."""

    def __post_init__(self):
        if self.attr not in ["input", "output"]:
            raise ValueError("attr must be either 'input' or 'output'.")
        if self.component is not None and self.component not in ["attn", "mlp"]:
            raise ValueError("component must be either 'attn' or 'mlp'.")
        if self.pos is not None and not isinstance(self.pos, (int, slice, tuple)):
            raise ValueError("pos must be either int, slice or tuple. You may have accidentally passed an InterventionPositions enum, which is meant for the DirectionApplier class during inference, not for direction generation.")

    def __eq__(self, other):
        if not isinstance(other, ActivationLocation):
            return NotImplemented
        return (self.layer == other.layer and
                self.attr == other.attr and
                self.component == other.component and
                self.pos == other.pos)
    
    def __lt__(self, other):
        """Define ordering for ActivationLocation objects.
        
        Ordering priority:
        1. Layer (ascending)
        2. Combined component/attribute: input (none) < attn < mlp < output (none)
        """
        if not isinstance(other, ActivationLocation):
            return NotImplemented
        
        # First sort by layer
        if self.layer != other.layer:
            return self.layer < other.layer
        
        # Then by combined component/attribute: input (none) < attn < mlp < output (none)
        def combined_order(comp, attr):
            if comp is None and attr == "input":
                return 0  # input (none)
            elif comp == "attn":
                return 1  # attn
            elif comp == "mlp":
                return 2  # mlp
            elif comp is None and attr == "output":
                return 3  # output (none)
            else:
                raise ValueError("Invalid component/attribute combination.")
        
        self_order = combined_order(self.component, self.attr)
        other_order = combined_order(other.component, other.attr)
        
        return self_order < other_order

@dataclass
class ActivationResults:
    """
    Dataclass to hold the activation results for different input labels.
    """
    loc: ActivationLocation
    dataset_metadata: Dict[str, Any]
    pos: List[torch.Tensor]
    neg: List[torch.Tensor]
    neutral: Optional[List[torch.Tensor]] = None

    @classmethod
    def from_dict(cls, loc: ActivationLocation, dataset_metadata: Dict[str, Any], results_dict: Dict[str, List[torch.Tensor]]):
        """
        Creates an ActivationResults instance from a dictionary.
        """
        return cls(
            loc=loc,
            dataset_metadata=dataset_metadata,
            pos=results_dict.get("pos", []),
            neg=results_dict.get("neg", []),
            neutral=results_dict.get("neutral"),
        )

    def free(self):
        """
        Frees the memory of the activations. Always call after generating a direction to avoid storing all activations in memory.
        """
        for tensor_list in [self.pos, self.neg, self.neutral]:
            if tensor_list is not None:
                for tensor in tensor_list:
                    del tensor
        torch.cuda.empty_cache()

    @classmethod
    def combine(cls, results_list: List['ActivationResults']) -> 'ActivationResults':
        if not results_list:
            raise ValueError("Input list cannot be empty.")

        first = results_list[0]
        loc = first.loc
        combined_metadata = []
        combined_pos = []
        combined_neg = []
        combined_neutral = [] if any(r.neutral is not None for r in results_list) else None

        for results in results_list:
            if results.loc != loc:
                raise ValueError("Cannot combine results with different locations.")

            combined_metadata.append(results.dataset_metadata)
            combined_pos.extend(results.pos)
            combined_neg.extend(results.neg)

            if combined_neutral is not None and results.neutral is not None:
                 combined_neutral.extend(results.neutral)

        return cls(
            loc=loc,
            dataset_metadata=combined_metadata,
            pos=combined_pos,
            neg=combined_neg,
            neutral=combined_neutral
        )

@dataclass
class CandidateDirection:
    """Data structure to hold candidate direction generated by a DirectionGenerator."""
    direction: torch.Tensor
    dataset_metadata: Dict[str, Any]
    model_name: str
    loc: ActivationLocation
    val_score: float = 0.0
    """Accuracy of the direction on the validation set. Default is 0.0, meaning not computed yet."""
    reference_activation: Optional[torch.Tensor] = None
    """Reference activation for affine steering (ACE). This corresponds to the harmless_mean in COSMIC."""

    def save(self, path: str):
        """Save the CandidateDirection to the given file path using torch.save."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> 'CandidateDirection':
        """Load a CandidateDirection object from the given file path."""
        return torch.load(path, weights_only=False)
