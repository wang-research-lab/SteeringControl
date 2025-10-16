"""Abstract base class for direction application component."""

from abc import ABC, abstractmethod
from typing import List, Union, Literal, Any, Optional
import torch
from utils.steering_utils import CandidateDirection
from dataclasses import dataclass
from enum import Enum

class InterventionType(Enum):
    """Enum for the default intervention type of a DirectionApplier."""
    POSITIVE = "positive"  # Promotes/enhances the direction (e.g., ActAdd)
    NEGATIVE = "negative"  # Suppresses/removes the direction (e.g., DirectionalAblation)

class InterventionPositions(Enum):
    """Enum for specifying positions in the sequence where the direction will be applied."""
    OUTPUT_ONLY = "output_only"  # The first forward pass and all subsequent positions in the sequence.
    POST_INSTRUCTION = "post_instr"  # All positions including and after the first post instruction token.
    ALL = "all"  # All positions in the sequence, including pre-instruction tokens.

@dataclass
class ApplicationLocation:
    layer: List[int]
    """Location in the model where the direction will be applied. This can be a single layer or a list of layers."""
    component: Optional[Union[str, List[Any]]]
    """Component(s) of the model where the direction will be applied. Can be a string or list of strings (e.g., ['attn', 'mlp']), or None for layer input."""
    attr: Union[str, List[str]]
    """Attribute(s) of the component to which the direction will be applied. Can be 'output', 'input', or list thereof."""
    pos: Union[List[int], Literal[InterventionPositions.POST_INSTRUCTION, InterventionPositions.ALL]]
    """Positions in the sequence where the direction will be applied. This can be a single position or a list of positions, or an InterventionPositions enum."""
    component_attr_paired: bool = False
    """Whether to pair components and attrs by index (True) or take cross-product (False) when both are lists."""

    @classmethod
    def from_dict(
        cls,
        layer: List[int],
        component: Optional[Union[str, List[Any]]],
        attr: Union[str, List[str]],
        pos: Union[List[int], Literal[InterventionPositions.POST_INSTRUCTION, InterventionPositions.ALL]],
        component_attr_paired: bool = False,
    ) -> "ApplicationLocation":
        """Creates an ApplicationLocation instance from a dictionary, with optional pairing flag."""
        if isinstance(pos, str):
            if pos == "ALL":
                pos = InterventionPositions.ALL
            elif pos == "POST_INSTRUCTION":
                pos = InterventionPositions.POST_INSTRUCTION
            elif pos == "OUTPUT_ONLY":
                pos = InterventionPositions.OUTPUT_ONLY
            else:
                try:
                    pos = eval(pos)
                except:
                    raise ValueError(f"Unsupported pos value: {pos}")
        
        return cls(
            layer=layer,
            component=component,
            attr=attr,
            pos=pos,
            component_attr_paired=component_attr_paired,
        )

class DirectionApplier(ABC):
    """Abstract base class for applying a selected direction during inference. This will be a parameter for the LLM generate method that specifies how to transform the hidden states."""

    @property
    @abstractmethod
    def default_intervention_type(self) -> InterventionType:
        """Indicates whether this applier's default behavior is a positive or negative intervention.
        
        Returns:
            InterventionType.POSITIVE if the applier promotes the direction by default (e.g., ActAdd)
            InterventionType.NEGATIVE if the applier suppresses the direction by default (e.g., DirectionalAblation)
        """
        pass

    @abstractmethod
    def apply(self, activation: Any, direction: CandidateDirection, reverse: bool = False, **kwargs) -> Any:
        """Apply the selected direction during inference.
        Args:
            activation: The model's hidden states or activations to which the direction will be applied.
            direction: The candidate direction to apply.
            reverse: Whether to reverse the direction, i.e., instead of promoting the behavior we try to remove it. Used in COSMIC for dual objective.
        Returns:
            Model outputs with steering applied.
        """
        pass
