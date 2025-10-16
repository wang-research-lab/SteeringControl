"""The direction is applied unconditionally (not based on specific tokens or inputs) during inference."""

from utils.steering_utils import CandidateDirection
from direction_application.base import DirectionApplier, InterventionType
from typing import Any
from einops import einsum, rearrange

class ActAdd(DirectionApplier):
    """Applies the direction by adding it to the activations at the specified location in the CandidateDirection."""
    
    @property
    def default_intervention_type(self) -> InterventionType:
        """ActAdd promotes the direction by default (positive intervention)."""
        return InterventionType.POSITIVE

    def apply(self, activation: Any, direction: CandidateDirection, tracer, reverse: bool, coefficient: float = 1.0) -> Any:
        """
        Apply the selected direction by adding it to the activation tensor.
        Formula: h <- h + coefficient * direction (or h <- h - coefficient * direction if reverse=True)

        Args:
            activation: The model's activations to which the direction will be applied.
            direction: The candidate direction to apply.
            reverse: Whether to reverse the direction (subtract instead of add).
            coefficient: A scaling factor for the direction.
        Returns:
            torch.Tensor: The modified activations after applying the direction.
        """
        # Apply reverse logic: negate coefficient if reverse=True
        effective_coefficient = -coefficient if reverse else coefficient
        modified_activation = activation + effective_coefficient * direction.direction.to(activation.device)
        
        return modified_activation

class DirectionalAblation(DirectionApplier):
    """Applies the direction by ablation, i.e., removing the part of the activations corresponding to the direction at each point the model writes to the residual stream. To do so we calculate the projection of the activations onto the direction and subtract it from the activations.
    
    This is described by [Arditi et al.](https://arxiv.org/pdf/2406.11717) and implemented [here](https://github.com/andyrdt/refusal_direction/blob/main/pipeline/utils/hook_utils.py#L41). We follow the COSMIC implementation which built on Arditi's work.
    """

    def __init__(self, use_affine: bool = False):
        super().__init__()
        self.use_affine = use_affine
    
    @property
    def default_intervention_type(self) -> InterventionType:
        """DirectionalAblation suppresses the direction by default (negative intervention)."""
        return InterventionType.NEGATIVE
    def apply(self, activation: Any, direction: CandidateDirection, tracer, reverse: bool, coefficient: float = None) -> Any:
        """
        Apply the directional ablation by removing the part of the activations corresponding to the direction.
        Formula: h <- h - proj(h, dir) + proj(reference, dir) * use_affine
            - where proj(h, dir) is the projection of h onto dir.
            - proj(reference, dir) is the projection of the reference activation onto dir to mimic ACE, if use_affine is True.
        """
        # we don't use a coefficient by default with Arditi's directional ablation.
        dir = direction.direction
        # Apply reverse logic: negate direction if reverse=True  
        if reverse:
            dir = -dir
        dir = dir.to(activation)
        # activation: (B, S, D)
        # dir: (D,)
        # Step 1: compute dot products along last dim
        # einsum -> dot: (B, S)
        dot = einsum(activation, dir, 'b s d, d -> b s')

        # Step 2: reshape to enable broadcasting
        # make dir shape: (1, 1, D)
        dir_broadcast = rearrange(dir, 'd -> 1 1 d')

        # Step 3: multiply projection
        projection = rearrange(dot, 'b s -> b s 1') * dir_broadcast  # shape: (B, S, D)

        # Step 4: subtract to get orthogonal component
        modified_activation = activation - projection  # shape: (B, S, D)

        # Step 5: if use_affine, add the projection of the reference activation onto the direction
        if self.use_affine:
            if direction.reference_activation is None:
                raise ValueError("Reference activation is required for affine steering.")
            reference = direction.reference_activation.to(activation.device)
            reference_projection = einsum(reference, dir, 'd, d ->') * dir_broadcast  # scalar * (1, 1, D)
            modified_activation += reference_projection
            
        return modified_activation
