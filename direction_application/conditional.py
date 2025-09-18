"""The direction is applied conditionally (based on specific tokens or inputs) during inference."""

import torch
from typing import Any, Optional, Literal
from utils.steering_utils import CandidateDirection, ActivationLocation
from direction_application.base import DirectionApplier, InterventionType


class ConditionalApplier(DirectionApplier):
    """
    Conditional applier that wraps another applier and applies it only when a condition is met.
    
    Based on activation-steering leash_layer implementation:
    - Checks condition using tanh-transformed cosine similarity on first forward call only
    - Uses condition_direction to create condition_projector for condition checking
    - Once condition is determined, always applies or never applies for rest of generation
    """
    
    def __init__(
        self,
        base_applier: DirectionApplier,
        condition_direction: torch.Tensor,
        condition_loc: ActivationLocation,
        condition_threshold: float = 0.0,
        condition_comparator: Literal["greater", "less"] = "greater", 
        condition_threshold_comparison_mode: Literal["mean", "last"] = "mean",
    ):
        """
        Initialize the conditional applier.
        
        Args:
            base_applier: The base applier to use when condition is met
            condition_direction: Direction tensor for condition checking
            condition_layer: Layer at which to check the condition
            condition_threshold: Threshold for condition activation  
            condition_comparator: Whether condition is met when similarity is "greater" or "less" than threshold
            condition_threshold_comparison_mode: How to aggregate tokens for condition checking ("mean" or "last")
        """
        self.base_applier = base_applier
        self.condition_direction = condition_direction
        self.condition_loc = condition_loc
        self.condition_threshold = condition_threshold
        self.condition_comparator = condition_comparator
        self.condition_threshold_comparison_mode = condition_threshold_comparison_mode
        
        # No need to track state - condition will be checked based on parameters
        
    @property
    def default_intervention_type(self) -> InterventionType:
        """Inherits intervention type from base applier."""
        return self.base_applier.default_intervention_type
    
    def reset_state(self):
        """Reset the applier state for a new generation."""
        # No state to reset - condition checking is stateless
        pass
    
    @staticmethod
    def create_condition_projector(condition_direction: torch.Tensor) -> torch.Tensor:
        """
        Create condition projector from condition direction.
        
        Args:
            condition_direction: The condition direction vector
            model_dtype: Model's data type
            model_device: Model's device
            
        Returns:
            Condition projector matrix
        """
        condition_tensor = condition_direction.clone().detach()
        condition_projector = torch.ger(condition_tensor, condition_tensor) / torch.dot(condition_tensor, condition_tensor)
        return condition_projector

    @staticmethod
    def project_hidden_state(hidden_state: torch.Tensor, condition_direction: torch.Tensor) -> torch.Tensor:
        # Create condition projector
        condition_projector = ConditionalApplier.create_condition_projector(condition_direction)
        
        # Apply tanh transformation to projected hidden state
        projected_hidden_state = torch.tanh(torch.matmul(condition_projector, hidden_state))
        
        return projected_hidden_state
    
    def check_condition(self, hidden_state: torch.Tensor) -> bool:
        """
        Check if condition is met using tanh similarity with condition projector.
        
        Args:
            hidden_state: Input hidden state tensor [batch, seq_len, d_model] or [seq_len, d_model]
            condition_direction: Condition direction vector to create projector from
            model_dtype: Model's data type
            model_device: Model's device
            
        Returns:
            Boolean indicating whether condition is met
        """
        # Handle batch dimension - take first sample if batched
        if hidden_state.dim() == 3:  # [batch, seq_len, d_model]
            hidden_state = hidden_state[0]  # [seq_len, d_model]
            
        # Aggregate tokens based on condition_threshold_comparison_mode
        if self.condition_threshold_comparison_mode == "mean":
            hidden_state = hidden_state.mean(dim=0)  # [d_model] 
        elif self.condition_threshold_comparison_mode == "last":
            hidden_state = hidden_state[-1, :]  # [d_model]
        
        # Create condition projector and project hidden state
        projected_hidden_state = self.project_hidden_state(hidden_state, self.condition_direction)
        
        # Compute cosine similarity  
        condition_similarity = self.compute_similarity(hidden_state, projected_hidden_state).item()
        
        # Check threshold based on comparator
        if self.condition_comparator == "greater":
            condition_met = condition_similarity > self.condition_threshold
        elif self.condition_comparator == "less": 
            condition_met = condition_similarity < self.condition_threshold
        else:
            raise ValueError(f"Unsupported condition_comparator: {self.condition_comparator}")
            
        return condition_met
    
    @staticmethod
    def compute_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two tensors.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Cosine similarity as a scalar tensor
        """
        return torch.dot(x.flatten(), y.flatten()) / (torch.norm(x) * torch.norm(y))
    
    def create_batch_mask(self, conditional_activation: torch.Tensor) -> torch.Tensor:
        """
        Create batch mask by checking condition for each sequence in the batch.
        
        Args:
            conditional_activation: Activation tensor for condition checking [batch_size, seq_len, d_model]
            
        Returns:
            Boolean tensor indicating which sequences meet the condition [batch_size]
        """
        batch_size = conditional_activation.shape[0]
        batch_mask = torch.zeros(batch_size, dtype=torch.bool, device=conditional_activation.device)
        
        for i in range(batch_size):
            # Check condition for each sequence individually
            condition_met = self.check_condition(conditional_activation[i:i+1])
            batch_mask[i] = condition_met
            
        return batch_mask

    def apply(
        self, 
        activation: torch.Tensor, 
        direction: CandidateDirection,
        tracer=None,
        conditional_activation: Optional[torch.Tensor] = None,
        batch_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply the base applier conditionally based on condition check.
        
        Args:
            activation: The model's activations (already sliced to positions)
            direction: The candidate direction for behavior application 
            tracer: NNsight tracer
            conditional_activation: The activation at the condition location. Must be passed so we can check the condition at the right place.
            batch_mask: Boolean tensor indicating which sequences in batch should have intervention applied [batch_size]
            **kwargs: Additional arguments passed to base applier
            
        Returns:
            Modified activations with base applier applied if condition is met
        """ 
        # Handle batch mask for conditional application
        if batch_mask is not None:
            # Vectorized approach: apply intervention to full batch, then selectively use results
            steered_activation = self.base_applier.apply(activation, direction, tracer=tracer, **kwargs)
            
            # Use torch.where to vectorize the selection
            # Expand batch_mask to match activation dimensions: [batch_size] -> [batch_size, seq_len, d_model]
            mask_expanded = batch_mask.view(-1, 1, 1).expand_as(activation)
            result = torch.where(mask_expanded, steered_activation, activation)
            
            return result
        else:
            raise ValueError("batch_mask must be provided to apply conditional intervention.")
