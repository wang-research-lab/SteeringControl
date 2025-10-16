"""
Utilities for applying interventions with proper position and masking logic.
Handles POST_INSTRUCTION positions and padding token exclusion.
"""

import torch
from typing import Union, List, Optional, Dict, Any
from direction_application.base import InterventionPositions
from utils.steering_utils import CandidateDirection
from direction_application.base import DirectionApplier


def create_intervention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_spec: Union[List[int], InterventionPositions],
    first_eoi_tok_indices: Optional[List[Optional[int]]] = None
) -> torch.Tensor:
    """
    Create a boolean mask for where interventions should be applied.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len] 
        pos_spec: Position specification for interventions
        first_eoi_tok_indices: Pre-computed EOI token indices (for POST_INSTRUCTION). These say for each item in the batch, where the first token after the prompt ends but before the post-instruction tokens (which are added with `add_generation_prompt=True`) starts. E.g., for Qwen2.5 the first_eoi_tok index should corresond to the first token in this: '<|im_end|>\n<|im_start|>assistant\n'.
        
    Returns:
        Boolean mask [batch_size, seq_len] where True means apply intervention
    """
    batch_size, seq_len = input_ids.shape
    
    # Start with attention mask to exclude padding tokens
    # attention_mask is 1 for real tokens, 0 for pad tokens
    intervention_mask = attention_mask.clone().bool()  # [batch_size, seq_len]
    
    if pos_spec == InterventionPositions.ALL:
        # Apply to all non-padding tokens (already handled by attention_mask)
        pass
        
    elif pos_spec == InterventionPositions.OUTPUT_ONLY:
        # Apply only to the last token per sequence.
        # Note we use left padding convention, so the last token is the rightmost one.
        # Find the last True position in each row
        new_mask = torch.zeros_like(intervention_mask)
        new_mask[:, -1] = attention_mask[:, -1]  # Set last token if it's not padding
        intervention_mask = new_mask
        
    elif pos_spec == InterventionPositions.POST_INSTRUCTION:
        if first_eoi_tok_indices is None:
            raise ValueError("first_eoi_tok_indices required for POST_INSTRUCTION positions")
            
        # Apply to tokens after instruction end, excluding padding
        new_mask = torch.zeros_like(intervention_mask)
        for i in range(batch_size):
            nopadding_eoi_idx = first_eoi_tok_indices[i]
            if nopadding_eoi_idx is not None:
                # Set True from eoi_idx to end of valid sequence
                # Ensure we get eoi_idx after adding any padding from attention_mask,
                # since eoi tokens are computed from non-padded input_ids.
                first_nonpad_idx = torch.nonzero(attention_mask[i], as_tuple=True)[0].min().item()
                eoi_idx = first_nonpad_idx + nopadding_eoi_idx
                new_mask[i, eoi_idx:] = True
        intervention_mask = new_mask
        
    else:
        raise ValueError(f"Unsupported position specification: {pos_spec}")
    
    return intervention_mask


def apply_intervention_with_mask(
    activation: torch.Tensor,
    direction: CandidateDirection,
    applier: DirectionApplier,
    intervention_mask: torch.Tensor,
    applier_kwargs: Dict[str, Any] = {}
) -> torch.Tensor:
    """
    Apply intervention to activation using the provided mask.
    
    Args:
        activation: Input activations [batch_size, seq_len, d_model]
        direction: Direction to apply
        applier: Applier to use for intervention
        intervention_mask: Boolean mask [batch_size, seq_len] for where to apply
        applier_kwargs: Additional arguments for applier
        
    Returns:
        Modified activation with intervention applied only at masked positions
    """
    # Check if any positions need intervention
    if not intervention_mask.any():
        return activation.clone()
    
    # Apply intervention to the FULL activation tensor (preserves expected shape)
    modified_activation_full = applier.apply(
        activation, 
        direction, 
        tracer=None, 
        **applier_kwargs
    )
    
    # Use mask to selectively keep modified vs original positions
    # Where mask is True, use modified; where False, use original
    final_activation = torch.where(
        intervention_mask.unsqueeze(-1),  # Broadcast to [batch, seq, 1] 
        modified_activation_full,         # Modified values
        activation                        # Original values
    )
    
    return final_activation