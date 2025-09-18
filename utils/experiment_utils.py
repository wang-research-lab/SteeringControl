"""
Shared utility functions for running experiments.

This module contains common functions used by run_experiment.py, run_ood_datasets.py,
and other experiment scripts to ensure consistency and avoid code duplication.
"""

import os
import yaml
import importlib
from typing import Dict, Any


def load_class(path: str):
    """Dynamically import a class from its full module path."""
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate(spec: dict):
    """Instantiate an object from a spec with `class` and `params` keys."""
    cls = load_class(spec['class'])
    params = spec.get('params', {}) or {}
    # Convert enum-like string parameters to actual Enum members (e.g., FormatMode.TRAIN)
    if 'supported_llms' not in spec['class']:
        for key, val in params.items():
            if isinstance(val, str) and '.' in val:
                enum_name, member_name = val.split('.', 1)
                try:
                    enums_mod = __import__('utils.enums', fromlist=[enum_name])
                    enum_cls = getattr(enums_mod, enum_name)
                    params[key] = getattr(enum_cls, member_name)
                except (ImportError, AttributeError):
                    pass
    return cls(**params)


def deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge two dicts: values in b override those in a.
    
    Special handling: if a[k] is a list and b[k] is a dict, merge the dict into each element of the list.
    """
    result = dict(a)
    for k, v in b.items():
        # merge nested dicts
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(a[k], v)
        # merge dict override into each element of a list of dicts
        elif k in a and isinstance(a[k], list) and isinstance(v, dict):
            # only merge into elements that are dicts
            merged_list = []
            for item in a[k]:
                if isinstance(item, dict):
                    merged_list.append(deep_merge(item, v))
                else:
                    merged_list.append(item)
            result[k] = merged_list
        else:
            # override or new key
            result[k] = v
    return result


def load_config(path: str) -> dict:
    """Recursively load a YAML config with support for 'extends'."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    # process inheritance if specified
    extends = cfg.get('extends')
    if extends:
        # resolve path relative to this file
        base_path = extends
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base config not found: {base_path}")
        # load base config recursively
        base_cfg = load_config(base_path)
        # merge, with child values overriding base
        child = {k: v for k, v in cfg.items() if k != 'extends'}
        return deep_merge(base_cfg, child)
    # no inheritance
    return cfg


def setup_applier_kwargs_from_steering_direction(applier, steering_direction: str) -> Dict[str, Any]:
    """Set up applier kwargs based on steering direction configuration.
    
    Args:
        applier: The direction applier object
        steering_direction: Either 'increase' or 'decrease'
        
    Returns:
        Dictionary of applier kwargs with appropriate 'reverse' setting
    """
    from direction_application.base import InterventionType
    
    applier_kwargs = {}
    
    if steering_direction == 'increase':
        # We want to increase the behavior - reverse if applier decreases by default
        applier_kwargs['reverse'] = (applier.default_intervention_type == InterventionType.NEGATIVE)
    elif steering_direction == 'decrease':
        # We want to decrease the behavior - reverse if applier increases by default  
        applier_kwargs['reverse'] = (applier.default_intervention_type == InterventionType.POSITIVE)
    elif steering_direction is not None:
        raise ValueError(f"steering_direction must be 'increase' or 'decrease', got: {steering_direction}")
    # If steering_direction is None, don't set reverse (backward compatibility)
    
    return applier_kwargs


def load_conditional_configuration(exp_dir: str):
    """Load conditional steering configuration from experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Tuple of (conditional_direction, conditional_metadata) or (None, None) if not found
    """
    import json
    from utils.steering_utils import CandidateDirection
    
    conditional_dir_path = os.path.join(exp_dir, 'conditional_direction.pt')
    metadata_path = os.path.join(exp_dir, 'metadata.json')
    
    if not (os.path.exists(conditional_dir_path) and os.path.exists(metadata_path)):
        return None, None
        
    # Load conditional metadata from metadata.json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    conditional_meta = metadata.get('conditional')
    if not conditional_meta:
        raise ValueError(f"conditional_direction.pt found but no conditional metadata in metadata.json")
    
    # Load conditional direction
    condition_direction = CandidateDirection.load(conditional_dir_path)
    
    return condition_direction, conditional_meta


def create_conditional_applier(base_applier, condition_direction, conditional_meta):
    """Create a ConditionalApplier from loaded configuration.
    
    Args:
        base_applier: Base direction applier
        condition_direction: Loaded conditional direction
        conditional_meta: Conditional metadata dict
        
    Returns:
        ConditionalApplier instance
    """
    from direction_application.conditional import ConditionalApplier
    
    return ConditionalApplier(
        base_applier=base_applier,
        condition_direction=condition_direction.direction,
        condition_loc=condition_direction.loc,
        condition_threshold=conditional_meta['condition_threshold'],
        condition_comparator=conditional_meta['condition_comparator'],
        condition_threshold_comparison_mode=conditional_meta['condition_threshold_comparison_mode']
    )


def load_inference_parameters(cfg: dict, concept_inference_map: dict = None, cls_path: str = None) -> dict:
    """Load inference parameters from config, with optional concept-specific overrides.
    
    Args:
        cfg: Main experiment configuration
        concept_inference_map: Optional mapping of dataset class paths to inference parameters
        cls_path: Optional dataset class path for concept-specific inference parameters
        
    Returns:
        Dictionary of inference parameters
    """
    from utils.enums import DEFAULT_MC_METHOD
    
    # Start with base inference parameters from config
    inference_params = cfg.get('inference', {}).copy()
    
    # Apply concept-specific overrides if available
    if concept_inference_map and cls_path and cls_path in concept_inference_map:
        concept_inference = concept_inference_map[cls_path]
        inference_params.update(concept_inference)
    
    # Ensure mc_evaluation_method has a default
    if 'mc_evaluation_method' not in inference_params:
        inference_params['mc_evaluation_method'] = DEFAULT_MC_METHOD
        
    return inference_params