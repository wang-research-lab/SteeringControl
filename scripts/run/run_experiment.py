"""
Command-line entry point to run a steering experiment from a YAML config.

Usage:
    python run_experiment.py configs/experiment.yaml
"""
import os
import sys
import yaml
import importlib
import asyncio

from data.steering_data import SteeringTrainData
from utils.experiment_saver import ExperimentSaver
from direction_application.base import InterventionType
from utils.enums import DEFAULT_MC_METHOD
from utils.experiment_utils import (
    instantiate,
    load_config,
    load_class,
    setup_applier_kwargs_from_steering_direction
)

# Set tokenizer parallelism to false to avoid async issues with transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def _validate_selector_config(sel_spec: dict, cfg: dict):
    """Validate selector configuration for common misconfigurations.
    
    Args:
        sel_spec: Direction selection specification from config.
        cfg: Full experiment configuration.
    """
    params = sel_spec.get('params', {})
    
    # Check for conflicting application location configurations
    has_app_locs = len(params.get('application_locations', [])) > 0
    has_pos_app_locs = len(params.get('positive_application_locations', [])) > 0  
    has_neg_app_locs = len(params.get('negative_application_locations', [])) > 0
    has_include_gen_loc = params.get('include_generation_loc', False)
    
    # Error: include_generation_loc=True with existing application_locations
    if has_include_gen_loc and has_app_locs:
        raise ValueError(
            "Configuration error: Cannot specify both 'include_generation_loc: true' and "
            "'application_locations' in direction_selection. Use one or the other."
        )
    
    # Error: include_generation_loc=True with positive_application_locations
    if has_include_gen_loc and has_pos_app_locs:
        raise ValueError(
            "Configuration error: Cannot specify both 'include_generation_loc: true' and "
            "'positive_application_locations' in direction_selection. Use one or the other."
        )
    
    # Error: include_generation_loc=True without generation_pos
    if has_include_gen_loc and not params.get('generation_pos'):
        raise ValueError(
            "Configuration error: 'include_generation_loc: true' requires 'generation_pos' to be specified."
        )
    
    # Warning: No application locations specified at all
    if not (has_app_locs or has_pos_app_locs or has_neg_app_locs or has_include_gen_loc):
        print("WARNING: No application locations specified. This may cause errors during direction selection.")




def main(config_path: str, save_intermediates: bool = False, output_dir: str = None, experiments_dir: str = 'experiments'):
    """
    Run steering experiment with optional intermediate result saving.
    
    Args:
        config_path: Path to experiment configuration
        save_intermediates: Whether to save intermediate results for validation
        output_dir: Directory to save intermediate results (if save_intermediates=True)
    """
    import pickle
    import json
    from datetime import datetime
    from pathlib import Path
    
    # Setup intermediate result saving
    intermediate_results = {}
    if save_intermediates:
        if output_dir is None:
            output_dir = f"experiment_intermediates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        intermediate_dir = Path(output_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        def save_intermediate_result(stage: str, data, metadata=None):
            """Save intermediate results for later comparison."""
            intermediate_results[stage] = {
                'data': data,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            stage_file = intermediate_dir / f"{stage}.pkl"
            with open(stage_file, 'wb') as f:
                pickle.dump(intermediate_results[stage], f)
                
            # Also save metadata as JSON
            metadata_file = intermediate_dir / f"{stage}_metadata.json"
            with open(metadata_file, 'w') as f:
                def json_encoder(obj):
                    if hasattr(obj, 'tolist'):  # torch tensors, numpy arrays
                        return obj.tolist()
                    elif hasattr(obj, 'shape'):
                        return f"<{type(obj).__name__} shape={obj.shape}>"
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    else:
                        return str(obj)
                        
                json.dump({
                    'stage': stage,
                    'metadata': metadata or {},
                    'timestamp': intermediate_results[stage]['timestamp'],
                    'data_type': type(data).__name__
                }, f, indent=2, default=json_encoder)
    else:
        def save_intermediate_result(stage: str, data, metadata=None):
            """No-op when not saving intermediates."""
            pass
    # load and merge full config
    cfg = load_config(config_path)

    # Instantiate LLM model
    model = instantiate(cfg['model'])
    
    # Save model loading stage
    save_intermediate_result('model_loading', {
        'model_name': model.name,
        'model_type': type(model).__name__,
        'num_layers': getattr(model, 'num_layers', None)
    }, {
        'config': cfg['model']
    })
    # Baseline mode: skip steering, just run inference on test set
    if cfg.get('baseline', False):
        # Prepare experiment output directory
        exp_root = ExperimentSaver(experiments_dir).create_experiment_dir(cfg.get('name'), model.name)
        # Save a copy of the config
        try:
            import shutil
            shutil.copy(config_path, os.path.join(exp_root, 'config.yaml'))
        except Exception:
            pass
        # Evaluate test datasets only
        test_specs = cfg.get('test_data', [])
        # Instantiate test datasets
        if isinstance(test_specs, list):
            test_ds_list = [instantiate(spec) for spec in test_specs]
        else:
            test_ds_list = [instantiate(test_specs)]
        # Run evaluation for each test dataset
        test_results = {}
        for ds in test_ds_list:
            ds_name = type(ds).__name__
            ds_save_dir = os.path.join(exp_root, ds_name)
            results = asyncio.run(
                ds.evaluate(
                    model=model,
                    direction=None,
                    applier=None,
                    application_location=None,
                    save_dir=ds_save_dir,
                    mc_evaluation_method=cfg.get('mc_evaluation_method', DEFAULT_MC_METHOD),
                    **cfg.get('inference', {})
                )
            )
            print(f"Generated outputs for {ds_name} saved under: {ds_save_dir}/{ds.format_mode.value}")
            results_metric = {k: sum(v) / len(v) for k, v in results.items() if isinstance(v, list)}
            test_results[ds_name] = {
                'outputs': results,
                'metrics': results_metric,
                'avg_metric': sum(results_metric.values()) / len(results_metric) if results_metric else None
            }
            print(f"Test results for {ds_name}: {test_results[ds_name]}")
        # Save test results to the experiment directory
        test_results_path = os.path.join(exp_root, 'test_results.yaml')
        with open(test_results_path, 'w') as wf:
            yaml.safe_dump(test_results, wf, sort_keys=False)
        print(f"Baseline test results saved to: {test_results_path}")
        return exp_root

    # Build train data (support single or multiple train dataset specs)
    train_specs = cfg['train_data']
    # Normalize to list of specs
    if isinstance(train_specs, dict):
        train_specs = [train_specs]
    elif not isinstance(train_specs, list):
        raise ValueError("`train_data` must be a dict or list of dicts.")
    train_data_list = []
    for spec in train_specs:
        # Instantiate positive dataset
        pos_ds = instantiate(spec['pos'])
        # Determine negative dataset spec
        neg_spec = spec.get('neg')
        # For paired datasets, ignore neg_spec
        if getattr(pos_ds, 'paired', False):
            neg_ds = None
        else:
            # For unpaired, if neg_spec lacks class, inherit from pos spec and merge params
            if isinstance(neg_spec, dict) and 'class' not in neg_spec:
                base_params = (spec['pos'].get('params') or {}).copy()
                override_params = (neg_spec.get('params') or {})
                merged_params = {**base_params, **override_params}
                neg_spec = {'class': spec['pos'].get('class'), 'params': merged_params}
            neg_ds = instantiate(neg_spec) if neg_spec else None
        # Neutral dataset
        neutral_spec = spec.get('neutral')
        neutral_ds = instantiate(neutral_spec) if neutral_spec else None
        train_data_list.append(SteeringTrainData(pos_ds, neg_ds, neutral_ds))
    print(f"Train data list: {train_data_list}")
    for train_data in train_data_list:
        print(f"Train data: {train_data}")

    # Save data preparation stage
    save_intermediate_result('data_preparation', {
        'num_train_datasets': len(train_data_list),
        'train_data_details': [
            {
                'pos_count': len(td.pos_instructions) if hasattr(td, 'pos_instructions') else 'N/A',
                'neg_count': len(td.neg_instructions) if hasattr(td, 'neg_instructions') else 'N/A',
                'concept': getattr(td, 'concept_name', 'N/A')
            }
            for td in train_data_list
        ]
    })

    # Validation and test datasets (support multiple entries)
    val_specs = cfg['validation_data']
    if isinstance(val_specs, list):
        val_ds_list = [instantiate(spec) for spec in val_specs]
    else:
        val_ds_list = [instantiate(val_specs)]
    
    # Optional negative validation data (for COSMIC dual-dataset approach)
    neg_val_ds_list = None
    if 'neg_validation_data' in cfg:
        neg_val_specs = cfg['neg_validation_data']
        if isinstance(neg_val_specs, list):
            neg_val_ds_list = [instantiate(spec) for spec in neg_val_specs]
        else:
            neg_val_ds_list = [instantiate(neg_val_specs)]
        print(f"Loaded negative validation data: {[ds.name for ds in neg_val_ds_list]}")
    
    test_specs = cfg['test_data']
    if isinstance(test_specs, list):
        test_ds_list = [instantiate(spec) for spec in test_specs]
    else:
        test_ds_list = [instantiate(test_specs)]

    # Direction applier (used both for grid search and for final evaluation)
    dap_spec = cfg['direction_application']
    applier = instantiate(dap_spec)
    
    # Handle applier kwargs for steering direction (used in COSMIC dual objectives)
    applier_kwargs = {}
    
    # General parameter: steering_direction can be "increase" (we want to amplify the existing direction) or "decrease" (opposite).
    # We do not have a default; we require it implemented for all.
    steering_direction = cfg.get('steering_direction')
    
    # Determine if we need reverse based on desired steering direction vs applier's default
    steering_kwargs = setup_applier_kwargs_from_steering_direction(applier, steering_direction)
    applier_kwargs.update(steering_kwargs)
    
    # Backward compatibility: still support reverse_directional_ablation (maps to decrease + DirectionalAblation)
    if cfg.get('reverse_directional_ablation', False):
        raise ValueError("Error: 'reverse_directional_ablation' is deprecated. Use 'steering_direction: increase' instead.")
        cls_path = dap_spec.get('class', '')
        if 'DirectionalAblation' in cls_path:
            applier_kwargs['reverse'] = True
        else:
            print(f"Warning: reverse_directional_ablation=True but applier is {cls_path}. Reverse only supported for DirectionalAblation.")
    # Prepare experiment output directory (include model name)
    exp_root = ExperimentSaver(experiments_dir).create_experiment_dir(cfg.get('name'), model.name)
    # Save a copy of the config for reproducibility
    try:
        import shutil
        shutil.copy(config_path, os.path.join(exp_root, 'config.yaml'))
    except Exception:
        pass
    # Optionally load a precomputed direction instead of running grid search
    pretrained_path = cfg.get('pretrained_direction')
    if pretrained_path:
        from utils.steering_utils import CandidateDirection
        print(f"Loading pretrained direction from {pretrained_path}")
        best_dir = CandidateDirection.load(pretrained_path)
        # Bring in hyperparams from config if not on the loaded object
        if hasattr(best_dir, 'factor'):
            factor = best_dir.factor
        else:
            raise ValueError("Pretrained direction must have a 'factor' attribute.")
            # factor = cfg['direction_selection']['params'].get('factors', [None])[0]
        application_location = getattr(best_dir, 'application_location', None)
    else:
        # Direction generator and its hyperparameter grid
        gen_spec = cfg['direction_generation']['generator']
        generator = instantiate(gen_spec)
        # Single source of truth for hyperparameter grid (with optional percentage-based layer ranges)
        generator_param_grid = cfg['direction_generation'].get('param_grid', {}) or {}
        # Support percentage-based layer ranges in generator param_grid, with optional stepping
        if 'layer_pct_start' in generator_param_grid or 'layer_pct_end' in generator_param_grid:
            start_vals = generator_param_grid.pop('layer_pct_start', None)
            end_vals = generator_param_grid.pop('layer_pct_end', None)
            # step size for layer ranges; default to every 2 layers if not provided
            step_vals = generator_param_grid.pop('layer_step', [2])
            if start_vals is None or end_vals is None:
                raise ValueError("Must specify both 'layer_pct_start' and 'layer_pct_end' in param_grid")
            if not isinstance(start_vals, list) or not isinstance(end_vals, list):
                raise ValueError("'layer_pct_start' and 'layer_pct_end' must be lists of floats between 0 and 1")
            # Normalize step_vals to list of positive integers
            if not isinstance(step_vals, list):
                step_vals = [step_vals]
            for st in step_vals:
                if not (isinstance(st, int) and st >= 1):
                    raise ValueError(f"'layer_step' must be a positive integer, got {st}")
            new_layers = []
            num_layers = model.num_layers
            for s in start_vals:
                for e in end_vals:
                    if not (0.0 <= s <= 1.0 and 0.0 <= e <= 1.0):
                        raise ValueError(f"layer_pct values must be between 0 and 1: got {s}, {e}")
                    start_idx = int(s * num_layers)
                    end_idx = int(e * num_layers)
                    if end_idx < start_idx:
                        raise ValueError(f"Invalid layer_pct range: start {s} > end {e}")
                    for step in step_vals:
                        # Generate layers from start to end (inclusive) with given step
                        new_layers.append(list(range(start_idx, end_idx + 1, step)))
            generator_param_grid['layer'] = new_layers
        # Validate generator param_grid layers within model's layer range (support nested lists)
        if hasattr(model, 'num_layers') and 'layer' in generator_param_grid:
            max_layer = model.num_layers - 1
            layers_spec = generator_param_grid.get('layer', [])
            # Flatten nested lists of layer indices
            flat_layers = []
            for item in layers_spec:
                if isinstance(item, list):
                    flat_layers.extend(item)
                else:
                    flat_layers.append(item)
            # Check each layer index
            for layer_idx in flat_layers:
                if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx > max_layer:
                    raise ValueError(
                        f"Invalid layer {layer_idx} in direction_generation.param_grid['layer']: "
                        f"must be an integer between 0 and {max_layer} for model {model.name}"
                    )
            # Now set the generator_param_grid['layer'] to the flattened list
            generator_param_grid['layer'] = flat_layers


        # Application locations: support explicit layer lists or ranges via layer_start/layer_end
        from direction_application.base import ApplicationLocation, InterventionPositions
        app_locs = []
        for loc in cfg['direction_selection']['params']['application_locations']:
            # Determine list of layers (support 'all', explicit list, or start/end ranges)
            if 'layer' in loc:
                raw_layer = loc['layer']
                if isinstance(raw_layer, str) and raw_layer.lower() == 'all':
                    if not hasattr(model, 'num_layers'):
                        raise ValueError("Cannot specify layer 'all': model has no 'num_layers' attribute")
                    layers = list(range(model.num_layers))
                elif isinstance(raw_layer, int):
                    layers = [raw_layer]
                elif isinstance(raw_layer, list):
                    layers = raw_layer
                else:
                    raise ValueError(f"Invalid layer specification: {raw_layer}")
            elif 'layer_start' in loc and 'layer_end' in loc:
                start = loc['layer_start']
                end = loc['layer_end']
                layers = list(range(start, end + 1))
            else:
                raise ValueError("Each application_locations entry must have 'layer' or 'layer_start' and 'layer_end'.")
            # Convert position to InterventionPositions enum if specified as string
            pos_val = loc.get('pos')
            if isinstance(pos_val, str):
                try:
                    pos_enum = InterventionPositions[pos_val]
                except KeyError:
                    pos_enum = pos_val
            else:
                pos_enum = pos_val
            app_locs.append(
                ApplicationLocation(
                    layer=layers,
                    component=loc['component'],
                    attr=loc['attr'],
                    pos=pos_enum,
                    component_attr_paired=loc.get('component_attr_paired', False)
                )
            )
        # Validate that specified layers are within model's layer range
        if hasattr(model, 'num_layers'):
            max_layer = model.num_layers - 1
            for app_loc in app_locs:
                for layer_idx in app_loc.layer:
                    if layer_idx < 0 or layer_idx > max_layer:
                        raise ValueError(
                            f"Invalid layer {layer_idx} in application_locations: must be between 0 and {max_layer} for model {model.name}"
                        )

        # 1) Generate candidate directions using base class method
        from direction_generation.base import generate_candidate_directions
        # do input cleaning -- cumulative layers is only set if empty application locations and include_generation_loc is True
        if cfg['direction_selection']['params'].get('cumulative_layers', False) and (len(app_locs) > 0 or not cfg['direction_selection']['params'].get('include_generation_loc', False)):
            raise ValueError("cumulative_layers can only be set if application_locations is empty and include_generation_loc is True.")
        dirs, dir_params, _orig_layers = generate_candidate_directions(model, generator, generator_param_grid, train_data_list, **cfg.get('inference', {}), cumulative_layers=cfg['direction_selection']['params'].get('cumulative_layers', False))
        candidate_directions = (dirs, dir_params)
        
        # Save direction generation stage
        save_intermediate_result('direction_generation', dirs, {
            'num_directions': len(dirs),
            'direction_params': dir_params,
            'generator_type': type(generator).__name__,
            'param_grid': generator_param_grid
        })

        # Selector: use the same grid defined under direction_generation
        sel_spec = cfg['direction_selection']
        SelectorCls = load_class(sel_spec['class'])
        
        # Handle separate positive/negative appliers for COSMIC
        selector_kwargs = {
            'applier': applier,
            # 'applier_kwargs': applier_kwargs,  # Do not want applier kwargs here; the selector class itself will take care of it, i.e., for COSMIC will handle reversing internally and for grid search it will reverse based on the dataset. Both don't need factors as they will iterate over all possibilities.
            'application_locations': app_locs,
            'factors': sel_spec['params'].get('factors'),
            'result_path': os.path.join(exp_root, sel_spec['params'].get('result_path', 'grid_search.csv')),
            'include_generation_loc': sel_spec['params'].get('include_generation_loc', False),
            'generation_pos': sel_spec['params'].get('generation_pos', None),
            'cumulative_layers': sel_spec['params'].get('cumulative_layers', False),
            'reverse': applier_kwargs['reverse'],  # keep track of if we want to steer towards positive or negative. necessary for when we actually apply the steering and do a full run on validation (i.e., grid search) or test.
            '_orig_layers': _orig_layers,  # Only for cumulative layers, else is None
        }
        
        # Add COSMIC-specific parameters if they exist (except application locations - handle those after model loading)
        if 'positive_applier' in sel_spec['params']:
            selector_kwargs['positive_applier'] = instantiate(sel_spec['params']['positive_applier'])
        if 'negative_applier' in sel_spec['params']:
            selector_kwargs['negative_applier'] = instantiate(sel_spec['params']['negative_applier'])
        if 'layer_selection_pct' in sel_spec['params']:
            selector_kwargs['layer_selection_pct'] = sel_spec['params']['layer_selection_pct']
        if 'kl_threshold' in sel_spec['params']:
            selector_kwargs['kl_threshold'] = sel_spec['params']['kl_threshold']
        if 'use_kl_divergence_check' in sel_spec['params']:
            selector_kwargs['use_kl_divergence_check'] = sel_spec['params']['use_kl_divergence_check']
        
        # Handle harmless validation data for KL divergence checking
        harmless_ds_list = None
        if 'harmless_validation_data' in cfg:
            # Explicit harmless validation data in concept config
            harmless_specs = cfg['harmless_validation_data']
            if isinstance(harmless_specs, list):
                harmless_ds_list = [instantiate(spec) for spec in harmless_specs]
            else:
                harmless_ds_list = [instantiate(harmless_specs)]
            selector_kwargs['harmless_validation_data'] = harmless_ds_list
        elif cfg.get('use_neg_validation_as_harmless', False) and neg_val_ds_list:
            # Use neg_validation_data as harmless data if concept config requests it
            selector_kwargs['harmless_validation_data'] = neg_val_ds_list
            harmless_ds_list = neg_val_ds_list
            print(f"Using neg_validation_data as harmless_validation_data for KL divergence checking")
        elif 'harmless_validation_data' in sel_spec['params']:
            # Fallback: method-specific harmless validation data (less common)
            selector_kwargs['harmless_validation_data'] = instantiate(sel_spec['params']['harmless_validation_data'])
            harmless_ds_list = selector_kwargs['harmless_validation_data']

        assert harmless_ds_list is not None, "We always need a harmless dataset, if not for KL div then at least for CAST and COSMIC."
        
        # Validate selector configuration before creating the selector
        _validate_selector_config(sel_spec, cfg)
        
        selector = SelectorCls(**selector_kwargs)
        
        # Handle COSMIC-specific application locations after model is loaded (so we can resolve 'all' layers)
        if 'positive_application_locations' in sel_spec['params']:
            from direction_application.base import ApplicationLocation, InterventionPositions
            positive_app_locs = []
            for loc in sel_spec['params']['positive_application_locations']:
                # Use the same layer resolution logic as standard application locations
                if 'layer' in loc:
                    raw_layer = loc['layer']
                    if isinstance(raw_layer, str) and raw_layer.lower() == 'all':
                        if not hasattr(model, 'num_layers'):
                            raise ValueError("Cannot specify layer 'all': model has no 'num_layers' attribute")
                        layers = list(range(model.num_layers))
                    elif isinstance(raw_layer, int):
                        layers = [raw_layer]
                    elif isinstance(raw_layer, list):
                        layers = raw_layer
                    else:
                        raise ValueError(f"Invalid layer specification: {raw_layer}")
                else:
                    raise ValueError("positive_application_locations entry must have 'layer'.")
                
                # Convert position to InterventionPositions enum if specified as string
                pos_val = loc.get('pos')
                if isinstance(pos_val, str):
                    try:
                        pos_enum = InterventionPositions[pos_val]
                    except KeyError:
                        pos_enum = pos_val
                else:
                    pos_enum = pos_val
                
                positive_app_locs.append(ApplicationLocation(
                    layer=layers,
                    component=loc['component'],
                    attr=loc['attr'],
                    pos=pos_enum,
                    component_attr_paired=loc.get('component_attr_paired', False)
                ))
            selector.positive_application_locations = positive_app_locs
        
        if 'negative_application_locations' in sel_spec['params']:
            negative_app_locs = []
            for loc in sel_spec['params']['negative_application_locations']:
                # Special handling for include_generation_loc
                if loc.get('include_generation_loc', False):
                    # This is a placeholder - actual generation location will be set per direction during selection
                    # We'll create a special marker that COSMIC can recognize, including the position
                    generation_pos = loc['generation_pos']
                    # TODO: May need to adjust the below for generality if we want multiple different generation_pos values for COSMIC
                    negative_app_locs.append({
                        'use_generation_loc': True,
                        'generation_pos': generation_pos
                    })
                    continue
                
                # Use the same layer resolution logic as standard application locations
                if 'layer' in loc:
                    raw_layer = loc['layer']
                    if isinstance(raw_layer, str) and raw_layer.lower() == 'all':
                        if not hasattr(model, 'num_layers'):
                            raise ValueError("Cannot specify layer 'all': model has no 'num_layers' attribute")
                        layers = list(range(model.num_layers))
                    elif isinstance(raw_layer, int):
                        layers = [raw_layer]
                    elif isinstance(raw_layer, list):
                        layers = raw_layer
                    else:
                        raise ValueError(f"Invalid layer specification: {raw_layer}")
                else:
                    raise ValueError("negative_application_locations entry must have 'layer'.")
                
                # Convert position to InterventionPositions enum if specified as string
                pos_val = loc.get('pos')
                if isinstance(pos_val, str):
                    try:
                        pos_enum = InterventionPositions[pos_val]
                    except KeyError:
                        pos_enum = pos_val
                else:
                    pos_enum = pos_val
                
                negative_app_locs.append(ApplicationLocation(
                    layer=layers,
                    component=loc['component'],
                    attr=loc['attr'],
                    pos=pos_enum,
                    component_attr_paired=loc.get('component_attr_paired', False)
                ))
            selector.negative_application_locations = negative_app_locs

        # 2) Run selection to get best direction
        best_dir = selector.select(
            model=model,
            candidate_directions=candidate_directions,
            validation_data=val_ds_list,
            train_data=train_data_list,
            neg_validation_data=harmless_ds_list,  # recall this is only for COSMIC. We will use Alpaca for neg prompts in every case, as pos + neg prompts in paired datasets are too close in terms of similarity in activations to make any substantive differences.
            **cfg.get('inference', {})
        )
        factor = getattr(best_dir, 'factor', None)
        application_location = getattr(best_dir, 'application_location', None)
        
        # Save direction selection stage
        save_intermediate_result('direction_selection', best_dir, {
            'factor': factor,
            'application_location': str(application_location) if application_location else None,
            'selector_type': type(selector).__name__,
            'layer': best_dir.loc.layer if best_dir.loc else None,
        })

        # 2.5) Handle conditional steering if enabled
        conditional_config = None
        conditional_applier = None
        if cfg.get('conditional', {}).get('enabled', False):
            print("Running conditional direction selection...")
            
            # Get conditional selector configuration
            cond_sel_spec = cfg['conditional']['condition_selection']
            CondSelectorCls = load_class(cond_sel_spec['class'])
            
            # Update result path to use experiment directory
            cond_selector_params = cond_sel_spec.get('params', {}).copy()
            cond_selector_params['result_path'] = os.path.join(exp_root, cond_selector_params.get('result_path', 'conditional_grid_search_results.csv'))
            
            cond_selector = CondSelectorCls(**cond_selector_params)
            
            # Get condition directions (same as behavior directions but no need for direction parameters)
            # need to check for if we are doing cumulative layers or not
            if cfg['direction_selection']['params'].get('cumulative_layers', False):
                # only select from the actual layers we care about from the yaml config, not every layer between min_layer and max_layer.
                condition_directions = [d for d in candidate_directions[0] if d.loc.layer in _orig_layers]
            else:
                condition_directions = candidate_directions[0]
            if not cfg['conditional'].get('use_behavior_directions_for_condition', True):
                # TODO: Add support for separate condition direction generation
                raise NotImplementedError("Separate condition directions not yet supported")
            
            # Run conditional selection
            conditional_config = cond_selector.select(
                model=model,
                condition_directions=condition_directions,
                validation_data=val_ds_list,
                neg_validation_data=harmless_ds_list  # because we want to have a classifier for main data vs harmless
                # neg_val_ds_list
            )
            
            # Create conditional applier
            from direction_application.conditional import ConditionalApplier
            conditional_applier = ConditionalApplier(
                base_applier=applier,
                condition_direction=conditional_config['condition_direction'].direction,
                condition_loc= conditional_config['condition_direction'].loc,
                condition_threshold=conditional_config['condition_threshold'],
                condition_comparator=conditional_config['condition_comparator'],
                condition_threshold_comparison_mode=conditional_config.get('condition_threshold_comparison_mode', 'last')
            )
            
            print(f"Selected conditional config: location={conditional_config['condition_direction'].loc}, "
                  f"threshold={conditional_config['condition_threshold']}, "
                  f"comparator={conditional_config['condition_comparator']}, "
                  f"F1={conditional_config['f1_score']:.3f}")
        else:
            print("Conditional steering disabled")
    print("Best direction:", best_dir)

    # Save experiment artifacts to the same experiment directory (including inference config)
    saver = ExperimentSaver(experiments_dir)
    exp_path = saver.save_experiment(
        direction=best_dir,
        factor=factor,
        application_location=application_location,
        exp_path=exp_root,
        extra={"inference": cfg.get("inference", {})},
        conditional_config=conditional_config
    )
    print(f"Experiment artifacts saved to: {exp_path}")

    # 3) Evaluate on test set(s) and save per-example outputs
    test_specs = cfg['test_data']
    if isinstance(test_specs, list):
        test_ds_list = [instantiate(spec) for spec in test_specs]
    else:
        test_ds_list = [instantiate(test_specs)]
    test_results = {}
    for ds in test_ds_list:
        ds_name = type(ds).__name__
        ds_save_dir = os.path.join(exp_path, ds_name)
        # Merge coefficient with any existing applier_kwargs (like reverse)
        eval_applier_kwargs = applier_kwargs.copy()
        eval_applier_kwargs["coefficient"] = factor
        
        # Use conditional applier if available, otherwise use regular applier
        eval_applier = conditional_applier if conditional_applier is not None else applier
        
        results = asyncio.run(
            ds.evaluate(
                model=model,
                direction=best_dir,
                applier=eval_applier,
                applier_kwargs=eval_applier_kwargs,
                application_location=best_dir.application_location,
                save_dir=ds_save_dir,
                mc_evaluation_method=cfg.get('mc_evaluation_method', DEFAULT_MC_METHOD),
                **cfg.get('inference', {})
            )
        )
        print(f"Generated outputs for {ds_name} saved under: {ds_save_dir}/{ds.format_mode.value}")
        results_metric = {k: sum(v) / len(v) for k, v in results.items() if isinstance(v, list)}
        test_results[ds_name] = {
            'outputs': results,
            'metrics': results_metric,
            'avg_metric': sum(results_metric.values()) / len(results_metric) if results_metric else None
        }
        print(f"Test results for {ds_name}: {test_results[ds_name]}")

    # Save test results to the experiment directory
    test_results_path = os.path.join(exp_path, 'test_results.yaml')
    with open(test_results_path, 'w') as wf:
        yaml.safe_dump(test_results, wf, sort_keys=False)
    print(f"Test results saved to: {test_results_path}")
    
    # Save final results stage
    save_intermediate_result('final_results', {
        'experiment_path': exp_path,
        'test_results': test_results,
        'factor': factor,
        'application_location': str(application_location) if application_location else None,
        'conditional_enabled': conditional_config is not None,
        'direction_summary': {
            'layer': best_dir.loc.layer if best_dir.loc else None,
            'component': best_dir.loc.component if best_dir.loc else None,
        }
    })
    
    # Save experiment summary if intermediate saving is enabled
    if save_intermediates:
        summary_file = intermediate_dir / 'experiment_summary.json'
        with open(summary_file, 'w') as f:
            def json_encoder(obj):
                if hasattr(obj, 'tolist'):  # torch tensors, numpy arrays
                    return obj.tolist()
                elif hasattr(obj, 'shape'):
                    return f"<{type(obj).__name__} shape={obj.shape}>"
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return str(obj)
                    
            summary = {
                'experiment_type': 'method_steering',
                'config_path': config_path,
                'output_dir': str(intermediate_dir),
                'timestamp': datetime.now().isoformat(),
                'config': cfg,
                'intermediate_stages': list(intermediate_results.keys()),
                'final_results': {
                    'experiment_path': exp_path,
                    'factor': factor,
                    'application_location': str(application_location) if application_location else None
                }
            }
            json.dump(summary, f, indent=2, default=json_encoder)
        print(f"Intermediate results saved to: {intermediate_dir}")
    
    return exp_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run steering experiment')
    parser.add_argument('config', help='Path to experiment configuration file')
    parser.add_argument('--save-intermediates', action='store_true', 
                       help='Save intermediate results for validation')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save intermediate results')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Directory to save experiments (default: experiments)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    main(args.config, args.save_intermediates, args.output_dir, args.experiments_dir)
