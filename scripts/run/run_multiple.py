#!/usr/bin/env python3
"""
run_multiple.py: Batch-run steering experiments over combinations of models, concepts, and methods.

This script loads YAML configs for each model in configs/models/, each concept in configs/concepts/,
and each method YAML defined below, merges them into a final experiment config, writes it under
configs/generated/, and invokes run_experiment.py on it.
"""
import os
import glob
import yaml
import subprocess
import argparse
from tqdm import tqdm
from utils.experiment_utils import deep_merge
from run_experiment import main as run_experiment_main

def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def main(selected_models=None, selected_concepts=None, selected_methods=None, debug=False, experiments_dir='experiments'):
    """
    Run multiple steering experiments.
    
    Returns:
        Tuple of (steering_experiment_dirs, baseline_experiment_dirs)
    """
    created_experiment_dirs = []
    baseline_experiment_dirs = []
    # File paths for experiments
    model_files = sorted(glob.glob('configs/models/*.yaml'))
    concept_files = sorted(glob.glob('configs/concepts/*.yaml'))
    # Define method config files here (method-only YAMLs under configs/)
    method_files = [
        'configs/baseline.yaml',

        'configs/pca.yaml',
        'configs/pca_nokl.yaml',
        'configs/pca_cosmic.yaml',
        'configs/pca_new_cosmic.yaml',
        'configs/pca_conditional.yaml',

        'configs/lat.yaml',
        'configs/lat_nokl.yaml',
        'configs/lat_conditional.yaml',

        'configs/caa.yaml',
        'configs/caa_nokl.yaml',
        'configs/caa_cosmic.yaml',
        'configs/caa_conditional.yaml',

        'configs/dim.yaml',
        'configs/dim_nokl.yaml',
        'configs/dim_conditional.yaml',
        'configs/dim_cosmic.yaml',

        'configs/ace.yaml',
        'configs/ace_nokl.yaml',
        'configs/ace_cosmic.yaml',
        'configs/ace_conditional.yaml',
        # TODO: Add more methods (e.g., configs/lat.yaml, configs/dim.yaml, etc.)
    ]
    # Filter lists based on user selections
    def name_of(path):
        return os.path.splitext(os.path.basename(path))[0]
    if selected_models:
        model_files = [f for f in model_files if name_of(f) in selected_models]
    if selected_concepts:
        concept_files = [f for f in concept_files if name_of(f) in selected_concepts]
    if selected_methods:
        method_files = [f for f in method_files if name_of(f) in selected_methods]

    # Directory to write merged experiment configs
    out_dir = os.path.join('configs', 'generated')
    os.makedirs(out_dir, exist_ok=True)

    # Handle baselines separately - run once per model with ALL concept datasets
    baseline_methods = [f for f in method_files if 'baseline' in os.path.basename(f)]
    non_baseline_methods = [f for f in method_files if 'baseline' not in os.path.basename(f)]
    
    # Run baselines once per model with all concept test datasets
    if baseline_methods:
        print(f"\n=== Running Baselines ===")
        
    for i, baseline_method_path in enumerate(baseline_methods):
        baseline_method_cfg = load_yaml(baseline_method_path)
        baseline_method_cfg.pop('extends', None)
        baseline_method_name = os.path.splitext(os.path.basename(baseline_method_path))[0]
        
        print(f"Baseline method {i+1}/{len(baseline_methods)}: {baseline_method_name}")
        
        for j, model_path in enumerate(model_files):
            model_cfg = load_yaml(model_path)
            model_cfg.pop('extends', None)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # Create baseline using FIRST concept only (others will be handled by OOD)
            if concept_files:
                first_concept_cfg = load_yaml(concept_files[0])
                first_concept_cfg.pop('extends', None)
                
                # Create baseline config with model + first concept
                baseline_cfg = deep_merge(model_cfg, baseline_method_cfg)
                
                # Copy all relevant sections from first concept
                for key in ['train_data', 'validation_data', 'neg_validation_data', 'harmless_validation_data', 'test_data', 'inference']:
                    if key in first_concept_cfg:
                        baseline_cfg[key] = first_concept_cfg[key]
                        
                first_concept_name = os.path.splitext(os.path.basename(concept_files[0]))[0]
                print(f"Creating baseline using primary concept: {first_concept_name}")
                if len(concept_files) > 1:
                    other_concepts = [os.path.splitext(os.path.basename(f))[0] for f in concept_files[1:]]
                    print(f"Other concepts will be evaluated via OOD: {other_concepts}")
            else:
                # Fallback if no concepts provided
                raise ValueError("No concept files found. Please provide at least one concept YAML file.")
                baseline_cfg = deep_merge(model_cfg, baseline_method_cfg)
            
            # Debug mode: override all n values to 8 for baseline (AFTER adding train/val data)
            if debug:
                def override_n_values(config_dict):
                    """Recursively override all 'n' values to 8 for debug mode."""
                    if isinstance(config_dict, dict):
                        for key, value in config_dict.items():
                            if key == 'n':
                                config_dict[key] = 8
                            elif isinstance(value, (dict, list)):
                                override_n_values(value)
                    elif isinstance(config_dict, list):
                        for item in config_dict:
                            override_n_values(item)
                
                override_n_values(baseline_cfg)
            
            exp_name = f"{baseline_method_name}_{model_name}"
            if debug:
                exp_name += "_debug"
            baseline_cfg['name'] = exp_name
            
            # Write baseline config
            out_path = os.path.join(out_dir, f"{exp_name}.yaml")
            with open(out_path, 'w') as wf:
                yaml.safe_dump(baseline_cfg, wf, sort_keys=False)
            
            print(f"  Model {j+1}/{len(model_files)}: {model_name}")
            
            # Count test datasets from first concept for logging
            test_data = baseline_cfg.get('test_data', [])
            num_datasets = len(test_data) if isinstance(test_data, list) else (1 if test_data else 0)
            print(f"\n🚀 Running: {exp_name} (testing on {num_datasets} datasets from {first_concept_name})")
            try:
                exp_path = run_experiment_main(out_path, experiments_dir=experiments_dir)
                baseline_experiment_dirs.append(exp_path)  # Track baseline separately
                print(f"✅ Completed: {exp_name}")
                
                # Clean up GPU memory after experiment
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            except Exception as e:
                print(f"❌ Failed: {exp_name} - {str(e)}")
                raise
    
    # Run regular steering experiments (method x concept x model)
    if non_baseline_methods:
        total_experiments = len(model_files) * len(concept_files) * len(non_baseline_methods)
        print(f"\n=== Running Steering Experiments ===")
        print(f"Total experiments to run: {total_experiments}")
        
        experiment_count = 0
        
    for model_path in model_files:
        model_cfg = load_yaml(model_path)
        model_cfg.pop('extends', None)
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        for concept_path in concept_files:
            concept_cfg = load_yaml(concept_path)
            concept_cfg.pop('extends', None)
            concept_name = os.path.splitext(os.path.basename(concept_path))[0]
            # Merge model + concept
            mid_cfg = deep_merge(model_cfg, concept_cfg)

            for method_path in non_baseline_methods:
                method_cfg = load_yaml(method_path)
                method_cfg.pop('extends', None)
                method_name = os.path.splitext(os.path.basename(method_path))[0]

                # Final merge: model -> concept -> method
                final_cfg = deep_merge(mid_cfg, method_cfg)
                # Override experiment name
                exp_name = f"{method_name}_{concept_name}_{model_name}"
                final_cfg['name'] = exp_name
                
                # Debug mode: override all n values to 8
                if debug:
                    def override_n_values(config_dict):
                        """Recursively override all 'n' values to 8 for debug mode."""
                        if isinstance(config_dict, dict):
                            for key, value in config_dict.items():
                                if key == 'n':
                                    config_dict[key] = 8
                                elif isinstance(value, (dict, list)):
                                    override_n_values(value)
                        elif isinstance(config_dict, list):
                            for item in config_dict:
                                override_n_values(item)
                    
                    override_n_values(final_cfg)
                    exp_name += "_debug"
                    final_cfg['name'] = exp_name

                # Write out merged config
                out_path = os.path.join(out_dir, f"{exp_name}.yaml")
                with open(out_path, 'w') as wf:
                    yaml.safe_dump(final_cfg, wf, sort_keys=False)

                experiment_count += 1
                
                print(f"\n🚀 Running: {exp_name} ({experiment_count}/{total_experiments})")
                try:
                    exp_path = run_experiment_main(out_path, experiments_dir=experiments_dir)
                    created_experiment_dirs.append(exp_path)
                    print(f"✅ Completed: {exp_name}")
                    
                    # Clean up GPU memory after experiment
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                except Exception as e:
                    print(f"❌ Failed: {exp_name} - {str(e)}")
                    raise
    
    return created_experiment_dirs, baseline_experiment_dirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch-run steering experiments: select models, concepts, and methods.')
    parser.add_argument(
        '-m', '--models', nargs='+',
        help='Model config basenames to include (e.g., qwen25-05b). Default: all models.')
    parser.add_argument(
        '-c', '--concepts', nargs='+',
        help='Concept config basenames to include (e.g., implicit_bias). Default: all concepts.')
    parser.add_argument(
        '-M', '--methods', nargs='+',
        help='Method config basenames to include (e.g., pca, baseline). Default: all methods.')
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug mode: set n=8 for all datasets to speed up testing')
    parser.add_argument(
        '--experiments-dir', default='experiments',
        help='Directory to save experiments (default: experiments)')
    args = parser.parse_args()
    main(selected_models=args.models,
         selected_concepts=args.concepts,
         selected_methods=args.methods,
         debug=args.debug,
         experiments_dir=args.experiments_dir)
