#!/usr/bin/env python3
"""
Script to compute concept metrics (effectiveness and entanglement) for steering experiments.

This script compares steered experiment results with baseline results to compute:
1. Effectiveness: How well steering improves performance on primary behaviors
2. Entanglement: How steering affects performance on non-targeted behaviors

Usage:
    python run_concept_metrics.py <experiment_dir> <baseline_dir> [--output <output_file>]
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from utils.evaluation_methods.concept_metrics import compute_all_metrics


def compute_concept_metrics(experiment_dir: str, baseline_dir: str, output_file: Optional[str] = None, exclude_from_entanglement: Optional[List[str]] = None) -> bool:
    """
    Compute concept metrics for a steering experiment.
    
    Args:
        experiment_dir: Path to steering experiment directory
        baseline_dir: Path to baseline experiment directory  
        output_file: Optional output file for metrics
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate input directories
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory '{experiment_dir}' does not exist", file=sys.stderr)
        return False
        
    if not os.path.exists(baseline_dir):
        print(f"Error: Baseline directory '{baseline_dir}' does not exist", file=sys.stderr)
        return False

    # Handle timestamp subdirectories - find the actual experiment directory
    def find_timestamp_dir(base_dir):
        if os.path.exists(os.path.join(base_dir, 'test_results.yaml')):
            return base_dir
        # Look for timestamp subdirectories
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.exists(os.path.join(subdir_path, 'test_results.yaml')):
                return subdir_path
        return None
    
    # Find actual experiment directories
    actual_exp_dir = find_timestamp_dir(experiment_dir)
    actual_baseline_dir = find_timestamp_dir(baseline_dir)

    if actual_exp_dir is None:
        print(f"Error: Could not find test_results.yaml in {experiment_dir} or its subdirectories", file=sys.stderr)
        return False
        
    if actual_baseline_dir is None:
        print(f"Error: Could not find test_results.yaml in {baseline_dir} or its subdirectories", file=sys.stderr)
        return False

    # Load all results (main + OOD) from both experiments
    exp_scores = load_all_scores_from_experiment(actual_exp_dir)
    baseline_scores = load_all_scores_from_experiment(actual_baseline_dir)
    
    if not exp_scores:
        print(f"Error: Could not load experiment results from {actual_exp_dir}", file=sys.stderr)
        return False
        
    if not baseline_scores:
        print(f"Error: Could not load baseline results from {actual_baseline_dir}", file=sys.stderr) 
        return False

    # Load config to identify primary behaviors  
    exp_config_path = os.path.join(actual_exp_dir, 'config.yaml')
    primary_behaviors = get_primary_behaviors_from_config_file(exp_config_path)
    
    config = load_results_yaml(exp_config_path)
    model_name = ""
    if config and 'train_data' in config and 'pos' in config['train_data']:
        model_path = config['train_data']['pos'].get('model_path', '')
        model_name = model_path.lower()
    
    if exclude_from_entanglement is None:
        exclude_from_entanglement = []
    else:
        exclude_from_entanglement = list(exclude_from_entanglement)  # Make a copy
    
    # Compute all metrics
    try:
        metrics = compute_all_metrics(
            steered_scores=exp_scores,
            baseline_scores=baseline_scores,
            primary_behaviors=primary_behaviors,
            exclude_from_entanglement=exclude_from_entanglement
        )
        
        # Output results
        if output_file:
            with open(output_file, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
            print(f"Metrics saved to: {output_file}")
        else:
            # Save to the actual experiment directory
            default_output = os.path.join(actual_exp_dir, 'concept_metrics.yaml')
            with open(default_output, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
            print(f"Metrics saved to: {default_output}")
            yaml.dump(metrics, sys.stdout, default_flow_style=False, sort_keys=False)
            
        return True
        
    except Exception as e:
        print(f"Error computing metrics: {e}", file=sys.stderr)
        return False


def load_results_yaml(file_path: str) -> Optional[Dict]:
    """Load and parse a YAML results file."""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None


def extract_scores_from_results(results: Dict) -> Dict[str, float]:
    """Extract dataset scores from results YAML structure."""
    scores = {}
    
    for dataset_name, dataset_results in results.items():
        if isinstance(dataset_results, dict) and 'avg_metric' in dataset_results:
            # Handle the case where avg_metric might be None
            avg_metric = dataset_results['avg_metric']
            if avg_metric is not None:
                score = float(avg_metric)
                # Transform DarkBench scores (1 - score) since higher scores mean worse behavior
                # DarkBench scores: 1 = exhibited dark pattern, 0 = didn't exhibit
                # We want: 1 = good (didn't exhibit), 0 = bad (exhibited)
                if dataset_name.startswith('DarkBench'):
                    score = 1.0 - score
                scores[dataset_name] = score
    
    return scores


def load_all_scores_from_experiment(exp_dir: str) -> Dict[str, float]:
    """Load scores from main, ood, and secondary test results files.

    Expects exp_dir to be the actual timestamp directory containing test_results.yaml.

    """
    all_scores = {}
    
    # Load main test results
    main_results_path = os.path.join(exp_dir, 'test_results.yaml')
    main_results = load_results_yaml(main_results_path)
    if main_results:
        main_scores = extract_scores_from_results(main_results)
        all_scores.update(main_scores)
    
    # Load OOD test results if they exist
    ood_results_path = os.path.join(exp_dir, 'ood', 'ood_test_results.yaml')
    ood_results = load_results_yaml(ood_results_path)
    if ood_results:
        ood_scores = extract_scores_from_results(ood_results)
        all_scores.update(ood_scores)
    
    # Load secondary test results if they exist
    secondary_results_path = os.path.join(exp_dir, 'ood', 'secondary_test_results.yaml')
    secondary_results = load_results_yaml(secondary_results_path)
    if secondary_results:
        secondary_scores = extract_scores_from_results(secondary_results)
        all_scores.update(secondary_scores)
    
    return all_scores


def load_experiment_config(exp_dir: str) -> Optional[Dict]:
    """Load experiment configuration to determine primary datasets."""
    config_path = os.path.join(exp_dir, 'config.yaml')
    return load_results_yaml(config_path)


def get_primary_datasets_from_config(config: Dict) -> Set[str]:
    """Extract primary dataset names from experiment configuration."""
    primary_datasets = set()
    
    # Check train_data section
    if 'train_data' in config and isinstance(config['train_data'], dict):
        for data_type in ['pos', 'neg', 'neutral']:
            if data_type in config['train_data'] and config['train_data'][data_type]:
                data_spec = config['train_data'][data_type]
                if isinstance(data_spec, dict) and 'class' in data_spec:
                    class_name = data_spec['class'].split('.')[-1]
                    primary_datasets.add(class_name)
    
    # Check test_data section  
    if 'test_data' in config and isinstance(config['test_data'], list):
        for data_spec in config['test_data']:
            if isinstance(data_spec, dict) and 'class' in data_spec:
                class_name = data_spec['class'].split('.')[-1]
                primary_datasets.add(class_name)
    
    return primary_datasets


def get_primary_behaviors_from_config_file(config_path: str) -> List[str]:
    """Extract primary behaviors from experiment config.yaml."""
    if not os.path.exists(config_path):
        print(f"Warning: config.yaml not found at {config_path}", file=sys.stderr)
        return []
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract primary dataset names from test_data section of config
        primary_datasets = get_primary_datasets_from_config(config)
        return list(primary_datasets)
    except Exception as e:
        print(f"Warning: Could not extract primary behaviors from {config_path}: {e}", file=sys.stderr)
        return []


def collect_all_results(exp_dir: str) -> Dict[str, float]:
    """Collect all results from experiment directory (test + OOD)."""
    all_scores = {}
    
    # Load main test results
    test_results_path = os.path.join(exp_dir, 'test_results.yaml')
    test_results = load_results_yaml(test_results_path)
    if test_results:
        all_scores.update(extract_scores_from_results(test_results))
    
    # Load OOD results if they exist
    ood_dir = os.path.join(exp_dir, 'ood')
    if os.path.exists(ood_dir):
        # Load primary OOD results
        ood_results_path = os.path.join(ood_dir, 'ood_test_results.yaml')
        ood_results = load_results_yaml(ood_results_path)
        if ood_results:
            all_scores.update(extract_scores_from_results(ood_results))
        
        # Load secondary results
        secondary_results_path = os.path.join(ood_dir, 'secondary_test_results.yaml')
        secondary_results = load_results_yaml(secondary_results_path)
        if secondary_results:
            all_scores.update(extract_scores_from_results(secondary_results))
    
    return all_scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute concept metrics for steering experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_concept_metrics.py exp_dir baseline_dir
    python run_concept_metrics.py exp_dir baseline_dir --output metrics.yaml
        """
    )
    
    parser.add_argument('experiment_dir', help='Path to steering experiment directory')
    parser.add_argument('baseline_dir', help='Path to baseline experiment directory')
    parser.add_argument('--output', '-o', help='Output file for metrics (default: stdout)')
    parser.add_argument('--exclude-from-entanglement', nargs='+', default=[], help='Dataset names to exclude from entanglement calculation (default: none)')
                        # ['TwinViews'],
                       # help='Dataset names to exclude from entanglement calculation (default: TwinViews)')
    
    args = parser.parse_args()
    
    print(f"Computing concept metrics...")
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Baseline directory: {args.baseline_dir}")
    
    success = compute_concept_metrics(
        experiment_dir=args.experiment_dir,
        baseline_dir=args.baseline_dir,
        output_file=args.output,
        exclude_from_entanglement=args.exclude_from_entanglement
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
