#!/usr/bin/env python3
"""
Evaluate pretrained steering directions on out-of-distribution primary datasets.
"""

import os
import sys
import traceback
import yaml
import argparse
import importlib
import glob
import asyncio

# Get script root directory for config and experiment paths
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.steering_utils import CandidateDirection
from direction_application.base import InterventionType
from utils.enums import DEFAULT_MC_METHOD
from utils.experiment_utils import (
    instantiate,
    load_conditional_configuration,
    create_conditional_applier,
    setup_applier_kwargs_from_steering_direction
)


def rebuild_aggregated_yaml(ood_root, filename='ood_test_results.yaml', subdirs=None):
    """Rebuild aggregated YAML from individual dataset metrics.yaml files.

    Args:
        ood_root: Root directory containing dataset subdirectories
        filename: Name of aggregated YAML file to create
        subdirs: List of subdirectories to search (e.g., ['secondary']). If None, search ood_root directly.
    """
    aggregated = {}

    # Determine search path
    if subdirs:
        search_paths = [os.path.join(ood_root, subdir) for subdir in subdirs]
    else:
        search_paths = [ood_root]

    # Collect all metrics from individual dataset directories
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        for dataset_name in os.listdir(search_path):
            dataset_dir = os.path.join(search_path, dataset_name)
            if not os.path.isdir(dataset_dir):
                continue

            metrics_file = os.path.join(dataset_dir, 'metrics.yaml')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = yaml.safe_load(f)
                        if metrics:
                            aggregated[dataset_name] = metrics
                except Exception as e:
                    print(f"Warning: Could not load {metrics_file}: {e}")

    # Write aggregated results
    output_path = os.path.join(ood_root, filename)
    with open(output_path, 'w') as wf:
        yaml.safe_dump(aggregated, wf, sort_keys=False)

    return aggregated


def get_primary_classes():
    """Collect all primary dataset class paths from concept configs."""
    concepts_dir = os.path.join(script_dir, 'configs', 'concepts')
    primary = set()
    if not os.path.isdir(concepts_dir):
        print(f"Concepts directory not found: {concepts_dir}", file=sys.stderr)
        return primary
    for fname in os.listdir(concepts_dir):
        if not fname.endswith('.yaml'):
            continue
        path = os.path.join(concepts_dir, fname)
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        test_data = cfg.get('test_data')
        if not test_data:
            continue
        entries = test_data if isinstance(test_data, list) else [test_data]
        for e in entries:
            cls = e.get('class')
            # only include bias, refusal, hallucination concepts
            if cls and (cls.startswith('data.bias.') or cls.startswith('data.refusal.') or cls.startswith('data.hallucination.')):
                primary.add(cls)
    return primary
    
def load_concept_inference():
    """Map primary dataset class paths to their inference args from concept configs."""
    # Scan both primary and secondary concept configs for inference settings
    # NOTE: We only load inference parameters (max_new_tokens, temperature, etc.)
    # NOT steering_direction, which should only come from the original experiment
    inference_map = {}
    for subdir in ['concepts', 'secondary_concepts']:
        concepts_dir = os.path.join(script_dir, 'configs', subdir)
        if not os.path.isdir(concepts_dir):
            continue
        for fname in os.listdir(concepts_dir):
            if not fname.endswith('.yaml'):
                continue
            path = os.path.join(concepts_dir, fname)
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            inference = cfg.get('inference', {}) or {}
            # Remove steering_direction from inference if it exists - this should only come from experiment config
            inference_filtered = {k: v for k, v in inference.items() if k != 'steering_direction'}
            test_data = cfg.get('test_data')
            if not test_data:
                continue
            entries = test_data if isinstance(test_data, list) else [test_data]
            for e in entries:
                cls = e.get('class')
                if cls:
                    inference_map[cls] = inference_filtered
    return inference_map
    
def get_secondary_tests():
    """Collect all secondary dataset specs from configs/secondary_concepts."""
    secondary_dir = os.path.join(script_dir, 'configs', 'secondary_concepts')
    secondary_tests = []
    if not os.path.isdir(secondary_dir):
        return secondary_tests
    for fname in os.listdir(secondary_dir):
        if not fname.endswith('.yaml'):
            continue
        path = os.path.join(secondary_dir, fname)
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        test_data = cfg.get('test_data')
        if not test_data:
            continue
        entries = test_data if isinstance(test_data, list) else [test_data]
        secondary_tests.extend(entries)
    return secondary_tests


def run_ood_evaluation(experiment_dirs=None, secondary=False, no_primary=False, skip=None, debug=False, mc_evaluation_method=None):
    """
    Core OOD evaluation logic.

    Args:
        experiment_dirs: List of experiment directory paths
        secondary: Whether to evaluate secondary datasets
        no_primary: Whether to skip primary OOD evaluation
        skip: List of dataset names to skip
        debug: Whether to enable debug mode
        mc_evaluation_method: Multiple choice evaluation method ('substring', 'likelihood', or 'both')
    """
    if skip is None:
        skip = []

    # Determine experiment run directories
    if experiment_dirs:
        exp_dirs = [os.path.abspath(p) for p in experiment_dirs if os.path.isdir(p)]
    else:
        pattern = os.path.join(script_dir, 'experiments', '*', '*')
        exp_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not exp_dirs:
        print("No experiment directories found.", file=sys.stderr)
        return False
    # Load secondary dataset specs if requested
    secondary_tests = get_secondary_tests() if secondary else []
    # Datasets to skip by name
    skip_set = set(skip) if skip else set()

    primary_classes = get_primary_classes()
    concept_inference_map = load_concept_inference()
    if not primary_classes:
        print("No primary dataset classes found.", file=sys.stderr)
        return False

    if debug:
        print(f"\n🔍 DEBUG MODE - OOD Evaluation Setup:")
        print(f"   📁 Found {len(exp_dirs)} experiment directories")
        print(f"   🎯 Found {len(primary_classes)} primary dataset classes")
        print(f"   📊 Found {len(secondary_tests)} secondary dataset specs")
        print(f"   ⏭️  Skipping datasets: {list(skip_set)}")
        print(f"   🔄 Secondary evaluation enabled: {secondary}")
        print(f"   🚫 No primary evaluation: {no_primary}")
        if primary_classes:
            print(f"   📝 Primary classes: {sorted(list(primary_classes))}")
        if secondary_tests:
            sec_names = [spec.get('class', '').split('.')[-1] for spec in secondary_tests]
            print(f"   📝 Secondary classes: {sec_names}")

    print(f"\n=== Processing {len(exp_dirs)} experiments ===\n")
    
    for i, exp_dir in enumerate(exp_dirs):
        print(f"\n🔍 Processing experiment {i+1}/{len(exp_dirs)}: {os.path.basename(os.path.dirname(exp_dir))}")
        config_path = os.path.join(exp_dir, 'config.yaml')
        dir_path = os.path.join(exp_dir, 'direction.pt')
        if not os.path.isfile(config_path):
            print(f"Skipping {exp_dir}: missing config.yaml", file=sys.stderr)
            continue

        # Load experiment config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}

        test_data = cfg.get('test_data')
        if not test_data:
            print(f"Skipping {exp_dir}: no test_data in config", file=sys.stderr)
            continue
        test_entries = test_data if isinstance(test_data, list) else [test_data]
        orig_classes = {entry.get('class') for entry in test_entries}
        # Determine sample size n from first entry
        orig_n = test_entries[0].get('params', {}).get('n')
        # Override n for debug mode
        if debug and orig_n:
            orig_n_original = orig_n
            orig_n = min(orig_n, 8)  # Use smaller of original n or 8 for debug
            if orig_n != orig_n_original:
                print(f"   🐛 Debug mode: limiting OOD evaluation n from {orig_n_original} to {orig_n}")

        # Instantiate model, applier, load pretrained direction (used for primary and secondary), handle baseline
        model = instantiate(cfg['model'])
        inference_args = cfg.get('inference', {}) or {}
        is_baseline = cfg.get('baseline', False)
        if not is_baseline and not os.path.isfile(dir_path):
            print(f"Skipping {exp_dir}: missing direction.pt", file=sys.stderr)
            continue
        exp_name = os.path.basename(os.path.dirname(exp_dir))
        if is_baseline:
            if not exp_name.startswith('baseline'):
                print(f"Error: baseline config but experiment dir '{exp_name}' not prefixed 'baseline'. Skipping.", file=sys.stderr)
                continue
            direction = None
            applier_obj = None
            app_loc = None
            applier_kwargs = {}
        else:
            applier_obj = instantiate(cfg['direction_application'])
            best_dir = CandidateDirection.load(dir_path)
            factor = getattr(best_dir, 'factor', None)
            app_loc = getattr(best_dir, 'application_location', None)
            direction = best_dir
            applier_kwargs = {"coefficient": factor}
            
            # Handle conditional steering if enabled in original experiment
            condition_direction, conditional_meta = load_conditional_configuration(exp_dir)
            if condition_direction is not None and conditional_meta is not None:
                print(f"  Loading conditional configuration from experiment metadata")
                applier_obj = create_conditional_applier(applier_obj, condition_direction, conditional_meta)
                print(f"  Using conditional steering: threshold={conditional_meta['condition_threshold']}, "
                      f"comparator={conditional_meta['condition_comparator']}, "
                      f"F1={conditional_meta.get('f1_score', 'N/A')}")
            else:
                print("  No conditional configuration found, using base applier")
            
            # Add reverse parameter based on steering direction from primary behavior (same logic as run_experiment.py)
            steering_direction = cfg.get('steering_direction')
            steering_kwargs = setup_applier_kwargs_from_steering_direction(applier_obj, steering_direction)
            applier_kwargs.update(steering_kwargs)

        # Prepare output directory for OOD results (needed for both primary and secondary evaluation)
        ood_root = os.path.join(exp_dir, 'ood')
        os.makedirs(ood_root, exist_ok=True)
        
        if debug:
            print(f"   📁 OOD directory created/verified: {ood_root}")
            print(f"   🏷️  Experiment is baseline: {is_baseline}")
            print(f"   📊 Original test datasets: {len(orig_classes)} classes")
            print(f"   🔢 Sample size (n): {orig_n}")

        # Primary OOD evaluation  
        if not no_primary:
            ood_results = {}
            raw_ood = sorted(primary_classes - orig_classes)
            ood_classes = [cls for cls in raw_ood if cls.split('.')[-1] not in skip_set]
            
            if debug:
                print(f"   🎯 PRIMARY OOD ANALYSIS:")
                print(f"      📊 All primary classes: {len(primary_classes)}")
                print(f"      📊 Original experiment classes: {len(orig_classes)}")
                print(f"      📊 Raw OOD candidates: {len(raw_ood)} -> {[cls.split('.')[-1] for cls in raw_ood]}")
                print(f"      📊 After skip filter: {len(ood_classes)} -> {[cls.split('.')[-1] for cls in ood_classes]}")
            
            if ood_classes:
                print(f"   🎯 Evaluating on {len(ood_classes)} OOD primary datasets")
                
            for cls_path in ood_classes:
                ds_name = cls_path.split('.')[-1]  
                print(f"     📊 Evaluating: {ds_name}")
                spec = {'class': cls_path, 'params': {'n': orig_n, 'format_mode': 'FormatMode.TEST'}}
                ds = instantiate(spec)
                ds_save_dir = os.path.join(ood_root, ds_name)
                os.makedirs(ds_save_dir, exist_ok=True)
                cache_file = os.path.join(ds_save_dir, 'metrics.yaml')
                
                if debug:
                    print(f"        🔍 Dataset spec: {spec}")
                    print(f"        📁 Save directory: {ds_save_dir}")
                    print(f"        💾 Cache file: {cache_file}")
                    print(f"        📊 Dataset size: {len(ds.dataset) if hasattr(ds, 'dataset') else 'Unknown'}")
                try:
                    if os.path.exists(cache_file):
                        print(f"       📁 Found cached metrics for {ds_name}")
                        with open(cache_file) as rf:
                            out = yaml.safe_load(rf)
                    else:
                        ood_inference = concept_inference_map.get(cls_path, inference_args)
                        # Use override if provided, otherwise use config value, otherwise use default
                        mc_method = mc_evaluation_method if mc_evaluation_method is not None else ood_inference.get('mc_evaluation_method', DEFAULT_MC_METHOD)
                        results = asyncio.run(
                            ds.evaluate(
                                model=model,
                                direction=direction,
                                applier=applier_obj,
                                application_location=app_loc,
                                applier_kwargs=applier_kwargs,
                                save_dir=ds_save_dir,
                                mc_evaluation_method=mc_method,
                                **ood_inference
                            )
                        )
                        metrics = {k: sum(v) / len(v) for k, v in results.items() if isinstance(v, list) and v}

                        # Compute avg_metric based on preferred method (default: substring)
                        preferred_method = ood_inference.get('preferred_avg_metric', 'substring')
                        if preferred_method in ['substring', 'likelihood']:
                            # Look for the preferred metric
                            metric_key = next((k for k in metrics if preferred_method in k.lower()), None)
                            avg_metric = metrics[metric_key] if metric_key else sum(metrics.values()) / len(metrics) if metrics else None
                        else:
                            # Average all metrics
                            avg_metric = sum(metrics.values()) / len(metrics) if metrics else None

                        out = {'metrics': metrics, 'avg_metric': avg_metric}
                        with open(cache_file, 'w') as wf:
                            yaml.safe_dump(out, wf, sort_keys=False)
                    ood_results[ds_name] = out
                    avg_score = out.get('avg_metric', 'N/A')
                    print(f"       ✅ Results for {ds_name}: avg={avg_score}")
                except Exception:
                    tb = traceback.format_exc()
                    error_file = os.path.join(ds_save_dir, 'error.log')
                    with open(error_file, 'w') as ef:
                        ef.write(tb)
                    print(f"       ❌ Error evaluating {ds_name}; see {error_file}")
            # Rebuild aggregated OOD results from all existing metrics files
            rebuild_aggregated_yaml(ood_root, 'ood_test_results.yaml', subdirs=None)
            results_path = os.path.join(ood_root, 'ood_test_results.yaml')
            print(f"   📁 OOD evaluation results saved to: {results_path}")
        # Evaluate secondary datasets if requested
        if secondary and secondary_tests:
            secondary_results = {}
            
            if debug:
                print(f"   📊 SECONDARY OOD ANALYSIS:")
                print(f"      📝 Total secondary specs: {len(secondary_tests)}")
                sec_names_all = [spec.get('class', '').split('.')[-1] for spec in secondary_tests]
                sec_names_filtered = [name for name in sec_names_all if name not in skip_set]
                print(f"      📝 All secondary names: {sec_names_all}")
                print(f"      📝 After skip filter: {sec_names_filtered}")
            
            for sec_spec in secondary_tests:
                cls_path = sec_spec.get('class')
                ds_name = cls_path.split('.')[-1]
                if ds_name in skip_set:
                    print(f"  ⏭️  Skipping secondary dataset: {ds_name}")
                    continue
                print(f"  📊 Evaluating on secondary dataset: {ds_name}")
                
                if debug:
                    print(f"     🔍 Secondary spec: {sec_spec}")
                
                # Override n for debug mode in secondary dataset specs
                if debug:
                    if 'params' in sec_spec:
                        if 'n' in sec_spec['params']:
                            orig_sec_n = sec_spec['params']['n']
                            if orig_sec_n is not None:  # Only override if n is not None
                                sec_spec['params']['n'] = min(orig_sec_n, 8)
                                if sec_spec['params']['n'] != orig_sec_n:
                                    print(f"     🐛 Debug mode: limiting secondary dataset n from {orig_sec_n} to {sec_spec['params']['n']}")
                            else:
                                # If n is None (full dataset, e.g., for DarkBench), set it to 8 for debug mode
                                sec_spec['params']['n'] = 8
                                print(f"     🐛 Debug mode: setting secondary dataset n from None to 8")
                        else:
                            # If no 'n' key exists, add it
                            sec_spec['params']['n'] = 8
                            print(f"     🐛 Debug mode: adding n=8 to secondary dataset spec")
                    else:
                        # If no params section, create it
                        sec_spec['params'] = {'n': 8, 'format_mode': 'FormatMode.TEST'}
                        print(f"     🐛 Debug mode: creating params with n=8")
                
                ds = instantiate(sec_spec)
                ds_save_dir = os.path.join(ood_root, 'secondary', ds_name)
                os.makedirs(ds_save_dir, exist_ok=True)
                cache_file = os.path.join(ds_save_dir, 'metrics.yaml')
                
                if debug:
                    print(f"     📁 Secondary save directory: {ds_save_dir}")
                    print(f"     💾 Secondary cache file: {cache_file}")
                    print(f"     📊 Secondary dataset size: {len(ds.dataset) if hasattr(ds, 'dataset') else 'Unknown'}")
                try:
                    if os.path.exists(cache_file):
                        print(f"       📁 Found cached metrics for {ds_name}")
                        with open(cache_file) as rf:
                            out = yaml.safe_load(rf)
                    else:
                        ood_inference = concept_inference_map.get(cls_path, inference_args)
                        # Use override if provided, otherwise use config value, otherwise use default
                        mc_method = mc_evaluation_method if mc_evaluation_method is not None else ood_inference.get('mc_evaluation_method', DEFAULT_MC_METHOD)
                        results = asyncio.run(
                            ds.evaluate(
                                model=model,
                                direction=direction,
                                applier=applier_obj,
                                application_location=app_loc,
                                applier_kwargs=applier_kwargs,
                                save_dir=ds_save_dir,
                                mc_evaluation_method=mc_method,
                                **ood_inference
                            )
                        )
                        metrics = {k: sum(v) / len(v) for k, v in results.items() if isinstance(v, list) and v}

                        # Compute avg_metric based on preferred method (default: substring)
                        preferred_method = ood_inference.get('preferred_avg_metric', 'substring')
                        if preferred_method in ['substring', 'likelihood']:
                            # Look for the preferred metric
                            metric_key = next((k for k in metrics if preferred_method in k.lower()), None)
                            avg_metric = metrics[metric_key] if metric_key else sum(metrics.values()) / len(metrics) if metrics else None
                        else:
                            # Average all metrics
                            avg_metric = sum(metrics.values()) / len(metrics) if metrics else None

                        out = {'metrics': metrics, 'avg_metric': avg_metric}
                        with open(cache_file, 'w') as wf:
                            yaml.safe_dump(out, wf, sort_keys=False)
                    secondary_results[ds_name] = out
                    print(f"    Secondary results for {ds_name}: {out}")
                except Exception:
                    tb = traceback.format_exc()
                    error_file = os.path.join(ds_save_dir, 'error.log')
                    with open(error_file, 'w') as ef:
                        ef.write(tb)
                    print(f"       ❌ Error evaluating {ds_name}; see {error_file}")
            # Rebuild aggregated secondary results from all existing metrics files
            rebuild_aggregated_yaml(ood_root, 'secondary_test_results.yaml', subdirs=['secondary'])
            sec_results_path = os.path.join(ood_root, 'secondary_test_results.yaml')
            print(f"📁 Secondary OOD evaluation results saved to: {sec_results_path}")
            
            if debug:
                print(f"   ✅ Secondary evaluation completed:")
                print(f"      📊 Evaluated {len(secondary_results)} datasets")
                print(f"      💾 Results file: {sec_results_path}")
                print(f"      📝 Dataset names: {list(secondary_results.keys())}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'experiment_dirs', nargs='*',
        help="Paths to experiment directories (timestamp folders). "
             "If not provided, will scan 'experiments/*/*'."
    )
    parser.add_argument(
        '--secondary', action='store_true',
        help="Also evaluate on secondary datasets defined under configs/secondary_concepts"
    )
    parser.add_argument(
        '--no-primary', action='store_true',
        help="Skip primary OOD evaluation and only run secondary when requested"
    )
    parser.add_argument(
        '--skip', nargs='+', default=[],
        help="List of dataset names (class suffixes) to skip in evaluations"
    )
    parser.add_argument(
        '--debug', action='store_true',
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        '--mc-evaluation-method', choices=['substring', 'likelihood', 'both'], default=None,
        help="Multiple choice evaluation method: 'substring', 'likelihood', or 'both' (overrides config)"
    )
    args = parser.parse_args()

    return run_ood_evaluation(
        experiment_dirs=args.experiment_dirs,
        secondary=args.secondary,
        no_primary=args.no_primary,
        skip=args.skip,
        mc_evaluation_method=args.mc_evaluation_method,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
