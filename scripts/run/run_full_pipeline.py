#!/usr/bin/env python3
"""
Unified pipeline script for steering experiments.

This script runs the complete pipeline:
1. Main experiments (run_multiple.py)
2. OOD dataset evaluation (run_ood_datasets.py) 
3. Concept metrics computation (run_concept_metrics.py)

Usage:
    python run_full_pipeline.py -m qwen25-05b -c refusal_base -M lat
    python run_full_pipeline.py --models qwen25-05b qwen25-7b --concepts implicit_bias explicit_bias --methods pca lat
    python run_full_pipeline.py --skip-main --ood-only  # Only run OOD evaluation
    python run_full_pipeline.py --skip-ood --metrics-only  # Only run concept metrics
"""

import os
import sys
import glob
import yaml
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import logging
from tqdm import tqdm
from run_multiple import main as run_multiple_main
from run_ood_datasets import run_ood_evaluation
from run_concept_metrics import compute_concept_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd: List[str], description: str, check: bool = True, show_progress: bool = False) -> bool:
    """Run a command and handle errors gracefully."""
    if show_progress:
        print(f"\n🚀 {description}")
    else:
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        if show_progress:
            # Stream output for better visibility
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)
            
            for line in process.stdout:
                line_clean = line.strip()
                if line_clean:  # Show all non-empty lines for pipeline scripts
                    print(f"   {line_clean}")
            
            process.wait()
            if process.returncode == 0:
                print(f"✅ Completed: {description}")
                return True
            else:
                print(f"❌ Failed: {description} (exit code: {process.returncode})")
                if check:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
                return False
        else:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            if result.stdout:
                logger.info(f"Output:\n{result.stdout}")
            return True
    except subprocess.CalledProcessError as e:
        if show_progress:
            print(f"❌ Command failed: {description}")
            print(f"   Return code: {e.returncode}")
        else:
            logger.error(f"Command failed: {description}")
            logger.error(f"Return code: {e.returncode}")
            if hasattr(e, 'stdout') and e.stdout:
                logger.error(f"Stdout:\n{e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"Stderr:\n{e.stderr}")
        return False


def run_main_experiments(models: Optional[List[str]] = None,
                        concepts: Optional[List[str]] = None, 
                        methods: Optional[List[str]] = None,
                        debug: bool = False,
                        experiments_dir: str = 'experiments') -> tuple[bool, List[str], List[str]]:
    """Run main steering experiments using run_multiple.py."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Running main steering experiments")
    logger.info("=" * 60)
    
    try:
        steering_dirs, baseline_dirs = run_multiple_main(
            selected_models=models,
            selected_concepts=concepts,
            selected_methods=methods,
            debug=debug,
            experiments_dir=experiments_dir
        )
        total_experiments = len(steering_dirs) + len(baseline_dirs)
        logger.info(f"✅ Main experiments completed successfully. Created {total_experiments} experiments ({len(steering_dirs)} steering, {len(baseline_dirs)} baseline).")
        return True, steering_dirs, baseline_dirs
    except Exception as e:
        logger.error(f"❌ Main experiments failed: {str(e)}")
        return False, [], []

def run_ood_evaluation_pipeline(experiment_dirs: Optional[List[str]] = None,
                               include_secondary: bool = True,
                               skip_datasets: Optional[List[str]] = None,
                               debug: bool = False) -> bool:
    """Run OOD dataset evaluation using run_ood_datasets.py."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Running OOD dataset evaluation")
    logger.info("=" * 60)
    
    try:
        success = run_ood_evaluation(
            experiment_dirs=experiment_dirs,
            secondary=include_secondary,
            no_primary=False,
            skip=skip_datasets,
            debug=debug
        )
        if success:
            logger.info("✅ OOD evaluation completed successfully")
        else:
            logger.error("❌ OOD evaluation failed")
        return success
    except Exception as e:
        logger.error(f"❌ OOD evaluation failed: {str(e)}")
        return False

def find_baseline_experiments() -> Dict[str, str]:
    """Find baseline experiments for concept metrics."""
    baselines = {}
    pattern = os.path.join('experiments', 'baseline_*', '*')
    baseline_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    
    for baseline_dir in baseline_dirs:
        exp_name = os.path.basename(os.path.dirname(baseline_dir))
        parts = exp_name.split('_')
        if len(parts) == 2 and parts[0] == 'baseline':
            # New format: baseline_model
            model = parts[1]
            # This baseline applies to ALL concepts for this model
            baselines[model] = baseline_dir
        elif len(parts) >= 3:
            # Legacy format: baseline_concept_model (for backward compatibility)
            model = parts[-1]
            concept_parts = parts[1:-1] 
            concept = '_'.join(concept_parts)
            
            key = f"{concept}_{model}"
            baselines[key] = baseline_dir
    
    return baselines

def run_concept_metrics(steering_dirs: List[str], baseline_dir: Optional[str] = None, baseline_dirs: Optional[List[str]] = None) -> bool:
    """Run concept metrics computation with flexible baseline specification."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Computing concept metrics")
    logger.info("=" * 60)
    
    if not steering_dirs:
        logger.info("No steering experiments found - skipping concept metrics")
        return True
    
    success_count = 0
    total_count = 0
    
    # Handle different baseline specification modes
    # Note in practice for our experiments, this code is not used at all, since we always provide baselines from the main phase that we can match by model.
    if baseline_dir:
        # Single baseline for all experiments
        baseline_name = os.path.basename(os.path.dirname(baseline_dir))
        logger.info(f"Using single baseline '{baseline_name}' for all experiments")
        
        for exp_dir in steering_dirs:
            exp_name = os.path.basename(os.path.dirname(exp_dir))
            total_count += 1
            logger.info(f"Computing metrics for {exp_name} vs {baseline_name}")
            
            try:
                success = compute_concept_metrics(
                    experiment_dir=exp_dir,
                    baseline_dir=baseline_dir,
                    output_file=None,
                    exclude_from_entanglement=['Twinviews'] if 'llama' in exp_name.lower() else None  # Twinviews is all over the place in LLaMA-based models from substring matching, so we skip it here.
                )
                if success:
                    success_count += 1
                    logger.info(f"✅ Completed metrics for {exp_name}")
                else:
                    logger.warning(f"❌ Failed to compute metrics for {exp_name}")
            except Exception as e:
                logger.error(f"❌ Error computing metrics for {exp_name}: {str(e)}")
                
    elif baseline_dirs:
        # Multiple baselines - match by model
        model_to_baseline = {}
        for b_dir in baseline_dirs:
            exp_name = os.path.basename(os.path.dirname(b_dir))
            parts = exp_name.split('_')
            if len(parts) >= 2 and parts[0] == 'baseline':
                # Handle both baseline_model and baseline_model_debug formats
                if len(parts) == 3 and parts[-1] == 'debug':
                    model = '_'.join(parts[1:])  # e.g., qwen25-05b_debug
                else:
                    model = parts[1]
                model_to_baseline[model] = b_dir
        
        logger.info(f"Using {len(model_to_baseline)} model-specific baselines")
        
        for exp_dir in steering_dirs:
            exp_name = os.path.basename(os.path.dirname(exp_dir))
            parts = exp_name.split('_')
            
            if len(parts) < 3:
                continue
                
            # Handle steering experiments with _debug suffix
            if len(parts) >= 4 and parts[-1] == 'debug':
                model = '_'.join(parts[-2:])  # e.g., qwen25-05b_debug
            else:
                model = parts[-1]  # e.g., qwen25-05b
            baseline_path = model_to_baseline.get(model)
            
            if baseline_path is None:
                logger.warning(f"No baseline found for {exp_name} (model: {model})")
                continue
            
            total_count += 1
            baseline_name = os.path.basename(os.path.dirname(baseline_path))
            logger.info(f"Computing metrics for {exp_name} vs {baseline_name}")
            
            try:
                success = compute_concept_metrics(
                    experiment_dir=exp_dir,
                    baseline_dir=baseline_path,
                    output_file=None,
                    exclude_from_entanglement=['Twinviews'] if 'llama' in model.lower() else None  # Twinviews is all over the place in LLaMA-based models from substring matching, so we skip it here.
                )
                if success:
                    success_count += 1
                    logger.info(f"✅ Completed metrics for {exp_name}")
                else:
                    logger.warning(f"❌ Failed to compute metrics for {exp_name}")
            except Exception as e:
                logger.error(f"❌ Error computing metrics for {exp_name}: {str(e)}")
    else:
        raise ValueError("Either baseline_dir or baseline_dirs must be provided for concept metrics.")
    
    logger.info(f"Concept metrics completed: {success_count}/{total_count} successful")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(
        description='Run the complete steering experiment pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline for specific configuration
    python run_full_pipeline.py -m qwen25-05b -c refusal_base -M lat
    
    # Run multiple models and methods
    python run_full_pipeline.py --models qwen25-05b qwen25-7b --methods pca lat caa
    
    # Skip main experiments, only run OOD evaluation
    python run_full_pipeline.py --skip-main --ood-only
    
    # Skip OOD evaluation, only run concept metrics  
    python run_full_pipeline.py --skip-ood --metrics-only
    
    # Run OOD and metrics on existing experiments
    python run_full_pipeline.py --skip-main -m qwen25-05b -c refusal_base
        """
    )
    
    # Selection arguments (same as run_multiple.py)
    parser.add_argument('-m', '--models', nargs='+',
                       help='Model config basenames to include (e.g., qwen25-05b)')
    parser.add_argument('-c', '--concepts', nargs='+', 
                       help='Concept config basenames to include (e.g., implicit_bias)')
    parser.add_argument('-M', '--methods', nargs='+',
                       help='Method config basenames to include (e.g., pca, baseline)')
    
    # Phase control arguments
    parser.add_argument('--skip-main', action='store_true',
                       help='Skip main experiment phase')
    parser.add_argument('--skip-ood', action='store_true', 
                       help='Skip OOD evaluation phase')
    parser.add_argument('--skip-metrics', action='store_true',
                       help='Skip concept metrics phase')
    
    # Individual phase arguments  
    parser.add_argument('--main-only', action='store_true',
                       help='Only run main experiments')
    parser.add_argument('--ood-only', action='store_true', 
                       help='Only run OOD evaluation')
    parser.add_argument('--metrics-only', action='store_true',
                       help='Only run concept metrics')
    
    # OOD evaluation options
    parser.add_argument('--no-secondary', action='store_true',
                       help='Skip secondary datasets in OOD evaluation')
    parser.add_argument('--skip-datasets', nargs='+', default=['GSM8K', 'EnronEmail'],
                       help='List of dataset names to skip in OOD evaluation (default: GSM8K, EnronEmail)')
    
    # Concept metrics options
    parser.add_argument('--baseline-dir', type=str,
                       help='Specific baseline experiment directory to use for concept metrics')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: set n=8 for all datasets to speed up testing')
    
    # Experiments directory options
    parser.add_argument('--experiments-dir', default='experiments',
                       help='Directory to save/load experiments (default: experiments)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create experiments directory if it doesn't exist
    experiments_dir = os.path.abspath(args.experiments_dir)
    os.makedirs(experiments_dir, exist_ok=True)
    logger.info(f"Using experiments directory: {experiments_dir}")
    
    # Set experiment directory for OpenAI caching
    os.environ['EXPERIMENT_DIR'] = experiments_dir
    
    # Determine which phases to run
    run_main = not args.skip_main and not args.ood_only and not args.metrics_only
    run_ood = not args.skip_ood and not args.main_only and not args.metrics_only
    run_metrics = not args.skip_metrics and not args.main_only and not args.ood_only
    
    if args.main_only:
        run_main, run_ood, run_metrics = True, False, False
    elif args.ood_only:
        run_main, run_ood, run_metrics = False, True, False  
    elif args.metrics_only:
        run_main, run_ood, run_metrics = False, False, True
    
    logger.info("Starting full steering experiment pipeline")
    logger.info(f"Phases: main={run_main}, ood={run_ood}, metrics={run_metrics}")
    
    success = True
    
    # Phase 1: Main experiments
    steering_experiment_dirs = []
    baseline_experiment_dirs = []
    if run_main:
        main_success, steering_experiment_dirs, baseline_experiment_dirs = run_main_experiments(args.models, args.concepts, args.methods, args.debug, experiments_dir)
        if not main_success:
            logger.error("Main experiments failed")
            success = False
            if not (run_ood or run_metrics):  # If this is the only phase, exit
                return 1
    
    # Phase 2: OOD evaluation  
    if run_ood:
        # Use the experiment directories we just created, or find existing ones if we skipped main
        exp_dirs = []
        if steering_experiment_dirs or baseline_experiment_dirs:
            exp_dirs = steering_experiment_dirs + baseline_experiment_dirs
            logger.info(f"Using {len(exp_dirs)} experiment directories from main phase ({len(steering_experiment_dirs)} steering, {len(baseline_experiment_dirs)} baseline)")
        
        if not exp_dirs:
            logger.warning("No experiment directories found for OOD evaluation -- skipping OOD phase")
        else:
            include_secondary = not args.no_secondary
            if not run_ood_evaluation_pipeline(exp_dirs, include_secondary, args.skip_datasets, args.debug):
                logger.error("OOD evaluation failed")
                success = False
    
    # Phase 3: Concept metrics
    if run_metrics:
        # If metrics-only mode and no experiments from current run, find existing experiments
        if not steering_experiment_dirs and args.metrics_only:
            logger.info("Metrics-only mode: searching for existing steering experiments...")
            
            # Find all non-baseline experiment directories
            pattern = os.path.join(experiments_dir, '*', '*')
            all_exp_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
            steering_experiment_dirs = [d for d in all_exp_dirs if not os.path.basename(os.path.dirname(d)).startswith('baseline_')]
            
            logger.info(f"Found {len(steering_experiment_dirs)} existing steering experiments")
            
            # Also find baseline directories for metrics computation
            baseline_pattern = os.path.join(experiments_dir, 'baseline_*', '*')
            baseline_experiment_dirs = [d for d in glob.glob(baseline_pattern) if os.path.isdir(d)]
            logger.info(f"Found {len(baseline_experiment_dirs)} existing baseline experiments")
        
        # Only process steering experiments that were created in this run or found in metrics-only mode
        if not steering_experiment_dirs:
            logger.info("No steering experiments to process - skipping concept metrics")
        elif args.baseline_dir:
            # Use explicitly provided baseline directory
            if not os.path.exists(args.baseline_dir):
                logger.error(f"Specified baseline directory does not exist: {args.baseline_dir}")
                success = False
            else:
                logger.info(f"Using specified baseline directory: {args.baseline_dir}")
                if not run_concept_metrics(steering_experiment_dirs, baseline_dir=args.baseline_dir):
                    logger.error("Concept metrics computation failed")
                    success = False
        elif baseline_experiment_dirs:
            # Use baseline directories from main phase
            logger.info(f"Using {len(baseline_experiment_dirs)} baseline directories from main phase")
            if not run_concept_metrics(steering_experiment_dirs, baseline_dirs=baseline_experiment_dirs):
                logger.error("Concept metrics computation failed")
                success = False
        else:
            # No baselines provided - skip concept metrics
            logger.info("No baseline directories provided - skipping concept metrics")
    
    if success:
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("PIPELINE COMPLETED WITH ERRORS")
        logger.error("=" * 60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
