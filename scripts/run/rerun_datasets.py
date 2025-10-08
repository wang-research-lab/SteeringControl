#!/usr/bin/env python3
"""
Rerun specific datasets across existing experiments.

This script finds all completed experiments and reruns evaluation on specified datasets
without having to recompute steering directions.

Usage:
    # Sequential rerun (default)
    python rerun_datasets.py --target-datasets GPQA Twinviews
    
    # Parallel rerun across multiple GPUs
    python rerun_datasets.py --target-datasets GPQA Twinviews --gpus 0 1 2 3
    
    # Parallel rerun with worker limit
    python rerun_datasets.py --target-datasets GPQA Twinviews --gpus 0 1 2 3 --max-workers 2
    
    # Only rerun missing datasets (not force rerun)  
    python rerun_datasets.py --target-datasets GPQA Twinviews --missing-only
    
    # Check what's missing first
    python rerun_datasets.py --list-missing
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Set
import logging
from tqdm import tqdm
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add necessary directories to Python path
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'analysis'))
sys.path.insert(0, script_dir)

from analyze_concept_metrics import parse_experiment_name
from run_ood_datasets import run_ood_evaluation


def gpu_rerun_worker(gpu_queue, job_queue, log_dir, results_queue, mc_evaluation_method=None):
    """Worker function that processes dataset reruns on assigned GPU."""
    while True:
        gpu_id = None
        try:
            # Get a GPU from the available pool
            gpu_id = gpu_queue.get(timeout=1)

            # Get the next job
            exp_info, datasets_to_skip = job_queue.get_nowait()
            experiment_name = exp_info['name']
            
            print(f"🚀 [GPU {gpu_id}] Starting dataset rerun: {experiment_name}")
            
            # Create individual log file for this experiment
            exp_log_file = os.path.join(log_dir, f"{experiment_name}_gpu{gpu_id}.log")
            
            start_time = time.time()
            success = False
            
            try:
                # Clear cached results
                ood_dir = os.path.join(exp_info['path'], 'ood')
                dirs_removed = []
                
                for dataset in exp_info['datasets_to_rerun']:
                    dataset_locations = [
                        os.path.join(ood_dir, dataset),
                        os.path.join(ood_dir, 'secondary', dataset),
                    ]
                    
                    for dataset_dir in dataset_locations:
                        if os.path.exists(dataset_dir):
                            import shutil
                            shutil.rmtree(dataset_dir)
                            dirs_removed.append(dataset_dir)
                
                # Create subprocess command
                run_ood_script = os.path.join(script_dir, 'run_ood_datasets.py')
                cmd = [
                    sys.executable, run_ood_script,
                    exp_info['path'],
                    '--secondary'
                ]

                # Add skip arguments
                if datasets_to_skip:
                    cmd.extend(['--skip'] + datasets_to_skip)

                # Add mc_evaluation_method if specified
                if mc_evaluation_method:
                    cmd.extend(['--mc-evaluation-method', mc_evaluation_method])

                # Set up environment with isolated GPU
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                # Run subprocess from project root (2 levels up from script_dir)
                project_root = os.path.dirname(os.path.dirname(script_dir))
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    cwd=project_root
                )
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    success = True
                else:
                    raise Exception(f"Subprocess failed with return code {result.returncode}:\\nSTDOUT: {result.stdout}\\nSTDERR: {result.stderr}")
                
                # Save success log
                with open(exp_log_file, 'w') as f:
                    f.write(f"Experiment: {experiment_name}\\n")
                    f.write(f"GPU: {gpu_id}\\n")
                    f.write(f"Command: {' '.join(cmd)}\\n")
                    f.write(f"Path: {exp_info['path']}\\n")
                    f.write(f"Datasets rerun: {exp_info['datasets_to_rerun']}\\n")
                    f.write(f"Directories removed: {dirs_removed}\\n")
                    f.write(f"Start time: {datetime.fromtimestamp(start_time)}\\n")
                    f.write(f"Duration: {duration:.1f}s\\n")
                    f.write(f"Return code: {result.returncode}\\n")
                    f.write(f"Status: SUCCESS\\n")
                    f.write("=" * 80 + "\\n")
                    f.write("STDOUT:\\n")
                    f.write(result.stdout)
                    f.write("\\n" + "=" * 80 + "\\n")
                    f.write("STDERR:\\n")
                    f.write(result.stderr)
                
                print(f"✅ [GPU {gpu_id}] Completed: {experiment_name} ({duration:.1f}s)")
                results_queue.put((True, gpu_id, experiment_name, duration, exp_log_file))
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                
                # Save error log  
                with open(exp_log_file, 'w') as f:
                    f.write(f"Experiment: {experiment_name}\\n")
                    f.write(f"GPU: {gpu_id}\\n")
                    f.write(f"Command: {' '.join(cmd) if 'cmd' in locals() else 'N/A'}\\n")
                    f.write(f"Path: {exp_info['path']}\\n")
                    f.write(f"Datasets rerun: {exp_info['datasets_to_rerun']}\\n")
                    f.write(f"Start time: {datetime.fromtimestamp(start_time)}\\n")
                    f.write(f"Duration: {duration:.1f}s\\n")
                    f.write(f"Status: FAILED\\n")
                    f.write("=" * 80 + "\\n")
                    f.write(f"ERROR: {error_msg}\\n")
                    if 'result' in locals():
                        f.write("\\n" + "=" * 80 + "\\n")
                        f.write("SUBPROCESS STDOUT:\\n")
                        f.write(result.stdout)
                        f.write("\\n" + "=" * 80 + "\\n")
                        f.write("SUBPROCESS STDERR:\\n")
                        f.write(result.stderr)
                
                print(f"❌ [GPU {gpu_id}] Failed: {experiment_name} - {error_msg}")
                results_queue.put((False, gpu_id, experiment_name, duration, exp_log_file))
            
            job_queue.task_done()
            # Put GPU back in queue for next job
            gpu_queue.put(gpu_id)
            gpu_id = None  # Mark as returned
            
        except queue.Empty:
            # Either no GPU available or no jobs available
            if gpu_id is not None:
                # We got a GPU but no job, put it back
                gpu_queue.put(gpu_id)
                gpu_id = None  # Mark as returned
            
            # Check if there are any jobs left
            if job_queue.empty():
                print(f"🏁 Worker exiting - no more jobs available")
                break
            else:
                # Jobs available but no GPU or got timeout, wait a bit and try again
                time.sleep(0.5)
                continue


def find_completed_experiments(base_dir: str, include_baselines: bool = True) -> List[Dict]:
    """Find all completed experiments (including baselines) with test results."""
    experiments = []
    
    for exp_name in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_path) or exp_name.startswith('.'):
            continue
            
        # Include baselines if requested
        is_baseline = 'baseline' in exp_name
        if is_baseline and not include_baselines:
            continue
            
        # Look for timestamp subdirectories with test results
        for subdir in os.listdir(exp_path):
            subdir_path = os.path.join(exp_path, subdir)
            if not os.path.isdir(subdir_path) or len(subdir) < 10:  # Not timestamp format
                continue
                
            # Check if this has test results (works for both experiments and baselines)
            config_file = os.path.join(subdir_path, 'config.yaml')
            has_results = (
                os.path.exists(os.path.join(subdir_path, 'test_results.yaml')) or
                os.path.exists(os.path.join(subdir_path, 'ood', 'ood_test_results.yaml')) or
                os.path.exists(os.path.join(subdir_path, 'ood', 'secondary_test_results.yaml'))
            )
            
            if has_results and os.path.exists(config_file):
                # Parse experiment info
                if is_baseline:
                    # Baseline experiments have format: baseline_model
                    parts = exp_name.split('_', 1)
                    if len(parts) == 2:
                        method = 'baseline'
                        concept = 'none'  # Baselines don't have concepts
                        model = parts[1]
                    else:
                        continue
                else:
                    # Regular experiments - create fake path for parse_experiment_name
                    fake_path = os.path.join(subdir_path, "concept_metrics.yaml")
                    method, concept, model = parse_experiment_name(fake_path)

                if method and model:
                    experiments.append({
                        'name': exp_name,
                        'method': method,
                        'concept': concept if concept else 'none',
                        'model': model,
                        'path': subdir_path,
                        'config_file': config_file,
                        'is_baseline': is_baseline
                    })
    
    return experiments


def get_existing_datasets(exp_path: str) -> Set[str]:
    """Get datasets that have successfully been evaluated for this experiment.

    A dataset is considered "existing" if the corresponding directory with actual
    evaluation data exists. We check directories directly rather than YAML files
    because YAML files can be incomplete/overwritten during partial reruns.
    """
    existing = set()

    # All possible dataset names to check
    all_possible_datasets = ['GPQA', 'Twinviews', 'ARC_C', 'TruthfulQA', 'CMTEST',
                           'DarkBenchAnthro', 'DarkBenchBrandBias', 'DarkBenchRetention',
                           'DarkBenchSneaking', 'DarkBenchSynchopancy', 'BBQ', 'ToxiGen',
                           'FaithEvalCounterfactual', 'FaithEvalInconsistent', 'FaithEvalUnanswerable',
                           'SaladBench', 'PreciseWiki', 'GSM8K', 'EnronEmail']

    for dataset in all_possible_datasets:
        # Check possible locations for this dataset
        # Secondary datasets have nested structure: secondary/GPQA/GPQA/
        possible_paths = [
            os.path.join(exp_path, 'ood', dataset),  # Primary OOD location
            os.path.join(exp_path, 'ood', 'secondary', dataset, dataset),  # Nested secondary location
        ]

        # If ANY of the possible paths exist, consider it existing
        if any(os.path.exists(p) for p in possible_paths):
            existing.add(dataset)

    return existing


def analyze_missing_datasets(experiments_dir: str, target_datasets: List[str] = None):
    """Analyze which datasets are missing across experiments."""
    if target_datasets is None:
        target_datasets = ['GPQA', 'Twinviews', 'ARC_C', 'TruthfulQA', 'CMTEST']
    
    experiments = find_completed_experiments(experiments_dir)
    
    print(f"\\n{'='*80}")
    print(f"MISSING DATASET ANALYSIS")
    print(f"{'='*80}")
    print(f"Target datasets: {target_datasets}")
    print(f"Found {len(experiments)} completed experiments\\n")
    
    missing_summary = {}
    for dataset in target_datasets:
        missing_summary[dataset] = []
    
    for exp in experiments:
        existing = get_existing_datasets(exp['path'])
        missing = [d for d in target_datasets if d not in existing]
        
        if missing:
            print(f"📊 {exp['name']}")
            print(f"   Path: {exp['path']}")
            print(f"   Missing: {missing}")
            for dataset in missing:
                missing_summary[dataset].append(exp)
        else:
            print(f"✅ {exp['name']} - All target datasets present")
    
    print(f"\\n🎯 SUMMARY:")
    for dataset, missing_exps in missing_summary.items():
        if missing_exps:
            print(f"   {dataset}: Missing from {len(missing_exps)} experiments")
        else:
            print(f"   {dataset}: ✅ Present in all experiments")
    
    return missing_summary


def rerun_datasets_for_experiment(exp_info: Dict, datasets_to_skip: List[str], dry_run: bool = False, mc_evaluation_method: str = None) -> bool:
    """Rerun OOD evaluation for target datasets."""
    
    print(f"\\n🔄 Processing {exp_info['name']}")
    print(f"   Method: {exp_info['method']}, Concept: {exp_info['concept']}, Model: {exp_info['model']}")
    print(f"   Path: {exp_info['path']}")
    print(f"   Baseline: {exp_info['is_baseline']}")
    print(f"   Datasets to rerun: {exp_info['datasets_to_rerun']}")
    print(f"   Previously missing: {exp_info['missing_datasets']}")
    
    if dry_run:
        print(f"   [DRY RUN] Would rerun OOD evaluation for: {exp_info['datasets_to_rerun']}")
        return True
        
    # Use the imported OOD runner function
    try:
        print(f"   Running OOD evaluation, skipping: {datasets_to_skip}")
        print(f"   Force rerunning: {exp_info['datasets_to_rerun']}")
        
        # Clear cached results for datasets we want to force rerun
        ood_dir = os.path.join(exp_info['path'], 'ood')
        dirs_to_remove = []
        
        for dataset in exp_info['datasets_to_rerun']:
            # Find dataset directories that exist
            dataset_locations = [
                os.path.join(ood_dir, dataset),              # ood/GPQA/
                os.path.join(ood_dir, 'secondary', dataset), # ood/secondary/Twinviews/
            ]
            
            for dataset_dir in dataset_locations:
                if os.path.exists(dataset_dir):
                    dirs_to_remove.append((dataset, dataset_dir))
        
        if dirs_to_remove:
            print(f"   🗑️  Will remove cached results for:")
            for dataset, dataset_dir in dirs_to_remove:
                print(f"      {dataset}: {dataset_dir}")
            
            response = 'y'  # input(f"   ❓ Remove these {len(dirs_to_remove)} directories? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                import shutil
                for dataset, dataset_dir in dirs_to_remove:
                    print(f"   🗑️  Removing {dataset}: {dataset_dir}")
                    shutil.rmtree(dataset_dir)
            else:
                print(f"   ❌ Cache removal cancelled - will use existing cached results")
                return False
        
        # Run both primary and secondary OOD evaluation
        run_ood_evaluation(
            experiment_dirs=[exp_info['path']],
            secondary=True,  # Include secondary datasets
            no_primary=False,  # Include primary OOD
            skip=datasets_to_skip,  # Skip datasets we already have
            debug=False,
            mc_evaluation_method=mc_evaluation_method
        )
        
        print(f"   ✅ Successfully completed OOD evaluation")
        return True
        
    except Exception as e:
        print(f"   ❌ Exception during OOD evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_parallel_reruns(experiments_to_process: List[Dict], gpu_ids: List[int], max_workers: int = None, mc_evaluation_method: str = None) -> int:
    """Run dataset reruns in parallel across multiple GPUs."""
    
    if max_workers is None:
        max_workers = len(gpu_ids)
    
    # Create log directory
    log_dir = "rerun_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting parallel rerun on GPUs {gpu_ids} with {max_workers} workers")
    print(f"📝 Logs will be saved to: {log_dir}/")
    
    # Create queues
    gpu_queue = queue.Queue()
    job_queue = queue.Queue()
    results_queue = queue.Queue()
    
    # Fill GPU queue
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    
    # Fill job queue
    for exp in experiments_to_process:
        job_queue.put((exp, exp['datasets_to_skip']))
    
    # Start worker threads - use number of GPUs, not min with experiments
    # This ensures we have enough workers to utilize all GPUs
    workers = []
    num_workers = min(max_workers, len(gpu_ids))
    for i in range(num_workers):
        worker = threading.Thread(
            target=gpu_rerun_worker,
            args=(gpu_queue, job_queue, log_dir, results_queue, mc_evaluation_method),
            daemon=True
        )
        worker.start()
        workers.append(worker)
    
    print(f"📍 Started {num_workers} worker threads for {len(gpu_ids)} GPUs")
    
    # Monitor progress
    total_jobs = len(experiments_to_process)
    success_count = 0
    completed = 0
    
    print(f"📊 Processing {total_jobs} experiments...")
    
    start_time = time.time()
    while completed < total_jobs:
        try:
            # Check for completed jobs
            success, gpu_id, experiment_name, duration, log_file = results_queue.get(timeout=1)
            completed += 1
            if success:
                success_count += 1
            
            # Print progress
            elapsed = time.time() - start_time
            progress = completed / total_jobs
            eta = elapsed / progress - elapsed if progress > 0 else 0
            
            print(f"📈 Progress: {completed}/{total_jobs} ({progress*100:.1f}%) - "
                  f"✅ {success_count} success, ❌ {completed-success_count} failed - "
                  f"⏱️ ETA: {eta/60:.1f}m")
            
        except queue.Empty:
            # Timeout, check if workers are still alive and if there are jobs remaining
            alive_workers = [w for w in workers if w.is_alive()]
            remaining_jobs = job_queue.qsize()
            
            if not alive_workers and remaining_jobs == 0:
                print("🏁 All workers finished and no jobs remaining")
                break
            elif remaining_jobs == 0:
                print("⌛ All jobs processed, waiting for workers to complete...")
                continue
            else:
                print(f"⏳ Waiting... {remaining_jobs} jobs remaining, {len(alive_workers)} workers active")
                continue
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=5)
    
    total_time = time.time() - start_time
    print(f"\\n🏁 Parallel processing completed in {total_time/60:.1f}m")
    print(f"📊 Results: ✅ {success_count} successful, ❌ {total_jobs-success_count} failed")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(description='Rerun specific datasets across existing experiments')
    parser.add_argument('--experiments-dir', default='experiments/',
                        help='Directory containing experiments')
    parser.add_argument('--target-datasets', nargs='+', default=['GPQA', 'Twinviews'],
                        help='Datasets to rerun (will be evaluated regardless of existing results)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually running')
    parser.add_argument('--list-missing', action='store_true',
                        help='Only analyze and list missing datasets')
    parser.add_argument('--force-rerun', action='store_true', default=True,
                        help='Force rerun target datasets even if they exist (default: True)')
    parser.add_argument('--missing-only', action='store_true',
                        help='Only rerun datasets that are actually missing (opposite of force-rerun)')
    parser.add_argument('--filter-method', help='Only process experiments with this method')
    parser.add_argument('--filter-concept', help='Only process experiments with this concept') 
    parser.add_argument('--filter-model', help='Only process experiments with this model')
    parser.add_argument('--gpus', nargs='+', type=int, default=None,
                        help='GPU IDs to use (e.g., --gpus 0 1 2 3). If not specified, runs sequentially.')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (defaults to number of GPUs)')
    parser.add_argument('--mc-evaluation-method', choices=['substring', 'likelihood', 'both'], default=None,
                        help="Multiple choice evaluation method: 'substring', 'likelihood', or 'both' (overrides experiment configs)")

    args = parser.parse_args()
    
    if not os.path.exists(args.experiments_dir):
        print(f"Error: Experiments directory not found: {args.experiments_dir}")
        return 1
        
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Analyze missing datasets
    missing_summary = analyze_missing_datasets(args.experiments_dir, args.target_datasets)
    
    if args.list_missing:
        return 0
        
    # Find experiments to process
    all_experiments = find_completed_experiments(args.experiments_dir)
    experiments_to_process = []
    
    # Override force_rerun if missing_only is specified
    force_rerun = args.force_rerun and not args.missing_only

    # Define all possible datasets once
    all_possible_datasets = ['GPQA', 'Twinviews', 'ARC_C', 'TruthfulQA', 'CMTEST',
                           'DarkBenchAnthro', 'DarkBenchBrandBias', 'DarkBenchRetention',
                           'DarkBenchSneaking', 'DarkBenchSynchopancy', 'BBQ', 'ToxiGen',
                           'FaithEvalCounterfactual', 'FaithEvalInconsistent', 'FaithEvalUnanswerable',
                           'SaladBench', 'PreciseWiki', 'GSM8K', 'EnronEmail']

    for exp in all_experiments:
        # Apply filters
        if args.filter_method and exp['method'].upper() != args.filter_method.upper():
            continue
        if args.filter_concept and args.filter_concept.lower() not in exp['concept'].lower():
            continue
        if args.filter_model and args.filter_model.lower() not in exp['model'].lower():
            continue

        existing = get_existing_datasets(exp['path'])

        if force_rerun:
            # Force rerun mode: rerun target datasets even if they already exist
            # This will delete and re-evaluate the target datasets
            datasets_to_rerun = args.target_datasets
            missing_for_exp = [d for d in args.target_datasets if d not in existing]

            # Skip everything EXCEPT our target datasets (force them to rerun)
            datasets_to_skip = [d for d in all_possible_datasets if d not in args.target_datasets]

            exp['datasets_to_rerun'] = datasets_to_rerun
            exp['missing_datasets'] = missing_for_exp  # For display purposes
            exp['datasets_to_skip'] = datasets_to_skip
            experiments_to_process.append(exp)

        else:
            # Missing-only mode: only rerun target datasets that are actually missing
            missing_for_exp = [d for d in args.target_datasets if d not in existing]

            if missing_for_exp:
                # Skip all datasets except the missing target ones
                # This includes: (1) existing datasets and (2) non-target datasets
                datasets_to_skip = list(existing) + [d for d in all_possible_datasets if d not in args.target_datasets]
                # Remove duplicates
                datasets_to_skip = list(set(datasets_to_skip))

                exp['datasets_to_rerun'] = missing_for_exp
                exp['missing_datasets'] = missing_for_exp
                exp['datasets_to_skip'] = datasets_to_skip
                experiments_to_process.append(exp)
    
    if not experiments_to_process:
        print(f"\\n🎉 No experiments need the specified datasets rerun!")
        return 0
        
    print(f"\\n{'='*80}")
    print(f"RERUNNING DATASETS")
    print(f"{'='*80}")
    print(f"Mode: {'FORCE RERUN' if force_rerun else 'MISSING ONLY'}")
    print(f"Target datasets: {args.target_datasets}")
    print(f"Will process {len(experiments_to_process)} experiments")
    
    if force_rerun:
        print("🔄 FORCE RERUN: Target datasets will be re-evaluated on ALL experiments")
    else:
        print("📋 MISSING ONLY: Target datasets will only be evaluated if missing")
    
    if args.dry_run:
        print("DRY RUN MODE - No actual changes will be made")
    
    # Process experiments - either in parallel or sequentially
    success_count = 0
    
    if args.gpus and not args.dry_run:
        # Parallel processing with GPU scheduling
        success_count = run_parallel_reruns(experiments_to_process, args.gpus, args.max_workers, args.mc_evaluation_method)
    else:
        # Sequential processing
        if args.gpus:
            print("Note: --gpus specified but running in dry-run mode, using sequential processing")
        
        for exp in tqdm(experiments_to_process, desc="Processing experiments"):
            success = rerun_datasets_for_experiment(exp, exp['datasets_to_skip'], args.dry_run, args.mc_evaluation_method)
            if success:
                success_count += 1
    
    print(f"\\n🎯 SUMMARY:")
    print(f"   Processed: {len(experiments_to_process)} experiments")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(experiments_to_process) - success_count}")
    
    return 0 if success_count == len(experiments_to_process) else 1


if __name__ == '__main__':
    sys.exit(main())
