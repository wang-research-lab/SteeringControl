#!/usr/bin/env python3
"""
Parallel experiment launcher for steering experiments.
Automatically manages GPU assignment and job scheduling across multiple GPUs.
"""

import os
import sys
import subprocess
import time
import argparse
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import queue

def gpu_worker(gpu_queue, job_queue, log_dir, results_queue):
    """Worker function that processes jobs on assigned GPU."""
    while True:
        try:
            # Get a GPU from the available pool
            gpu_id = gpu_queue.get(timeout=1)
            
            try:
                # Get the next job
                cmd, experiment_name = job_queue.get_nowait()
                
                print(f"🚀 [GPU {gpu_id}] Starting: {experiment_name}")
                print(f"📋 [GPU {gpu_id}] Command: {' '.join(cmd)}")
                
                # Create individual log file for this experiment
                exp_log_file = os.path.join(log_dir, f"{experiment_name}_gpu{gpu_id}.log")
                
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                start_time = time.time()
                try:
                    # Set working directory to project root (two levels up from scripts/run/)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    result = subprocess.run(
                        cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        cwd=project_root
                    )
                    
                    duration = time.time() - start_time
                    
                    # Save full output to individual log file
                    with open(exp_log_file, 'w') as f:
                        f.write(f"Experiment: {experiment_name}\n")
                        f.write(f"GPU: {gpu_id}\n")
                        f.write(f"Command: {' '.join(cmd)}\n")
                        f.write(f"Start time: {datetime.fromtimestamp(start_time)}\n")
                        f.write(f"Duration: {duration:.1f}s\n")
                        f.write(f"Return code: {result.returncode}\n")
                        f.write("=" * 80 + "\n")
                        f.write("STDOUT:\n")
                        f.write(result.stdout)
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("STDERR:\n")
                        f.write(result.stderr)
                    
                    if result.returncode == 0:
                        print(f"✅ [GPU {gpu_id}] Completed: {experiment_name} ({duration:.1f}s)")
                        print(f"📝 [GPU {gpu_id}] Full log: {exp_log_file}")
                        results_queue.put((True, gpu_id, experiment_name, duration, exp_log_file, cmd))
                    else:
                        print(f"❌ [GPU {gpu_id}] Failed: {experiment_name} ({duration:.1f}s)")
                        print(f"🔍 [GPU {gpu_id}] Last stdout: {result.stdout[-200:] if result.stdout else 'None'}")
                        print(f"🔍 [GPU {gpu_id}] Last stderr: {result.stderr[-200:] if result.stderr else 'None'}")
                        print(f"📝 [GPU {gpu_id}] Full error log: {exp_log_file}")
                        results_queue.put((False, gpu_id, experiment_name, duration, exp_log_file, cmd))
                        
                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = f"Exception: {str(e)}"
                    
                    # Save exception to log file
                    with open(exp_log_file, 'w') as f:
                        f.write(f"Experiment: {experiment_name}\n")
                        f.write(f"GPU: {gpu_id}\n")
                        f.write(f"Command: {' '.join(cmd)}\n")
                        f.write(f"Start time: {datetime.fromtimestamp(start_time)}\n")
                        f.write(f"Duration: {duration:.1f}s\n")
                        f.write("=" * 80 + "\n")
                        f.write("EXCEPTION:\n")
                        f.write(error_msg)
                    
                    print(f"💥 [GPU {gpu_id}] Exception: {experiment_name} - {str(e)}")
                    print(f"📝 [GPU {gpu_id}] Exception log: {exp_log_file}")
                    results_queue.put((False, gpu_id, experiment_name, duration, exp_log_file, cmd))
                
                # Mark job as done
                job_queue.task_done()
                
            except queue.Empty:
                # No more jobs, put GPU back and exit
                gpu_queue.put(gpu_id)
                break
                
            finally:
                # Always return the GPU to the pool
                gpu_queue.put(gpu_id)
                
        except queue.Empty:
            # No GPUs available, exit worker
            break

def generate_experiments(models, methods, concepts, base_args=None):
    """Generate all experiment combinations."""
    if base_args is None:
        base_args = []
    
    experiments = []
    
    for model, method, concept in product(models, methods, concepts):
        cmd = [
            'python', 'scripts/run/run_full_pipeline.py',
            '-m', model,
            '-M', method,
            '-c', concept
        ] + base_args
        
        exp_name = f"{method}_{concept}_{model}"
        experiments.append((cmd, exp_name))
    
    return experiments

def main():
    parser = argparse.ArgumentParser(description='Run steering experiments in parallel across multiple GPUs')
    
    # Experiment specification
    parser.add_argument('--models', nargs='+', default=['qwen25-7b'],
                       help='Models to run (default: qwen25-7b)')
    parser.add_argument('--methods', nargs='+', 
                       default=['caa', 'lat', 'pca', 'dim'],
                       help='Methods to run')
    parser.add_argument('--concepts', nargs='+',
                       default=['explicit_bias', 'implicit_bias', 'hallucination_extrinsic', 
                               'refusal_base'],  # 'hallucination_intrinsic', 
                       help='Concepts to run')
    
    # Resource management
    parser.add_argument('--gpus', nargs='+', type=int,
                       default=list(range(8)),  # GPUs 0-7
                       help='GPU IDs to use (default: 0-7)')
    parser.add_argument('--max-concurrent', type=int, default=None,
                       help='Maximum concurrent jobs (default: number of GPUs)')
    
    # Experiment options
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (faster, smaller datasets)')
    parser.add_argument('--skip-ood', action='store_true',
                       help='Skip OOD evaluation')
    parser.add_argument('--skip-metrics', action='store_true', 
                       help='Skip concept metrics')
    parser.add_argument('--experiments-dir', default='experiments',
                       help='Experiments directory')
    
    # Job management  
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    parser.add_argument('--resume', action='store_true',
                       help='Skip experiments that already exist')
    parser.add_argument('--log-file', default=None,
                       help='Log file for results (default: parallel_experiments_TIMESTAMP.log)')
    
    args = parser.parse_args()
    
    # Set up logging directories
    if args.log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_file = f'parallel_experiments_{timestamp}.log'
    
    # Create logs directory for individual experiment logs
    log_dir = f'experiment_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Build base arguments for run_full_pipeline.py
    base_args = []
    if args.debug:
        base_args.append('--debug')
    if args.skip_ood:
        base_args.append('--skip-ood')  
    if args.skip_metrics:
        base_args.append('--skip-metrics')
    if args.experiments_dir != 'experiments':
        base_args.extend(['--experiments-dir', args.experiments_dir])
    
    # Generate all experiments
    experiments = generate_experiments(args.models, args.methods, args.concepts, base_args)
    
    print(f"🎯 Generated {len(experiments)} experiments")
    print(f"🖥️  Using GPUs: {args.gpus}")
    print(f"⚡ Max concurrent: {args.max_concurrent or len(args.gpus)}")
    print(f"📝 Summary log: {args.log_file}")
    print(f"📁 Individual logs: {log_dir}/")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN - Would execute:")
        for i, (cmd, name) in enumerate(experiments):
            gpu_id = args.gpus[i % len(args.gpus)]
            print(f"   [GPU {gpu_id}] {name}: {' '.join(cmd)}")
        return
    
    # Filter existing experiments if resuming
    if args.resume:
        print("🔄 Checking for existing experiments...")
        # TODO: Add logic to check if experiment already exists
        # For now, just run all experiments
    
    # Set up queues for dynamic GPU assignment
    gpu_queue = queue.Queue()
    job_queue = queue.Queue()
    results_queue = queue.Queue()
    
    # Fill GPU queue
    for gpu_id in args.gpus:
        gpu_queue.put(gpu_id)
    
    # Fill job queue
    for cmd, exp_name in experiments:
        job_queue.put((cmd, exp_name))
    
    # Start worker threads (one per GPU)
    workers = []
    for _ in args.gpus:
        worker = threading.Thread(target=gpu_worker, args=(gpu_queue, job_queue, log_dir, results_queue))
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    print(f"🚀 Starting {len(experiments)} experiments with {len(args.gpus)} GPUs")
    print("=" * 80)
    
    # Process results as they complete
    completed = 0
    failed = 0
    
    with open(args.log_file, 'w') as log_file:
        log_file.write(f"Parallel Experiments Log - Started {datetime.now()}\n")
        log_file.write(f"Total experiments: {len(experiments)}\n")
        log_file.write(f"GPUs: {args.gpus}\n")
        log_file.write("=" * 80 + "\n\n")
        
        # Collect results
        while completed + failed < len(experiments):
            try:
                success, gpu_id, exp_name, duration, log_file_path, cmd = results_queue.get(timeout=10)
                
                if success:
                    completed += 1
                    status = "✅ COMPLETED"
                else:
                    failed += 1
                    status = "❌ FAILED"
                
                # Log to summary file
                log_entry = f"[{datetime.now()}] {status} - GPU {gpu_id} - {exp_name} ({duration:.1f}s)\n"
                log_file.write(log_entry)
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write(f"Detailed log: {log_file_path}\n")
                log_file.write("-" * 40 + "\n")
                log_file.flush()
                
                # Progress update
                total_done = completed + failed
                print(f"📊 Progress: {total_done}/{len(experiments)} ({completed} ✅, {failed} ❌)")
                
            except queue.Empty:
                print("⏳ Waiting for experiments to complete...")
                continue
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏁 All experiments completed!")
    print(f"✅ Successful: {completed}")
    print(f"❌ Failed: {failed}")
    print(f"📝 Summary log: {args.log_file}")
    print(f"📁 Individual logs: {log_dir}/")
    
    if failed > 0:
        print(f"\n❗ {failed} experiments failed. Check individual log files in {log_dir}/ for full details.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
