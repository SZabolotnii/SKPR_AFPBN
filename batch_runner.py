#!/usr/bin/env python3
"""
Batch Experiment Runner for AFPBN

This script provides automated batch execution of AFPBN experiments with
comprehensive logging, error handling, and results aggregation. It supports
reproducing all results from the paper systematically.

Features:
1. Parallel experiment execution
2. Comprehensive error handling and recovery  
3. Real-time progress monitoring
4. Automatic results aggregation and comparison
5. Performance validation against expected results

Usage:
    python batch_runner.py --config_file configs/table_4.json
    python batch_runner.py --reproduce_table 4 --parallel_jobs 2
    python batch_runner.py --reproduce_all_tables --validate_results
"""

import argparse
import json
import subprocess
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import signal
from dataclasses import dataclass, asdict

# Import our configuration system
try:
    from experiment_configs import (
        ExperimentSpec, PaperExperimentConfigs, 
        ExperimentConfigManager, TableType
    )
except ImportError:
    print("Error: experiment_configs.py not found. Make sure it's in the same directory.")
    sys.exit(1)

@dataclass
class ExperimentResult:
    """Container for experiment execution results"""
    spec: ExperimentSpec
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    result_file: Optional[Path] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

class BatchExperimentRunner:
    """Manages batch execution of AFPBN experiments"""
    
    def __init__(self, results_dir: Path = Path("./results"), 
                 logs_dir: Path = Path("./logs"),
                 parallel_jobs: int = 1):
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir)
        self.parallel_jobs = parallel_jobs
        self.experiment_results: List[ExperimentResult] = []
        self._shutdown_requested = False
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self._shutdown_requested = True
    
    def _build_command(self, spec: ExperimentSpec) -> List[str]:
        """Build command-line arguments for an experiment"""
        cmd = [
            sys.executable, "main_afpbn.py",
            "--data_type", spec.data_type,
            "--case", str(spec.case),
            "--experiment_type", spec.experiment_type,
            "--n_basis", str(spec.n_basis),
            "--epochs", str(spec.epochs),
            "--lambda_recon", str(spec.lambda_recon),
            "--save_model"
        ]
        
        if not spec.use_both_channels:
            cmd.append("--single_channel")
        
        if spec.basis_powers:
            cmd.extend(["--basis_powers"] + [str(p) for p in spec.basis_powers])
        
        return cmd
    
    def _run_single_experiment(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run a single experiment and capture results"""
        result = ExperimentResult(
            spec=spec,
            success=False,
            start_time=datetime.now()
        )
        
        try:
            print(f"🚀 Starting: {spec.name}")
            print(f"   Description: {spec.description}")
            
            # Build command
            cmd = self._build_command(spec)
            
            # Create log files
            log_prefix = self.logs_dir / f"{spec.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            stdout_log = log_prefix.with_suffix(".out")
            stderr_log = log_prefix.with_suffix(".err")
            
            # Run experiment
            with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
                process = subprocess.run(
                    cmd,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )
            
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Read logs
            with open(stdout_log, 'r') as f:
                result.stdout = f.read()
            with open(stderr_log, 'r') as f:
                result.stderr = f.read()
            
            if process.returncode == 0:
                result.success = True
                
                # Find and load result file
                result_files = list(self.results_dir.glob(f"*{spec.data_type}_case{spec.case}*{spec.experiment_type}*_results.json"))
                if result_files:
                    # Get the most recent result file
                    result.result_file = max(result_files, key=lambda p: p.stat().st_mtime)
                    
                    try:
                        with open(result.result_file, 'r') as f:
                            result_data = json.load(f)
                            result.metrics = result_data.get('final_metrics', {})
                    except Exception as e:
                        print(f"⚠️  Warning: Could not load result file for {spec.name}: {e}")
                
                print(f"✅ Completed: {spec.name} ({result.duration_seconds:.1f}s)")
                if result.metrics:
                    mae = result.metrics.get('mae_ns', 'N/A')
                    val_rmse = result.metrics.get('val_rmse', 'N/A')
                    print(f"   Results: MAE={mae:.4f}ns, Val RMSE={val_rmse:.4f}")
            else:
                result.error_message = f"Process exited with code {process.returncode}"
                print(f"❌ Failed: {spec.name} - {result.error_message}")
                if result.stderr:
                    print(f"   Error: {result.stderr[-200:]}")  # Last 200 chars
        
        except subprocess.TimeoutExpired:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.error_message = "Experiment timed out (2 hours)"
            print(f"⏰ Timeout: {spec.name}")
        
        except Exception as e:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.error_message = str(e)
            print(f"💥 Error: {spec.name} - {e}")
        
        return result
    
    def run_experiments(self, experiments: List[ExperimentSpec], 
                       validate_results: bool = False) -> List[ExperimentResult]:
        """Run a batch of experiments with optional parallel execution"""
        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENT EXECUTION")
        print(f"Total experiments: {len(experiments)}")
        print(f"Parallel jobs: {self.parallel_jobs}")
        print(f"Results directory: {self.results_dir}")
        print(f"Logs directory: {self.logs_dir}")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        results = []
        
        if self.parallel_jobs == 1:
            # Sequential execution
            for i, spec in enumerate(experiments, 1):
                if self._shutdown_requested:
                    print("Shutdown requested, stopping execution...")
                    break
                
                print(f"\n[{i}/{len(experiments)}] Running experiment sequentially...")
                result = self._run_single_experiment(spec)
                results.append(result)
                
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
                # Submit all experiments
                future_to_spec = {
                    executor.submit(self._run_single_experiment, spec): spec 
                    for spec in experiments
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_spec):
                    if self._shutdown_requested:
                        print("Shutdown requested, cancelling remaining experiments...")
                        for f in future_to_spec:
                            f.cancel()
                        break
                    
                    completed += 1
                    result = future.result()
                    results.append(result)
                    
                    print(f"\n[{completed}/{len(experiments)}] Parallel execution progress: "
                          f"{completed/len(experiments)*100:.1f}% complete")
        
        # Sort results by experiment name for consistency
        results.sort(key=lambda r: r.spec.name)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"\n{'='*80}")
        print(f"BATCH EXECUTION SUMMARY")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {failed}/{len(results)}")
        print(f"{'='*80}")
        
        # Validate results if requested
        if validate_results:
            self._validate_results(results)
        
        # Save batch summary
        self._save_batch_summary(results, total_time)
        
        self.experiment_results = results
        return results
    
    def _validate_results(self, results: List[ExperimentResult]):
        """Validate experimental results against expected values"""
        print(f"\n{'='*60}")
        print(f"RESULTS VALIDATION")
        print(f"{'='*60}")
        
        validation_passed = 0
        validation_failed = 0
        tolerance = 0.05  # 5% tolerance for expected results
        
        for result in results:
            if not result.success or not result.metrics:
                continue
            
            spec = result.spec
            if spec.expected_mae is None:
                continue
            
            actual_mae = result.metrics.get('mae_ns', float('inf'))
            expected_mae = spec.expected_mae
            
            error_pct = abs(actual_mae - expected_mae) / expected_mae * 100
            
            if error_pct <= tolerance * 100:
                status = "✅ PASS"
                validation_passed += 1
            else:
                status = "❌ FAIL"
                validation_failed += 1
            
            print(f"{status} {spec.name}")
            print(f"      Expected MAE: {expected_mae:.4f} ns")
            print(f"      Actual MAE:   {actual_mae:.4f} ns")
            print(f"      Error:        {error_pct:.1f}%")
            print()
        
        print(f"Validation Summary:")
        print(f"  Passed: {validation_passed}")
        print(f"  Failed: {validation_failed}")
        print(f"  Success Rate: {validation_passed/(validation_passed+validation_failed)*100:.1f}%")
    
    def _save_batch_summary(self, results: List[ExperimentResult], total_time: float):
        """Save batch execution summary to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.results_dir / f"batch_summary_{timestamp}.json"
        
        summary_data = {
            "execution_info": {
                "timestamp": timestamp,
                "total_experiments": len(results),
                "successful_experiments": sum(1 for r in results if r.success),
                "failed_experiments": sum(1 for r in results if not r.success),
                "total_time_seconds": total_time,
                "parallel_jobs": self.parallel_jobs
            },
            "experiment_results": []
        }
        
        for result in results:
            result_dict = {
                "name": result.spec.name,
                "description": result.spec.description,
                "success": result.success,
                "duration_seconds": result.duration_seconds,
                "error_message": result.error_message,
                "result_file": str(result.result_file) if result.result_file else None,
                "metrics": result.metrics,
                "config": asdict(result.spec)
            }
            summary_data["experiment_results"].append(result_dict)
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"Batch summary saved to: {summary_path}")
    
    def generate_comparison_table(self, results: List[ExperimentResult]) -> str:
        """Generate a comparison table of all results"""
        table_lines = []
        table_lines.append("="*100)
        table_lines.append("BATCH EXPERIMENT RESULTS COMPARISON")
        table_lines.append("="*100)
        table_lines.append(f"{'Experiment Name':<35} {'Status':<8} {'MAE (ns)':<10} {'Val RMSE':<10} {'Time (s)':<10} {'Expected':<10}")
        table_lines.append("-"*100)
        
        for result in results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            
            if result.metrics:
                mae_str = f"{result.metrics.get('mae_ns', 0):.4f}"
                rmse_str = f"{result.metrics.get('val_rmse', 0):.4f}"
            else:
                mae_str = "N/A"
                rmse_str = "N/A"
            
            time_str = f"{result.duration_seconds:.1f}" if result.duration_seconds else "N/A"
            expected_str = f"{result.spec.expected_mae:.4f}" if result.spec.expected_mae else "N/A"
            
            table_lines.append(
                f"{result.spec.name[:34]:<35} {status:<8} {mae_str:<10} {rmse_str:<10} {time_str:<10} {expected_str:<10}"
            )
        
        table_lines.append("="*100)
        return "\n".join(table_lines)

def create_paper_reproduction_batches() -> Dict[str, List[ExperimentSpec]]:
    """Create experiment batches for reproducing paper results"""
    all_configs = PaperExperimentConfigs.get_all_paper_configs()
    
    # Define batches for systematic reproduction
    batches = {
        "main_results": all_configs.get("table_4", []),  # Table 4: Main comparison
        "complex_scenarios": all_configs.get("table_5", []),  # Table 5: Interpolation/extrapolation
        "ablation_study": all_configs.get("table_6", []),  # Table 6: Component ablation
        "basis_analysis": all_configs.get("table_7", []),  # Table 7: Number of basis functions
        "state_of_the_art": all_configs.get("state_of_the_art", [])  # Best configurations
    }
    
    return batches

def main():
    """Main function for batch experiment execution"""
    parser = argparse.ArgumentParser(
        description="AFPBN Batch Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments from config file
  python batch_runner.py --config_file configs/table_4.json
  
  # Reproduce Table 4 with validation
  python batch_runner.py --reproduce_table 4 --validate_results
  
  # Run all paper experiments with 2 parallel jobs
  python batch_runner.py --reproduce_all_tables --parallel_jobs 2
  
  # Run specific batch
  python batch_runner.py --batch main_results --parallel_jobs 1
        """
    )
    
    # Input options
    parser.add_argument('--config_file', type=str,
                       help='JSON configuration file with experiment specs')
    parser.add_argument('--reproduce_table', type=str, choices=['4', '5', '6', '7'],
                       help='Reproduce specific table from the paper')
    parser.add_argument('--reproduce_all_tables', action='store_true',
                       help='Reproduce all tables from the paper')
    parser.add_argument('--batch', type=str, 
                       choices=['main_results', 'complex_scenarios', 'ablation_study', 
                               'basis_analysis', 'state_of_the_art'],
                       help='Run predefined experiment batch')
    
    # Execution options
    parser.add_argument('--parallel_jobs', type=int, default=1,
                       help='Number of parallel experiments to run')
    parser.add_argument('--validate_results', action='store_true',
                       help='Validate results against expected values')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--logs_dir', type=str, default='./logs',
                       help='Directory to save execution logs')
    
    # Analysis options
    parser.add_argument('--generate_comparison_table', action='store_true',
                       help='Generate and print comparison table after execution')
    parser.add_argument('--timeout_minutes', type=int, default=120,
                       help='Timeout for individual experiments in minutes')
    
    args = parser.parse_args()
    
    # Determine experiments to run
    experiments = []
    
    if args.config_file:
        # Load from configuration file
        config_manager = ExperimentConfigManager()
        config_file = Path(args.config_file)
        if not config_file.exists():
            print(f"Error: Configuration file {config_file} not found")
            return
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for exp_dict in config_data["experiments"]:
                experiments.append(ExperimentSpec(**exp_dict))
        except Exception as e:
            print(f"Error loading config file: {e}")
            return
    
    elif args.reproduce_table:
        # Reproduce specific table
        table_map = {'4': 'table_4', '5': 'table_5', '6': 'table_6', '7': 'table_7'}
        table_key = table_map[args.reproduce_table]
        all_configs = PaperExperimentConfigs.get_all_paper_configs()
        experiments = all_configs.get(table_key, [])
        
    elif args.reproduce_all_tables:
        # Reproduce all tables
        all_configs = PaperExperimentConfigs.get_all_paper_configs()
        for table_experiments in all_configs.values():
            experiments.extend(table_experiments)
        
        # Remove duplicates based on experiment name
        seen_names = set()
        unique_experiments = []
        for exp in experiments:
            if exp.name not in seen_names:
                unique_experiments.append(exp)
                seen_names.add(exp.name)
        experiments = unique_experiments
        
    elif args.batch:
        # Run predefined batch
        batches = create_paper_reproduction_batches()
        experiments = batches.get(args.batch, [])
        
    else:
        print("Error: Must specify experiments to run")
        print("Use --config_file, --reproduce_table, --reproduce_all_tables, or --batch")
        return
    
    if not experiments:
        print("No experiments found to run")
        return
    
    print(f"Found {len(experiments)} experiments to run")
    
    # Create batch runner
    runner = BatchExperimentRunner(
        results_dir=Path(args.results_dir),
        logs_dir=Path(args.logs_dir),
        parallel_jobs=args.parallel_jobs
    )
    
    # Run experiments
    try:
        results = runner.run_experiments(experiments, args.validate_results)
        
        # Generate comparison table if requested
        if args.generate_comparison_table:
            comparison_table = runner.generate_comparison_table(results)
            print(f"\n{comparison_table}")
        
        # Print final summary
        successful = sum(1 for r in results if r.success)
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   Total experiments: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        print(f"   Success rate: {successful/len(results)*100:.1f}%")
        
        if successful > 0:
            print(f"\n📊 Analysis commands:")
            print(f"   python analysis_utils.py --results_dir {args.results_dir} --analysis_type comprehensive")
            if args.reproduce_table:
                print(f"   python analysis_utils.py --results_dir {args.results_dir} --analysis_type ablation_comparison")
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Batch execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())