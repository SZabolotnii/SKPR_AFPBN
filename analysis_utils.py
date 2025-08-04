#!/usr/bin/env python3
"""
Analysis and Visualization Utilities for AFPBN Experiments

This module provides utilities for analyzing and visualizing results from AFPBN
experiments and ablation studies, supporting the research described in:
"Adaptive Hybrid Neural Networks Based on Decomposition in Space with Generating Element"

Features:
1. Ablation study results analysis and comparison
2. Performance visualization and statistical analysis
3. Alpha evolution tracking and basis function analysis
4. Comprehensive result reporting

Usage:
    python analysis_utils.py --results_dir ./results --analysis_type ablation_comparison
    python analysis_utils.py --results_file results.json --analysis_type single_experiment
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import scipy.stats
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AFPBNAnalyzer:
    """Class for analyzing AFPBN experimental results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True, parents=True)
    
    def load_results(self, pattern: str = "*_results.json") -> Dict[str, Dict]:
        """Load all result files matching the pattern"""
        results = {}
        for file_path in self.results_dir.glob(pattern):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results[file_path.stem] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return results
    
    def analyze_single_experiment(self, result_data: Dict) -> Dict:
        """Analyze results from a single experiment"""
        analysis = {
            'experiment_info': {
                'name': result_data.get('experiment_name', 'Unknown'),
                'config': result_data.get('config', {}),
                'final_alpha': result_data.get('final_alpha'),
                'final_powers': result_data.get('final_powers')
            },
            'performance_metrics': result_data.get('final_metrics', {}),
            'training_analysis': {}
        }
        
        # Analyze training history if available
        history = result_data.get('history', {})
        if history:
            analysis['training_analysis'] = {
                'total_epochs': len(history.get('train_loss', [])),
                'final_train_loss': history.get('train_loss', [])[-1] if history.get('train_loss') else None,
                'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
                'best_val_rmse': min(history.get('val_rmse', [])) if history.get('val_rmse') else None,
                'convergence_epoch': self._find_convergence_epoch(history.get('val_loss', [])),
                'alpha_stability': self._analyze_alpha_stability(history.get('alpha_values', []))
            }
        
        return analysis
    
    def _find_convergence_epoch(self, val_losses: List[float], patience: int = 5) -> Optional[int]:
        """Find the epoch where the model converged (stopped improving significantly)"""
        if len(val_losses) < patience:
            return None
        
        best_loss = float('inf')
        no_improve_count = 0
        
        for epoch, loss in enumerate(val_losses):
            if loss < best_loss * 0.999:  # 0.1% improvement threshold
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                return epoch - patience + 1
        
        return None
    
    def _analyze_alpha_stability(self, alpha_values: List[float]) -> Dict:
        """Analyze the stability of alpha parameter evolution"""
        if not alpha_values:
            return {}
        
        alpha_array = np.array(alpha_values)
        return {
            'mean': float(np.mean(alpha_array)),
            'std': float(np.std(alpha_array)),
            'min': float(np.min(alpha_array)),
            'max': float(np.max(alpha_array)),
            'final': float(alpha_array[-1]),
            'stability_score': float(1.0 / (1.0 + np.std(alpha_array)))  # Higher is more stable
        }
    
    def compare_ablation_results(self, ablation_data: Dict[str, Dict]) -> pd.DataFrame:
        """Compare results from ablation studies"""
        comparison_data = []
        
        for experiment_type, results in ablation_data.items():
            if 'error' in results:
                continue  # Skip failed experiments
                
            metrics = results.get('final_metrics', {})
            config = results.get('config', {})
            
            row = {
                'Experiment': experiment_type.replace('_', ' ').title(),
                'MAE (ns)': metrics.get('mae_ns', np.nan),
                'RMSE (ns)': metrics.get('rmse_ns', np.nan),
                'Val RMSE': metrics.get('val_rmse', np.nan),
                'Best Val Loss': metrics.get('best_val_loss', np.nan),
                'Data Type': config.get('data_type', 'unknown'),
                'Case': config.get('case', 'unknown'),
                'N Basis': config.get('n_basis', 0),
                'Use Both Channels': config.get('use_both_channels', False),
                'Lambda Recon': config.get('lambda_recon', 0.0),
                'Final Alpha': results.get('final_alpha')
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('MAE (ns)') if not df.empty else df
    
    def plot_training_history(self, result_data: Dict, save_path: Optional[Path] = None):
        """Plot training history for a single experiment"""
        history = result_data.get('history', {})
        if not history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        experiment_name = result_data.get('experiment_name', 'Unknown')
        
        # Loss plot
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Training Loss', alpha=0.8)
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', alpha=0.8)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Progress')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE plot
        if 'val_rmse' in history:
            axes[0, 1].plot(history['val_rmse'], color='red', alpha=0.8)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Validation RMSE')
            axes[0, 1].set_title('Validation RMSE Evolution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'], color='green', alpha=0.8)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Alpha evolution plot
        if 'alpha_values' in history and history['alpha_values']:
            axes[1, 1].plot(history['alpha_values'], color='purple', alpha=0.8)
            axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                             label='Forbidden α=0.5')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Alpha Value')
            axes[1, 1].set_title('Adaptive Alpha Evolution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No adaptive alpha\nin this experiment', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Alpha Evolution (N/A)')
        
        plt.suptitle(f'Training Analysis: {experiment_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_ablation_comparison(self, comparison_df: pd.DataFrame, 
                                save_path: Optional[Path] = None):
        """Plot comparison of ablation study results"""
        if comparison_df.empty:
            print("No data available for ablation comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # MAE comparison
        df_sorted = comparison_df.sort_values('MAE (ns)')
        bars1 = axes[0, 0].bar(range(len(df_sorted)), df_sorted['MAE (ns)'], 
                              alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Experiment')
        axes[0, 0].set_ylabel('MAE (ns)')
        axes[0, 0].set_title('Mean Absolute Error Comparison')
        axes[0, 0].set_xticks(range(len(df_sorted)))
        axes[0, 0].set_xticklabels(df_sorted['Experiment'], rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add values on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            if not np.isnan(height):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
        
        # Val RMSE comparison
        bars2 = axes[0, 1].bar(range(len(df_sorted)), df_sorted['Val RMSE'], 
                              alpha=0.7, color='lightcoral')
        axes[0, 1].set_xlabel('Experiment')
        axes[0, 1].set_ylabel('Validation RMSE')
        axes[0, 1].set_title('Validation RMSE Comparison')
        axes[0, 1].set_xticks(range(len(df_sorted)))
        axes[0, 1].set_xticklabels(df_sorted['Experiment'], rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add values on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if not np.isnan(height):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
        
        # Performance improvement over baseline
        baseline_mae = df_sorted[df_sorted['Experiment'].str.contains('No Basis', case=False)]['MAE (ns)']
        if not baseline_mae.empty:
            baseline_value = baseline_mae.iloc[0]
            improvements = ((baseline_value - df_sorted['MAE (ns)']) / baseline_value * 100)
            
            bars3 = axes[1, 0].bar(range(len(df_sorted)), improvements, 
                                  alpha=0.7, color='lightgreen')
            axes[1, 0].set_xlabel('Experiment')
            axes[1, 0].set_ylabel('Improvement over Baseline (%)')
            axes[1, 0].set_title('Performance Improvement Analysis')
            axes[1, 0].set_xticks(range(len(df_sorted)))
            axes[1, 0].set_xticklabels(df_sorted['Experiment'], rotation=45, ha='right')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, bar in enumerate(bars3):
                height = bar.get_height()
                if not np.isnan(height):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}%', ha='center', 
                                   va='bottom' if height >= 0 else 'top')
        
        # Alpha values distribution (if available)
        alpha_data = comparison_df.dropna(subset=['Final Alpha'])
        if not alpha_data.empty:
            axes[1, 1].scatter(alpha_data['Final Alpha'], alpha_data['MAE (ns)'], 
                             alpha=0.7, s=100)
            axes[1, 1].set_xlabel('Final Alpha Value')
            axes[1, 1].set_ylabel('MAE (ns)')
            axes[1, 1].set_title('Alpha vs Performance Relationship')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add experiment labels
            for _, row in alpha_data.iterrows():
                axes[1, 1].annotate(row['Experiment'][:10] + '...', 
                                  (row['Final Alpha'], row['MAE (ns)']),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.7)
        else:
            axes[1, 1].text(0.5, 0.5, 'No alpha data\navailable', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Alpha Analysis (N/A)')
        
        plt.suptitle('Ablation Study Results Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Ablation comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, results_data: Dict[str, Dict], 
                       save_path: Optional[Path] = None) -> str:
        """Generate a comprehensive analysis report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("AFPBN EXPERIMENTAL RESULTS ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total experiments analyzed: {len(results_data)}")
        report_lines.append("")
        
        # Overall statistics
        all_metrics = []
        for results in results_data.values():
            if 'final_metrics' in results:
                all_metrics.append(results['final_metrics'])
        
        if all_metrics:
            mae_values = [m.get('mae_ns', np.nan) for m in all_metrics]
            val_rmse_values = [m.get('val_rmse', np.nan) for m in all_metrics]
            
            mae_values = [x for x in mae_values if not np.isnan(x)]
            val_rmse_values = [x for x in val_rmse_values if not np.isnan(x)]
            
            if mae_values:
                report_lines.append("OVERALL PERFORMANCE STATISTICS")
                report_lines.append("-" * 40)
                report_lines.append(f"MAE Statistics:")
                report_lines.append(f"  Best (minimum): {min(mae_values):.4f} ns")
                report_lines.append(f"  Mean: {np.mean(mae_values):.4f} ns")
                report_lines.append(f"  Std: {np.std(mae_values):.4f} ns")
                report_lines.append(f"  Worst (maximum): {max(mae_values):.4f} ns")
                report_lines.append("")
            
            if val_rmse_values:
                report_lines.append(f"Validation RMSE Statistics:")
                report_lines.append(f"  Best (minimum): {min(val_rmse_values):.4f}")
                report_lines.append(f"  Mean: {np.mean(val_rmse_values):.4f}")
                report_lines.append(f"  Std: {np.std(val_rmse_values):.4f}")
                report_lines.append(f"  Worst (maximum): {max(val_rmse_values):.4f}")
                report_lines.append("")
        
        # Individual experiment analysis
        report_lines.append("INDIVIDUAL EXPERIMENT RESULTS")
        report_lines.append("-" * 40)
        
        # Sort experiments by MAE performance
        sorted_experiments = sorted(
            results_data.items(),
            key=lambda x: x[1].get('final_metrics', {}).get('mae_ns', float('inf'))
        )
        
        for i, (exp_name, results) in enumerate(sorted_experiments, 1):
            if 'error' in results:
                report_lines.append(f"{i}. {exp_name} - FAILED: {results['error']}")
                continue
            
            metrics = results.get('final_metrics', {})
            config = results.get('config', {})
            
            report_lines.append(f"{i}. {exp_name}")
            report_lines.append(f"   Configuration: {config.get('experiment_type', 'unknown')} "
                              f"({config.get('data_type', 'unknown')}, case {config.get('case', 'unknown')})")
            report_lines.append(f"   MAE: {metrics.get('mae_ns', 'N/A'):.4f} ns")
            report_lines.append(f"   Val RMSE: {metrics.get('val_rmse', 'N/A'):.4f}")
            
            if results.get('final_alpha') is not None:
                report_lines.append(f"   Final Alpha: {results['final_alpha']:.3f}")
            
            if results.get('final_powers'):
                powers_str = ', '.join([f"{p:.3f}" for p in results['final_powers'][:3]])
                report_lines.append(f"   Final Powers: [{powers_str}...]")
            
            report_lines.append("")
        
        # Best performing experiments
        report_lines.append("TOP PERFORMING EXPERIMENTS")
        report_lines.append("-" * 40)
        
        top_3 = sorted_experiments[:3]
        for i, (exp_name, results) in enumerate(top_3, 1):
            if 'error' in results:
                continue
            metrics = results.get('final_metrics', {})
            improvement = ""
            if i > 1:
                baseline_mae = top_3[0][1].get('final_metrics', {}).get('mae_ns', 0)
                current_mae = metrics.get('mae_ns', 0)
                if baseline_mae > 0:
                    diff_pct = ((current_mae - baseline_mae) / baseline_mae) * 100
                    improvement = f" ({diff_pct:+.1f}%)"
            
            report_lines.append(f"{i}. {exp_name}")
            report_lines.append(f"   MAE: {metrics.get('mae_ns', 'N/A'):.4f} ns{improvement}")
            report_lines.append(f"   Val RMSE: {metrics.get('val_rmse', 'N/A'):.4f}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Analysis report saved to {save_path}")
        
        return report_text
    
    def statistical_significance_test(self, results_data: Dict[str, Dict]) -> Dict:
        """Perform statistical significance tests between experiments"""
        # This would require multiple runs of each experiment to be meaningful
        # For now, return a placeholder structure
        
        experiments_with_metrics = {}
        for exp_name, results in results_data.items():
            if 'final_metrics' in results and 'error' not in results:
                experiments_with_metrics[exp_name] = results['final_metrics']
        
        significance_results = {
            'note': 'Statistical significance testing requires multiple runs of each experiment',
            'available_experiments': list(experiments_with_metrics.keys()),
            'metrics_available': len(experiments_with_metrics) > 0
        }
        
        return significance_results

def analyze_results_directory(results_dir: Path, analysis_type: str = "comprehensive"):
    """Analyze all results in a directory"""
    analyzer = AFPBNAnalyzer(results_dir)
    
    if analysis_type == "ablation_comparison":
        # Look for ablation study files
        ablation_files = list(results_dir.glob("ablation_study_*.json"))
        
        if not ablation_files:
            print("No ablation study files found")
            return
        
        # Load the most recent ablation study
        latest_ablation = max(ablation_files, key=lambda p: p.stat().st_mtime)
        print(f"Analyzing ablation study: {latest_ablation}")
        
        with open(latest_ablation, 'r') as f:
            ablation_data = json.load(f)
        
        # Create comparison DataFrame
        comparison_df = analyzer.compare_ablation_results(ablation_data)
        print("\nAblation Study Results:")
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = analyzer.figures_dir / f"ablation_comparison_{timestamp}.png"
        analyzer.plot_ablation_comparison(comparison_df, plot_path)
        
        # Save comparison table
        csv_path = analyzer.figures_dir / f"ablation_results_{timestamp}.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"Results table saved to {csv_path}")
        
    elif analysis_type == "comprehensive":
        # Load all result files
        all_results = analyzer.load_results()
        
        if not all_results:
            print("No result files found")
            return
        
        print(f"Loaded {len(all_results)} experiment results")
        
        # Generate comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = analyzer.figures_dir / f"analysis_report_{timestamp}.txt"
        report = analyzer.generate_report(all_results, report_path)
        print("\nAnalysis Report:")
        print(report)
        
        # Plot training histories for recent experiments
        recent_results = sorted(all_results.items(), 
                              key=lambda x: x[0], reverse=True)[:3]
        
        for exp_name, results in recent_results:
            if 'history' in results:
                plot_path = analyzer.figures_dir / f"training_history_{exp_name}.png"
                analyzer.plot_training_history(results, plot_path)

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(
        description="Analyze AFPBN experimental results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing result files')
    parser.add_argument('--analysis_type', type=str, 
                       choices=['comprehensive', 'ablation_comparison', 'single_experiment'],
                       default='comprehensive',
                       help='Type of analysis to perform')
    parser.add_argument('--results_file', type=str,
                       help='Specific result file to analyze (for single_experiment)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return
    
    if args.analysis_type == 'single_experiment':
        if not args.results_file:
            print("--results_file is required for single_experiment analysis")
            return
        
        results_file = Path(args.results_file)
        if not results_file.exists():
            print(f"Results file {results_file} does not exist")
            return
        
        with open(results_file, 'r') as f:
            result_data = json.load(f)
        
        analyzer = AFPBNAnalyzer(results_dir)
        analysis = analyzer.analyze_single_experiment(result_data)
        
        print("Single Experiment Analysis:")
        print(json.dumps(analysis, indent=2))
        
        # Plot training history
        plot_path = analyzer.figures_dir / f"single_experiment_analysis.png"
        analyzer.plot_training_history(result_data, plot_path)
        
    else:
        analyze_results_directory(results_dir, args.analysis_type)

if __name__ == "__main__":
    main()