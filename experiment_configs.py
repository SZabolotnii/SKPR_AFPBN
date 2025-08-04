#!/usr/bin/env python3
"""
Experiment Configuration System for AFPBN

This module provides predefined configurations for reproducing all experiments
from the paper "Adaptive Hybrid Neural Networks Based on Decomposition in Space 
with Generating Element: Robust Parameter Estimation for UWB Radar Signals"

Usage:
    python experiment_configs.py --reproduce_table 4
    python experiment_configs.py --reproduce_table 6 --data_type cir --case 1
    python experiment_configs.py --reproduce_all_tables
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum

class TableType(Enum):
    """Tables from the paper to reproduce"""
    TABLE_4 = "table_4"  # Main results comparison (Table 4)
    TABLE_5 = "table_5"  # Complex scenarios (Table 5) 
    TABLE_6 = "table_6"  # Ablation study (Table 6)
    TABLE_7 = "table_7"  # Basis functions analysis (Table 7)
    TABLE_8 = "table_8"  # Classical methods comparison (Table 8)

@dataclass
class ExperimentSpec:
    """Specification for a single experiment"""
    name: str
    data_type: str
    case: int
    experiment_type: str
    n_basis: int = 6
    use_both_channels: bool = True
    adaptive_alpha: bool = True
    lambda_recon: float = 0.1
    basis_powers: List[float] = None
    epochs: int = 100
    description: str = ""
    expected_mae: float = None
    expected_val_rmse: float = None

class PaperExperimentConfigs:
    """Predefined configurations to reproduce paper results"""
    
    @staticmethod
    def get_table_4_configs() -> List[ExperimentSpec]:
        """
        Table 4: Results on UWB dataset for different data types and scenarios
        
        This reproduces the main comparison between Baseline CNN, CNN + Fixed Basis,
        and AFPBN (our method) as reported in Table 4 of the paper.
        """
        configs = []
        
        # CIR Data, Case 1
        configs.extend([
            ExperimentSpec(
                name="baseline_cnn_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="no_basis",
                description="Baseline CNN for CIR data",
                expected_mae=1.0121,
                expected_val_rmse=12.0088
            ),
            ExperimentSpec(
                name="cnn_fixed_basis_cir_case1", 
                data_type="cir",
                case=1,
                experiment_type="fixed_basis",
                basis_powers=[0.5, 0.333, 0.25],
                description="CNN + Fixed Basis for CIR data",
                expected_mae=0.9483,
                expected_val_rmse=9.6881
            ),
            ExperimentSpec(
                name="afpbn_full_cir_case1",
                data_type="cir", 
                case=1,
                experiment_type="full",
                description="Full AFPBN model for CIR data",
                expected_mae=0.8834,
                expected_val_rmse=8.8972
            )
        ])
        
        # Variance Data, Case 1  
        configs.extend([
            ExperimentSpec(
                name="baseline_cnn_variance_case1",
                data_type="variance",
                case=1,
                experiment_type="no_basis",
                description="Baseline CNN for Variance data",
                expected_mae=1.1350,
                expected_val_rmse=8.7457
            ),
            ExperimentSpec(
                name="cnn_fixed_basis_variance_case1",
                data_type="variance",
                case=1, 
                experiment_type="fixed_basis",
                basis_powers=[0.5, 0.333, 0.25],
                description="CNN + Fixed Basis for Variance data",
                expected_mae=1.1432,
                expected_val_rmse=7.6014
            ),
            ExperimentSpec(
                name="afpbn_full_variance_case1",
                data_type="variance",
                case=1,
                experiment_type="full",
                description="Full AFPBN model for Variance data", 
                expected_mae=1.0834,
                expected_val_rmse=7.2156
            )
        ])
        
        return configs
    
    @staticmethod
    def get_table_5_configs() -> List[ExperimentSpec]:
        """
        Table 5: Performance on interpolation and extrapolation scenarios
        
        This reproduces the results showing performance degradation on complex
        scenarios (Cases 2 and 3) as reported in Table 5.
        """
        configs = []
        
        # Baseline CNN on all cases
        for case in [1, 2, 3]:
            configs.append(ExperimentSpec(
                name=f"baseline_cnn_cir_case{case}",
                data_type="cir",
                case=case,
                experiment_type="no_basis",
                description=f"Baseline CNN, Case {case}",
                expected_mae={1: 1.0121, 2: 1.8234, 3: 2.1456}[case]
            ))
        
        # AFPBN on all cases
        for case in [1, 2, 3]:
            configs.append(ExperimentSpec(
                name=f"afpbn_full_cir_case{case}",
                data_type="cir",
                case=case,
                experiment_type="full", 
                description=f"Full AFPBN, Case {case}",
                expected_mae={1: 0.8834, 2: 1.2156, 3: 1.4823}[case]
            ))
            
        return configs
    
    @staticmethod
    def get_table_6_configs() -> List[ExperimentSpec]:
        """
        Table 6: Ablation study of components (CIR, Case 1)
        
        This reproduces the systematic ablation study showing the contribution
        of each component as reported in Table 6.
        """
        return [
            ExperimentSpec(
                name="afpbn_full_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="full",
                description="Full AFPBN model",
                expected_mae=0.8834,
                expected_val_rmse=8.8972
            ),
            ExperimentSpec(
                name="afpbn_no_adaptive_alpha_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="no_adaptive_alpha",
                description="Without adaptive α (fixed α=0.3)",
                expected_mae=0.9234,
                expected_val_rmse=9.4521
            ),
            ExperimentSpec(
                name="afpbn_no_reconstruction_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="no_reconstruction",
                lambda_recon=0.0,
                description="Without reconstruction (λ=0)",
                expected_mae=0.9512,
                expected_val_rmse=10.2134
            ),
            ExperimentSpec(
                name="afpbn_no_basis_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="no_basis",
                description="Without basis features",
                expected_mae=0.9876,
                expected_val_rmse=11.3245
            ),
            ExperimentSpec(
                name="afpbn_single_channel_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="single_channel", 
                use_both_channels=False,
                description="Without both channels (only dynamic)",
                expected_mae=0.9156,
                expected_val_rmse=9.8234
            ),
            ExperimentSpec(
                name="baseline_cnn_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="no_basis",
                description="Baseline CNN",
                expected_mae=1.0121,
                expected_val_rmse=12.0088
            )
        ]
    
    @staticmethod
    def get_table_7_configs() -> List[ExperimentSpec]:
        """
        Table 7: Impact of number of basis functions
        
        This reproduces the analysis of optimal number of basis functions
        as reported in Table 7.
        """
        configs = []
        
        for n_basis in [3, 6, 9, 12]:
            configs.append(ExperimentSpec(
                name=f"afpbn_nbasis{n_basis}_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="full",
                n_basis=n_basis,
                description=f"AFPBN with {n_basis} basis functions",
                expected_mae={3: 0.9234, 6: 0.8834, 9: 0.8798, 12: 0.8812}[n_basis]
            ))
            
        return configs
    
    @staticmethod
    def get_state_of_the_art_configs() -> List[ExperimentSpec]:
        """
        Configurations for achieving state-of-the-art results
        
        These are the best performing configurations from the paper,
        designed to achieve the highest accuracy.
        """
        return [
            ExperimentSpec(
                name="afpbn_sota_cir_case1",
                data_type="cir",
                case=1,
                experiment_type="full",
                n_basis=6,
                use_both_channels=True,
                adaptive_alpha=True,
                lambda_recon=0.1,
                epochs=100,
                description="State-of-the-art AFPBN for CIR data",
                expected_mae=0.8834,
                expected_val_rmse=8.8972
            ),
            ExperimentSpec(
                name="afpbn_sota_variance_case1_2ch",
                data_type="variance",
                case=1,
                experiment_type="full",
                n_basis=6,
                use_both_channels=True,
                adaptive_alpha=True, 
                lambda_recon=0.1,
                epochs=100,
                description="State-of-the-art AFPBN for Variance data (2 channels)",
                expected_mae=1.1432,
                expected_val_rmse=7.6014
            )
        ]
    
    @staticmethod
    def get_all_paper_configs() -> Dict[str, List[ExperimentSpec]]:
        """Get all predefined configurations organized by table"""
        return {
            "table_4": PaperExperimentConfigs.get_table_4_configs(),
            "table_5": PaperExperimentConfigs.get_table_5_configs(), 
            "table_6": PaperExperimentConfigs.get_table_6_configs(),
            "table_7": PaperExperimentConfigs.get_table_7_configs(),
            "state_of_the_art": PaperExperimentConfigs.get_state_of_the_art_configs()
        }

class ExperimentConfigManager:
    """Manager for experiment configurations"""
    
    def __init__(self, config_dir: Path = Path("./configs")):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
    def save_configs_to_file(self, configs: List[ExperimentSpec], 
                           filename: str) -> Path:
        """Save experiment configurations to JSON file"""
        config_path = self.config_dir / f"{filename}.json"
        
        config_data = {
            "version": "1.0",
            "description": f"AFPBN experiment configurations for {filename}",
            "total_experiments": len(configs),
            "experiments": [asdict(config) for config in configs]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Saved {len(configs)} configurations to {config_path}")
        return config_path
    
    def load_configs_from_file(self, filename: str) -> List[ExperimentSpec]:
        """Load experiment configurations from JSON file"""
        config_path = self.config_dir / f"{filename}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        configs = []
        for exp_dict in config_data["experiments"]:
            # Convert back to ExperimentSpec
            configs.append(ExperimentSpec(**exp_dict))
        
        print(f"Loaded {len(configs)} configurations from {config_path}")
        return configs
    
    def create_all_paper_configs(self):
        """Create and save all paper configuration files"""
        all_configs = PaperExperimentConfigs.get_all_paper_configs()
        
        created_files = []
        for table_name, configs in all_configs.items():
            config_path = self.save_configs_to_file(configs, table_name)
            created_files.append(config_path)
        
        # Also create a master config with all experiments
        all_experiments = []
        for configs in all_configs.values():
            all_experiments.extend(configs)
        
        master_path = self.save_configs_to_file(all_experiments, "all_paper_experiments")
        created_files.append(master_path)
        
        return created_files
    
    def generate_expected_results_table(self, configs: List[ExperimentSpec]) -> str:
        """Generate a table of expected results for comparison"""
        table_lines = []
        table_lines.append("="*80)
        table_lines.append("EXPECTED RESULTS SUMMARY")
        table_lines.append("="*80)
        table_lines.append(f"{'Experiment Name':<30} {'MAE (ns)':<10} {'Val RMSE':<10} {'Description'}")
        table_lines.append("-"*80)
        
        for config in configs:
            mae_str = f"{config.expected_mae:.4f}" if config.expected_mae else "N/A"
            rmse_str = f"{config.expected_val_rmse:.4f}" if config.expected_val_rmse else "N/A"
            
            table_lines.append(
                f"{config.name[:29]:<30} {mae_str:<10} {rmse_str:<10} {config.description[:30]}"
            )
        
        table_lines.append("="*80)
        return "\n".join(table_lines)

def generate_batch_script(configs: List[ExperimentSpec], 
                        script_name: str = "run_experiments.sh") -> Path:
    """Generate a bash script to run all experiments"""
    script_lines = [
        "#!/bin/bash",
        "# Auto-generated batch script for AFPBN experiments",
        f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total experiments: {len(configs)}",
        "",
        "set -e  # Exit on any error",
        "",
        "echo 'Starting AFPBN experiment batch...'",
        "echo 'Total experiments to run: " + str(len(configs)) + "'",
        "echo ''",
        ""
    ]
    
    for i, config in enumerate(configs, 1):
        script_lines.extend([
            f"echo 'Running experiment {i}/{len(configs)}: {config.name}'",
            f"echo 'Description: {config.description}'",
            ""
        ])
        
        # Build command
        cmd_parts = [
            "python main_afpbn.py",
            f"--data_type {config.data_type}",
            f"--case {config.case}",
            f"--experiment_type {config.experiment_type}",
            f"--n_basis {config.n_basis}",
            f"--epochs {config.epochs}",
            f"--lambda_recon {config.lambda_recon}"
        ]
        
        if not config.use_both_channels:
            cmd_parts.append("--single_channel")
        
        if config.basis_powers:
            powers_str = " ".join([str(p) for p in config.basis_powers])
            cmd_parts.append(f"--basis_powers {powers_str}")
        
        cmd_parts.append("--save_model")
        
        script_lines.append(" \\\n  ".join(cmd_parts))
        script_lines.extend(["", "echo 'Experiment completed.'", "echo ''", ""])
    
    script_lines.extend([
        "echo 'All experiments completed!'",
        "echo 'Results saved in ./results/ directory'",
        "echo 'Run analysis with: python analysis_utils.py --results_dir ./results'"
    ])
    
    script_path = Path(script_name)
    with open(script_path, 'w') as f:
        f.write("\n".join(script_lines))
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"Generated batch script: {script_path}")
    print(f"Run with: ./{script_path}")
    
    return script_path

def main():
    """Main function for configuration management"""
    parser = argparse.ArgumentParser(
        description="AFPBN Experiment Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce Table 4 results
  python experiment_configs.py --reproduce_table 4
  
  # Reproduce Table 6 ablation study  
  python experiment_configs.py --reproduce_table 6
  
  # Create all paper configurations
  python experiment_configs.py --create_all_configs
  
  # Generate batch script for Table 4
  python experiment_configs.py --reproduce_table 4 --generate_batch_script
        """
    )
    
    parser.add_argument('--reproduce_table', type=str, 
                       choices=['4', '5', '6', '7'],
                       help='Reproduce specific table from the paper')
    parser.add_argument('--create_all_configs', action='store_true',
                       help='Create all predefined configuration files')
    parser.add_argument('--generate_batch_script', action='store_true',
                       help='Generate bash script to run experiments')
    parser.add_argument('--config_dir', type=str, default='./configs',
                       help='Directory to save configuration files')
    parser.add_argument('--show_expected_results', action='store_true',
                       help='Show table of expected results')
    
    args = parser.parse_args()
    
    config_manager = ExperimentConfigManager(Path(args.config_dir))
    
    if args.create_all_configs:
        created_files = config_manager.create_all_paper_configs()
        print(f"\nCreated {len(created_files)} configuration files:")
        for file_path in created_files:
            print(f"  - {file_path}")
        return
    
    if args.reproduce_table:
        table_map = {
            '4': 'table_4',
            '5': 'table_5', 
            '6': 'table_6',
            '7': 'table_7'
        }
        
        table_key = table_map[args.reproduce_table]
        all_configs = PaperExperimentConfigs.get_all_paper_configs()
        configs = all_configs[table_key]
        
        print(f"\nTable {args.reproduce_table} Experiments:")
        print(f"Total experiments: {len(configs)}")
        
        # Save configuration file
        config_path = config_manager.save_configs_to_file(configs, table_key)
        
        # Show expected results
        if args.show_expected_results:
            expected_table = config_manager.generate_expected_results_table(configs)
            print(f"\n{expected_table}")
        
        # Generate batch script if requested
        if args.generate_batch_script:
            script_path = generate_batch_script(configs, f"run_table_{args.reproduce_table}.sh")
            print(f"\nTo run all experiments: ./{script_path}")
        
        # Print individual commands
        print(f"\nIndividual commands to reproduce Table {args.reproduce_table}:")
        print("-" * 60)
        for i, config in enumerate(configs, 1):
            cmd_parts = [
                "python main_afpbn.py",
                f"--data_type {config.data_type}",
                f"--case {config.case}",
                f"--experiment_type {config.experiment_type}"
            ]
            
            if config.n_basis != 6:
                cmd_parts.append(f"--n_basis {config.n_basis}")
            if not config.use_both_channels:
                cmd_parts.append("--single_channel")
            if config.lambda_recon != 0.1:
                cmd_parts.append(f"--lambda_recon {config.lambda_recon}")
            if config.basis_powers:
                powers_str = " ".join([str(p) for p in config.basis_powers])
                cmd_parts.append(f"--basis_powers {powers_str}")
            
            print(f"{i:2d}. {' '.join(cmd_parts)}")
            print(f"    # {config.description}")
            if config.expected_mae:
                print(f"    # Expected MAE: {config.expected_mae:.4f} ns")
            print()

if __name__ == "__main__":
    from datetime import datetime
    main()