# AFPBN: Adaptive Fractional-Power Basis Networks

This repository contains the refactored implementation of **Adaptive Fractional-Power Basis Networks (AFPBN)** for robust UWB signal processing, as described in the paper:

> **"Adaptive Hybrid Neural Networks Based on Decomposition in Space with Generating Element: Robust Parameter Estimation for UWB Radar Signals"**

## 🎯 Key Features

- **Hybrid Architecture**: Combines CNN encoder with adaptive fractional-power basis features
- **Adaptive Alpha Selection**: Automatically adjusts basis parameters based on excess kurtosis
- **Dual-Task Learning**: Simultaneous regression and reconstruction for better generalization
- **Comprehensive Ablation Studies**: Built-in support for systematic component analysis
- **Statistical Robustness**: Designed for heavy-tailed, non-Gaussian UWB signal distributions

## 📁 Project Structure

```
├── main_afpbn.py      # Main AFPBN implementation
├── analysis_utils.py        # Results analysis and visualization utilities  
├── README.md               # This file
├── data/                   # UWB dataset (auto-downloaded)
├── results/                # Experiment results and logs
├── models/                 # Saved model checkpoints
└── figures/                # Generated plots and visualizations
```

## 🚀 Quick Start

### Installation

```bash
pip install torch numpy scipy matplotlib seaborn pandas tqdm gdown
```

### Basic Usage

```bash
# Run full AFPBN model with adaptive alpha
python main_afpbn.py --data_type variance --case 1 --experiment_type full

# Run baseline CNN (no basis features)
python main_afpbn.py --data_type cir --case 1 --experiment_type no_basis
```

## 🧪 Experiment Types

The implementation supports all ablation studies described in the paper:

| Experiment Type | Description | Paper Reference |
|----------------|-------------|-----------------|
| `full` | Complete AFPBN model | Section 5.2 |
| `no_adaptive_alpha` | Fixed α=0.3 | Table 6 |
| `no_reconstruction` | λ=0 (no reconstruction loss) | Table 6 |
| `no_basis` | CNN only (baseline) | Table 6 |
| `single_channel` | Dynamic channel only | Table 6 |
| `fixed_basis` | Fixed powers [0.5, 0.333, 0.25] | Section 5.3 |

## 📊 Data Configuration

### Data Types
- **`cir`**: Channel Impulse Response (accumulated and aligned)
- **`variance`**: Per-sample CIR variance (computed in sliding window)

### Cases (Train/Test Splits)
- **Case 1**: Standard split (66.7%/33.3%) - Table 4
- **Case 2**: Interpolation (test data from middle of trajectory) - Table 5  
- **Case 3**: Extrapolation (test data from beginning of trajectory) - Table 5

## 🔬 Running Experiments

### Single Experiments

```bash
# Full AFPBN with adaptive alpha (reproduces Table 4 results)
python main_afpbn.py --data_type cir --case 1 --experiment_type full

# Ablation: without adaptive alpha (Table 6)
python main_afpbn.py --data_type cir --case 1 --experiment_type no_adaptive_alpha

# Ablation: without reconstruction loss (Table 6)
python main_afpbn.py --data_type variance --case 1 --experiment_type no_reconstruction

# Ablation: different number of basis functions (Table 7)
python main_afpbn.py --data_type cir --case 1 --experiment_type full --n_basis 3
```

### Comprehensive Ablation Study

```bash
# Run all ablation experiments for given data type and case
python main_afpbn.py --data_type variance --case 1 --run_ablation_study
```

This will automatically run all experiment types and generate a comprehensive comparison.

### Advanced Options

```bash
# Custom configuration
python main_afpbn.py \
  --data_type cir \
  --case 2 \
  --experiment_type full \
  --n_basis 6 \
  --epochs 100 \
  --lambda_recon 0.1 \
  --save_model

# Fixed basis powers (expert knowledge)
python main_afpbn.py \
  --data_type variance \
  --case 1 \
  --experiment_type fixed_basis \
  --basis_powers 0.5 0.333 0.25
```

## 📈 Results Analysis

### Automatic Analysis

Results are automatically saved in JSON format with comprehensive metrics:

```json
{
  "experiment_name": "variance_case1_full_20250804_143022",
  "final_metrics": {
    "mae_ns": 0.8834,
    "rmse_ns": 1.1923,
    "val_rmse": 8.8972
  },
  "final_alpha": 0.102,
  "final_powers": [0.225, 0.275, 0.325, 0.358]
}
```

### Visualization and Reporting

```bash
# Analyze all results in directory
python analysis_utils.py --results_dir ./results --analysis_type comprehensive

# Compare ablation study results
python analysis_utils.py --results_dir ./results --analysis_type ablation_comparison

# Analyze single experiment
python analysis_utils.py --results_file results/experiment_results.json --analysis_type single_experiment
```

### Generated Outputs

The analysis utilities automatically generate:
- **Training history plots** (loss curves, alpha evolution)
- **Ablation comparison charts** (performance metrics)
- **Statistical analysis reports** (comprehensive summaries)
- **Performance improvement analysis** (relative to baseline)

## 🔧 Implementation Details

### Adaptive Alpha Mechanism

The alpha parameter is automatically updated every 5 epochs based on excess kurtosis:

```python
# Kurtosis-based alpha selection (Equation 11 in paper)
if kurtosis > 3:     alpha = 0.0  # Very heavy tails
elif kurtosis > 1.5: alpha = 0.2  # Heavy tails  
elif kurtosis > 0:   alpha = 0.3  # Moderate tails
elif kurtosis > -1:  alpha = 0.7  # Close to normal
else:                alpha = 1.0  # Light tails
```

### Fractional-Power Basis

Powers are generated using piecewise-linear interpolation (Equation 10):

```python
# Adaptive power generation
if alpha < 0.3:     # Fractional mode
    p_i = (1-α) * (1/i) + α * (1/√i)
elif alpha > 0.7:   # Integer mode  
    p_i = (1-α) * √i + α * i
else:               # Transition mode
    p_i = (1-α) * (1/i) + α * √i
```

### Architecture Configuration

```python
# Model hyperparameters (from paper)
MODEL_CONFIG = {
    'learning_rate': 0.001,
    'max_epochs': 100,
    'early_stop_patience': 10,
    'cnn_output_dim': 128,
    'max_basis_functions': 6,
    'lambda_recon': 0.1,          # Reconstruction loss weight
    'alpha_update_freq': 5,        # Update alpha every 5 epochs
    'alpha_smoothing': 0.7         # Exponential smoothing factor
}
```

## 📋 Expected Results

Based on the paper (Table 4), you should expect:

| Method | Data Type | MAE (ns) | Val RMSE | Improvement |
|--------|-----------|----------|----------|-------------|
| Baseline CNN | CIR | 1.0121 | 12.0088 | - |
| **AFPBN (Full)** | **CIR** | **0.8834** | **8.8972** | **+12.7% MAE, +25.9% RMSE** |
| Baseline CNN | Variance | 1.1350 | 8.7457 | - |
| **AFPBN (Full)** | **Variance** | **1.0834** | **7.2156** | **+4.5% MAE, +17.5% RMSE** |

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Download Issues**
   ```bash
   # Manual download if automatic fails
   # Check internet connection and Google Drive access
   ```

2. **CUDA Memory Issues**
   ```bash
   # Reduce batch size or use CPU
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   ```

3. **Numerical Instability**
   ```bash
   # Fractional powers can be sensitive - adjust basis_shift parameter
   python main_afpbn.py --experiment_type full --basis_shift 1.0
   ```

### Performance Expectations

- **Training Time**: ~5-15 minutes per experiment (depending on hardware)
- **Memory Usage**: ~2-4 GB GPU memory
- **Storage**: ~50 MB per saved model + results

## 🤝 Contributing

This implementation follows the exact specifications from the paper. For modifications:

1. Maintain compatibility with the paper's experimental protocol
2. Preserve all ablation study configurations  
3. Update both code and documentation
4. Test all experiment types before submitting

## 📚 Citation

If you use this implementation, please cite:

```bibtex
@article{zabolotnii2025afpbn,
  title={Adaptive Hybrid Neural Networks Based on Decomposition in Space with Generating Element: Robust Parameter Estimation for UWB Radar Signals},
  author={Zabolotnii, Serhii},
  journal={Submitted for publication},
  year={2025}
}
```

## 📞 Support

For questions about the implementation:
1. Check the paper for theoretical details
2. Review the code comments for implementation specifics
3. Use the analysis utilities to understand results
4. Compare with expected outcomes in the paper

---

**Note**: This implementation is designed to exactly reproduce the results reported in the paper. All hyperparameters, architectural choices, and experimental protocols match the published work.