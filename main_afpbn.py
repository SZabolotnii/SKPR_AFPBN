#!/usr/bin/env python3
"""
AFPBN: Adaptive Fractional-Power Basis Networks for Robust UWB Signal Processing

This script implements the AFPBN architecture with comprehensive ablation study support
as described in "Adaptive Hybrid Neural Networks Based on Decomposition in Space with 
Generating Element: Robust Parameter Estimation for UWB Radar Signals"

Key Features:
1. Adaptive alpha parameter selection based on excess kurtosis
2. Dual-task learning (regression + reconstruction)
3. Multi-channel fractional-power basis features
4. Comprehensive ablation study configurations
5. Flexible experimental setup

Usage Examples:
# Full AFPBN model with adaptive alpha
python main_afpbn.py --data_type variance --case 1 --experiment_type full

# Ablation: without adaptive alpha (fixed alpha=0.3)
python main_afpbn.py --data_type cir --case 1 --experiment_type no_adaptive_alpha

# Ablation: without reconstruction loss
python main_afpbn.py --data_type variance --case 2 --experiment_type no_reconstruction

# Ablation: without basis features (baseline CNN)
python main_afpbn.py --data_type cir --case 3 --experiment_type no_basis

# Different number of basis functions
python main_afpbn.py --data_type variance --case 1 --experiment_type full --n_basis 3

Author: Serhii Zabolotnii
Date: August 2025
"""

import argparse
import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import gdown
import numpy as np
import scipy.io as sio
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ===============================================================================
# CONSTANTS AND CONFIGURATION
# ===============================================================================

# Dataset configuration
DATASET_CONFIG = {
    'file_id': "1jo-PErF5nnqWJ8UUdzZv_OpWcDMesgxB",
    'data_dir': Path("./data"),
    'results_dir': Path("./results"),
    'models_dir': Path("./models"),
    'mat_file_name': "Dyn_CIR_VAR.mat"
}

# Model hyperparameters from the paper
MODEL_CONFIG = {
    'learning_rate': 0.001,
    'max_epochs': 100,
    'early_stop_patience': 10,
    'resampled_length': 500,
    'cnn_output_dim': 128,
    'max_basis_functions': 6,
    'lambda_recon': 0.1,
    'gamma_l2': 0.0001,
    'dropout_rate': 0.1,
    'basis_dropout_rate': 0.2,
    'basis_shift': 1.0,
    'alpha_update_freq': 5,
    'alpha_smoothing': 0.7,
    'gradient_clip_norm': 1.0
}

RANDOM_SEED = 42

class ExperimentType(Enum):
    """Enumeration of different experiment types for ablation studies"""
    FULL = "full"                           # Complete AFPBN model
    NO_ADAPTIVE_ALPHA = "no_adaptive_alpha" # Fixed alpha = 0.3
    NO_RECONSTRUCTION = "no_reconstruction"  # Lambda = 0
    NO_BASIS = "no_basis"                   # CNN only (baseline)
    SINGLE_CHANNEL = "single_channel"       # Only dynamic channel
    FIXED_BASIS = "fixed_basis"             # Fixed fractional powers [0.5, 0.333, 0.25]

@dataclass
class ExperimentConfig:
    """Configuration class for experiment parameters"""
    data_type: str
    case: int
    experiment_type: ExperimentType
    n_basis: int = MODEL_CONFIG['max_basis_functions']
    use_both_channels: bool = True
    adaptive_alpha: bool = True
    lambda_recon: float = MODEL_CONFIG['lambda_recon']
    basis_powers: Optional[List[float]] = None
    epochs: int = MODEL_CONFIG['max_epochs']
    save_model: bool = False
    
    def __post_init__(self):
        """Adjust configuration based on experiment type"""
        if self.experiment_type == ExperimentType.NO_ADAPTIVE_ALPHA:
            self.adaptive_alpha = False
            self.basis_powers = None  # Will use fixed alpha=0.3
        elif self.experiment_type == ExperimentType.NO_RECONSTRUCTION:
            self.lambda_recon = 0.0
        elif self.experiment_type == ExperimentType.NO_BASIS:
            self.n_basis = 0
            self.adaptive_alpha = False
            self.basis_powers = None
        elif self.experiment_type == ExperimentType.SINGLE_CHANNEL:
            self.use_both_channels = False
        elif self.experiment_type == ExperimentType.FIXED_BASIS:
            self.adaptive_alpha = False
            self.basis_powers = [0.5, 0.333, 0.25]

# ===============================================================================
# DATA LOADING AND PREPROCESSING
# ===============================================================================

class UWBDataLoader:
    """Class for handling UWB data loading and preprocessing"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.mat_file_path = data_dir / DATASET_CONFIG['mat_file_name']
        
    def download_dataset(self) -> Path:
        """Download and verify the UWB dataset from Google Drive"""
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        if not self.mat_file_path.exists() or self.mat_file_path.stat().st_size < 10_000_000:
            print(f"Downloading {self.mat_file_path} from Google Drive...")
            gdown.download(
                f"https://drive.google.com/uc?id={DATASET_CONFIG['file_id']}", 
                str(self.mat_file_path), 
                quiet=False
            )
        else:
            print(f"{self.mat_file_path} already exists.")
        
        # Verify the downloaded file
        try:
            sio.loadmat(self.mat_file_path)
            print("MAT file successfully verified.")
        except Exception as e:
            raise RuntimeError(f"Invalid MAT file. Please delete it and try again.") from e
        
        return self.mat_file_path
    
    @staticmethod
    def per_sample_normalize(sample: np.ndarray) -> np.ndarray:
        """Per-sample min-max normalization (equivalent to MATLAB's mat2gray)"""
        min_val, max_val = sample.min(), sample.max()
        return (sample - min_val) / (max_val - min_val) if max_val > min_val else sample
    
    def load_and_prepare_data(self, data_type: str, case: int) -> Tuple:
        """
        Load and prepare UWB data for training and testing
        
        Args:
            data_type: 'cir' or 'variance'
            case: 1, 2, or 3 (different train/test splits)
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test), batch_size, re_samp_time
        """
        print(f"\n{'='*60}")
        print(f"Loading data - Type: {data_type.upper()}, Case: {case}")
        print(f"{'='*60}\n")
        
        mat_data = sio.loadmat(self.mat_file_path)
        
        # Load resampling time and compute labels
        re_samp_time = mat_data['re_SampTime'].squeeze()
        label_dict = {}
        link_ids = ["01", "02", "04", "12", "14", "24"]
        
        for link_id in link_ids:
            real_tof = mat_data[f'Dyn_real_ToF{link_id}'].squeeze()
            trx_tof = np.min(real_tof)
            diff_tof = np.abs(real_tof - trx_tof)
            label_dict[link_id] = np.argmin(np.abs(diff_tof[:, None] - re_samp_time), axis=1)
        
        # Determine train/test splits based on case
        split_indices = {}
        for link_id in link_ids:
            time_stamps = mat_data[f'Dyn_re_tUWB{link_id}'].squeeze()
            if case == 1:
                train_idx = np.where(time_stamps < 73.8)[0]
                test_idx = np.where(time_stamps >= 73.8)[0]
            elif case == 2:
                test_idx = np.where((time_stamps > 47.6) & (time_stamps < 73.8))[0]
                train_idx = np.where((time_stamps <= 47.6) | (time_stamps >= 73.8))[0]
            else:  # case == 3
                test_idx = np.where(time_stamps < 47.6)[0]
                train_idx = np.where(time_stamps >= 47.6)[0]
            split_indices[link_id] = {'train': train_idx, 'test': test_idx}
        
        # Load and process signals
        all_X_train, all_y_train = [], []
        all_X_test, all_y_test = [], []
        
        prefix = "Dyn_var_CIR" if data_type == 'variance' else "Dyn_re_CIR"
        
        for link_id in link_ids:
            # Load raw signal
            dyn_signal_raw = mat_data[f"{prefix}{link_id}"]
            if np.iscomplexobj(dyn_signal_raw):
                processed_signal = np.abs(dyn_signal_raw)
            else:
                processed_signal = dyn_signal_raw
            
            # Handle NaN values
            if np.isnan(processed_signal).any():
                processed_signal = np.nan_to_num(
                    processed_signal, 
                    nan=np.nanmedian(processed_signal)
                )
            
            # Compute background signal
            static_samples = min(1700, len(processed_signal) // 4)
            bg_signal = np.mean(processed_signal[:static_samples], axis=0)
            bg_norm = self.per_sample_normalize(bg_signal)
            
            # Get labels and indices
            labels = label_dict[link_id]
            train_idx = split_indices[link_id]['train']
            test_idx = split_indices[link_id]['test']
            
            # Prepare training data
            X_train_link = np.zeros((len(train_idx), 2, MODEL_CONFIG['resampled_length']))
            for i, idx in enumerate(train_idx):
                X_train_link[i, 0, :] = self.per_sample_normalize(processed_signal[idx, :])
                X_train_link[i, 1, :] = bg_norm
            
            # Prepare test data
            X_test_link = np.zeros((len(test_idx), 2, MODEL_CONFIG['resampled_length']))
            for i, idx in enumerate(test_idx):
                X_test_link[i, 0, :] = self.per_sample_normalize(processed_signal[idx, :])
                X_test_link[i, 1, :] = bg_norm
            
            all_X_train.append(X_train_link)
            all_y_train.append(labels[train_idx])
            all_X_test.append(X_test_link)
            all_y_test.append(labels[test_idx])
        
        # Combine all links
        X_train_final = np.vstack(all_X_train)
        y_train_final = np.concatenate(all_y_train)
        X_test_final = np.vstack(all_X_test)
        y_test_final = np.concatenate(all_y_test)
        
        # Determine batch size (10% of training set)
        batch_size = int(0.10 * len(X_train_final))
        
        # Convert to tensors
        tensors = tuple(
            torch.from_numpy(d).float() 
            for d in [X_train_final, y_train_final, X_test_final, y_test_final]
        )
        
        return tensors, batch_size, re_samp_time

# ===============================================================================
# FRACTIONAL-POWER BASIS FUNCTIONS
# ===============================================================================

class FractionalBasisGenerator:
    """Class for generating and managing fractional-power basis functions"""
    
    def __init__(self, max_functions: int = MODEL_CONFIG['max_basis_functions']):
        self.max_functions = max_functions
        self.forbidden_alpha = 0.5  # Alpha value to avoid (leads to degenerate basis)
    
    def estimate_optimal_alpha(self, residuals: np.ndarray) -> float:
        """
        Estimate optimal alpha parameter based on excess kurtosis of residuals
        
        Args:
            residuals: Model residuals for alpha adaptation
            
        Returns:
            Optimal alpha value in [0, 1], avoiding forbidden value 0.5
        """
        try:
            # Compute statistical characteristics
            kurtosis = scipy.stats.kurtosis(residuals, fisher=True)  # Excess kurtosis
            skewness = scipy.stats.skew(residuals)
            std_residuals = np.std(residuals)
            
            # Heuristic rules based on tail heaviness
            if kurtosis > 3:  # Very heavy tails
                alpha = 0.0
            elif kurtosis > 1.5:  # Heavy tails
                alpha = 0.2
            elif kurtosis > 0:  # Moderately heavy tails
                alpha = 0.3
            elif kurtosis > -1:  # Close to normal
                alpha = 0.7
            else:  # Light tails
                alpha = 1.0
            
            # Correction for skewness
            if abs(skewness) > 2:
                alpha = max(0.0, alpha - 0.2)
            elif abs(skewness) > 1:
                alpha = max(0.0, alpha - 0.1)
            
            # Correction for variability
            if std_residuals > 10:  # High variability
                alpha = min(1.0, alpha + 0.1)
            
            # Avoid forbidden value
            if abs(alpha - self.forbidden_alpha) < 0.1:
                if alpha < self.forbidden_alpha:
                    alpha = self.forbidden_alpha - 0.15
                else:
                    alpha = self.forbidden_alpha + 0.15
            
            # Final clipping
            alpha = np.clip(alpha, 0.0, 1.0)
            
            return alpha
            
        except Exception as e:
            print(f"Error in estimate_optimal_alpha: {e}")
            return 0.3  # Safe default value
    
    def generate_adaptive_powers(self, alpha: float, max_order: int = 5) -> List[float]:
        """
        Generate power exponents for the adaptive fractional-power basis
        
        Args:
            alpha: Adaptation parameter in [0, 1]
            max_order: Maximum order for power generation
            
        Returns:
            List of power exponents (fixed length for stability)
        """
        powers = []
        
        # Special cases
        if alpha == 0.0:
            # Pure fractional mode
            powers = [0.5, 0.333, 0.25, 0.2, 0.167, 0.143][:self.max_functions]
        elif alpha == 1.0:
            # Pure integer mode
            powers = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0][:self.max_functions]
        else:
            # Adaptive generation using piecewise-linear interpolation
            for i in range(2, max_order + 2):
                if alpha < 0.3:  # Fractional mode
                    p_i = (1 - alpha) * (1.0 / i) + alpha * (1.0 / np.sqrt(i))
                elif alpha > 0.7:  # Integer mode
                    p_i = (1 - alpha) * np.sqrt(i) + alpha * i
                else:  # Transition mode
                    p_i = (1 - alpha) * (1.0 / i) + alpha * np.sqrt(i)
                
                # Avoid powers close to 1
                if abs(p_i - 1.0) < 0.15:
                    if p_i < 1.0:
                        p_i = 0.85
                    else:
                        p_i = 1.2
                
                # Add only unique powers
                if all(abs(p_i - p) > 0.05 for p in powers):
                    powers.append(p_i)
                
                # Stop when we have enough powers
                if len(powers) >= self.max_functions:
                    break
        
        # Fill to fixed length if needed
        while len(powers) < self.max_functions:
            if len(powers) >= 2:
                new_p = (powers[-1] + powers[-2]) / 2
                if all(abs(new_p - p) > 0.05 for p in powers):
                    powers.append(new_p)
                else:
                    powers.append(powers[-1] + 0.1)
            else:
                powers.append(0.1 * (len(powers) + 1))
        
        # Clip to maximum length and sort
        return sorted(powers[:self.max_functions])
    
    def calculate_basis_features(self, x_batch: torch.Tensor, powers: List[float], 
                               use_both_channels: bool = False, b: float = MODEL_CONFIG['basis_shift'],
                               normalize: bool = True) -> torch.Tensor:
        """
        Calculate fractional-power basis features with fixed dimensionality
        
        Args:
            x_batch: Input tensor (batch_size, channels, length)
            powers: List of power exponents
            use_both_channels: Whether to use both input channels
            b: Shift parameter for numerical stability
            normalize: Whether to normalize features
            
        Returns:
            Feature tensor (batch_size, max_functions * num_channels)
        """
        if not powers:
            num_channels = 2 if use_both_channels else 1
            return torch.zeros(
                x_batch.size(0), 
                self.max_functions * num_channels, 
                device=x_batch.device
            )
        
        # Process input dimensions
        if x_batch.dim() == 3:  # (batch, channels, length)
            if use_both_channels and x_batch.size(1) >= 2:
                channels_to_process = [x_batch[:, 0, :], x_batch[:, 1, :]]
            else:
                channels_to_process = [x_batch[:, 0, :]]
        else:  # (batch, length)
            channels_to_process = [x_batch]
        
        num_channels = len(channels_to_process)
        batch_size = x_batch.size(0)
        
        # Initialize output tensor with fixed size
        all_features = torch.zeros(
            batch_size, 
            self.max_functions * num_channels, 
            device=x_batch.device
        )
        
        # Limit the number of active powers
        active_powers = powers[:self.max_functions]
        powers_tensor = torch.tensor(
            active_powers, 
            dtype=torch.float32, 
            device=x_batch.device
        ).view(1, 1, -1)
        eps = 1e-8
        
        for ch_idx, channel_data in enumerate(channels_to_process):
            # Add shift parameter
            x_shifted = channel_data + b
            
            # Safe computation of fractional powers
            x_abs = torch.abs(x_shifted).unsqueeze(-1)
            x_sign = torch.sign(x_shifted).unsqueeze(-1)
            x_abs_safe = torch.clamp(x_abs, min=eps)
            
            # Compute (b + x)^p
            phi_x = x_sign * torch.pow(x_abs_safe, powers_tensor)
            
            # Normalize each basis function (optional)
            if normalize:
                # L2 normalization across signal length
                phi_x = F.normalize(phi_x, p=2, dim=1)
            
            # Average across signal length
            features_channel = torch.mean(phi_x, dim=1)  # (batch_size, num_active_powers)
            
            # Store in correct positions of output tensor
            start_idx = ch_idx * self.max_functions
            end_idx = start_idx + len(active_powers)
            all_features[:, start_idx:end_idx] = features_channel
        
        return all_features

# ===============================================================================
# NEURAL NETWORK ARCHITECTURE
# ===============================================================================

class ResBlock(nn.Module):
    """Residual block for the CNN encoder"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=4, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=4, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Handle size mismatch
        if out.size(-1) != identity.size(-1):
            identity = F.interpolate(identity, size=out.size(-1), 
                                   mode='linear', align_corners=False)
        
        out += identity
        return F.relu(out)

class AFPBNModel(nn.Module):
    """
    Adaptive Fractional-Power Basis Network (AFPBN) model
    
    This hybrid architecture combines:
    1. CNN encoder for automatic feature learning
    2. Fractional-power basis for robust statistical features  
    3. Dual-task decoder for regression and reconstruction
    """
    
    def __init__(self, config: ExperimentConfig):
        super(AFPBNModel, self).__init__()
        self.config = config
        
        # CNN Encoder (based on Li et al. baseline)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 8, 10, padding=4), 
            nn.BatchNorm1d(8), 
            nn.ReLU(),
            nn.MaxPool1d(10, stride=5),
            nn.Conv1d(8, 16, 4, padding=1), 
            nn.BatchNorm1d(16), 
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            ResBlock(16, 32), 
            ResBlock(32, 64), 
            ResBlock(64, 128),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Calculate feature dimensions
        cnn_dim = MODEL_CONFIG['cnn_output_dim']
        if config.experiment_type == ExperimentType.NO_BASIS:
            basis_dim = 0
        else:
            num_channels = 2 if config.use_both_channels else 1
            basis_dim = config.n_basis * num_channels
        
        combined_dim = cnn_dim + basis_dim
        
        # Dropout for basis features
        if basis_dim > 0:
            self.basis_dropout = nn.Dropout(MODEL_CONFIG['basis_dropout_rate'])
        else:
            self.basis_dropout = None
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(64, 1)
        )
        
        # Reconstruction head (uses only CNN features)
        if config.experiment_type != ExperimentType.NO_RECONSTRUCTION:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(cnn_dim, 256), 
                nn.ReLU(),
                nn.Linear(256, MODEL_CONFIG['resampled_length'])
            )
        else:
            self.reconstruction_head = None
    
    def forward(self, x: torch.Tensor, basis_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # CNN encoding
        cnn_features = torch.flatten(self.encoder(x), 1)
        
        # Combine features with regularization
        if basis_features.size(1) > 0:
            if self.basis_dropout is not None:
                basis_features = self.basis_dropout(basis_features)
            combined_features = torch.cat([cnn_features, basis_features], dim=1)
        else:
            combined_features = cnn_features
        
        # Main task: ToF regression
        tof_prediction = self.regression_head(combined_features)
        
        # Auxiliary task: signal reconstruction
        if self.reconstruction_head is not None:
            signal_reconstruction = self.reconstruction_head(cnn_features)
        else:
            signal_reconstruction = None
        
        return tof_prediction, signal_reconstruction

# ===============================================================================
# TRAINING AND EVALUATION
# ===============================================================================

class AFPBNTrainer:
    """Class for training and evaluating AFPBN models"""
    
    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device
        self.basis_generator = FractionalBasisGenerator(config.n_basis)
        
        # Create directories
        for directory in [DATASET_CONFIG['results_dir'], DATASET_CONFIG['models_dir']]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def create_experiment_name(self) -> str:
        """Create unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.data_type}_case{self.config.case}_{self.config.experiment_type.value}_{timestamp}"
    
    def create_data_loaders(self, X_train: torch.Tensor, y_train: torch.Tensor, 
                          X_test: torch.Tensor, y_test: torch.Tensor, 
                          batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders"""
        # Split training data into train/validation
        val_size = int(len(X_train) * 0.15)
        indices = torch.randperm(len(X_train))
        
        X_train_split = X_train[indices[:-val_size]]
        y_train_split = y_train[indices[:-val_size]]
        X_val = X_train[indices[-val_size:]]
        y_val = y_train[indices[-val_size:]]
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(X_train_split, y_train_split), 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), 
            batch_size=batch_size
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), 
            batch_size=batch_size
        )
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate(self, data_tensors: Tuple, batch_size: int, re_samp_time: np.ndarray) -> Dict:
        """
        Main training and evaluation function
        
        Args:
            data_tensors: (X_train, y_train, X_test, y_test)
            batch_size: Batch size for training
            re_samp_time: Resampling time array for converting indices to nanoseconds
            
        Returns:
            Dictionary with results and metrics
        """
        X_train, y_train, X_test, y_test = data_tensors
        experiment_name = self.create_experiment_name()
        
        print(f"\n{'='*80}")
        print(f"TRAINING AFPBN MODEL")
        print(f"Experiment: {experiment_name}")
        print(f"Configuration: {self.config}")
        print(f"{'='*80}\n")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            X_train, y_train, X_test, y_test, batch_size
        )
        
        # Initialize model
        model = AFPBNModel(self.config).to(self.device)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Setup optimization
        criterion_mse = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5
        )
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Initialize adaptive alpha if needed
        if self.config.adaptive_alpha:
            current_alpha = 0.0  # Start with pure fractional mode
            current_powers = self.basis_generator.generate_adaptive_powers(current_alpha)
        else:
            if self.config.basis_powers:
                current_powers = self.config.basis_powers.copy()
            elif self.config.experiment_type == ExperimentType.NO_ADAPTIVE_ALPHA:
                # Fixed alpha = 0.3
                current_powers = self.basis_generator.generate_adaptive_powers(0.3)
            else:
                current_powers = [0.5, 0.333, 0.25]  # Default fractional powers
            current_alpha = None
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'val_rmse': [],
            'learning_rate': [], 'alpha_values': [], 'powers': []
        }
        
        print(f"Initial powers: {[f'{p:.3f}' for p in current_powers] if current_powers else 'None'}")
        if current_alpha is not None:
            print(f"Initial alpha: {current_alpha}")
        
        # Training loop
        for epoch in range(self.config.epochs):
            model.train()
            epoch_residuals = []
            train_loss_sum = 0
            
            # Training step
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device).view(-1, 1)
                
                # Compute basis features
                if self.config.experiment_type != ExperimentType.NO_BASIS:
                    basis_features = self.basis_generator.calculate_basis_features(
                        inputs, current_powers, self.config.use_both_channels, 
                        MODEL_CONFIG['basis_shift'], normalize=True
                    )
                else:
                    basis_features = torch.zeros(
                        inputs.size(0), 0, device=self.device
                    )
                
                # Forward pass
                tof_pred, signal_recon = model(inputs, basis_features)
                
                # Compute losses
                loss_regression = criterion_mse(tof_pred, targets)
                
                if signal_recon is not None:
                    loss_reconstruction = criterion_mse(signal_recon, inputs[:, 0, :])
                    total_loss = loss_regression + self.config.lambda_recon * loss_reconstruction
                else:
                    total_loss = loss_regression
                
                # Collect residuals for adaptive alpha
                if self.config.adaptive_alpha:
                    residuals = (tof_pred - targets).detach().cpu().numpy().flatten()
                    epoch_residuals.extend(residuals)
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    MODEL_CONFIG['gradient_clip_norm']
                )
                
                optimizer.step()
                
                train_loss_sum += total_loss.item() * inputs.size(0)
            
            # Adaptive alpha update
            if (self.config.adaptive_alpha and epoch > 0 and 
                epoch % MODEL_CONFIG['alpha_update_freq'] == 0 and epoch_residuals):
                
                new_alpha = self.basis_generator.estimate_optimal_alpha(
                    np.array(epoch_residuals)
                )
                
                # Exponential smoothing for stability
                if epoch > 5:
                    current_alpha = (MODEL_CONFIG['alpha_smoothing'] * current_alpha + 
                                   (1 - MODEL_CONFIG['alpha_smoothing']) * new_alpha)
                else:
                    current_alpha = new_alpha
                
                # Update powers
                current_powers = self.basis_generator.generate_adaptive_powers(current_alpha)
                print(f"\nEpoch {epoch}: α={current_alpha:.3f}, "
                      f"powers={[f'{p:.3f}' for p in current_powers]}")
            
            # Validation
            model.eval()
            val_total_loss = 0
            val_reg_loss = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device).view(-1, 1)
                    
                    # Compute basis features
                    if self.config.experiment_type != ExperimentType.NO_BASIS:
                        basis_features = self.basis_generator.calculate_basis_features(
                            inputs, current_powers, self.config.use_both_channels,
                            MODEL_CONFIG['basis_shift'], normalize=True
                        )
                    else:
                        basis_features = torch.zeros(
                            inputs.size(0), 0, device=self.device
                        )
                    
                    # Forward pass
                    tof_pred, signal_recon = model(inputs, basis_features)
                    
                    # Compute losses
                    loss_regression = criterion_mse(tof_pred, targets)
                    
                    if signal_recon is not None:
                        loss_reconstruction = criterion_mse(signal_recon, inputs[:, 0, :])
                        total_loss = loss_regression + self.config.lambda_recon * loss_reconstruction
                    else:
                        total_loss = loss_regression
                    
                    val_total_loss += total_loss.item() * inputs.size(0)
                    val_reg_loss += loss_regression.item() * inputs.size(0)
            
            # Compute metrics
            train_loss_avg = train_loss_sum / len(train_loader.dataset)
            val_total_loss /= len(val_loader.dataset)
            val_reg_rmse = np.sqrt(val_reg_loss / len(val_loader.dataset))
            
            # Update history
            history['train_loss'].append(train_loss_avg)
            history['val_loss'].append(val_total_loss)
            history['val_rmse'].append(val_reg_rmse)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            if current_alpha is not None:
                history['alpha_values'].append(current_alpha)
            if current_powers:
                history['powers'].append(current_powers.copy())
            
            # Learning rate scheduling
            scheduler.step(val_total_loss)
            
            print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss_avg:.4f}, "
                  f"Val Loss: {val_total_loss:.4f}, Val RMSE: {val_reg_rmse:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_alpha = current_alpha
                best_powers = current_powers.copy() if current_powers else []
            else:
                patience_counter += 1
            
            if patience_counter >= MODEL_CONFIG['early_stop_patience']:
                print(f"Early stopping at epoch {epoch+1}.")
                break
        
        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"Restored best model with Val Loss: {best_val_loss:.4f}")
            if self.config.adaptive_alpha:
                current_alpha = best_alpha
                current_powers = best_powers
        
        # Final testing
        model.eval()
        with torch.no_grad():
            pred_indices_list = []
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                
                # Compute basis features
                if self.config.experiment_type != ExperimentType.NO_BASIS:
                    basis_features = self.basis_generator.calculate_basis_features(
                        inputs, current_powers, self.config.use_both_channels,
                        MODEL_CONFIG['basis_shift'], normalize=True
                    )
                else:
                    basis_features = torch.zeros(
                        inputs.size(0), 0, device=self.device
                    )
                
                tof_pred, _ = model(inputs, basis_features)
                pred_indices_list.append(tof_pred.cpu())
            
            pred_indices = torch.cat(pred_indices_list).numpy().squeeze()
        
        # Compute final metrics
        true_indices = y_test.numpy()
        pred_tof = re_samp_time[np.round(pred_indices).astype(int).clip(0, len(re_samp_time)-1)]
        true_tof = re_samp_time[true_indices.astype(int)]
        
        final_mae_ns = np.mean(np.abs(pred_tof - true_tof))
        final_rmse_ns = np.sqrt(np.mean((pred_tof - true_tof)**2))
        
        # Print results
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"Experiment: {experiment_name}")
        if self.config.adaptive_alpha:
            print(f"Final α: {current_alpha:.3f}")
        print(f"Final powers: {[f'{p:.3f}' for p in current_powers] if current_powers else 'None'}")
        print(f"{'='*80}")
        print(f"Mean Absolute Error (MAE): {final_mae_ns:.4f} ns")
        print(f"Root Mean Square Error (RMSE): {final_rmse_ns:.4f} ns")
        print(f"Validation RMSE: {val_reg_rmse:.4f}")
        
        # Create results dictionary
        results = {
            'experiment_name': experiment_name,
            'config': {
                'data_type': self.config.data_type,
                'case': self.config.case,
                'experiment_type': self.config.experiment_type.value,
                'n_basis': self.config.n_basis,
                'use_both_channels': self.config.use_both_channels,
                'adaptive_alpha': self.config.adaptive_alpha,
                'lambda_recon': self.config.lambda_recon,
                'basis_powers': self.config.basis_powers
            },
            'final_metrics': {
                'mae_ns': float(final_mae_ns),
                'rmse_ns': float(final_rmse_ns),
                'val_rmse': float(val_reg_rmse),
                'best_val_loss': float(best_val_loss)
            },
            'final_alpha': float(current_alpha) if current_alpha is not None else None,
            'final_powers': [float(p) for p in current_powers] if current_powers else None,
            'history': history
        }
        
        # Save results
        results_path = DATASET_CONFIG['results_dir'] / f"{experiment_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        # Save model if requested
        if self.config.save_model:
            model_path = DATASET_CONFIG['models_dir'] / f"{experiment_name}_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.config,
                'final_alpha': current_alpha,
                'final_powers': current_powers,
                'final_metrics': results['final_metrics']
            }, model_path)
            print(f"Model saved to {model_path}")
        
        return results

# ===============================================================================
# MAIN EXECUTION AND ABLATION STUDIES
# ===============================================================================

def setup_device_and_seed(seed: int = RANDOM_SEED) -> torch.device:
    """Setup random seed and determine compute device"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    return device

def run_experiment(config: ExperimentConfig) -> Dict:
    """Run a single experiment with the given configuration"""
    device = setup_device_and_seed()
    
    # Load data
    data_loader = UWBDataLoader(DATASET_CONFIG['data_dir'])
    data_path = data_loader.download_dataset()
    data_tensors, batch_size, re_samp_time = data_loader.load_and_prepare_data(
        config.data_type, config.case
    )
    
    # Train and evaluate
    trainer = AFPBNTrainer(config, device)
    results = trainer.train_and_evaluate(data_tensors, batch_size, re_samp_time)
    
    return results

def run_ablation_study(data_type: str, case: int) -> Dict[str, Dict]:
    """
    Run comprehensive ablation study for given data type and case
    
    Args:
        data_type: 'cir' or 'variance'
        case: 1, 2, or 3
        
    Returns:
        Dictionary containing results for all ablation experiments
    """
    print(f"\n{'='*100}")
    print(f"RUNNING COMPREHENSIVE ABLATION STUDY")
    print(f"Data Type: {data_type.upper()}, Case: {case}")
    print(f"{'='*100}\n")
    
    ablation_results = {}
    
    # Define all ablation experiments
    experiments = [
        (ExperimentType.FULL, "Full AFPBN Model"),
        (ExperimentType.NO_ADAPTIVE_ALPHA, "Fixed Alpha (α=0.3)"),
        (ExperimentType.NO_RECONSTRUCTION, "No Reconstruction Loss"),
        (ExperimentType.NO_BASIS, "No Basis Features (Baseline CNN)"),
        (ExperimentType.SINGLE_CHANNEL, "Single Channel Only"),
        (ExperimentType.FIXED_BASIS, "Fixed Fractional Powers [0.5, 0.333, 0.25]")
    ]
    
    for experiment_type, description in experiments:
        print(f"\n{'-'*60}")
        print(f"Running: {description}")
        print(f"{'-'*60}")
        
        try:
            config = ExperimentConfig(
                data_type=data_type,
                case=case,
                experiment_type=experiment_type,
                epochs=50  # Reduced epochs for ablation study
            )
            
            results = run_experiment(config)
            ablation_results[experiment_type.value] = results
            
            print(f"✅ Completed: {description}")
            print(f"MAE: {results['final_metrics']['mae_ns']:.4f} ns, "
                  f"Val RMSE: {results['final_metrics']['val_rmse']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed: {description} - {str(e)}")
            ablation_results[experiment_type.value] = {"error": str(e)}
    
    # Save comprehensive ablation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_path = DATASET_CONFIG['results_dir'] / f"ablation_study_{data_type}_case{case}_{timestamp}.json"
    
    with open(ablation_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"ABLATION STUDY COMPLETED")
    print(f"Results saved to: {ablation_path}")
    print(f"{'='*100}")
    
    return ablation_results

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="AFPBN: Adaptive Fractional-Power Basis Networks for UWB Signal Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full AFPBN model
  python main_afpbn.py --data_type variance --case 1 --experiment_type full
  
  # Ablation studies
  python main_afpbn.py --data_type cir --case 1 --experiment_type no_adaptive_alpha
  python main_afpbn.py --data_type variance --case 2 --experiment_type no_reconstruction
  python main_afpbn.py --data_type cir --case 3 --experiment_type no_basis
  
  # Run complete ablation study
  python main_afpbn.py --data_type variance --case 1 --run_ablation_study
  
  # Different number of basis functions
  python main_afpbn.py --data_type cir --case 1 --experiment_type full --n_basis 3
        """
    )
    
    # Required arguments
    parser.add_argument('--data_type', type=str, required=True, 
                       choices=['cir', 'variance'],
                       help='Type of UWB data to use')
    parser.add_argument('--case', type=int, required=True, 
                       choices=[1, 2, 3],
                       help='Data split case (1: standard, 2: interpolation, 3: extrapolation)')
    
    # Experiment configuration
    parser.add_argument('--experiment_type', type=str, 
                       choices=[e.value for e in ExperimentType],
                       default=ExperimentType.FULL.value,
                       help='Type of experiment/ablation to run')
    
    # Optional parameters
    parser.add_argument('--n_basis', type=int, 
                       default=MODEL_CONFIG['max_basis_functions'],
                       help='Number of basis functions to use')
    parser.add_argument('--epochs', type=int, 
                       default=MODEL_CONFIG['max_epochs'],
                       help='Number of training epochs')
    parser.add_argument('--lambda_recon', type=float, 
                       default=MODEL_CONFIG['lambda_recon'],
                       help='Reconstruction loss weight')
    parser.add_argument('--basis_powers', type=float, nargs='*',
                       help='Fixed basis powers to use (overrides adaptive selection)')
    parser.add_argument('--single_channel', action='store_true',
                       help='Use only dynamic channel for basis features')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model to disk')
    
    # Special modes
    parser.add_argument('--run_ablation_study', action='store_true',
                       help='Run comprehensive ablation study')
    
    args = parser.parse_args()
    
    # Run ablation study if requested
    if args.run_ablation_study:
        run_ablation_study(args.data_type, args.case)
        return
    
    # Create experiment configuration
    try:
        experiment_type = ExperimentType(args.experiment_type)
    except ValueError:
        print(f"Invalid experiment type: {args.experiment_type}")
        print(f"Valid options: {[e.value for e in ExperimentType]}")
        return
    
    config = ExperimentConfig(
        data_type=args.data_type,
        case=args.case,
        experiment_type=experiment_type,
        n_basis=args.n_basis,
        use_both_channels=not args.single_channel,
        lambda_recon=args.lambda_recon,
        basis_powers=args.basis_powers,
        epochs=args.epochs,
        save_model=args.save_model
    )
    
    # Run single experiment
    results = run_experiment(config)
    
    # Print summary
    metrics = results['final_metrics']
    print(f"\n🎯 EXPERIMENT SUMMARY:")
    print(f"Experiment Type: {experiment_type.value}")
    print(f"MAE: {metrics['mae_ns']:.4f} ns")
    print(f"RMSE: {metrics['rmse_ns']:.4f} ns") 
    print(f"Val RMSE: {metrics['val_rmse']:.4f}")
    
    # Performance assessment
    target_mae = 0.95
    if metrics['mae_ns'] <= target_mae:
        status = "🎉 EXCELLENT! (State-of-the-art level)"
    elif metrics['mae_ns'] <= target_mae * 1.5:
        status = "✅ GOOD"
    else:
        status = "🔶 NEEDS FURTHER OPTIMIZATION"
    
    print(f"Status: {status}")

if __name__ == "__main__":
    main()