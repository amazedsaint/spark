"""
Comprehensive SPD Router Evaluation

This module implements rigorous evaluation of the SPD Router's ability to separate
structured signals from pseudo-random noise in realistic scenarios.

Tests include:
1. Financial time series with trends + market noise
2. Audio signals with speech + background noise  
3. Scientific data with systematic patterns + measurement error
4. Synthetic signals with known ground truth separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
from scipy import signal
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from tqdm import tqdm
import json

from src.spd_router import SPDRouter, AdaptiveSPDRouter


class FinancialTimeSeriesDataset(Dataset):
    """Real-world-like financial time series with known structure + noise components"""
    
    def __init__(self, num_series: int = 1000, seq_len: int = 128, d_model: int = 64):
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.data = []
        self.ground_truth = []
        
        for _ in tqdm(range(num_series), desc="Generating financial time series"):
            # Generate structured components
            t = np.linspace(0, 10, seq_len)
            
            # Trend component
            trend_coeff = np.random.uniform(-0.5, 0.5)
            trend = trend_coeff * t
            
            # Seasonal components (multiple frequencies)
            seasonal = np.zeros(seq_len)
            for freq in [1/12, 1/4, 1/52]:  # Annual, quarterly, weekly patterns
                amplitude = np.random.uniform(0.1, 0.8)
                phase = np.random.uniform(0, 2*np.pi)
                seasonal += amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Cyclical component (business cycle)
            cycle_freq = np.random.uniform(1/120, 1/60)  # 5-10 year cycles
            cycle_amplitude = np.random.uniform(0.5, 1.5)
            cycle = cycle_amplitude * np.sin(2 * np.pi * cycle_freq * t)
            
            # Mean reversion component
            mean_rev_strength = np.random.uniform(0.05, 0.2)
            mean_rev = np.zeros(seq_len)
            for i in range(1, seq_len):
                mean_rev[i] = mean_rev[i-1] * (1 - mean_rev_strength) + np.random.normal(0, 0.1)
            
            # Combine structured components
            structured_base = trend + seasonal + cycle + mean_rev
            
            # Replicate across features with slight variations
            structured = np.zeros((seq_len, d_model))
            for j in range(d_model):
                variation = np.random.normal(0, 0.1, seq_len)
                correlation = np.random.uniform(0.7, 0.95)  # Features are correlated
                structured[:, j] = correlation * structured_base + (1-correlation) * variation
            
            # Generate noise components
            # Market noise (correlated across assets)
            market_noise = np.random.normal(0, 0.3, seq_len)
            
            # Idiosyncratic noise (uncorrelated)
            idio_noise = np.random.normal(0, 0.2, (seq_len, d_model))
            
            # Jump processes (sudden shocks)
            jump_prob = 0.05
            jump_times = np.random.random(seq_len) < jump_prob
            jump_magnitudes = np.random.normal(0, 1.0, seq_len)
            jumps = jump_times * jump_magnitudes
            
            # Volatility clustering (GARCH-like)
            vol = np.zeros(seq_len)
            vol[0] = 0.2
            for i in range(1, seq_len):
                vol[i] = 0.1 + 0.05 * (idio_noise[i-1, 0]**2) + 0.9 * vol[i-1]
            
            # Combined noise
            noise = np.zeros((seq_len, d_model))
            for j in range(d_model):
                noise_correlation = np.random.uniform(0.3, 0.7)
                noise[:, j] = (noise_correlation * market_noise + 
                              (1-noise_correlation) * idio_noise[:, j] + 
                              jumps) * vol
            
            # Final series
            final_series = structured + noise
            
            self.data.append(final_series)
            self.ground_truth.append({
                'structured': structured,
                'noise': noise,
                'trend': trend,
                'seasonal': seasonal,
                'cycle': cycle
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'structured_gt': torch.tensor(self.ground_truth[idx]['structured'], dtype=torch.float32),
            'noise_gt': torch.tensor(self.ground_truth[idx]['noise'], dtype=torch.float32),
            'trend': torch.tensor(self.ground_truth[idx]['trend'], dtype=torch.float32),
            'seasonal': torch.tensor(self.ground_truth[idx]['seasonal'], dtype=torch.float32)
        }


class AudioSignalDataset(Dataset):
    """Synthetic audio-like signals with speech patterns + background noise"""
    
    def __init__(self, num_signals: int = 500, seq_len: int = 256, d_model: int = 32):
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.data = []
        self.ground_truth = []
        
        for _ in tqdm(range(num_signals), desc="Generating audio-like signals"):
            # Generate speech-like structured signal
            t = np.linspace(0, 2, seq_len)  # 2 seconds
            
            # Fundamental frequency and harmonics (like vowel sounds)
            f0 = np.random.uniform(80, 300)  # Fundamental frequency
            speech_signal = np.zeros(seq_len)
            
            # Add harmonics with decreasing amplitude
            for harmonic in range(1, 6):
                amplitude = 1.0 / harmonic
                freq = f0 * harmonic
                phase = np.random.uniform(0, 2*np.pi)
                speech_signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Modulate with envelope (like speech amplitude)
            envelope_freq = np.random.uniform(2, 8)  # Syllable rate
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * envelope_freq * t)
            speech_signal *= envelope
            
            # Formant structure (frequency peaks characteristic of vowels)
            formant_freqs = [np.random.uniform(500, 1000), np.random.uniform(1000, 2000)]
            for formant_freq in formant_freqs:
                formant_bw = 50  # Bandwidth
                # Create formant as band-pass filtered noise
                formant_noise = np.random.normal(0, 0.3, seq_len)
                sos = signal.butter(4, [formant_freq-formant_bw, formant_freq+formant_bw], 
                                  btype='band', fs=seq_len/2, output='sos')
                formant_signal = signal.sosfilt(sos, formant_noise)
                speech_signal += 0.2 * formant_signal
            
            # Create multi-channel structured signal (like stereo or multiple mics)
            structured = np.zeros((seq_len, d_model))
            for j in range(d_model):
                # Simulate different microphone positions with delays and attenuation
                delay = np.random.randint(0, 5)
                attenuation = np.random.uniform(0.7, 1.0)
                
                if delay > 0:
                    delayed_signal = np.concatenate([np.zeros(delay), speech_signal[:-delay]])
                else:
                    delayed_signal = speech_signal
                
                structured[:, j] = attenuation * delayed_signal
            
            # Generate noise components
            # Environmental noise (correlated across channels)
            env_noise_types = ['pink', 'white', 'brown']
            noise_type = np.random.choice(env_noise_types)
            
            if noise_type == 'pink':
                # Pink noise (1/f spectrum)
                white_noise = np.random.normal(0, 1, seq_len)
                freqs = np.fft.fftfreq(seq_len)
                fft_white = np.fft.fft(white_noise)
                # Apply 1/f filter
                fft_pink = fft_white * (1 / np.sqrt(np.abs(freqs) + 1e-8))
                env_noise_base = np.real(np.fft.ifft(fft_pink))
            elif noise_type == 'white':
                env_noise_base = np.random.normal(0, 1, seq_len)
            else:  # brown
                env_noise_base = np.cumsum(np.random.normal(0, 1, seq_len))
            
            # Normalize
            env_noise_base = env_noise_base / np.std(env_noise_base)
            
            # Electronic noise (uncorrelated across channels)
            electronic_noise = np.random.normal(0, 0.1, (seq_len, d_model))
            
            # Impulse noise (sudden spikes)
            impulse_prob = 0.02
            impulse_locations = np.random.random((seq_len, d_model)) < impulse_prob
            impulse_magnitudes = np.random.normal(0, 2.0, (seq_len, d_model))
            impulse_noise = impulse_locations * impulse_magnitudes
            
            # Combined noise
            noise_level = np.random.uniform(0.2, 0.8)
            noise = np.zeros((seq_len, d_model))
            for j in range(d_model):
                correlation = np.random.uniform(0.3, 0.8)
                noise[:, j] = noise_level * (correlation * env_noise_base + 
                                           (1-correlation) * electronic_noise[:, j] + 
                                           impulse_noise[:, j])
            
            # Final signal
            final_signal = structured + noise
            
            self.data.append(final_signal)
            self.ground_truth.append({
                'structured': structured,
                'noise': noise,
                'speech_component': speech_signal
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'structured_gt': torch.tensor(self.ground_truth[idx]['structured'], dtype=torch.float32),
            'noise_gt': torch.tensor(self.ground_truth[idx]['noise'], dtype=torch.float32)
        }


class ScientificDataDataset(Dataset):
    """Scientific measurement data with systematic patterns + measurement error"""
    
    def __init__(self, num_experiments: int = 800, seq_len: int = 100, d_model: int = 48):
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.data = []
        self.ground_truth = []
        
        for _ in tqdm(range(num_experiments), desc="Generating scientific data"):
            # Different types of scientific phenomena
            experiment_type = np.random.choice(['exponential_decay', 'oscillation', 'growth_curve', 'phase_transition'])
            
            t = np.linspace(0, 10, seq_len)
            
            if experiment_type == 'exponential_decay':
                # Radioactive decay, chemical reactions, etc.
                decay_rate = np.random.uniform(0.1, 2.0)
                initial_value = np.random.uniform(50, 200)
                structured_base = initial_value * np.exp(-decay_rate * t)
                
            elif experiment_type == 'oscillation':
                # Pendulum, AC circuits, vibrations
                freq = np.random.uniform(0.5, 3.0)
                amplitude = np.random.uniform(10, 100)
                damping = np.random.uniform(0.05, 0.3)
                phase = np.random.uniform(0, 2*np.pi)
                structured_base = amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * freq * t + phase)
                
            elif experiment_type == 'growth_curve':
                # Population growth, bacterial cultures
                growth_rate = np.random.uniform(0.1, 1.0)
                carrying_capacity = np.random.uniform(80, 150)
                initial_pop = np.random.uniform(1, 10)
                structured_base = carrying_capacity / (1 + ((carrying_capacity - initial_pop) / initial_pop) * np.exp(-growth_rate * t))
                
            else:  # phase_transition
                # Magnetic transitions, chemical phase changes
                transition_point = np.random.uniform(3, 7)
                transition_sharpness = np.random.uniform(1, 5)
                low_value = np.random.uniform(10, 30)
                high_value = np.random.uniform(70, 120)
                structured_base = low_value + (high_value - low_value) / (1 + np.exp(-transition_sharpness * (t - transition_point)))
            
            # Add systematic experimental effects
            # Drift (instrument drift over time)
            drift_rate = np.random.uniform(-0.5, 0.5)
            drift = drift_rate * t
            
            # Calibration offset
            offset = np.random.uniform(-5, 5)
            
            # Temperature dependence (slow variation)
            temp_variation = np.random.uniform(0.5, 2.0) * np.sin(2 * np.pi * 0.1 * t)
            
            # Create multi-sensor measurements
            structured = np.zeros((seq_len, d_model))
            for j in range(d_model):
                # Each sensor has slightly different characteristics
                sensor_gain = np.random.uniform(0.95, 1.05)
                sensor_offset = np.random.uniform(-2, 2)
                sensor_nonlinearity = np.random.uniform(0.98, 1.02)
                
                structured[:, j] = (sensor_gain * (structured_base + offset + drift + temp_variation) ** sensor_nonlinearity + 
                                  sensor_offset)
            
            # Generate measurement noise
            # Thermal noise (Gaussian)
            thermal_noise = np.random.normal(0, np.random.uniform(1, 5), (seq_len, d_model))
            
            # Shot noise (Poisson-like)
            shot_noise_intensity = np.random.uniform(0.1, 1.0)
            shot_noise = np.random.poisson(shot_noise_intensity, (seq_len, d_model)) - shot_noise_intensity
            
            # Quantization noise (from ADC)
            quantization_step = np.random.uniform(0.1, 0.5)
            quantization_noise = np.random.uniform(-quantization_step/2, quantization_step/2, (seq_len, d_model))
            
            # Environmental interference (correlated)
            interference_freq = np.random.uniform(5, 15)  # 50/60 Hz power line interference
            interference_amplitude = np.random.uniform(0.5, 3.0)
            interference_base = interference_amplitude * np.sin(2 * np.pi * interference_freq * t)
            
            # Combined noise
            noise = np.zeros((seq_len, d_model))
            for j in range(d_model):
                coupling_strength = np.random.uniform(0.2, 0.8)
                noise[:, j] = (thermal_noise[:, j] + 
                              shot_noise[:, j] + 
                              quantization_noise[:, j] + 
                              coupling_strength * interference_base)
            
            # Final measurement
            final_data = structured + noise
            
            self.data.append(final_data)
            self.ground_truth.append({
                'structured': structured,
                'noise': noise,
                'base_phenomenon': structured_base,
                'experiment_type': experiment_type
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'structured_gt': torch.tensor(self.ground_truth[idx]['structured'], dtype=torch.float32),
            'noise_gt': torch.tensor(self.ground_truth[idx]['noise'], dtype=torch.float32),
            'experiment_type': self.ground_truth[idx]['experiment_type']
        }


class SPDEvaluationBenchmark:
    """Comprehensive SPD Router evaluation framework"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
    
    def evaluate_separation_quality(self, spd_router, dataset, dataset_name):
        """Evaluate how well SPD router separates structure from noise"""
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        spd_router.eval()
        
        all_structure_correlations = []
        all_noise_correlations = []
        all_cross_correlations = []
        all_snr_improvements = []
        all_reconstruction_errors = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                data = batch['data'].to(self.device)
                struct_gt = batch['structured_gt'].to(self.device)
                noise_gt = batch['noise_gt'].to(self.device)
                
                # Forward pass through SPD router
                output, x_struct, x_pseudo, sep_loss = spd_router(data)
                
                # Calculate metrics for each sample in batch
                for i in range(data.size(0)):
                    # Correlations with ground truth
                    struct_corr = self._calculate_correlation(x_struct[i], struct_gt[i])
                    noise_corr = self._calculate_correlation(x_pseudo[i], noise_gt[i])
                    
                    # Cross-correlation (should be low)
                    cross_corr = self._calculate_correlation(x_struct[i], x_pseudo[i])
                    
                    # SNR improvement
                    original_snr = self._calculate_snr(struct_gt[i], noise_gt[i])
                    separated_snr = self._calculate_snr(x_struct[i], x_pseudo[i])
                    snr_improvement = separated_snr - original_snr
                    
                    # Reconstruction error
                    recon_error = F.mse_loss(output[i], data[i]).item()
                    
                    all_structure_correlations.append(struct_corr)
                    all_noise_correlations.append(noise_corr)
                    all_cross_correlations.append(abs(cross_corr))
                    all_snr_improvements.append(snr_improvement)
                    all_reconstruction_errors.append(recon_error)
        
        return {
            'structure_correlation': np.mean(all_structure_correlations),
            'noise_correlation': np.mean(all_noise_correlations),
            'cross_correlation': np.mean(all_cross_correlations),
            'snr_improvement_db': np.mean(all_snr_improvements),
            'reconstruction_error': np.mean(all_reconstruction_errors),
            'structure_correlation_std': np.std(all_structure_correlations),
            'snr_improvement_std': np.std(all_snr_improvements)
        }
    
    def _calculate_correlation(self, x, y):
        """Calculate correlation coefficient between two tensors"""
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        x_centered = x_flat - x_flat.mean()
        y_centered = y_flat - y_flat.mean()
        
        correlation = (x_centered * y_centered).sum() / (x_centered.norm() * y_centered.norm() + 1e-8)
        return correlation.item()
    
    def _calculate_snr(self, signal, noise):
        """Calculate SNR in dB"""
        signal_power = (signal ** 2).mean()
        noise_power = (noise ** 2).mean()
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return snr_db.item()
    
    def evaluate_baseline_separation(self, data, struct_gt, noise_gt):
        """Baseline separation using simple filtering"""
        # Simple low-pass filter as baseline
        from scipy.signal import butter, filtfilt
        
        all_results = []
        
        for i in range(data.size(0)):
            sample = data[i].cpu().numpy()
            struct_true = struct_gt[i].cpu().numpy()
            noise_true = noise_gt[i].cpu().numpy()
            
            # Apply low-pass filter to each channel
            filtered = np.zeros_like(sample)
            residual = np.zeros_like(sample)
            
            for j in range(sample.shape[1]):
                # Design low-pass filter
                b, a = butter(3, 0.3, btype='low')
                filtered[:, j] = filtfilt(b, a, sample[:, j])
                residual[:, j] = sample[:, j] - filtered[:, j]
            
            # Calculate metrics
            struct_corr = np.corrcoef(filtered.flatten(), struct_true.flatten())[0, 1]
            noise_corr = np.corrcoef(residual.flatten(), noise_true.flatten())[0, 1]
            
            all_results.append({
                'structure_correlation': struct_corr if not np.isnan(struct_corr) else 0,
                'noise_correlation': noise_corr if not np.isnan(noise_corr) else 0
            })
        
        return {
            'structure_correlation': np.mean([r['structure_correlation'] for r in all_results]),
            'noise_correlation': np.mean([r['noise_correlation'] for r in all_results])
        }
    
    def train_spd_router(self, spd_router, dataset, epochs=20):
        """Train SPD router on dataset"""
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(spd_router.parameters(), lr=1e-3)
        
        spd_router.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                data = batch['data'].to(self.device)
                struct_gt = batch['structured_gt'].to(self.device)
                
                optimizer.zero_grad()
                
                output, x_struct, x_pseudo, sep_loss = spd_router(data)
                
                # Combined loss: reconstruction + separation + structure supervision
                recon_loss = F.mse_loss(output, data)
                struct_supervision = F.mse_loss(x_struct, struct_gt)
                
                total_loss_batch = recon_loss + 0.1 * sep_loss + 0.5 * struct_supervision
                total_loss_batch.backward()
                
                optimizer.step()
                total_loss += total_loss_batch.item()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def run_comprehensive_evaluation(self):
        """Run complete SPD Router evaluation"""
        print("Starting Comprehensive SPD Router Evaluation")
        print("=" * 60)
        
        results = {}
        
        # Test datasets
        datasets = {
            'financial': FinancialTimeSeriesDataset(num_series=200, seq_len=128, d_model=32),
            'audio': AudioSignalDataset(num_signals=200, seq_len=256, d_model=16),
            'scientific': ScientificDataDataset(num_experiments=200, seq_len=100, d_model=24)
        }
        
        for dataset_name, dataset in datasets.items():
            print(f"\n--- Evaluating on {dataset_name.upper()} data ---")
            
            d_model = dataset[0]['data'].shape[1]
            
            # Create SPD routers
            standard_spd = SPDRouter(d_model=d_model, separation_strength=0.1).to(self.device)
            adaptive_spd = AdaptiveSPDRouter(d_model=d_model).to(self.device)
            
            # Train both routers
            print("Training Standard SPD...")
            self.train_spd_router(standard_spd, dataset, epochs=15)
            
            print("Training Adaptive SPD...")
            self.train_spd_router(adaptive_spd, dataset, epochs=15)
            
            # Evaluate both routers
            print("Evaluating Standard SPD...")
            standard_results = self.evaluate_separation_quality(standard_spd, dataset, f"{dataset_name}_standard")
            
            print("Evaluating Adaptive SPD...")
            adaptive_results = self.evaluate_separation_quality(adaptive_spd, dataset, f"{dataset_name}_adaptive")
            
            # Evaluate baseline
            print("Evaluating Baseline...")
            sample_batch = next(iter(DataLoader(dataset, batch_size=50)))
            baseline_results = self.evaluate_baseline_separation(
                sample_batch['data'], 
                sample_batch['structured_gt'], 
                sample_batch['noise_gt']
            )
            
            results[dataset_name] = {
                'standard_spd': standard_results,
                'adaptive_spd': adaptive_results,
                'baseline_filter': baseline_results
            }
            
            # Print summary
            print(f"\nResults for {dataset_name}:")
            print(f"  Baseline - Structure Corr: {baseline_results['structure_correlation']:.3f}")
            print(f"  Standard SPD - Structure Corr: {standard_results['structure_correlation']:.3f}, SNR Gain: {standard_results['snr_improvement_db']:.2f} dB")
            print(f"  Adaptive SPD - Structure Corr: {adaptive_results['structure_correlation']:.3f}, SNR Gain: {adaptive_results['snr_improvement_db']:.2f} dB")
        
        self.results['spd_evaluation'] = results
        return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benchmark = SPDEvaluationBenchmark(device=device)
    results = benchmark.run_comprehensive_evaluation()
    
    # Save results
    with open('spd_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("SPD ROUTER EVALUATION COMPLETE")
    print("="*60)
    print("Results saved to spd_evaluation_results.json")