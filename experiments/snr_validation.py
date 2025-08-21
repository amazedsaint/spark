import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spd_router import SPDRouter, AdaptiveSPDRouter


def create_structured_signal(batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
    """Create a signal with clear structure (periodic patterns, trends)"""
    
    signals = []
    for _ in range(batch_size):
        # Base structured patterns
        t = torch.linspace(0, 4 * np.pi, seq_len).unsqueeze(1)
        
        # Combine multiple structured components
        signal = torch.zeros(seq_len, d_model)
        
        # Sine waves with different frequencies
        for i in range(d_model // 4):
            freq = (i + 1) * 0.5
            phase = np.random.uniform(0, 2 * np.pi)
            signal[:, i] = torch.sin(freq * t.squeeze() + phase)
        
        # Polynomial trends
        for i in range(d_model // 4, d_model // 2):
            coeffs = np.random.normal(0, 0.1, 3)
            t_norm = torch.linspace(-1, 1, seq_len)
            signal[:, i] = coeffs[0] + coeffs[1] * t_norm + coeffs[2] * t_norm ** 2
        
        # Periodic square waves
        for i in range(d_model // 2, 3 * d_model // 4):
            period = np.random.randint(10, 30)
            signal[:, i] = torch.sign(torch.sin(2 * np.pi * torch.arange(seq_len) / period))
        
        # Linear ramps and steps
        for i in range(3 * d_model // 4, d_model):
            if np.random.rand() > 0.5:
                # Linear ramp
                signal[:, i] = torch.linspace(-1, 1, seq_len)
            else:
                # Step function
                step_point = seq_len // 2
                signal[:step_point, i] = -0.5
                signal[step_point:, i] = 0.5
        
        signals.append(signal)
    
    return torch.stack(signals)


def create_noise_signal(batch_size: int, seq_len: int, d_model: int, noise_level: float = 1.0) -> torch.Tensor:
    """Create pseudo-random noise signal"""
    return torch.randn(batch_size, seq_len, d_model) * noise_level


def add_structured_noise(clean_signal: torch.Tensor, noise_level: float = 0.5) -> torch.Tensor:
    """Add structured noise to clean signal"""
    batch_size, seq_len, d_model = clean_signal.shape
    
    # Generate different types of noise
    noise_components = []
    
    # Gaussian noise
    gaussian_noise = torch.randn_like(clean_signal) * 0.3
    noise_components.append(gaussian_noise)
    
    # Impulsive noise (sparse outliers)
    impulse_mask = torch.rand_like(clean_signal) < 0.05  # 5% impulse noise
    impulse_noise = impulse_mask.float() * torch.randn_like(clean_signal) * 2.0
    noise_components.append(impulse_noise)
    
    # Colored noise (low-frequency disturbance)
    colored_noise = torch.randn_like(clean_signal)
    # Apply simple low-pass filter
    for b in range(batch_size):
        for d in range(d_model):
            colored_noise[b, 1:, d] = 0.7 * colored_noise[b, :-1, d] + 0.3 * colored_noise[b, 1:, d]
    noise_components.append(colored_noise * 0.4)
    
    # Combine all noise components
    total_noise = sum(noise_components) * noise_level
    
    return clean_signal + total_noise


class BaselineProcessor(nn.Module):
    """Baseline processor without SPD routing"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x):
        return self.processor(x), x, x, 0.0  # Match SPD interface


def compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB"""
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr.item()


def compute_spectral_whiteness(signal: torch.Tensor) -> float:
    """Compute spectral whiteness measure (lower is whiter)"""
    # Compute power spectral density
    fft_signal = torch.fft.fft(signal, dim=1)
    power_spectrum = torch.abs(fft_signal) ** 2
    
    # Compute variance of power spectrum (whiter = more uniform = lower variance)
    spectral_variance = torch.var(power_spectrum, dim=1).mean()
    return spectral_variance.item()


def evaluate_snr_improvement():
    """Evaluate SNR improvement with SPD routing"""
    
    print("=== SNR Improvement Experiment ===")
    
    # Parameters
    batch_size = 16
    seq_len = 128
    d_model = 64
    num_test_batches = 50
    noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    spd_router = SPDRouter(d_model=d_model, separation_strength=0.1).to(device)
    adaptive_spd = AdaptiveSPDRouter(d_model=d_model).to(device)
    baseline = BaselineProcessor(d_model=d_model).to(device)
    
    models = {
        "Baseline": baseline,
        "SPD Router": spd_router, 
        "Adaptive SPD": adaptive_spd
    }
    
    # Training phase (brief)
    print("Training models...")
    for name, model in models.items():
        if name == "Baseline":
            continue
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for _ in tqdm(range(100), desc=f"Training {name}"):
            # Create training data
            clean_structured = create_structured_signal(batch_size, seq_len, d_model).to(device)
            noisy_signal = add_structured_noise(clean_structured, noise_level=0.5)
            
            optimizer.zero_grad()
            output, x_struct, x_pseudo, sep_loss = model(noisy_signal)
            
            # Loss: reconstruction + separation penalty
            recon_loss = F.mse_loss(output, clean_structured)
            total_loss = recon_loss + sep_loss
            
            total_loss.backward()
            optimizer.step()
    
    # Evaluation phase
    print("\nEvaluating SNR improvements...")
    
    results = {name: {"snr_improvements": [], "whiteness_improvements": []} 
               for name in models.keys()}
    
    for noise_level in noise_levels:
        print(f"\nNoise level: {noise_level}")
        
        model_snr_improvements = {}
        model_whiteness_improvements = {}
        
        for name, model in models.items():
            model.eval()
            snr_improvements = []
            whiteness_improvements = []
            
            with torch.no_grad():
                for _ in tqdm(range(num_test_batches), desc=f"Testing {name}"):
                    # Create test data
                    clean_structured = create_structured_signal(batch_size, seq_len, d_model).to(device)
                    noisy_signal = add_structured_noise(clean_structured, noise_level=noise_level)
                    
                    # Process through model
                    output, x_struct, x_pseudo, _ = model(noisy_signal)
                    
                    # Compute metrics for each sample in batch
                    for b in range(batch_size):
                        # Original SNR
                        noise = noisy_signal[b] - clean_structured[b]
                        original_snr = compute_snr(clean_structured[b], noise)
                        
                        # Processed SNR
                        processed_noise = output[b] - clean_structured[b]
                        processed_snr = compute_snr(clean_structured[b], processed_noise)
                        
                        snr_improvement = processed_snr - original_snr
                        snr_improvements.append(snr_improvement)
                        
                        # Spectral whiteness
                        original_whiteness = compute_spectral_whiteness(noise)
                        processed_whiteness = compute_spectral_whiteness(processed_noise)
                        
                        whiteness_improvement = original_whiteness - processed_whiteness  # Lower is better
                        whiteness_improvements.append(whiteness_improvement)
            
            avg_snr_improvement = np.mean(snr_improvements)
            avg_whiteness_improvement = np.mean(whiteness_improvements)
            
            model_snr_improvements[name] = avg_snr_improvement
            model_whiteness_improvements[name] = avg_whiteness_improvement
            
            print(f"  {name}: SNR +{avg_snr_improvement:.2f} dB, Whiteness +{avg_whiteness_improvement:.4f}")
        
        # Store results
        for name in models.keys():
            results[name]["snr_improvements"].append(model_snr_improvements[name])
            results[name]["whiteness_improvements"].append(model_whiteness_improvements[name])
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    print("Noise Level | Baseline SNR | SPD Router SNR | Adaptive SPD SNR")
    print("-" * 65)
    
    for i, noise_level in enumerate(noise_levels):
        baseline_snr = results["Baseline"]["snr_improvements"][i]
        spd_snr = results["SPD Router"]["snr_improvements"][i]
        adaptive_snr = results["Adaptive SPD"]["snr_improvements"][i]
        
        print(f"   {noise_level:.1f}      |    {baseline_snr:+.2f} dB   |    {spd_snr:+.2f} dB     |     {adaptive_snr:+.2f} dB")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # SNR improvements
    for name, data in results.items():
        ax1.plot(noise_levels, data["snr_improvements"], 'o-', label=name, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('SNR Improvement (dB)', fontsize=12)
    ax1.set_title('SNR Improvement vs Noise Level', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Whiteness improvements
    for name, data in results.items():
        ax2.plot(noise_levels, data["whiteness_improvements"], 's-', label=name, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Spectral Whiteness Improvement', fontsize=12)
    ax2.set_title('Noise Whitening Performance', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('/home/claude-user/projects/spark/snr_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Demonstrate structure vs pseudo separation
    print("\n=== Structure vs Pseudo-randomness Separation ===")
    
    with torch.no_grad():
        # Create test signal
        clean_structured = create_structured_signal(1, seq_len, d_model).to(device)
        noisy_signal = add_structured_noise(clean_structured, noise_level=0.5)
        
        # Process with SPD router
        spd_router.eval()
        output, x_struct, x_pseudo, _ = spd_router(noisy_signal)
        
        # Compute correlation between separated components and ground truth
        clean_np = clean_structured[0].cpu().numpy()
        noise_np = (noisy_signal[0] - clean_structured[0]).cpu().numpy()
        struct_np = x_struct[0].cpu().numpy()
        pseudo_np = x_pseudo[0].cpu().numpy()
        
        # Correlation analysis
        clean_struct_corr = np.corrcoef(clean_np.flatten(), struct_np.flatten())[0, 1]
        noise_pseudo_corr = np.corrcoef(noise_np.flatten(), pseudo_np.flatten())[0, 1]
        
        print(f"Correlation between clean signal and extracted structure: {clean_struct_corr:.3f}")
        print(f"Correlation between noise and extracted pseudo-randomness: {noise_pseudo_corr:.3f}")
        
        # Mutual information proxy (negative correlation between structure and pseudo)
        struct_pseudo_corr = np.corrcoef(struct_np.flatten(), pseudo_np.flatten())[0, 1]
        print(f"Cross-correlation (structure vs pseudo): {struct_pseudo_corr:.3f} (lower is better)")
    
    return results


if __name__ == "__main__":
    results = evaluate_snr_improvement()
    print("\nSNR experiment completed! Results saved to snr_results.png")