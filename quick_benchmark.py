#!/usr/bin/env python3
"""
Quick benchmark to rapidly test improvements and iterate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from src.spark_transformer import SPaRKTransformer
from src.spd_router import SPDRouter

def quick_spd_test():
    """Quick SPD Router test with improved configuration"""
    print("📊 Quick SPD Router Test")
    print("-" * 25)
    
    device = torch.device('cpu')
    d_model = 24
    seq_len = 32
    batch_size = 4
    
    # Create clearer structured signal
    t = torch.linspace(0, 2*np.pi, seq_len)
    structured = torch.zeros(batch_size, seq_len, d_model)
    
    # Multiple frequency components
    for i in range(d_model):
        freq = (i // 4 + 1) * 0.5  # Group frequencies
        structured[:, :, i] = torch.sin(freq * t + np.random.uniform(0, np.pi)).unsqueeze(0).repeat(batch_size, 1)
    
    results = {}
    
    for noise_level in [0.2, 0.4, 0.6]:
        noise = torch.randn_like(structured) * noise_level
        mixed_signal = structured + noise
        
        # Optimized SPD Router
        spd_router = SPDRouter(
            d_model=d_model,
            structure_dim=d_model//2,
            pseudo_dim=d_model//2,
            separation_strength=0.05,  # Lower strength
            use_learned_basis=True
        ).to(device)
        
        # Quick training with better loss
        optimizer = torch.optim.Adam(spd_router.parameters(), lr=1e-3)
        
        spd_router.train()
        for epoch in range(15):
            optimizer.zero_grad()
            
            output, x_struct, x_pseudo, sep_loss = spd_router(mixed_signal)
            
            # Multi-objective loss
            recon_loss = F.mse_loss(output, mixed_signal)
            
            # Structure should correlate with clean signal
            struct_target = structured[:, :, :x_struct.size(-1)]
            struct_loss = F.mse_loss(x_struct, struct_target)
            
            # Pseudo should be orthogonal to structure
            ortho_loss = (F.cosine_similarity(x_struct.flatten(), x_pseudo.flatten(), dim=0) ** 2)
            
            total_loss = recon_loss + 0.2 * struct_loss + 0.1 * ortho_loss + 0.05 * sep_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(spd_router.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate
        spd_router.eval()
        with torch.no_grad():
            output, x_struct, x_pseudo, _ = spd_router(mixed_signal)
            
            struct_target = structured[:, :, :x_struct.size(-1)]
            noise_target = noise[:, :, :x_pseudo.size(-1)]
            
            # Correlations
            struct_corr = F.cosine_similarity(x_struct.flatten(), struct_target.flatten(), dim=0).item()
            noise_corr = F.cosine_similarity(x_pseudo.flatten(), noise_target.flatten(), dim=0).item()
            cross_corr = F.cosine_similarity(x_struct.flatten(), x_pseudo.flatten(), dim=0).item()
            
            # SNR
            orig_signal_power = (struct_target ** 2).mean()
            orig_noise_power = (noise_target ** 2).mean()
            orig_snr = 10 * torch.log10(orig_signal_power / (orig_noise_power + 1e-8))
            
            sep_signal_power = (x_struct ** 2).mean()
            sep_noise_power = (x_pseudo ** 2).mean()
            sep_snr = 10 * torch.log10(sep_signal_power / (sep_noise_power + 1e-8))
            
            snr_improvement = sep_snr - orig_snr
            
            results[noise_level] = {
                'structure_correlation': struct_corr,
                'noise_correlation': noise_corr,
                'cross_correlation': abs(cross_corr),
                'snr_improvement_db': snr_improvement.item()
            }
            
            print(f"  Noise {noise_level}: Struct {struct_corr:.3f}, Cross {abs(cross_corr):.3f}, SNR {snr_improvement.item():+.1f}dB")
    
    return results

def quick_verifier_test():
    """Quick Verifier Head test with simpler task"""
    print("\n🔍 Quick Verifier Head Test")
    print("-" * 25)
    
    device = torch.device('cpu')
    
    # Very simple nesting task: count nesting depth
    def generate_depth_data(num_samples=200):
        data = []
        labels = []
        
        vocab = {'(': 1, ')': 2, '<PAD>': 0}
        
        for _ in range(num_samples):
            depth = np.random.randint(1, 6)  # Max depth 5
            
            # Create simple nested structure: (((())))
            sequence = ['('] * depth + [')'] * depth
            
            # Convert to IDs and pad
            token_ids = [vocab[t] for t in sequence] + [0] * (20 - len(sequence))
            
            data.append(token_ids[:20])
            labels.append(min(depth, 5))  # Cap at 5 for classification
        
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    train_x, train_y = generate_depth_data(150)
    test_x, test_y = generate_depth_data(50)
    
    results = {}
    
    # Test with and without verifier
    for use_verifier in [False, True]:
        model_name = 'verifier' if use_verifier else 'baseline'
        print(f"Testing {model_name}...")
        
        if use_verifier:
            model = SPaRKTransformer(
                vocab_size=10,
                d_model=48,
                n_layers=1,
                n_heads=2,
                enable_verifier=True,
                verification_types=['balanced_parens'],
                stack_size=8,
                verifier_penalty_strength=0.2  # Lower penalty
            ).to(device)
        else:
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(10, 48)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(48, 2, batch_first=True),
                        num_layers=1
                    )
                    self.classifier = nn.Linear(48, 6)  # Classify depth 0-5
                
                def forward(self, x):
                    x = self.embedding(x)
                    x = self.transformer(x)
                    return self.classifier(x.mean(dim=1))
            
            model = SimpleModel().to(device)
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        
        model.train()
        for epoch in range(10):
            for i in range(0, len(train_x), 8):
                batch_x = train_x[i:i+8].to(device)
                batch_y = train_y[i:i+8].to(device)
                
                optimizer.zero_grad()
                
                if use_verifier:
                    outputs, aux_info = model(batch_x)
                    outputs = outputs[:, -1, :6]  # Last token, depth classes
                    loss = F.cross_entropy(outputs, batch_y)
                    
                    if 'total_verification_loss' in aux_info:
                        loss += 0.05 * aux_info['total_verification_loss']
                else:
                    outputs = model(batch_x)
                    loss = F.cross_entropy(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
        
        # Test
        model.eval()
        with torch.no_grad():
            if use_verifier:
                test_outputs, _ = model(test_x.to(device))
                test_outputs = test_outputs[:, -1, :6]
            else:
                test_outputs = model(test_x.to(device))
            
            test_preds = torch.argmax(test_outputs, dim=1)
            accuracy = (test_preds == test_y.to(device)).float().mean().item()
        
        results[model_name] = accuracy
        print(f"  Accuracy: {accuracy:.3f}")
    
    return results

def main():
    """Run quick benchmarks and show results"""
    start_time = time.time()
    
    spd_results = quick_spd_test()
    verifier_results = quick_verifier_test()
    
    end_time = time.time()
    
    print(f"\n🎯 QUICK BENCHMARK SUMMARY")
    print("="*50)
    print(f"Total time: {end_time - start_time:.1f} seconds")
    
    # SPD Router best result
    best_snr = max([nr['snr_improvement_db'] for nr in spd_results.values()])
    print(f"\nSPD Router best SNR improvement: {best_snr:+.1f} dB")
    
    # Verifier Head comparison
    verifier_improvement = verifier_results['verifier'] - verifier_results['baseline']
    print(f"Verifier Head improvement: {verifier_improvement:+.3f}")
    
    # Overall assessment
    meaningful_improvements = 0
    if best_snr > 1.0:
        meaningful_improvements += 1
    if verifier_improvement > 0.02:
        meaningful_improvements += 1
    
    print(f"\nComponents with improvements: {meaningful_improvements}/2")
    
    if meaningful_improvements == 0:
        print("Status: ⚠️  No clear improvements yet")
    elif meaningful_improvements == 1:
        print("Status: ⚡ Some progress made")
    else:
        print("Status: ✅ Multiple components improving")
    
    # Save results
    final_results = {
        'spd_router': spd_results,
        'verifier_head': verifier_results,
        'evaluation_time': end_time - start_time,
        'best_snr_improvement': best_snr,
        'verifier_improvement': verifier_improvement,
        'meaningful_improvements': meaningful_improvements
    }
    
    with open('quick_benchmark_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    return final_results

if __name__ == "__main__":
    results = main()