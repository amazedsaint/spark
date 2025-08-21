#!/usr/bin/env python3
"""
Iterative optimization based on benchmark findings.
Fix identified issues and re-evaluate to get better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from src.spark_transformer import SPaRKTransformer
from src.spd_router import SPDRouter
from src.verifier_head import VerifierHead

def optimize_spd_router():
    """Optimize SPD Router based on benchmark findings"""
    print("🔧 Optimizing SPD Router")
    print("-" * 30)
    
    device = torch.device('cpu')
    d_model = 32
    seq_len = 64
    batch_size = 8
    
    results = {}
    
    # Test different configurations
    configs = [
        {'separation_strength': 0.01, 'name': 'low_sep'},
        {'separation_strength': 0.1, 'name': 'medium_sep'},
        {'separation_strength': 0.5, 'name': 'high_sep'}
    ]
    
    for config in configs:
        print(f"Testing {config['name']} configuration...")
        
        # Create optimized SPD router
        spd_router = SPDRouter(
            d_model=d_model,
            separation_strength=config['separation_strength'],
            use_learned_basis=True,
            dropout=0.05  # Reduced dropout
        ).to(device)
        
        # Test on multiple noise levels
        noise_results = {}
        
        for noise_level in [0.1, 0.3, 0.5, 0.7]:
            # Generate test data
            t = torch.linspace(0, 4*np.pi, seq_len)
            
            # More complex structured signal
            structured = torch.zeros(batch_size, seq_len, d_model)
            for i in range(d_model):
                freq = (i + 1) * 0.5
                phase = np.random.uniform(0, 2*np.pi)
                structured[:, :, i] = torch.sin(freq * t + phase).unsqueeze(0).repeat(batch_size, 1)
            
            # Add noise
            noise = torch.randn_like(structured) * noise_level
            mixed_signal = structured + noise
            
            # Extended training
            optimizer = torch.optim.Adam(spd_router.parameters(), lr=5e-4)  # Lower LR
            spd_router.train()
            
            best_struct_corr = -1
            for epoch in range(30):  # More epochs
                optimizer.zero_grad()
                output, x_struct, x_pseudo, sep_loss = spd_router(mixed_signal)
                
                # Improved loss function
                recon_loss = F.mse_loss(output, mixed_signal)
                struct_target = structured[:, :, :x_struct.size(-1)]
                struct_supervision = F.mse_loss(x_struct, struct_target)
                
                # Add orthogonality constraint
                cross_corr = F.cosine_similarity(x_struct.flatten(), x_pseudo.flatten(), dim=0)
                ortho_loss = cross_corr ** 2
                
                total_loss = recon_loss + 0.3 * struct_supervision + sep_loss + 0.2 * ortho_loss
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(spd_router.parameters(), 1.0)
                optimizer.step()
                
                # Track best correlation
                with torch.no_grad():
                    struct_corr = F.cosine_similarity(
                        x_struct.flatten(), struct_target.flatten(), dim=0
                    ).item()
                    best_struct_corr = max(best_struct_corr, struct_corr)
            
            # Final evaluation
            spd_router.eval()
            with torch.no_grad():
                output, x_struct, x_pseudo, _ = spd_router(mixed_signal)
                
                struct_target = structured[:, :, :x_struct.size(-1)]
                noise_target = noise[:, :, :x_pseudo.size(-1)]
                
                struct_corr = F.cosine_similarity(x_struct.flatten(), struct_target.flatten(), dim=0).item()
                noise_corr = F.cosine_similarity(x_pseudo.flatten(), noise_target.flatten(), dim=0).item()
                cross_corr = F.cosine_similarity(x_struct.flatten(), x_pseudo.flatten(), dim=0).item()
                
                # SNR calculation
                signal_power = (struct_target ** 2).mean()
                noise_power = (noise_target ** 2).mean()
                original_snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                
                separated_signal_power = (x_struct ** 2).mean()
                separated_noise_power = (x_pseudo ** 2).mean()
                separated_snr = 10 * torch.log10(separated_signal_power / (separated_noise_power + 1e-8))
                
                snr_improvement = separated_snr - original_snr
                
                noise_results[noise_level] = {
                    'structure_correlation': struct_corr,
                    'noise_correlation': noise_corr,
                    'cross_correlation': abs(cross_corr),
                    'snr_improvement_db': snr_improvement.item(),
                    'best_training_correlation': best_struct_corr
                }
                
                print(f"  Noise {noise_level}: Struct corr {struct_corr:.3f}, SNR {snr_improvement.item():+.1f} dB")
        
        results[config['name']] = noise_results
    
    return results

def optimize_verifier_head():
    """Optimize Verifier Head based on benchmark findings"""
    print("\n🔧 Optimizing Verifier Head")
    print("-" * 30)
    
    device = torch.device('cpu')
    d_model = 64
    seq_len = 30
    vocab_size = 10
    
    # Generate more challenging parentheses data
    def generate_complex_parentheses(num_samples=500, max_depth=8):
        data = []
        labels = []
        
        for _ in range(num_samples):
            depth = np.random.randint(2, max_depth + 1)
            
            if np.random.random() < 0.5:
                # Generate properly nested structure
                sequence = []
                stack = []
                
                # Build up to depth
                for d in range(depth):
                    bracket_type = np.random.choice(['()', '[]', '{}'])
                    sequence.append(bracket_type[0])
                    stack.append(bracket_type[1])
                
                # Add some interleaving
                for _ in range(np.random.randint(0, depth)):
                    if stack and np.random.random() < 0.7:
                        # Close a bracket
                        sequence.append(stack.pop())
                    else:
                        # Open new bracket
                        bracket_type = np.random.choice(['()', '[]', '{}'])
                        sequence.append(bracket_type[0])
                        stack.append(bracket_type[1])
                
                # Close remaining
                while stack:
                    sequence.append(stack.pop())
                
                is_balanced = True
            else:
                # Generate unbalanced
                length = np.random.randint(4, max_depth * 2)
                sequence = []
                for _ in range(length):
                    bracket = np.random.choice(['(', ')', '[', ']', '{', '}'])
                    sequence.append(bracket)
                is_balanced = False
            
            # Convert to IDs
            token_map = {'(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6}
            token_ids = [token_map.get(t, 0) for t in sequence]
            
            # Pad
            if len(token_ids) < seq_len:
                token_ids += [0] * (seq_len - len(token_ids))
            else:
                token_ids = token_ids[:seq_len]
            
            data.append(token_ids)
            labels.append(1 if is_balanced else 0)
        
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    results = {}
    
    for model_type in ['optimized_verifier', 'standard', 'original_verifier']:
        print(f"Testing {model_type}...")
        
        if model_type == 'optimized_verifier':
            # Optimized verifier configuration
            model = SPaRKTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=2,  # More layers
                n_heads=4,
                enable_verifier=True,
                verification_types=['balanced_parens'],
                stack_size=16,
                verifier_penalty_strength=0.5,  # Reduced penalty
                dropout=0.05  # Less dropout
            ).to(device)
        elif model_type == 'original_verifier':
            # Original configuration
            model = SPaRKTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=1,
                n_heads=2,
                enable_verifier=True,
                verification_types=['balanced_parens']
            ).to(device)
        else:
            # Standard baseline
            class StandardModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, d_model)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model, 4, batch_first=True),
                        num_layers=2
                    )
                    self.classifier = nn.Linear(d_model, 2)
                
                def forward(self, x):
                    x = self.embedding(x)
                    x = self.transformer(x)
                    return self.classifier(x.mean(dim=1))
            
            model = StandardModel().to(device)
        
        # Generate training data (shallow) and test data (deep)
        train_x, train_y = generate_complex_parentheses(400, max_depth=4)  # Shallow
        test_x, test_y = generate_complex_parentheses(200, max_depth=8)    # Deep
        
        # Extended training
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
        
        model.train()
        best_val_acc = 0
        
        for epoch in range(25):  # More training
            epoch_loss = 0
            
            for i in range(0, len(train_x), 8):
                batch_x = train_x[i:i+8].to(device)
                batch_y = train_y[i:i+8].to(device)
                
                optimizer.zero_grad()
                
                if 'verifier' in model_type:
                    outputs, aux_info = model(batch_x)
                    outputs = outputs[:, -1, :2]
                    loss = F.cross_entropy(outputs, batch_y)
                    
                    # Add verifier loss with curriculum learning
                    if 'total_verification_loss' in aux_info:
                        verifier_weight = min(0.1, epoch * 0.004)  # Gradually increase
                        loss += verifier_weight * aux_info['total_verification_loss']
                else:
                    outputs = model(batch_x)
                    loss = F.cross_entropy(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            # Validation check every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    if 'verifier' in model_type:
                        val_outputs, _ = model(test_x[:32].to(device))
                        val_outputs = val_outputs[:, -1, :2]
                    else:
                        val_outputs = model(test_x[:32].to(device))
                    
                    val_preds = torch.argmax(val_outputs, dim=1)
                    val_acc = (val_preds == test_y[:32].to(device)).float().mean().item()
                    best_val_acc = max(best_val_acc, val_acc)
                
                model.train()
                print(f"  Epoch {epoch}: Loss {epoch_loss/(len(train_x)//8):.4f}, Val Acc {val_acc:.3f}")
        
        # Final test
        model.eval()
        with torch.no_grad():
            if 'verifier' in model_type:
                test_outputs, _ = model(test_x.to(device))
                test_outputs = test_outputs[:, -1, :2]
            else:
                test_outputs = model(test_x.to(device))
            
            test_preds = torch.argmax(test_outputs, dim=1)
            final_accuracy = (test_preds == test_y.to(device)).float().mean().item()
        
        results[model_type] = {
            'final_accuracy': final_accuracy,
            'best_val_accuracy': best_val_acc,
            'num_parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"  Final test accuracy: {final_accuracy:.3f}")
        print(f"  Best validation accuracy: {best_val_acc:.3f}")
    
    return results

def optimize_fk_attention():
    """Create more challenging tasks for FK-Attention"""
    print("\n🔧 Testing FK-Attention on Harder Tasks")
    print("-" * 30)
    
    device = torch.device('cpu')
    
    # Create path reasoning task with explicit multi-hop requirements
    def create_path_data(num_samples=300):
        data = []
        labels = []
        
        # Simple graph: 0->1->2->3->4->5 (linear chain)
        # Task: Given start and path length, predict end position
        
        for _ in range(num_samples):
            start_pos = np.random.randint(0, 4)  # Start positions 0-3
            path_length = np.random.randint(1, 4)  # Path lengths 1-3
            end_pos = min(start_pos + path_length, 5)
            
            # Create sequence: [start] [path_length] [padding...]
            sequence = [start_pos + 10, path_length + 20] + [0] * 18  # Pad to length 20
            
            data.append(sequence)
            labels.append(end_pos)
        
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    # Test different beta values for FK-Attention
    beta_values = [0.1, 0.3, 0.5, 0.7]
    results = {}
    
    for beta in beta_values:
        print(f"Testing beta={beta}...")
        
        # Create model with FK-Attention
        model = SPaRKTransformer(
            vocab_size=50,
            d_model=64,
            n_layers=2,
            n_heads=2,
            fk_beta=beta,
            fk_max_path_length=6,
            enable_verifier=False,
            use_adaptive_spd=False
        ).to(device)
        
        # Generate data
        train_x, train_y = create_path_data(250)
        test_x, test_y = create_path_data(100)
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        for epoch in range(20):
            for i in range(0, len(train_x), 8):
                batch_x = train_x[i:i+8].to(device)
                batch_y = train_y[i:i+8].to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(batch_x)
                outputs = outputs[:, -1, :6]  # Predict positions 0-5
                
                loss = F.cross_entropy(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Test
        model.eval()
        with torch.no_grad():
            test_outputs, _ = model(test_x.to(device))
            test_outputs = test_outputs[:, -1, :6]
            test_preds = torch.argmax(test_outputs, dim=1)
            accuracy = (test_preds == test_y.to(device)).float().mean().item()
        
        results[f'beta_{beta}'] = accuracy
        print(f"  Path reasoning accuracy: {accuracy:.3f}")
    
    # Compare with standard transformer
    class SimplePathModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(50, 64)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(64, 2, batch_first=True),
                num_layers=2
            )
            self.classifier = nn.Linear(64, 6)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            return self.classifier(x[:, -1, :])
    
    baseline_model = SimplePathModel().to(device)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    
    baseline_model.train()
    for epoch in range(20):
        for i in range(0, len(train_x), 8):
            batch_x = train_x[i:i+8].to(device)
            batch_y = train_y[i:i+8].to(device)
            
            optimizer.zero_grad()
            outputs = baseline_model(batch_x)
            loss = F.cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    baseline_model.eval()
    with torch.no_grad():
        baseline_outputs = baseline_model(test_x.to(device))
        baseline_preds = torch.argmax(baseline_outputs, dim=1)
        baseline_accuracy = (baseline_preds == test_y.to(device)).float().mean().item()
    
    results['baseline'] = baseline_accuracy
    print(f"Standard baseline accuracy: {baseline_accuracy:.3f}")
    
    return results

def run_optimization_iteration():
    """Run complete optimization iteration"""
    print("🚀 SPaR-K Optimization Iteration")
    print("Author: Anoop Madhusudanan (amazedsaint@gmail.com)")
    print("="*50)
    
    start_time = time.time()
    
    # Run optimizations
    spd_results = optimize_spd_router()
    fk_results = optimize_fk_attention()
    verifier_results = optimize_verifier_head()
    
    end_time = time.time()
    
    # Compile results
    optimization_results = {
        'spd_router_optimization': spd_results,
        'fk_attention_optimization': fk_results,
        'verifier_head_optimization': verifier_results,
        'optimization_time': end_time - start_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Analysis
    print(f"\n📊 OPTIMIZATION RESULTS")
    print("="*50)
    
    # SPD Router analysis
    print(f"\nSPD Router - Best Configuration:")
    best_spd_config = None
    best_spd_score = -100
    
    for config_name, noise_results in spd_results.items():
        # Calculate average improvement across noise levels
        improvements = [nr['snr_improvement_db'] for nr in noise_results.values()]
        avg_improvement = np.mean(improvements)
        
        print(f"  {config_name}: {avg_improvement:+.2f} dB average")
        
        if avg_improvement > best_spd_score:
            best_spd_score = avg_improvement
            best_spd_config = config_name
    
    print(f"  Best: {best_spd_config} ({best_spd_score:+.2f} dB)")
    
    # FK-Attention analysis
    print(f"\nFK-Attention - Best Configuration:")
    best_fk_beta = None
    best_fk_score = 0
    
    for beta_name, accuracy in fk_results.items():
        if beta_name != 'baseline':
            print(f"  {beta_name}: {accuracy:.3f}")
            if accuracy > best_fk_score:
                best_fk_score = accuracy
                best_fk_beta = beta_name
    
    baseline_acc = fk_results.get('baseline', 0)
    fk_improvement = best_fk_score - baseline_acc
    print(f"  Baseline: {baseline_acc:.3f}")
    print(f"  Best FK improvement: {fk_improvement:+.3f}")
    
    # Verifier Head analysis
    print(f"\nVerifier Head - Performance:")
    for model_name, result in verifier_results.items():
        print(f"  {model_name}: {result['final_accuracy']:.3f} (params: {result['num_parameters']:,})")
    
    # Overall assessment
    print(f"\n🎯 OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Optimization completed in {end_time - start_time:.1f} seconds")
    
    improvements_found = 0
    total_components = 3
    
    if best_spd_score > 0:
        improvements_found += 1
    if fk_improvement > 0.02:
        improvements_found += 1
    if any(result['final_accuracy'] > 0.76 for result in verifier_results.values() if 'verifier' in result):
        improvements_found += 1
    
    print(f"Components with improvements: {improvements_found}/{total_components}")
    
    if improvements_found == 0:
        status = "⚠️  No significant improvements found"
        recommendation = "Major architectural changes needed"
    elif improvements_found == total_components:
        status = "✅ All components improved"
        recommendation = "Architecture ready for advanced evaluation"
    else:
        status = "⚡ Partial improvements"
        recommendation = "Continue optimization on underperforming components"
    
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    
    # Save results
    with open('optimization_results.json', 'w') as f:
        json.dump(optimization_results, f, indent=2, default=str)
    
    return optimization_results

if __name__ == "__main__":
    results = run_optimization_iteration()
    print(f"\n📁 Optimization results saved to optimization_results.json")