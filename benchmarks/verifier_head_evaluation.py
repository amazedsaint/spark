"""
Comprehensive Verifier Head Evaluation

This module implements rigorous evaluation of the Verifier Head's ability to
maintain algorithmic invariants and enable systematic generalization.

Tests include:
1. Balanced parentheses/brackets with varying depths
2. Context-free grammar parsing (Dyck languages)
3. Stack-based computation (reverse Polish notation)
4. Nested structure validation (JSON, XML, code blocks)
5. Systematic generalization to longer sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import random
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.verifier_head import VerifierHead
from src.spark_transformer import SPaRKTransformer


class BalancedParenthesesDataset(Dataset):
    """Dataset for testing balanced parentheses validation at various depths"""
    
    def __init__(self, 
                 num_samples: int = 5000,
                 min_depth: int = 1,
                 max_depth: int = 10,
                 min_length: int = 4,
                 max_length: int = 50):
        
        self.samples = []
        self.vocab = {'(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6, '<PAD>': 0, '<START>': 7, '<END>': 8}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Generate balanced sequences
        for _ in tqdm(range(num_samples // 2), desc="Generating balanced sequences"):
            depth = random.randint(min_depth, max_depth)
            sequence, actual_depth, balance_trace = self._generate_balanced_sequence(depth, max_length)
            
            self.samples.append({
                'sequence': sequence,
                'is_balanced': True,
                'max_depth': actual_depth,
                'balance_trace': balance_trace,
                'length': len(sequence)
            })
        
        # Generate unbalanced sequences
        for _ in tqdm(range(num_samples // 2), desc="Generating unbalanced sequences"):
            # Start with balanced, then corrupt
            depth = random.randint(min_depth, max_depth)
            sequence, actual_depth, balance_trace = self._generate_balanced_sequence(depth, max_length)
            
            # Introduce imbalance
            sequence, balance_trace = self._introduce_imbalance(sequence)
            
            self.samples.append({
                'sequence': sequence,
                'is_balanced': False,
                'max_depth': actual_depth,
                'balance_trace': balance_trace,
                'length': len(sequence)
            })
    
    def _generate_balanced_sequence(self, target_depth: int, max_length: int) -> Tuple[List[str], int, List[int]]:
        """Generate a balanced sequence with approximately target depth"""
        bracket_types = [('(', ')'), ('[', ']'), ('{', '}')]
        sequence = []
        stack = []
        balance_trace = []
        max_depth_reached = 0
        
        # Generate opening brackets to reach target depth
        for _ in range(target_depth):
            bracket_type = random.choice(bracket_types)
            sequence.append(bracket_type[0])
            stack.append(bracket_type[1])
            balance_trace.append(len(stack))
            max_depth_reached = max(max_depth_reached, len(stack))
        
        # Add more complexity
        remaining_length = random.randint(0, max_length - len(sequence) - len(stack))
        
        for _ in range(remaining_length):
            if len(sequence) + len(stack) >= max_length:
                break
                
            if len(stack) == 0 or random.random() < 0.5:
                # Add opening bracket
                bracket_type = random.choice(bracket_types)
                sequence.append(bracket_type[0])
                stack.append(bracket_type[1])
                balance_trace.append(len(stack))
                max_depth_reached = max(max_depth_reached, len(stack))
            else:
                # Add closing bracket
                closing = stack.pop()
                sequence.append(closing)
                balance_trace.append(len(stack))
        
        # Close remaining brackets
        while stack:
            closing = stack.pop()
            sequence.append(closing)
            balance_trace.append(len(stack))
        
        return sequence, max_depth_reached, balance_trace
    
    def _introduce_imbalance(self, sequence: List[str]) -> Tuple[List[str], List[int]]:
        """Introduce imbalance into a sequence"""
        corrupted = sequence.copy()
        
        corruption_type = random.choice(['remove_bracket', 'wrong_bracket', 'extra_bracket'])
        
        if corruption_type == 'remove_bracket' and len(corrupted) > 1:
            # Remove a random bracket
            idx = random.randint(0, len(corrupted) - 1)
            corrupted.pop(idx)
        
        elif corruption_type == 'wrong_bracket' and len(corrupted) > 0:
            # Change a bracket to wrong type
            idx = random.randint(0, len(corrupted) - 1)
            if corrupted[idx] in ['(', ')', '[', ']']:
                corrupted[idx] = random.choice(['{', '}'])
            elif corrupted[idx] in ['{', '}']:
                corrupted[idx] = random.choice(['(', ')'])
        
        elif corruption_type == 'extra_bracket':
            # Add an extra opening bracket
            bracket_type = random.choice(['(', '[', '{'])
            idx = random.randint(0, len(corrupted))
            corrupted.insert(idx, bracket_type)
        
        # Recalculate balance trace
        balance_trace = self._calculate_balance_trace(corrupted)
        
        return corrupted, balance_trace
    
    def _calculate_balance_trace(self, sequence: List[str]) -> List[int]:
        """Calculate balance trace for a sequence"""
        stack_depth = 0
        trace = []
        
        for token in sequence:
            if token in ['(', '[', '{']:
                stack_depth += 1
            elif token in [')', ']', '}']:
                stack_depth = max(0, stack_depth - 1)
            trace.append(stack_depth)
        
        return trace
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to token IDs
        sequence_ids = [self.vocab['<START>']]
        for token in sample['sequence']:
            sequence_ids.append(self.vocab.get(token, 0))
        sequence_ids.append(self.vocab['<END>'])
        
        # Pad to fixed length
        max_len = 80
        if len(sequence_ids) > max_len:
            sequence_ids = sequence_ids[:max_len]
        else:
            sequence_ids += [self.vocab['<PAD>']] * (max_len - len(sequence_ids))
        
        return {
            'input_ids': torch.tensor(sequence_ids, dtype=torch.long),
            'is_balanced': torch.tensor(1 if sample['is_balanced'] else 0, dtype=torch.long),
            'max_depth': sample['max_depth'],
            'length': sample['length'],
            'balance_trace': torch.tensor(sample['balance_trace'] + [0] * (max_len - len(sample['balance_trace'])), dtype=torch.float32)
        }


class DyckLanguageDataset(Dataset):
    """Multi-bracket Dyck language for testing complex nesting"""
    
    def __init__(self, 
                 num_samples: int = 3000,
                 num_bracket_types: int = 3,
                 max_depth: int = 8,
                 max_length: int = 60):
        
        self.samples = []
        self.num_bracket_types = num_bracket_types
        
        # Define bracket pairs
        self.bracket_pairs = [
            ('(', ')'), ('[', ']'), ('{', '}'), ('<', '>'), ('«', '»')
        ][:num_bracket_types]
        
        # Vocabulary
        self.vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2}
        idx = 3
        for open_b, close_b in self.bracket_pairs:
            self.vocab[open_b] = idx
            self.vocab[close_b] = idx + 1
            idx += 2
        
        # Generate valid Dyck words
        for _ in tqdm(range(num_samples // 2), desc="Generating valid Dyck words"):
            dyck_word = self._generate_dyck_word(max_depth, max_length)
            
            self.samples.append({
                'sequence': dyck_word,
                'is_valid': True,
                'complexity': self._calculate_complexity(dyck_word)
            })
        
        # Generate invalid sequences
        for _ in tqdm(range(num_samples // 2), desc="Generating invalid sequences"):
            # Start with valid, then corrupt
            dyck_word = self._generate_dyck_word(max_depth, max_length)
            corrupted = self._corrupt_dyck_word(dyck_word)
            
            self.samples.append({
                'sequence': corrupted,
                'is_valid': False,
                'complexity': self._calculate_complexity(corrupted)
            })
    
    def _generate_dyck_word(self, max_depth: int, max_length: int) -> List[str]:
        """Generate a valid Dyck word"""
        if max_length <= 0:
            return []
        
        # Choose random strategy
        if random.random() < 0.3:
            # Simple nested structure
            return self._generate_simple_nested(max_depth, max_length)
        else:
            # Complex interleaved structure
            return self._generate_complex_dyck(max_depth, max_length)
    
    def _generate_simple_nested(self, max_depth: int, max_length: int) -> List[str]:
        """Generate simple nested structure like ((()))(()())"""
        depth = random.randint(1, min(max_depth, max_length // 2))
        sequence = []
        
        # Choose bracket type
        bracket_type = random.choice(self.bracket_pairs)
        
        # Add opening brackets
        for _ in range(depth):
            sequence.append(bracket_type[0])
        
        # Add closing brackets
        for _ in range(depth):
            sequence.append(bracket_type[1])
        
        # Add more complexity if length allows
        remaining = max_length - len(sequence)
        if remaining >= 4:
            extra_word = self._generate_simple_nested(max_depth - depth, remaining)
            sequence.extend(extra_word)
        
        return sequence
    
    def _generate_complex_dyck(self, max_depth: int, max_length: int) -> List[str]:
        """Generate complex interleaved Dyck word"""
        sequence = []
        stack = []
        
        while len(sequence) < max_length - 1:
            if len(stack) == 0 or (len(stack) < max_depth and random.random() < 0.6):
                # Add opening bracket
                bracket_type = random.choice(self.bracket_pairs)
                sequence.append(bracket_type[0])
                stack.append(bracket_type[1])
            else:
                # Add closing bracket
                if stack:
                    closing = stack.pop()
                    sequence.append(closing)
        
        # Close remaining brackets
        while stack:
            closing = stack.pop()
            sequence.append(closing)
        
        return sequence
    
    def _corrupt_dyck_word(self, dyck_word: List[str]) -> List[str]:
        """Corrupt a valid Dyck word to make it invalid"""
        if not dyck_word:
            return dyck_word
        
        corrupted = dyck_word.copy()
        corruption_type = random.choice(['mismatch', 'unbalanced', 'wrong_order'])
        
        if corruption_type == 'mismatch':
            # Change a closing bracket to wrong type
            closing_positions = [i for i, token in enumerate(corrupted) 
                               if token in [pair[1] for pair in self.bracket_pairs]]
            if closing_positions:
                pos = random.choice(closing_positions)
                # Change to a different closing bracket
                current_type = corrupted[pos]
                available_types = [pair[1] for pair in self.bracket_pairs if pair[1] != current_type]
                if available_types:
                    corrupted[pos] = random.choice(available_types)
        
        elif corruption_type == 'unbalanced':
            # Remove a random bracket
            if len(corrupted) > 1:
                idx = random.randint(0, len(corrupted) - 1)
                corrupted.pop(idx)
        
        elif corruption_type == 'wrong_order':
            # Swap two adjacent brackets
            if len(corrupted) > 1:
                idx = random.randint(0, len(corrupted) - 2)
                corrupted[idx], corrupted[idx + 1] = corrupted[idx + 1], corrupted[idx]
        
        return corrupted
    
    def _calculate_complexity(self, sequence: List[str]) -> int:
        """Calculate structural complexity of sequence"""
        max_depth = 0
        current_depth = 0
        
        for token in sequence:
            if token in [pair[0] for pair in self.bracket_pairs]:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif token in [pair[1] for pair in self.bracket_pairs]:
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to token IDs
        sequence_ids = [self.vocab['<START>']]
        for token in sample['sequence']:
            sequence_ids.append(self.vocab.get(token, 0))
        sequence_ids.append(self.vocab['<END>'])
        
        # Pad to fixed length
        max_len = 100
        if len(sequence_ids) > max_len:
            sequence_ids = sequence_ids[:max_len]
        else:
            sequence_ids += [self.vocab['<PAD>']] * (max_len - len(sequence_ids))
        
        return {
            'input_ids': torch.tensor(sequence_ids, dtype=torch.long),
            'is_valid': torch.tensor(1 if sample['is_valid'] else 0, dtype=torch.long),
            'complexity': sample['complexity']
        }


class RPNCalculatorDataset(Dataset):
    """Reverse Polish Notation evaluation - tests stack-based computation"""
    
    def __init__(self, num_samples: int = 4000, max_length: int = 20):
        self.samples = []
        self.vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            '+': 3, '-': 4, '*': 5, '/': 6,
            '0': 10, '1': 11, '2': 12, '3': 13, '4': 14,
            '5': 15, '6': 16, '7': 17, '8': 18, '9': 19
        }
        
        for _ in tqdm(range(num_samples), desc="Generating RPN expressions"):
            expression, result, is_valid = self._generate_rpn_expression(max_length)
            
            self.samples.append({
                'expression': expression,
                'result': result,
                'is_valid': is_valid,
                'length': len(expression)
            })
    
    def _generate_rpn_expression(self, max_length: int) -> Tuple[List[str], float, bool]:
        """Generate RPN expression with known result"""
        if random.random() < 0.8:
            # Generate valid expression
            return self._generate_valid_rpn(max_length)
        else:
            # Generate invalid expression
            return self._generate_invalid_rpn(max_length)
    
    def _generate_valid_rpn(self, max_length: int) -> Tuple[List[str], float, bool]:
        """Generate valid RPN expression"""
        # Start with two numbers
        expression = [str(random.randint(1, 9)), str(random.randint(1, 9))]
        
        # Add operations
        operations = ['+', '-', '*']  # Avoid division for simplicity
        
        while len(expression) < max_length and len(expression) < 15:
            if random.random() < 0.7 and len(expression) >= 2:
                # Add operation
                op = random.choice(operations)
                expression.append(op)
            else:
                # Add number
                expression.append(str(random.randint(1, 9)))
        
        # Evaluate expression
        try:
            result = self._evaluate_rpn(expression)
            return expression, result, True
        except:
            # If evaluation fails, make it simpler
            simple_expr = [str(random.randint(1, 9)), str(random.randint(1, 9)), '+']
            result = int(simple_expr[0]) + int(simple_expr[1])
            return simple_expr, result, True
    
    def _generate_invalid_rpn(self, max_length: int) -> Tuple[List[str], float, bool]:
        """Generate invalid RPN expression"""
        length = random.randint(3, max_length)
        expression = []
        
        # Create imbalanced expression (too many operators or operands)
        if random.random() < 0.5:
            # Too many operators
            expression = [str(random.randint(1, 9)), '+', '+']
        else:
            # Too many operands
            expression = [str(random.randint(1, 9)), str(random.randint(1, 9)), str(random.randint(1, 9))]
        
        return expression, 0.0, False
    
    def _evaluate_rpn(self, expression: List[str]) -> float:
        """Evaluate RPN expression"""
        stack = []
        
        for token in expression:
            if token.isdigit():
                stack.append(float(token))
            elif token in ['+', '-', '*', '/']:
                if len(stack) < 2:
                    raise ValueError("Invalid RPN expression")
                
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    if b == 0:
                        raise ValueError("Division by zero")
                    result = a / b
                
                stack.append(result)
        
        if len(stack) != 1:
            raise ValueError("Invalid RPN expression")
        
        return stack[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to token IDs
        sequence_ids = [self.vocab['<START>']]
        for token in sample['expression']:
            sequence_ids.append(self.vocab.get(token, 0))
        sequence_ids.append(self.vocab['<END>'])
        
        # Pad to fixed length
        max_len = 40
        if len(sequence_ids) > max_len:
            sequence_ids = sequence_ids[:max_len]
        else:
            sequence_ids += [self.vocab['<PAD>']] * (max_len - len(sequence_ids))
        
        return {
            'input_ids': torch.tensor(sequence_ids, dtype=torch.long),
            'is_valid': torch.tensor(1 if sample['is_valid'] else 0, dtype=torch.long),
            'result': torch.tensor(sample['result'], dtype=torch.float32),
            'length': sample['length']
        }


class VerifierHeadBenchmark:
    """Comprehensive benchmark for Verifier Head algorithmic capabilities"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
    
    def create_baseline_model(self, vocab_size: int, d_model: int = 128) -> nn.Module:
        """Create standard Transformer baseline without verifier head"""
        class StandardTransformer(nn.Module):
            def __init__(self, vocab_size, d_model):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Embedding(200, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.classifier = nn.Linear(d_model, 2)  # Binary classification
                
            def forward(self, x):
                seq_len = x.size(1)
                pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
                
                x = self.embedding(x) + self.pos_embedding(pos)
                x = self.transformer(x)
                x = self.classifier(x[:, -1, :])  # Use last token
                return x
        
        return StandardTransformer(vocab_size, d_model)
    
    def evaluate_systematic_generalization(self, model, dataset, model_type='spark'):
        """Test generalization to longer/deeper sequences than training"""
        # Split dataset by complexity for generalization testing
        train_samples = []
        test_samples = []
        
        if hasattr(dataset[0], 'max_depth'):
            # Split by depth
            for i, sample in enumerate(dataset):
                if dataset[i]['max_depth'] <= 5:  # Train on shallow
                    train_samples.append(i)
                else:  # Test on deep
                    test_samples.append(i)
        elif hasattr(dataset[0], 'complexity'):
            # Split by complexity
            for i, sample in enumerate(dataset):
                if dataset[i]['complexity'] <= 4:
                    train_samples.append(i)
                else:
                    test_samples.append(i)
        else:
            # Split by length
            for i, sample in enumerate(dataset):
                if dataset[i]['length'] <= 20:
                    train_samples.append(i)
                else:
                    test_samples.append(i)
        
        # Create data loaders
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_samples),
            batch_size=16, shuffle=True
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_samples),
            batch_size=16, shuffle=False
        )
        
        # Train model
        self._train_classification_model(model, train_loader, epochs=20, model_type=model_type)
        
        # Evaluate on train and test sets
        train_accuracy = self._evaluate_classification_model(model, train_loader, model_type)
        test_accuracy = self._evaluate_classification_model(model, test_loader, model_type)
        
        generalization_gap = train_accuracy - test_accuracy
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'generalization_gap': generalization_gap,
            'train_samples': len(train_samples),
            'test_samples': len(test_samples)
        }
    
    def _train_classification_model(self, model, dataloader, epochs=15, model_type='spark'):
        """Train model for binary classification"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                
                if 'is_balanced' in batch:
                    targets = batch['is_balanced'].to(self.device)
                elif 'is_valid' in batch:
                    targets = batch['is_valid'].to(self.device)
                else:
                    continue
                
                optimizer.zero_grad()
                
                if model_type == 'spark':
                    logits, aux_info = model(input_ids)
                    # Use last token for classification
                    logits = logits[:, -1, :2]  # First 2 classes
                    
                    # Add verifier loss if available
                    total_loss_batch = F.cross_entropy(logits, targets)
                    if 'total_verification_loss' in aux_info:
                        total_loss_batch += 0.1 * aux_info['total_verification_loss']
                else:
                    logits = model(input_ids)
                    total_loss_batch = F.cross_entropy(logits, targets)
                
                total_loss_batch.backward()
                optimizer.step()
                total_loss += total_loss_batch.item()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def _evaluate_classification_model(self, model, dataloader, model_type='spark'):
        """Evaluate classification accuracy"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                
                if 'is_balanced' in batch:
                    targets = batch['is_balanced'].to(self.device)
                elif 'is_valid' in batch:
                    targets = batch['is_valid'].to(self.device)
                else:
                    continue
                
                if model_type == 'spark':
                    logits, _ = model(input_ids)
                    logits = logits[:, -1, :2]
                else:
                    logits = model(input_ids)
                
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        return accuracy
    
    def run_comprehensive_evaluation(self):
        """Run complete Verifier Head evaluation"""
        print("Starting Comprehensive Verifier Head Evaluation")
        print("=" * 60)
        
        results = {}
        
        # Test datasets
        datasets = {
            'balanced_parens': BalancedParenthesesDataset(num_samples=2000, max_depth=8),
            'dyck_language': DyckLanguageDataset(num_samples=1500, num_bracket_types=3, max_depth=6),
            'rpn_calculator': RPNCalculatorDataset(num_samples=1500, max_length=15)
        }
        
        for dataset_name, dataset in datasets.items():
            print(f"\n--- Evaluating on {dataset_name.upper()} ---")
            
            # Determine vocab size from first sample
            vocab_size = max(dataset[0]['input_ids'].max().item() + 1, 50)
            
            # Create models
            spark_model = SPaRKTransformer(
                vocab_size=vocab_size,
                d_model=128,
                n_layers=2,
                n_heads=4,
                enable_verifier=True,
                verification_types=['balanced_parens'],
                stack_size=32,
                verifier_penalty_strength=1.0
            ).to(self.device)
            
            baseline_model = self.create_baseline_model(vocab_size, 128).to(self.device)
            
            # Test systematic generalization
            print("Testing SPaR-K systematic generalization...")
            spark_results = self.evaluate_systematic_generalization(
                spark_model, dataset, 'spark'
            )
            
            print("Testing Baseline systematic generalization...")
            baseline_results = self.evaluate_systematic_generalization(
                baseline_model, dataset, 'baseline'
            )
            
            results[dataset_name] = {
                'spark': spark_results,
                'baseline': baseline_results
            }
            
            # Print summary
            print(f"\nResults for {dataset_name}:")
            print(f"  SPaR-K - Train: {spark_results['train_accuracy']:.3f}, Test: {spark_results['test_accuracy']:.3f}, Gap: {spark_results['generalization_gap']:.3f}")
            print(f"  Baseline - Train: {baseline_results['train_accuracy']:.3f}, Test: {baseline_results['test_accuracy']:.3f}, Gap: {baseline_results['generalization_gap']:.3f}")
            
            # Analyze by complexity
            if dataset_name == 'balanced_parens':
                self._analyze_by_depth(dataset, spark_model, baseline_model, results, dataset_name)
        
        self.results['verifier_evaluation'] = results
        return results
    
    def _analyze_by_depth(self, dataset, spark_model, baseline_model, results, dataset_name):
        """Analyze performance by nesting depth"""
        depth_results = {'spark': {}, 'baseline': {}}
        
        # Group samples by depth
        depth_groups = {}
        for i, sample in enumerate(dataset):
            depth = sample['max_depth']
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(i)
        
        for depth, indices in depth_groups.items():
            if len(indices) < 10:  # Skip depths with too few samples
                continue
            
            subset_loader = DataLoader(
                torch.utils.data.Subset(dataset, indices),
                batch_size=16, shuffle=False
            )
            
            spark_acc = self._evaluate_classification_model(spark_model, subset_loader, 'spark')
            baseline_acc = self._evaluate_classification_model(baseline_model, subset_loader, 'baseline')
            
            depth_results['spark'][depth] = spark_acc
            depth_results['baseline'][depth] = baseline_acc
        
        results[dataset_name]['depth_analysis'] = depth_results
        
        print(f"\nDepth Analysis for {dataset_name}:")
        for depth in sorted(depth_results['spark'].keys()):
            spark_acc = depth_results['spark'][depth]
            baseline_acc = depth_results['baseline'][depth]
            print(f"  Depth {depth}: SPaR-K {spark_acc:.3f}, Baseline {baseline_acc:.3f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benchmark = VerifierHeadBenchmark(device=device)
    results = benchmark.run_comprehensive_evaluation()
    
    # Save results
    with open('verifier_head_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("VERIFIER HEAD EVALUATION COMPLETE")
    print("="*60)
    print("Results saved to verifier_head_results.json")