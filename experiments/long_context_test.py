import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.verifier_head import VerifierHead
from src.spark_transformer import SPaRKTransformerBlock


def generate_dyck_language(length: int, num_pairs: int = 3) -> tuple:
    """
    Generate balanced parentheses strings (Dyck language)
    Returns string and validity labels for each position
    """
    symbols = ['(', ')', '[', ']', '{', '}'][:2*num_pairs]
    open_symbols = symbols[::2]
    close_symbols = symbols[1::2]
    
    sequence = []
    stack = []
    validity_labels = []
    
    for _ in range(length):
        if len(stack) == 0 or (len(stack) < length // 2 and np.random.rand() < 0.6):
            # Push open bracket
            bracket_type = np.random.randint(0, num_pairs)
            open_bracket = open_symbols[bracket_type]
            sequence.append(open_bracket)
            stack.append(bracket_type)
            validity_labels.append(1.0)  # Valid so far
        else:
            # Pop with matching close bracket
            if stack:
                bracket_type = stack.pop()
                close_bracket = close_symbols[bracket_type] 
                sequence.append(close_bracket)
                validity_labels.append(1.0)
            else:
                # This shouldn't happen in our generation
                sequence.append(')')
                validity_labels.append(0.0)  # Invalid
    
    # Close remaining brackets
    while stack:
        bracket_type = stack.pop()
        close_bracket = close_symbols[bracket_type]
        sequence.append(close_bracket)
        validity_labels.append(1.0)
    
    return sequence, validity_labels


def create_vocab_from_symbols():
    """Create vocabulary for parentheses symbols"""
    symbols = ['<pad>', '<start>', '<end>', '(', ')', '[', ']', '{', '}']
    vocab = {symbol: idx for idx, symbol in enumerate(symbols)}
    return vocab, symbols


class DyckDataset:
    """Dataset for Dyck language (balanced parentheses) tasks"""
    
    def __init__(self, num_samples: int = 1000, min_length: int = 20, max_length: int = 100):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.vocab, self.symbols = create_vocab_from_symbols()
        
        # Generate dataset
        self.sequences = []
        self.labels = []
        
        for _ in range(num_samples):
            length = np.random.randint(min_length, max_length + 1)
            sequence, validity_labels = generate_dyck_language(length)
            
            # Convert to indices
            sequence_indices = [self.vocab.get(symbol, 0) for symbol in sequence]
            
            self.sequences.append(sequence_indices)
            self.labels.append(validity_labels)
    
    def get_batch(self, batch_size: int, max_seq_len: int = None):
        """Get batch of sequences"""
        
        # Sample random sequences
        indices = np.random.choice(len(self.sequences), batch_size, replace=True)
        
        batch_sequences = []
        batch_labels = []
        
        for idx in indices:
            seq = self.sequences[idx]
            labels = self.labels[idx]
            
            # Truncate if needed
            if max_seq_len and len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
                labels = labels[:max_seq_len]
            
            batch_sequences.append(seq)
            batch_labels.append(labels)
        
        # Pad to same length
        max_len = max(len(seq) for seq in batch_sequences)
        
        padded_sequences = []
        padded_labels = []
        
        for seq, labels in zip(batch_sequences, batch_labels):
            # Pad with <pad> token (index 0)
            padded_seq = seq + [0] * (max_len - len(seq))
            padded_label = labels + [0.0] * (max_len - len(labels))
            
            padded_sequences.append(padded_seq)
            padded_labels.append(padded_label)
        
        return (
            torch.tensor(padded_sequences, dtype=torch.long),
            torch.tensor(padded_labels, dtype=torch.float32)
        )


class SimplifiedTransformerBlock(nn.Module):
    """Simplified transformer block without verifier for comparison"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        # Self attention
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x, {"verification_loss": 0.0}


class DyckModel(nn.Module):
    """Model for Dyck language validation"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, use_verifier: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.use_verifier = use_verifier
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(1000, d_model)  # Support up to 1000 positions
        
        # Transformer block
        if use_verifier:
            self.transformer_block = SPaRKTransformerBlock(
                d_model=d_model,
                enable_verifier=True,
                verification_types=["balanced_parens"],
                stack_size=32,
                verifier_penalty_strength=1.0
            )
        else:
            self.transformer_block = SimplifiedTransformerBlock(d_model)
        
        # Output head for validity prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        
        x = token_embeds + pos_embeds
        
        # Transformer processing
        if self.use_verifier:
            x, aux_info = self.transformer_block(x)
        else:
            x, aux_info = self.transformer_block(x)
        
        # Validity prediction
        validity_logits = self.output_head(x).squeeze(-1)
        
        return validity_logits, aux_info


def train_model(model, dataset, device, num_epochs=20, batch_size=8, max_seq_len=None):
    """Train model on Dyck language task"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0.0
        num_batches = 20
        
        for _ in range(num_batches):
            sequences, labels = dataset.get_batch(batch_size, max_seq_len)
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            validity_pred, aux_info = model(sequences)
            
            # Compute loss
            task_loss = F.binary_cross_entropy(validity_pred, labels)
            verification_loss = aux_info.get("verification_loss", 0.0)
            
            if isinstance(verification_loss, torch.Tensor):
                total_loss = task_loss + 0.1 * verification_loss
            else:
                total_loss = task_loss + 0.1 * torch.tensor(verification_loss, device=device)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        losses.append(epoch_loss / num_batches)
    
    return losses


def evaluate_generalization():
    """Test generalization to longer contexts"""
    
    print("=== Long Context Generalization Experiment ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vocab, _ = create_vocab_from_symbols()
    vocab_size = len(vocab)
    d_model = 128
    
    # Training parameters
    train_min_length = 20
    train_max_length = 100
    test_lengths = [150, 200, 300, 500]  # Longer than training
    
    print("Creating datasets...")
    
    # Training dataset (shorter sequences)
    train_dataset = DyckDataset(
        num_samples=2000, 
        min_length=train_min_length, 
        max_length=train_max_length
    )
    
    # Test datasets (longer sequences)
    test_datasets = {}
    for length in test_lengths:
        test_datasets[length] = DyckDataset(
            num_samples=500,
            min_length=length - 10, 
            max_length=length + 10
        )
    
    # Models to compare
    models = {
        "Without Verifier": DyckModel(vocab_size, d_model, use_verifier=False).to(device),
        "With Verifier": DyckModel(vocab_size, d_model, use_verifier=True).to(device)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train on short sequences
        train_losses = train_model(
            model, train_dataset, device, 
            num_epochs=25, batch_size=16, 
            max_seq_len=train_max_length
        )
        
        print(f"Final training loss: {train_losses[-1]:.4f}")
        
        # Test on different sequence lengths
        model.eval()
        accuracies = []
        
        with torch.no_grad():
            for test_length in test_lengths:
                print(f"Testing on length {test_length}...")
                
                test_dataset = test_datasets[test_length]
                correct = 0
                total = 0
                
                # Test on multiple batches
                for _ in range(20):
                    sequences, labels = test_dataset.get_batch(8, max_seq_len=test_length)
                    sequences, labels = sequences.to(device), labels.to(device)
                    
                    validity_pred, _ = model(sequences)
                    
                    # Convert to binary predictions
                    predictions = (validity_pred > 0.5).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.numel()
                
                accuracy = correct / total
                accuracies.append(accuracy)
                print(f"  Accuracy: {accuracy:.3f}")
        
        results[model_name] = accuracies
    
    # Print comparison
    print("\n=== GENERALIZATION RESULTS ===")
    print("Length | Without Verifier | With Verifier | Improvement")
    print("-" * 55)
    
    for i, length in enumerate(test_lengths):
        without_acc = results["Without Verifier"][i]
        with_acc = results["With Verifier"][i]
        improvement = with_acc - without_acc
        
        print(f" {length:3d}   |      {without_acc:.3f}      |    {with_acc:.3f}     |   {improvement:+.3f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    plt.plot(test_lengths, results["Without Verifier"], 'o-', 
             label='Without Verifier', linewidth=2, markersize=8, color='red')
    plt.plot(test_lengths, results["With Verifier"], 's-', 
             label='With Verifier Head', linewidth=2, markersize=8, color='blue')
    
    plt.xlabel('Test Sequence Length', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Generalization to Longer Contexts (Trained on 20-100 tokens)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Add training range indicator
    plt.axvspan(train_min_length, train_max_length, alpha=0.2, color='green', 
                label='Training Range')
    plt.legend(fontsize=11)
    
    # Add improvement annotations
    for i, (length, without_acc, with_acc) in enumerate(zip(test_lengths, results["Without Verifier"], results["With Verifier"])):
        improvement = with_acc - without_acc
        if improvement > 0:
            plt.annotate(f'+{improvement:.3f}', 
                        xy=(length, with_acc), xytext=(length, with_acc + 0.05),
                        ha='center', fontsize=10, color='green',
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/claude-user/projects/spark/long_context_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    results = evaluate_generalization()
    print("\nLong context experiment completed! Results saved to long_context_results.png")