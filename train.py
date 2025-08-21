import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import os
import math
from tqdm import tqdm
from typing import Dict, Any, Optional

from src.spark_transformer import SPaRKTransformer


class SimpleTextDataset(Dataset):
    """Simple text dataset for training SPaR-K model"""
    
    def __init__(self, texts: list, tokenizer_vocab: dict, max_length: int = 512):
        self.texts = texts
        self.vocab = tokenizer_vocab
        self.reverse_vocab = {v: k for k, v in tokenizer_vocab.items()}
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def tokenize(self, text: str) -> list:
        """Simple word-level tokenization"""
        tokens = text.lower().split()
        return [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenize(text)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.vocab.get('<pad>', 1)] * (self.max_length - len(tokens))
        
        # Input and target (for language modeling)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "targets": targets
        }


def create_synthetic_data(vocab_size: int = 1000, num_samples: int = 10000) -> tuple:
    """Create synthetic text data for training"""
    
    # Create vocabulary
    vocab = {f"<pad>": 0, f"<unk>": 1, f"<start>": 2, f"<end>": 3}
    for i in range(4, vocab_size):
        vocab[f"token_{i}"] = i
    
    # Generate synthetic texts with some structure
    texts = []
    for _ in range(num_samples):
        # Random length
        length = torch.randint(10, 100, (1,)).item()
        
        # Generate text with some patterns
        tokens = ["<start>"]
        
        # Add structured patterns (parentheses, repeated sequences)
        if torch.rand(1).item() < 0.3:
            # Balanced parentheses pattern
            depth = 0
            for _ in range(length // 2):
                if depth == 0 or torch.rand(1).item() < 0.6:
                    tokens.append("(")
                    depth += 1
                else:
                    tokens.append(")")
                    depth -= 1
                    
                # Add random token
                random_token = f"token_{torch.randint(4, min(50, vocab_size), (1,)).item()}"
                tokens.append(random_token)
            
            # Close remaining parentheses
            while depth > 0:
                tokens.append(")")
                depth -= 1
        else:
            # Random sequence with some repetitions
            for _ in range(length):
                if torch.rand(1).item() < 0.2 and len(tokens) > 1:
                    # Repeat previous token
                    tokens.append(tokens[-1])
                else:
                    random_token = f"token_{torch.randint(4, min(100, vocab_size), (1,)).item()}"
                    tokens.append(random_token)
        
        tokens.append("<end>")
        texts.append(" ".join(tokens))
    
    return texts, vocab


def train_epoch(
    model: SPaRKTransformer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_verifier: float = 0.1,
    lambda_separation: float = 0.05,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch"""
    
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_verification_loss = 0.0
    total_separation_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        
        # Forward pass
        logits, aux_info = model(input_ids)
        
        # Compute loss
        total_loss_batch, loss_components = model.compute_total_loss(
            logits, targets, aux_info, lambda_verifier, lambda_separation
        )
        
        # Backward pass
        total_loss_batch.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_task_loss += loss_components["task_loss"].item()
        
        if isinstance(loss_components["verification_loss"], torch.Tensor):
            total_verification_loss += loss_components["verification_loss"].item()
        else:
            total_verification_loss += loss_components["verification_loss"]
            
        if isinstance(loss_components["separation_loss"], torch.Tensor):
            total_separation_loss += loss_components["separation_loss"].item()
        else:
            total_separation_loss += loss_components["separation_loss"]
            
        num_batches += 1
    
    return {
        "total_loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "verification_loss": total_verification_loss / num_batches,
        "separation_loss": total_separation_loss / num_batches
    }


def evaluate(
    model: SPaRKTransformer,
    dataloader: DataLoader, 
    device: torch.device,
    lambda_verifier: float = 0.1,
    lambda_separation: float = 0.05
) -> Dict[str, float]:
    """Evaluate the model"""
    
    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_verification_loss = 0.0
    total_separation_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            
            # Forward pass
            logits, aux_info = model(input_ids)
            
            # Compute loss
            total_loss_batch, loss_components = model.compute_total_loss(
                logits, targets, aux_info, lambda_verifier, lambda_separation
            )
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_task_loss += loss_components["task_loss"].item()
            
            if isinstance(loss_components["verification_loss"], torch.Tensor):
                total_verification_loss += loss_components["verification_loss"].item()
            else:
                total_verification_loss += loss_components["verification_loss"]
                
            if isinstance(loss_components["separation_loss"], torch.Tensor):
                total_separation_loss += loss_components["separation_loss"].item()
            else:
                total_separation_loss += loss_components["separation_loss"]
                
            num_batches += 1
    
    return {
        "total_loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "verification_loss": total_verification_loss / num_batches,
        "separation_loss": total_separation_loss / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Train SPaR-K Transformer")
    parser.add_argument("--config", type=str, default="configs/spark_config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "model": {
                "vocab_size": 1000,
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 8,
                "max_seq_length": 512,
                "dropout": 0.1,
                "fk_beta": 0.5,
                "fk_approximation": "krylov",
                "use_adaptive_spd": True,
                "enable_verifier": True,
                "verification_types": ["balanced_parens", "sequence_length"]
            },
            "training": {
                "batch_size": 16,
                "num_epochs": 10,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "lambda_verifier": 0.1,
                "lambda_separation": 0.05,
                "grad_clip": 1.0,
                "warmup_steps": 1000
            },
            "data": {
                "num_samples": 10000,
                "train_split": 0.8,
                "max_length": 256
            }
        }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic data
    texts, vocab = create_synthetic_data(
        vocab_size=config["model"]["vocab_size"],
        num_samples=config["data"]["num_samples"]
    )
    
    # Split data
    train_size = int(config["data"]["train_split"] * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    # Create datasets
    train_dataset = SimpleTextDataset(train_texts, vocab, config["data"]["max_length"])
    val_dataset = SimpleTextDataset(val_texts, vocab, config["data"]["max_length"])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    model = SPaRKTransformer(**config["model"])
    model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["training"]["num_epochs"],
        eta_min=config["training"]["learning_rate"] * 0.1
    )
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            config["training"]["lambda_verifier"],
            config["training"]["lambda_separation"],
            config["training"]["grad_clip"]
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader, device,
            config["training"]["lambda_verifier"],
            config["training"]["lambda_separation"]
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Task: {train_metrics['task_loss']:.4f}, "
              f"Ver: {train_metrics['verification_loss']:.4f}, "
              f"Sep: {train_metrics['separation_loss']:.4f})")
        
        print(f"Val Loss: {val_metrics['total_loss']:.4f} "
              f"(Task: {val_metrics['task_loss']:.4f}, "
              f"Ver: {val_metrics['verification_loss']:.4f}, "
              f"Sep: {val_metrics['separation_loss']:.4f})")
        
        # Tensorboard logging
        writer.add_scalar("train/total_loss", train_metrics["total_loss"], epoch)
        writer.add_scalar("train/task_loss", train_metrics["task_loss"], epoch) 
        writer.add_scalar("train/verification_loss", train_metrics["verification_loss"], epoch)
        writer.add_scalar("train/separation_loss", train_metrics["separation_loss"], epoch)
        
        writer.add_scalar("val/total_loss", val_metrics["total_loss"], epoch)
        writer.add_scalar("val/task_loss", val_metrics["task_loss"], epoch)
        writer.add_scalar("val/verification_loss", val_metrics["verification_loss"], epoch)
        writer.add_scalar("val/separation_loss", val_metrics["separation_loss"], epoch)
        
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        # Save best model
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["total_loss"],
                "config": config
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            print("Saved best model!")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["total_loss"],
                "config": config
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    writer.close()
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()