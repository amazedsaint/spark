"""
SPaR-K: Structure-Pseudo-Randomness with Kinetic Attention

Complete SPaR-K Transformer implementation combining three novel components:
1. Feynman-Kac Attention: Path-integral formulation for multi-hop reasoning
2. SPD Router: Structure vs pseudo-randomness decomposition (Tao's principle)
3. Verifier Head: Stack-based algorithmic verification

Author: Anoop Madhusudanan (amazedsaint@gmail.com)
Repository: https://github.com/amazedsaint/spark

Reference:
Madhusudanan, A. (2025). SPaR-K: Structure-Pseudo-Randomness with Kinetic 
Attention for Enhanced Transformer Reasoning.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math

from .feynman_kac_attention import FeynmanKacAttention
from .spd_router import SPDRouter, AdaptiveSPDRouter
from .verifier_head import VerifierHead


class SPaRKTransformerBlock(nn.Module):
    """
    Complete SPaR-K Transformer Block combining:
    - Feynman-Kac Attention for multi-hop reasoning
    - SPD Router for structure vs pseudo-randomness decomposition  
    - Verifier Head for algorithmic verification
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        # FK Attention params
        fk_beta: float = 0.5,
        fk_approximation: str = "krylov",
        fk_max_path_length: int = 10,
        # SPD Router params  
        use_adaptive_spd: bool = False,
        spd_separation_strength: float = 0.1,
        # Verifier Head params
        enable_verifier: bool = True,
        stack_size: int = 64,
        num_stacks: int = 1,
        verification_types: list = None,
        verifier_penalty_strength: float = 1.0,
        # Integration params
        residual_connection: str = "standard",  # "standard", "highway", "dense"
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.enable_verifier = enable_verifier
        self.residual_connection = residual_connection
        
        # Pre-attention layer norm
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # SPD Router (before attention)
        if use_adaptive_spd:
            self.spd_router = AdaptiveSPDRouter(
                d_model=d_model,
                separation_strength=spd_separation_strength,
                dropout=dropout
            )
        else:
            self.spd_router = SPDRouter(
                d_model=d_model, 
                separation_strength=spd_separation_strength,
                dropout=dropout
            )
        
        # Feynman-Kac Attention
        self.fk_attention = FeynmanKacAttention(
            d_model=d_model,
            n_heads=n_heads,
            beta=fk_beta,
            max_path_length=fk_max_path_length,
            approximation_method=fk_approximation,
            dropout=dropout
        )
        
        # Post-attention layer norm
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Verifier Head (processes intermediate states)
        if self.enable_verifier:
            self.verifier_head = VerifierHead(
                d_model=d_model,
                stack_size=stack_size,
                num_stacks=num_stacks,
                verification_types=verification_types or ["balanced_parens", "sequence_length"],
                penalty_strength=verifier_penalty_strength
            )
        
        # Residual connection variants
        if residual_connection == "highway":
            self.highway_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
        elif residual_connection == "dense":
            self.dense_weights = nn.Parameter(torch.ones(3))  # For input, attention, ffn
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        verifier_state: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            verifier_state: Previous verifier state
            
        Returns:
            output: Transformed output (batch_size, seq_len, d_model)  
            aux_info: Dictionary containing verification signals, penalties, and intermediate states
        """
        batch_size, seq_len, d_model = x.shape
        aux_info = {}
        
        # Store input for residual connections
        residual_input = x
        
        # Pre-attention layer norm
        x_norm = self.ln1(x)
        
        # SPD Router decomposition
        spd_output, x_struct, x_pseudo, separation_loss = self.spd_router(x_norm)
        aux_info["spd_separation_loss"] = separation_loss
        aux_info["structured_component"] = x_struct
        aux_info["pseudo_component"] = x_pseudo
        
        # Feynman-Kac Attention on SPD output
        attended = self.fk_attention(spd_output, spd_output, spd_output, attention_mask)
        
        # Post-attention residual connection
        if self.residual_connection == "highway":
            gate = self.highway_gate(residual_input)
            x = gate * attended + (1 - gate) * residual_input
        elif self.residual_connection == "dense":
            weights = torch.softmax(self.dense_weights, dim=0)
            x = weights[0] * residual_input + weights[1] * attended
        else:  # standard
            x = residual_input + self.dropout(attended)
        
        # Post-attention layer norm  
        x_norm2 = self.ln2(x)
        
        # Verifier Head processing (on intermediate state)
        if self.enable_verifier:
            verification_signals, new_verifier_state, verification_penalties = self.verifier_head(
                x_norm2, verifier_state
            )
            aux_info["verification_signals"] = verification_signals
            aux_info["verifier_state"] = new_verifier_state
            aux_info["verification_penalties"] = verification_penalties
            aux_info["verification_loss"] = self.verifier_head.compute_verification_loss(verification_penalties)
        
        # Feed-forward network
        ffn_output = self.ffn(x_norm2)
        
        # Final residual connection
        if self.residual_connection == "dense":
            output = weights[0] * residual_input + weights[1] * attended + weights[2] * ffn_output
        else:
            output = x + self.dropout(ffn_output)
        
        return output, aux_info


class SPaRKTransformer(nn.Module):
    """
    Complete SPaR-K Transformer model with multiple SPaR-K blocks
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = None,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
        # SPaR-K specific params
        **spark_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # SPaR-K Transformer blocks
        self.layers = nn.ModuleList([
            SPaRKTransformerBlock(
                d_model=d_model,
                n_heads=n_heads, 
                d_ff=d_ff,
                dropout=dropout,
                **spark_kwargs
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output head
        self.ln_final = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        self.output_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following GPT-style initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        verifier_states: Optional[list] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len) 
            verifier_states: List of verifier states for each layer
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            aux_info: Auxiliary information from all layers
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Initialize verifier states if not provided
        if verifier_states is None:
            verifier_states = [None] * self.n_layers
        
        # Process through SPaR-K layers
        all_aux_info = []
        new_verifier_states = []
        
        for i, layer in enumerate(self.layers):
            x, aux_info = layer(x, attention_mask, verifier_states[i])
            all_aux_info.append(aux_info)
            
            if "verifier_state" in aux_info:
                new_verifier_states.append(aux_info["verifier_state"])
            else:
                new_verifier_states.append(None)
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.output_head(x)
        
        # Aggregate auxiliary information
        aggregated_aux = {
            "layer_aux_info": all_aux_info,
            "verifier_states": new_verifier_states,
            "total_verification_loss": sum(
                aux.get("verification_loss", 0.0) for aux in all_aux_info
            ),
            "total_separation_loss": sum(
                aux.get("spd_separation_loss", 0.0) for aux in all_aux_info
            )
        }
        
        return logits, aggregated_aux
    
    def compute_total_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_info: Dict,
        lambda_verifier: float = 0.1,
        lambda_separation: float = 0.05
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss including task loss and auxiliary losses
        
        Args:
            logits: Model output logits
            targets: Target token IDs
            aux_info: Auxiliary information from forward pass
            lambda_verifier: Weight for verification loss
            lambda_separation: Weight for separation loss
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        # Task loss (standard language modeling)
        task_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        
        # Verification loss
        verification_loss = aux_info.get("total_verification_loss", 0.0)
        if isinstance(verification_loss, (int, float)):
            verification_loss = torch.tensor(verification_loss, device=logits.device)
        
        # Separation loss  
        separation_loss = aux_info.get("total_separation_loss", 0.0)
        if isinstance(separation_loss, (int, float)):
            separation_loss = torch.tensor(separation_loss, device=logits.device)
        
        # Total loss
        total_loss = task_loss + lambda_verifier * verification_loss + lambda_separation * separation_loss
        
        loss_components = {
            "task_loss": task_loss,
            "verification_loss": verification_loss, 
            "separation_loss": separation_loss,
            "total_loss": total_loss
        }
        
        return total_loss, loss_components