"""
Structure-Pseudo-randomness Decomposition Router

This module implements the SPD Router component of the SPaR-K architecture.
Based on Tao's structure vs randomness principle, it decomposes inputs into
structured and pseudo-random components, routing each through appropriate
computational pathways.

Key Innovation: X = X_struct + X_pseudo decomposition with specialized processing

Author: Anoop Madhusudanan (amazedsaint@gmail.com)
Part of: SPaR-K Architecture
Repository: https://github.com/amazedsaint/spark

References:
- Tao, T. (2012). Topics in random matrix theory. (Structure vs randomness principle)
- Ahmed, N. et al. (1974). Discrete cosine transform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SPDRouter(nn.Module):
    """
    Structure-Pseudo-randomness Decomposition Router
    
    Implements Tao's structure vs randomness principle by decomposing
    input into structured and pseudo-random components, routing each
    through appropriate computational pathways.
    """
    
    def __init__(
        self,
        d_model: int,
        structure_dim: int = None,
        pseudo_dim: int = None,
        gate_temperature: float = 1.0,
        separation_strength: float = 0.1,
        use_learned_basis: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.structure_dim = structure_dim or d_model // 2
        self.pseudo_dim = pseudo_dim or d_model // 2
        self.gate_temperature = gate_temperature
        self.separation_strength = separation_strength
        self.use_learned_basis = use_learned_basis
        
        # Decomposition networks
        self.structure_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.structure_dim),
            nn.Sigmoid()
        )
        
        self.pseudo_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2), 
            nn.ReLU(),
            nn.Linear(d_model // 2, self.pseudo_dim),
            nn.Sigmoid()
        )
        
        # Structure processing pathway - stable, low-complexity operators
        if self.use_learned_basis:
            self.structure_basis = nn.Parameter(
                torch.randn(d_model, self.structure_dim) / math.sqrt(d_model)
            )
            self.structure_coeffs = nn.Linear(self.structure_dim, self.structure_dim)
        else:
            # Use predefined structured operators (e.g., circulant, Toeplitz)
            self.register_buffer(
                "structure_basis", 
                self._create_structured_basis(d_model, self.structure_dim)
            )
        
        self.structure_processor = nn.Sequential(
            nn.Linear(self.structure_dim, self.structure_dim),
            nn.LayerNorm(self.structure_dim),
            nn.ReLU(),
            nn.Linear(self.structure_dim, self.structure_dim)
        )
        
        # Pseudo-random extraction and processing
        self.pseudo_proj = nn.Linear(d_model, self.pseudo_dim)
        self.pseudo_processor = nn.Sequential(
            nn.Linear(self.pseudo_dim, self.pseudo_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.pseudo_dim * 2, self.pseudo_dim),
            nn.LayerNorm(self.pseudo_dim)
        )
        
        # Reconstruction
        self.reconstruct = nn.Linear(self.structure_dim + self.pseudo_dim, d_model)
        
        # Separation penalty computation
        self.separation_predictor = nn.Sequential(
            nn.Linear(self.structure_dim, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            output: Reconstructed output (batch_size, seq_len, d_model)
            x_struct: Structured component (batch_size, seq_len, structure_dim)  
            x_pseudo: Pseudo-random component (batch_size, seq_len, pseudo_dim)
            separation_loss: Mutual information penalty for disentanglement
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute gating probabilities
        struct_gate = self.structure_gate(x)  # (B, L, structure_dim)
        pseudo_gate = self.pseudo_gate(x)     # (B, L, pseudo_dim)
        
        # Soft decomposition with temperature control
        struct_gate = F.gumbel_softmax(
            torch.log(struct_gate + 1e-8), 
            tau=self.gate_temperature, 
            hard=False
        )
        pseudo_gate = F.gumbel_softmax(
            torch.log(pseudo_gate + 1e-8),
            tau=self.gate_temperature,
            hard=False
        )
        
        # Extract structured and pseudo-random components
        x_struct = self._extract_structured(x, struct_gate)
        x_pseudo = self._extract_pseudorandom(x, pseudo_gate)
        
        # Process through respective pathways
        x_struct_processed = self._process_structured(x_struct)
        x_pseudo_processed = self._process_pseudorandom(x_pseudo)
        
        # Reconstruct output
        combined = torch.cat([x_struct_processed, x_pseudo_processed], dim=-1)
        output = self.reconstruct(combined)
        output = self.dropout(output)
        
        # Compute separation penalty
        separation_loss = self._compute_separation_loss(x_struct, x_pseudo, x)
        
        return output, x_struct, x_pseudo, separation_loss
    
    def _extract_structured(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Extract structured component using learned/fixed basis"""
        # Project input through structure-sensitive transformation
        if self.use_learned_basis:
            # Learnable basis extraction
            projected = torch.matmul(x, self.structure_basis)
        else:
            # Fixed structured basis (e.g., DCT, DFT-like)
            projected = self._structured_transform(x)
            
        return projected * gate
    
    def _extract_pseudorandom(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Extract pseudo-random component"""
        # Simple linear projection for pseudo-random component
        projected = self.pseudo_proj(x)
        return projected * gate
    
    def _process_structured(self, x_struct: torch.Tensor) -> torch.Tensor:
        """Process structured component through stable operators"""
        if self.use_learned_basis:
            # Apply learned structured transformation
            transformed = self.structure_coeffs(x_struct)
            # Skip the basis multiplication for now - just use the coefficients output
        else:
            transformed = x_struct
            
        return self.structure_processor(transformed)
    
    def _process_pseudorandom(self, x_pseudo: torch.Tensor) -> torch.Tensor:
        """Process pseudo-random component through high-capacity pathway"""
        return self.pseudo_processor(x_pseudo)
    
    def _compute_separation_loss(
        self, 
        x_struct: torch.Tensor, 
        x_pseudo: torch.Tensor, 
        x_orig: torch.Tensor
    ) -> float:
        """
        Compute mutual information penalty to encourage disentanglement
        Uses a predictor that should succeed from structured component but fail from pseudo
        """
        # Predict original input from structured component
        pred_from_struct = self.separation_predictor(x_struct)
        struct_loss = F.mse_loss(pred_from_struct, x_orig)
        
        # Encourage pseudo component to be unpredictive of structure
        # (implicitly through the structured predictor success)
        pseudo_entropy = -torch.mean(torch.sum(torch.abs(x_pseudo) * torch.log(torch.abs(x_pseudo) + 1e-8), dim=-1))
        
        # Combined separation objective
        separation_loss = struct_loss - self.separation_strength * pseudo_entropy
        
        # Handle potential NaN values
        if torch.isnan(separation_loss) or torch.isinf(separation_loss):
            separation_loss = torch.tensor(0.0, device=separation_loss.device)
        
        return separation_loss.item()
    
    def _create_structured_basis(self, input_dim: int, output_dim: int) -> torch.Tensor:
        """Create a structured basis (e.g., DCT, circulant)"""
        # Create DCT-like basis
        basis = torch.zeros(input_dim, output_dim)
        for i in range(input_dim):
            for j in range(output_dim):
                basis[i, j] = math.cos(math.pi * i * (j + 0.5) / output_dim)
        
        # Normalize
        basis = basis / (torch.norm(basis, dim=0, keepdim=True) + 1e-8)
        return basis
    
    def _structured_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply structured transformation (DCT, circulant convolution, etc.)"""
        # Simple implementation: use the structured basis for transformation
        return torch.matmul(x, self.structure_basis)


class AdaptiveSPDRouter(SPDRouter):
    """
    Adaptive version that learns to adjust structure/pseudo-randomness 
    decomposition based on input statistics
    """
    
    def __init__(self, d_model: int, **kwargs):
        super().__init__(d_model, **kwargs)
        
        # Adaptive gating network
        self.adaptive_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(), 
            nn.Linear(d_model // 4, 2),  # [structure_weight, pseudo_weight]
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        # Compute adaptive weights based on input characteristics
        input_stats = torch.mean(x, dim=1)  # (B, d_model)
        adaptive_weights = self.adaptive_controller(input_stats)  # (B, 2)
        
        # Scale gates by adaptive weights
        output, x_struct, x_pseudo, sep_loss = super().forward(x)
        
        # Apply adaptive weighting
        struct_weight = adaptive_weights[:, 0:1].unsqueeze(1)  # (B, 1, 1)  
        pseudo_weight = adaptive_weights[:, 1:2].unsqueeze(1)   # (B, 1, 1)
        
        x_struct = x_struct * struct_weight
        x_pseudo = x_pseudo * pseudo_weight
        
        # Recompute output with adaptive weighting
        combined = torch.cat([x_struct, x_pseudo], dim=-1)
        output = self.reconstruct(combined)
        
        return output, x_struct, x_pseudo, sep_loss