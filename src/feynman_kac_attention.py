"""
Feynman-Kac Attention: Path-integral formulation for multi-hop reasoning

This module implements the FK-Attention component of the SPaR-K architecture.
FK-Attention extends standard attention to capture multi-hop dependencies through
a resolvent formulation inspired by the Feynman-Kac formula for path integrals.

Key Innovation: (I - βA)^(-1)V formulation enables single-layer multi-hop reasoning

Author: Anoop Madhusudanan (amazedsaint@gmail.com)
Part of: SPaR-K Architecture
Repository: https://github.com/amazedsaint/spark

References:
- Kac, M. (1949). On distributions of certain Wiener functionals.
- Vaswani, A. et al. (2017). Attention is all you need.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FeynmanKacAttention(nn.Module):
    """
    Feynman-Kac Attention mechanism that computes path-integral attention
    using resolvent formulation: (I - βA)^(-1)V
    
    This enables multi-hop reasoning within a single attention layer by
    considering all possible paths between query and key positions.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        beta: float = 0.5,
        max_path_length: int = 10,
        approximation_method: str = "krylov",  # "krylov", "chebyshev", "exact"
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.beta = beta
        self.max_path_length = max_path_length
        self.approximation_method = approximation_method
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Stability parameters
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(self.d_head)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)  
            value: (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention weights (adjacency matrix A)
        A = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            A = A.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))
            
        A = F.softmax(A, dim=-1)
        A = self.dropout(A)
        
        # Apply Feynman-Kac resolvent: (I - βA)^(-1)V
        resolvent_output = self._compute_resolvent(A, V)
        
        # Reshape and project output
        output = resolvent_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.out_proj(output)
        
        return output
    
    def _compute_resolvent(self, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Compute (I - βA)^(-1)V using specified approximation method
        """
        if self.approximation_method == "exact":
            return self._exact_resolvent(A, V)
        elif self.approximation_method == "krylov":
            return self._krylov_resolvent(A, V)
        elif self.approximation_method == "chebyshev":
            return self._chebyshev_resolvent(A, V)
        else:
            raise ValueError(f"Unknown approximation method: {self.approximation_method}")
    
    def _exact_resolvent(self, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Exact resolvent computation using matrix inverse"""
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
        I = I.unsqueeze(0).unsqueeze(0).expand_as(A)
        
        # Ensure stability: β * ρ(A) < 1
        beta_stable = min(self.beta, 0.99 / (torch.max(torch.abs(A)).item() + 1e-8))
        
        resolvent_matrix = torch.inverse(I - beta_stable * A)
        return torch.matmul(resolvent_matrix, V)
    
    def _krylov_resolvent(self, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Krylov subspace approximation of (I - βA)^(-1)V
        Using power series expansion: Σ_{k=0}^∞ (βA)^k V
        """
        result = V.clone()
        power_term = V.clone()
        
        beta_stable = min(self.beta, 0.99)
        
        for k in range(1, self.max_path_length + 1):
            power_term = torch.matmul(beta_stable * A, power_term)
            result = result + power_term
            
            # Early stopping if contribution becomes negligible
            if torch.norm(power_term) < 1e-6 * torch.norm(result):
                break
                
        return result
    
    def _chebyshev_resolvent(self, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Chebyshev polynomial approximation of the resolvent
        More stable for larger beta values
        """
        # Map eigenvalues of A to [-1, 1] for Chebyshev stability
        spectral_norm = torch.norm(A, p=2, dim=(-2, -1), keepdim=True) + 1e-8
        A_normalized = A / spectral_norm
        
        beta_normalized = self.beta / spectral_norm.squeeze(-1).squeeze(-1)
        
        # Chebyshev coefficients for (1-βz)^(-1)
        T0 = V
        T1 = torch.matmul(beta_normalized.unsqueeze(-1).unsqueeze(-1) * A_normalized, V)
        
        result = T0 + T1
        
        for n in range(2, self.max_path_length + 1):
            T_new = 2 * torch.matmul(beta_normalized.unsqueeze(-1).unsqueeze(-1) * A_normalized, T1) - T0
            result = result + T_new
            T0, T1 = T1, T_new
            
        return result


class MultiHeadFeynmanKacAttention(nn.Module):
    """Wrapper for easier integration into existing Transformer architectures"""
    
    def __init__(self, d_model: int, n_heads: int = 8, **kwargs):
        super().__init__()
        self.attention = FeynmanKacAttention(d_model, n_heads, **kwargs)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.attention(x, x, x, mask)