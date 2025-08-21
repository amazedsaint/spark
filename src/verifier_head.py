"""
Verifier Head: Stack-based algorithmic verification

This module implements the Verifier Head component of the SPaR-K architecture.
It maintains algorithmic invariants through differentiable stack operations
and provides training penalties for trace failures.

Key Innovation: Differentiable stack with verification signals for algorithmic priors

Author: Anoop Madhusudanan (amazedsaint@gmail.com)
Part of: SPaR-K Architecture
Repository: https://github.com/amazedsaint/spark

References:
- Graves, A. et al. (2014). Neural turing machines.
- Joulin, A. & Mikolov, T. (2015). Inferring algorithmic patterns with stack-augmented recurrent nets.
- Graves, A. et al. (2016). Hybrid computing using a neural network with dynamic external memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math


class DifferentiableStack(nn.Module):
    """
    Differentiable stack implementation for maintaining algorithmic invariants
    Supports push, pop, and no-op operations with soft attention-based addressing
    """
    
    def __init__(self, stack_size: int, element_dim: int):
        super().__init__()
        self.stack_size = stack_size
        self.element_dim = element_dim
        
        # Stack memory and pointer
        self.register_buffer("stack_memory", torch.zeros(stack_size, element_dim))
        self.register_buffer("stack_pointer", torch.tensor(0.0))
        
    def forward(
        self, 
        operation: torch.Tensor,  # [batch, 3] - [push_prob, pop_prob, noop_prob]
        element: torch.Tensor,    # [batch, element_dim] - element to push
        current_memory: torch.Tensor,  # [batch, stack_size, element_dim]
        current_pointer: torch.Tensor  # [batch, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply stack operation differentiably
        
        Returns:
            new_memory: Updated stack memory
            new_pointer: Updated stack pointer
            popped_element: Element that was popped (or zeros)
        """
        batch_size = operation.shape[0]
        
        push_prob = operation[:, 0:1]    # [batch, 1]
        pop_prob = operation[:, 1:2]     # [batch, 1] 
        noop_prob = operation[:, 2:3]    # [batch, 1]
        
        # Current pointer as continuous value
        ptr = current_pointer.clamp(0, self.stack_size - 1)
        
        # Push operation: write element at current pointer, increment pointer
        push_indices = torch.arange(self.stack_size, device=current_memory.device, dtype=torch.float32)
        push_indices = push_indices.unsqueeze(0).expand(batch_size, -1)  # [batch, stack_size]
        
        push_mask = torch.exp(-(push_indices - ptr) ** 2 / 2.0)  # Soft addressing
        push_mask = push_mask / (push_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        push_update = push_prob.unsqueeze(-1) * push_mask.unsqueeze(-1) * element.unsqueeze(1)
        
        # Pop operation: read element at pointer-1, decrement pointer  
        pop_indices = push_indices - 1
        pop_mask = torch.exp(-(pop_indices - ptr) ** 2 / 2.0)
        pop_mask = pop_mask / (pop_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        popped_element = torch.sum(
            pop_prob.unsqueeze(-1) * pop_mask.unsqueeze(-1) * current_memory, 
            dim=1
        )
        
        # Update memory
        memory_decay = 1.0 - push_prob.unsqueeze(-1) * push_mask.unsqueeze(-1)
        new_memory = current_memory * memory_decay + push_update
        
        # Update pointer
        pointer_delta = push_prob - pop_prob
        new_pointer = (current_pointer + pointer_delta).clamp(0, self.stack_size - 1)
        
        return new_memory, new_pointer, popped_element


class VerifierHead(nn.Module):
    """
    Verifier Head that maintains algorithmic invariants through stack-based verification
    Emits verification signals and computes training penalties for trace failures
    """
    
    def __init__(
        self,
        d_model: int,
        stack_size: int = 64,
        num_stacks: int = 1,
        verification_types: List[str] = None,
        penalty_strength: float = 1.0,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.stack_size = stack_size
        self.num_stacks = num_stacks
        self.verification_types = verification_types or ["balanced_parens", "sequence_length", "invariant_check"]
        self.penalty_strength = penalty_strength
        self.temperature = temperature
        
        # Stack operations controller
        self.stack_controller = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 3),  # [push, pop, noop]
                nn.Softmax(dim=-1)
            ) for _ in range(num_stacks)
        ])
        
        # Element encoders for different verification types
        self.element_encoders = nn.ModuleDict()
        for vtype in self.verification_types:
            self.element_encoders[vtype] = nn.Linear(d_model, d_model // 2)
        
        # Differentiable stacks
        self.stacks = nn.ModuleList([
            DifferentiableStack(stack_size, d_model // 2) 
            for _ in range(num_stacks)
        ])
        
        # Verification predictors
        self.verification_heads = nn.ModuleDict()
        for vtype in self.verification_types:
            self.verification_heads[vtype] = nn.Sequential(
                nn.Linear(d_model + d_model // 2, d_model // 4), 
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )
        
        # State tracking
        self.register_buffer("init_memory", torch.zeros(1, num_stacks, stack_size, d_model // 2))
        self.register_buffer("init_pointer", torch.zeros(1, num_stacks, 1))
        
    def forward(
        self, 
        hidden_states: torch.Tensor,  # [batch, seq_len, d_model]
        previous_state: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Args:
            hidden_states: Hidden states from transformer layers
            previous_state: Previous verifier state (memory, pointers)
            
        Returns:
            verification_signals: Verification outputs for each position
            current_state: Current verifier state
            penalties: Dictionary of verification penalties
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Initialize state if not provided
        if previous_state is None:
            stack_memories = self.init_memory.expand(batch_size, -1, -1, -1)
            stack_pointers = self.init_pointer.expand(batch_size, -1, -1)
        else:
            stack_memories = previous_state["memories"]
            stack_pointers = previous_state["pointers"]
        
        # Process each position in sequence
        verification_outputs = []
        penalties = {vtype: 0.0 for vtype in self.verification_types}
        
        for t in range(seq_len):
            current_hidden = hidden_states[:, t, :]  # [batch, d_model]
            
            # Update stacks based on current hidden state
            new_memories = []
            new_pointers = []
            popped_elements = []
            
            for stack_idx in range(self.num_stacks):
                # Decide stack operation
                operation_logits = self.stack_controller[stack_idx](current_hidden)
                operation_probs = F.gumbel_softmax(
                    torch.log(operation_logits + 1e-8), 
                    tau=self.temperature, 
                    hard=False
                )
                
                # Encode element to push (based on verification type)
                element_to_push = torch.zeros(batch_size, self.d_model // 2, device=hidden_states.device)
                for vtype in self.verification_types:
                    element_to_push += self.element_encoders[vtype](current_hidden)
                element_to_push /= len(self.verification_types)
                
                # Apply stack operation
                new_mem, new_ptr, popped_elem = self.stacks[stack_idx](
                    operation_probs,
                    element_to_push,
                    stack_memories[:, stack_idx],
                    stack_pointers[:, stack_idx]
                )
                
                new_memories.append(new_mem)
                new_pointers.append(new_ptr)
                popped_elements.append(popped_elem)
                
                # Compute stack penalties
                penalties = self._update_stack_penalties(
                    penalties, operation_probs, new_ptr, stack_idx
                )
            
            # Stack new memory/pointer states
            stack_memories = torch.stack(new_memories, dim=1)
            stack_pointers = torch.stack(new_pointers, dim=1)
            
            # Generate verification signals
            verification_output = self._generate_verification_signals(
                current_hidden, popped_elements, stack_memories, stack_pointers
            )
            verification_outputs.append(verification_output)
            
            # Update verification penalties
            penalties = self._update_verification_penalties(
                penalties, current_hidden, verification_output, popped_elements
            )
        
        # Stack verification outputs across sequence
        verification_signals = torch.stack(verification_outputs, dim=1)
        
        current_state = {
            "memories": stack_memories,
            "pointers": stack_pointers
        }
        
        return verification_signals, current_state, penalties
    
    def _update_stack_penalties(
        self, 
        penalties: Dict, 
        operation_probs: torch.Tensor, 
        pointer: torch.Tensor, 
        stack_idx: int
    ) -> Dict:
        """Update penalties for stack violations"""
        
        # Stack underflow penalty (trying to pop from empty stack)
        underflow_penalty = operation_probs[:, 1] * F.relu(-pointer.squeeze())
        penalties["stack_underflow"] = penalties.get("stack_underflow", 0.0) + torch.mean(underflow_penalty)
        
        # Stack overflow penalty (pointer exceeds stack size)  
        overflow_penalty = F.relu(pointer.squeeze() - self.stack_size + 1)
        penalties["stack_overflow"] = penalties.get("stack_overflow", 0.0) + torch.mean(overflow_penalty)
        
        return penalties
    
    def _generate_verification_signals(
        self,
        hidden: torch.Tensor,
        popped_elements: List[torch.Tensor], 
        memories: torch.Tensor,
        pointers: torch.Tensor
    ) -> torch.Tensor:
        """Generate verification signals based on stack state"""
        
        # Create stack summary with correct dimensions
        # memories: (batch, num_stacks, stack_size, element_dim)
        # pointers: (batch, num_stacks, 1)
        
        batch_size = hidden.shape[0]
        
        # Average over stack and memory dimensions
        memory_summary = torch.mean(memories, dim=(1, 2))  # (batch, element_dim)
        pointer_summary = torch.mean(pointers, dim=(1, 2))  # (batch,)
        
        # Ensure pointer summary has correct shape
        if pointer_summary.dim() == 1:
            pointer_summary = pointer_summary.unsqueeze(-1)  # (batch, 1)
        
        # Stack summary should match expected dimension
        expected_summary_dim = self.d_model // 2
        if memory_summary.shape[-1] != expected_summary_dim:
            # Project to correct dimension
            if not hasattr(self, 'memory_proj'):
                self.memory_proj = nn.Linear(memory_summary.shape[-1], expected_summary_dim).to(hidden.device)
            memory_summary = self.memory_proj(memory_summary)
        
        # Pad pointer summary to match if needed
        pointer_dim = expected_summary_dim - memory_summary.shape[-1]
        if pointer_dim > 0:
            pointer_padding = torch.zeros(batch_size, pointer_dim, device=hidden.device)
            stack_summary = torch.cat([memory_summary, pointer_summary, pointer_padding], dim=-1)
        else:
            stack_summary = memory_summary
        
        # Combine with hidden state
        combined_input = torch.cat([hidden, stack_summary], dim=-1)
        
        # Generate verification signals for each type
        verification_signals = []
        for vtype in self.verification_types:
            signal = self.verification_heads[vtype](combined_input)
            verification_signals.append(signal)
        
        return torch.cat(verification_signals, dim=-1)
    
    def _update_verification_penalties(
        self,
        penalties: Dict,
        hidden: torch.Tensor, 
        verification_output: torch.Tensor,
        popped_elements: List[torch.Tensor]
    ) -> Dict:
        """Update verification-specific penalties"""
        
        # Balanced parentheses penalty
        if "balanced_parens" in self.verification_types:
            paren_signal = verification_output[:, 0]
            # Penalty for unbalanced state (should be near 0.5 for balanced)
            balance_penalty = torch.mean((paren_signal - 0.5) ** 2)
            penalties["balanced_parens"] = penalties.get("balanced_parens", 0.0) + balance_penalty
        
        # Sequence length consistency
        if "sequence_length" in self.verification_types:
            length_signal = verification_output[:, 1] if verification_output.shape[1] > 1 else verification_output[:, 0]
            # Length should increase monotonically
            length_penalty = torch.mean(F.relu(-length_signal))  # Penalize decreasing length
            penalties["sequence_length"] = penalties.get("sequence_length", 0.0) + length_penalty
        
        # General invariant check
        if "invariant_check" in self.verification_types:
            invariant_signal = verification_output[:, -1]
            # Invariant should remain stable (close to previous values)
            stability_penalty = torch.var(invariant_signal)
            penalties["invariant_check"] = penalties.get("invariant_check", 0.0) + stability_penalty
        
        return penalties
    
    def compute_verification_loss(self, penalties: Dict) -> torch.Tensor:
        """Compute total verification loss from penalties"""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for penalty_name, penalty_value in penalties.items():
            if isinstance(penalty_value, torch.Tensor):
                total_loss += self.penalty_strength * penalty_value
            else:
                total_loss += self.penalty_strength * torch.tensor(penalty_value, device=total_loss.device)
        
        return total_loss
    
    def reset_state(self, batch_size: int) -> Dict:
        """Reset verifier state for new sequence"""
        return {
            "memories": self.init_memory.expand(batch_size, -1, -1, -1),
            "pointers": self.init_pointer.expand(batch_size, -1, -1)
        }