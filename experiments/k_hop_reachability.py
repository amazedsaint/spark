import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
import networkx as nx
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feynman_kac_attention import FeynmanKacAttention


class GraphReachabilityDataset:
    """Dataset for K-hop reachability tasks"""
    
    def __init__(self, num_graphs: int = 1000, num_nodes: int = 20, edge_prob: float = 0.3, max_k: int = 5):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes  
        self.edge_prob = edge_prob
        self.max_k = max_k
        
        self.graphs = []
        self.node_pairs = []
        self.k_hop_labels = {}
        
        self._generate_data()
    
    def _generate_data(self):
        """Generate random graphs and compute k-hop reachability"""
        
        for _ in tqdm(range(self.num_graphs), desc="Generating graphs"):
            # Create random graph
            G = nx.erdos_renyi_graph(self.num_nodes, self.edge_prob, directed=True)
            adj_matrix = nx.adjacency_matrix(G).todense().astype(np.float32)
            
            self.graphs.append(adj_matrix)
            
            # Generate node pairs and compute k-hop reachability
            pairs = []
            k_hop_dict = {}
            
            for source in range(self.num_nodes):
                for target in range(self.num_nodes):
                    if source != target:
                        pairs.append((source, target))
                        
                        # Compute k-hop reachability for different k values
                        k_hop_dict[(source, target)] = {}
                        
                        for k in range(1, self.max_k + 1):
                            try:
                                path = nx.shortest_path(G, source, target)
                                path_length = len(path) - 1
                                k_hop_dict[(source, target)][k] = 1 if path_length <= k else 0
                            except nx.NetworkXNoPath:
                                k_hop_dict[(source, target)][k] = 0
            
            self.node_pairs.append(pairs)
            self.k_hop_labels[len(self.graphs) - 1] = k_hop_dict
    
    def get_batch(self, batch_size: int, k: int = 3):
        """Get batch of graph data for k-hop reachability"""
        
        batch_graphs = []
        batch_queries = []
        batch_labels = []
        
        for _ in range(batch_size):
            # Sample random graph
            graph_idx = np.random.randint(0, len(self.graphs))
            adj_matrix = self.graphs[graph_idx]
            
            # Sample random node pair
            pair_idx = np.random.randint(0, len(self.node_pairs[graph_idx]))
            source, target = self.node_pairs[graph_idx][pair_idx]
            
            # Get k-hop label
            label = self.k_hop_labels[graph_idx][(source, target)][k]
            
            batch_graphs.append(adj_matrix)
            batch_queries.append([source, target])
            batch_labels.append(label)
        
        return (
            torch.tensor(np.stack(batch_graphs), dtype=torch.float32),
            torch.tensor(batch_queries, dtype=torch.long),
            torch.tensor(batch_labels, dtype=torch.float32)
        )


class VanillaAttention(nn.Module):
    """Standard vanilla attention for comparison"""
    
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = 1.0 / np.sqrt(self.d_head)
    
    def forward(self, query, key, value, adjacency_mask=None):
        batch_size, seq_len, _ = query.shape
        
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Standard attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if adjacency_mask is not None:
            attn_weights = attn_weights + adjacency_mask.unsqueeze(1).unsqueeze(1) * -1e9
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output)


class ReachabilityModel(nn.Module):
    """Model for testing k-hop reachability performance"""
    
    def __init__(self, num_nodes: int, d_model: int = 64, attention_type: str = "vanilla"):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.attention_type = attention_type
        
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        
        # Attention mechanism
        if attention_type == "fk":
            self.attention = FeynmanKacAttention(
                d_model=d_model, 
                n_heads=1, 
                beta=0.7,
                approximation_method="krylov",
                max_path_length=8
            )
        else:  # vanilla
            self.attention = VanillaAttention(d_model, n_heads=1)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, adj_matrix, queries):
        """
        Args:
            adj_matrix: (batch_size, num_nodes, num_nodes)
            queries: (batch_size, 2) - [source, target] pairs
        """
        batch_size = adj_matrix.shape[0]
        
        # Create node embeddings
        node_indices = torch.arange(self.num_nodes, device=adj_matrix.device)
        node_indices = node_indices.unsqueeze(0).expand(batch_size, -1)
        node_embeds = self.node_embedding(node_indices)  # (batch_size, num_nodes, d_model)
        
        # Create adjacency mask (prevent attention to non-connected nodes)
        adj_mask = (adj_matrix == 0).float() * -1e9
        
        # Apply attention
        if self.attention_type == "fk":
            # FK attention uses adjacency matrix directly
            attended = self.attention(node_embeds, node_embeds, node_embeds, adj_mask)
        else:
            attended = self.attention(node_embeds, node_embeds, node_embeds, adj_mask)
        
        # Extract source and target embeddings
        source_indices = queries[:, 0]  # (batch_size,)
        target_indices = queries[:, 1]  # (batch_size,)
        
        source_embeds = attended[torch.arange(batch_size), source_indices]  # (batch_size, d_model)
        target_embeds = attended[torch.arange(batch_size), target_indices]  # (batch_size, d_model)
        
        # Predict reachability
        combined = torch.cat([source_embeds, target_embeds], dim=-1)
        output = self.output_proj(combined).squeeze(-1)
        
        return output


def evaluate_k_hop_performance():
    """Evaluate FK-Attention vs Vanilla Attention on K-hop reachability"""
    
    print("=== K-Hop Reachability Experiment ===")
    
    # Parameters
    num_nodes = 20
    d_model = 64
    batch_size = 32
    num_batches = 200
    k_values = [1, 2, 3, 4, 5]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    print("Generating graph dataset...")
    dataset = GraphReachabilityDataset(num_graphs=1000, num_nodes=num_nodes, max_k=max(k_values))
    
    # Test both attention mechanisms
    attention_types = ["vanilla", "fk"]
    results = {}
    
    for attention_type in attention_types:
        print(f"\nTesting {attention_type.upper()} attention...")
        
        # Create model
        model = ReachabilityModel(num_nodes, d_model, attention_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Results for this attention type
        results[attention_type] = {}
        
        for k in k_values:
            print(f"  K={k} hop reachability:")
            
            # Training phase
            model.train()
            for batch_idx in tqdm(range(num_batches), desc=f"Training K={k}"):
                adj_matrices, queries, labels = dataset.get_batch(batch_size, k)
                adj_matrices, queries, labels = adj_matrices.to(device), queries.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(adj_matrices, queries)
                loss = F.binary_cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Evaluation phase
            model.eval()
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for _ in tqdm(range(50), desc=f"Evaluating K={k}"):
                    adj_matrices, queries, labels = dataset.get_batch(batch_size, k)
                    adj_matrices, queries, labels = adj_matrices.to(device), queries.to(device), labels.to(device)
                    
                    outputs = model(adj_matrices, queries)
                    all_predictions.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Compute metrics
            auc = roc_auc_score(all_labels, all_predictions)
            results[attention_type][k] = auc
            print(f"    AUC: {auc:.3f}")
    
    # Print comparison
    print("\n=== RESULTS COMPARISON ===")
    print("K-hop | Vanilla AUC | FK-Attention AUC | Improvement")
    print("-" * 55)
    
    for k in k_values:
        vanilla_auc = results["vanilla"][k]
        fk_auc = results["fk"][k]
        improvement = fk_auc - vanilla_auc
        print(f"  {k}   |    {vanilla_auc:.3f}    |      {fk_auc:.3f}      |   +{improvement:.3f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    k_range = list(k_values)
    vanilla_aucs = [results["vanilla"][k] for k in k_values]
    fk_aucs = [results["fk"][k] for k in k_values]
    
    plt.plot(k_range, vanilla_aucs, 'o-', label='Vanilla Attention', linewidth=2, markersize=8)
    plt.plot(k_range, fk_aucs, 's-', label='Feynman-Kac Attention', linewidth=2, markersize=8)
    
    plt.xlabel('K (hop distance)', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('K-Hop Reachability Performance Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Add improvement annotations
    for k, vanilla_auc, fk_auc in zip(k_values, vanilla_aucs, fk_aucs):
        if fk_auc > vanilla_auc:
            plt.annotate(f'+{fk_auc - vanilla_auc:.3f}', 
                        xy=(k, fk_auc), xytext=(k, fk_auc + 0.05),
                        ha='center', fontsize=10, color='green',
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/claude-user/projects/spark/k_hop_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    results = evaluate_k_hop_performance()
    print("\nExperiment completed! Results saved to k_hop_results.png")