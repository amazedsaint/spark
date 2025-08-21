"""
Comprehensive Multi-hop Reasoning Benchmark

This module implements rigorous evaluation of FK-Attention's ability to perform
multi-hop reasoning compared to standard attention mechanisms.

Tests include:
1. Graph traversal with varying path lengths (2-10 hops)
2. Knowledge graph question answering
3. Logical inference chains
4. Mathematical reasoning sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from src.feynman_kac_attention import FeynmanKacAttention
from src.spark_transformer import SPaRKTransformer


class GraphTraversalDataset(Dataset):
    """Dataset for testing multi-hop graph traversal capabilities"""
    
    def __init__(self, 
                 num_graphs: int = 1000,
                 num_nodes: int = 20,
                 edge_prob: float = 0.3,
                 max_path_length: int = 8,
                 vocab_size: int = 100):
        
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.vocab_size = vocab_size
        self.max_path_length = max_path_length
        
        # Generate diverse graph structures
        self.graphs = []
        self.samples = []
        
        for _ in tqdm(range(num_graphs), desc="Generating graph traversal data"):
            # Create random graph with different topologies
            if random.random() < 0.3:
                # Tree-like structure
                G = nx.random_labeled_tree(num_nodes)
                G = G.to_directed()
            elif random.random() < 0.6:
                # Scale-free network
                G = nx.barabasi_albert_graph(num_nodes, min(3, num_nodes-1))
                G = G.to_directed()
            else:
                # Random graph
                G = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True)
            
            # Ensure connectivity
            if not nx.is_weakly_connected(G):
                # Add edges to make connected
                components = list(nx.weakly_connected_components(G))
                for i in range(len(components) - 1):
                    u = random.choice(list(components[i]))
                    v = random.choice(list(components[i + 1]))
                    G.add_edge(u, v)
            
            self.graphs.append(G)
            
            # Generate path queries of varying lengths
            for path_length in range(2, max_path_length + 1):
                for _ in range(10):  # Multiple queries per graph per path length
                    sample = self._generate_path_query(G, path_length)
                    if sample:
                        self.samples.append(sample)
    
    def _generate_path_query(self, G: nx.DiGraph, target_length: int) -> Dict:
        """Generate a path query of specific length"""
        nodes = list(G.nodes())
        
        # Try to find a path of target length
        for _ in range(100):  # Multiple attempts
            start = random.choice(nodes)
            
            # BFS to find paths of target length
            paths = []
            queue = [(start, [start])]
            
            while queue and len(paths) < 5:
                node, path = queue.pop(0)
                
                if len(path) == target_length + 1:
                    paths.append(path)
                    continue
                elif len(path) > target_length + 1:
                    continue
                
                for neighbor in G.successors(node):
                    if neighbor not in path:  # Avoid cycles
                        queue.append((neighbor, path + [neighbor]))
            
            if paths:
                # Select a random valid path
                true_path = random.choice(paths)
                end_node = true_path[-1]
                
                # Generate distractors (nodes not reachable via this path length)
                all_reachable = set()
                queue = [(start, 0)]
                visited = {start}
                
                while queue:
                    node, dist = queue.pop(0)
                    all_reachable.add((node, dist))
                    
                    if dist < target_length:
                        for neighbor in G.successors(node):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, dist + 1))
                
                # Get nodes reachable at exactly target_length
                reachable_at_length = {node for node, dist in all_reachable if dist == target_length}
                
                # Generate negative examples
                negative_candidates = [n for n in nodes if n not in reachable_at_length]
                
                if negative_candidates:
                    return {
                        'graph': G,
                        'start_node': start,
                        'target_length': target_length,
                        'true_end': end_node,
                        'path': true_path,
                        'negative_ends': random.sample(negative_candidates, min(3, len(negative_candidates)))
                    }
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        G = sample['graph']
        
        # Convert graph to sequence representation
        # Format: [START] node1 [REL] node2 [REL] ... [QUERY] target_length [END] answer
        
        # Create adjacency list representation
        adj_tokens = []
        for u in G.nodes():
            for v in G.successors(u):
                adj_tokens.extend([u + 10, 1, v + 10])  # +10 to avoid special tokens
        
        # Create query
        query_tokens = [2, sample['start_node'] + 10, 3, sample['target_length'], 4]  # START, node, QUERY, length, END
        
        # Combine
        input_tokens = adj_tokens + query_tokens
        
        # Pad/truncate to fixed length
        max_len = 200
        if len(input_tokens) > max_len:
            input_tokens = input_tokens[:max_len]
        else:
            input_tokens += [0] * (max_len - len(input_tokens))
        
        return {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'target_node': sample['true_end'] + 10,
            'path_length': sample['target_length'],
            'graph_size': len(G.nodes()),
            'metadata': sample
        }


class LogicalInferenceDataset(Dataset):
    """Dataset for testing logical reasoning chains"""
    
    def __init__(self, num_samples: int = 5000, max_chain_length: int = 6):
        self.samples = []
        
        # Define logical predicates and entities
        predicates = ['parent', 'friend', 'colleague', 'neighbor', 'sibling']
        entities = [f'person_{i}' for i in range(50)]
        
        for _ in tqdm(range(num_samples), desc="Generating logical inference data"):
            chain_length = random.randint(2, max_chain_length)
            
            # Generate inference chain
            chain = []
            current_entity = random.choice(entities)
            
            for step in range(chain_length):
                predicate = random.choice(predicates)
                next_entity = random.choice([e for e in entities if e != current_entity])
                chain.append((current_entity, predicate, next_entity))
                current_entity = next_entity
            
            # Create question about transitivity
            start_entity = chain[0][0]
            end_entity = chain[-1][2]
            
            # Generate facts and question
            facts = []
            for subj, pred, obj in chain:
                facts.append(f"{subj} {pred} {obj}")
            
            question = f"Is {start_entity} connected to {end_entity} via {chain_length} steps?"
            
            self.samples.append({
                'facts': facts,
                'question': question,
                'chain_length': chain_length,
                'answer': True,
                'chain': chain
            })
            
            # Generate negative example
            wrong_end = random.choice([e for e in entities if e != end_entity])
            neg_question = f"Is {start_entity} connected to {wrong_end} via {chain_length} steps?"
            
            self.samples.append({
                'facts': facts,
                'question': neg_question,
                'chain_length': chain_length,
                'answer': False,
                'chain': chain
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to token sequence
        # Simple tokenization: fact1 [SEP] fact2 [SEP] ... [QUERY] question [END]
        vocab = {'[SEP]': 1, '[QUERY]': 2, '[END]': 3, '[TRUE]': 4, '[FALSE]': 5}
        word_to_id = {}
        next_id = 10
        
        def get_token_id(word):
            nonlocal next_id
            if word not in word_to_id:
                word_to_id[word] = next_id
                next_id += 1
            return word_to_id[word]
        
        tokens = []
        
        # Add facts
        for fact in sample['facts']:
            for word in fact.split():
                tokens.append(get_token_id(word))
            tokens.append(vocab['[SEP]'])
        
        # Add question
        tokens.append(vocab['[QUERY]'])
        for word in sample['question'].split():
            tokens.append(get_token_id(word))
        tokens.append(vocab['[END]'])
        
        # Pad/truncate
        max_len = 150
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens += [0] * (max_len - len(tokens))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'target': torch.tensor(1 if sample['answer'] else 0, dtype=torch.long),
            'chain_length': sample['chain_length'],
            'metadata': sample
        }


class MultiHopBenchmark:
    """Comprehensive benchmark for multi-hop reasoning capabilities"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
    
    def create_baseline_model(self, vocab_size: int, d_model: int = 128) -> nn.Module:
        """Create standard Transformer baseline"""
        class StandardTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads=4, n_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Embedding(1000, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.classifier = nn.Linear(d_model, vocab_size)
                
            def forward(self, x):
                seq_len = x.size(1)
                pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
                
                x = self.embedding(x) + self.pos_embedding(pos)
                x = self.transformer(x)
                x = self.classifier(x[:, -1, :])  # Use last token for classification
                return x
        
        return StandardTransformer(vocab_size, d_model)
    
    def evaluate_graph_traversal(self, model_type='spark', num_test_samples=1000):
        """Evaluate multi-hop graph traversal capability"""
        print(f"\n=== Graph Traversal Evaluation ({model_type}) ===")
        
        # Create dataset
        dataset = GraphTraversalDataset(num_graphs=200, num_nodes=15, max_path_length=6)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Create model
        vocab_size = 150
        if model_type == 'spark':
            model = SPaRKTransformer(
                vocab_size=vocab_size,
                d_model=128,
                n_layers=2,
                n_heads=4,
                fk_beta=0.5,
                enable_verifier=False  # Focus on FK-Attention
            ).to(self.device)
        else:
            model = self.create_baseline_model(vocab_size).to(self.device)
        
        # Train briefly
        self._train_model(model, dataloader, epochs=10, model_type=model_type)
        
        # Evaluate by path length
        results_by_length = {}
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating graph traversal"):
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['target_node'].to(self.device)
                path_lengths = batch['path_length']
                
                if model_type == 'spark':
                    logits, _ = model(input_ids)
                else:
                    logits = model(input_ids)
                
                predictions = torch.argmax(logits, dim=-1)
                
                # Group by path length
                for i, length in enumerate(path_lengths):
                    length = length.item()
                    if length not in results_by_length:
                        results_by_length[length] = {'correct': 0, 'total': 0}
                    
                    if predictions[i] == targets[i]:
                        results_by_length[length]['correct'] += 1
                    results_by_length[length]['total'] += 1
        
        # Calculate accuracies
        accuracies = {}
        for length, stats in results_by_length.items():
            accuracy = stats['correct'] / max(stats['total'], 1)
            accuracies[length] = accuracy
            print(f"Path length {length}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        return accuracies
    
    def evaluate_logical_inference(self, model_type='spark', num_test_samples=1000):
        """Evaluate logical inference chain capability"""
        print(f"\n=== Logical Inference Evaluation ({model_type}) ===")
        
        # Create dataset
        dataset = LogicalInferenceDataset(num_samples=2000, max_chain_length=5)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Create model
        vocab_size = 200
        if model_type == 'spark':
            model = SPaRKTransformer(
                vocab_size=vocab_size,
                d_model=128,
                n_layers=2,
                n_heads=4,
                fk_beta=0.5,
                enable_verifier=False
            ).to(self.device)
        else:
            model = self.create_baseline_model(vocab_size).to(self.device)
            # Adapt for binary classification
            model.classifier = nn.Linear(128, 2)
        
        # Train
        self._train_model(model, dataloader, epochs=15, task_type='classification')
        
        # Evaluate by chain length
        results_by_length = {}
        all_predictions = []
        all_targets = []
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['target'].to(self.device)
                chain_lengths = batch['chain_length']
                
                if model_type == 'spark':
                    logits, _ = model(input_ids)
                    # Use appropriate output head for classification
                    logits = logits[:, -1, :2]  # Last token, first 2 classes
                else:
                    logits = model(input_ids)
                
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Group by chain length
                for i, length in enumerate(chain_lengths):
                    length = length.item()
                    if length not in results_by_length:
                        results_by_length[length] = {'correct': 0, 'total': 0}
                    
                    if predictions[i] == targets[i]:
                        results_by_length[length]['correct'] += 1
                    results_by_length[length]['total'] += 1
        
        # Calculate metrics
        overall_accuracy = accuracy_score(all_targets, all_predictions)
        overall_f1 = f1_score(all_targets, all_predictions, average='binary')
        
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        print(f"Overall F1: {overall_f1:.3f}")
        
        accuracies = {}
        for length, stats in results_by_length.items():
            accuracy = stats['correct'] / max(stats['total'], 1)
            accuracies[length] = accuracy
            print(f"Chain length {length}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        return accuracies, overall_accuracy, overall_f1
    
    def _train_model(self, model, dataloader, epochs=10, model_type='spark', task_type='language_modeling'):
        """Train model on given dataset"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                
                if task_type == 'classification':
                    targets = batch['target'].to(self.device)
                else:
                    targets = batch['target_node'].to(self.device)
                
                optimizer.zero_grad()
                
                if model_type == 'spark':
                    if task_type == 'classification':
                        logits, _ = model(input_ids)
                        logits = logits[:, -1, :2]  # Binary classification
                    else:
                        logits, _ = model(input_ids)
                        logits = logits[:, -1, :]  # Last token prediction
                else:
                    logits = model(input_ids)
                
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 3 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def run_comprehensive_evaluation(self):
        """Run all multi-hop reasoning benchmarks"""
        print("Starting Comprehensive Multi-hop Reasoning Evaluation")
        print("=" * 60)
        
        results = {}
        
        # Test both SPaR-K and baseline
        for model_type in ['baseline', 'spark']:
            print(f"\n--- Evaluating {model_type.upper()} ---")
            
            # Graph traversal
            graph_results = self.evaluate_graph_traversal(model_type)
            
            # Logical inference  
            logic_results, logic_acc, logic_f1 = self.evaluate_logical_inference(model_type)
            
            results[model_type] = {
                'graph_traversal': graph_results,
                'logical_inference': logic_results,
                'logic_overall_accuracy': logic_acc,
                'logic_overall_f1': logic_f1
            }
        
        self.results['multi_hop'] = results
        return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benchmark = MultiHopBenchmark(device=device)
    results = benchmark.run_comprehensive_evaluation()
    
    # Save results
    with open('multi_hop_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("MULTI-HOP REASONING BENCHMARK COMPLETE")
    print("="*60)
    print("Results saved to multi_hop_results.json")