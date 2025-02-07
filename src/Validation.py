
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from joblib import Parallel, delayed

class AttentionFlow:
    """
    This class is a re-implemmentation of attention weights explanation from:
    https://github.com/samiraabnar/attention_flow/blob/master/attention_graph_util.py
    """
    def __init__(self, attentions, top_k=0):
        num_layers, _, num_heads, seq_len, _ = attentions.shape
        
        self.__num_layers = num_layers
        self.__seq_len = seq_len
        self.__top_k = seq_len if top_k <= 0 or top_k > seq_len else top_k
        self.lambda_residual = 0.5
        self.attentions = attentions
        self.__avg_attention = attentions.mean(dim=2)

    def __normalize_raw_weights(self):
        """Applies residual connections and normalizes attention weights."""
        
        identity = torch.eye(self.__seq_len).unsqueeze(0)
        return self.lambda_residual * identity + (1 - self.lambda_residual) * self.__avg_attention

    def __build_attention_graphs(self):
        """Creates a directed graph with attention weights as edge capacities."""
        normalized_weights = self.__normalize_raw_weights()
        normalized_weights = normalized_weights.squeeze(1).detach().cpu().numpy()
        
        G = nx.DiGraph()

        # Create nodes
        for layer in range(self.__num_layers):
            for token_position in range(self.__seq_len):
                G.add_node((layer, token_position))

       # Add edges with attention weights as capacities
        for layer in range(self.__num_layers - 1):
            for i in range(self.__seq_len):
                for j in range(self.__seq_len):
                    #aggregate_capacity = np.sum([attentions[layer, 0, head, i, j].item() for head in range(num_heads)])
                    attn_weight = normalized_weights[layer, i, j]
                    if attn_weight > 0:
                        G.add_edge((layer, j), (layer + 1, i), capacity=attn_weight)

        return G

    def compute_rollout(self):
        """
        Computes the product of weights of edges between input nodes and the target token across layers.
        Based on: https://github.com/samiraabnar/attention_flow
        """        
        normalized_weights = self.__normalize_raw_weights()
        identity = torch.eye(self.__seq_len).unsqueeze(0)
        rollout = identity.clone()
        for layer in range(self.__num_layers):
            rollout = torch.matmul(normalized_weights[layer], rollout)
        print(rollout.data.squeeze(0)[:,28])
        return rollout.data.squeeze(0)

    def compute_attention_flow(self, target_token_position):
        """
        Computes attention flow from each input token to a specific target token.
        Based on Attention Flow: https://github.com/samiraabnar/attention_flow
        """
        
        G = self.__build_attention_graphs()
        input_nodes = [(0, i) for i in range(self.__seq_len)]  # Input tokens (layer 0)
        target_node_per_layer = [(i, target_token_position) for i in range(1,self.__num_layers)]  # The masked token at the final layer
        
        flow_matrix = np.zeros((self.__seq_len, self.__num_layers - 1))
        def compute_flow(inp_node):
            return [nx.maximum_flow_value(G, inp_node, target_node, flow_func=nx.algorithms.flow.boykov_kolmogorov)
                    for target_node in target_node_per_layer]

        # Parallelize computation
        flow_matrix = np.array(Parallel(n_jobs=-1)(delayed(compute_flow)(node) for node in input_nodes))
        
        # No parallelization solution commented for now
        """ # Compute max-flow for each input token to target token 
        for inp_idx, inp_node in enumerate(input_nodes):
            for layer_idx, target_node_layer in enumerate(target_node_per_layer):
                flow = nx.maximum_flow_value(G, inp_node, target_node_layer, flow_func=nx.algorithms.flow.boykov_kolmogorov)
                flow_matrix[inp_idx, layer_idx] = flow """
        
        return flow_matrix
        

    def __select_top_k_tokens(self, values, tokens):
        
        """Selects the top-k most important tokens based on the given values."""

        top_k_indices = np.argsort(-values)[:self.__top_k]
        top_k_values = values[top_k_indices]
        top_k_labels = [tokens[i] for i in top_k_indices] if tokens else top_k_indices

        # Sort in descending order for better visualization
        sorted_indices = np.argsort(-top_k_values)
        top_k_values = top_k_values[sorted_indices]
        top_k_labels = [top_k_labels[i] for i in sorted_indices]
        return top_k_labels, top_k_values

    def visualize_attention_flow(self, tokens, flow_matrix, target_token_position):
        """Visualizes the most important tokens contributing to a target token via attention flow."""

        token_importance = flow_matrix.sum(axis=1)
        top_k_labels, top_k_values = self.__select_top_k_tokens(token_importance, tokens)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_k_values, y=top_k_labels, palette="Blues_r")
        plt.xlabel("Cumulative Attention Flow")
        plt.ylabel("Tokens")
        plt.title(f"Top-{self.__top_k} Important Tokens (Flow) to Position {target_token_position} - Attention Flow")
        plt.show(block = False)
    
    def visualize_attention_rollout(self, tokens, rollout, target_token_position):
            """Visualizes top-k influential tokens using rollout attention."""

            print(rollout[:,28])
            top_k_labels, top_k_values = self.__select_top_k_tokens(rollout[:, target_token_position], tokens)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_k_values, y=top_k_labels, palette="Blues_r")
            plt.xlabel(f"Cumulative Attentions")
            plt.ylabel("Tokens")
            plt.title(f"Top-{self.__top_k} Important Tokens to Position {target_token_position} - (Rollout)")
            plt.show(block = False)