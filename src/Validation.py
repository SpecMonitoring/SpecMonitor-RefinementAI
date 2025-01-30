import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch

class Validation:
    def __init__(self):
        super(Validation, self).__init__()
        
    # Visualize positional encodings
    def visualize_positional_encodings(encoding, seq_length):
        sns.heatmap(encoding[:seq_length].cpu().numpy(), cmap='viridis')
        plt.title('Positional Encodings')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Position')
        plt.show()

    def plot_attention_heatmap(attn_weights,token_ids, tokenizer, title="Attention Weights Heatmap"):
        """
        Plots a heatmap for the attention weights matrix.
        
        :param attn_weights: The attention weights matrix ()\\
        :param tokens: A list of tokens corresponding to the sequence positions (optional)
        :param title: The title for the plot
        """
        
        # Extract attention weights for a single head (e.g., head 0 for visualization)
        attention_for_head = attn_weights[6][0, 11].detach().cpu().numpy()  # Shape is [16, 16]
        #average_tensor = torch.mean(torch.stack(attn_weights), dim=(0,2)).detach().cpu().numpy()
        #squeezed = torch.stack(attn_weights).squeeze(1)
        #average_tensor = torch.mean(squeezed, dim=0).detach().cpu().numpy()  # Shape: [50, 50]        

        # Plotting the heatmap for the subset
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(attention_for_head, xticklabels=(tokenizer.tokens + tokenizer.token_types), yticklabels=(tokenizer.tokens + tokenizer.token_types), cmap="viridis", annot=True, annot_kws={"size": 5})
        plt.title(title)
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
        ax.set_yticklabels(ax.get_xticklabels(), fontsize=5)
        plt.show()