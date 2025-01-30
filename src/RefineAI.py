import torch
import torch.nn as nn
from transformers import RobertaModel,RobertaTokenizer, RobertaConfig

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """
        Forward pass with additional handling for attention weights.
        Args:
        - src: Input tensor (seq_len, batch_size, hidden_dim)
        - src_mask: Optional mask for attention
        - src_key_padding_mask: Optional padding mask
        - is_causal: Whether to use causal attention (unused in this case)
        """
        # Multi-head self-attention
        src2, self.attention_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        # Add & normalize
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class RefineAI(nn.Module):
    def __init__(self, tokenizer, vocab_size, num_heads, ff_dim):
        super(RefineAI, self).__init__()

        config = RobertaConfig.from_pretrained("microsoft/codebert-base-mlm",)
        self.bert = RobertaModel.from_pretrained("microsoft/codebert-base-mlm", config=config) 
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(0.1)
        
        # TLA-specific transformer encoder layers
        tla_encoder_layers = CustomTransformerEncoderLayer(d_model=config.hidden_size, 
                                                           nhead=num_heads, 
                                                           dim_feedforward=ff_dim, 
                                                           dropout=0.1)
        self.additional_layers = nn.TransformerEncoder(tla_encoder_layers, num_layers=4)
        
        # Java-specific transformer encoder layers
        java_encoder_layers = nn.TransformerEncoderLayer(d_model=config.hidden_size, 
                                                         nhead=num_heads, 
                                                         dim_feedforward=ff_dim, 
                                                         dropout=0.1)
        self.java_encoder = nn.TransformerEncoder(java_encoder_layers, num_layers=4)
 
         
        # Linear layer for MLM prediction (projection to vocabulary space)
        self.linear = nn.Linear(config.hidden_size, vocab_size)
        
        # Project embeddings for alignment task
        self.alignment_linear = nn.Linear(config.hidden_size, config.hidden_size)

   
    def forward(self, token_indices,task,language='TLA'):
        
        # Check if input is a single sequence (1D) or a batch of sequences (2D)
        token_indices = torch.tensor(token_indices, dtype=torch.long)
        if token_indices.dim() == 1:
            token_indices = token_indices.unsqueeze(0)  # Add batch dimension if 1D
        
         # Truncate token_indices to the max sequence length of CodeBERT 
        if token_indices.size(1) > 512:
            token_indices = token_indices[:, :512]
        
        attention_mask = (token_indices != self.tokenizer.vocab['__PAD__']).long()
        # Forward pass through CodeBERT (outputs hidden states and attention weights)
        outputs = self.bert(input_ids=token_indices,attention_mask=attention_mask,output_attentions=True)
        hidden_states = outputs.last_hidden_state
        attention_weights = outputs.attentions  # [num_layers, batch_size, num_heads, seq_len, seq_len]
        
        causal_mask = None
        if task.lower() == 'ntp':
            seq_len = token_indices.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        padding_mask = (token_indices == self.tokenizer.vocab['__PAD__'])
        # Language-specific layer
        if language.lower() == 'tla':
            for layer in self.additional_layers.layers:  
                hidden_states = layer(
                    hidden_states.permute(1, 0, 2),  # Shape: [seq_len, batch_size, hidden_dim]
                    src_mask=causal_mask,
                    src_key_padding_mask=padding_mask
                ).permute(1, 0, 2)  # Revert to [batch_size, seq_len, hidden_dim]
            
            # Retrieve attention weights for each layer
            """ attention_weights = [
                layer.attention_weights for layer in self.additional_layers.layers
            ] """
            # Get token embeddings from hidden_states
            #token_embeddings = hidden_states
        elif language.lower() == 'java':
            for layer in self.java_encoder.layers:
                hidden_states = layer(
                    hidden_states.permute(1, 0, 2),  
                    src_mask=causal_mask,
                    src_key_padding_mask=padding_mask
                ).permute(1, 0, 2) 
            # Get token embeddings from hidden_states
            #token_embeddings = hidden_states
        else:
            raise ValueError("Unknown language. Please specify either 'TLA' or 'Java'.")

        # Task-specific outputs
        if task.lower() == 'mlm' or task.lower() == 'ntp':
            logits = self.linear(hidden_states) # Project the hidden states to vocabulary space for MLM and reshape back to [batch_size, seq_len, vocab_size]
            return logits, attention_weights
        elif task.lower() == 'alignment':
            token_embeddings = self.alignment_linear(hidden_states)
            return token_embeddings
        else:
            raise ValueError(f"Unknown task: {task}. Supported tasks: mlm, alignment.")

    