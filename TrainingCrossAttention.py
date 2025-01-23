
import torch
from torch.utils.data import Dataset, DataLoader
from tree_sitter import Language
import tree_sitter_java as tsjava
import tree_sitter_tlaplus as tstlaplus
from itertools import zip_longest
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import re
import random


class PairedCodeDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
       
        tla_code = example['tla_code']
        self.tokenizer.parser.language = Language(tstlaplus.language())
        tla_token_ids = self.tokenizer.encode(tla_code)
        java_code = example['java_code']
        self.tokenizer.parser.language = Language(tsjava.language())
        java_token_ids = self.tokenizer.encode(java_code)
        positive_examples = example['positive_examples']
        negative_examples = example['negative_examples']
        
        return torch.tensor(tla_token_ids), torch.tensor(java_token_ids), positive_examples, negative_examples



class TrainingCrosssAttention:
    def __init__(self, model, tokenizer, batch_size, epochs, contrastive_margin=0.3, mapping_examples=None):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.contrastive_margin = contrastive_margin
        self.mapping_examples = mapping_examples
        self.positive_similarities = [] # monitoring variable
        self.negative_similarities = [] # monitoring variable
    
    

    def __sorting_key(self,example):
        return len(example['positive_examples']) + len(example['negative_examples'])
        """ # Define the order for each complexity level
        complexity_order = {"simple": 1, "intermediate": 2, "advanced": 3}
        # Extract level and number from the name
        match = re.match(r"(simple|intermediate|advanced)_(\d+)", example['name'], re.IGNORECASE)
        if match:
            level, num = match.groups()
            level_order = complexity_order[level.lower()]  # Get order based on level
            return (level_order, int(num))  # Convert number part to integer for proper sorting
        else:
            # Default to the end if there's a name that doesn't match the pattern
            return (float('inf'), 0) """
    
    def __getDynamicSample(self, epoch, total_epochs):
        sorted_examples = sorted(self.mapping_examples, key=self.__sorting_key)
        sorted_examples = sorted_examples
        # Gradually include more complex sequences
        difficulty_threshold = int((epoch / total_epochs) * len(sorted_examples))
        difficulty_threshold = max(1, difficulty_threshold)
       
        # Select sequences up to the difficulty threshold
        sampled_sequences = sorted_examples

        # Shuffle the selected sequences
        random.shuffle(sampled_sequences)
        
        # Return a limited number of sequences
        return sampled_sequences

    def load_data(self, epoch):
        # Sort examples by complexity level
        examples = self.__getDynamicSample(epoch, self.epochs)
        dataset = PairedCodeDataset(examples, self.tokenizer)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)

    def _collate_fn(self, batch):

        tla_token_batch = [item[0] for item in batch]
        java_token_batch = [item[1] for item in batch]
        positive_batch = [item[2] for item in batch]
        negative_batch = [item[3] for item in batch]
        
        tla_token_batch = torch.nn.utils.rnn.pad_sequence(tla_token_batch, batch_first=True, padding_value= self.tokenizer.vocab['__PAD__'])
        java_token_batch = torch.nn.utils.rnn.pad_sequence(java_token_batch, batch_first=True, padding_value= self.tokenizer.vocab['__PAD__'])
        
        return tla_token_batch, java_token_batch, positive_batch, negative_batch
    
    def _get_example_embedding(self, token_embeddings, token_ids, example_token_ids, max_gap=7):
        """
        Retrieves the embedding vector for a sequence of tokens, allowing for gaps between tokens.
        """
        
        selected_embeddings = []
        example_len = len(example_token_ids)
        # Define a window size (number of tokens to check forward or backward)
        window_size = example_len - 1   # You can adjust this size as needed

        # Convert example_token_ids to a tensor and move to the correct device 
        example_token_tensor = torch.tensor(example_token_ids).to(self.device)
        sequence_length = token_ids.size(1)
        # Iterate over token_ids to find a valid match for the entire sequence
        for i in range(sequence_length):
            # Check if the current token matches the first token of the example
            if torch.equal(token_ids[0][i], example_token_tensor[0]):
                # Assume initially that the sequence is fully matched
                is_valid_match = True

                # Check the next tokens within the specified window to see if they match
                for k in range(1, window_size + 1):  # example_len is the length of the positive example sequence

                    if k < example_len and i + k < sequence_length:
                        if not torch.equal(token_ids[0][i + k], example_token_tensor[k]):
                            is_valid_match = False  # If any token does not match, set to False and break
                            break
                
                # If the entire sequence matches, collect the embeddings
                if is_valid_match:
                    
                    # Collect embeddings for the entire matched sequence
                    for k in range(example_len):
                        selected_embeddings.append(token_embeddings[:, i + k, :])

                    # Once the full sequence is matched and embeddings are collected, break
                    break

        # Aggregate embeddings if necessary
        if len(selected_embeddings) > 0:  # Ensure all example tokens were found in order
            element_embedding = torch.mean(torch.stack(selected_embeddings), dim=0)#torch.max(torch.stack(selected_embeddings), dim=0)[0] #torch.mean(torch.stack(selected_embeddings), dim=0)
        else:
            # Return a zero vector if not all tokens were matched
            element_embedding = None

        return element_embedding


    def train(self):
        # Lists to store similarities
        positive_similarities_over_time = []
        negative_similarities_over_time = []
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            self.load_data(epoch+1)
            self._train_one_epoch(epoch + 1)
            self.scheduler.step()
            # Append average similarity for this epoch
            positive_similarities_over_time.append(np.mean(self.positive_similarities))
            negative_similarities_over_time.append(np.mean(self.negative_similarities))

        # Plotting the similarity trends after training
        # Plotting
        plt.figure(figsize=(12, 8))
        plt.plot(positive_similarities_over_time, label='Positive Pair Similarity')
        plt.plot(negative_similarities_over_time, label='Negative Pair Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Average Similarity')
        plt.legend()
        plt.title('Similarity Between Positive and Negative Pairs Over Epochs')
        plt.show()

    
    
    

    def _train_one_epoch(self,epoch):
        self.model.train()

        labels_tla = []
        tla_embeddings_list = []
        java_embeddings_list = []
        labels_java = []
        total_loss = 0.0
       
        for tla_token_batch, java_token_batch, positive_batch, negative_batch in self.train_loader:
            
            # Move to device
            tla_token_batch, java_token_batch= tla_token_batch.to(self.device), java_token_batch.to(self.device)
            
            self.optimizer.zero_grad()
            # Step 2: Forward pass
            tla_embeddings = self.model(tla_token_batch,'alignment', 'TLA')
            java_embeddings = self.model(java_token_batch,'alignment', 'Java')

            # Process positive and negative examples
            positive_loss = self._process_batch(positive_batch,tla_token_batch,java_token_batch, tla_embeddings, java_embeddings, tla_embeddings_list, 
                                                java_embeddings_list, labels_tla, labels_java, is_positive=True)
            negative_loss = self._process_batch(negative_batch,tla_token_batch,java_token_batch, tla_embeddings, java_embeddings, tla_embeddings_list, 
                                                java_embeddings_list, labels_tla, labels_java, is_positive=False)

            # Compute total loss with weighted contributions
            positive_weight = 1.0 / max(len(self.positive_similarities), 1)
            negative_weight = 1.0 / max(len(self.negative_similarities), 1)
            batch_loss = positive_loss * positive_weight + negative_loss * negative_weight
            total_loss += batch_loss

            if batch_loss > 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        avg_loss = total_loss / len(self.train_loader)
        print(f'Average Loss: {avg_loss:.4f}')

        # Generate and visualize embeddings
        if tla_embeddings_list or java_embeddings_list:
            self._visualize_embeddings(epoch, tla_embeddings_list, java_embeddings_list, labels_tla, labels_java)
        else:
            print("No positive examples to visualize.")
       
    
    def _process_batch(self, batch,tla_token_batch,java_token_batch, tla_embeddings, java_embeddings, tla_list, java_list, labels_tla, labels_java, is_positive):
        total_loss = 0.0
        for pairs in batch:
            for pair in pairs:
                self.tokenizer.parser.language = Language(tstlaplus.language())
                example_token_ids = self.tokenizer.encode(pair['tla_element'])
                tla_embedding = self._get_example_embedding(tla_embeddings,tla_token_batch, example_token_ids)
                self.tokenizer.parser.language = Language(tsjava.language())
                example_token_ids = self.tokenizer.encode(pair['java_element'])
                java_embedding = self._get_example_embedding(java_embeddings,java_token_batch, example_token_ids)

                if tla_embedding is not None and java_embedding is not None:
                    similarity = F.cosine_similarity(tla_embedding[0], java_embedding[0], dim=0).item()
                    if is_positive:
                        self.positive_similarities.append(similarity)
                        if similarity < 0.75:
                            total_loss += F.cosine_embedding_loss(tla_embedding, java_embedding, torch.ones(1, device=self.device))
                            self._store_embedding(tla_embedding, tla_list, labels_tla, pair['tla_element'], "TLA+")
                            self._store_embedding(java_embedding, java_list, labels_java, pair['java_element'], "Java")
                    else:
                        self.negative_similarities.append(similarity)
                        if similarity > 0.0:
                            total_loss += F.cosine_embedding_loss(tla_embedding, java_embedding, -torch.ones(1, device=self.device))

        return total_loss
    
    def _store_embedding(self, embedding, embeddings_list, labels, element, prefix):
        embedding_np = embedding.cpu().detach().numpy()[:1, :]
        if not any(np.array_equal(embedding_np, existing) for existing in embeddings_list) and element not in labels:
            embeddings_list.append(embedding_np)
            labels.append(f"{prefix}_{element}")

    def _visualize_embeddings(self, epoch, tla_embeddings_list, java_embeddings_list, labels_tla, labels_java):
        all_embeddings = np.vstack(tla_embeddings_list + java_embeddings_list)
        if all_embeddings.shape[0] < 3:
            print("Skipping UMAP for small dataset.")
            return

        # Apply UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
        reduced_embeddings = umap_reducer.fit_transform(all_embeddings)

        num_tla_embeddings = len(tla_embeddings_list)
        tla_reduced = reduced_embeddings[:num_tla_embeddings]
        java_reduced = reduced_embeddings[num_tla_embeddings:]

        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(tla_reduced[:, 0], tla_reduced[:, 1], color='blue', label='TLA+', alpha=0.6, edgecolors='w', s=100)
        plt.scatter(java_reduced[:, 0], java_reduced[:, 1], color='red', label='Java', alpha=0.6, edgecolors='w', s=100)

        for i, (x, y) in enumerate(tla_reduced):
            plt.text(x, y, labels_tla[i], fontsize=9, alpha=0.8)

        for i, (x, y) in enumerate(java_reduced):
            plt.text(x, y, labels_java[i], fontsize=9, alpha=0.8)

        plt.title(f'TLA+ and Java Embeddings Visualization with UMAP - Epoch {epoch}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show(block=False)

    def save_model_state(self, path):
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, path)