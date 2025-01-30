import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter
import math
from tree_sitter import Language
import tree_sitter_java as tsjava
import tree_sitter_tlaplus as tstlaplus

class CorpusFile:
    def __init__(self, file_path,separator, tokenizer, max_sequences=1000):
        super(CorpusFile, self).__init__()
        self.tokenizer = tokenizer
        self.dataset_file_path = file_path
        self.separator = separator
        self.token_counter = Counter()

        self.sequences = self.__load_sequences_of_code_snippet(max_sequences)
        self.__initialize_token_lengths_for_each_code_sequence()
        if tokenizer.parser.language == Language(tstlaplus.language()):
            self.__extend_sequence_and_token_length_after_generating_synthetic_identifier()
        self.__save_vocab()
        
    
    def __save_vocab(self):
        self.tokenizer.save_vocab('vocab.json',self.tokenizer.vocab)
        print(f"Vocabulary for the dataset {self.dataset_file_path} is saved in current execution path.")


    def __initialize_token_lengths_for_each_code_sequence(self):
        self.token_lengths = []
        for code in self.sequences:
            token_ids = self.tokenizer.encode(code)
            self.token_lengths.append(len(token_ids))
            self.token_counter.update(token_ids)
        large_specs = [length for length in self.token_lengths if length>512]
        print(f"The total number of specs: {len(self.token_lengths)}, The number of specs with token lengths greater than 512: {len(large_specs)}",)    

    def __extend_sequence_and_token_length_after_generating_synthetic_identifier(self):
        anonym_seqs = []
        anonym_token_lengths =[]
        for code in self.sequences:
            anonym_code = self.tokenizer.anonymize_variable_names_for_tla(code)
            anonym_seqs.append(anonym_code)
            token_ids = self.tokenizer.encode(anonym_code)
            anonym_token_lengths.append(len(token_ids))
            self.token_counter.update(token_ids)
        
        self.sequences.extend(anonym_seqs)
        self.token_lengths.extend(anonym_token_lengths)
            
    def __load_sequences_of_code_snippet(self, max_codes):
        with open(self.dataset_file_path, 'r') as f:
            data = f.read()
        sequences = data.split(self.separator)
        return sequences[:max_codes]

    def getSortedSequences(self):
        # Sort sequences by complexity score, i.e., token lengths
        sorted_sequences = sorted(zip(self.sequences, self.token_lengths), key=lambda x: x[1])
        return [seq for seq, _ in sorted_sequences]

    def getDynamicSample(self, epoch, total_epochs):
        # Adjust difficulty of samples based on current epoch
        sorted_sequences = self.getSortedSequences()
        # Gradually include more complex sequences
        difficulty_threshold = int((epoch / total_epochs) * len(sorted_sequences))
        # Ensure at least one sequence is included
        if difficulty_threshold == 0:
            difficulty_threshold = 1  # Select at least one sample
        
        # Select sequences up to the difficulty threshold
        sampled_sequences = sorted_sequences[:difficulty_threshold]
        
        # Shuffle the selected sequences
        random.shuffle(sampled_sequences)
        
        return sampled_sequences

    def getMaxSequenceLength(self):
        max_len = max(self.token_lengths)

        return max_len
    
    def fetchSequences(self):
        return self.sequences

    def readContent(self):
        return self.data

class CodeDataset(Dataset):
    def __init__(self, specs, tokenizer, language):
        self.specs = specs
        self.tokenizer = tokenizer
        self.language = language
    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        max_retries = len(self.specs)
        retries = 0
        while retries < max_retries:
            spec = self.specs[idx]
            if self.language == 'TLA':
                self.tokenizer.parser.language = Language(tstlaplus.language())
                token_ids = self.tokenizer.encode(spec)
                state_variables = self.tokenizer.extract_all_tla_state_variables(spec)
                set_of_keywords = self.tokenizer.extract_all_keywords(spec)
                if token_ids:
                    return torch.tensor(token_ids),state_variables,set_of_keywords
            elif self.language == 'Java':
                self.tokenizer.parser.language = Language(tsjava.language())
                set_of_keywords = self.tokenizer.extract_all_keywords(spec)
                class_fields = self.tokenizer.extract_all_java_class_fields(spec)
                token_ids = self.tokenizer.encode(spec)
                if token_ids:
                    return torch.tensor(token_ids), class_fields, set_of_keywords
            idx = (idx + 1) % len(self.specs)
            retries += 1
        raise ValueError("No valid code found in the dataset.")

class TrainingSelfAttention:
    
    def __init__(self, model, tokenizer, mask_prob, batch_size, epochs, tla_corpus, java_corpus):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.mask_prob = mask_prob
        self.epochs = epochs
        self.device = 'cpu' #torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100,label_smoothing=0.1)
        self.tla_corpus = tla_corpus
        self.java_corpus = java_corpus
        self.language = ''

    def load_data(self, epoch, training_task):

        if epoch % 2 == 0:
            print("Training on TLA+ data for this epoch.")
            specs = self.tla_corpus.getDynamicSample(epoch, self.epochs)
            self.language = 'TLA'
            dataset = CodeDataset(specs, self.tokenizer, 'TLA')
        else:
            print("Training on Java data for this epoch.")
            code = self.java_corpus.getDynamicSample(epoch, self.epochs)
            self.language = 'Java'
            dataset = CodeDataset(code, self.tokenizer, 'Java')

        if training_task.lower() == 'mlm':
            self.train_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn, shuffle=True)    
        elif  training_task.lower() == 'ntp':
            self.train_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_ntp, shuffle=True) 
        else:
            raise ValueError(f'Training task {training_task} is not supported.')   

    
    def _collate_fn_ntp(self, batch):
        token_batch = [item[0] for item in batch]
        token_batch = torch.nn.utils.rnn.pad_sequence(token_batch, batch_first=True, padding_value= self.tokenizer.vocab['__PAD__'])
        input,labels = create_ntp_inputs(token_batch,padding_value= self.tokenizer.vocab['__PAD__'])
        return input,labels

    def _collate_fn(self, batch):

        # Batch contains pairs (tokens, state_vars)
        token_batch = [item[0] for item in batch]
        state_variables = [item[1] for item in batch]
        set_of_keywords = set() 
        for item in batch: 
            if item[2] is not None: 
                set_of_keywords.update(item[2])
        
        token_batch = torch.nn.utils.rnn.pad_sequence(token_batch, batch_first=True, padding_value= self.tokenizer.vocab['__PAD__'])
        masked_input, labels, var_masked_indices = create_mlm_inputs(token_batch, state_variables, set_of_keywords, self.tokenizer,self.mask_prob)

        return masked_input, labels, var_masked_indices, state_variables
    
    def train(self):
        epoch_range = range(self.epochs)
        time_for_ntp_training = [self.epochs // 2, (self.epochs // 2)+1]
        for epoch in epoch_range:
            if epoch in time_for_ntp_training:
                print(f'Epoch {epoch + 1}/{self.epochs} - NTP training starts.')
                self.load_data(epoch, 'ntp')
                self._train_one_epoch_ntp()
            else:
                print(f'Epoch {epoch + 1}/{self.epochs} - MLM training task.')
                self.load_data(epoch, 'mlm')
                self._train_one_epoch()
            
            self.scheduler.step()


    def __get_spans(self,mask):
        """
        Extract contiguous spans of True values from a boolean mask.
        Args:
            mask (torch.Tensor): Boolean mask of shape (seq_len,).
        Returns:
            List[Tuple[int, int]]: List of (start_idx, end_idx) tuples for each span.
        """
        spans = []
        start = None

        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                spans.append((start, i))  # End the current span
                start = None
        
        if start is not None:  # Handle the case where the last span reaches the end
            spans.append((start, len(mask)))
        
        return spans
    
    def __compute_non_identifier_loss(self, labels, var_masked_indices, logits):
        _,_,vocab_size = logits.size()
        non_identifier_mask = (labels != -100) & ~var_masked_indices  # Exclude identifier tokens
        non_identifier_logits = logits[non_identifier_mask]
        non_identifier_labels = labels[non_identifier_mask]
        #class_weights = self.weights
            
        if non_identifier_logits.size(0) > 0:  # Ensure there are tokens to compute loss for
            non_identifier_loss = torch.nn.functional.cross_entropy(
                non_identifier_logits.view(-1,vocab_size),
                non_identifier_labels.view(-1)
                #weight=class_weights.to(self.device)
            )
        else:
            non_identifier_loss = 0.0  # No non-identifier tokens to compute loss

        return non_identifier_loss 

    def __compute_loss_for_label_span(self, mean_logits, unique_labels):   
        target_distribution = torch.zeros_like(mean_logits)
        target_distribution[unique_labels] = 1.0
        # Normalize the target distribution
        target_distribution /= target_distribution.sum()                         
                        
        # Compute KL divergence loss
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(mean_logits, dim=0),
            target_distribution,
            reduction="batchmean",
        )
        return loss
    
    def __compute_identifier_loss(self, labels, var_masked_indices, logits, variables):
        
        batch_size,_,_ = logits.size()
        identifier_loss = 0.0
        penalty_loss = 0.0
        total_spans = 0
        
        for i in range(batch_size):
                identifier_masked = var_masked_indices[i]
                identifer_labels = labels[i]
                identifer_logits = logits[i]  
                spans = self.__get_spans(identifier_masked)
                for start, end in spans:
                    
                    span_logits = identifer_logits[start:end]
                    span_labels = identifer_labels[start:end]
                    mean_logits = span_logits.mean(dim=0)
                    
                    # Create the target distribution
                    unique_labels = span_labels[span_labels != -100].unique()
                    inverse_vocab_elements = [self.tokenizer.inverse_vocab[item] for item in unique_labels.tolist()]
                    #print(self.tokenizer.inverse_vocab[torch.argmax(mean_logits).item()], inverse_vocab_elements, variables[i])
                    if len(unique_labels) == 0:
                        continue  # Skip spans with no valid labels
                    
                    identifier_loss += self.__compute_loss_for_label_span(mean_logits, unique_labels)
                   
                    """ predicted_token = torch.argmax(mean_logits).item()
                    confidence = torch.softmax(mean_logits, dim=0)[predicted_token]
                    add_penalty = self.__add_penalty(predicted_token, span_labels,variables[i])
                    if add_penalty:
                        print("-Adding penalty-")
                        penalty_loss += confidence**2
                    else:
                        penalty_loss -= confidence """

                    total_spans += 1
                    
        # Normalize the identifier loss
        if total_spans > 0:
            identifier_loss /= total_spans
            penalty_loss /= total_spans
        
        identifier_loss = identifier_loss + penalty_loss
        return identifier_loss
    
    def __compute_correct_predictions(self,labels, logits,padding_token=None):
        if padding_token is not None:
            # For NTP, ignore padding tokens
            masked_positions = labels != padding_token
        else:
            # For MLM, ignore tokens with label -100
            masked_positions = labels != -100
        predicted_tokens = torch.argmax(logits, dim=-1)
        masked_predictions = predicted_tokens[masked_positions]
        masked_labels = labels[masked_positions]

        correct_predictions = (masked_predictions == masked_labels).sum().item()

        return correct_predictions
    
    def __compute_number_of_masked_tokens(self,labels):
        masked_positions = labels != -100
        masked_labels = labels[masked_positions]
        return masked_labels.numel()
    
    def __print_stats(self,total_loss, total_correct_predictions, total_masked_tokens):
        avg_loss = total_loss / len(self.train_loader)
        print(total_loss)
        print(f'Average Loss: {avg_loss:.4f}')
        
        overall_accuracy = total_correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0
        #self.model.set_accuracy(overall_accuracy)
        print(f"Overall Token Prediction Accuracy: {overall_accuracy * 100:.2f}%")
    

    def _train_one_epoch_ntp(self):
        self.model.train()
        total_correct_predictions = 0
        total_masked_tokens = 0
        
        total_batches = len(self.train_loader)
        print(f"Processing {total_batches} sequences in the current epoch.")
        padding_token = self.tokenizer.vocab['__PAD__']
        milestone_indices = [
            total_batches // 10,
            total_batches // 2,   
            total_batches * 8 // 10
        ]
        milestone_indices.append(total_batches - 1)  

        for batch_idx, (input, labels) in enumerate(self.train_loader):
            input, labels = input.to(self.device), labels.to(self.device)
            if input.size(1) == 0:
                continue;
            self.optimizer.zero_grad()
            total_loss = 0.0
            logits,_ = self.model(input,task='ntp',language= self.language)  

            _, _, vocab_size = logits.size()
            loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=padding_token
                )
            
            total_loss += loss.item()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_correct_predictions += self.__compute_correct_predictions(labels, logits,padding_token)
            total_masked_tokens += labels[labels!=padding_token].numel()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            progress_percentage = (batch_idx + 1) / total_batches * 100
            if batch_idx in milestone_indices:
                print(f"Progress: {progress_percentage:.2f}% ({batch_idx + 1}/{total_batches} sequences)")
                self.__print_stats(total_loss, total_correct_predictions, total_masked_tokens)
    
    
    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_correct_predictions = 0
        total_masked_tokens = 0
        total_batches = len(self.train_loader)
        print(f"Processing {total_batches} sequences in current epoch.")
        # Calculate indices for 10%, 50%, and 80% milestones
        milestone_indices = [
            total_batches // 10,
            total_batches // 2,   
            total_batches * 8 // 10
        ]
        
        # Add the last batch as a milestone (100%)
        milestone_indices.append(total_batches - 1)
        for batch_idx, (masked_input, labels, var_masked_indices, state_vars) in enumerate(self.train_loader):
            masked_input, labels,var_masked_indices = masked_input.to(self.device), labels.to(self.device), var_masked_indices.to(self.device)
            if masked_input.size(1) == 0:
                continue;
            self.optimizer.zero_grad()
            # Step 1: Forward pass
            logits,_ = self.model(masked_input,task='mlm',language= self.language)
            # Step 2: Compute Non-Identifier Loss
            non_identifier_loss = self.__compute_non_identifier_loss(labels,var_masked_indices,logits)
            # Step 3: Compute Identifier Loss
            identifier_loss = self.__compute_identifier_loss(labels,var_masked_indices,logits, state_vars)
            # Total loss combines both components
            if identifier_loss == 0 and non_identifier_loss == 0:
                print("Warning: One of the batches has zero loss!")
                continue
            total_batch_loss = non_identifier_loss + identifier_loss
            total_batch_loss.backward()  # Backpropagation
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # Step 4: Track Total Loss
            total_loss += total_batch_loss.item()
            # Step 5: Compute Accuracy
            total_correct_predictions += self.__compute_correct_predictions(labels, logits)
            total_masked_tokens += self.__compute_number_of_masked_tokens(labels)
            # Release unused memory after each training step
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            progress_percentage = (batch_idx + 1) / total_batches * 100
            if batch_idx in milestone_indices:
                print(f"Progress: {progress_percentage:.2f}% ({batch_idx + 1}/{total_batches} sequences)")
        self.__print_stats(total_loss, total_correct_predictions, total_masked_tokens)
       
        
    def save_model(self, path):
        # After training completes
        # Save model 
        torch.save(self.model, path)

       
    def save_model_state(self, path):
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, path)


def __apply_random_masking(sequence, mask_prob, padded_token_id):
    # Find the first padding token
    pad_idx = (sequence == padded_token_id).nonzero(as_tuple=True)[0]  # Indices of padding tokens
    max_valid_idx = pad_idx[0].item() if len(pad_idx) > 0 else sequence.size(0)  # First padding or end of sequence
    max_valid_idx = max_valid_idx - 1
    mask = torch.zeros_like(sequence, dtype=torch.bool)
    if max_valid_idx > 0:
        random_mask = torch.bernoulli(torch.full((max_valid_idx,), mask_prob, device=sequence.device)).bool()
        mask[:max_valid_idx] = random_mask
    return mask

def __apply_block_masking(sequence, block_mask_prob, block_size_range, padded_token_id):
    pad_idx = (sequence == padded_token_id).nonzero(as_tuple=True)[0]  # Indices of padding tokens
    max_valid_idx = pad_idx[0].item() if len(pad_idx) > 0 else sequence.size(0)  # First padding or end of sequence
    max_valid_idx = max_valid_idx - 1
    # Use block_mask_prob to decide whether to apply block masking to each sample
    mask = torch.zeros_like(sequence, dtype=torch.bool)
    if torch.bernoulli(torch.tensor(block_mask_prob, device=sequence.device)):
        block_size = torch.randint(block_size_range[0], block_size_range[1] + 1, (1,)).item()
        if max_valid_idx >= block_size:
            start_pos = torch.randint(0, max_valid_idx - block_size + 1, (1,)).item()
            mask[start_pos:start_pos + block_size] = True
    return mask

def __apply_keyword_masking(sequence, keywords, tokenizer,masking_fraction=0.10):
    mask = torch.zeros_like(sequence, dtype=torch.bool)
    if keywords:
            num_samples = math.ceil(len(keywords) * masking_fraction)
            chosen_keywords = random.sample(list(keywords), num_samples)
            if chosen_keywords:
                for keyword in chosen_keywords:
                    keyword_id = tokenizer.vocab.get(keyword) or tokenizer.vocab.get('Ġ'+keyword)
                    if keyword_id:
                        condition = sequence == keyword_id
                        keyword_indices = torch.where(condition)[0].tolist()
                        mask[keyword_indices] = True
                    else:
                        print(f"Warning: keyword {keyword} is not in the vocab. The id is {keyword_id}")
    return mask

def __apply_variable_masking(sequence, chosen_var, tokenizer, padded_token_id,masking_fraction):
    
    mask = torch.zeros_like(sequence, dtype=torch.bool)
    pad_idx = (sequence == padded_token_id).nonzero(as_tuple=True)[0]  # Indices of padding tokens
    max_valid_idx = pad_idx[0].item() if len(pad_idx) > 0 else sequence.size(0)  # First padding or end of sequence
    max_valid_idx = max_valid_idx - 1
    # Encode state variables into token IDs
    chosen_var_ids = tokenizer.encode(chosen_var) 

    var_len = len(chosen_var_ids)

    # Find all positions where the full variable appears as a contiguous span
    positions_to_mask = []
    for i in range(max_valid_idx - var_len + 1):  # Ensure we stay within bounds
        if torch.all(sequence[i : i + var_len] == torch.tensor(chosen_var_ids, device=sequence.device)):
            positions_to_mask.append(i)
            

    # Sample a fraction of the positions for masking
    all_selected_variable_indices = set()
    if positions_to_mask:
        sampled_positions = random.sample(positions_to_mask, max(1, int(len(positions_to_mask) * masking_fraction)))
        all_selected_variable_indices.update(positions_to_mask)
        # Mask the full variable at the sampled positions
        for pos in sampled_positions:
            mask[pos : pos + var_len] = True
            all_selected_variable_indices.update(range(pos,pos + var_len))
           
    sorted_indices = list(all_selected_variable_indices)
    sorted_indices.sort()
    return mask, sorted_indices

def generate_80_10_10_masking_indices(masked_sequence):
    num_masked_tokens = masked_sequence.sum()
    # Indices of tokens to mask
    masked_indices = torch.where(masked_sequence)[0]
    # Shuffle the indices to randomly assign them to different categories
    shuffled_indices = masked_indices[torch.randperm(len(masked_indices))]

    # Compute split points for 80% with mask_id, 10% random token id, 10% unchanged
    num_mask = int(num_masked_tokens * 0.8)
    num_random = int(num_masked_tokens * 0.1)

    # Assign indices to each group
    mask_indices = shuffled_indices[:num_mask]
    random_indices = shuffled_indices[num_mask:num_mask + num_random]
    unchanged_indices = shuffled_indices[num_mask + num_random:]
    return mask_indices,random_indices, unchanged_indices

def __apply_combined_masking(
    sequence, mask_prob, block_mask_prob, block_size_range, keywords, 
    tokenizer, variables_to_mask, padded_token_id
):
    """
    Combines random, block, keyword, and variable masking into a single mask.
    """
    sequence_mask = torch.zeros_like(sequence, dtype=torch.bool)
    random_mask = __apply_random_masking(sequence, mask_prob, padded_token_id)
    block_mask = __apply_block_masking(sequence, block_mask_prob, block_size_range, padded_token_id)
    keyword_mask = __apply_keyword_masking(sequence, keywords, tokenizer, masking_fraction=0.10)
    
    variable_mask = torch.zeros_like(sequence, dtype=torch.bool)
    
    if variables_to_mask:
        # Randomly select one state variable to mask
        chosen_var = random.choice(variables_to_mask)
        variable_mask, positions_list = __apply_variable_masking(sequence, chosen_var, tokenizer, padded_token_id, masking_fraction = 0.50)
        random_mask[positions_list] = False
        block_mask[positions_list] = False

    sequence_mask = random_mask | block_mask | keyword_mask | variable_mask  
    # Combine masks
    return sequence_mask, variable_mask

def __apply_80_10_10_masking(sequence, sequence_mask, variable_mask, tokenizer, vocab_size, device):
    """
    Applies the 80-10-10 masking rule to a sequence.
    """
    masked_input = sequence.clone()
    labels = torch.full_like(sequence, -100)  # Default label is -100 (ignored index)
    masked_sequence_except_variable_indices = sequence_mask & ~variable_mask
    mask_indices,random_indices,unchanged_indices = generate_80_10_10_masking_indices(masked_sequence_except_variable_indices)
    
    ordered_mask_indices = mask_indices.sort().values  # Sort indices in ascending order
    mask_token_id = None
    # Replace 80% with [MASK]
    for idx, mask_position in enumerate(ordered_mask_indices):
        # Use distinct MASK tokens based on position
        mask_token_id = tokenizer.vocab[f'__MASK{0}__'] 
        masked_input[mask_position] = mask_token_id

    # Replace 10% with random tokens and Leave 10% unchanged
    after_special_tokens = len(tokenizer.special_tokens) + 1
    random_token_ids = torch.randint(after_special_tokens, vocab_size, size=random_indices.size(), device=device)
    masked_input[random_indices] = random_token_ids

    """ if mask_token_id is not None:
        mask_token_id = mask_token_id + 1 # Todo: we must ensure it is  witin valid range
    else:
        mask_token_id = tokenizer.vocab[f'__MASK0__'] """ 
    mask_token_id = tokenizer.vocab[f'__MASKVAR__']
    masked_var_indices = torch.where(variable_mask)[0]
    for position in masked_var_indices:
        masked_input[position] = mask_token_id

    # Update labels
    labels[mask_indices] = sequence[mask_indices]
    labels[random_indices] = sequence[random_indices]
    labels[unchanged_indices] = sequence[unchanged_indices]
    labels[masked_var_indices] = sequence[masked_var_indices]
    return masked_input, labels

def create_mlm_inputs(batch, variables_to_mask, keywords, tokenizer, mask_prob, block_mask_prob=0.0, block_size_range=(2, 4)):
    """
    Given a batch of tokenized sequences, mask some tokens.
    """
    device = batch.device
    #mask_token_id = tokenizer.vocab['__MASK__']
    padded_token_id = tokenizer.vocab['__PAD__']
    vocab_size = len(tokenizer.vocab)
    
    if not (0 <= mask_prob <= 1):
        raise ValueError("mask_prob should be between 0 and 1")
    
    if not (0 <= block_mask_prob <= 1):
        raise ValueError("block_mask_prob should be between 0 and 1")
    
    if batch.size(1) == 0:
        print("Empty sequences detected, skipping.")
        return batch, torch.tensor([], device=device)
    
    if batch.size(1) > 512:
        batch = batch[:, :512]

    masked_input = batch.clone()
    labels = torch.full_like(batch, -100)  # Default ignored value for labels
    var_indices = torch.zeros_like(batch, dtype=torch.bool)
    for i, sequence in enumerate(batch):
        # Apply various masking strategies
        sequence_mask,variable_mask = __apply_combined_masking(
            sequence, mask_prob, block_mask_prob, block_size_range, keywords, 
            tokenizer, variables_to_mask[i], padded_token_id
        )
        var_indices[i] = variable_mask
        if sequence_mask.sum() > 0:
            masked_input[i], labels[i] = __apply_80_10_10_masking(
                sequence, sequence_mask, var_indices[i], tokenizer, vocab_size, device
            )

    masked_tokens = (labels != -100).sum().item()          
    total_tokens = (masked_input != padded_token_id).sum().item()
    percentage_masked = (masked_tokens / total_tokens) * 100    
    #print(f"Percentage of masked tokens: {percentage_masked:.2f}%")
    #print("Masked input (tokens):", [[tokenizer.inverse_vocab[token_id.item()] for token_id in seq] for seq in masked_input])
    #print("Labels (token IDs):", labels)
    #print("Labels (tokens):", [[tokenizer.inverse_vocab[token_id.item()] if token_id != -100  else 'ignored' for token_id in seq] for seq in labels])



    return masked_input, labels, var_indices

def create_ntp_inputs(batch,padding_value):
    device = batch.device
    if batch.size(1) == 0:
        print("Empty sequences detected, skipping.")
        return batch, torch.tensor([], device=device)
    
    if batch.size(1) > 512:
        batch = batch[:, :512]
    
    # Shift the token batch to create the labels 
    labels = batch.clone() 
    labels[:, :-1] = batch[:, 1:] # Shift left 
    labels[:, -1] = padding_value # Set last token to padding
    
    return batch, labels 