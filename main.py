
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_tlaplus as tstlaplus
from ASTCodeTokenizer import ASTCodeTokenizer
from RefineAI import RefineAI
from Validation import Validation
from TrainingSelfAttention import TrainingSelfAttention, CorpusFile
from TrainingCrossAttention import TrainingCrosssAttention
import torch
import torch.nn as nn
import pickle
import json
import argparse

CORPUS_FILE_TLAPLUS = 'tla_code_corpus_extended.txt'
CORPUS_FILE_JAVA = 'java_code_corpus.txt'
MAPPING_DATASET = 'mapping_dataset.json'
MAPPING_DATASET_VALIDATION = 'mapping_dataset_validation.json'
JAVA_LANGUAGE = Language(tsjava.language())
TLAPLUS_LANGUAGE = Language(tstlaplus.language())
# Create parser instances for each language
java_parser = Parser()
java_parser.language=JAVA_LANGUAGE
tlaplus_parser = Parser()
tlaplus_parser.language=TLAPLUS_LANGUAGE

REFINEAIMODEL_PATH = 'RefineAI.pth'
# Load the tokenizer from a file if it exists, otherwise create a new one
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    print("tokenizer state file has not been found.")
    tokenizer = ASTCodeTokenizer(tlaplus_parser)  # Initialize a new tokenizer if not found
    tokenizer.train_bpe_tokenizer([CORPUS_FILE_TLAPLUS,CORPUS_FILE_JAVA])

embed_dim = 256
num_heads = 8
ff_dim = 4 * embed_dim

def load_weights_from_saved_model(model_path, task):
    vocab_size = len(tokenizer.vocab)
    print("Current vocab size is: ", vocab_size)
    # Create the model
    model = RefineAI(tokenizer, vocab_size, num_heads,ff_dim)
    # Load the model if it exists, to resume training
    try:
        state = torch.load(model_path)
        
        # Extract old linear layer weights from the checkpoint
        if task.lower() == "self":
            old_linear_weight = state['model_state_dict']['linear.weight']
            old_linear_bias = state['model_state_dict']['linear.bias']
        elif task.lower() == "alignment":
            old_linear_weight = state['model_state_dict']['alignment_linear.weight']
            old_linear_bias = state['model_state_dict']['alignment_linear.bias']
        else:
            print("Error! old weights could not be read. starting training from scratch.")
            return model

        # Reinitialize the linear layer with the new vocabulary size
        new_linear = nn.Linear(model.bert.config.hidden_size, vocab_size)
        
        # Determine the size overlap between old and new linear layers
        num_weights_to_copy = min(new_linear.weight.size(0), old_linear_weight.size(0))
        num_biases_to_copy = min(new_linear.bias.size(0), old_linear_bias.size(0))

        # Copy over the old weights to the new linear layer
        with torch.no_grad():
            new_linear.weight[:num_weights_to_copy, :] = old_linear_weight[:num_weights_to_copy, :]
            new_linear.bias[:num_biases_to_copy] = old_linear_bias[:num_biases_to_copy]
        
        model.linear = new_linear  # Update model's linear layer
        # Load the rest of the model's state_dict
        if task.lower() == "self":
            model.load_state_dict({k: v for k, v in state['model_state_dict'].items() 
                                if 'linear.weight' not in k and 'linear.bias' not in k}, strict=False)
        else:
            model.load_state_dict({k: v for k, v in state['model_state_dict'].items() 
                                if 'alignment_linear.weight' not in k and 'alignment_linear.bias' not in k}, strict=False)
    
    except FileNotFoundError:
        print("No previous model found, starting training from scratch")

    return model
    

def visualizeSelfAttention():
    
    example_code ="""
-------------------------- MODULE OpenAddressing --------------------------
EXTENDS Naturals, FiniteSets, Sequences, TLC
CONSTANT K
VARIABLES x
vars == <<x>>
Init == x = 1 
Spec == Init /\ [][Next]_vars
========
"""
    language = 'TLA'
    task = 'mlm'
    tokenizer.parser.language = TLAPLUS_LANGUAGE
    #load from trained model
    vocab_size = len(tokenizer.vocab)
    model = RefineAI(tokenizer, vocab_size, num_heads,ff_dim)
    state = torch.load(REFINEAIMODEL_PATH)
    model.load_state_dict(state['model_state_dict'],strict=False)
    model.eval() 
    token_ids = tokenizer.encode(example_code)
    _, attn_weights = model(token_ids,task=task, language= language) # currently the model returns att_weights when task is mlm

    Validation.plot_attention_heatmap(attn_weights,token_ids, tokenizer, 
                                      title="Average Attention Weights")


def doTrainSelfAttention():

    tokenizer.parser.language = TLAPLUS_LANGUAGE
    tla_corpus = CorpusFile(CORPUS_FILE_TLAPLUS, '__END_OF_CODE__', tokenizer)
    tokenizer.parser.language = JAVA_LANGUAGE
    java_corpus = CorpusFile(CORPUS_FILE_JAVA, '__END_OF_CODE__', tokenizer)
    
    model = load_weights_from_saved_model(REFINEAIMODEL_PATH, "self")

    trainer = TrainingSelfAttention(
        model=model,
        tokenizer=tokenizer,
        batch_size=4,
        mask_prob=0.15,
        epochs=5,
        tla_corpus=tla_corpus,
        java_corpus=java_corpus
    )
    trainer.train()
    trainer.save_model_state(REFINEAIMODEL_PATH)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)


def doTrainCrossAttention():
    
    model = load_weights_from_saved_model(REFINEAIMODEL_PATH, "alignment")
    with open(MAPPING_DATASET, 'r') as f:
        examples = json.load(f)
    # Create the trainer
    trainer = TrainingCrosssAttention(
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
        epochs=6,
        mapping_examples = examples['examples']
    )

    trainer.train()
    trainer.save_model_state(REFINEAIMODEL_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Refinement AI",description="""This program fine-tunes CodeBERT to identify the implementation of state variables from TLA+ in Java. 
                                     It combines masked language modeling (MLM) and next token prediction (NTP) tasks to understand the structure of TLA+ and Java individually. 
                                     A cross-alignment task then maps semantically relevant elements from TLA+ state variables to their Java counterparts. """) 
    parser.add_argument("task", choices=["train_self", "train_cross"], help="""Task to perform: (1) train_self that commbines MLM and  NTP learninng  strategy.
                                                                                            (2) train_cross that commbines uses cosine similarity for alingment. """) 
    args = parser.parse_args()
    if args.task == "train_self": 
        doTrainSelfAttention() 
    elif args.task == "train_cross": 
        doTrainCrossAttention()
    #visualizeSelfAttention()


    







