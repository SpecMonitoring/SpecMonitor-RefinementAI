
import os
import torch
import torch.nn as nn
import pickle
import json
import argparse
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_tlaplus as tstlaplus
from src.ASTCodeTokenizer import ASTCodeTokenizer
from src.RefineAI import RefineAI
from src.Validation import Validation
from src.TrainingSelfAttention import TrainingSelfAttention, CorpusFile
from src.TrainingCrossAttention import TrainingCrosssAttention


RAW_FILE_TLAPLUS = 'data/tla_code_corpus_extended.txt'
RAW_FILE_JAVA = 'data/java_code_corpus.txt'
MAPPING_DATASET = 'data/mapping_dataset.json'
MAPPING_DATASET_VALIDATION = 'data/mapping_dataset_validation.json'
REFINEAIMODEL_PATH = 'model/RefineAI.pth'
JAVA_LANGUAGE = Language(tsjava.language())
TLAPLUS_LANGUAGE = Language(tstlaplus.language())


def check_file_exist(files):
    # Check for the existence of each required file
    missing_files = [file for file in files if not os.path.isfile(file)]

    if missing_files:
        missing_files_list = '\n'.join(missing_files)
        raise FileNotFoundError(f"The following required files are missing:\n{missing_files_list}\nPlease ensure these files exist.")

def load_tokenizer():

    try:
        with open('model/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            return tokenizer
    except FileNotFoundError:
        print("Warning! Tokenizer state file has not been found. Continues with creating a new tokenizer instance.")
        tokenizer = ASTCodeTokenizer()  # Initialize a new tokenizer if not found
        tokenizer.train_bpe_tokenizer([RAW_FILE_JAVA,RAW_FILE_TLAPLUS])
        return tokenizer


def load_weights_from_saved_model(model_path, task):

    tokenizer = load_tokenizer()
    embed_dim = 256
    num_heads = 8
    ff_dim = 4 * embed_dim
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
    tokenizer = load_tokenizer()
    tokenizer.parser.language = TLAPLUS_LANGUAGE
    #load from trained model
    embed_dim = 256
    num_heads = 8
    ff_dim = 4 * embed_dim
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
    files = [RAW_FILE_JAVA,RAW_FILE_TLAPLUS]
    check_file_exist(files)
    tokenizer = load_tokenizer()
    tokenizer.parser.language = TLAPLUS_LANGUAGE
    tla_corpus = CorpusFile(RAW_FILE_TLAPLUS, '__END_OF_CODE__', tokenizer)
    
    tokenizer.parser.language = JAVA_LANGUAGE
    java_corpus = CorpusFile(RAW_FILE_JAVA, '__END_OF_CODE__', tokenizer)
    
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
    files = [REFINEAIMODEL_PATH,MAPPING_DATASET]
    check_file_exist(files)
    tokenizer = load_tokenizer()
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


    







