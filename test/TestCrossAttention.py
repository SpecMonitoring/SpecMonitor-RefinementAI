
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_tlaplus as tstlaplus

from src.RefineAI import RefineAI
import pickle
import torch
import json


REFINEAIMODEL_PATH = 'RefineAI.pth'
TEST_DATASET = 'mapping_dataset_validation.json'

def __get_example_embedding(token_embeddings,token_ids, example_token_ids):
        """
        Retrieves the embedding vector for specified token ids.
        """

        token_ids = torch.tensor(token_ids)
        selected_embeddings = []
        example_len = len(example_token_ids)
        window_size = example_len - 1   # Adjust this size as needed 
        example_token_tensor = torch.tensor(example_token_ids)
        sequence_length = token_ids.size(0)

        for i in range(sequence_length):
            if torch.equal(token_ids[i], example_token_tensor[0]):  # Check for a match in order
                 # Assume initially that the sequence is fully matched
                is_valid_match = True
                 # Check the next tokens within the specified window to see if they match
                for k in range(1, window_size + 1):  # example_len is the length of the positive example sequence
                    if k < example_len and i + k < sequence_length:
                        if not torch.equal(token_ids[i + k], example_token_tensor[k]):
                            is_valid_match = False  # If any token does not match, set to False and break
                            break
                
                if is_valid_match:
                    # Collect embeddings for the entire matched sequence
                    for k in range(example_len):
                        selected_embeddings.append(token_embeddings[:, i + k, :])
                    # Once the full sequence is matched and embeddings are collected, break
                    break

        # Aggregate embeddings if necessary
        if len(selected_embeddings) > 0:  
            element_embedding = torch.mean(torch.stack(selected_embeddings), dim=0)#torch.max(torch.stack(selected_embeddings), dim=0)[0]
        else:
            # Return a zero vector if not all tokens were matched
            element_embedding = None

        return element_embedding

def extract_fields_and_methods_java_code(code): 
    statements =[]
    # Load the Java language 
    JAVA_LANGUAGE = Language(tsjava.language())
    code_bytes = code.encode()
    java_parser = Parser()
    java_parser.language=JAVA_LANGUAGE
    tree = java_parser.parse(code_bytes)
    root_node = tree.root_node
    def extract_statements(node, code):
        if node.type == 'field_declaration':
            field_name = None
            field_type = None  
            for child in node.children:
             
                if child.type == 'variable_declarator':
                    
                    field_name = code[child.start_byte:child.end_byte].decode("utf8")

                elif 'type' in child.type:
                    field_type = code[child.start_byte:child.end_byte].decode("utf8")

            if field_type and field_name:
                statements.append(f"{field_type} {field_name};")
        elif node.type in {'method_declaration', 'constructor_declaration'}:
            name = None
            param_name=None
            param_type = None
            parameters =[]
            body_statements = []

            for child in node.children:

                if child.type == 'identifier':
                    name = code[child.start_byte:child.end_byte].decode("utf8")
                elif child.type == 'formal_parameters':
                    for param in child.children:
                        if param.type == 'formal_parameter':
                            for sub_child in param.children:
                                if sub_child.type == 'identifier':
                                    param_name = code[sub_child.start_byte:sub_child.end_byte].decode("utf8")
                                elif 'type' in sub_child.type:
                                    param_type = code[sub_child.start_byte:sub_child.end_byte].decode("utf8")
                            if param_type and param_name:
                                parameters.append(f"{param_type} {param_name}")
                        

                elif child.type in {'block','constructor_body'}:
                   
                    for grandchild in child.children:
                        if grandchild.type == 'expression_statement':
                            body_statement = code[grandchild.start_byte:grandchild.end_byte].decode("utf8")
                            statements.append(body_statement)
            
            statements.append(f"{name}()")
            """ if parameters: 
                statements.append(f"{name}({', '.join(parameters)})") 
            else: 
                statements.append(f"{name}()") """

        # Recursively extract from child nodes
        for child in node.children:
            extract_statements(child, code)
    
   
    extract_statements(root_node, code_bytes)
    return statements

def find_closest_embeddings(model, tokenizer, tla_code, java_code, tla_token, device='cpu'):

    tokenizer.parser.language = Language(tstlaplus.language())
    tla_tokens_ids = tokenizer.encode(tla_code)
    example_token_ids = tokenizer.encode(tla_token)    

    tokenizer.parser.language = Language(tsjava.language())
    java_token_ids = tokenizer.encode(java_code) 

    with torch.no_grad():
        tla_embeddings = model(torch.tensor(tla_tokens_ids).to(device),task='alignment', language= 'TLA')
        java_embeddings = model(torch.tensor(java_token_ids).to(device),task='alignment', language= 'Java')

    tla_embedding = __get_example_embedding(tla_embeddings, tla_tokens_ids, example_token_ids)
    if tla_embedding is None:
        print("Can not extract embedding. No similarity will be computed.")
        return
    

    closest_statements = []
    java_statements = extract_fields_and_methods_java_code(java_code)

    for java_statement in java_statements:
        statement_token_ids = tokenizer.encode(java_statement)
        java_statement_embedding = __get_example_embedding(java_embeddings, java_token_ids, statement_token_ids)
        if java_statement_embedding is None:
            print("Could not extract embedding for statement: " ,java_statement)
        else:
            similarity = torch.cosine_similarity(tla_embedding[0], java_statement_embedding[0], dim=0).item()
            similarity2 = torch.dot(tla_embedding[0], java_statement_embedding[0])
            #print(java_statement,similarity,distance,loss.mean())
            closest_statements.append((java_statement, similarity))

    # Sort phrases by similarity score in descending order
    closest_statements = sorted(closest_statements, key=lambda x: x[1], reverse=True)
    return closest_statements 

def main():
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            tokenizer.parser.language = Language(tsjava.language())
    except FileNotFoundError:
            print("tokenizer state file has not been found.")
            return
    embed_dim = 256
    num_heads = 8
    ff_dim = 4 * embed_dim
    vocab_size = len(tokenizer.vocab)
    model = RefineAI(tokenizer, vocab_size, num_heads,ff_dim)
    state = torch.load(REFINEAIMODEL_PATH)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    with open(TEST_DATASET, 'r') as f:
        examples = json.load(f)
    
    for example in examples['examples']:
        print("-------Example report block-------")
        print()
        for positive_example in example['positive_examples']: 
            closest_embeddings = find_closest_embeddings(model, tokenizer, example["tla_code"], example["java_code"], positive_example["tla_element"])
            print()
            print("Closest Java tokens to TLA+ token:", positive_example["tla_element"])
            print()
            for statement, similarity in closest_embeddings:
                print(statement,"---->",similarity)
        print("--------End report-----------")
        print()
            


if __name__ == "__main__":
    main()