import json
import re
from collections import Counter
from tqdm import tqdm

def simple_tokenize(text):
    """
    Tokenizes the input text using a regular expression.
    This function extracts alphanumeric word tokens.
    
    Args:
        text (str): The text to tokenize.
        
    Returns:
        list: A list of word tokens.
    """
    # \b\w+\b finds whole words (alphanumeric sequences)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def build_vocab(records, vocab_size=50000, special_tokens=["<pad>", "<unk>"]):
    """
    Builds a vocabulary dictionary from the text of all records.
    
    Args:
        records (list): List of record dictionaries, each with a "text" field.
        vocab_size (int): Maximum size of the vocabulary (including special tokens).
        special_tokens (list): List of special tokens to include.
    
    Returns:
        dict: A dictionary mapping tokens to unique integer IDs.
    """
    counter = Counter()
    for record in records:
        tokens = simple_tokenize(record["text"])
        counter.update(tokens)
    
    # Initialize vocabulary with special tokens.
    vocab = {}
    for token in special_tokens:
        vocab[token] = len(vocab)
    
    # Add the most common tokens to the vocabulary.
    for token, count in counter.most_common(vocab_size - len(vocab)):
        vocab[token] = len(vocab)
    
    return vocab

def tokenize_text(text, vocab):
    """
    Tokenizes the given text and converts tokens to token IDs using the vocabulary.
    
    Args:
        text (str): The input text.
        vocab (dict): A dictionary mapping tokens to IDs.
        
    Returns:
        tuple: A tuple (tokens, token_ids)
            - tokens (list): List of tokens.
            - token_ids (list): List of corresponding token IDs.
    """
    tokens = simple_tokenize(text)
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    return tokens, token_ids

def main():
    # Path to your cleaned JSON file from the WET file.
    input_json_path = r"C:\Users\gunne\OneDrive\Desktop\testing\wet_output.json"
    output_json_path = "wet_tokenized.json"
    vocab_output_path = "vocab.json"
    
    # Load the cleaned data.
    with open(input_json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    print("Building vocabulary from the dataset...")
    vocab = build_vocab(records, vocab_size=50000)
    print(f"Vocabulary size: {len(vocab)} tokens.")
    
    # Tokenize each record's text.
    print("Tokenizing records...")
    for record in tqdm(records, desc="Tokenizing"):
        tokens, token_ids = tokenize_text(record["text"], vocab)
        record["tokens"] = tokens
        record["token_ids"] = token_ids
    
    # Save the vocabulary to a JSON file.
    with open(vocab_output_path, "w", encoding="utf-8") as vf:
        json.dump(vocab, vf, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {vocab_output_path}.")
    
    # Save the tokenized records to a new JSON file.
    with open(output_json_path, 'w', encoding='utf-8') as out_f:
        json.dump(records, out_f, ensure_ascii=False, indent=2)
    print(f"Tokenized data written to {output_json_path}.")

if __name__ == "__main__":
    main()
