import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Set folder, dataset, and chunk size
save_folder = "edu_fineweb10B"
dataset_name = "sample-10BT"
tokens_per_chunk = int(1e8)

# Create save folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Download dataset
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=dataset_name, split="train")

# Initialize tokenizer and special end token
tokenizer = tiktoken.get_encoding("gpt2")

end_token = tokenizer._special_tokens['<|endoftext|>']

def tokenize_text(doc):
    """
    Tokenizes the given document into an array of uint16 tokens.
    Necessary because tokenizer we use outputs tokens as ints,
    which allows for more efficient storage and processing.
    
    The special end-of-text token is prepended to ensure proper segmentation
    Args:
        doc (dict): dictionary containing the document text.

    Returns:
        np.array: A numpy array of uint16 tokens.
    """
    tokens = [end_token] + tokenizer.encode_ordinary(doc["text"])
    return np.array(tokens, dtype=np.uint16)

def save_chunk(filename, tokens):
    """
    Saves chunk of tokens to file.
    Includes error handling to ensure the save process does not fail silently.
    
    Args:
        filename (str): The name of the file to save the tokens to.
        tokens (np.array): A numpy array of tokens to save.
    """
    try:
        np.save(filename, tokens)
    except Exception as e:
        print(f"Error saving chunk {filename}: {e}")

def handle_token_overflow(token_buffer, doc_tokens, current_token_index, chunk_size, chunk_num, save_dir):
    """
    Handles the case where the current chunk doesn't have enough space for all tokens.
    It fills up the current chunk and saves it, then begins a new chunk with the leftover tokens.

    Args:
        token_buffer (np.array): The current buffer for tokens.
        doc_tokens (np.array): The tokens from the current document.
        current_token_index (int): index in the buffer where new tokens are being added.
        chunk_size (int): The maximum size of the buffer.
        chunk_num (int): current chunk number.
        save_dir (str): Directory where chunks are saved.

    Returns:
        int, np.array, int: Updated token index, leftover document tokens, and updated chunk number.
    """
    remaining_space = chunk_size - current_token_index
    token_buffer[current_token_index:] = doc_tokens[:remaining_space]
    save_chunk(f"{save_dir}/edu_fineweb_{chunk_num:06d}.npy", token_buffer)
    
    # Start a new chunk with the leftover tokens
    return len(doc_tokens) - remaining_space, doc_tokens[remaining_space:], chunk_num + 1

def process_data(dataset, chunk_size, save_dir):
    """
    Tokenizes dataset, saves chunks of tokens to files. It processes the dataset 1 document at a time,
    ensuring that the tokens are split into chunks of a defined size and saved incrementally.

    Args:
        dataset (Dataset): The dataset to process.
        chunk_size (int): The maximum number of tokens per chunk.
        save_dir (str): Directory where chunks are saved.
    """
    chunk_num = 0  # Tracks which chunk is currently being saved
    current_token_index = 0  # Tracks how many tokens have been written into the current buffer
    token_buffer = np.empty(chunk_size, dtype=np.uint16)  # Pre-allocated buffer for tokens
    
    # Track progress across the dataset using tqdm
    with tqdm(total=len(dataset), unit="documents") as pbar:
        for doc in dataset:
            # Tokenize the current document
            doc_tokens = tokenize_text(doc)

            # Check if the current buffer has enough space for the new tokens
            if len(doc_tokens) > (chunk_size - current_token_index):
                # Handle the case where the buffer would overflow
                current_token_index, doc_tokens, chunk_num = handle_token_overflow(
                    token_buffer, doc_tokens, current_token_index, chunk_size, chunk_num, save_dir
                )

            # Add remaining tokens to the buffer
            token_buffer[current_token_index:current_token_index + len(doc_tokens)] = doc_tokens
            current_token_index += len(doc_tokens)
            pbar.update(1)
        
        # After all documents are processed, save the last incomplete chunk if it has tokens
        if current_token_index > 0:
            save_chunk(f"{save_dir}/edu_fineweb_{chunk_num:06d}.npy", token_buffer[:current_token_index])

process_data(dataset, tokens_per_chunk, save_folder)
