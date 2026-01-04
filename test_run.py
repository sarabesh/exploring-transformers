#here we will try to use the classes, create a model and run a forward pass

import torch
from models.gpt import GPT  
from tokenizer.bpe import BPE
if __name__ == "__main__":

    # Define model parameters
    vocab_size = 1000  
    embed_size = 64    
    num_blocks = 2     
    heads = 4          

    # Initialize the GPT model
    model = GPT(vocab_size, embed_size, num_blocks, heads)

    # Create a simple tokenizer
    tokenizer = BPE().load_trained('bpe_tokenizer.json')

    # Sample input text
    input_text = "hello world"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor([input_ids])  # Shape: (1, seq_length)

    # Run a forward pass through the model
    output_logits = model(input_tensor)

    print("Input IDs:", input_ids)
    print("Output logits shape:", output_logits.shape)  # Should be (1, seq_length, vocab_size)
