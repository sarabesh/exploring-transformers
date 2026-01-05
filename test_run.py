#here we will try to use the classes, create a model and run a forward pass

import torch
from models.gpt import GPT  
from tokenizer.bpe import BPE


if __name__ == "__main__":

    # Define model parameters
    vocab_size = 512 
    embed_size = 512 
    num_blocks = 2     
    heads = 4          

    # Initialize the GPT model
    model = GPT(vocab_size, embed_size, num_blocks, heads)

    # Create a simple tokenizer

    tokenizer = BPE()
    tokenizer.load_trained('bpe_tokenizer.txt')

    # Sample input text
    input_text = "hello, i am your friendly neighborhood"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor([input_ids])  # Shape: (1, seq_length)

    # Run a forward pass through the model
    output_logits = model(input_tensor)

    print("Input IDs:", input_ids)
    print("Output logits shape:", output_logits.shape)  # Should be (1, seq_length, vocab_size)

    softmax_output = torch.softmax(output_logits, dim=-1)
    predicted_tokens = torch.argmax(softmax_output, dim=-1)
    print("Predicted token IDs:", predicted_tokens)

    # Decode the predicted token IDs back to text
    predicted_text = tokenizer.decode(predicted_tokens[0].tolist())
    print("Predicted text:", predicted_text)
