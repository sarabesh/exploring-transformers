#trying to profile a simple GPT model using torch profiler
#here we will try to use the classes, create a model and run a forward pass

import torch
from models.gpt import GPT  
from tokenizer.bpe import BPE
from torch.profiler import profile, ProfilerActivity, record_function


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

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        device = "cuda"
        activities += [ProfilerActivity.CUDA]
    elif torch.xpu.is_available():
        device = "xpu"
        activities += [ProfilerActivity.XPU]
    else:
        print(
            "Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices"
        )
        import sys

        sys.exit(0)

    sort_by_keyword = device + "_time_total"


    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            output_logits =model(input_tensor.to(device))

    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

    print("Input IDs:", input_ids)
    print("Output logits shape:", output_logits.shape)  # Should be (1, seq_length, vocab_size)

    softmax_output = torch.softmax(output_logits, dim=-1)
    predicted_tokens = torch.argmax(softmax_output, dim=-1)
    print("Predicted token IDs:", predicted_tokens)

    # Decode the predicted token IDs back to text
    predicted_text = tokenizer.decode(predicted_tokens[0].tolist())
    print("Predicted text:", predicted_text)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
