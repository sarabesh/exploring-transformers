#Byte Pair Encoding Tokenizer implementation, buut not the most optimized one.

import os
import heapq


class BPETokenizer:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def get_max_pair_heap(self, tokens):
        pair_freq = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] = pair_freq.get(pair, 0) + 1
        if not pair_freq:
            return None
        
        #build max heap and get max
        max_pair_freq = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(max_pair_freq)

        return max_pair_freq

    def train(self, text, max_vocab_size=1000):
        # Build BPE merges
        bytes = text.encode('utf-8')
        tokens = list(map(int, bytes))
        max_pair_freq = self.get_max_pair_heap(tokens)

        while max_pair_freq and len(self.vocab) < max_vocab_size:
            max_pair = heapq.heappop(max_pair_freq)[1]

            #add new token to vocab
            self.merges[max_pair] = 257 + len(self.merges)
            new_token = self.merges[max_pair]
            self.vocab[new_token] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]

            # Replace all occurrences of max_pair in tokens with new_token
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == max_pair:
                    tokens = tokens[:i] + [new_token] + tokens[i+2:]
                else:
                    i += 1

        return tokens


    def encode(self, text):
        bytes = text.encode('utf-8')
        tokens = list(map(int, bytes))

        for merge in self.merges.keys():
            i = 0
            while i < len(tokens) - 1:
                t1, t2 = tokens[i], tokens[i + 1]
                if (t1, t2) == merge:
                    new_token = self.merges[merge]
                    tokens = tokens[:i] + [new_token] + tokens[i+2:]
                else:
                    i += 1
        
        return tokens

    def decode(self, ids):

        tokens = b"".join(self.vocab[id] for id in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text
    
    def save_trained(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for idx, token_bytes in self.vocab.items():
                f.write(f"{idx}\t{token_bytes}\n")
            f.write("Merges:\n")
            for (t1, t2), idx in self.merges.items():
                f.write(f"{t1} {t2} -> {idx}\n")
    
    def load_trained(self, filepath):
        self.vocab = {}
        self.merges = {}
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            merge_section = False
            for line in lines:
                line = line.strip()
                if line == "Merges:":
                    merge_section = True
                    continue
                if not merge_section:
                    idx, token_bytes = line.split("\t")
                    self.vocab[int(idx)] = bytes(map(int, token_bytes.strip("b'").split(b' ')))
                else:
                    parts = line.split(" -> ")
                    pair = tuple(map(int, parts[0].split(" ")))
                    idx = int(parts[1])
                    self.merges[pair] = idx
    

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    #read text file for training
    
    BASE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE, "data/max.txt"), "r", encoding="utf-8") as file:
        sample_text = file.read()
    sample_tokens = tokenizer.train(sample_text, max_vocab_size=512)
    tokenizer.save_trained("bpe_tokenizer.txt")
    tokenizer.load_trained("bpe_tokenizer.txt")
    
    encoded = tokenizer.encode(sample_text)
    # print("Encoded:", encoded)
    print(len(sample_tokens), len(encoded))
    print(encoded==sample_tokens)
    decoded = tokenizer.decode(encoded)
    # print("Decoded:", decoded)

    sample_text2 = """And the encode function does opposite of decode. The encode function takes a string of text as input and converts it into a list of integers (tokens) representing the encoded sequence. Here’s how it works step by step: Text Encoding: Initially, the function converts the input text into a list of integers using UTF-8 encoding. Each character in the text is encoded into one or more bytes, and these byte values are stored in tokens.
Merging Process: While there are at least two tokens in the list (while len(tokens) >= 2), the function calculates statistics (stats) about potential pairs of tokens that can be merged together to form new tokens. This is done using a helper function get_stats.
Choosing the Best Merge: Among all possible pairs (stats), the function selects the pair that is most eligible to be merged based on a predefined merging dictionary (merges). This dictionary keeps track of which pairs have been merged and assigned a new index (idx).
Merging and Updating Tokens: If the selected pair (pair) is found in merges, indicating it can be merged, the function merges the pair into a single token identified by idx. This process updates the tokens list by replacing the pair with the new token.
Termination: The process continues until no more eligible pairs can be found (if pair not in merges), indicating that all possible merges have been completed.
Return Tokens: Finally, the function returns the list of tokens representing the encoded sequence of the input text."""
    encoded2 = tokenizer.encode(sample_text2)
    # print("Encoded2:", encoded2)
    decoded2 = tokenizer.decode(encoded2)
    # print("Decoded2:", decoded2)
    print(sample_text2==decoded2)