#Byte Pair Encoding Tokenizer implementation, buut not the most optimized one.

import os
import collections
import json
from typing import Any



class BPETokenizer:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)} # initial vocab: byte values 0-255 mapped to their byte representation
        self.merges = {}
        self.next_id: int = 256


    def _get_max_pair(self, tokens): # return most common pair of consecutive tokens, using Counter
        if len(tokens) < 2:
            return None
            
        pair_freq = collections.Counter(
            zip(tokens, tokens[1:])
        )
        
        return pair_freq.most_common(1)[0][0]

    def _merge_once(self, tokens, pair, new_id):
        """One linear pass: replace every non-overlapping occurrence of `pair` with `new_id`."""
        out = []
        i = 0
        a, b = pair
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b: #merging the pair
                i += 2
                out.append(new_id)
            else:
                out.append(tokens[i])
                i += 1
        return out


    def train(self, text, max_vocab_size=1000):
        # Build BPE merges
        bytes = text.encode('utf-8') #encoding each char to bytes
       
        tokens = list(map(int, bytes)) #maps each byte to its integer value (0-255)

        while len(self.vocab) < max_vocab_size:

            max_pair = self._get_max_pair(tokens)

            if max_pair is None:
                print("No more pairs to merge. Stoping with vocab size:", len(self.vocab))
                break

            #add new token to vocab
            
            self.merges[max_pair] = self.next_id #adding to merges dict
            self.vocab[self.next_id] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]] #adding new token to vocab

            # Replace all occurrences of max_pair in tokens with new_token
            tokens = self._merge_once(tokens, max_pair, self.next_id)
            self.next_id += 1

        return tokens


    def encode(self, text):
        bytes = text.encode('utf-8')
        tokens = list(map(int, bytes))

        # apply merges greedily until stable (fixed point)
        while True:
            changed = False
            out = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and (tokens[i], tokens[i + 1]) in self.merges:
                    out.append(self.merges[(tokens[i], tokens[i + 1])])
                    i += 2
                    changed = True
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
            if not changed:
                break

        return tokens

    def encode_ordered(self, text):
        ##Encode by applying merges in training order and exhausting each merge before moving on.
        ##- Assumes self.merges[pair] == new_token_id allocated in training order (lower id = earlier).
        ##- Deterministic and training-order compliant.
        
        b = text.encode("utf-8", errors="surrogatepass")
        tokens = list(map(int, b))

        if not self.merges:
            return tokens

        # Build ordered list of (pair, merge_id) sorted by merge_id (lower id => earlier merge)
        ordered_merges = sorted(self.merges.items(), key=lambda kv: kv[1])  # [((a,b), id), ...]

        # For each merge in training order, repeatedly scan the token list and replace non-overlapping occurrences
        for (pair, merge_id) in ordered_merges:
            a, b = pair
            while True:
                i = 0
                changed = False
                out = []
                n = len(tokens)
                while i < n:
                    if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                        out.append(merge_id)   # apply merge (non-overlapping)
                        i += 2
                        changed = True
                    else:
                        out.append(tokens[i])
                        i += 1
                tokens = out
                if not changed:
                    break

        return tokens

    def decode(self, ids):

        tokens = b"".join(self.vocab[id] for id in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text
    
    def save_trained(self, filepath: str) -> None:
        """
        Persist tokenizer state to `filepath` in JSON format.

        Format:
        {
            "next_id": int,
            "vocab": { "<id>": [byte0, byte1, ...], ... },
            "merges": [ [t1, t2, new_id], ... ]
        }
        """
        payload = {
            "next_id": self.next_id,
            # store bytes as list of ints so JSON is safe and exact
            "vocab": {str(idx): list(token_bytes) for idx, token_bytes in self.vocab.items()},
            # store merges as list triples for deterministic ordering and cross-language compatibility
            "merges": [[int(a), int(b), int(new_id)] for (a, b), new_id in self.merges.items()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_trained(self, filepath: str) -> None:
        """
        Load tokenizer state from JSON file created by save_trained.
        This replaces self.vocab, self.merges, and self.next_id.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            payload: Any = json.load(f)

        # basic validation
        if not isinstance(payload, dict):
            raise ValueError("Invalid tokenizer file: root element must be an object.")

        # next_id
        self.next_id = int(payload.get("next_id", 256))

        # vocab: convert lists of ints back to bytes
        raw_vocab = payload.get("vocab", {})
        if not isinstance(raw_vocab, dict):
            raise ValueError("Invalid tokenizer file: 'vocab' must be a mapping.")
        self.vocab = {}
        for k, byte_list in raw_vocab.items():
            idx = int(k)
            if not isinstance(byte_list, list):
                raise ValueError(f"Invalid vocab entry for id {k}: expected list of byte ints.")
            self.vocab[idx] = bytes(int(b) for b in byte_list)

        # merges: load list-of-triples or dict form (backwards compatibility)
        raw_merges = payload.get("merges", [])
        self.merges = {}
        # accept either a list-of-triples [[a,b,new_id], ...] or a dict {"a b": new_id}
        if isinstance(raw_merges, list):
            for triple in raw_merges:
                if len(triple) != 3:
                    raise ValueError("Invalid merges entry: expected [a, b, new_id].")
                a, b, new_id = map(int, triple)
                self.merges[(a, b)] = new_id
        elif isinstance(raw_merges, dict):
            # legacy support: keys like "a b" -> id
            for k, v in raw_merges.items():
                parts = k.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid merge key: {k}")
                a, b = map(int, parts)
                self.merges[(a, b)] = int(v)
        else:
            raise ValueError("Invalid tokenizer file: 'merges' has unexpected type.")
    

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    #read text file for training
    
    BASE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE, "data/max.txt"), "r", encoding="utf-8") as file:
        sample_text = file.read()
    sample_tokens = tokenizer.train(sample_text, max_vocab_size=512)
    encoded = tokenizer.encode(sample_text)
    encoded2 = tokenizer.encode_ordered(sample_text)
    print("training output may not match encode:", sample_tokens == encoded)
    print("training output matches encode_ordered:", sample_tokens == encoded2)
  
    tokenizer.save_trained("bpe_tokenizer.txt")
    tokenizer.load_trained("bpe_tokenizer.txt")
    
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(sample_text==decoded)
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

