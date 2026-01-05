"""CoNLL format NER data loading."""
import numpy as np
from typing import List, Tuple

class CoNLLLoader:
    def __init__(self, label_list=None):
        self.label_list = label_list or []
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}

    def load_conll(self, filepath):
        sentences, labels = [], []
        current_tokens, current_labels = [], []
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if current_tokens:
                            sentences.append(current_tokens)
                            labels.append(current_labels)
                            current_tokens, current_labels = [], []
                    else:
                        parts = line.split()
                        current_tokens.append(parts[0])
                        current_labels.append(parts[-1] if len(parts) > 1 else "O")
            if current_tokens:
                sentences.append(current_tokens)
                labels.append(current_labels)
        except FileNotFoundError:
            pass
        print(f"Loaded {len(sentences)} sentences")
        return sentences, labels

    def build_vocab(self, sentences):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for sent in sentences:
            for token in sent:
                if token.lower() not in vocab:
                    vocab[token.lower()] = len(vocab)
        return vocab

    def encode_sentences(self, sentences, vocab, max_len=128):
        encoded = []
        for sent in sentences:
            ids = [vocab.get(t.lower(), vocab["<UNK>"]) for t in sent[:max_len]]
            ids += [vocab["<PAD>"]] * (max_len - len(ids))
            encoded.append(ids)
        return np.array(encoded)

    def encode_labels(self, label_sequences, max_len=128):
        encoded = []
        for seq in label_sequences:
            ids = [self.label2id.get(l, 0) for l in seq[:max_len]]
            ids += [0] * (max_len - len(ids))
            encoded.append(ids)
        return np.array(encoded)
