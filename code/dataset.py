import os
import re
from typing import List, Tuple

from torch.utils.data import Dataset


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.lower().strip()


def load_mr_data(pos_path: str, neg_path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    with open(pos_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(clean_text(line))
                labels.append(1)

    with open(neg_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(clean_text(line))
                labels.append(0)

    return texts, labels


def build_vocab(texts: List[str], min_freq: int = 1):
    freq = {}
    for text in texts:
        for token in text.split():
            freq[token] = freq.get(token, 0) + 1

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in freq.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict, max_len: int) -> List[int]:
    tokens = text.split()
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class MRDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: dict, max_len: int):
        self.labels = labels
        self.encoded = [encode_text(text, vocab, max_len) for text in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.encoded[idx], self.labels[idx]


def load_all_mr(data_dir: str):
    pos_path = os.path.join(data_dir, "rt-polarity.pos")
    neg_path = os.path.join(data_dir, "rt-polarity.neg")
    texts, labels = load_mr_data(pos_path, neg_path)
    max_len = max(len(t.split()) for t in texts)
    return texts, labels, max_len