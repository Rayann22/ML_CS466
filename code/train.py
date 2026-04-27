import random
import numpy as np
import torch
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MRDataset, build_vocab, load_all_mr
from model import KimCNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch):
    xs, ys = zip(*batch)
    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor

#
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = float((all_preds == all_labels).mean()) if len(all_labels) > 0 else 0.0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return acc, precision, recall, f1


def load_pretrained_word2vec(path):
    print("Loading pretrained word2vec from:", path)
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)
    print("Loaded %d vectors of dim %d" % (len(w2v.key_to_index), w2v.vector_size))
    return w2v


def build_embedding_matrix(vocab, w2v, embed_dim=300):
    matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim)).astype(np.float32)
    matrix[vocab["<pad>"]] = np.zeros(embed_dim, dtype=np.float32)

    hits = 0
    for token, idx in vocab.items():
        if token in ("<pad>", "<unk>"):
            continue
        if token in w2v:
            matrix[idx] = w2v[token]
            hits += 1

    print("Pretrained hits: %d / %d" % (hits, len(vocab)))
    return torch.tensor(matrix, dtype=torch.float32)


def train_one_fold(train_texts, train_labels, val_texts, val_labels, max_len, device, model_mode, w2v=None):
    batch_size = 32
    epochs = 6
    lr = 1.0
    embed_dim = 300

    vocab = build_vocab(train_texts)

    pretrained_embeddings = None
    static = False

    if model_mode == "static":
        pretrained_embeddings = build_embedding_matrix(vocab, w2v, embed_dim=embed_dim)
        static = True
    elif model_mode == "non_static":
        pretrained_embeddings = build_embedding_matrix(vocab, w2v, embed_dim=embed_dim)
        static = False

    train_dataset = MRDataset(train_texts, train_labels, vocab, max_len)
    val_dataset = MRDataset(val_texts, val_labels, vocab, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = KimCNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_classes=2,
        filter_sizes=(3, 4, 5),
        num_filters=100,
        dropout=0.5,
        pad_idx=vocab["<pad>"],
        pretrained_embeddings=pretrained_embeddings,
        static=static,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_metrics = {
        "acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc="Epoch %d" % (epoch + 1), leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        avg_loss = running_loss / len(train_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                "acc": val_acc,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1,
            }

        print(
            "    Epoch %d | loss %.4f | acc %.4f | precision %.4f | recall %.4f | f1 %.4f"
            % (epoch + 1, avg_loss, val_acc, val_precision, val_recall, val_f1)
        )

    return best_metrics


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model_mode = "static"   # choose: "rand", "static", "non_static"
    w2v = None

    if model_mode in ("static", "non_static"):
        w2v_path = "../embeddings/GoogleNews-vectors-negative300.bin.gz"
        w2v = load_pretrained_word2vec(w2v_path)

    data_dir = "../data"
    texts, labels, max_len = load_all_mr(data_dir)

    texts = np.array(texts, dtype=object)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
        print("\n========== Fold %d / 10 ==========" % fold_idx)

        train_texts = texts[train_idx].tolist()
        train_labels = labels[train_idx].tolist()
        val_texts = texts[val_idx].tolist()
        val_labels = labels[val_idx].tolist()

        fold_metrics = train_one_fold(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            max_len,
            device,
            model_mode,
            w2v,
        )

        fold_accuracies.append(fold_metrics["acc"])
        fold_precisions.append(fold_metrics["precision"])
        fold_recalls.append(fold_metrics["recall"])
        fold_f1s.append(fold_metrics["f1"])

        print(
            "Fold %d best | acc %.4f | precision %.4f | recall %.4f | f1 %.4f"
            % (
                fold_idx,
                fold_metrics["acc"],
                fold_metrics["precision"],
                fold_metrics["recall"],
                fold_metrics["f1"],
            )
        )

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    mean_precision = float(np.mean(fold_precisions))
    mean_recall = float(np.mean(fold_recalls))
    mean_f1 = float(np.mean(fold_f1s))

    print("\n========== Final 10-Fold CV Result ==========")
    for i in range(len(fold_accuracies)):
        print(
            "Fold %d | acc %.4f | precision %.4f | recall %.4f | f1 %.4f"
            % (
                i + 1,
                fold_accuracies[i],
                fold_precisions[i],
                fold_recalls[i],
                fold_f1s[i],
            )
        )

    print("\nMean accuracy: %.4f" % mean_acc)
    print("Std accuracy: %.4f" % std_acc)
    print("Mean precision: %.4f" % mean_precision)
    print("Mean recall: %.4f" % mean_recall)
    print("Mean f1-score: %.4f" % mean_f1)


if __name__ == "__main__":
    main()
