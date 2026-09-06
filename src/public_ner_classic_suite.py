from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from torchcrf import CRF
except Exception:  # pragma: no cover
    CRF = None


PAD = "<PAD>"
UNK = "<UNK>"


def normalize_tags(tags: list[str]) -> list[str]:
    out = []
    prev = "O"
    for tag in tags:
        if tag == "O":
            out.append(tag)
            prev = tag
            continue
        if "-" not in tag:
            tag = "O"
        prefix, label = tag.split("-", 1) if "-" in tag else ("O", "")
        if prefix == "S":
            tag = "B-" + label
        elif prefix in {"M", "E"}:
            tag = "I-" + label
        elif prefix not in {"B", "I"}:
            tag = "O"
        if tag.startswith("I-") and not (prev == tag or prev == "B-" + tag[2:]):
            tag = "B-" + tag[2:]
        out.append(tag)
        prev = tag
    return out


def read_bio(path: Path) -> list[tuple[list[str], list[str]]]:
    sentences = []
    chars: list[str] = []
    tags: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if chars:
                    sentences.append((chars, normalize_tags(tags)))
                    chars, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            chars.append(parts[0])
            tags.append(parts[-1])
    if chars:
        sentences.append((chars, normalize_tags(tags)))
    return sentences


def split_long_sentences(sentences: list[tuple[list[str], list[str]]], max_len: int) -> list[tuple[list[str], list[str]]]:
    out = []
    for chars, tags in sentences:
        start = 0
        while start < len(chars):
            end = min(len(chars), start + max_len)
            while end < len(chars) and end > start and tags[end].startswith("I-"):
                end -= 1
            if end <= start:
                end = min(len(chars), start + max_len)
            out.append((chars[start:end], tags[start:end]))
            start = end
    return out


def tags_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if not tag.startswith("B-"):
            i += 1
            continue
        label = tag[2:]
        start = i
        i += 1
        while i < len(tags) and tags[i] == "I-" + label:
            i += 1
        spans.append((start, i, label))
    return spans


def evaluate_entities(gold_sentences: list[tuple[list[str], list[str]]], pred_tags: list[list[str]]) -> dict:
    tp = Counter()
    pred_n = Counter()
    gold_n = Counter()
    for (_, gold_tags), pred in zip(gold_sentences, pred_tags):
        gold = set(tags_to_spans(gold_tags))
        pred_set = set(tags_to_spans(pred))
        for _, _, label in gold:
            gold_n[label] += 1
        for _, _, label in pred_set:
            pred_n[label] += 1
        for _, _, label in gold & pred_set:
            tp[label] += 1
    labels = sorted(set(gold_n) | set(pred_n))
    total_tp = sum(tp.values())
    total_pred = sum(pred_n.values())
    total_gold = sum(gold_n.values())

    def prf(t: int, p: int, g: int) -> tuple[float, float, float]:
        precision = t / p if p else 0.0
        recall = t / g if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision, recall, f1 = prf(total_tp, total_pred, total_gold)
    by_label = {}
    for label in labels:
        p, r, f = prf(tp[label], pred_n[label], gold_n[label])
        by_label[label] = {"precision": p, "recall": r, "f1": f, "tp": tp[label], "pred": pred_n[label], "gold": gold_n[label]}
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "pred": total_pred, "gold": total_gold, "by_label": by_label}


def evaluate_tokens(gold_sentences: list[tuple[list[str], list[str]]], pred_tags: list[list[str]]) -> dict:
    gold = [tag for _, tags in gold_sentences for tag in tags]
    pred = [tag for sent in pred_tags for tag in sent]
    labels = sorted(set(gold) | set(pred))
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    out = {"accuracy": correct / len(gold) if gold else 0.0, "count": len(gold)}
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score

        out.update(
            {
                "macro_precision": float(precision_score(gold, pred, labels=labels, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(gold, pred, labels=labels, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
                "macro_f1_without_O": float(f1_score(gold, pred, labels=[x for x in labels if x != "O"], average="macro", zero_division=0)),
            }
        )
    except Exception as exc:
        out["sklearn_error"] = repr(exc)
    return out


def load_dataset(data_root: Path, name: str, max_len: int):
    if name == "GNER":
        base = data_root / "GNER_zenodo_5758466"
        train = read_bio(base / "train.txt")
        dev = []
        test = read_bio(base / "test.txt")
    elif name == "FGNER":
        base = data_root / "FGNER_zenodo_7262862" / "extracted" / "chen-echo-FGNER-corpus-db4b401"
        train = read_bio(base / "FGNER_train.txt")
        dev = read_bio(base / "FGNER_dev.txt")
        test = read_bio(base / "FGNER_test.txt")
    else:
        base = data_root / name
        train = read_bio(base / "train.txt")
        dev_path = base / "dev.txt"
        dev = read_bio(dev_path) if dev_path.exists() else []
        test = read_bio(base / "test.txt")
    return split_long_sentences(train, max_len), split_long_sentences(dev, max_len), split_long_sentences(test, max_len)


def build_vocab(train, all_sentences, min_freq: int):
    char_counts = Counter(ch for chars, _ in train for ch in chars)
    chars = [PAD, UNK] + sorted([ch for ch, n in char_counts.items() if n >= min_freq])
    tags = sorted({tag for _, sent_tags in all_sentences for tag in sent_tags}, key=lambda x: (x != "O", x))
    return {ch: i for i, ch in enumerate(chars)}, {tag: i for i, tag in enumerate(tags)}


class NerDataset(Dataset):
    def __init__(self, sentences, char2id, tag2id):
        self.items = []
        unk = char2id[UNK]
        for chars, tags in sentences:
            self.items.append(
                (
                    torch.tensor([char2id.get(ch, unk) for ch in chars], dtype=torch.long),
                    torch.tensor([tag2id[tag] for tag in tags], dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max())
    chars = torch.zeros((len(batch), max_len), dtype=torch.long)
    tags = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        chars[i, : len(x)] = x
        tags[i, : len(y)] = y
        mask[i, : len(x)] = True
    return chars, tags, lengths, mask


class ClassicTagger(nn.Module):
    def __init__(
        self,
        arch: str,
        vocab_size: int,
        tag_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        use_crf: bool,
        num_heads: int,
    ):
        super().__init__()
        if use_crf and CRF is None:
            raise RuntimeError("torchcrf is not installed.")
        self.arch = arch
        self.use_crf = use_crf
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        if arch in {"bilstm", "bilstm_attention", "bigru"}:
            if arch == "bigru":
                self.encoder = nn.GRU(emb_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
            else:
                self.encoder = nn.LSTM(emb_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
            out_dim = hidden_dim
            if arch == "bilstm_attention":
                heads = max(1, min(num_heads, hidden_dim))
                while hidden_dim % heads != 0 and heads > 1:
                    heads -= 1
                self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True, dropout=dropout)
                self.norm = nn.LayerNorm(hidden_dim)
        elif arch in {"idcnn", "idcnn2"}:
            channels = hidden_dim
            self.proj = nn.Conv1d(emb_dim, channels, kernel_size=1)
            blocks = 2 if arch == "idcnn" else 5
            layers = []
            for _ in range(blocks):
                for dilation in (1, 2, 1):
                    layers.extend(
                        [
                            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        ]
                    )
            self.encoder = nn.Sequential(*layers)
            out_dim = channels
        else:
            raise ValueError(arch)
        self.classifier = nn.Linear(out_dim, tag_size)
        self.crf = CRF(tag_size, batch_first=True) if use_crf else None

    def emissions(self, x, lengths, mask):
        emb = self.dropout(self.embedding(x))
        if self.arch in {"bilstm", "bilstm_attention", "bigru"}:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.encoder(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            if self.arch == "bilstm_attention":
                attended, _ = self.attention(out, out, out, key_padding_mask=~mask)
                out = self.norm(out + attended)
            return self.classifier(self.dropout(out))
        conv_in = emb.transpose(1, 2)
        out = self.proj(conv_in)
        out = self.encoder(out).transpose(1, 2)
        return self.classifier(self.dropout(out))

    def loss(self, x, tags, lengths, mask, criterion):
        emissions = self.emissions(x, lengths, mask)
        if self.use_crf:
            return -self.crf(emissions, tags, mask=mask.byte(), reduction="mean")
        ce_tags = tags.masked_fill(~mask, -100)
        return criterion(emissions.reshape(-1, emissions.shape[-1]), ce_tags.reshape(-1))

    def decode(self, x, lengths, mask, id2tag):
        emissions = self.emissions(x, lengths, mask)
        if self.use_crf:
            pred_ids = self.crf.decode(emissions, mask=mask.byte())
            return [normalize_tags([id2tag[idx] for idx in row]) for row in pred_ids]
        pred_ids = emissions.argmax(-1).cpu().tolist()
        mask_l = mask.cpu().tolist()
        return [normalize_tags([id2tag[idx] for idx, keep in zip(row, mrow) if keep]) for row, mrow in zip(pred_ids, mask_l)]


@torch.no_grad()
def predict_loader(model, loader, id2tag, device):
    model.eval()
    pred = []
    for chars, _, lengths, mask in loader:
        pred.extend(model.decode(chars.to(device), lengths.to(device), mask.to(device), id2tag))
    return pred


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(args, dataset_name: str) -> dict:
    train, dev, test = load_dataset(Path(args.data_root), dataset_name, args.max_len)
    char2id, tag2id = build_vocab(train, train + dev + test, args.min_freq)
    id2tag = {i: tag for tag, i in tag2id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ClassicTagger(args.arch, len(char2id), len(tag2id), args.emb_dim, args.hidden_dim, args.dropout, args.use_crf, args.num_heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    train_loader = DataLoader(NerDataset(train, char2id, tag2id), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(NerDataset(dev, char2id, tag2id), batch_size=args.batch_size, shuffle=False, collate_fn=collate) if dev else None
    test_loader = DataLoader(NerDataset(test, char2id, tag2id), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    best_state = None
    best_dev_f1 = -1.0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for chars, tags, lengths, mask in train_loader:
            chars = chars.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            loss = model.loss(chars, tags, lengths, mask, criterion)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            tokens = int(mask.sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
        row = {"epoch": epoch, "loss": total_loss / max(1, total_tokens)}
        if dev_loader:
            dev_pred = predict_loader(model, dev_loader, id2tag, device)
            dev_metrics = evaluate_entities(dev, dev_pred)
            row["dev_entity_f1"] = dev_metrics["f1"]
            if dev_metrics["f1"] > best_dev_f1:
                best_dev_f1 = dev_metrics["f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps({"dataset": dataset_name, "arch": args.arch, **row}, ensure_ascii=False), flush=True)
        history.append(row)
    if best_state:
        model.load_state_dict(best_state)
    pred = predict_loader(model, test_loader, id2tag, device)
    return {
        "dataset": dataset_name,
        "arch": args.arch,
        "train_sentences": len(train),
        "dev_sentences": len(dev),
        "test_sentences": len(test),
        "vocab_size": len(char2id),
        "tag_size": len(tag2id),
        "use_crf": args.use_crf,
        "device": str(device),
        "history": history,
        "test_entity": evaluate_entities(test, pred),
        "test_token": evaluate_tokens(test, pred),
        "seconds": time.time() - start,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--arch", choices=["bilstm", "bilstm_attention", "bigru", "idcnn", "idcnn2"], required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-crf", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"config": vars(args), "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(), "runs": []}
    for dataset in args.datasets:
        run = train_one(args, dataset)
        result["runs"].append(run)
        (out_dir / f"{dataset.lower()}_{args.arch}_result.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "classic_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
