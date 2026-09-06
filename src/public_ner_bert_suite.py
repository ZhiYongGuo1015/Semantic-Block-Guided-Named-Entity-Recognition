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
except Exception:  # pragma: no cover - optional dependency on server
    CRF = None


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


def split_long_sentences(sentences: list[tuple[list[str], list[str]]], max_chars: int) -> list[tuple[list[str], list[str]]]:
    out = []
    for chars, tags in sentences:
        start = 0
        while start < len(chars):
            end = min(len(chars), start + max_chars)
            while end < len(chars) and end > start and tags[end].startswith("I-"):
                end -= 1
            if end <= start:
                end = min(len(chars), start + max_chars)
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


def spans_to_tags(length: int, spans: list[tuple[int, int, str]]) -> list[str]:
    tags = ["O"] * length
    for start, end, label in spans:
        if start < 0 or end > length or start >= end:
            continue
        if any(tag != "O" for tag in tags[start:end]):
            continue
        tags[start] = "B-" + label
        for i in range(start + 1, end):
            tags[i] = "I-" + label
    return tags


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
    except Exception as exc:  # pragma: no cover - environment diagnostic only
        out["sklearn_error"] = repr(exc)
    return out


def build_lexicon(train, min_freq: int, min_len: int) -> dict[str, dict]:
    surface_counts: dict[str, Counter] = {}
    for chars, tags in train:
        text = "".join(chars)
        for start, end, label in tags_to_spans(tags):
            surface = text[start:end]
            if len(surface) < min_len:
                continue
            surface_counts.setdefault(surface, Counter())[label] += 1
    lexicon = {}
    for surface, counts in surface_counts.items():
        total = sum(counts.values())
        if total < min_freq:
            continue
        label, freq = counts.most_common(1)[0]
        lexicon[surface] = {"label": label, "freq": freq, "total": total}
    return lexicon


def predict_with_lexicon(chars: list[str], lexicon: dict[str, dict], max_entity_len: int) -> list[str]:
    text = "".join(chars)
    spans = []
    i = 0
    while i < len(text):
        best = None
        upper = min(len(text), i + max_entity_len)
        for end in range(upper, i, -1):
            item = lexicon.get(text[i:end])
            if item:
                best = (i, end, item["label"])
                break
        if best:
            spans.append(best)
            i = best[1]
        else:
            i += 1
    return spans_to_tags(len(chars), spans)


def fuse_by_label(bert_tags: list[str], lex_tags: list[str], label_source: dict[str, str]) -> list[str]:
    candidates = []
    for source, tags in [("bert", bert_tags), ("lexicon", lex_tags)]:
        for start, end, label in tags_to_spans(tags):
            if label_source.get(label, "bert") == source:
                priority = 1 if source == "bert" else 0
                candidates.append((priority, end - start, start, end, label))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    spans = []
    occupied = [False] * len(bert_tags)
    for _, _, start, end, label in candidates:
        if any(occupied[start:end]):
            continue
        spans.append((start, end, label))
        for i in range(start, end):
            occupied[i] = True
    spans.sort(key=lambda x: x[0])
    return spans_to_tags(len(bert_tags), spans)


def select_label_sources(dev_gold, bert_dev_pred, lex_dev_pred, margin: float):
    bert_metrics = evaluate_entities(dev_gold, bert_dev_pred)
    lex_metrics = evaluate_entities(dev_gold, lex_dev_pred)
    labels = sorted(set(bert_metrics["by_label"]) | set(lex_metrics["by_label"]))
    source = {}
    detail = {}
    for label in labels:
        bert_f1 = bert_metrics["by_label"].get(label, {}).get("f1", 0.0)
        lex_f1 = lex_metrics["by_label"].get(label, {}).get("f1", 0.0)
        source[label] = "lexicon" if lex_f1 > bert_f1 + margin else "bert"
        detail[label] = {"bert_dev_f1": bert_f1, "lexicon_dev_f1": lex_f1, "source": source[label]}
    return source, detail


def load_dataset(data_root: Path, name: str, max_chars: int):
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
        train_path = base / "train.txt"
        dev_path = base / "dev.txt"
        test_path = base / "test.txt"
        if not train_path.exists() or not test_path.exists():
            raise ValueError(f"unknown dataset {name!r}; expected {train_path} and {test_path}")
        train = read_bio(train_path)
        dev = read_bio(dev_path) if dev_path.exists() else []
        test = read_bio(test_path)
    return split_long_sentences(train, max_chars), split_long_sentences(dev, max_chars), split_long_sentences(test, max_chars)


def build_tag_vocab(sentences: list[tuple[list[str], list[str]]]) -> dict[str, int]:
    tags = sorted({tag for _, sent_tags in sentences for tag in sent_tags}, key=lambda x: (x != "O", x))
    return {tag: i for i, tag in enumerate(tags)}


class BertNerDataset(Dataset):
    def __init__(self, sentences, tokenizer, tag2id, max_seq_len: int):
        self.items = []
        cls_id = tokenizer.cls_token_id or tokenizer.convert_tokens_to_ids("[CLS]")
        sep_id = tokenizer.sep_token_id or tokenizer.convert_tokens_to_ids("[SEP]")
        unk_id = tokenizer.unk_token_id or tokenizer.convert_tokens_to_ids("[UNK]")
        o_id = tag2id["O"]
        for chars, tags in sentences:
            ids = [cls_id] + [tokenizer.convert_tokens_to_ids(ch) for ch in chars] + [sep_id]
            ids = [unk_id if token_id is None or token_id < 0 else token_id for token_id in ids]
            label_ids = [o_id] + [tag2id[tag] for tag in tags] + [o_id]
            valid_mask = [0] + [1] * len(chars) + [0]
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                label_ids = label_ids[:max_seq_len]
                valid_mask = valid_mask[:max_seq_len]
            self.items.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor(label_ids, dtype=torch.long),
                    torch.tensor(valid_mask, dtype=torch.bool),
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max())
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    tags = torch.zeros((len(batch), max_len), dtype=torch.long)
    valid_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, (ids, label_ids, vmask) in enumerate(batch):
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = True
        tags[i, : len(label_ids)] = label_ids
        valid_mask[i, : len(vmask)] = vmask
    return input_ids, attention_mask, tags, valid_mask


class BertTagger(nn.Module):
    def __init__(
        self,
        model_path: str,
        tag_size: int,
        dropout: float,
        use_crf: bool,
        model_type: str,
        rnn_type: str,
        rnn_hidden: int,
        rnn_attention: bool,
        attention_heads: int,
    ):
        super().__init__()
        if use_crf and CRF is None:
            raise RuntimeError("torchcrf is not installed in this environment.")
        from transformers import AutoModel, BertModel

        self.bert = BertModel.from_pretrained(model_path) if model_type == "bert" else AutoModel.from_pretrained(model_path)
        hidden = int(self.bert.config.hidden_size)
        self.rnn_type = rnn_type
        if rnn_type == "none":
            out_hidden = hidden
            self.rnn = None
        else:
            rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                hidden,
                rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            out_hidden = rnn_hidden * 2
        self.rnn_attention = rnn_attention
        if rnn_attention:
            heads = max(1, min(attention_heads, out_hidden))
            while out_hidden % heads != 0 and heads > 1:
                heads -= 1
            self.attention = nn.MultiheadAttention(out_hidden, heads, batch_first=True, dropout=dropout)
            self.attention_norm = nn.LayerNorm(out_hidden)
            self.attention_heads = heads
        else:
            self.attention = None
            self.attention_norm = None
            self.attention_heads = 0
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_hidden, tag_size)
        self.use_crf = use_crf
        self.crf = CRF(tag_size, batch_first=True) if use_crf else None

    def emissions(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask.long())
        sequence_output = outputs[0]
        if self.rnn is not None:
            lengths = attention_mask.long().sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence_output,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            sequence_output, _ = self.rnn(packed)
            sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
                sequence_output,
                batch_first=True,
                total_length=input_ids.shape[1],
            )
        if self.attention is not None:
            attended, _ = self.attention(
                sequence_output,
                sequence_output,
                sequence_output,
                key_padding_mask=~attention_mask,
            )
            sequence_output = self.attention_norm(sequence_output + attended)
        return self.classifier(self.dropout(sequence_output))

    def loss(self, input_ids, attention_mask, tags, criterion):
        emissions = self.emissions(input_ids, attention_mask)
        if self.use_crf:
            return -self.crf(emissions, tags, mask=attention_mask.byte(), reduction="mean")
        ce_tags = tags.masked_fill(~attention_mask, -100)
        return criterion(emissions.reshape(-1, emissions.shape[-1]), ce_tags.reshape(-1))

    def decode(self, input_ids, attention_mask):
        emissions = self.emissions(input_ids, attention_mask)
        if self.use_crf:
            return self.crf.decode(emissions, mask=attention_mask.byte())
        return emissions.argmax(-1).cpu().tolist()


def decode_batch(model, loader, id2tag, device):
    model.eval()
    pred_tags = []
    with torch.no_grad():
        for input_ids, attention_mask, _, valid_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            decoded = model.decode(input_ids, attention_mask)
            vmask = valid_mask.cpu().tolist()
            for row, mask_row in zip(decoded, vmask):
                pred_tags.append(normalize_tags([id2tag[idx] for idx, keep in zip(row, mask_row) if keep]))
    return pred_tags


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_dataset(args, dataset_name: str) -> dict:
    train, dev, test = load_dataset(Path(args.data_root), dataset_name, args.max_seq_len - 2)
    original_train_sentences = len(train)
    if args.train_fraction < 1.0:
        sample_size = max(1, int(len(train) * args.train_fraction))
        rng = random.Random(args.train_sample_seed)
        indices = list(range(len(train)))
        rng.shuffle(indices)
        keep = set(indices[:sample_size])
        train = [sent for idx, sent in enumerate(train) if idx in keep]
    tag2id = build_tag_vocab(train + dev + test)
    id2tag = {i: tag for tag, i in tag2id.items()}
    from transformers import AutoTokenizer, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(args.model_path) if args.tokenizer_type == "bert" else AutoTokenizer.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BertTagger(
        args.model_path,
        len(tag2id),
        args.dropout,
        args.use_crf,
        args.model_type,
        args.rnn_type,
        args.rnn_hidden,
        args.rnn_attention,
        args.attention_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    train_loader = DataLoader(BertNerDataset(train, tokenizer, tag2id, args.max_seq_len), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(BertNerDataset(dev, tokenizer, tag2id, args.max_seq_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate) if dev else None
    test_loader = DataLoader(BertNerDataset(test, tokenizer, tag2id, args.max_seq_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    history = []
    best_state = None
    best_dev_f1 = -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for input_ids, attention_mask, tags, _ in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags = tags.to(device)
            loss = model.loss(input_ids, attention_mask, tags, criterion)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        row = {"epoch": epoch, "loss": total_loss / max(1, steps)}
        if dev_loader:
            pred = decode_batch(model, dev_loader, id2tag, device)
            dev_metrics = evaluate_entities(dev, pred)
            row["dev_entity_f1"] = dev_metrics["f1"]
            if dev_metrics["f1"] > best_dev_f1:
                best_dev_f1 = dev_metrics["f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps({"dataset": dataset_name, **row}, ensure_ascii=False), flush=True)
        history.append(row)
    if best_state:
        model.load_state_dict(best_state)
    pred = decode_batch(model, test_loader, id2tag, device)
    run = {
        "dataset": dataset_name,
        "train_sentences": len(train),
        "original_train_sentences": original_train_sentences,
        "train_fraction": args.train_fraction,
        "train_sample_seed": args.train_sample_seed,
        "dev_sentences": len(dev),
        "test_sentences": len(test),
        "tag_size": len(tag2id),
        "device": str(device),
        "use_crf": args.use_crf,
        "model_path": args.model_path,
        "model_type": args.model_type,
        "tokenizer_type": args.tokenizer_type,
        "rnn_type": args.rnn_type,
        "rnn_hidden": args.rnn_hidden,
        "rnn_attention": args.rnn_attention,
        "attention_heads": args.attention_heads,
        "history": history,
        "test_entity": evaluate_entities(test, pred),
        "test_token": evaluate_tokens(test, pred),
        "seconds": time.time() - start,
    }
    if args.lexicon_fusion:
        lexicon = build_lexicon(train, args.lexicon_min_freq, args.lexicon_min_len)
        max_entity_len = max([len(x) for x in lexicon] or [1])
        if args.lexicon_max_entity_len:
            max_entity_len = min(max_entity_len, args.lexicon_max_entity_len)
        lex_test_pred = [predict_with_lexicon(chars, lexicon, max_entity_len) for chars, _ in test]
        run["lexicon_only"] = {
            "lexicon_size": len(lexicon),
            "max_entity_len": max_entity_len,
            "test_entity": evaluate_entities(test, lex_test_pred),
            "test_token": evaluate_tokens(test, lex_test_pred),
        }
        if dev and dev_loader:
            bert_dev_pred = decode_batch(model, dev_loader, id2tag, device)
            lex_dev_pred = [predict_with_lexicon(chars, lexicon, max_entity_len) for chars, _ in dev]
            label_source, source_detail = select_label_sources(dev, bert_dev_pred, lex_dev_pred, args.fusion_margin)
            fused_dev_pred = [fuse_by_label(bert_tags, lex_tags, label_source) for bert_tags, lex_tags in zip(bert_dev_pred, lex_dev_pred)]
            fused_test_pred = [fuse_by_label(bert_tags, lex_tags, label_source) for bert_tags, lex_tags in zip(pred, lex_test_pred)]
            dev_candidates = {
                "bert": (evaluate_entities(dev, bert_dev_pred), pred),
                "lexicon": (evaluate_entities(dev, lex_dev_pred), lex_test_pred),
                "per_label_fusion": (evaluate_entities(dev, fused_dev_pred), fused_test_pred),
            }
            selected_name, (selected_dev_metrics, selected_test_pred) = max(
                dev_candidates.items(),
                key=lambda item: (item[1][0]["f1"], item[1][0]["precision"], item[1][0]["recall"]),
            )
            run["lexicon_fusion"] = {
                "strategy": "per_label_dev_f1_source_selection",
                "fusion_margin": args.fusion_margin,
                "label_source": source_detail,
                "test_entity": evaluate_entities(test, fused_test_pred),
                "test_token": evaluate_tokens(test, fused_test_pred),
            }
            run["adaptive_dev_selection"] = {
                "strategy": "select_bert_lexicon_or_per_label_fusion_by_dev_f1",
                "selected": selected_name,
                "dev_candidates": {name: metrics for name, (metrics, _) in dev_candidates.items()},
                "selected_dev_entity": selected_dev_metrics,
                "test_entity": evaluate_entities(test, selected_test_pred),
                "test_token": evaluate_tokens(test, selected_test_pred),
            }
        else:
            run["lexicon_fusion"] = {"skipped": "dev set is required for per-label source selection"}
    return run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", choices=["auto", "bert"], default="auto")
    parser.add_argument("--tokenizer-type", choices=["auto", "bert"], default="auto")
    parser.add_argument("--rnn-type", choices=["none", "lstm", "gru"], default="none")
    parser.add_argument("--rnn-hidden", type=int, default=128)
    parser.add_argument("--rnn-attention", action="store_true")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=["GNER", "FGNER"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--train-sample-seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-crf", action="store_true")
    parser.add_argument("--lexicon-fusion", action="store_true")
    parser.add_argument("--lexicon-min-freq", type=int, default=1)
    parser.add_argument("--lexicon-min-len", type=int, default=2)
    parser.add_argument("--lexicon-max-entity-len", type=int, default=0)
    parser.add_argument("--fusion-margin", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"config": vars(args), "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(), "runs": []}
    for dataset in args.datasets:
        run = train_one_dataset(args, dataset)
        result["runs"].append(run)
        suffix = "crf" if args.use_crf else "softmax"
        (out_dir / f"{dataset.lower()}_bert_{suffix}_result.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "bert_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
