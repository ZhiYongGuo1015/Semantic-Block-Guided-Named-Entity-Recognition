from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import public_ner_bert_suite as base

try:
    from torchcrf import CRF
except Exception:  # pragma: no cover
    CRF = None


def read_meta(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_long_items(items: list[tuple[list[str], list[str], str]], max_chars: int) -> list[tuple[list[str], list[str], str]]:
    out = []
    for chars, tags, category in items:
        start = 0
        while start < len(chars):
            end = min(len(chars), start + max_chars)
            while end < len(chars) and end > start and tags[end].startswith("I-"):
                end -= 1
            if end <= start:
                end = min(len(chars), start + max_chars)
            out.append((chars[start:end], tags[start:end], category))
            start = end
    return out


def load_semantic_dataset(data_root: Path, name: str, max_chars: int):
    base_dir = data_root / name

    def load_split(split: str):
        bio_path = base_dir / f"{split}.txt"
        meta_path = base_dir / f"{split}_meta.jsonl"
        if not bio_path.exists():
            return []
        sentences = base.read_bio(bio_path)
        metas = read_meta(meta_path)
        if metas and len(metas) != len(sentences):
            raise ValueError(f"{split} meta length {len(metas)} != bio sentence length {len(sentences)}")
        if not metas:
            metas = [{"category": "未知"} for _ in sentences]
        return [(chars, tags, meta.get("category") or "未知") for (chars, tags), meta in zip(sentences, metas)]

    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")
    if not train or not test:
        raise ValueError(f"missing train/test files under {base_dir}")
    return (
        split_long_items(train, max_chars),
        split_long_items(dev, max_chars),
        split_long_items(test, max_chars),
    )


def strip_category(items: list[tuple[list[str], list[str], str]]) -> list[tuple[list[str], list[str]]]:
    return [(chars, tags) for chars, tags, _ in items]


def build_category_vocab(items: list[tuple[list[str], list[str], str]]) -> dict[str, int]:
    categories = sorted({category for _, _, category in items})
    return {category: idx for idx, category in enumerate(categories)}


def build_decoupled_tag_maps(tag2id: dict[str, int]) -> dict:
    entity_types = sorted(
        {
            tag.split("-", 1)[1]
            for tag in tag2id
            if tag != "O" and "-" in tag
        }
    )
    if not entity_types:
        raise ValueError("BIO tag vocabulary contains no entity types")
    type2id = {
        entity_type: index
        for index, entity_type in enumerate(entity_types)
    }
    boundary_ids = torch.zeros(len(tag2id), dtype=torch.long)
    type_ids = torch.zeros(len(tag2id), dtype=torch.long)
    entity_mask = torch.zeros(len(tag2id), dtype=torch.float)
    for tag, tag_id in tag2id.items():
        if tag == "O":
            boundary_ids[tag_id] = 0
            continue
        prefix, entity_type = tag.split("-", 1)
        boundary_ids[tag_id] = 1 if prefix == "B" else 2
        type_ids[tag_id] = type2id[entity_type]
        entity_mask[tag_id] = 1.0
    return {
        "entity_types": entity_types,
        "boundary_ids": boundary_ids,
        "type_ids": type_ids,
        "entity_mask": entity_mask,
    }


def build_category_label_bias(train, category2id, tag2id, alpha: float) -> torch.Tensor:
    labels = sorted({tag[2:] for tag in tag2id if tag != "O" and "-" in tag})
    counts: dict[str, Counter] = {cat: Counter() for cat in category2id}
    global_counts = Counter()
    for _, tags, category in train:
        for _, _, label in base.tags_to_spans(tags):
            counts[category][label] += 1
            global_counts[label] += 1
    total_global = sum(global_counts.values()) + alpha * len(labels)
    global_log = {label: torch.log(torch.tensor((global_counts[label] + alpha) / total_global)) for label in labels}
    bias = torch.zeros((len(category2id), len(tag2id)), dtype=torch.float)
    for category, cid in category2id.items():
        total = sum(counts[category].values()) + alpha * len(labels)
        category_log = {label: torch.log(torch.tensor((counts[category][label] + alpha) / total)) for label in labels}
        for tag, tid in tag2id.items():
            if tag == "O" or "-" not in tag:
                continue
            label = tag.split("-", 1)[1]
            bias[cid, tid] = category_log[label] - global_log[label]
    return bias


def build_category_lexicon(train, min_freq: int, min_len: int) -> dict[str, dict[str, dict]]:
    surface_counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for chars, tags, category in train:
        text = "".join(chars)
        for start, end, label in base.tags_to_spans(tags):
            surface = text[start:end]
            if len(surface) < min_len:
                continue
            surface_counts[category][surface][label] += 1
    lexicons: dict[str, dict[str, dict]] = {}
    for category, category_counts in surface_counts.items():
        lexicons[category] = {}
        for surface, counts in category_counts.items():
            total = sum(counts.values())
            if total < min_freq:
                continue
            label, freq = counts.most_common(1)[0]
            lexicons[category][surface] = {"label": label, "freq": freq, "total": total}
    return lexicons


def predict_with_category_lexicon(chars: list[str], category: str, lexicons: dict[str, dict[str, dict]], max_entity_len: int) -> list[str]:
    return base.predict_with_lexicon(chars, lexicons.get(category, {}), max_entity_len)


def category_lexicon_size(lexicons: dict[str, dict[str, dict]]) -> int:
    return sum(len(lexicon) for lexicon in lexicons.values())


def build_category_lexicon_features(
    chars: list[str],
    category: str,
    lexicons: dict[str, dict[str, dict]],
    label2feature: dict[str, int],
    max_entity_len: int,
) -> torch.Tensor:
    feature_size = 4 + len(label2feature)
    features = torch.zeros((len(chars), feature_size), dtype=torch.float)
    if not chars or not lexicons:
        return features
    lexicon = lexicons.get(category, {})
    if not lexicon:
        return features
    text = "".join(chars)
    max_len = max(1, max_entity_len)
    index = 0
    while index < len(chars):
        matched_surface = ""
        matched_info = None
        upper_len = min(max_len, len(chars) - index)
        for length in range(upper_len, 0, -1):
            surface = text[index : index + length]
            if surface in lexicon:
                matched_surface = surface
                matched_info = lexicon[surface]
                break
        if matched_info is None:
            index += 1
            continue
        end = index + len(matched_surface)
        for pos in range(index, end):
            if pos == index:
                features[pos, 0] = 1.0
            else:
                features[pos, 1] = 1.0
            if pos == end - 1:
                features[pos, 2] = 1.0
            features[pos, 3] = min(len(matched_surface), 10) / 10.0
            label = matched_info.get("label")
            if label in label2feature:
                features[pos, 4 + label2feature[label]] = 1.0
        index = end
    return features


class SemanticBertNerDataset(Dataset):
    def __init__(
        self,
        items,
        tokenizer,
        tag2id,
        category2id,
        max_seq_len: int,
        use_category_prefix: bool = False,
        category_lexicons: dict[str, dict[str, dict]] | None = None,
        label2feature: dict[str, int] | None = None,
        max_entity_len: int = 0,
    ):
        self.items = []
        self.lexicon_feature_size = 4 + len(label2feature or {}) if category_lexicons else 0
        cls_id = tokenizer.cls_token_id or tokenizer.convert_tokens_to_ids("[CLS]")
        sep_id = tokenizer.sep_token_id or tokenizer.convert_tokens_to_ids("[SEP]")
        unk_id = tokenizer.unk_token_id or tokenizer.convert_tokens_to_ids("[UNK]")
        o_id = tag2id["O"]
        for chars, tags, category in items:
            prefix_chars = list(f"{category}：") if use_category_prefix else []
            model_chars = prefix_chars + chars
            ids = [cls_id] + [tokenizer.convert_tokens_to_ids(ch) for ch in model_chars] + [sep_id]
            ids = [unk_id if token_id is None or token_id < 0 else token_id for token_id in ids]
            label_ids = [o_id] + [o_id] * len(prefix_chars) + [tag2id[tag] for tag in tags] + [o_id]
            valid_mask = [0] + [0] * len(prefix_chars) + [1] * len(chars) + [0]
            if category_lexicons and label2feature is not None:
                char_features = build_category_lexicon_features(
                    chars,
                    category,
                    category_lexicons,
                    label2feature,
                    max_entity_len,
                )
                prefix_features = torch.zeros((len(prefix_chars), self.lexicon_feature_size), dtype=torch.float)
                special_feature = torch.zeros((1, self.lexicon_feature_size), dtype=torch.float)
                lexicon_features = torch.cat(
                    [special_feature, prefix_features, char_features, special_feature],
                    dim=0,
                )
            else:
                lexicon_features = torch.zeros((len(ids), 0), dtype=torch.float)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                label_ids = label_ids[:max_seq_len]
                valid_mask = valid_mask[:max_seq_len]
                lexicon_features = lexicon_features[:max_seq_len]
            self.items.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor(label_ids, dtype=torch.long),
                    torch.tensor(valid_mask, dtype=torch.bool),
                    torch.tensor(category2id[category], dtype=torch.long),
                    lexicon_features,
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max())
    feature_size = int(batch[0][4].shape[-1])
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    tags = torch.zeros((len(batch), max_len), dtype=torch.long)
    valid_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    category_ids = torch.zeros((len(batch),), dtype=torch.long)
    lexicon_features = torch.zeros((len(batch), max_len, feature_size), dtype=torch.float)
    for i, (ids, label_ids, vmask, category_id, lex_features) in enumerate(batch):
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = True
        tags[i, : len(label_ids)] = label_ids
        valid_mask[i, : len(vmask)] = vmask
        category_ids[i] = category_id
        if feature_size:
            lexicon_features[i, : len(lex_features)] = lex_features
    return input_ids, attention_mask, tags, valid_mask, category_ids, lexicon_features


class SemanticBertTagger(nn.Module):
    def __init__(
        self,
        model_path: str,
        tag_size: int,
        decoupled_tag_maps: dict,
        dropout: float,
        use_crf: bool,
        model_type: str,
        rnn_type: str,
        rnn_hidden: int,
        rnn_attention: bool,
        attention_heads: int,
        rnn_semantic_input_mode: str,
        rnn_semantic_input_scale: float,
        rnn_semantic_init: bool,
        rnn_semantic_bridge_scale: float,
        attention_semantic_memory: bool,
        category_count: int,
        category_emb_dim: int,
        category_fusion_mode: str,
        category_classifier_mode: str,
        category_classifier_scale: float,
        category_factor_rank: int,
        category_uncertainty_gate: bool,
        category_contrastive_weight: float,
        category_contrastive_margin: float,
        category_contrastive_scope: str,
        category_contrastive_wrong_mode: str,
        category_aux_loss_weight: float,
        category_label_bias: torch.Tensor | None,
        category_prior_bias_scale: float,
        lexicon_feature_size: int,
        semantic_adapter_dim: int,
        semantic_adapter_scale: float,
        type_emission_gate_scale: float,
        boundary_aux_loss_weight: float,
        type_aux_loss_weight: float,
        block_adaptive_crf: bool,
        block_crf_rank: int,
        block_crf_scale: float,
        block_crf_start_end: bool,
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
            self.rnn = rnn_cls(hidden, rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
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
        self.category_emb_dim = category_emb_dim
        self.category_fusion_mode = category_fusion_mode
        self.category_embedding = nn.Embedding(category_count, category_emb_dim) if category_emb_dim > 0 else None
        self.rnn_semantic_input_mode = rnn_semantic_input_mode
        self.rnn_semantic_input_scale = rnn_semantic_input_scale
        self.rnn_semantic_init = rnn_semantic_init
        self.rnn_semantic_bridge_scale = rnn_semantic_bridge_scale
        self.attention_semantic_memory = attention_semantic_memory
        if self.category_embedding is not None and self.rnn is not None:
            if rnn_semantic_input_mode == "residual_gate":
                self.rnn_semantic_input_projection = nn.Linear(category_emb_dim, hidden)
                self.rnn_semantic_input_gate = nn.Linear(hidden + category_emb_dim, hidden)
                self.rnn_semantic_input_film = None
                self.rnn_semantic_input_norm = nn.LayerNorm(hidden)
            elif rnn_semantic_input_mode == "film":
                self.rnn_semantic_input_projection = None
                self.rnn_semantic_input_gate = None
                self.rnn_semantic_input_film = nn.Linear(category_emb_dim, hidden * 2)
                self.rnn_semantic_input_norm = nn.LayerNorm(hidden)
                nn.init.zeros_(self.rnn_semantic_input_film.weight)
                nn.init.zeros_(self.rnn_semantic_input_film.bias)
            else:
                self.rnn_semantic_input_projection = None
                self.rnn_semantic_input_gate = None
                self.rnn_semantic_input_film = None
                self.rnn_semantic_input_norm = None
            if rnn_semantic_init:
                self.rnn_semantic_hidden_init = nn.Linear(category_emb_dim, rnn_hidden * 2)
                nn.init.zeros_(self.rnn_semantic_hidden_init.weight)
                nn.init.zeros_(self.rnn_semantic_hidden_init.bias)
                if rnn_type == "lstm":
                    self.rnn_semantic_cell_init = nn.Linear(category_emb_dim, rnn_hidden * 2)
                    nn.init.zeros_(self.rnn_semantic_cell_init.weight)
                    nn.init.zeros_(self.rnn_semantic_cell_init.bias)
                else:
                    self.rnn_semantic_cell_init = None
            else:
                self.rnn_semantic_hidden_init = None
                self.rnn_semantic_cell_init = None
            if rnn_semantic_bridge_scale > 0:
                self.rnn_semantic_bridge_projection = nn.Linear(hidden, out_hidden, bias=False)
                self.rnn_semantic_bridge_gate = nn.Linear(out_hidden + category_emb_dim, out_hidden)
                self.rnn_semantic_bridge_norm = nn.LayerNorm(out_hidden)
                nn.init.normal_(self.rnn_semantic_bridge_projection.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.rnn_semantic_bridge_gate.weight)
                nn.init.constant_(self.rnn_semantic_bridge_gate.bias, -1.0)
            else:
                self.rnn_semantic_bridge_projection = None
                self.rnn_semantic_bridge_gate = None
                self.rnn_semantic_bridge_norm = None
        else:
            self.rnn_semantic_input_projection = None
            self.rnn_semantic_input_gate = None
            self.rnn_semantic_input_film = None
            self.rnn_semantic_input_norm = None
            self.rnn_semantic_hidden_init = None
            self.rnn_semantic_cell_init = None
            self.rnn_semantic_bridge_projection = None
            self.rnn_semantic_bridge_gate = None
            self.rnn_semantic_bridge_norm = None
        if self.category_embedding is not None and self.attention is not None and attention_semantic_memory:
            self.attention_semantic_memory_projection = nn.Linear(category_emb_dim, out_hidden)
            nn.init.normal_(self.attention_semantic_memory_projection.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.attention_semantic_memory_projection.bias)
        else:
            self.attention_semantic_memory_projection = None
        if self.category_embedding is not None and category_fusion_mode in {
            "gate",
            "residual_gate",
            "film",
            "conditional_layer_norm",
        }:
            self.category_projection = nn.Linear(category_emb_dim, out_hidden)
            self.category_gate = nn.Linear(out_hidden + category_emb_dim, out_hidden)
            self.category_gate_norm = nn.LayerNorm(out_hidden)
            self.category_film = nn.Linear(category_emb_dim, out_hidden * 2)
            self.category_film_norm = nn.LayerNorm(out_hidden)
            self.category_cond_norm = nn.LayerNorm(out_hidden, elementwise_affine=False)
            self.category_cond_norm_affine = nn.Linear(category_emb_dim, out_hidden * 2)
            nn.init.zeros_(self.category_film.weight)
            nn.init.zeros_(self.category_film.bias)
            nn.init.zeros_(self.category_cond_norm_affine.weight)
            nn.init.zeros_(self.category_cond_norm_affine.bias)
            classifier_in = out_hidden
        elif self.category_embedding is not None and category_fusion_mode == "concat":
            self.category_projection = None
            self.category_gate = None
            self.category_gate_norm = None
            self.category_film = None
            self.category_film_norm = None
            self.category_cond_norm = None
            self.category_cond_norm_affine = None
            classifier_in = out_hidden + category_emb_dim
        else:
            self.category_projection = None
            self.category_gate = None
            self.category_gate_norm = None
            self.category_film = None
            self.category_film_norm = None
            self.category_cond_norm = None
            self.category_cond_norm_affine = None
            classifier_in = out_hidden
        self.semantic_adapter_scale = semantic_adapter_scale
        if self.category_embedding is not None and semantic_adapter_dim > 0:
            self.semantic_adapter_down = nn.Linear(out_hidden + category_emb_dim, semantic_adapter_dim)
            self.semantic_adapter_up = nn.Linear(semantic_adapter_dim, out_hidden)
            self.semantic_adapter_norm = nn.LayerNorm(out_hidden)
            nn.init.zeros_(self.semantic_adapter_up.weight)
            nn.init.zeros_(self.semantic_adapter_up.bias)
        else:
            self.semantic_adapter_down = None
            self.semantic_adapter_up = None
            self.semantic_adapter_norm = None
        self.lexicon_feature_size = lexicon_feature_size
        classifier_in += lexicon_feature_size
        self.dropout = nn.Dropout(dropout)
        self.category_classifier_mode = category_classifier_mode
        self.category_classifier_scale = category_classifier_scale
        self.category_count = category_count
        self.tag_size = tag_size
        self.decoupled_decoder = category_classifier_mode in {
            "decoupled",
            "decoupled_factorized",
        }
        self.entity_types = list(decoupled_tag_maps["entity_types"])
        self.entity_type_count = len(self.entity_types)
        self.register_buffer(
            "tag_boundary_ids",
            decoupled_tag_maps["boundary_ids"].long(),
        )
        self.register_buffer(
            "tag_type_ids",
            decoupled_tag_maps["type_ids"].long(),
        )
        self.register_buffer(
            "tag_entity_mask",
            decoupled_tag_maps["entity_mask"].float(),
        )
        if self.decoupled_decoder:
            self.classifier = None
            self.boundary_classifier = nn.Linear(classifier_in, 3)
            self.type_classifier = nn.Linear(
                classifier_in,
                self.entity_type_count,
            )
        else:
            self.classifier = nn.Linear(classifier_in, tag_size)
            self.boundary_classifier = None
            self.type_classifier = None
        if category_classifier_mode in {"specific", "residual"}:
            self.category_classifiers = nn.ModuleList([nn.Linear(classifier_in, tag_size) for _ in range(category_count)])
        else:
            self.category_classifiers = None
        if category_classifier_mode in {"factorized", "decoupled_factorized"}:
            factor_output_size = (
                self.entity_type_count
                if category_classifier_mode == "decoupled_factorized"
                else tag_size
            )
            self.category_factor_projection = nn.Linear(classifier_in, category_factor_rank, bias=False)
            self.category_factor_embedding = nn.Embedding(category_count, category_factor_rank)
            self.category_factor_output = nn.Linear(category_factor_rank, factor_output_size, bias=False)
            nn.init.xavier_uniform_(self.category_factor_projection.weight)
            nn.init.xavier_uniform_(self.category_factor_embedding.weight)
            if category_classifier_mode == "decoupled_factorized":
                nn.init.zeros_(self.category_factor_output.weight)
            else:
                nn.init.xavier_uniform_(self.category_factor_output.weight)
        else:
            self.category_factor_projection = None
            self.category_factor_embedding = None
            self.category_factor_output = None
        self.category_factor_rank = category_factor_rank
        self.category_uncertainty_gate = category_uncertainty_gate
        self.category_contrastive_weight = category_contrastive_weight
        self.category_contrastive_margin = category_contrastive_margin
        self.category_contrastive_scope = category_contrastive_scope
        self.category_contrastive_wrong_mode = (
            category_contrastive_wrong_mode
        )
        self.category_aux_loss_weight = category_aux_loss_weight
        self.category_aux_classifier = nn.Linear(out_hidden, category_count) if category_aux_loss_weight > 0 else None
        self.type_emission_gate_scale = type_emission_gate_scale
        self.type_emission_gate = (
            nn.Linear(category_emb_dim, self.entity_type_count)
            if self.category_embedding is not None and type_emission_gate_scale > 0
            else None
        )
        self.boundary_aux_loss_weight = boundary_aux_loss_weight
        self.type_aux_loss_weight = type_aux_loss_weight
        self.boundary_aux_classifier = (
            nn.Linear(classifier_in, 3) if boundary_aux_loss_weight > 0 else None
        )
        self.type_aux_classifier = (
            nn.Linear(classifier_in, self.entity_type_count)
            if type_aux_loss_weight > 0
            else None
        )
        self.use_crf = use_crf
        self.crf = CRF(tag_size, batch_first=True) if use_crf else None
        self.block_adaptive_crf = block_adaptive_crf
        self.block_crf_rank = block_crf_rank
        self.block_crf_scale = block_crf_scale
        self.block_crf_start_end = block_crf_start_end
        if block_adaptive_crf:
            self.block_crf_coeff = nn.Embedding(category_count, block_crf_rank)
            self.block_crf_transition_basis = nn.Parameter(
                torch.empty(block_crf_rank, tag_size, tag_size)
            )
            nn.init.normal_(self.block_crf_coeff.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.block_crf_transition_basis, mean=0.0, std=0.02)
            if block_crf_start_end:
                self.block_crf_start_basis = nn.Parameter(
                    torch.empty(block_crf_rank, tag_size)
                )
                self.block_crf_end_basis = nn.Parameter(
                    torch.empty(block_crf_rank, tag_size)
                )
                nn.init.normal_(self.block_crf_start_basis, mean=0.0, std=0.02)
                nn.init.normal_(self.block_crf_end_basis, mean=0.0, std=0.02)
            else:
                self.block_crf_start_basis = None
                self.block_crf_end_basis = None
        else:
            self.block_crf_coeff = None
            self.block_crf_transition_basis = None
            self.block_crf_start_basis = None
            self.block_crf_end_basis = None
        self.category_prior_bias_scale = category_prior_bias_scale
        if category_label_bias is None:
            self.register_buffer("category_label_bias", torch.zeros((category_count, tag_size), dtype=torch.float))
        else:
            self.register_buffer("category_label_bias", category_label_bias.float())

    def adaptive_crf_parameters(self, category_ids):
        transitions = self.crf.transitions.unsqueeze(0).expand(
            category_ids.shape[0],
            -1,
            -1,
        )
        start_transitions = self.crf.start_transitions.unsqueeze(0).expand(
            category_ids.shape[0],
            -1,
        )
        end_transitions = self.crf.end_transitions.unsqueeze(0).expand(
            category_ids.shape[0],
            -1,
        )
        if not self.block_adaptive_crf:
            return start_transitions, transitions, end_transitions
        coeff = self.block_crf_coeff(category_ids)
        transitions = transitions + self.block_crf_scale * torch.einsum(
            "br,rij->bij",
            coeff,
            self.block_crf_transition_basis,
        )
        if self.block_crf_start_end:
            start_transitions = start_transitions + self.block_crf_scale * torch.einsum(
                "br,ri->bi",
                coeff,
                self.block_crf_start_basis,
            )
            end_transitions = end_transitions + self.block_crf_scale * torch.einsum(
                "br,ri->bi",
                coeff,
                self.block_crf_end_basis,
            )
        return start_transitions, transitions, end_transitions

    def adaptive_crf_score(self, emissions, tags, mask, category_ids):
        mask = mask.bool()
        start_transitions, transitions, end_transitions = self.adaptive_crf_parameters(
            category_ids,
        )
        batch_index = torch.arange(tags.shape[0], device=tags.device)
        score = start_transitions.gather(1, tags[:, :1]).squeeze(1)
        score = score + emissions[batch_index, 0, tags[:, 0]]
        for timestep in range(1, tags.shape[1]):
            prev_tags = tags[:, timestep - 1]
            curr_tags = tags[:, timestep]
            transition_score = transitions[batch_index, prev_tags, curr_tags]
            emission_score = emissions[batch_index, timestep, curr_tags]
            score = score + (transition_score + emission_score) * mask[:, timestep]
        lengths = mask.long().sum(dim=1).clamp_min(1)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + end_transitions.gather(1, last_tags.unsqueeze(1)).squeeze(1)
        return score

    def adaptive_crf_normalizer(self, emissions, mask, category_ids):
        mask = mask.bool()
        start_transitions, transitions, end_transitions = self.adaptive_crf_parameters(
            category_ids,
        )
        score = start_transitions + emissions[:, 0]
        for timestep in range(1, emissions.shape[1]):
            next_score = (
                score.unsqueeze(2)
                + transitions
                + emissions[:, timestep].unsqueeze(1)
            )
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, timestep].unsqueeze(1), next_score, score)
        score = score + end_transitions
        return torch.logsumexp(score, dim=1)

    def adaptive_crf_nll(self, emissions, tags, mask, category_ids):
        numerator = self.adaptive_crf_score(emissions, tags, mask, category_ids)
        denominator = self.adaptive_crf_normalizer(emissions, mask, category_ids)
        return denominator - numerator

    def crf_nll(self, emissions, tags, mask, category_ids):
        if self.block_adaptive_crf:
            return self.adaptive_crf_nll(emissions, tags, mask, category_ids)
        return -self.crf(emissions, tags, mask=mask.byte(), reduction="none")

    def adaptive_crf_decode(self, emissions, mask, category_ids):
        mask = mask.bool()
        start_transitions, transitions, end_transitions = self.adaptive_crf_parameters(
            category_ids,
        )
        decoded = []
        for batch_index in range(emissions.shape[0]):
            seq_len = int(mask[batch_index].long().sum().item())
            seq_len = max(1, seq_len)
            transition = transitions[batch_index]
            score = start_transitions[batch_index] + emissions[batch_index, 0]
            history = []
            for timestep in range(1, seq_len):
                next_score = (
                    score.unsqueeze(1)
                    + transition
                    + emissions[batch_index, timestep].unsqueeze(0)
                )
                best_score, best_path = next_score.max(dim=0)
                history.append(best_path)
                score = best_score
            score = score + end_transitions[batch_index]
            best_last_tag = int(score.argmax().item())
            best_tags = [best_last_tag]
            for best_path in reversed(history):
                best_last_tag = int(best_path[best_tags[-1]].item())
                best_tags.append(best_last_tag)
            best_tags.reverse()
            decoded.append(best_tags)
        return decoded

    def uncertainty_gate(self, emissions):
        if not self.category_uncertainty_gate:
            return None
        base_probabilities = torch.softmax(emissions.detach(), dim=-1)
        entropy = -(
            base_probabilities
            * torch.log(base_probabilities.clamp_min(1e-12))
        ).sum(dim=-1, keepdim=True)
        return entropy / math.log(self.tag_size)

    def type_uncertainty_gate(self, type_logits):
        if not self.category_uncertainty_gate:
            return None
        probabilities = torch.softmax(type_logits.detach(), dim=-1)
        entropy = -(
            probabilities
            * torch.log(probabilities.clamp_min(1e-12))
        ).sum(dim=-1, keepdim=True)
        return entropy / math.log(self.entity_type_count)

    def compose_decoupled_emissions(self, boundary_logits, type_logits):
        boundary_by_tag = boundary_logits.index_select(
            -1,
            self.tag_boundary_ids,
        )
        type_by_tag = type_logits.index_select(
            -1,
            self.tag_type_ids,
        )
        return (
            boundary_by_tag
            + type_by_tag * self.tag_entity_mask.view(1, 1, -1)
        )

    def decoupled_type_logits(self, decoder_input, category_ids):
        type_logits = self.type_classifier(decoder_input)
        if self.category_classifier_mode != "decoupled_factorized":
            return type_logits
        token_factor = torch.tanh(
            self.category_factor_projection(decoder_input)
        )
        category_factor = self.category_factor_embedding(
            category_ids
        ).unsqueeze(1)
        category_type_residual = self.category_factor_output(
            token_factor * category_factor
        )
        category_type_residual = (
            self.category_classifier_scale * category_type_residual
        )
        uncertainty_gate = self.type_uncertainty_gate(type_logits)
        if uncertainty_gate is not None:
            category_type_residual = (
                uncertainty_gate * category_type_residual
            )
        return type_logits + category_type_residual

    def classify(self, decoder_input, category_ids):
        if self.decoupled_decoder:
            boundary_logits = self.boundary_classifier(decoder_input)
            type_logits = self.decoupled_type_logits(
                decoder_input,
                category_ids,
            )
            emissions = self.compose_decoupled_emissions(
                boundary_logits,
                type_logits,
            )
            uncertainty_gate = None
        else:
            emissions = self.classifier(decoder_input)
            uncertainty_gate = self.uncertainty_gate(emissions)
        if not self.decoupled_decoder and self.category_classifiers is not None:
            category_emissions = torch.stack([head(decoder_input) for head in self.category_classifiers], dim=1)
            gather_index = category_ids.view(-1, 1, 1, 1).expand(-1, 1, category_emissions.shape[2], category_emissions.shape[3])
            category_emissions = category_emissions.gather(1, gather_index).squeeze(1)
            if self.category_classifier_mode == "specific":
                emissions = category_emissions
            else:
                category_residual = self.category_classifier_scale * category_emissions
                if uncertainty_gate is not None:
                    category_residual = uncertainty_gate * category_residual
                emissions = emissions + category_residual
        elif not self.decoupled_decoder and self.category_classifier_mode == "factorized":
            token_factor = torch.tanh(self.category_factor_projection(decoder_input))
            category_factor = self.category_factor_embedding(category_ids).unsqueeze(1)
            category_emissions = self.category_factor_output(token_factor * category_factor)
            category_residual = self.category_classifier_scale * category_emissions
            if uncertainty_gate is not None:
                category_residual = uncertainty_gate * category_residual
            emissions = emissions + category_residual
        if self.category_prior_bias_scale:
            category_prior = (
                self.category_prior_bias_scale
                * self.category_label_bias[category_ids].unsqueeze(1)
            )
            if uncertainty_gate is not None:
                category_prior = uncertainty_gate * category_prior
            emissions = emissions + category_prior
        if self.type_emission_gate is not None:
            category_vec = self.category_embedding(category_ids)
            type_gate = torch.tanh(self.type_emission_gate(category_vec))
            tag_gate = type_gate.index_select(-1, self.tag_type_ids)
            tag_gate = tag_gate * self.tag_entity_mask.view(1, -1)
            emissions = emissions + self.type_emission_gate_scale * tag_gate.unsqueeze(1)
        return emissions

    def emissions(
        self,
        input_ids,
        attention_mask,
        category_ids,
        lexicon_features=None,
        return_category_logits: bool = False,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask.long())
        bert_output = outputs[0]
        sequence_output = bert_output
        category_embedding = (
            self.category_embedding(category_ids)
            if self.category_embedding is not None
            else None
        )
        if self.rnn is not None:
            if self.rnn_semantic_input_mode != "none":
                category_input = category_embedding.unsqueeze(1).expand(
                    -1,
                    sequence_output.shape[1],
                    -1,
                )
                if self.rnn_semantic_input_mode == "residual_gate":
                    category_context = self.rnn_semantic_input_projection(category_input)
                    input_gate = torch.sigmoid(
                        self.rnn_semantic_input_gate(
                            torch.cat([sequence_output, category_input], dim=-1)
                        )
                    )
                    sequence_output = self.rnn_semantic_input_norm(
                        sequence_output
                        + self.rnn_semantic_input_scale
                        * input_gate
                        * category_context
                    )
                elif self.rnn_semantic_input_mode == "film":
                    gamma, beta = self.rnn_semantic_input_film(category_input).chunk(
                        2,
                        dim=-1,
                    )
                    sequence_output = self.rnn_semantic_input_norm(
                        sequence_output
                        * (1.0 + self.rnn_semantic_input_scale * gamma)
                        + self.rnn_semantic_input_scale * beta
                    )
            lengths = attention_mask.long().sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(sequence_output, lengths, batch_first=True, enforce_sorted=False)
            initial_state = None
            if self.rnn_semantic_hidden_init is not None:
                initial_hidden = self.rnn_semantic_hidden_init(category_embedding)
                initial_hidden = initial_hidden.view(
                    category_ids.shape[0],
                    2,
                    self.rnn.hidden_size,
                ).transpose(0, 1).contiguous()
                if self.rnn_type == "lstm":
                    initial_cell = self.rnn_semantic_cell_init(category_embedding)
                    initial_cell = initial_cell.view(
                        category_ids.shape[0],
                        2,
                        self.rnn.hidden_size,
                    ).transpose(0, 1).contiguous()
                    initial_state = (initial_hidden, initial_cell)
                else:
                    initial_state = initial_hidden
            if initial_state is None:
                sequence_output, _ = self.rnn(packed)
            else:
                sequence_output, _ = self.rnn(packed, initial_state)
            sequence_output, _ = nn.utils.rnn.pad_packed_sequence(sequence_output, batch_first=True, total_length=input_ids.shape[1])
            if self.rnn_semantic_bridge_projection is not None:
                category_output = category_embedding.unsqueeze(1).expand(
                    -1,
                    sequence_output.shape[1],
                    -1,
                )
                bridge = self.rnn_semantic_bridge_projection(bert_output)
                bridge_gate = torch.sigmoid(
                    self.rnn_semantic_bridge_gate(
                        torch.cat([sequence_output, category_output], dim=-1)
                    )
                )
                sequence_output = self.rnn_semantic_bridge_norm(
                    sequence_output
                    + self.rnn_semantic_bridge_scale * bridge_gate * bridge
                )
        if self.attention is not None:
            if self.attention_semantic_memory_projection is not None:
                semantic_memory = self.attention_semantic_memory_projection(
                    category_embedding
                ).unsqueeze(1)
                attention_memory = torch.cat(
                    [sequence_output, semantic_memory],
                    dim=1,
                )
                memory_padding_mask = torch.cat(
                    [
                        ~attention_mask.bool(),
                        torch.zeros(
                            (attention_mask.shape[0], 1),
                            dtype=torch.bool,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                attended, _ = self.attention(
                    sequence_output,
                    attention_memory,
                    attention_memory,
                    key_padding_mask=memory_padding_mask,
                )
            else:
                attended, _ = self.attention(sequence_output, sequence_output, sequence_output, key_padding_mask=~attention_mask.bool())
            sequence_output = self.attention_norm(sequence_output + attended)
        category_logits = self.category_aux_classifier(self.dropout(sequence_output[:, 0])) if self.category_aux_classifier is not None else None
        if category_embedding is not None:
            category_vec = category_embedding.unsqueeze(1).expand(-1, sequence_output.shape[1], -1)
            if self.semantic_adapter_down is not None:
                adapter_input = torch.cat([sequence_output, category_vec], dim=-1)
                adapter_delta = self.semantic_adapter_up(
                    torch.relu(self.semantic_adapter_down(adapter_input))
                )
                sequence_output = self.semantic_adapter_norm(
                    sequence_output + self.semantic_adapter_scale * adapter_delta
                )
            if self.category_fusion_mode in {"gate", "residual_gate"}:
                category_context = self.category_projection(category_vec)
                gate = torch.sigmoid(self.category_gate(torch.cat([sequence_output, category_vec], dim=-1)))
                if self.category_fusion_mode == "residual_gate":
                    sequence_output = self.category_gate_norm(sequence_output + gate * category_context)
                else:
                    sequence_output = self.category_gate_norm(gate * sequence_output + (1.0 - gate) * category_context)
            elif self.category_fusion_mode == "film":
                gamma, beta = self.category_film(category_vec).chunk(2, dim=-1)
                sequence_output = self.category_film_norm(
                    sequence_output * (1.0 + gamma) + beta
                )
            elif self.category_fusion_mode == "conditional_layer_norm":
                gamma, beta = self.category_cond_norm_affine(category_vec).chunk(2, dim=-1)
                sequence_output = self.category_cond_norm(sequence_output)
                sequence_output = sequence_output * (1.0 + gamma) + beta
            elif self.category_fusion_mode == "concat":
                sequence_output = torch.cat([sequence_output, category_vec], dim=-1)
        if lexicon_features is not None and self.lexicon_feature_size:
            sequence_output = torch.cat([sequence_output, lexicon_features.float()], dim=-1)
        dropped = self.dropout(sequence_output)
        emissions = self.classify(dropped, category_ids)
        if return_category_logits:
            return emissions, category_logits, dropped
        return emissions

    def loss(
        self,
        input_ids,
        attention_mask,
        tags,
        category_ids,
        lexicon_features,
        criterion,
        return_emissions: bool = False,
    ):
        emissions, category_logits, decoder_input = self.emissions(
            input_ids,
            attention_mask,
            category_ids,
            lexicon_features,
            return_category_logits=True,
        )
        if self.use_crf:
            sequence_nll = self.crf_nll(
                emissions,
                tags,
                attention_mask,
                category_ids,
            )
            loss = sequence_nll.mean()
        else:
            ce_tags = tags.masked_fill(~attention_mask, -100)
            loss = criterion(emissions.reshape(-1, emissions.shape[-1]), ce_tags.reshape(-1))
        if (
            self.category_contrastive_weight > 0
            and self.category_contrastive_scope == "entity_type"
        ):
            correct_type_logits = self.decoupled_type_logits(
                decoder_input,
                category_ids,
            )
            gold_type_ids = self.tag_type_ids[tags]
            correct_type_nll = -torch.log_softmax(
                correct_type_logits,
                dim=-1,
            ).gather(-1, gold_type_ids.unsqueeze(-1)).squeeze(-1)
            wrong_type_nlls = []
            if self.category_contrastive_wrong_mode == "all":
                wrong_category_batches = [
                    (category_ids + offset) % self.category_count
                    for offset in range(1, self.category_count)
                ]
            else:
                offsets = torch.randint(
                    1,
                    self.category_count,
                    category_ids.shape,
                    device=category_ids.device,
                )
                wrong_category_batches = [
                    (category_ids + offsets) % self.category_count
                ]
            for wrong_category_ids in wrong_category_batches:
                wrong_type_logits = self.decoupled_type_logits(
                    decoder_input,
                    wrong_category_ids,
                )
                wrong_type_nlls.append(
                    -torch.log_softmax(
                        wrong_type_logits,
                        dim=-1,
                    ).gather(
                        -1,
                        gold_type_ids.unsqueeze(-1),
                    ).squeeze(-1)
                )
            hardest_wrong_type_nll = torch.stack(
                wrong_type_nlls,
                dim=0,
            ).amin(dim=0)
            entity_token_mask = (
                self.tag_entity_mask[tags] * attention_mask.float()
            )
            token_contrastive_loss = nn.functional.relu(
                self.category_contrastive_margin
                + correct_type_nll
                - hardest_wrong_type_nll
            )
            contrastive_loss = (
                token_contrastive_loss * entity_token_mask
            ).sum() / entity_token_mask.sum().clamp_min(1.0)
            loss = (
                loss
                + self.category_contrastive_weight * contrastive_loss
            )
        elif self.category_contrastive_weight > 0:
            offsets = torch.randint(
                1,
                self.category_count,
                category_ids.shape,
                device=category_ids.device,
            )
            wrong_category_ids = (
                category_ids + offsets
            ) % self.category_count
            wrong_emissions = self.classify(
                decoder_input,
                wrong_category_ids,
            )
            wrong_sequence_nll = self.crf_nll(
                wrong_emissions,
                tags,
                attention_mask,
                wrong_category_ids,
            )
            sequence_lengths = attention_mask.sum(dim=1).clamp_min(1)
            normalized_sequence_nll = sequence_nll / sequence_lengths
            normalized_wrong_sequence_nll = (
                wrong_sequence_nll / sequence_lengths
            )
            contrastive_loss = nn.functional.relu(
                self.category_contrastive_margin
                + normalized_sequence_nll
                - normalized_wrong_sequence_nll
            ).mean()
            loss = loss + self.category_contrastive_weight * contrastive_loss
        if category_logits is not None:
            loss = loss + self.category_aux_loss_weight * nn.functional.cross_entropy(category_logits, category_ids)
        if self.boundary_aux_classifier is not None:
            boundary_logits = self.boundary_aux_classifier(decoder_input)
            boundary_targets = self.tag_boundary_ids[tags].masked_fill(
                ~attention_mask,
                -100,
            )
            boundary_loss = nn.functional.cross_entropy(
                boundary_logits.reshape(-1, 3),
                boundary_targets.reshape(-1),
                ignore_index=-100,
            )
            loss = loss + self.boundary_aux_loss_weight * boundary_loss
        if self.type_aux_classifier is not None:
            type_logits = self.type_aux_classifier(decoder_input)
            entity_token_mask = (
                (self.tag_entity_mask[tags] > 0) & attention_mask
            )
            type_targets = self.tag_type_ids[tags].masked_fill(
                ~entity_token_mask,
                -100,
            )
            if entity_token_mask.any():
                type_loss = nn.functional.cross_entropy(
                    type_logits.reshape(-1, self.entity_type_count),
                    type_targets.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + self.type_aux_loss_weight * type_loss
        if return_emissions:
            return loss, emissions
        return loss

    def decode(self, input_ids, attention_mask, category_ids, lexicon_features=None):
        emissions = self.emissions(input_ids, attention_mask, category_ids, lexicon_features)
        if self.use_crf:
            if self.block_adaptive_crf:
                return self.adaptive_crf_decode(emissions, attention_mask, category_ids)
            return self.crf.decode(emissions, mask=attention_mask.byte())
        return emissions.argmax(-1).cpu().tolist()


def decode_batch(model, loader, id2tag, device, category_offset: int = 0):
    model.eval()
    pred_tags = []
    with torch.no_grad():
        for input_ids, attention_mask, _, valid_mask, category_ids, lexicon_features in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            category_ids = category_ids.to(device)
            lexicon_features = lexicon_features.to(device)
            if category_offset:
                category_ids = (category_ids + category_offset) % model.category_count
            decoded = model.decode(input_ids, attention_mask, category_ids, lexicon_features)
            vmask = valid_mask.cpu().tolist()
            for row, mask_row in zip(decoded, vmask):
                pred_tags.append(base.normalize_tags([id2tag[idx] for idx, keep in zip(row, mask_row) if keep]))
    return pred_tags


def symmetric_kl_loss(emissions_a, emissions_b, attention_mask):
    log_prob_a = torch.log_softmax(emissions_a, dim=-1)
    log_prob_b = torch.log_softmax(emissions_b, dim=-1)
    prob_a = log_prob_a.exp().detach()
    prob_b = log_prob_b.exp().detach()
    kl_ab = nn.functional.kl_div(log_prob_a, prob_b, reduction="none").sum(dim=-1)
    kl_ba = nn.functional.kl_div(log_prob_b, prob_a, reduction="none").sum(dim=-1)
    mask = attention_mask.float()
    return ((kl_ab + kl_ba) * mask).sum() / mask.sum().clamp_min(1.0) / 2.0


def apply_fgm_perturbation(model, epsilon: float):
    if epsilon <= 0:
        return []
    backups = []
    for name, parameter in model.named_parameters():
        if (
            parameter.requires_grad
            and parameter.grad is not None
            and "word_embeddings" in name
        ):
            norm = torch.norm(parameter.grad)
            if torch.isfinite(norm).item() and norm.item() > 0:
                backups.append((parameter, parameter.data.detach().clone()))
                parameter.data.add_(epsilon * parameter.grad / norm)
    return backups


def restore_fgm_perturbation(backups):
    for parameter, value in backups:
        parameter.data.copy_(value)


def prediction_change_stats(reference_tags, perturbed_tags) -> dict:
    changed_sentences = 0
    changed_entity_sets = 0
    changed_tokens = 0
    total_tokens = 0
    for reference, perturbed in zip(reference_tags, perturbed_tags):
        if reference != perturbed:
            changed_sentences += 1
        if set(base.tags_to_spans(reference)) != set(base.tags_to_spans(perturbed)):
            changed_entity_sets += 1
        for reference_tag, perturbed_tag in zip(reference, perturbed):
            changed_tokens += int(reference_tag != perturbed_tag)
            total_tokens += 1
    sentence_count = len(reference_tags)
    return {
        "changed_sentences": changed_sentences,
        "sentence_count": sentence_count,
        "sentence_change_rate": changed_sentences / sentence_count if sentence_count else 0.0,
        "changed_entity_sets": changed_entity_sets,
        "entity_set_change_rate": changed_entity_sets / sentence_count if sentence_count else 0.0,
        "changed_tokens": changed_tokens,
        "token_count": total_tokens,
        "token_change_rate": changed_tokens / total_tokens if total_tokens else 0.0,
    }


def category_sequence_score_diagnostics(model, loader, device) -> dict:
    if not model.use_crf or model.category_count < 2:
        return {}
    correct_scores = []
    wrong_scores_by_sample: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, tags, _, category_ids, lexicon_features in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags = tags.to(device)
            category_ids = category_ids.to(device)
            lexicon_features = lexicon_features.to(device)
            lengths = attention_mask.sum(dim=1).clamp_min(1)
            correct_emissions = model.emissions(
                input_ids,
                attention_mask,
                category_ids,
                lexicon_features,
            )
            correct_nll = model.crf_nll(
                correct_emissions,
                tags,
                attention_mask,
                category_ids,
            )
            correct_nll = (correct_nll / lengths).cpu().tolist()
            batch_wrong = [[] for _ in correct_nll]
            for category_offset in range(1, model.category_count):
                wrong_category_ids = (
                    category_ids + category_offset
                ) % model.category_count
                wrong_emissions = model.emissions(
                    input_ids,
                    attention_mask,
                    wrong_category_ids,
                    lexicon_features,
                )
                wrong_nll = model.crf_nll(
                    wrong_emissions,
                    tags,
                    attention_mask,
                    wrong_category_ids,
                )
                wrong_nll = (wrong_nll / lengths).cpu().tolist()
                for sample_index, score in enumerate(wrong_nll):
                    batch_wrong[sample_index].append(score)
            correct_scores.extend(correct_nll)
            wrong_scores_by_sample.extend(batch_wrong)

    wrong_means = [
        sum(scores) / len(scores)
        for scores in wrong_scores_by_sample
    ]
    wrong_mins = [min(scores) for scores in wrong_scores_by_sample]
    gains = [
        wrong_mean - correct
        for correct, wrong_mean in zip(correct_scores, wrong_means)
    ]
    sample_count = len(correct_scores)
    return {
        "samples": sample_count,
        "mean_correct_nll_per_token": (
            sum(correct_scores) / sample_count
            if sample_count
            else 0.0
        ),
        "mean_wrong_nll_per_token": (
            sum(wrong_means) / sample_count
            if sample_count
            else 0.0
        ),
        "mean_correct_score_gain_vs_wrong": (
            sum(gains) / sample_count
            if sample_count
            else 0.0
        ),
        "correct_preferred_over_mean_wrong_rate": (
            sum(
                correct < wrong_mean
                for correct, wrong_mean in zip(correct_scores, wrong_means)
            )
            / sample_count
            if sample_count
            else 0.0
        ),
        "correct_preferred_over_all_wrong_rate": (
            sum(
                correct < wrong_min
                for correct, wrong_min in zip(correct_scores, wrong_mins)
            )
            / sample_count
            if sample_count
            else 0.0
        ),
    }


def category_type_score_diagnostics(model, loader, device) -> dict:
    if not model.decoupled_decoder or model.category_count < 2:
        return {}
    correct_values = []
    wrong_means = []
    wrong_mins = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, tags, _, category_ids, lexicon_features in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags = tags.to(device)
            category_ids = category_ids.to(device)
            lexicon_features = lexicon_features.to(device)
            _, _, decoder_input = model.emissions(
                input_ids,
                attention_mask,
                category_ids,
                lexicon_features,
                return_category_logits=True,
            )
            gold_type_ids = model.tag_type_ids[tags]
            entity_mask = (
                model.tag_entity_mask[tags] > 0
            ) & attention_mask
            correct_logits = model.decoupled_type_logits(
                decoder_input,
                category_ids,
            )
            correct_nll = -torch.log_softmax(
                correct_logits,
                dim=-1,
            ).gather(-1, gold_type_ids.unsqueeze(-1)).squeeze(-1)
            wrong_nll_batches = []
            for category_offset in range(1, model.category_count):
                wrong_category_ids = (
                    category_ids + category_offset
                ) % model.category_count
                wrong_logits = model.decoupled_type_logits(
                    decoder_input,
                    wrong_category_ids,
                )
                wrong_nll_batches.append(
                    -torch.log_softmax(
                        wrong_logits,
                        dim=-1,
                    ).gather(
                        -1,
                        gold_type_ids.unsqueeze(-1),
                    ).squeeze(-1)
                )
            stacked_wrong = torch.stack(wrong_nll_batches, dim=0)
            correct_values.extend(
                correct_nll[entity_mask].cpu().tolist()
            )
            wrong_means.extend(
                stacked_wrong.mean(dim=0)[entity_mask].cpu().tolist()
            )
            wrong_mins.extend(
                stacked_wrong.amin(dim=0)[entity_mask].cpu().tolist()
            )
    token_count = len(correct_values)
    gains = [
        wrong - correct
        for correct, wrong in zip(correct_values, wrong_means)
    ]
    return {
        "entity_tokens": token_count,
        "mean_correct_type_nll": (
            sum(correct_values) / token_count if token_count else 0.0
        ),
        "mean_wrong_type_nll": (
            sum(wrong_means) / token_count if token_count else 0.0
        ),
        "mean_correct_type_score_gain_vs_wrong": (
            sum(gains) / token_count if token_count else 0.0
        ),
        "correct_preferred_over_mean_wrong_rate": (
            sum(
                correct < wrong
                for correct, wrong in zip(correct_values, wrong_means)
            )
            / token_count
            if token_count
            else 0.0
        ),
        "correct_preferred_over_all_wrong_rate": (
            sum(
                correct < wrong
                for correct, wrong in zip(correct_values, wrong_mins)
            )
            / token_count
            if token_count
            else 0.0
        ),
    }


def entity_records(chars: list[str], tags: list[str]) -> list[dict]:
    text = "".join(chars)
    return [
        {"start": start, "end": end, "label": label, "text": text[start:end]}
        for start, end, label in base.tags_to_spans(tags)
    ]


def save_prediction_records(path: Path, items, pred_tags, metas: list[dict] | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        for index, ((chars, gold_tags, category), predicted_tags) in enumerate(zip(items, pred_tags)):
            row = {
                "index": index,
                "text": "".join(chars),
                "category": category,
                "gold_tags": gold_tags,
                "predicted_tags": predicted_tags,
                "gold_entities": entity_records(chars, gold_tags),
                "predicted_entities": entity_records(chars, predicted_tags),
            }
            if metas is not None:
                meta = metas[index]
                row.update(
                    {
                        "sample_id": meta.get("sample_id", ""),
                        "db_name": meta.get("db_name", ""),
                        "source_dataset": meta.get("dataset", ""),
                    }
                )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_by_category(items, pred_tags):
    out = {}
    grouped_gold: dict[str, list[tuple[list[str], list[str]]]] = defaultdict(list)
    grouped_pred: dict[str, list[list[str]]] = defaultdict(list)
    for (chars, tags, category), pred in zip(items, pred_tags):
        grouped_gold[category].append((chars, tags))
        grouped_pred[category].append(pred)
    for category in sorted(grouped_gold):
        out[category] = base.evaluate_entities(grouped_gold[category], grouped_pred[category])
    return out


def sample_train_fraction(train, fraction: float, seed: int):
    if fraction >= 1.0:
        return train
    sample_size = max(1, int(len(train) * fraction))
    rng = random.Random(seed)
    indices = list(range(len(train)))
    rng.shuffle(indices)
    keep = set(indices[:sample_size])
    return [sent for idx, sent in enumerate(train) if idx in keep]


def train_one_dataset(args, dataset_name: str) -> dict:
    train, dev, test = load_semantic_dataset(Path(args.data_root), dataset_name, args.max_seq_len - 2)
    raw_test_metas = read_meta(
        Path(args.data_root) / dataset_name / "test_meta.jsonl"
    )
    test_metas = raw_test_metas if len(raw_test_metas) == len(test) else None
    original_train_sentences = len(train)
    train = sample_train_fraction(train, args.train_fraction, args.train_sample_seed)
    plain_train = strip_category(train)
    plain_dev = strip_category(dev)
    plain_test = strip_category(test)
    tag2id = base.build_tag_vocab(plain_train)
    id2tag = {i: tag for tag, i in tag2id.items()}
    decoupled_tag_maps = build_decoupled_tag_maps(tag2id)
    category2id = build_category_vocab(train)
    unseen_tags = {
        tag
        for _, tags in plain_dev + plain_test
        for tag in tags
        if tag not in tag2id
    }
    unseen_categories = {
        category
        for _, _, category in dev + test
        if category not in category2id
    }
    if unseen_tags:
        raise ValueError(f"dev/test contain unseen BIO tags: {sorted(unseen_tags)}")
    if unseen_categories:
        raise ValueError(
            f"dev/test contain unseen semantic blocks: {sorted(unseen_categories)}"
        )
    from transformers import AutoTokenizer, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(args.model_path) if args.tokenizer_type == "bert" else AutoTokenizer.from_pretrained(args.model_path)
    category_bias = build_category_label_bias(train, category2id, tag2id, args.category_prior_alpha)
    category_lexicons_for_features = None
    lexicon_feature_size = 0
    lexicon_feature_max_entity_len = 0
    label2feature = {
        label: index
        for index, label in enumerate(decoupled_tag_maps["entity_types"])
    }
    if args.category_lexicon_train_feature:
        category_lexicons_for_features = build_category_lexicon(
            train,
            args.lexicon_min_freq,
            args.lexicon_min_len,
        )
        lexicon_feature_size = 4 + len(label2feature)
        lexicon_feature_max_entity_len = max(
            [
                len(surface)
                for lexicon in category_lexicons_for_features.values()
                for surface in lexicon
            ]
            or [1]
        )
        if args.lexicon_max_entity_len:
            lexicon_feature_max_entity_len = min(
                lexicon_feature_max_entity_len,
                args.lexicon_max_entity_len,
            )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SemanticBertTagger(
        args.model_path,
        len(tag2id),
        decoupled_tag_maps,
        args.dropout,
        args.use_crf,
        args.model_type,
        args.rnn_type,
        args.rnn_hidden,
        args.rnn_attention,
        args.attention_heads,
        args.rnn_semantic_input_mode,
        args.rnn_semantic_input_scale,
        args.rnn_semantic_init,
        args.rnn_semantic_bridge_scale,
        args.attention_semantic_memory,
        len(category2id),
        args.category_emb_dim if args.use_category_feature else 0,
        args.category_fusion_mode,
        args.category_classifier_mode,
        args.category_classifier_scale,
        args.category_factor_rank,
        args.category_uncertainty_gate,
        args.category_contrastive_weight,
        args.category_contrastive_margin,
        args.category_contrastive_scope,
        args.category_contrastive_wrong_mode,
        args.category_aux_loss_weight,
        category_bias,
        args.category_prior_bias_scale,
        lexicon_feature_size,
        args.semantic_adapter_dim,
        args.semantic_adapter_scale,
        args.type_emission_gate_scale,
        args.boundary_aux_loss_weight,
        args.type_aux_loss_weight,
        args.block_adaptive_crf,
        args.block_crf_rank,
        args.block_crf_scale,
        args.block_crf_start_end,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    train_loader = DataLoader(
        SemanticBertNerDataset(
            train,
            tokenizer,
            tag2id,
            category2id,
            args.max_seq_len,
            args.use_category_prefix,
            category_lexicons_for_features,
            label2feature,
            lexicon_feature_max_entity_len,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = (
        DataLoader(
            SemanticBertNerDataset(
                dev,
                tokenizer,
                tag2id,
                category2id,
                args.max_seq_len,
                args.use_category_prefix,
                category_lexicons_for_features,
                label2feature,
                lexicon_feature_max_entity_len,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        if dev
        else None
    )
    test_loader = DataLoader(
        SemanticBertNerDataset(
            test,
            tokenizer,
            tag2id,
            category2id,
            args.max_seq_len,
            args.use_category_prefix,
            category_lexicons_for_features,
            label2feature,
            lexicon_feature_max_entity_len,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    history = []
    best_state = None
    best_dev_f1 = -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for input_ids, attention_mask, tags, _, category_ids, lexicon_features in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags = tags.to(device)
            category_ids = category_ids.to(device)
            lexicon_features = lexicon_features.to(device)
            optimizer.zero_grad()
            if args.rdrop_alpha > 0:
                loss_a, emissions_a = model.loss(
                    input_ids,
                    attention_mask,
                    tags,
                    category_ids,
                    lexicon_features,
                    criterion,
                    return_emissions=True,
                )
                loss_b, emissions_b = model.loss(
                    input_ids,
                    attention_mask,
                    tags,
                    category_ids,
                    lexicon_features,
                    criterion,
                    return_emissions=True,
                )
                loss = (
                    (loss_a + loss_b) / 2.0
                    + args.rdrop_alpha
                    * symmetric_kl_loss(emissions_a, emissions_b, attention_mask)
                )
            else:
                loss = model.loss(
                    input_ids,
                    attention_mask,
                    tags,
                    category_ids,
                    lexicon_features,
                    criterion,
                )
            loss.backward()
            backups = apply_fgm_perturbation(model, args.fgm_epsilon)
            if backups:
                adv_loss = model.loss(
                    input_ids,
                    attention_mask,
                    tags,
                    category_ids,
                    lexicon_features,
                    criterion,
                )
                adv_loss.backward()
                restore_fgm_perturbation(backups)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        row = {"epoch": epoch, "loss": total_loss / max(1, steps)}
        if dev_loader:
            pred = decode_batch(model, dev_loader, id2tag, device)
            dev_metrics = base.evaluate_entities(plain_dev, pred)
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
        "entity_types": decoupled_tag_maps["entity_types"],
        "category2id": category2id,
        "device": str(device),
        "use_crf": args.use_crf,
        "model_path": args.model_path,
        "model_type": args.model_type,
        "tokenizer_type": args.tokenizer_type,
        "rnn_type": args.rnn_type,
        "rnn_hidden": args.rnn_hidden,
        "rnn_attention": args.rnn_attention,
        "attention_heads": args.attention_heads,
        "rnn_semantic_input_mode": args.rnn_semantic_input_mode,
        "rnn_semantic_input_scale": args.rnn_semantic_input_scale,
        "rnn_semantic_init": args.rnn_semantic_init,
        "rnn_semantic_bridge_scale": args.rnn_semantic_bridge_scale,
        "attention_semantic_memory": args.attention_semantic_memory,
        "use_category_feature": args.use_category_feature,
        "use_category_prefix": args.use_category_prefix,
        "category_emb_dim": args.category_emb_dim if args.use_category_feature else 0,
        "category_fusion_mode": args.category_fusion_mode if args.use_category_feature else "none",
        "category_classifier_mode": args.category_classifier_mode,
        "category_classifier_scale": args.category_classifier_scale,
        "category_factor_rank": args.category_factor_rank,
        "category_uncertainty_gate": args.category_uncertainty_gate,
        "category_contrastive_weight": args.category_contrastive_weight,
        "category_contrastive_margin": args.category_contrastive_margin,
        "category_contrastive_scope": args.category_contrastive_scope,
        "category_contrastive_wrong_mode": (
            args.category_contrastive_wrong_mode
        ),
        "category_aux_loss_weight": args.category_aux_loss_weight,
        "category_prior_bias_scale": args.category_prior_bias_scale,
        "category_prior_alpha": args.category_prior_alpha,
        "semantic_adapter_dim": args.semantic_adapter_dim,
        "semantic_adapter_scale": args.semantic_adapter_scale,
        "type_emission_gate_scale": args.type_emission_gate_scale,
        "boundary_aux_loss_weight": args.boundary_aux_loss_weight,
        "type_aux_loss_weight": args.type_aux_loss_weight,
        "fgm_epsilon": args.fgm_epsilon,
        "rdrop_alpha": args.rdrop_alpha,
        "block_adaptive_crf": args.block_adaptive_crf,
        "block_crf_rank": args.block_crf_rank,
        "block_crf_scale": args.block_crf_scale,
        "block_crf_start_end": args.block_crf_start_end,
        "category_lexicon_train_feature": args.category_lexicon_train_feature,
        "lexicon_feature_size": lexicon_feature_size,
        "lexicon_feature_max_entity_len": lexicon_feature_max_entity_len,
        "lexicon_feature_size_by_category": (
            {category: len(lexicon) for category, lexicon in category_lexicons_for_features.items()}
            if category_lexicons_for_features
            else {}
        ),
        "history": history,
        "test_entity": base.evaluate_entities(plain_test, pred),
        "test_entity_by_category": evaluate_by_category(test, pred),
        "test_token": base.evaluate_tokens(plain_test, pred),
        "seconds": time.time() - start,
    }
    if args.evaluate_category_perturbations:
        category_perturbations = {}
        for category_offset in range(1, len(category2id)):
            perturbed_pred = decode_batch(
                model,
                test_loader,
                id2tag,
                device,
                category_offset=category_offset,
            )
            category_perturbations[f"offset_{category_offset}"] = {
                "test_entity": base.evaluate_entities(plain_test, perturbed_pred),
                "test_entity_by_category": evaluate_by_category(test, perturbed_pred),
                "prediction_change": prediction_change_stats(pred, perturbed_pred),
            }
        run["category_perturbations"] = category_perturbations
        run["category_sequence_score_diagnostics"] = (
            category_sequence_score_diagnostics(
                model,
                test_loader,
                device,
            )
        )
        run["category_type_score_diagnostics"] = (
            category_type_score_diagnostics(
                model,
                test_loader,
                device,
            )
        )
    if args.save_predictions:
        prediction_path = Path(args.out_dir) / f"{dataset_name.lower()}_test_predictions.jsonl"
        save_prediction_records(prediction_path, test, pred, test_metas)
        run["test_predictions_path"] = str(prediction_path)
        if dev_loader:
            dev_prediction_path = Path(args.out_dir) / f"{dataset_name.lower()}_dev_predictions.jsonl"
            dev_pred = decode_batch(model, dev_loader, id2tag, device)
            raw_dev_metas = read_meta(
                Path(args.data_root) / dataset_name / "dev_meta.jsonl"
            )
            dev_metas = raw_dev_metas if len(raw_dev_metas) == len(dev) else None
            save_prediction_records(dev_prediction_path, dev, dev_pred, dev_metas)
            run["dev_predictions_path"] = str(dev_prediction_path)
    if args.lexicon_fusion:
        lexicon = base.build_lexicon(plain_train, args.lexicon_min_freq, args.lexicon_min_len)
        max_entity_len = max([len(x) for x in lexicon] or [1])
        if args.lexicon_max_entity_len:
            max_entity_len = min(max_entity_len, args.lexicon_max_entity_len)
        lex_test_pred = [base.predict_with_lexicon(chars, lexicon, max_entity_len) for chars, _ in plain_test]
        run["lexicon_only"] = {
            "lexicon_size": len(lexicon),
            "max_entity_len": max_entity_len,
            "test_entity": base.evaluate_entities(plain_test, lex_test_pred),
            "test_token": base.evaluate_tokens(plain_test, lex_test_pred),
        }
        if dev and dev_loader:
            bert_dev_pred = decode_batch(model, dev_loader, id2tag, device)
            lex_dev_pred = [base.predict_with_lexicon(chars, lexicon, max_entity_len) for chars, _ in plain_dev]
            label_source, source_detail = base.select_label_sources(plain_dev, bert_dev_pred, lex_dev_pred, args.fusion_margin)
            fused_dev_pred = [base.fuse_by_label(bert_tags, lex_tags, label_source) for bert_tags, lex_tags in zip(bert_dev_pred, lex_dev_pred)]
            fused_test_pred = [base.fuse_by_label(bert_tags, lex_tags, label_source) for bert_tags, lex_tags in zip(pred, lex_test_pred)]
            dev_candidates = {
                "bert": (base.evaluate_entities(plain_dev, bert_dev_pred), pred),
                "lexicon": (base.evaluate_entities(plain_dev, lex_dev_pred), lex_test_pred),
                "per_label_fusion": (base.evaluate_entities(plain_dev, fused_dev_pred), fused_test_pred),
            }
            selected_name, (selected_dev_metrics, selected_test_pred) = max(
                dev_candidates.items(),
                key=lambda item: (item[1][0]["f1"], item[1][0]["precision"], item[1][0]["recall"]),
            )
            run["lexicon_fusion"] = {
                "strategy": "per_label_dev_f1_source_selection",
                "fusion_margin": args.fusion_margin,
                "label_source": source_detail,
                "test_entity": base.evaluate_entities(plain_test, fused_test_pred),
                "test_token": base.evaluate_tokens(plain_test, fused_test_pred),
            }
            run["adaptive_dev_selection"] = {
                "strategy": "select_bert_lexicon_or_per_label_fusion_by_dev_f1",
                "selected": selected_name,
                "dev_candidates": {name: metrics for name, (metrics, _) in dev_candidates.items()},
                "selected_dev_entity": selected_dev_metrics,
                "test_entity": base.evaluate_entities(plain_test, selected_test_pred),
                "test_token": base.evaluate_tokens(plain_test, selected_test_pred),
            }
    if args.category_lexicon_fusion:
        lexicons = build_category_lexicon(train, args.lexicon_min_freq, args.lexicon_min_len)
        max_entity_len = max([len(surface) for lexicon in lexicons.values() for surface in lexicon] or [1])
        if args.lexicon_max_entity_len:
            max_entity_len = min(max_entity_len, args.lexicon_max_entity_len)
        lex_test_pred = [predict_with_category_lexicon(chars, category, lexicons, max_entity_len) for chars, _, category in test]
        run["category_lexicon_only"] = {
            "lexicon_size": category_lexicon_size(lexicons),
            "lexicon_size_by_category": {category: len(lexicon) for category, lexicon in lexicons.items()},
            "max_entity_len": max_entity_len,
            "test_entity": base.evaluate_entities(plain_test, lex_test_pred),
            "test_token": base.evaluate_tokens(plain_test, lex_test_pred),
        }
        if dev and dev_loader:
            bert_dev_pred = decode_batch(model, dev_loader, id2tag, device)
            lex_dev_pred = [predict_with_category_lexicon(chars, category, lexicons, max_entity_len) for chars, _, category in dev]
            label_source, source_detail = base.select_label_sources(plain_dev, bert_dev_pred, lex_dev_pred, args.fusion_margin)
            fused_dev_pred = [base.fuse_by_label(bert_tags, lex_tags, label_source) for bert_tags, lex_tags in zip(bert_dev_pred, lex_dev_pred)]
            fused_test_pred = [base.fuse_by_label(bert_tags, lex_tags, label_source) for bert_tags, lex_tags in zip(pred, lex_test_pred)]
            dev_candidates = {
                "bert": (base.evaluate_entities(plain_dev, bert_dev_pred), pred),
                "category_lexicon": (base.evaluate_entities(plain_dev, lex_dev_pred), lex_test_pred),
                "per_label_category_lexicon_fusion": (base.evaluate_entities(plain_dev, fused_dev_pred), fused_test_pred),
            }
            selected_name, (selected_dev_metrics, selected_test_pred) = max(
                dev_candidates.items(),
                key=lambda item: (item[1][0]["f1"], item[1][0]["precision"], item[1][0]["recall"]),
            )
            run["category_lexicon_fusion"] = {
                "strategy": "per_label_dev_f1_source_selection_with_category_lexicon",
                "fusion_margin": args.fusion_margin,
                "label_source": source_detail,
                "test_entity": base.evaluate_entities(plain_test, fused_test_pred),
                "test_token": base.evaluate_tokens(plain_test, fused_test_pred),
            }
            run["category_adaptive_dev_selection"] = {
                "strategy": "select_bert_category_lexicon_or_per_label_fusion_by_dev_f1",
                "selected": selected_name,
                "dev_candidates": {name: metrics for name, (metrics, _) in dev_candidates.items()},
                "selected_dev_entity": selected_dev_metrics,
                "test_entity": base.evaluate_entities(plain_test, selected_test_pred),
                "test_token": base.evaluate_tokens(plain_test, selected_test_pred),
            }
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
    parser.add_argument(
        "--rnn-semantic-input-mode",
        choices=["none", "residual_gate", "film"],
        default="none",
    )
    parser.add_argument("--rnn-semantic-input-scale", type=float, default=1.0)
    parser.add_argument("--rnn-semantic-init", action="store_true")
    parser.add_argument("--rnn-semantic-bridge-scale", type=float, default=0.0)
    parser.add_argument("--attention-semantic-memory", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["SELF_GEO_NER"])
    parser.add_argument("--epochs", type=int, default=10)
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
    parser.add_argument("--use-category-feature", action="store_true")
    parser.add_argument("--use-category-prefix", action="store_true")
    parser.add_argument("--category-emb-dim", type=int, default=32)
    parser.add_argument(
        "--category-fusion-mode",
        choices=[
            "concat",
            "none",
            "gate",
            "residual_gate",
            "film",
            "conditional_layer_norm",
        ],
        default="concat",
    )
    parser.add_argument(
        "--category-classifier-mode",
        choices=[
            "shared",
            "specific",
            "residual",
            "factorized",
            "decoupled",
            "decoupled_factorized",
        ],
        default="shared",
    )
    parser.add_argument("--category-classifier-scale", type=float, default=0.5)
    parser.add_argument("--category-factor-rank", type=int, default=8)
    parser.add_argument("--category-uncertainty-gate", action="store_true")
    parser.add_argument("--category-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--category-contrastive-margin", type=float, default=0.1)
    parser.add_argument(
        "--category-contrastive-scope",
        choices=["sequence", "entity_type"],
        default="sequence",
    )
    parser.add_argument(
        "--category-contrastive-wrong-mode",
        choices=["random", "all"],
        default="random",
    )
    parser.add_argument("--category-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--category-prior-bias-scale", type=float, default=0.0)
    parser.add_argument("--category-prior-alpha", type=float, default=1.0)
    parser.add_argument("--semantic-adapter-dim", type=int, default=0)
    parser.add_argument("--semantic-adapter-scale", type=float, default=1.0)
    parser.add_argument("--type-emission-gate-scale", type=float, default=0.0)
    parser.add_argument("--boundary-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--type-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--fgm-epsilon", type=float, default=0.0)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0)
    parser.add_argument("--block-adaptive-crf", action="store_true")
    parser.add_argument("--block-crf-rank", type=int, default=4)
    parser.add_argument("--block-crf-scale", type=float, default=0.2)
    parser.add_argument("--block-crf-start-end", action="store_true")
    parser.add_argument("--lexicon-fusion", action="store_true")
    parser.add_argument("--category-lexicon-fusion", action="store_true")
    parser.add_argument("--category-lexicon-train-feature", action="store_true")
    parser.add_argument("--lexicon-min-freq", type=int, default=1)
    parser.add_argument("--lexicon-min-len", type=int, default=2)
    parser.add_argument("--lexicon-max-entity-len", type=int, default=0)
    parser.add_argument("--fusion-margin", type=float, default=0.0)
    parser.add_argument("--evaluate-category-perturbations", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    if args.category_factor_rank <= 0:
        parser.error("--category-factor-rank must be positive")
    if args.block_crf_rank <= 0:
        parser.error("--block-crf-rank must be positive")
    if args.semantic_adapter_dim < 0:
        parser.error("--semantic-adapter-dim must be non-negative")
    if args.semantic_adapter_scale < 0:
        parser.error("--semantic-adapter-scale must be non-negative")
    if args.rnn_semantic_input_scale < 0:
        parser.error("--rnn-semantic-input-scale must be non-negative")
    if args.rnn_semantic_bridge_scale < 0:
        parser.error("--rnn-semantic-bridge-scale must be non-negative")
    if args.type_emission_gate_scale < 0:
        parser.error("--type-emission-gate-scale must be non-negative")
    if args.boundary_aux_loss_weight < 0:
        parser.error("--boundary-aux-loss-weight must be non-negative")
    if args.type_aux_loss_weight < 0:
        parser.error("--type-aux-loss-weight must be non-negative")
    if args.fgm_epsilon < 0:
        parser.error("--fgm-epsilon must be non-negative")
    if args.rdrop_alpha < 0:
        parser.error("--rdrop-alpha must be non-negative")
    if (args.semantic_adapter_dim > 0 or args.type_emission_gate_scale > 0) and not args.use_category_feature:
        parser.error("semantic adapter and type emission gate require --use-category-feature")
    uses_rnn_semantics = (
        args.rnn_semantic_input_mode != "none"
        or args.rnn_semantic_init
        or args.rnn_semantic_bridge_scale > 0
    )
    if uses_rnn_semantics and not args.use_category_feature:
        parser.error("RNN semantic modules require --use-category-feature")
    if uses_rnn_semantics and args.rnn_type == "none":
        parser.error("RNN semantic modules require --rnn-type lstm or gru")
    if args.attention_semantic_memory and not args.use_category_feature:
        parser.error("--attention-semantic-memory requires --use-category-feature")
    if args.attention_semantic_memory and not args.rnn_attention:
        parser.error("--attention-semantic-memory requires --rnn-attention")
    if args.block_adaptive_crf and not args.use_crf:
        parser.error("--block-adaptive-crf requires --use-crf")
    if args.category_contrastive_weight < 0:
        parser.error("--category-contrastive-weight must be non-negative")
    if args.category_contrastive_weight > 0 and not args.use_crf:
        parser.error("--category-contrastive-weight requires --use-crf")
    if (
        args.category_contrastive_weight > 0
        and args.category_classifier_mode
        not in {"residual", "factorized", "decoupled_factorized"}
    ):
        parser.error(
            "--category-contrastive-weight requires a residual, factorized, "
            "or decoupled_factorized category classifier"
        )
    if (
        args.category_contrastive_scope == "entity_type"
        and args.category_classifier_mode != "decoupled_factorized"
    ):
        parser.error(
            "--category-contrastive-scope entity_type requires "
            "--category-classifier-mode decoupled_factorized"
        )
    if args.category_contrastive_weight > 0 and args.use_category_feature:
        parser.error("counterfactual training currently targets the decoder; do not combine it with --use-category-feature")
    if (
        args.category_uncertainty_gate
        and args.category_classifier_mode
        not in {"residual", "factorized", "decoupled_factorized"}
        and not args.category_prior_bias_scale
    ):
        parser.error(
            "--category-uncertainty-gate requires a residual/factorized classifier "
            "or non-zero category prior"
        )
    if args.evaluate_category_perturbations and args.use_category_prefix:
        parser.error("category perturbation cannot consistently replace a tokenized category prefix")

    base.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"config": vars(args), "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(), "runs": []}
    for dataset in args.datasets:
        run = train_one_dataset(args, dataset)
        result["runs"].append(run)
        suffix_bits = ["crf" if args.use_crf else "softmax"]
        if args.use_category_prefix:
            suffix_bits.append("catprefix")
        if args.use_category_feature:
            if args.category_fusion_mode == "none":
                suffix_bits.append("catinternal")
            elif args.category_fusion_mode == "residual_gate":
                suffix_bits.append("catresgate")
            elif args.category_fusion_mode == "gate":
                suffix_bits.append("catgate")
            elif args.category_fusion_mode == "film":
                suffix_bits.append("catfilm")
            elif args.category_fusion_mode == "conditional_layer_norm":
                suffix_bits.append("catcln")
            else:
                suffix_bits.append("catfeat")
        if args.rnn_semantic_input_mode != "none":
            suffix_bits.append(
                f"rnnsemin{args.rnn_semantic_input_mode}"
                f"{args.rnn_semantic_input_scale:g}"
            )
        if args.rnn_semantic_init:
            suffix_bits.append("rnnseminstate")
        if args.rnn_semantic_bridge_scale:
            suffix_bits.append(f"rnnsembridge{args.rnn_semantic_bridge_scale:g}")
        if args.attention_semantic_memory:
            suffix_bits.append("attsemmemory")
        if args.category_classifier_mode != "shared":
            suffix_bits.append(f"catclf{args.category_classifier_mode}{args.category_classifier_scale:g}")
        if args.category_classifier_mode in {
            "factorized",
            "decoupled_factorized",
        }:
            suffix_bits.append(f"catrank{args.category_factor_rank}")
        if args.category_uncertainty_gate:
            suffix_bits.append("catugate")
        if args.category_contrastive_weight:
            suffix_bits.append(
                f"catcf{args.category_contrastive_weight:g}"
                f"m{args.category_contrastive_margin:g}"
                f"{args.category_contrastive_scope}"
                f"{args.category_contrastive_wrong_mode}"
            )
        if args.category_aux_loss_weight:
            suffix_bits.append(f"cataux{args.category_aux_loss_weight:g}")
        if args.category_prior_bias_scale:
            suffix_bits.append(f"catprior{args.category_prior_bias_scale:g}")
        if args.semantic_adapter_dim:
            suffix_bits.append(
                f"semadpt{args.semantic_adapter_dim}s{args.semantic_adapter_scale:g}"
            )
        if args.type_emission_gate_scale:
            suffix_bits.append(f"typegate{args.type_emission_gate_scale:g}")
        if args.boundary_aux_loss_weight:
            suffix_bits.append(f"baux{args.boundary_aux_loss_weight:g}")
        if args.type_aux_loss_weight:
            suffix_bits.append(f"taux{args.type_aux_loss_weight:g}")
        if args.fgm_epsilon:
            suffix_bits.append(f"fgm{args.fgm_epsilon:g}")
        if args.rdrop_alpha:
            suffix_bits.append(f"rdrop{args.rdrop_alpha:g}")
        if args.block_adaptive_crf:
            suffix_bits.append(
                f"blockcrf{args.block_crf_scale:g}r{args.block_crf_rank}"
            )
            if args.block_crf_start_end:
                suffix_bits.append("blockcrfse")
        if args.category_lexicon_fusion:
            suffix_bits.append("catlex")
        if args.category_lexicon_train_feature:
            suffix_bits.append("catlexfeat")
        suffix = "_".join(suffix_bits)
        (out_dir / f"{dataset.lower()}_semantic_bert_{suffix}_result.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "semantic_bert_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
