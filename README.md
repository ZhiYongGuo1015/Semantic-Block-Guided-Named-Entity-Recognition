# Semantic Block Guided Geological NER

This repository contains the code used for semantic-block guided named entity recognition (NER) in Chinese geological reports.

The proposed model is **RoBERTa-SBG-CRF**, which injects geological semantic block information into a RoBERTa-CRF sequence tagging framework through representation modulation, type-aware emission adjustment, and block-adaptive CRF decoding.

## Repository Contents

| Path | Description |
|---|---|
| `src/geo_semantic_bert_suite.py` | Main training and evaluation code for RoBERTa/BERT, CRF, BiLSTM, BiGRU, attention, and semantic-block guided variants. |
| `src/public_ner_bert_suite.py` | Shared BIO reader, tagging utilities, baseline PLM-NER model, and evaluation utilities. |
| `src/public_ner_classic_suite.py` | Classical NER baseline utilities. |
| `src/build_auto_k_semantic_splits.py` | Optional automatic semantic-block split construction script used in exploratory experiments. |
| `scripts/*.sh` | Clean reproducibility scripts without server-specific absolute paths. |
| `scripts/summarize/*.py` | Result summarization scripts for the main experiments. |
| `docs/` | Code, data format, model, and experiment notes. |
| `data/` | Empty placeholder. Put BIO datasets here before running experiments. |
| `outputs/` | Empty placeholder. Experiment outputs will be written here. |

## Data

The annotated Gold data are not included in this code-only release. To reproduce the experiments, place the prepared BIO dataset folders under `data/`.

Expected dataset layout:

```text
data/
  gold12_50reports_core4_bio_sample_stratified_split_20260814/
    train.txt
    dev.txt
    test.txt
    train_meta.jsonl
    dev_meta.jsonl
    test_meta.jsonl
    split_summary.json
```

The `.txt` files use character-level BIO format, with one character and one BIO tag per line. Blank lines separate samples. The corresponding `*_meta.jsonl` files store paragraph-level metadata, including the semantic block label in the `category` field.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

The code expects local pretrained model directories or Hugging Face model names. For the manuscript experiments, the RoBERTa backbone used a Chinese RoBERTa whole-word-masking model.

## Quick Start

Set paths first:

```bash
export DATA_ROOT=/path/to/repo/data
export OUTPUT_ROOT=/path/to/repo/outputs
export ROBERTA_PATH=/path/to/chinese_roberta_wwm_large_ext
export BERT_PATH=/path/to/bert-base-chinese
```

Run the paper's seed=42 main comparison:

```bash
bash scripts/run_seed42_main.sh
```

Run the multi-seed stability experiment:

```bash
bash scripts/run_multiseed_main7.sh
```

Run report-level split evaluation:

```bash
bash scripts/run_report_level_main5.sh
```

Run backbone-specific semantic adaptation:

```bash
bash scripts/run_backbone_specific_semantic.sh
```

## Main Model Configuration

The main RoBERTa-SBG-CRF setting uses:

```text
--use-crf
--use-category-feature
--category-fusion-mode residual_gate
--category-emb-dim 32
--semantic-adapter-dim 64
--semantic-adapter-scale 1.0
--type-emission-gate-scale 0.20
--rdrop-alpha 0.5
--block-adaptive-crf
--block-crf-rank 8
--block-crf-scale 0.2
```

## Notes for Public Release

Before uploading to GitHub, check whether the annotated data and pretrained model paths can be publicly released. This code-only package intentionally excludes private data, experiment logs, model checkpoints, server IP addresses, and SSH credentials.

