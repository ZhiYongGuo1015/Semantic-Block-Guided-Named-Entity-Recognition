# Reproducibility Guide

## 1. Prepare Data

Place the dataset folder under `data/`.

For the main sample-level experiment:

```text
data/gold12_50reports_core4_bio_sample_stratified_split_20260814/
```

For report-level evaluation:

```text
data/gold12_50reports_core4_bio_report_split_20260814/
```

## 2. Prepare Pretrained Models

Set the pretrained model paths before running scripts:

```bash
export ROBERTA_PATH=/path/to/chinese_roberta_wwm_large_ext
export BERT_PATH=/path/to/bert-base-chinese
```

Alternatively, set them to Hugging Face model names if your environment has internet access or local cache.

## 3. Run Main Experiments

Seed=42 main comparison:

```bash
bash scripts/run_seed42_main.sh
```

Multi-seed stability experiment:

```bash
bash scripts/run_multiseed_main7.sh
python scripts/summarize/summarize_multiseed_main7.py --project-root .
```

Report-level split experiment:

```bash
bash scripts/run_report_level_main5.sh
python scripts/summarize/summarize_report_level_main5.py --project-root .
```

Backbone-specific semantic adaptation:

```bash
bash scripts/run_backbone_specific_semantic.sh
python scripts/summarize/summarize_backbone_specific_semantic.py --project-root .
```

## 4. Output Files

Each run directory contains:

```text
semantic_bert_results.json
train.screen.log
*_dev_predictions.jsonl
*_test_predictions.jsonl
```

The key metrics are stored in `semantic_bert_results.json` under `test_entity`.

