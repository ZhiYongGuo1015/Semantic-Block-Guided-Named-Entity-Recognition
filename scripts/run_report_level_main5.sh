#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
ROBERTA_PATH="${ROBERTA_PATH:?Set ROBERTA_PATH to a local RoBERTa pretrained model path or Hugging Face model name.}"

DATASET="${DATASET:-gold12_50reports_core4_bio_report_split_20260814}"
OUT_BASE="$OUTPUT_ROOT/gold12_50reports_report_level_e6_seed42_main5_20260824"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
LR="${LR:-2e-5}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

mkdir -p "$OUT_BASE"

run_one() {
  local name="$1"
  shift
  local out="$OUT_BASE/$name"
  mkdir -p "$out"
  "$PYTHON_BIN" "$REPO_ROOT/src/geo_semantic_bert_suite.py" \
    --data-root "$DATA_ROOT" \
    --out-dir "$out" \
    --model-path "$ROBERTA_PATH" \
    --model-type bert \
    --tokenizer-type bert \
    --datasets "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len 256 \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --weight-decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --train-fraction 1.0 \
    --train-sample-seed "$SEED" \
    --save-predictions \
    "$@" | tee "$out/train.screen.log"
}

run_one "01_roberta_crf" --use-crf
run_one "02_roberta_bilstm_crf" --use-crf --rnn-type lstm --rnn-hidden 128
run_one "03_roberta_bigru_crf" --use-crf --rnn-type gru --rnn-hidden 128
run_one "04_roberta_bigru_attention_crf" --use-crf --rnn-type gru --rnn-hidden 128 --rnn-attention
run_one "05_roberta_sbg_crf" \
  --use-crf \
  --use-category-feature \
  --category-fusion-mode residual_gate \
  --category-emb-dim 32 \
  --semantic-adapter-dim 64 \
  --semantic-adapter-scale 1.0 \
  --type-emission-gate-scale 0.20 \
  --rdrop-alpha 0.5 \
  --block-adaptive-crf \
  --block-crf-rank 8 \
  --block-crf-scale 0.2 \
  --evaluate-category-perturbations

"$PYTHON_BIN" "$REPO_ROOT/scripts/summarize/summarize_report_level_main5.py" --project-root "$REPO_ROOT"

