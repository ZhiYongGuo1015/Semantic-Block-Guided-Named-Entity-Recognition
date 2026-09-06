#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
ROBERTA_PATH="${ROBERTA_PATH:?Set ROBERTA_PATH to a local RoBERTa pretrained model path or Hugging Face model name.}"

DATASET="${DATASET:-gold12_50reports_core4_bio_sample_stratified_split_20260814}"
OUT_BASE="$OUTPUT_ROOT/gold12_50reports_backbone_specific_semantic_20260823"
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

run_one "01_bilstm/04_lstm_state_bridge_type_lightcrf" \
  --use-crf --rnn-type lstm --rnn-hidden 128 \
  --use-category-feature \
  --rnn-semantic-init \
  --rnn-semantic-bridge-scale 0.2 \
  --type-emission-gate-scale 0.20 \
  --block-adaptive-crf \
  --block-crf-rank 4 \
  --block-crf-scale 0.1

run_one "02_bigru/08_gru_resinput_state_bridge_decoupled" \
  --use-crf --rnn-type gru --rnn-hidden 128 \
  --use-category-feature \
  --rnn-semantic-input-mode residual_gate \
  --rnn-semantic-input-scale 0.2 \
  --rnn-semantic-init \
  --rnn-semantic-bridge-scale 0.2 \
  --category-classifier-mode decoupled_factorized \
  --category-classifier-scale 0.5 \
  --category-factor-rank 8

run_one "03_bigru_attention/10_att_memory_state_type" \
  --use-crf --rnn-type gru --rnn-hidden 128 --rnn-attention \
  --use-category-feature \
  --attention-semantic-memory \
  --rnn-semantic-init \
  --type-emission-gate-scale 0.20

"$PYTHON_BIN" "$REPO_ROOT/scripts/summarize/summarize_backbone_specific_semantic.py" --project-root "$REPO_ROOT"

