#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
BERT_PATH="${BERT_PATH:?Set BERT_PATH to a local BERT pretrained model path or Hugging Face model name.}"
ROBERTA_PATH="${ROBERTA_PATH:?Set ROBERTA_PATH to a local RoBERTa pretrained model path or Hugging Face model name.}"

DATASET="${DATASET:-gold12_50reports_core4_bio_sample_stratified_split_20260814}"
OUT_BASE="$OUTPUT_ROOT/gold12_50reports_epoch6_multiseed_main7_20260822"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-5}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SEEDS=(${SEEDS:-13 21 42 87 100})

mkdir -p "$OUT_BASE"

run_one() {
  local seed="$1"
  local name="$2"
  local model_path="$3"
  shift 3
  local out="$OUT_BASE/seed_${seed}/$name"
  mkdir -p "$out"
  "$PYTHON_BIN" "$REPO_ROOT/src/geo_semantic_bert_suite.py" \
    --data-root "$DATA_ROOT" \
    --out-dir "$out" \
    --model-path "$model_path" \
    --model-type bert \
    --tokenizer-type bert \
    --datasets "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len 256 \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --weight-decay "$WEIGHT_DECAY" \
    --seed "$seed" \
    --train-fraction 1.0 \
    --train-sample-seed "$seed" \
    --save-predictions \
    "$@" | tee "$out/train.screen.log"
}

for seed in "${SEEDS[@]}"; do
  run_one "$seed" "01_bert_fc" "$BERT_PATH"
  run_one "$seed" "02_roberta_fc" "$ROBERTA_PATH"
  run_one "$seed" "03_roberta_crf" "$ROBERTA_PATH" --use-crf
  run_one "$seed" "04_roberta_bilstm_crf" "$ROBERTA_PATH" --use-crf --rnn-type lstm --rnn-hidden 128
  run_one "$seed" "05_roberta_bigru_crf" "$ROBERTA_PATH" --use-crf --rnn-type gru --rnn-hidden 128
  run_one "$seed" "06_roberta_bigru_att_crf" "$ROBERTA_PATH" --use-crf --rnn-type gru --rnn-hidden 128 --rnn-attention
  run_one "$seed" "07_roberta_sbg_crf" "$ROBERTA_PATH" \
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
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/summarize/summarize_multiseed_main7.py" --project-root "$REPO_ROOT"

