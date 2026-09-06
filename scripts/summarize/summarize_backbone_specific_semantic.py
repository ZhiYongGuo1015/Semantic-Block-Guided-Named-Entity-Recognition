from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_REL = Path("outputs/gold12_50reports_backbone_specific_semantic_20260823")
PREVIOUS_REL = Path("outputs/gold12_50reports_plm10_aligned_20260820")

BACKBONES = {
    "RoBERTa-BiLSTM-CRF": {
        "group": "01_bilstm",
        "baseline_name": "08_roberta_bilstm_crf",
        "previous_name": "08_roberta_bilstm_crf_semantic",
        "fallback_baseline": 0.8491,
        "fallback_previous": 0.8587,
    },
    "RoBERTa-BiGRU-CRF": {
        "group": "02_bigru",
        "baseline_name": "09_roberta_bigru_crf",
        "previous_name": "09_roberta_bigru_crf_semantic",
        "fallback_baseline": 0.8522,
        "fallback_previous": 0.8464,
    },
    "RoBERTa-BiGRU-Attention-CRF": {
        "group": "03_bigru_attention",
        "baseline_name": "10_roberta_bigru_att_crf",
        "previous_name": "10_roberta_bigru_att_crf_semantic",
        "fallback_baseline": 0.8498,
        "fallback_previous": 0.8541,
    },
}

CANDIDATES = [
    ("RoBERTa-BiLSTM-CRF", "01_lstm_block_state_type", "block state init + type gate"),
    ("RoBERTa-BiLSTM-CRF", "02_lstm_film_state_type", "pre-LSTM FiLM + block state init + type gate"),
    ("RoBERTa-BiLSTM-CRF", "03_lstm_resinput_state_bridge_type", "pre-LSTM residual gate + block state init + RoBERTa bridge + type gate"),
    ("RoBERTa-BiLSTM-CRF", "04_lstm_state_bridge_type_lightcrf", "block state init + RoBERTa bridge + type gate + light block CRF"),
    ("RoBERTa-BiGRU-CRF", "05_gru_block_state_type", "block state init + type gate"),
    ("RoBERTa-BiGRU-CRF", "06_gru_resinput_state_type", "pre-GRU residual gate + block state init + type gate"),
    ("RoBERTa-BiGRU-CRF", "07_gru_film_state_bridge_type", "pre-GRU FiLM + block state init + RoBERTa bridge + type gate"),
    ("RoBERTa-BiGRU-CRF", "08_gru_resinput_state_bridge_decoupled", "pre-GRU residual gate + block state init + RoBERTa bridge + decoupled type decoder"),
    ("RoBERTa-BiGRU-Attention-CRF", "09_att_block_memory_type", "attention block memory + type gate"),
    ("RoBERTa-BiGRU-Attention-CRF", "10_att_memory_state_type", "attention block memory + GRU state init + type gate"),
    ("RoBERTa-BiGRU-Attention-CRF", "11_att_memory_resinput_state_type", "attention block memory + pre-GRU residual gate + state init + type gate"),
    ("RoBERTa-BiGRU-Attention-CRF", "12_att_memory_film_bridge_type_lightcrf", "attention block memory + pre-GRU FiLM + RoBERTa bridge + type gate + light block CRF"),
]


def load_metric(path: Path) -> tuple[float, float, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs") or []
    if not runs:
        return None
    metric = runs[0].get("test_entity") or {}
    if "f1" not in metric:
        return None
    return (
        float(metric.get("precision", 0.0)),
        float(metric.get("recall", 0.0)),
        float(metric["f1"]),
    )


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def delta(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:+.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT_REL / "backbone_specific_semantic_summary_20260823.md",
    )
    args = parser.parse_args()
    root = args.project_root

    reference = {}
    for backbone, config in BACKBONES.items():
        baseline = load_metric(
            root
            / PREVIOUS_REL
            / "01_baseline"
            / config["baseline_name"]
            / "semantic_bert_results.json"
        )
        previous = load_metric(
            root
            / PREVIOUS_REL
            / "02_semantic"
            / config["previous_name"]
            / "semantic_bert_results.json"
        )
        reference[backbone] = {
            "baseline": baseline[2] if baseline else config["fallback_baseline"],
            "previous": previous[2] if previous else config["fallback_previous"],
        }

    rows = []
    for backbone, run_name, method in CANDIDATES:
        group = BACKBONES[backbone]["group"]
        run_dir = root / ROOT_REL / group / run_name
        metric = load_metric(run_dir / "semantic_bert_results.json")
        if (run_dir / ".done").exists():
            status = "completed"
        elif (run_dir / ".failed").exists():
            status = "failed"
        else:
            status = "pending/running"
        f1 = metric[2] if metric else None
        rows.append(
            {
                "backbone": backbone,
                "run": run_name,
                "method": method,
                "metric": metric,
                "baseline_delta": None if f1 is None else f1 - reference[backbone]["baseline"],
                "previous_delta": None if f1 is None else f1 - reference[backbone]["previous"],
                "status": status,
            }
        )

    completed = sum(row["metric"] is not None for row in rows)
    lines = [
        "# Gold12 backbone-specific semantic-block structure experiment",
        "",
        "本轮不再把 RoBERTa-SBG-CRF 的统一后置结构直接套到循环模型，而是分别改造 BiLSTM 状态、BiGRU 输入与残差通路，以及 BiGRU-Attention 的注意力记忆。",
        "",
        "Common setting: dataset=gold12_50reports_core4_bio_sample_stratified_split_20260814, epochs=6, batch=8, seed=42, lr=2e-5, dropout=0.1, weight_decay=0.01, R-Drop alpha=0.3.",
        "",
        f"Completed: {completed}/{len(rows)}",
        "",
        "| Backbone | Targeted structure | Test P | Test R | Test F1 | vs baseline | vs previous generic semantic | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metric = row["metric"]
        lines.append(
            "| {backbone} | {method} | {p} | {r} | {f1} | {db} | {dp} | {status} |".format(
                backbone=row["backbone"],
                method=row["method"],
                p=pct(metric[0]) if metric else "-",
                r=pct(metric[1]) if metric else "-",
                f1=pct(metric[2]) if metric else "-",
                db=delta(row["baseline_delta"]),
                dp=delta(row["previous_delta"]),
                status=row["status"],
            )
        )

    lines.extend(["", "## Best completed variant by backbone", ""])
    for backbone in BACKBONES:
        candidates = [row for row in rows if row["backbone"] == backbone and row["metric"]]
        if not candidates:
            lines.append(f"- `{backbone}`: no completed candidate")
            continue
        best = max(candidates, key=lambda row: row["metric"][2])
        lines.append(
            f"- `{backbone}`: `{best['run']}`, Test F1 {pct(best['metric'][2])}, "
            f"vs baseline {delta(best['baseline_delta'])}, vs previous generic semantic {delta(best['previous_delta'])}"
        )

    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "completed": completed, "total": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
