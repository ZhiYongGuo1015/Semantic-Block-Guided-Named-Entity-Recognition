from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT_REL = Path("outputs/gold12_50reports_epoch6_multiseed_main7_20260822")
SEEDS = [13, 21, 42, 87, 100]
MODELS = [
    ("BERT-FC", "01_bert_fc"),
    ("RoBERTa-FC", "02_roberta_fc"),
    ("RoBERTa-CRF", "03_roberta_crf"),
    ("RoBERTa-BiLSTM-CRF", "04_roberta_bilstm_crf"),
    ("RoBERTa-BiGRU-CRF", "05_roberta_bigru_crf"),
    ("RoBERTa-BiGRU-Attention-CRF", "06_roberta_bigru_att_crf"),
    ("RoBERTa-SBG-CRF (Ours)", "07_roberta_sbg_crf"),
]


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs") or []
    if not runs:
        return None
    run = runs[0]
    ent = run.get("test_entity") or {}
    if "f1" not in ent:
        return None
    return {
        "p": float(ent.get("precision", 0.0)),
        "r": float(ent.get("recall", 0.0)),
        "f1": float(ent["f1"]),
        "best_dev_f1": float(run.get("best_dev_f1", 0.0)),
        "best_epoch": run.get("best_epoch"),
    }


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{statistics.mean(values) * 100:.2f}% ± {statistics.stdev(values) * 100:.2f}%" if len(values) > 1 else f"{values[0] * 100:.2f}% ± 0.00%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT_REL / "epoch6_multiseed_main7_summary_20260822.md",
    )
    args = parser.parse_args()

    project_root = args.project_root
    root = args.root if args.root is not None else project_root / ROOT_REL
    if not root.is_absolute():
        root = project_root / root

    rows = []
    for model, run_name in MODELS:
        seed_results = []
        for seed in SEEDS:
            run_dir = root / f"seed_{seed}" / run_name
            result = load_result(run_dir / "semantic_bert_results.json")
            status = "completed" if (run_dir / ".done").exists() and result else "failed" if (run_dir / ".failed").exists() else "pending"
            seed_results.append((seed, status, result))
        rows.append((model, run_name, seed_results))

    completed = sum(1 for _, _, seed_results in rows for _, status, _ in seed_results if status == "completed")
    total = len(MODELS) * len(SEEDS)

    lines = [
        "# Gold12 50 reports epoch=6 multi-seed main 7-model summary",
        "",
        "本表用于论文主实验稳定性分析。7 个模型在同一数据集、同一 epoch=6 设置和同一组随机种子下运行；每个 run 内部按 Dev F1 选择最佳 checkpoint，再报告 Test entity-level P/R/F1。",
        "",
        "Common setting: dataset=gold12_50reports_core4_bio_sample_stratified_split_20260814, epochs=6, batch=8, lr=2e-5, dropout=0.1, weight_decay=0.01, max_seq_len=256.",
        "",
        f"Seeds: {', '.join(str(s) for s in SEEDS)}",
        f"Completed runs: {completed}/{total}",
        "",
        "## Mean ± std",
        "",
        "| Model | Test P mean ± std | Test R mean ± std | Test F1 mean ± std | Completed seeds |",
        "|---|---:|---:|---:|---:|",
    ]

    summary = {}
    for model, _, seed_results in rows:
        vals = [result for _, status, result in seed_results if status == "completed" and result]
        ps = [v["p"] for v in vals]
        rs = [v["r"] for v in vals]
        f1s = [v["f1"] for v in vals]
        summary[model] = {
            "p_mean": mean(ps),
            "r_mean": mean(rs),
            "f1_mean": mean(f1s),
            "f1_std": std(f1s),
            "completed": len(vals),
        }
        lines.append(f"| {model} | {mean_std(ps)} | {mean_std(rs)} | {mean_std(f1s)} | {len(vals)}/{len(SEEDS)} |")

    lines.extend(["", "## Per-seed Test F1", ""])
    header = "| Model | " + " | ".join(f"seed {seed}" for seed in SEEDS) + " |"
    sep = "|---" + "|---:" * len(SEEDS) + "|"
    lines.extend([header, sep])
    for model, _, seed_results in rows:
        f1_cells = []
        for _, status, result in seed_results:
            f1_cells.append(pct(result["f1"]) if status == "completed" and result else status)
        lines.append("| " + model + " | " + " | ".join(f1_cells) + " |")

    lines.extend(["", "## Best Dev checkpoint details", ""])
    header = "| Model | " + " | ".join(f"seed {seed}" for seed in SEEDS) + " |"
    lines.extend([header, sep])
    for model, _, seed_results in rows:
        cells = []
        for _, status, result in seed_results:
            if status == "completed" and result:
                cells.append(f"e{result['best_epoch']} / dev {pct(result['best_dev_f1'])}")
            else:
                cells.append(status)
        lines.append("| " + model + " | " + " | ".join(cells) + " |")

    ours = summary.get("RoBERTa-SBG-CRF (Ours)", {})
    roberta_crf = summary.get("RoBERTa-CRF", {})
    if ours.get("f1_mean") is not None and roberta_crf.get("f1_mean") is not None:
        gain = ours["f1_mean"] - roberta_crf["f1_mean"]
        lines.extend(["", "## Main comparison", ""])
        lines.append(f"- RoBERTa-SBG-CRF vs RoBERTa-CRF mean Test F1 gain: {pct(gain)}")
        lines.append("- 论文中应优先报告 mean ± std；单个 seed=42 的结果可作为补充说明。")

    out = args.out
    if not out.is_absolute():
        out = project_root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "completed": completed, "total": total}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
