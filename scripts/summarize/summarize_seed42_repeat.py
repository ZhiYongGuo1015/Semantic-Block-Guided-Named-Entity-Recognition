from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT_REL = Path("outputs/gold12_50reports_seed42_repeat_check_20260823")
REPEATS = [1, 2, 3]
MODELS = [
    ("RoBERTa-CRF", "01_roberta_crf"),
    ("RoBERTa-SBG-CRF", "02_roberta_sbg_crf"),
]


def load_metric(path: Path) -> dict | None:
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


def mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0] * 100:.2f}% ± 0.00%"
    return f"{statistics.mean(values) * 100:.2f}% ± {statistics.stdev(values) * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT_REL / "seed42_repeat_check_summary_20260823.md")
    args = parser.parse_args()

    project_root = args.project_root
    root = args.root if args.root is not None else project_root / ROOT_REL
    if not root.is_absolute():
        root = project_root / root

    rows = []
    completed = 0
    for model, run_name in MODELS:
        results = []
        for repeat in REPEATS:
            run_dir = root / f"repeat_{repeat}" / run_name
            result = load_metric(run_dir / "semantic_bert_results.json")
            status = "completed" if (run_dir / ".done").exists() and result else "failed" if (run_dir / ".failed").exists() else "pending"
            if status == "completed":
                completed += 1
            results.append((repeat, status, result))
        rows.append((model, results))

    lines = [
        "# Gold12 seed=42 repeat reproducibility check",
        "",
        "本实验只重复运行 RoBERTa-CRF 与 RoBERTa-SBG-CRF，用于检查 seed=42 下 +2.41% 的单次提升是否具有运行可重复性。",
        "",
        "Common setting: dataset=gold12_50reports_core4_bio_sample_stratified_split_20260814, epochs=6, seed=42, batch=8, lr=2e-5, dropout=0.1, weight_decay=0.01.",
        "",
        f"Completed runs: {completed}/{len(MODELS) * len(REPEATS)}",
        "",
        "## Mean ± std",
        "",
        "| Model | Test P mean ± std | Test R mean ± std | Test F1 mean ± std | Completed repeats |",
        "|---|---:|---:|---:|---:|",
    ]

    summary = {}
    for model, results in rows:
        vals = [result for _, status, result in results if status == "completed" and result]
        ps = [v["p"] for v in vals]
        rs = [v["r"] for v in vals]
        f1s = [v["f1"] for v in vals]
        summary[model] = statistics.mean(f1s) if f1s else None
        lines.append(f"| {model} | {mean_std(ps)} | {mean_std(rs)} | {mean_std(f1s)} | {len(vals)}/{len(REPEATS)} |")

    lines.extend(["", "## Per-repeat Test F1", ""])
    lines.append("| Model | " + " | ".join(f"repeat {r}" for r in REPEATS) + " |")
    lines.append("|---" + "|---:" * len(REPEATS) + "|")
    for model, results in rows:
        cells = []
        for _, status, result in results:
            cells.append(pct(result["f1"]) if status == "completed" and result else status)
        lines.append("| " + model + " | " + " | ".join(cells) + " |")

    crf = summary.get("RoBERTa-CRF")
    sbg = summary.get("RoBERTa-SBG-CRF")
    if crf is not None and sbg is not None:
        lines.extend(["", "## Comparison", ""])
        lines.append(f"- Mean Test F1 gain: {pct(sbg - crf)}")

    out = args.out
    if not out.is_absolute():
        out = project_root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "completed": completed, "total": len(MODELS) * len(REPEATS)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
