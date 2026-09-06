from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_REL = Path("outputs/gold12_50reports_report_level_e6_seed42_main5_20260824")
DATASET_REL = Path("data/gold12_50reports_core4_bio_report_split_20260814")

MODELS = [
    ("RoBERTa-CRF", "01_roberta_crf"),
    ("RoBERTa-BiLSTM-CRF", "02_roberta_bilstm_crf"),
    ("RoBERTa-BiGRU-CRF", "03_roberta_bigru_crf"),
    ("RoBERTa-BiGRU-Attention-CRF", "04_roberta_bigru_attention_crf"),
    ("RoBERTa-SBG-CRF (本文)", "05_roberta_sbg_crf"),
]


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def signed_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}%"


def load_result(run_dir: Path) -> dict | None:
    path = run_dir / "semantic_bert_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs") or []
    if not runs:
        return None
    run = runs[0]
    entity = run.get("test_entity") or {}
    if "f1" not in entity:
        return None
    return {
        "p": float(entity.get("precision", 0.0)),
        "r": float(entity.get("recall", 0.0)),
        "f1": float(entity["f1"]),
        "seconds": float(run.get("seconds", 0.0)),
        "train_sentences": int(run.get("train_sentences", 0)),
        "dev_sentences": int(run.get("dev_sentences", 0)),
        "test_sentences": int(run.get("test_sentences", 0)),
    }


def load_dataset_summary(project_root: Path) -> dict:
    path = project_root / DATASET_REL / "split_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT_REL / "report_level_e6_seed42_main5_summary_20260824.md")
    args = parser.parse_args()

    project_root = args.project_root
    root = args.root if args.root is not None else project_root / ROOT_REL
    if not root.is_absolute():
        root = project_root / root

    dataset_summary = load_dataset_summary(project_root)
    rows = []
    completed = 0
    for label, run_name in MODELS:
        run_dir = root / run_name
        result = load_result(run_dir)
        status = "completed" if (run_dir / ".done").exists() and result else "failed" if (run_dir / ".failed").exists() else "pending"
        if status == "completed":
            completed += 1
        rows.append((label, run_name, result, status))

    roberta_crf = next((result for label, _, result, status in rows if label == "RoBERTa-CRF" and status == "completed"), None)
    best_baseline = None
    for label, _, result, status in rows:
        if label == "RoBERTa-SBG-CRF (本文)" or status != "completed" or result is None:
            continue
        if best_baseline is None or result["f1"] > best_baseline["f1"]:
            best_baseline = {"label": label, **result}

    lines = [
        "# Gold12 50 reports report-level split e6 seed42 main-5 summary",
        "",
        "本轮用于论文稳健性验证：同一份报告的段落不会同时出现在 train/dev/test，测试集由未参与训练的完整报告构成。",
        "",
        "Common setting: dataset=gold12_50reports_core4_bio_report_split_20260814, epochs=6, batch=8, seed=42, lr=2e-5, dropout=0.1, weight_decay=0.01, max_seq_len=256.",
        "",
        f"Completed runs: {completed}/{len(MODELS)}",
        "",
    ]
    if dataset_summary:
        lines.extend(
            [
                "## Dataset Split",
                "",
                f"- Split strategy: `{dataset_summary.get('split_strategy', '-')}`",
                f"- Reports: {dataset_summary.get('report_count', '-')} total; "
                f"train/dev/test = {dataset_summary.get('report_counts_by_split', {})}",
                f"- Samples: {dataset_summary.get('sample_count', '-')} total; "
                f"train/dev/test = {dataset_summary.get('split_counts', {})}",
                f"- Entities: {dataset_summary.get('entity_count', '-')}",
                f"- Sample IDs disjoint: {dataset_summary.get('sample_id_disjoint', '-')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Main Results",
            "",
            "| Model | Test P | Test R | Test F1 | vs RoBERTa-CRF | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for label, _, result, status in rows:
        if status == "completed" and result:
            gain = result["f1"] - roberta_crf["f1"] if roberta_crf else None
            lines.append(
                f"| {label} | {pct(result['p'])} | {pct(result['r'])} | {pct(result['f1'])} | {signed_pct(gain)} | {status} |"
            )
        else:
            lines.append(f"| {label} | - | - | - | - | {status} |")

    ours = next((result for label, _, result, status in rows if label == "RoBERTa-SBG-CRF (本文)" and status == "completed"), None)
    if ours and roberta_crf:
        lines.extend(["", "## Key Comparison", ""])
        lines.append(f"- RoBERTa-SBG-CRF vs RoBERTa-CRF: Test F1 {pct(ours['f1'])} vs {pct(roberta_crf['f1'])}, gain {signed_pct(ours['f1'] - roberta_crf['f1'])}.")
        if best_baseline:
            lines.append(f"- RoBERTa-SBG-CRF vs best non-SBG baseline ({best_baseline['label']}): gain {signed_pct(ours['f1'] - best_baseline['f1'])}.")
        lines.append("- 若该结果仍保持正向提升，可作为样本级主实验之外的报告级泛化验证。")

    out = args.out
    if not out.is_absolute():
        out = project_root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "completed": completed, "total": len(MODELS)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
