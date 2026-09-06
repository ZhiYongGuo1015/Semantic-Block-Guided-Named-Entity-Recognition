from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TERM_GROUPS = {
    "strata": ["地层", "群", "组", "段", "系", "界", "统", "岩组", "Pt", "Z", "D", "C", "P", "T", "J", "K"],
    "structure": ["断裂", "褶皱", "构造", "背斜", "向斜", "节理", "破碎带", "剪切带", "走向", "倾向", "倾角"],
    "magmatic": ["岩体", "岩株", "岩脉", "花岗", "闪长", "辉绿", "侵入", "岩浆", "火山", "喷发"],
    "deposit": ["矿体", "矿脉", "矿床", "矿点", "矿化", "矿石", "品位", "资源量", "储量", "金", "铜", "铅", "锌"],
    "lithology": ["砂岩", "板岩", "灰岩", "白云岩", "片岩", "片麻岩", "凝灰岩", "角砾岩", "泥岩", "页岩"],
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def make_bio(text: str, entities: list[dict]) -> list[tuple[str, str]]:
    chars = list(text)
    tags = ["O"] * len(chars)
    for entity in sorted(entities, key=lambda item: (int(item.get("start", 0)), -int(item.get("end", 0)))):
        start = int(entity.get("start", 0))
        end = int(entity.get("end", 0))
        label = str(entity.get("label", "")).strip()
        if not label or start < 0 or end > len(chars) or start >= end:
            continue
        if any(tag != "O" for tag in tags[start:end]):
            continue
        tags[start] = f"B-{label}"
        for index in range(start + 1, end):
            tags[index] = f"I-{label}"
    return list(zip(chars, tags))


def write_bio(path: Path, samples: list[dict]) -> None:
    lines: list[str] = []
    for sample in samples:
        for char, tag in make_bio(sample.get("content", ""), sample.get("entities", [])):
            lines.append(f"{char} {tag}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def density_features(texts: list[str]) -> np.ndarray:
    features = []
    for text in texts:
        length = max(1, len(text))
        row = [
            len(text),
            sum(ch.isdigit() for ch in text) / length,
            sum(ch in "，。；、（）()[]【】" for ch in text) / length,
        ]
        for terms in TERM_GROUPS.values():
            count = sum(text.count(term) for term in terms)
            row.append(count / length)
        features.append(row)
    return np.asarray(features, dtype=float)


def weak_text_features(texts: list[str]) -> np.ndarray:
    features = []
    punctuation = set("，。；、：（）()[]【】《》“”\"'！？!?.,;:-")
    for text in texts:
        length = max(1, len(text))
        features.append(
            [
                len(text),
                sum(ch.isdigit() for ch in text) / length,
                sum(ch in punctuation for ch in text) / length,
                sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text) / length,
                sum("\u4e00" <= ch <= "\u9fff" for ch in text) / length,
            ]
        )
    return np.asarray(features, dtype=float)


def make_word_analyzer():
    try:
        import jieba  # type: ignore

        def analyzer(text: str) -> list[str]:
            return [token.strip() for token in jieba.lcut(text) if len(token.strip()) >= 2]

        return analyzer, "jieba_word_tfidf"
    except Exception:
        pattern = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}")

        def analyzer(text: str) -> list[str]:
            tokens: list[str] = []
            for match in pattern.findall(text):
                if re.fullmatch(r"[\u4e00-\u9fff]{2,}", match):
                    tokens.extend(match[i : i + 2] for i in range(max(0, len(match) - 1)))
                else:
                    tokens.append(match)
            return tokens

        return analyzer, "fallback_bigram_word_tfidf"


def stratified_three_way_split(rows: list[dict], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, list[dict]]:
    labels = [row["category"] for row in rows]
    train_rows, temp_rows, train_labels, temp_labels = train_test_split(
        rows,
        labels,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    dev_fraction_of_temp = dev_ratio / (1.0 - train_ratio)
    dev_rows, test_rows = train_test_split(
        temp_rows,
        train_size=dev_fraction_of_temp,
        random_state=seed + 1,
        shuffle=True,
        stratify=temp_labels,
    )
    return {"train": list(train_rows), "dev": list(dev_rows), "test": list(test_rows)}


def source_stratified_three_way_split(rows: list[dict], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, list[dict]]:
    labels = [
        row.get("category")
        or row.get("semantic_block")
        or row.get("original_category")
        or "UNKNOWN"
        for row in rows
    ]
    train_rows, temp_rows, train_labels, temp_labels = train_test_split(
        rows,
        labels,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    dev_fraction_of_temp = dev_ratio / (1.0 - train_ratio)
    dev_rows, test_rows = train_test_split(
        temp_rows,
        train_size=dev_fraction_of_temp,
        random_state=seed + 1,
        shuffle=True,
        stratify=temp_labels,
    )
    return {"train": list(train_rows), "dev": list(dev_rows), "test": list(test_rows)}


def fit_transform_features(
    train_texts: list[str],
    other_texts: list[str],
    feature_mode: str,
    max_features: int,
):
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    train_char = char_vectorizer.fit_transform(train_texts)
    other_char = (
        char_vectorizer.transform(other_texts)
        if other_texts
        else csr_matrix((0, train_char.shape[1]))
    )

    if feature_mode == "domain_tfidf_density":
        scaler = StandardScaler()
        train_dense = scaler.fit_transform(density_features(train_texts))
        other_dense = (
            scaler.transform(density_features(other_texts))
            if other_texts
            else np.zeros((0, train_dense.shape[1]), dtype=float)
        )
        train_features = hstack([train_char, csr_matrix(train_dense)], format="csr")
        other_features = hstack([other_char, csr_matrix(other_dense)], format="csr")
        feature_desc = "char_tfidf_2_4_plus_domain_term_density"
        return train_features, other_features, feature_desc

    word_analyzer, word_analyzer_name = make_word_analyzer()
    word_vectorizer = TfidfVectorizer(
        analyzer=word_analyzer,
        min_df=2,
        max_features=max(1000, max_features // 2),
        sublinear_tf=True,
    )
    train_word = word_vectorizer.fit_transform(train_texts)
    other_word = (
        word_vectorizer.transform(other_texts)
        if other_texts
        else csr_matrix((0, train_word.shape[1]))
    )
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(weak_text_features(train_texts))
    other_dense = (
        scaler.transform(weak_text_features(other_texts))
        if other_texts
        else np.zeros((0, train_dense.shape[1]), dtype=float)
    )
    train_features = hstack([train_char, train_word, csr_matrix(train_dense)], format="csr")
    other_features = hstack([other_char, other_word, csr_matrix(other_dense)], format="csr")
    feature_desc = f"pure_char_tfidf_2_4_plus_{word_analyzer_name}_plus_weak_text_stats"
    return train_features, other_features, feature_desc


def assign_auto_cluster(row: dict, k: int, label: int, output_tag: str) -> dict:
    copied = dict(row)
    copied["original_category"] = row.get("category", "")
    copied["original_semantic_block"] = row.get("semantic_block", "")
    copied["auto_cluster"] = int(label)
    copied["category"] = f"AutoK{k}_C{label}"
    copied["semantic_block_source"] = f"auto_{output_tag}_kmeans"
    return copied


def summarize(rows_by_split: dict[str, list[dict]], k: int, inertia: float) -> dict:
    summary = {
        "k": k,
        "inertia": inertia,
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "cluster_counts_by_split": {},
        "original_block_by_cluster": {},
    }
    cluster_block_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split, rows in rows_by_split.items():
        summary["cluster_counts_by_split"][split] = dict(Counter(row["category"] for row in rows))
        for row in rows:
            cluster_block_counts[row["category"]][row.get("semantic_block", "")]
    for split_rows in rows_by_split.values():
        for row in split_rows:
            cluster_block_counts[row["category"]][row.get("semantic_block", "")] += 1
    summary["original_block_by_cluster"] = {
        cluster: dict(counts)
        for cluster, counts in sorted(cluster_block_counts.items())
    }
    return summary


def write_dataset(out_dir: Path, rows_by_split: dict[str, list[dict]], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        sorted_rows = sorted(rows, key=lambda row: (row["category"], row.get("report_id", ""), row.get("doc_order", ""), row.get("sample_id", "")))
        for row in sorted_rows:
            row["split"] = split
        write_bio(out_dir / f"{split}.txt", sorted_rows)
        write_bio(out_dir / f"{split}.bio", sorted_rows)
        write_jsonl(out_dir / f"{split}_meta.jsonl", sorted_rows)
        with (out_dir / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["sample_id", "split", "category", "auto_cluster", "semantic_block", "report_id", "content"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted_rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    write_jsonl(out_dir / "all_samples.jsonl", [row for rows in rows_by_split.values() for row in rows])
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        f"# {out_dir.name}\n\n"
        f"自动 K 类语义块数据集：先对全部样本做 {summary.get('feature', '')} KMeans 聚类，"
        f"划分流程：{summary.get('split_protocol', '')}。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="data/gold12_50reports_core4_bio_sample_stratified_split_20260814",
        help="Input BIO dataset directory.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Output root for generated automatic semantic-block datasets.",
    )
    parser.add_argument("--k-values", default="4,5,6,7,8")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument(
        "--feature-mode",
        choices=["pure_tfidf_weak", "domain_tfidf_density"],
        default="pure_tfidf_weak",
        help="pure_tfidf_weak excludes geology-type dictionary densities; domain_tfidf_density keeps the earlier domain-enhanced version.",
    )
    parser.add_argument("--output-tag", default="")
    parser.add_argument(
        "--split-before-cluster",
        action="store_true",
        help="Strict protocol: split train/dev/test first, fit TF-IDF/KMeans only on train, then predict dev/test clusters.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    data_root = Path(args.data_root)
    rows = read_jsonl(source_dir / "all_samples.jsonl")
    output_tag = args.output_tag or (
        "tfidf_density" if args.feature_mode == "domain_tfidf_density" else "pure_tfidf_weak"
    )

    outputs = {}
    for k in [int(value.strip()) for value in args.k_values.split(",") if value.strip()]:
        if args.split_before_cluster:
            raw_split = source_stratified_three_way_split(
                rows,
                args.seed + k,
                args.train_ratio,
                args.dev_ratio,
            )
            train_texts = [row.get("content", "") for row in raw_split["train"]]
            dev_test_rows = raw_split["dev"] + raw_split["test"]
            other_texts = [row.get("content", "") for row in dev_test_rows]
            train_features, other_features, feature_desc = fit_transform_features(
                train_texts,
                other_texts,
                args.feature_mode,
                args.max_features,
            )
            kmeans = KMeans(n_clusters=k, random_state=args.seed + k, n_init=20)
            train_labels = kmeans.fit_predict(train_features)
            other_labels = kmeans.predict(other_features)
            dev_labels = other_labels[: len(raw_split["dev"])]
            test_labels = other_labels[len(raw_split["dev"]) :]
            rows_by_split = {
                "train": [
                    assign_auto_cluster(row, k, label, output_tag)
                    for row, label in zip(raw_split["train"], train_labels)
                ],
                "dev": [
                    assign_auto_cluster(row, k, label, output_tag)
                    for row, label in zip(raw_split["dev"], dev_labels)
                ],
                "test": [
                    assign_auto_cluster(row, k, label, output_tag)
                    for row, label in zip(raw_split["test"], test_labels)
                ],
            }
            split_protocol = "strict_split_first_train_fit_kmeans_then_devtest_predict"
            leakage_note = "strict: TF-IDF, scaler, and KMeans are fitted on train only"
        else:
            texts = [row.get("content", "") for row in rows]
            features, _, feature_desc = fit_transform_features(
                texts,
                [],
                args.feature_mode,
                args.max_features,
            )
            kmeans = KMeans(n_clusters=k, random_state=args.seed + k, n_init=20)
            labels = kmeans.fit_predict(features)
            clustered_rows = [
                assign_auto_cluster(row, k, label, output_tag)
                for row, label in zip(rows, labels)
            ]
            rows_by_split = stratified_three_way_split(
                clustered_rows,
                args.seed + k,
                args.train_ratio,
                args.dev_ratio,
            )
            split_protocol = "cluster_all_samples_then_stratified_split"
            leakage_note = "clustered on all samples before train/dev/test split as requested"

        suffix = f"{output_tag}_strict" if args.split_before_cluster else output_tag
        out_dir = data_root / f"gold12_50reports_autoK{k}_{suffix}_split_20260815"
        summary = summarize(rows_by_split, k, float(kmeans.inertia_))
        summary.update(
            {
                "source_dir": str(source_dir),
                "out_dir": str(out_dir),
                "seed": args.seed,
                "feature": feature_desc,
                "feature_mode": args.feature_mode,
                "output_tag": output_tag,
                "split_before_cluster": args.split_before_cluster,
                "split_protocol": split_protocol,
                "leakage_note": leakage_note,
            }
        )
        write_dataset(out_dir, rows_by_split, summary)
        outputs[k] = str(out_dir)
    print(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
