# Data Format

The code expects character-level BIO files and one metadata file per split.

## BIO Files

Each non-empty line contains a character and its BIO tag, separated by whitespace:

```text
寒 B-ROCK_TYPE
武 I-ROCK_TYPE
系 I-ROCK_TYPE

断 B-STRUCTURE
裂 I-STRUCTURE
发 O
育 O
```

Blank lines separate paragraph samples.

Required files:

```text
train.txt
dev.txt
test.txt
```

The code also accepts `.bio` files in the dataset folder if you adapt the loader, but the manuscript scripts use `.txt`.

## Metadata Files

Each split should have a matching JSONL metadata file:

```text
train_meta.jsonl
dev_meta.jsonl
test_meta.jsonl
```

Each line corresponds to one BIO sample. The most important field is:

```json
{"category": "地层"}
```

Supported semantic block labels in the manuscript experiments:

```text
地层
构造
岩浆岩
矿床
```

Other metadata fields, such as report name, sample id, source paragraph id, or batch id, can be retained. The model only requires `category` for semantic-block modules.

