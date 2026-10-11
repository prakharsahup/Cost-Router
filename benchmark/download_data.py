"""Download real TriviaQA / GSM8K samples for the benchmark (REQ-BENCH-1).

Fetches the official splits from the Hugging Face Hub (parquet export) and
writes a deterministic, seeded sample in the formats ``datasets.py`` reads:

- ``data/triviaqa_sample.json``  — TriviaQA ``rc.nocontext`` validation split
- ``data/gsm8k_sample.jsonl``    — GSM8K ``main`` test split

Usage::

    python -m benchmark.download_data --n 50 --seed 42
"""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

TRIVIAQA_URL = (
    "https://huggingface.co/api/datasets/mandarjoshi/trivia_qa/parquet/"
    "rc.nocontext/validation/0.parquet"
)
GSM8K_URL = "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _fetch_parquet(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return pd.read_parquet(io.BytesIO(resp.read()))


def _sample_indices(total: int, n: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(total), n))


def download_triviaqa(n: int, seed: int, out: Path) -> int:
    df = _fetch_parquet(TRIVIAQA_URL)
    records = []
    for i in _sample_indices(len(df), n, seed):
        row = df.iloc[i]
        ans = row["answer"]
        records.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": {
                "value": ans["value"],
                "aliases": list(ans["aliases"]),
            },
        })
    out.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(records)


def download_gsm8k(n: int, seed: int, out: Path) -> int:
    df = _fetch_parquet(GSM8K_URL)
    with out.open("w", encoding="utf-8") as f:
        for i in _sample_indices(len(df), n, seed):
            row = df.iloc[i]
            f.write(json.dumps({"question": row["question"], "answer": row["answer"]},
                               ensure_ascii=False) + "\n")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=50, help="items per dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    t = download_triviaqa(args.n, args.seed, DATA_DIR / "triviaqa_sample.json")
    g = download_gsm8k(args.n, args.seed, DATA_DIR / "gsm8k_sample.jsonl")
    print(f"Wrote {t} TriviaQA and {g} GSM8K items to {DATA_DIR}")


if __name__ == "__main__":
    main()
