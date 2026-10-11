"""Dataset loaders for TriviaQA and GSM8K benchmarks (REQ-BENCH-1).

Loaders return ``(task_id, prompt, gold_answer)`` tuples.
Loaders are separate from grading logic (see ``graders.py``).

For v1.0, we load small subsets from standard JSON/JSONL files.
If the datasets are not available locally, synthetic mini-datasets
are provided for smoke testing.
"""

from __future__ import annotations

import json
import os
from typing import Union


# Type alias for a single benchmark item: (task_id, prompt, gold_answer).
# ``gold_answer`` is a string, or a list of acceptable aliases (TriviaQA).
BenchmarkItem = tuple[str, str, Union[str, list[str]]]


# ---------------------------------------------------------------------------
# TriviaQA / SQuAD loader (easy set)
# ---------------------------------------------------------------------------

def load_triviaqa(
    path: str | None = None, limit: int | None = None
) -> list[BenchmarkItem]:
    """Load TriviaQA items from a JSON file.

    Expected format (HuggingFace-style)::

        [{"question": "...", "answer": {"value": "...", "aliases": [...]}}]

    Falls back to a built-in mini dataset if *path* is ``None`` or missing.
    """
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for i, entry in enumerate(data):
            q = entry.get("question", entry.get("Question", ""))
            ans = entry.get("answer", {})
            if isinstance(ans, dict):
                value = ans.get("value", ans.get("Value", ""))
                aliases = ans.get("aliases", ans.get("Aliases", [])) or []
                # Keep every alias so the grader accepts any valid answer.
                gold = [value] + [a for a in aliases if a != value]
            else:
                gold = str(ans)
            items.append((f"triviaqa-{i}", q, gold))
        if limit:
            items = items[:limit]
        return items

    # Built-in mini dataset for smoke testing
    return _MINI_TRIVIAQA[:limit] if limit else list(_MINI_TRIVIAQA)


_MINI_TRIVIAQA: list[BenchmarkItem] = [
    ("triviaqa-0", "What is the capital of France?", "Paris"),
    ("triviaqa-1", "Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("triviaqa-2", "What is the chemical symbol for water?", "H2O"),
    ("triviaqa-3", "In which year did World War II end?", "1945"),
    ("triviaqa-4", "What planet is known as the Red Planet?", "Mars"),
    ("triviaqa-5", "Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("triviaqa-6", "What is the largest ocean on Earth?", "Pacific Ocean"),
    ("triviaqa-7", "What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
    ("triviaqa-8", "Who discovered penicillin?", "Alexander Fleming"),
    ("triviaqa-9", "What is the square root of 144?", "12"),
]


# ---------------------------------------------------------------------------
# GSM8K loader (hard set)
# ---------------------------------------------------------------------------

def load_gsm8k(
    path: str | None = None, limit: int | None = None
) -> list[BenchmarkItem]:
    """Load GSM8K items from a JSONL file.

    Expected format (one JSON object per line)::

        {"question": "...", "answer": "...\\n#### 42"}

    The gold answer is extracted from the ``#### <number>`` suffix.
    Falls back to a built-in mini dataset if *path* is ``None`` or missing.
    """
    if path and os.path.isfile(path):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                entry = json.loads(line.strip())
                q = entry.get("question", "")
                raw_ans = entry.get("answer", "")
                # Extract the #### <number> from GSM8K format
                gold = _extract_gsm8k_answer(raw_ans)
                items.append((f"gsm8k-{i}", q, gold))
        if limit:
            items = items[:limit]
        return items

    return _MINI_GSM8K[:limit] if limit else list(_MINI_GSM8K)


def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K ``#### <answer>`` format."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    # Fallback: return the whole string
    return answer_text.strip()


_MINI_GSM8K: list[BenchmarkItem] = [
    (
        "gsm8k-0",
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
        "and bakes muffins for her friends every day with four. She sells every remaining "
        "egg at the farmers' market daily for $2 per egg. How much in dollars does she "
        "make every day at the farmers' market?",
        "18",
    ),
    (
        "gsm8k-1",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
        "bolts in total does it take?",
        "3",
    ),
    (
        "gsm8k-2",
        "Josh decides to try flipping a house. He buys a house for $80,000 and then "
        "puts in $50,000 in repairs. This increased the value of the house by 150%. "
        "How much profit did he make?",
        "70000",
    ),
    (
        "gsm8k-3",
        "James writes a 3-page letter to 2 different friends twice a week. How many "
        "pages does he write a year?",
        "624",
    ),
    (
        "gsm8k-4",
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, "
        "containing seeds, mealworms and vegetables to help keep them healthy. She gives "
        "the chickens their feed in three separate meals. In the morning, she gives her "
        "flock of chickens 15 cups of feed. In the afternoon, she gives her chickens "
        "another 25 cups of feed. If Wendi has 20 chickens, how many cups of feed does "
        "she need to give her chickens in the final meal of the day?",
        "20",
    ),
]
