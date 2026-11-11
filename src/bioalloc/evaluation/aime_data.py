"""AIME competition problems, references and grading (decision D31).

Sources, pinned: Hugging Face ``gneubig/aime-1983-2024`` (CC0) for 1983-2024 and
``opencompass/AIME2025`` (MIT) for 2025. Only 2020-2025 enter the pool (180 problems): recent
problems are hard for the backbone without extended thinking and less likely to be memorized.
Every AIME answer is an integer from 0 to 999, so grading is exact and needs no parser. The
public side reuses the GSM8K family (one numeric answer, a calculator, no goal check), so the
model cannot verify an answer: this is the regime where independent votes carry information.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from bioalloc.decomposition import PublicTask
from bioalloc.evaluation.data import PREPARED, RAW, fetch, read_jsonl, write_jsonl, write_manifest
from bioalloc.evaluation.gsm8k_data import public_task as _gsm8k_task
from bioalloc.evaluation.toy_data import Grade
from bioalloc.tasks.common import normalize_number

HISTORIC = (
    "https://huggingface.co/datasets/gneubig/aime-1983-2024/resolve/"
    "5d610df981dec508dd93d0a16333a029ac1739d8/AIME_Dataset_1983_2024.csv",
    "959358884aa93b30b8d85ecb80fd8b577014498c08f7569faf88f675cc3fb9f4",
)
Y2025 = {
    "I": (
        "https://huggingface.co/datasets/opencompass/AIME2025/resolve/"
        "a6ad95f611d72cf628a80b58bd0432ef6638f958/aime2025-I.jsonl",
        "b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e",
    ),
    "II": (
        "https://huggingface.co/datasets/opencompass/AIME2025/resolve/"
        "a6ad95f611d72cf628a80b58bd0432ef6638f958/aime2025-II.jsonl",
        "16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e",
    ),
}
Y2024 = (
    "https://huggingface.co/datasets/Maxwell-Jia/AIME_2024/resolve/"
    "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792/aime_2024_problems.parquet",
    "3cb34696a75837f848ea16f15a42f4130be655d5995e8cd24efcc54f4e6e1524",
)  # MIT; the 1983-2024 file lacks AIME 2024 I, so 2024 comes from here
FIRST_YEAR = 2020
PUBLIC = Path("aime/public.jsonl")
PRIVATE = Path("aime/private.jsonl")
INSTRUCTION = (
    "Solve the competition math problem in the input 'question'. The answer is an integer from "
    '0 to 999. Reply with the final answer as {"answer": <integer>}.'
)


def task_id(year: int, part: str, number: int) -> str:
    return f"aime-{year}-{part}-{number:02d}"


def prepare() -> Path:
    rows: list[tuple[str, int, str, str]] = []  # id, year, question, answer
    local = fetch(HISTORIC[0], HISTORIC[1], RAW / "aime" / "aime_1983_2024.csv")
    with local.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            year = int(r["Year"])
            if year < FIRST_YEAR or year == 2024:
                continue
            part = (r["Part"] or "").strip() or "I"
            rows.append(
                (task_id(year, part, int(r["Problem Number"])), year, r["Question"], r["Answer"])
            )
    import pyarrow.parquet as pq  # preparation-only dependency (uv group "prep")

    path24 = fetch(Y2024[0], Y2024[1], RAW / "aime" / "aime_2024.parquet")
    for item in pq.read_table(path24).to_pylist():
        _, part, num = str(item["ID"]).split("-")
        rows.append(
            (task_id(2024, part, int(num)), 2024, str(item["Problem"]), str(item["Answer"]))
        )
    for part, (url, digest) in Y2025.items():
        path = fetch(url, digest, RAW / "aime" / f"aime2025-{part}.jsonl")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            item = json.loads(line)
            rows.append((task_id(2025, part, n), 2025, item["question"], item["answer"]))
    public: list[dict[str, JsonValue]] = []
    private: list[dict[str, JsonValue]] = []
    excluded: list[JsonValue] = []
    for tid, year, question, answer in sorted(rows):
        ref = normalize_number(answer)
        if ref is None or not ref.isdigit() or not 0 <= int(ref) <= 999:
            excluded.append({"id": tid, "answer": answer})  # e.g. two accepted answers
            continue
        norm = " ".join(question.lower().split())
        group = "aime:" + hashlib.sha256(norm.encode()).hexdigest()[:16]
        public.append({"id": tid, "year": year, "question": question, "group": group})
        private.append({"id": tid, "answer": ref})
    if len({p["id"] for p in public}) != len(public):
        raise ValueError("duplicate AIME ids")
    pub_hash = write_jsonl(PREPARED / PUBLIC, public)
    priv_hash = write_jsonl(PREPARED / PRIVATE, private)
    return write_manifest(
        "aime",
        {
            "source": [HISTORIC[0], Y2024[0], *(u for u, _ in Y2025.values())],
            "licence": "CC0 (1983-2023), MIT (2024, 2025)",
            "raw": {
                HISTORIC[0]: HISTORIC[1],
                Y2024[0]: Y2024[1],
                **{u: d for u, d in Y2025.values()},
            },
            "prepared": {str(PUBLIC): pub_hash, str(PRIVATE): priv_hash},
            "counts": {"tasks": len(public), "first_year": FIRST_YEAR},
            "excluded": excluded,
        },
    )


@dataclass(frozen=True)
class AIMEItem:
    task: PublicTask
    group: str
    year: int
    reference: str


def load(root: Path = PREPARED) -> list[AIMEItem]:
    answers = {str(r["id"]): str(r["answer"]) for r in read_jsonl(root / PRIVATE)}
    items = []
    for r in read_jsonl(root / PUBLIC):
        base = _gsm8k_task(str(r["id"]), str(r["question"]))
        items.append(
            AIMEItem(
                task=base.model_copy(update={"instruction": INSTRUCTION}),
                group=str(r["group"]),
                year=int(str(r["year"])),
                reference=answers[str(r["id"])],
            )
        )
    return items


def grade(item: AIMEItem, answer: JsonValue) -> Grade:
    value = answer.get("answer") if isinstance(answer, dict) else None
    got = normalize_number(value)
    if got is None:
        return Grade(correct=False, detail="no numeric answer")
    if got == item.reference:
        return Grade(correct=True)
    return Grade(correct=False, detail=f"expected {item.reference}, got {got}")
