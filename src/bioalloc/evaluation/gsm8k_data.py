"""GSM8K snapshot, references and grading (Cobbe et al., 2021; MIT licence).

Source: Hugging Face ``openai/gsm8k``, config ``main``, pinned revision below. The public file
holds only questions; the private file holds the reference solutions and is read only by the
grader and, on the training split, the label builder.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from bioalloc.decomposition import PublicTask
from bioalloc.evaluation.data import PREPARED, RAW, fetch, read_jsonl, write_jsonl, write_manifest
from bioalloc.evaluation.toy_data import Grade
from bioalloc.tasks.common import normalize_number
from bioalloc.tasks.gsm8k import ANSWER_SCHEMA, CONTRACT, INSTRUCTION

REPO = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
FILES = {
    "train": (
        "main/train-00000-of-00001.parquet",
        "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    ),
    "test": (
        "main/test-00000-of-00001.parquet",
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    ),
}
PUBLIC = Path("gsm8k/public.jsonl")
PRIVATE = Path("gsm8k/private.jsonl")


def reference_from_solution(solution: str) -> str:
    """The number after '####' in a GSM8K solution, normalized."""
    match = re.search(r"####\s*(.+)$", solution.strip())
    value = normalize_number(match.group(1)) if match else None
    if value is None:
        raise ValueError(f"no final answer in solution: {solution[-80:]!r}")
    return value


def group_id(question: str) -> str:
    """Template group: exact duplicates after whitespace and case normalization share a group."""
    norm = " ".join(question.lower().split())
    return "gsm8k:" + hashlib.sha256(norm.encode()).hexdigest()[:16]


def prepare() -> Path:
    import pyarrow.parquet as pq  # preparation-only dependency (uv group "prep")

    public: list[dict[str, JsonValue]] = []
    private: list[dict[str, JsonValue]] = []
    raw_hashes: dict[str, JsonValue] = {}
    for split, (path, digest) in FILES.items():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
        local = fetch(url, digest, RAW / "gsm8k" / Path(path).name)
        raw_hashes[path] = digest
        table = pq.read_table(local).to_pylist()
        for row, item in enumerate(table):
            task_id = f"gsm8k-{split}-{row:05d}"
            question = str(item["question"])
            solution = str(item["answer"])
            public.append(
                {
                    "id": task_id,
                    "upstream_split": split,
                    "row": row,
                    "question": question,
                    "group": group_id(question),
                }
            )
            private.append(
                {"id": task_id, "solution": solution, "answer": reference_from_solution(solution)}
            )
    pub_hash = write_jsonl(PREPARED / PUBLIC, public)
    priv_hash = write_jsonl(PREPARED / PRIVATE, private)
    return write_manifest(
        "gsm8k",
        {
            "source": f"https://huggingface.co/datasets/{REPO}",
            "config": "main",
            "revision": REVISION,
            "licence": "MIT",
            "raw": raw_hashes,
            "prepared": {str(PUBLIC): pub_hash, str(PRIVATE): priv_hash},
            "counts": {"tasks": len(public), "groups": len({p["group"] for p in public})},
        },
    )


@dataclass(frozen=True)
class GSM8KItem:
    task: PublicTask
    group: str
    upstream_split: str
    reference: str


def public_task(task_id: str, question: str) -> PublicTask:
    return PublicTask(
        id=task_id,
        instruction=INSTRUCTION,
        inputs={"question": question},
        output_schema=ANSWER_SCHEMA,
        allowed_tools=("calculate",),
        contract=CONTRACT,
    )


def load(root: Path = PREPARED) -> list[GSM8KItem]:
    answers = {str(r["id"]): str(r["answer"]) for r in read_jsonl(root / PRIVATE)}
    return [
        GSM8KItem(
            task=public_task(str(r["id"]), str(r["question"])),
            group=str(r["group"]),
            upstream_split=str(r["upstream_split"]),
            reference=answers[str(r["id"])],
        )
        for r in read_jsonl(root / PUBLIC)
    ]


def grade(item: GSM8KItem, answer: JsonValue) -> Grade:
    value = answer.get("answer") if isinstance(answer, dict) else None
    got = normalize_number(value)
    if got is None:
        return Grade(correct=False, detail="no numeric answer")
    if got == item.reference:
        return Grade(correct=True)
    return Grade(correct=False, detail=f"expected {item.reference}, got {got}")
