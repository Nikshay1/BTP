"""Grouped train / tune / test splits (research plan, Section 4.4; Sol's plan, section E).

Splits are made once, before any decomposition or model run, at the level of template groups: every
task of a group lands in the same split, so paraphrases, renamings and duplicates can never sit on
both sides. Groups are ordered by SHA-256(split_version, seed, group id) and whole groups are
assigned in that order until each family's quotas are met. Each family reserves part of its test
quota for one preregistered test-only variant; groups of that variant are used only there.

Quotas per family (plan): 50 training, 30 tuning, 70 test, of which 20 are the test-only variant.
The assignment is saved and committed; later code reads it and never re-derives splits.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from bioalloc.schemas import Wire

SPLIT_VERSION = 1
SEED = 20261026
QUOTAS = {"train": 50, "tune": 30, "test": 50, "test_variant": 20}


class Candidate(Wire):
    task_id: str
    family: str
    group: str
    variant: bool  # belongs to the family's preregistered test-only variant


class Assignment(Wire):
    task_id: str
    family: str
    group: str
    split: str  # train | tune | test | test_variant


@dataclass(frozen=True)
class SplitResult:
    assignments: list[Assignment]
    groups_per_split: dict[str, dict[str, int]]


def _order(group: str) -> str:
    return hashlib.sha256(f"{SPLIT_VERSION}|{SEED}|{group}".encode()).hexdigest()


def assign(candidates: Iterable[Candidate], quotas: dict[str, int] = QUOTAS) -> SplitResult:
    by_family: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_family[c.family].append(c)
    out: list[Assignment] = []
    groups_per_split: dict[str, dict[str, int]] = {}
    for family, cands in sorted(by_family.items()):
        groups: dict[str, list[Candidate]] = defaultdict(list)
        for c in cands:
            groups[c.group].append(c)
        mixed = [g for g, members in groups.items() if len({m.variant for m in members}) > 1]
        if mixed:
            raise ValueError(f"{family}: groups mix variant and ordinary tasks: {mixed[:3]}")
        filled = dict.fromkeys(quotas, 0)
        counted = dict.fromkeys(quotas, 0)
        for g in sorted(groups, key=_order):
            members = sorted(groups[g], key=lambda m: m.task_id)
            variant = members[0].variant
            targets = ["test_variant"] if variant else ["train", "tune", "test"]
            for split in targets:
                if filled[split] + len(members) <= quotas[split]:
                    filled[split] += len(members)
                    counted[split] += 1
                    out += [
                        Assignment(task_id=m.task_id, family=family, group=g, split=split)
                        for m in members
                    ]
                    break
        short = {s: (filled[s], quotas[s]) for s in quotas if filled[s] < quotas[s]}
        if short:
            raise ValueError(f"{family}: not enough whole groups to fill {short}")
        groups_per_split[family] = counted
    return SplitResult(out, groups_per_split)


def canonical(assignments: list[Assignment]) -> str:
    rows = sorted((a.model_dump() for a in assignments), key=lambda r: r["task_id"])
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def save(result: SplitResult, path: Path, provenance: dict[str, JsonValue]) -> str:
    text = canonical(result.assignments)
    digest = hashlib.sha256(text.encode()).hexdigest()
    payload = {
        "split_version": SPLIT_VERSION,
        "seed": SEED,
        "quotas": QUOTAS,
        "sha256": digest,
        "provenance": provenance,
        "groups_per_split": result.groups_per_split,
        "assignments": json.loads(text),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return digest


def load(path: Path) -> list[Assignment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = [Assignment.model_validate(a) for a in payload["assignments"]]
    digest = hashlib.sha256(canonical(assignments).encode()).hexdigest()
    if digest != payload["sha256"]:
        raise ValueError(f"{path} does not match its recorded hash")
    return assignments
