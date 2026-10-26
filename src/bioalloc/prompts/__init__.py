"""Prompt templates shared by all systems. Their hashes go into every run manifest."""

from __future__ import annotations

from importlib.resources import files


def load(name: str) -> str:
    return (files(__package__) / f"{name}.txt").read_text(encoding="utf-8")


def all_prompts() -> dict[str, str]:
    return {
        p.name.removesuffix(".txt"): p.read_text(encoding="utf-8")
        for p in files(__package__).iterdir()
        if p.name.endswith(".txt")
    }
