"""Data preparation: the only code allowed to use the network (decision D16).

Files are downloaded once from pinned upstream revisions, checked against their SHA-256, and turned
into local snapshots. Runs and CI read only the snapshots. Each prepared dataset gets a manifest
recording the upstream source, the revision, and the hash of every raw and prepared file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from pydantic import JsonValue

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data" / "raw"
PREPARED = REPO_ROOT / "data" / "prepared"
MANIFESTS = REPO_ROOT / "data" / "manifests"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, sha256: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless an identical file is already there. Fails if the hash
    does not match, so a changed upstream file can never slip in."""
    if dest.exists() and file_sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # curl rather than Python's HTTP stack: it follows the CDN redirects and falls back between
    # address families reliably; the hash check below is what guarantees the content.
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            "300",
            "--output",
            str(tmp),
            url,
        ],
        check=True,
    )
    got = file_sha256(tmp)
    if got != sha256:
        tmp.unlink()
        raise ValueError(f"{url}: expected sha256 {sha256}, got {got}")
    tmp.replace(dest)
    return dest


def write_jsonl(path: Path, rows: list[dict[str, JsonValue]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return file_sha256(path)


def read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(name: str, manifest: dict[str, JsonValue]) -> Path:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_prepared(name: str) -> dict[str, JsonValue]:
    """Check the prepared files against the committed manifest before any run uses them."""
    manifest: dict[str, JsonValue] = json.loads((MANIFESTS / f"{name}.json").read_text())
    prepared = manifest["prepared"]
    assert isinstance(prepared, dict)
    for rel, digest in prepared.items():
        path = PREPARED / rel
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/prepare_data.py {name}")
        if file_sha256(path) != digest:
            raise ValueError(f"{path} does not match the manifest")
    return manifest
