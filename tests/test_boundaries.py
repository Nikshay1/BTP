"""Systems under test must not be able to reach reference answers (decision D5)."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "bioalloc"
HARNESS_ONLY = ("bioalloc.evaluation",)
SYSTEM_MODULES = [
    p
    for p in SRC.rglob("*.py")
    if p.relative_to(SRC).parts[0] not in ("evaluation", "cli.py", "harness.py")
]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_system_modules_never_import_the_evaluation_harness() -> None:
    assert SYSTEM_MODULES, "no modules found"
    offenders = {
        str(p.relative_to(SRC)): sorted(
            m for m in imported_modules(p) if m.startswith(HARNESS_ONLY)
        )
        for p in SYSTEM_MODULES
    }
    assert {k: v for k, v in offenders.items() if v} == {}
