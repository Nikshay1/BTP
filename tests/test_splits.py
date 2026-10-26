import pytest

from bioalloc.evaluation.splits import Candidate, assign, canonical


def cands(family: str, n_groups: int, size: int = 1, variant_every: int = 0) -> list[Candidate]:
    out = []
    for g in range(n_groups):
        variant = bool(variant_every) and g % variant_every == 0
        out += [
            Candidate(
                task_id=f"{family}-{g}-{i}", family=family, group=f"{family}:g{g}", variant=variant
            )
            for i in range(size)
        ]
    return out


QUOTAS = {"train": 5, "tune": 3, "test": 5, "test_variant": 2}


def test_quotas_met_and_groups_never_cross_splits() -> None:
    result = assign(
        cands("f", 60, size=1, variant_every=5) + cands("g", 40, size=1, variant_every=4), QUOTAS
    )
    for fam in ("f", "g"):
        counts = {
            s: sum(1 for a in result.assignments if a.family == fam and a.split == s)
            for s in QUOTAS
        }
        assert counts == QUOTAS
    split_of: dict[str, set[str]] = {}
    for a in result.assignments:
        split_of.setdefault(a.group, set()).add(a.split)
    assert all(len(s) == 1 for s in split_of.values())


def test_variant_groups_only_in_variant_split() -> None:
    cs = cands("f", 60, variant_every=5)
    variant_groups = {c.group for c in cs if c.variant}
    result = assign(cs, QUOTAS)
    for a in result.assignments:
        assert (a.group in variant_groups) == (a.split == "test_variant")


def test_assignment_is_deterministic_and_order_independent() -> None:
    cs = cands("f", 60, size=2, variant_every=5)
    a = canonical(assign(cs, {"train": 6, "tune": 4, "test": 6, "test_variant": 2}).assignments)
    b = canonical(
        assign(
            list(reversed(cs)), {"train": 6, "tune": 4, "test": 6, "test_variant": 2}
        ).assignments
    )
    assert a == b


def test_whole_groups_are_never_cut_and_shortfalls_are_errors() -> None:
    with pytest.raises(ValueError, match="not enough"):
        assign(cands("f", 4, size=3, variant_every=2), QUOTAS)
    mixed = [
        Candidate(task_id="t1", family="f", group="g", variant=True),
        Candidate(task_id="t2", family="f", group="g", variant=False),
    ]
    with pytest.raises(ValueError, match="mix variant"):
        assign(mixed, QUOTAS)
