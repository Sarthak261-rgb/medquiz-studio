"""Automatic checks and the rule that turns them into an auto verdict."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ChecksCfg
from ..schemas import CHECK_FAILURE, FAILURE_PRIORITY, CheckResult, FailureType


def severity(name: str, cfg: ChecksCfg) -> str:
    if name in cfg.hard:
        return "hard"
    if name in cfg.soft:
        return "soft"
    return "info"


@dataclass
class AutoVerdict:
    status: str  # "pass", "fail" or "incomplete"
    failed_hard: list[str] = field(default_factory=list)
    failed_soft: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[FailureType]:
        types = {CHECK_FAILURE[n] for n in self.failed_hard if n in CHECK_FAILURE}
        return [t for t in FAILURE_PRIORITY if t in types]

    @property
    def primary(self) -> FailureType | None:
        f = self.failures
        return f[0] if f else None


def auto_verdict(checks: list[CheckResult], cfg: ChecksCfg, applicable: set[str] | None = None) -> AutoVerdict:
    """A question passes only if every hard check that applies to it passed.
    A judge call that errored counts as missing, not as a pass."""
    by_name = {c.name: c for c in checks}
    hard = [n for n in cfg.hard if applicable is None or n in applicable]
    v = AutoVerdict(status="pass")
    for name in hard:
        c = by_name.get(name)
        if c is None or (c.passed is None and "error" in c.detail):
            v.missing.append(name)
        elif c.passed is False:
            v.failed_hard.append(name)
    v.failed_soft = [n for n in cfg.soft if by_name.get(n) and by_name[n].passed is False]
    if v.failed_hard:
        v.status = "fail"
    elif v.missing:
        v.status = "incomplete"
    return v
