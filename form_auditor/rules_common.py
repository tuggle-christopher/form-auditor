"""Shared types for the rule engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .settings import Settings
from .template import Box


@dataclass
class Finding:
    section_id: str
    section_title: str
    item_id: str
    item_label: str
    check: str
    status: str                 # PASS | FAIL | REVIEW | SKIP
    observed: str
    expected: str
    reason: str
    confidence: float
    boxes: list[Box] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    marker: Optional[int] = None
    rule_refs: list[str] = field(default_factory=list)
    value_read: str = ""        # raw transcription
    value_normalized: str = ""  # e.g. ISO date, upper-cased id, NA
    mark_type: str = ""         # circle | tick | cancel | written ...
    ocr_engine: str = ""
    ocr_confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def escalate(self, status: str, reason: str) -> None:
        """Raise status (PASS < REVIEW < FAIL); never lower it."""
        order = {"SKIP": 0, "PASS": 1, "REVIEW": 2, "FAIL": 3}
        if order.get(status, 0) > order.get(self.status, 0):
            self.status = status
            self.reason = reason
        elif status == self.status and reason and reason not in self.reason:
            self.reason = f"{self.reason}; {reason}" if self.reason else reason


def classify_presence(ink: float, vis: float, s: Settings) -> tuple[str, float]:
    """-> (state, confidence) with state in filled | blank | uncertain | occluded."""
    if vis < s.min_visible_frac:
        return "occluded", round(vis, 2)
    span = max(s.fill_hi - s.fill_lo, 1e-9)
    if ink >= s.fill_hi:
        return "filled", round(min(1.0, 0.6 + 0.4 * (ink - s.fill_hi) / span), 2)
    if ink <= s.fill_lo:
        return "blank", round(min(1.0, 0.6 + 0.4 * (s.fill_lo - ink) / max(s.fill_lo, 1e-9)), 2)
    return "uncertain", round(0.5 * abs(ink - (s.fill_lo + s.fill_hi) / 2) / (span / 2), 2)
