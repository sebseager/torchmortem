"""Interpreter protocol -- the interface all interpreters implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torchmortem.types import Finding, Report, RunMetadata, CollectorState


@runtime_checkable
class Interpreter(Protocol):
    """Protocol for interpreters.

    An interpreter takes a list of findings from all detectors, plus run
    metadata and collector states, and produces a Report with an executive
    summary, cross-signal insights, and prioritized findings.
    """

    def interpret(
        self,
        findings: list[Finding],
        metadata: RunMetadata,
        collector_states: dict[str, CollectorState],
    ) -> Report:
        """Synthesize findings into a complete report."""
        ...
