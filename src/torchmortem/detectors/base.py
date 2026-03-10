"""Detector protocol -- the interface all detectors implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torchmortem.types import CollectorState, Finding, RunMetadata


@runtime_checkable
class Detector(Protocol):
    """Protocol for diagnostic detectors.

    Detectors consume collector states (post-training) and produce a list
    of Finding objects describing detected pathologies.
    """

    name: str
    required_collectors: list[str]

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        """Analyze collected data and return diagnostic findings."""
        ...
