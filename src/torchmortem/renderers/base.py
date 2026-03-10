"""Renderer protocol -- the interface all renderers implement."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from torchmortem.types import Report


@runtime_checkable
class Renderer(Protocol):
    """Protocol for report renderers.

    Each renderer takes a Report and writes it to a file in a specific format.
    """

    format_name: str

    def render(self, report: Report, output_path: Path) -> None:
        """Render the report to the given output path."""
        ...
