"""JSONRenderer -- machine-readable JSON report output."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from torchmortem.registry import register_renderer
from torchmortem.types import Report


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


@register_renderer
class JSONRenderer:
    """Renders a Report as a self-contained JSON file.

    Useful for CI pipelines, programmatic comparison between runs,
    and feeding into other tools.
    """

    format_name: str = "json"

    def render(self, report: Report, output_path: Path) -> None:
        data = self._report_to_dict(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, indent=2, cls=_NumpyEncoder, ensure_ascii=False),
            encoding="utf-8",
        )

    def _report_to_dict(self, report: Report) -> dict[str, Any]:
        result: dict[str, Any] = {
            "metadata": asdict(report.metadata),
            "executive_summary": report.executive_summary,
            "findings": [self._finding_to_dict(f) for f in report.findings],
            "insights": [asdict(i) for i in report.insights],
            "health_scores": [asdict(hs) for hs in report.health_scores],
        }
        # Include collector states as raw series data (for downstream tools).
        result["collector_data"] = {}
        for name, cs in report.collector_states.items():
            result["collector_data"][name] = {
                "steps": cs.steps,
                "layers": cs.layers,
                "series": cs.series,
                "metadata": cs.metadata,
            }
        return result

    def _finding_to_dict(self, finding: Any) -> dict[str, Any]:
        d = asdict(finding)
        # Convert Severity enum to string for readability.
        d["severity"] = finding.severity.name.lower()
        return d
