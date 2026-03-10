"""Core data types for torchmortem."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Severity(enum.IntEnum):
    """Finding severity level, ordered by importance."""

    INFO = 0
    WARNING = 1
    CRITICAL = 2


class CollectorCost(enum.Enum):
    """How expensive a collector is to run each step."""

    TRIVIAL = "trivial"
    CHEAP = "cheap"
    EXPENSIVE = "expensive"


# ---------------------------------------------------------------------------
# Sampling configuration
# ---------------------------------------------------------------------------

# Preset name -> (default_interval, expensive_interval)
_SAMPLING_PRESETS: dict[str, tuple[int, int]] = {
    "thorough": (1, 20),
    "balanced": (1, 50),
    "fast": (5, 200),
}


@dataclass(frozen=True)
class SamplingConfig:
    """Controls how frequently each collector records data.

    Collectors declare their cost tier (trivial/cheap/expensive).
    This config maps cost tiers to step intervals, with per-collector overrides.

    Args:
        default_interval: Step interval for trivial/cheap collectors.
        expensive_interval: Step interval for expensive collectors.
        overrides: Per-collector name -> step interval override.
    """

    default_interval: int = 1
    expensive_interval: int = 50
    overrides: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_preset(cls, name: str) -> SamplingConfig:
        if name not in _SAMPLING_PRESETS:
            valid = ", ".join(sorted(_SAMPLING_PRESETS))
            raise ValueError(f"Unknown sampling preset {name!r}. Choose from: {valid}")
        default_iv, expensive_iv = _SAMPLING_PRESETS[name]
        return cls(default_interval=default_iv, expensive_interval=expensive_iv)

    def interval_for(self, collector_name: str, cost: CollectorCost) -> int:
        """Return the step interval for a given collector."""
        if collector_name in self.overrides:
            return self.overrides[collector_name]
        if cost == CollectorCost.EXPENSIVE:
            return self.expensive_interval
        return self.default_interval

    def should_collect(self, collector_name: str, cost: CollectorCost, step: int) -> bool:
        """Return True if the collector should record at this step."""
        interval = self.interval_for(collector_name, cost)
        return step % interval == 0


def resolve_sampling(sampling: str | SamplingConfig | None) -> SamplingConfig:
    """Resolve a sampling argument into a SamplingConfig."""
    if sampling is None:
        return SamplingConfig.from_preset("balanced")
    if isinstance(sampling, str):
        return SamplingConfig.from_preset(sampling)
    if isinstance(sampling, SamplingConfig):
        return sampling
    raise TypeError(f"Expected str, SamplingConfig, or None -- got {type(sampling).__name__}")


# ---------------------------------------------------------------------------
# Reference (for citations in findings)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """A citation to a paper, blog post, or documentation page."""

    title: str
    url: str = ""
    authors: str = ""
    year: int | None = None

    def __str__(self) -> str:
        parts = []
        if self.authors:
            parts.append(self.authors)
        parts.append(self.title)
        if self.year:
            parts.append(f"({self.year})")
        return " -- ".join(parts)


# ---------------------------------------------------------------------------
# CollectorState -- per-collector output
# ---------------------------------------------------------------------------


@dataclass
class CollectorState:
    """Container for data recorded by a single collector.

    Attributes:
        name: Collector name (matches the collector that produced this).
        steps: 1-D array of step numbers at which data was recorded.
        layers: Ordered list of layer names that were tracked.
        series: Mapping of metric_name -> numpy array.
            For per-layer metrics: shape (num_steps, num_layers).
            For scalar metrics: shape (num_steps,).
        metadata: Arbitrary extra info from the collector.
    """

    name: str
    steps: np.ndarray  # (num_steps,)
    layers: list[str] = field(default_factory=list)
    series: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Finding -- output of a single detector
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single diagnostic finding produced by a detector.

    Attributes:
        detector: Name of the detector that produced this.
        severity: How serious this finding is.
        category: Grouping key (e.g. "gradient_flow", "dead_units").
        title: Short human-readable title.
        summary: One-line summary.
        detail: Multi-paragraph explanation with ML context.
        evidence: Raw data backing the finding (used by renderers for charts).
        affected_layers: Which layers are involved.
        step_range: (start_step, end_step) when this was observed.
        remediation: Actionable suggestions.
        references: Citations to papers/blogs.
    """

    detector: str
    severity: Severity
    category: str
    title: str
    summary: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    affected_layers: list[str] = field(default_factory=list)
    step_range: tuple[int, int] = (0, 0)
    remediation: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Insight -- cross-signal correlation produced by the interpreter
# ---------------------------------------------------------------------------


@dataclass
class Insight:
    """A cross-signal insight synthesized from multiple findings.

    Attributes:
        rule_name: Name of the correlation rule that produced this.
        title: Short human-readable title.
        summary: One-line summary.
        detail: Multi-paragraph explanation.
        contributing_findings: The findings that contributed to this insight.
        remediation: Prioritized actionable suggestions.
        references: Citations.
    """

    rule_name: str
    title: str
    summary: str
    detail: str
    contributing_findings: list[Finding] = field(default_factory=list)
    remediation: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CorrelationRule -- data-driven cross-signal interpretation
# ---------------------------------------------------------------------------


@dataclass
class CorrelationRule:
    """A rule that correlates findings from multiple detectors.

    The rule engine matches rules against the set of findings, evaluates
    a condition predicate, and, if true, synthesizes a cross-signal insight.

    Attributes:
        name: Unique rule name.
        description: Human-readable description.
        required_categories: Finding categories that must be present for this rule.
        priority: Higher priority rules are evaluated first.
        condition: Predicate that checks if the matched findings actually correlate.
        synthesize: Produces an Insight from the matched findings.
    """

    name: str
    description: str
    required_categories: list[str]
    priority: int = 0
    condition: Any = None  # Callable[[list[Finding]], bool]
    synthesize: Any = None  # Callable[[list[Finding]], Insight]


# ---------------------------------------------------------------------------
# HealthScore -- per-layer health assessment
# ---------------------------------------------------------------------------


@dataclass
class HealthScore:
    """Normalized health score for a layer.

    Score range: 0.0 (many critical issues) to 1.0 (perfectly healthy).
    """

    layer_name: str
    score: float
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RunMetadata -- context about the training run
# ---------------------------------------------------------------------------


@dataclass
class RunMetadata:
    """Metadata about the training run, gathered by the Autopsy orchestrator."""

    model_name: str = ""
    total_steps: int = 0
    total_parameters: int = 0
    optimizer_name: str = ""
    learning_rate: float | None = None
    layer_names: list[str] = field(default_factory=list)
    device: str = ""
    is_complete: bool = True  # False if training crashed


# ---------------------------------------------------------------------------
# Report -- final output of the interpreter
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """Complete diagnostic report, ready to be rendered.

    Attributes:
        metadata: Info about the training run.
        executive_summary: 3-5 sentence overall assessment.
        findings: All findings from all detectors, sorted by severity.
        insights: Cross-signal insights from the interpreter.
        health_scores: Per-layer health scores.
        collector_states: Raw collector data (for charts).
    """

    metadata: RunMetadata
    executive_summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    health_scores: list[HealthScore] = field(default_factory=list)
    collector_states: dict[str, CollectorState] = field(default_factory=dict)
