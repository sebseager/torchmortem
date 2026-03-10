"""Plugin registry for collectors, detectors, interpreters, and renderers."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Global registries (simple dicts keyed by name)
# ---------------------------------------------------------------------------

_collectors: dict[str, type] = {}
_detectors: dict[str, type] = {}
_renderers: dict[str, type] = {}
_correlation_rules: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------


def register_collector(cls: type) -> type:
    """Register a collector class. Must have a ``name`` class attribute."""
    name = getattr(cls, "name", None)
    if name is None:
        raise ValueError(f"Collector {cls.__name__} must define a 'name' class attribute")
    _collectors[name] = cls
    return cls


def register_detector(cls: type) -> type:
    """Register a detector class. Must have a ``name`` class attribute."""
    name = getattr(cls, "name", None)
    if name is None:
        raise ValueError(f"Detector {cls.__name__} must define a 'name' class attribute")
    _detectors[name] = cls
    return cls


def register_renderer(cls: type) -> type:
    """Register a renderer class. Must have a ``format_name`` class attribute."""
    fmt = getattr(cls, "format_name", None)
    if fmt is None:
        raise ValueError(f"Renderer {cls.__name__} must define a 'format_name' class attribute")
    _renderers[fmt] = cls
    return cls


def register_correlation_rule(rule: Any) -> Any:
    """Register a correlation rule instance. Must have a ``name`` attribute."""
    name = getattr(rule, "name", None)
    if name is None:
        raise ValueError("CorrelationRule must have a 'name' attribute")
    _correlation_rules[name] = rule
    return rule


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_all_collectors() -> dict[str, type]:
    return dict(_collectors)


def get_all_detectors() -> dict[str, type]:
    return dict(_detectors)


def get_all_renderers() -> dict[str, type]:
    return dict(_renderers)


def get_renderer(fmt: str) -> type:
    if fmt not in _renderers:
        available = ", ".join(sorted(_renderers)) or "(none registered)"
        raise KeyError(f"No renderer registered for format {fmt!r}. Available: {available}")
    return _renderers[fmt]


def get_all_correlation_rules() -> dict[str, Any]:
    return dict(_correlation_rules)
