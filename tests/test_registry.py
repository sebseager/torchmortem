"""Tests for the plugin registry."""

from __future__ import annotations

import pytest

from torchmortem.registry import (
    _collectors,
    _detectors,
    _renderers,
    get_all_collectors,
    get_all_detectors,
    get_all_renderers,
    get_renderer,
    register_collector,
    register_detector,
    register_renderer,
)


class TestRegistry:
    """Tests for plugin registry."""

    def test_built_in_collectors_registered(self) -> None:
        """Built-in collectors are registered on import."""
        collectors = get_all_collectors()
        assert "gradient" in collectors
        assert "loss" in collectors

    def test_built_in_detectors_registered(self) -> None:
        """Built-in detectors are registered on import."""
        detectors = get_all_detectors()
        assert "gradient_flow" in detectors

    def test_built_in_renderers_registered(self) -> None:
        """Built-in renderers are registered on import."""
        renderers = get_all_renderers()
        assert "html" in renderers
        assert "json" in renderers

    def test_get_renderer_exists(self) -> None:
        """get_renderer returns the correct renderer class."""
        cls = get_renderer("json")
        assert cls.format_name == "json"

    def test_get_renderer_nonexistent(self) -> None:
        """get_renderer raises KeyError for unknown formats."""
        with pytest.raises(KeyError, match="No renderer registered"):
            get_renderer("pdf")

    def test_register_collector_requires_name(self) -> None:
        """Collectors without a name attribute are rejected."""
        with pytest.raises(ValueError, match="must define a 'name'"):

            @register_collector
            class BadCollector:
                pass

    def test_register_detector_requires_name(self) -> None:
        """Detectors without a name attribute are rejected."""
        with pytest.raises(ValueError, match="must define a 'name'"):

            @register_detector
            class BadDetector:
                pass

    def test_register_renderer_requires_format_name(self) -> None:
        """Renderers without a format_name attribute are rejected."""
        with pytest.raises(ValueError, match="must define a 'format_name'"):

            @register_renderer
            class BadRenderer:
                pass

    def test_custom_collector_registration(self) -> None:
        """Custom collectors can be registered."""

        @register_collector
        class MyCollector:
            name = "_test_custom_collector"

        try:
            collectors = get_all_collectors()
            assert "_test_custom_collector" in collectors
        finally:
            _collectors.pop("_test_custom_collector", None)

    def test_custom_detector_registration(self) -> None:
        """Custom detectors can be registered."""

        @register_detector
        class MyDetector:
            name = "_test_custom_detector"

        try:
            detectors = get_all_detectors()
            assert "_test_custom_detector" in detectors
        finally:
            _detectors.pop("_test_custom_detector", None)

    def test_custom_renderer_registration(self) -> None:
        """Custom renderers can be registered."""

        @register_renderer
        class MyRenderer:
            format_name = "_test_custom_format"

        try:
            renderers = get_all_renderers()
            assert "_test_custom_format" in renderers
        finally:
            _renderers.pop("_test_custom_format", None)
