"""Tests for core types."""

from __future__ import annotations

import pytest

from torchmortem.types import (
    CollectorCost,
    SamplingConfig,
    Severity,
    resolve_sampling,
)


class TestSeverity:
    def test_ordering(self) -> None:
        assert Severity.INFO < Severity.WARNING < Severity.CRITICAL

    def test_names(self) -> None:
        assert Severity.INFO.name == "INFO"
        assert Severity.WARNING.name == "WARNING"
        assert Severity.CRITICAL.name == "CRITICAL"


class TestSamplingConfig:
    def test_default_values(self) -> None:
        config = SamplingConfig()
        assert config.default_interval == 1
        assert config.expensive_interval == 50

    def test_from_preset_balanced(self) -> None:
        config = SamplingConfig.from_preset("balanced")
        assert config.default_interval == 1
        assert config.expensive_interval == 50

    def test_from_preset_thorough(self) -> None:
        config = SamplingConfig.from_preset("thorough")
        assert config.default_interval == 1
        assert config.expensive_interval == 20

    def test_from_preset_fast(self) -> None:
        config = SamplingConfig.from_preset("fast")
        assert config.default_interval == 5
        assert config.expensive_interval == 200

    def test_from_preset_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown sampling preset"):
            SamplingConfig.from_preset("nonexistent")

    def test_interval_for_cheap(self) -> None:
        config = SamplingConfig(default_interval=2, expensive_interval=100)
        assert config.interval_for("gradient", CollectorCost.CHEAP) == 2
        assert config.interval_for("loss", CollectorCost.TRIVIAL) == 2

    def test_interval_for_expensive(self) -> None:
        config = SamplingConfig(default_interval=1, expensive_interval=50)
        assert config.interval_for("curvature", CollectorCost.EXPENSIVE) == 50

    def test_interval_override(self) -> None:
        config = SamplingConfig(
            default_interval=1,
            expensive_interval=50,
            overrides={"curvature": 20},
        )
        assert config.interval_for("curvature", CollectorCost.EXPENSIVE) == 20
        assert config.interval_for("gradient", CollectorCost.CHEAP) == 1

    def test_should_collect(self) -> None:
        config = SamplingConfig(default_interval=3)
        assert config.should_collect("gradient", CollectorCost.CHEAP, 0) is True
        assert config.should_collect("gradient", CollectorCost.CHEAP, 1) is False
        assert config.should_collect("gradient", CollectorCost.CHEAP, 2) is False
        assert config.should_collect("gradient", CollectorCost.CHEAP, 3) is True


class TestResolveSampling:
    def test_none_returns_balanced(self) -> None:
        config = resolve_sampling(None)
        assert config.default_interval == 1
        assert config.expensive_interval == 50

    def test_string_preset(self) -> None:
        config = resolve_sampling("thorough")
        assert config.expensive_interval == 20

    def test_config_passthrough(self) -> None:
        original = SamplingConfig(default_interval=7, expensive_interval=77)
        config = resolve_sampling(original)
        assert config is original

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected str"):
            resolve_sampling(42)  # type: ignore[arg-type]
