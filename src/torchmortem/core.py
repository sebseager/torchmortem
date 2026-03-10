"""Autopsy -- coordinates collectors, detectors, interpreter, and renderer."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from types import TracebackType
from typing import Any

import torch.nn as nn
import torch.optim as optim

from torchmortem.interpreters.default import DefaultInterpreter
from torchmortem.registry import get_all_collectors, get_all_detectors, get_renderer
from torchmortem.types import (
    CollectorState,
    Finding,
    Report,
    RunMetadata,
    SamplingConfig,
    resolve_sampling,
)

logger = logging.getLogger("torchmortem")


class Autopsy:
    """Drop-in diagnostic hook for PyTorch training loops.

    Usage (context manager)::

        with Autopsy(model, optimizer=optimizer) as autopsy:
            for batch in dataloader:
                loss = model(batch)
                loss.backward()
                optimizer.step()
                autopsy.step(loss=loss.item())

        autopsy.report("report.html")

    Usage (explicit)::

        autopsy = Autopsy(model, optimizer=optimizer)
        autopsy.attach()
        # ... training loop with autopsy.step(loss=...) ...
        autopsy.detach()
        autopsy.report("report.html")

    Args:
        model: The PyTorch model being trained.
        optimizer: The optimizer (optional, used for metadata + some collectors).
        sampling: Sampling preset name ("thorough", "balanced", "fast")
                  or a SamplingConfig instance.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        sampling: str | SamplingConfig | None = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._sampling = resolve_sampling(sampling)

        self._step_count: int = 0
        self._attached: bool = False
        self._report_generated: bool = False
        self._atexit_registered: bool = False

        # Instantiate all registered collectors.
        self._collectors: list[Any] = []
        for name, cls in get_all_collectors().items():
            self._collectors.append(cls())
            logger.debug("Registered collector: %s", name)

        # Instantiate all registered detectors.
        self._detectors: list[Any] = []
        for name, cls in get_all_detectors().items():
            self._detectors.append(cls())
            logger.debug("Registered detector: %s", name)

        # Interpreter (could be made pluggable later).
        self._interpreter = DefaultInterpreter()

        # Cached report.
        self._report: Report | None = None

    # -------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------

    def __enter__(self) -> Autopsy:
        self.attach()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.detach(is_complete=exc_type is None)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def attach(self) -> None:
        """Attach all collectors to the model. Call once before training."""
        if self._attached:
            logger.warning("Autopsy is already attached -- skipping.")
            return

        for collector in self._collectors:
            collector.attach(self._model, self._optimizer, self._sampling)

        self._attached = True
        self._report = None
        self._report_generated = False

        # Register atexit handler for crash resilience.
        if not self._atexit_registered:
            atexit.register(self._atexit_handler)
            self._atexit_registered = True

        logger.info(
            "Autopsy attached (%d collectors, %d detectors)",
            len(self._collectors),
            len(self._detectors),
        )

    def detach(self, is_complete: bool = True) -> None:
        """Detach all collectors and run analysis."""
        if not self._attached:
            return

        for collector in self._collectors:
            collector.detach()

        self._attached = False
        self._run_analysis(is_complete=is_complete)

    def step(self, **kwargs: Any) -> None:
        """Call after each training step.

        Args:
            **kwargs: Arbitrary keyword arguments passed to collectors.
                      Typically ``loss=loss.item()``.
        """
        if not self._attached:
            return

        for collector in self._collectors:
            collector.on_step(self._step_count, **kwargs)

        self._step_count += 1

    # -------------------------------------------------------------------
    # Analysis & reporting
    # -------------------------------------------------------------------

    def _run_analysis(self, is_complete: bool = True) -> None:
        """Run detectors and interpreter on collected data."""
        # Gather collector states.
        collector_states: dict[str, CollectorState] = {}
        for collector in self._collectors:
            state = collector.state()
            collector_states[state.name] = state

        # Build metadata.
        metadata = self._build_metadata(is_complete)

        # Run detectors.
        findings: list[Finding] = []
        for detector in self._detectors:
            # Check that required collectors are present.
            missing = [r for r in detector.required_collectors if r not in collector_states]
            if missing:
                logger.warning(
                    "Detector '%s' skipped -- missing collectors: %s",
                    detector.name,
                    missing,
                )
                continue

            try:
                found = detector.analyze(collector_states, metadata)
                findings.extend(found)
            except Exception:
                logger.exception("Detector '%s' failed", detector.name)

        # Interpret.
        self._report = self._interpreter.interpret(findings, metadata, collector_states)
        self._report_generated = True

        n_crit = sum(1 for f in findings if f.severity.name == "CRITICAL")
        n_warn = sum(1 for f in findings if f.severity.name == "WARNING")
        logger.info(
            "Analysis complete: %d findings (%d critical, %d warnings)",
            len(findings),
            n_crit,
            n_warn,
        )

    def _build_metadata(self, is_complete: bool) -> RunMetadata:
        model_name = self._model.__class__.__name__
        total_params = sum(p.numel() for p in self._model.parameters())

        optimizer_name = ""
        learning_rate: float | None = None
        if self._optimizer is not None:
            optimizer_name = self._optimizer.__class__.__name__
            param_groups = self._optimizer.param_groups
            if param_groups:
                learning_rate = param_groups[0].get("lr")

        # Determine device from first parameter.
        device = ""
        first_param = next(self._model.parameters(), None)
        if first_param is not None:
            device = str(first_param.device)

        # Layer names (from gradient collector if available, else from model).
        layer_names: list[str] = []
        for collector in self._collectors:
            state = collector.state()
            if state.layers:
                layer_names = state.layers
                break
        if not layer_names:
            layer_names = [n for n, _ in self._model.named_modules() if n]

        return RunMetadata(
            model_name=model_name,
            total_steps=self._step_count,
            total_parameters=total_params,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            layer_names=layer_names,
            device=device,
            is_complete=is_complete,
        )

    def report(
        self,
        output_path: str | Path,
        fmt: str | None = None,
    ) -> Report:
        """Generate a report and write it to a file.

        Args:
            output_path: Path to write report to.
            fmt: Output format ("html" or "json"). If None, inferred from extension.

        Returns:
            The Report object.
        """
        path = Path(output_path)

        if self._report is None:
            if self._attached:
                # Still attached -- detach first.
                self.detach()
            else:
                raise RuntimeError(
                    "No report data available. Did you forget to call attach()/step()/detach()?"
                )

        assert self._report is not None

        # Infer format from extension if not specified.
        if fmt is None:
            ext = path.suffix.lower()
            fmt_map = {".html": "html", ".htm": "html", ".json": "json"}
            fmt = fmt_map.get(ext, "html")

        renderer_cls = get_renderer(fmt)
        renderer = renderer_cls()
        renderer.render(self._report, path)

        logger.info("Report written to %s (format: %s)", path, fmt)
        return self._report

    def get_report(self) -> Report | None:
        """Return the report object without writing to disk, or None if not yet generated."""
        return self._report

    # -------------------------------------------------------------------
    # Crash resilience
    # -------------------------------------------------------------------

    def _atexit_handler(self) -> None:
        """Generate partial report if training crashes."""
        if self._attached and not self._report_generated:
            logger.warning("torchmortem: training did not complete -- generating partial report.")
            try:
                self.detach(is_complete=False)
                self.report("autopsy_crash_report.html")
            except Exception:
                logger.exception("Failed to generate crash report")
