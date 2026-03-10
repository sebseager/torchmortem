"""Interpreters package."""

from torchmortem.interpreters.default import DefaultInterpreter

import torchmortem.interpreters.rules  # noqa: F401  -- register built-in rules

__all__ = ["DefaultInterpreter"]
