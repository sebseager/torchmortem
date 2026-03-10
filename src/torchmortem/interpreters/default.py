"""DefaultInterpreter -- rule-engine-based cross-signal interpretation.

Evaluates registered CorrelationRule objects against the set of findings,
generates cross-signal insights, computes per-layer health scores,
and produces an executive summary.
"""

from __future__ import annotations

from collections import defaultdict

from torchmortem.registry import get_all_correlation_rules
from torchmortem.types import (
    CollectorState,
    Finding,
    HealthScore,
    Insight,
    Report,
    RunMetadata,
    Severity,
)


class DefaultInterpreter:
    """Default interpreter with a correlation rule engine.

    Processing pipeline:
        1. Sort findings by severity.
        2. Match registered CorrelationRules against finding categories.
        3. For matched rules, evaluate condition predicate.
        4. For passing rules, call synthesize to produce Insights.
        5. Compute per-layer health scores.
        6. Generate executive summary.
    """

    def interpret(
        self,
        findings: list[Finding],
        metadata: RunMetadata,
        collector_states: dict[str, CollectorState],
    ) -> Report:
        sorted_findings = sorted(findings, key=lambda f: f.severity, reverse=True)

        # Build category -> findings index
        by_category: dict[str, list[Finding]] = defaultdict(list)
        for f in sorted_findings:
            by_category[f.category].append(f)

        # Run correlation rules
        insights = self._run_correlation_rules(sorted_findings, by_category)

        # Compute per-layer health scores
        health_scores = self._compute_health_scores(sorted_findings, metadata)

        # Generate executive summary (now includes insights)
        summary = self._generate_summary(sorted_findings, insights, metadata)

        return Report(
            metadata=metadata,
            executive_summary=summary,
            findings=sorted_findings,
            insights=insights,
            health_scores=health_scores,
            collector_states=collector_states,
        )

    def _run_correlation_rules(
        self,
        findings: list[Finding],
        by_category: dict[str, list[Finding]],
    ) -> list[Insight]:
        """Evaluate all registered correlation rules."""
        rules = get_all_correlation_rules()
        if not rules:
            return []

        # Sort rules by priority (highest first)
        sorted_rules = sorted(rules.values(), key=lambda r: r.priority, reverse=True)

        insights: list[Insight] = []
        # Track which findings have been "claimed" by a rule
        claimed_findings: set[int] = set()

        for rule in sorted_rules:
            # Check if required categories are all present
            if not all(cat in by_category for cat in rule.required_categories):
                continue

            # Gather matching findings
            matched: list[Finding] = []
            for cat in rule.required_categories:
                matched.extend(by_category[cat])

            # Skip if all matched findings claimed by higher-priority rules
            unclaimed = [f for f in matched if id(f) not in claimed_findings]
            if not unclaimed:
                continue

            # Step 2: Evaluate condition
            if rule.condition is not None:
                try:
                    if not rule.condition(matched):
                        continue
                except Exception:
                    continue

            # Step 3: Synthesize insight
            if rule.synthesize is not None:
                try:
                    insight = rule.synthesize(matched)
                    insights.append(insight)
                    # Claim matched findings
                    for f in matched:
                        claimed_findings.add(id(f))
                except Exception:
                    continue

        return insights

    def _compute_health_scores(
        self,
        findings: list[Finding],
        metadata: RunMetadata,
    ) -> list[HealthScore]:
        """Compute a 0-1 health score per layer based on findings."""
        layer_names = metadata.layer_names
        if not layer_names:
            return []

        # Aggregate penalties per layer
        layer_penalties: dict[str, float] = defaultdict(float)
        layer_issues: dict[str, list[str]] = defaultdict(list)

        severity_penalties = {
            Severity.CRITICAL: 0.5,
            Severity.WARNING: 0.2,
            Severity.INFO: 0.05,
        }

        for finding in findings:
            penalty = severity_penalties.get(finding.severity, 0.1)
            affected = finding.affected_layers if finding.affected_layers else layer_names
            for layer in affected:
                if layer in layer_names or not finding.affected_layers:
                    layer_penalties[layer] += penalty
                    layer_issues[layer].append(finding.title)

        scores = []
        for layer in layer_names:
            penalty = layer_penalties.get(layer, 0.0)
            score = max(0.0, 1.0 - penalty)
            scores.append(
                HealthScore(
                    layer_name=layer,
                    score=round(score, 2),
                    issues=layer_issues.get(layer, []),
                )
            )

        return scores

    def _generate_summary(
        self,
        findings: list[Finding],
        insights: list[Insight],
        metadata: RunMetadata,
    ) -> str:
        if not findings:
            return (
                "No significant issues were detected during this training run. "
                "All monitored signals (gradient flow, loss dynamics, etc.) "
                "appear healthy."
            )

        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        infos = [f for f in findings if f.severity == Severity.INFO]

        parts: list[str] = []

        # Opening line
        model_desc = metadata.model_name or "the model"
        total = metadata.total_steps
        parts.append(
            f"Training run of {model_desc} ({total} steps) completed "
            f"{'with issues' if critical or warnings else 'cleanly'}."
        )

        if critical:
            titles = [f.title for f in critical]
            parts.append(
                f"CRITICAL: {len(critical)} critical issue(s) found: {'; '.join(titles)}."
            )

        if warnings:
            parts.append(f"{len(warnings)} warning(s) detected.")

        if infos and not critical and not warnings:
            parts.append(f"{len(infos)} informational note(s).")

        # Include cross-signal insights
        if insights:
            insight_titles = [i.title for i in insights[:2]]
            parts.append(f"Cross-signal analysis: {'; '.join(insight_titles)}.")

        # Top recommendation
        if findings:
            top = findings[0]  # Highest severity
            if top.remediation:
                parts.append(f"Top recommendation: {top.remediation[0]}")

        return " ".join(parts)
