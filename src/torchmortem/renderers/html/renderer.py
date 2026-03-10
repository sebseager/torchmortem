"""HTMLRenderer -- self-contained static HTML report with Plotly charts."""

from __future__ import annotations

import json
from pathlib import Path

import jinja2

from torchmortem.registry import register_renderer
from torchmortem.types import CollectorState, HealthScore, Report, Severity


def _severity_badge(severity: Severity) -> str:
    colors = {
        Severity.CRITICAL: ("#dc2626", "#fef2f2"),
        Severity.WARNING: ("#d97706", "#fffbeb"),
        Severity.INFO: ("#2563eb", "#eff6ff"),
    }
    fg, bg = colors.get(severity, ("#6b7280", "#f3f4f6"))
    label = severity.name
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:9999px;'
        f'font-size:0.75rem;font-weight:600;color:{fg};background:{bg};">'
        f"{label}</span>"
    )


def _build_gradient_chart_json(
    collector_states: dict[str, CollectorState],
) -> str | None:
    """Build Plotly JSON data for gradient norm chart."""
    grad = collector_states.get("gradient")
    if grad is None:
        return None

    norms = grad.series.get("grad_norm")
    if norms is None or len(grad.steps) == 0:
        return None

    traces = []
    for layer_idx, layer_name in enumerate(grad.layers):
        traces.append(
            {
                "x": grad.steps.tolist(),
                "y": norms[:, layer_idx].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": layer_name,
                "line": {"width": 1.5},
            }
        )

    layout = {
        "title": {"text": "Gradient L2 Norm by Layer"},
        "xaxis": {"title": {"text": "Step"}},
        "yaxis": {"title": {"text": "Gradient Norm"}, "type": "log"},
        "template": "plotly_white",
        "height": 500,
        "margin": {"b": 150},
        "legend": {
            "orientation": "h",
            "y": -0.25,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
        },
    }

    return json.dumps({"data": traces, "layout": layout})


def _build_loss_chart_json(collector_states: dict[str, CollectorState]) -> str | None:
    """Build Plotly JSON data for loss curve chart."""
    loss = collector_states.get("loss")
    if loss is None:
        return None

    loss_vals = loss.series.get("loss")
    smoothed = loss.series.get("loss_smoothed")
    if loss_vals is None or len(loss.steps) == 0:
        return None

    traces = [
        {
            "x": loss.steps.tolist(),
            "y": loss_vals.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Loss",
            "line": {"width": 1, "color": "rgba(99, 102, 241, 0.3)"},
        },
    ]
    if smoothed is not None:
        traces.append(
            {
                "x": loss.steps.tolist(),
                "y": smoothed.tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "Smoothed Loss",
                "line": {"width": 2, "color": "#6366f1"},
            }
        )

    layout = {
        "title": {"text": "Training Loss"},
        "xaxis": {"title": {"text": "Step"}},
        "yaxis": {"title": {"text": "Loss"}},
        "template": "plotly_white",
        "height": 350,
    }

    return json.dumps({"data": traces, "layout": layout})


def _build_weight_norm_chart_json(
    collector_states: dict[str, CollectorState],
) -> str | None:
    """Build Plotly JSON data for weight norm chart."""
    weight = collector_states.get("weight")
    if weight is None:
        return None

    norms = weight.series.get("weight_norm")
    if norms is None or len(weight.steps) == 0:
        return None

    traces = []
    for layer_idx, layer_name in enumerate(weight.layers):
        traces.append(
            {
                "x": weight.steps.tolist(),
                "y": norms[:, layer_idx].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": layer_name,
                "line": {"width": 1.5},
            }
        )

    layout = {
        "title": {"text": "Weight Norm by Layer"},
        "xaxis": {"title": {"text": "Step"}},
        "yaxis": {"title": {"text": "Weight Norm (L2)"}},
        "template": "plotly_white",
        "height": 500,
        "margin": {"b": 150},
        "legend": {
            "orientation": "h",
            "y": -0.25,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
        },
    }

    return json.dumps({"data": traces, "layout": layout})


def _build_update_ratio_chart_json(
    collector_states: dict[str, CollectorState],
) -> str | None:
    """Build Plotly JSON for update-to-weight ratio."""
    weight = collector_states.get("weight")
    if weight is None:
        return None

    ratios = weight.series.get("update_ratio")
    if ratios is None or len(weight.steps) == 0:
        return None

    traces = []
    for layer_idx, layer_name in enumerate(weight.layers):
        traces.append(
            {
                "x": weight.steps.tolist(),
                "y": ratios[:, layer_idx].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": layer_name,
                "line": {"width": 1.5},
            }
        )

    # Add reference band for the ~1e-3 healthy zone
    traces.append(
        {
            "x": [weight.steps[0].item(), weight.steps[-1].item()],
            "y": [1e-3, 1e-3],
            "type": "scatter",
            "mode": "lines",
            "name": "Target (~1e-3)",
            "line": {"width": 2, "dash": "dash", "color": "#10b981"},
        }
    )

    layout = {
        "title": {"text": "Update / Weight Ratio by Layer"},
        "xaxis": {"title": {"text": "Step"}},
        "yaxis": {"title": {"text": "Ratio"}, "type": "log"},
        "template": "plotly_white",
        "height": 500,
        "margin": {"b": 150},
        "legend": {
            "orientation": "h",
            "y": -0.25,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
        },
    }

    return json.dumps({"data": traces, "layout": layout})


def _build_activation_chart_json(
    collector_states: dict[str, CollectorState],
) -> str | None:
    """Build Plotly JSON for dead-unit fraction over training."""
    act = collector_states.get("activation")
    if act is None:
        return None

    dead = act.series.get("act_dead_frac")
    if dead is None or len(act.steps) == 0:
        return None

    traces = []
    for layer_idx, layer_name in enumerate(act.layers):
        traces.append(
            {
                "x": act.steps.tolist(),
                "y": dead[:, layer_idx].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": layer_name,
                "line": {"width": 1.5},
            }
        )

    layout = {
        "title": {"text": "Dead Unit Fraction by Layer"},
        "xaxis": {"title": {"text": "Step"}},
        "yaxis": {"title": {"text": "Fraction Dead"}, "range": [0, 1]},
        "template": "plotly_white",
        "height": 500,
        "margin": {"b": 150},
        "legend": {
            "orientation": "h",
            "y": -0.25,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
        },
    }

    return json.dumps({"data": traces, "layout": layout})


def _build_rank_chart_json(
    collector_states: dict[str, CollectorState],
) -> str | None:
    """Build Plotly JSON for effective rank over training."""
    rank = collector_states.get("rank")
    if rank is None:
        return None

    eff_rank = rank.series.get("effective_rank")
    if eff_rank is None or len(rank.steps) == 0:
        return None

    traces = []
    for layer_idx, layer_name in enumerate(rank.layers):
        traces.append(
            {
                "x": rank.steps.tolist(),
                "y": eff_rank[:, layer_idx].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": layer_name,
                "line": {"width": 1.5},
            }
        )

    layout = {
        "title": {"text": "Effective Rank by Layer"},
        "xaxis": {"title": {"text": "Step"}},
        "yaxis": {"title": {"text": "Effective Rank"}},
        "template": "plotly_white",
        "height": 500,
        "margin": {"b": 150},
        "legend": {
            "orientation": "h",
            "y": -0.25,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
        },
    }

    return json.dumps({"data": traces, "layout": layout})


def _build_health_heatmap_json(health_scores: list[HealthScore]) -> str | None:
    """Build Plotly JSON for per-layer health heatmap."""
    if not health_scores:
        return None

    layers = [hs.layer_name for hs in health_scores]
    scores = [hs.score for hs in health_scores]

    # Build colorscale: red (0) -> yellow (0.5) -> green (1)
    traces = [
        {
            "z": [scores],
            "x": layers,
            "y": ["Health"],
            "type": "heatmap",
            "colorscale": [
                [0, "#dc2626"],
                [0.5, "#facc15"],
                [1.0, "#22c55e"],
            ],
            "zmin": 0,
            "zmax": 1,
            "text": [[f"{s:.0%}" for s in scores]],
            "texttemplate": "%{text}",
            "textangle": -90,
            "textfont": {"size": 10},
            "hovertemplate": "%{x}: %{z:.2f}<extra></extra>",
            "showscale": True,
            "colorbar": {
                "title": {"text": "Score", "side": "right"},
                "orientation": "h",
                "y": 1.15,
                "len": 0.4,
                "thickness": 12,
                "xanchor": "center",
                "yanchor": "bottom",
                # "yref": "paper",
                "x": 0.5,
            },
        }
    ]

    layout = {
        "title": {"text": "Layer Health"},
        "xaxis": {"title": {"text": ""}, "tickangle": -45},
        "yaxis": {"visible": False},
        "template": "plotly_white",
        "height": 300,
        "margin": {"t": 80, "b": 120},
    }

    return json.dumps({"data": traces, "layout": layout})


@register_renderer
class HTMLRenderer:
    """Renders a Report as a self-contained static HTML file.

    Uses Jinja2 templates and embeds Plotly.js from CDN for interactive charts.
    All CSS is inlined for portability -- the output is a single HTML file.
    """

    format_name: str = "html"

    def __init__(self) -> None:
        template_dir = Path(__file__).parent / "templates"
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        self._env.globals["severity_badge"] = _severity_badge

    def render(self, report: Report, output_path: Path) -> None:
        template = self._env.get_template("report.html.j2")

        gradient_chart = _build_gradient_chart_json(report.collector_states)
        loss_chart = _build_loss_chart_json(report.collector_states)
        weight_norm_chart = _build_weight_norm_chart_json(report.collector_states)
        update_ratio_chart = _build_update_ratio_chart_json(report.collector_states)
        activation_chart = _build_activation_chart_json(report.collector_states)
        rank_chart = _build_rank_chart_json(report.collector_states)
        health_heatmap = _build_health_heatmap_json(report.health_scores)

        html = template.render(
            report=report,
            findings=report.findings,
            insights=report.insights,
            health_scores=report.health_scores,
            metadata=report.metadata,
            gradient_chart_json=gradient_chart,
            loss_chart_json=loss_chart,
            weight_norm_chart_json=weight_norm_chart,
            update_ratio_chart_json=update_ratio_chart,
            activation_chart_json=activation_chart,
            rank_chart_json=rank_chart,
            health_heatmap_json=health_heatmap,
            Severity=Severity,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
