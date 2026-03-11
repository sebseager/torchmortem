# 🩻 torchmortem

**Diagnostic autopsy for PyTorch training runs.**

`torchmortem` hooks into your PyTorch training loop and produces a diagnostic 
report (an "autopsy", if you will) telling you why your training might be broken 
and how to fix it.

If you need full-fledged experiment tracking, hyperparameter sweeps, or collaborative 
dashboards, this is probably not the right tool. Instead, go look at platforms like
[Weights & Biases](https://wandb.ai) or 
[TensorBoard](https://www.tensorflow.org/tensorboard).

## Installation

Install into a virtual environment with [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or install from PyPI (once published):

```bash
pip install torchmortem
```

Requires Python >=3.10 and PyTorch >=2.0.

## Quick Start

```python
from torchmortem import Autopsy

with Autopsy(model, optimizer=optimizer) as autopsy:
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            autopsy.step(loss=loss.item())

autopsy.report("autopsy_report.html")
```

## How it works

`torchmortem` is built using a plugin architecture for maximum extensibility.

- **Collectors** (implementing `Collector` in `collectors/base.py`) attach PyTorch hooks to a model and record raw signals during training.
- **Detectors** (implementing `Detector` in `detectors/base.py`) analyze the collected signals and return **findings**.
- The **interpreter** (`DefaultInterpreter` in `interpreters/default.py`, override-able via the protocol in `interpreters/base.py`) synthesizes the findings from all detectors by applying the **rules** defined in `interpreters/rules`.
- The interpreter produces human-readable reports using **renderers** (defined in `renderers/`).

## Built-in feature set

**Individual detectors:**
- [x] **Vanishing / exploding gradients** -- inter-layer gradient ratio analysis
- [x] **Dead units** -- persistently inactive neurons (dead ReLU problem)
- [x] **Activation saturation** -- sigmoid/tanh layers stuck in flat regions
- [x] **Unhealthy update ratios** -- ||update||/||weight|| deviating from ~1e-3
- [x] **Loss dynamics** -- catapult phase, edge-of-stability, plateaus, divergence
- [x] **Rank collapse** -- representation dimensionality shrinking over training
- [x] **Weight norm pathologies** -- explosion, stagnation, inter-layer imbalance
- [x] **Gradient noise** -- SNR and batch size efficiency

**Cross-signal insights** (correlation rules):
- [x] **Gradient starvation** -- vanishing gradients + dead units
- [x] **Instability feedback loop** -- exploding gradients + weight explosion
- [x] **Representation bottleneck** -- rank collapse + loss stagnation
- [x] **Curvature traps** -- edge-of-stability + plateau

**Report features**:
- [x] **Executive summary** -- 3-5 sentence assessment with the top recommendation
- [x] **Per-layer health scores** -- 0-1 score for each layer, visualized as a heatmap
- [x] **Interactive charts** -- loss curve, gradient norms, weight norms, update ratios, dead unit fractions, effective rank
- [x] **Cross-signal insights** -- root-cause explanations synthesized from multiple detectors
- [x] **Findings** -- each with severity, explanation, affected layers, remediation, and references
- [x] **JSON output** -- for CI pipelines and programmatic analysis

## Sampling Configuration

Control the overhead/detail tradeoff:

```python
# Presets
Autopsy(model, sampling="thorough")  # max detail
Autopsy(model, sampling="balanced")  # default
Autopsy(model, sampling="fast")      # minimal overhead

# Granular control
from torchmortem import SamplingConfig
Autopsy(model, sampling=SamplingConfig(
    default_interval=1,
    expensive_interval=50,
    overrides={"curvature": 20},
))
```

## Examples

See the `examples/` directory:
- **`basic_mlp.py`** -- Deep MLP with sigmoid activations (vanishing gradients, dead units)
- **`healthy_resnet.py`** -- Well-configured residual network
- **`transformer_debug.py`** -- Transformer with high LR and no clipping
- **`cnn_overfit.py`** -- Small CNN that overfits on a toy image dataset
- **`lstm_vanishing.py`** -- Vanilla LSTM with extreme sequence length

## Contributing

Contributions are welcome! `torchmortem`'s plugin architecture aims to make it 
relatively easy for contributors to add new features. Please refer to 
[CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Custom detector example

Here is a complete example of a custom detector that flags any layer whose
gradient norm exceeds a configurable threshold:

```python
import numpy as np
from torchmortem.registry import register_detector
from torchmortem.types import CollectorState, Finding, RunMetadata, Severity


@register_detector
class LargeGradientDetector:
    """Flags layers where the gradient norm exceeds a fixed threshold."""

    name: str = "large_gradient"
    required_collectors: list[str] = ["gradient"]

    def __init__(self, threshold: float = 100.0) -> None:
        self._threshold = threshold

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        grad_state = collector_states["gradient"]
        norms = grad_state.series.get("grad_norm")
        if norms is None or len(grad_state.steps) == 0:
            return []

        findings: list[Finding] = []
        for idx, layer in enumerate(grad_state.layers):
            layer_norms = norms[:, idx]
            max_norm = float(np.max(layer_norms))
            if max_norm > self._threshold:
                findings.append(Finding(
                    detector=self.name,
                    severity=Severity.WARNING,
                    category="gradient_flow",
                    title=f"Large gradient in {layer}",
                    summary=f"Gradient norm in {layer} reached {max_norm:.1f}, exceeding the {self._threshold:.1f} threshold.",
                    detail=f"The maximum gradient L2 norm observed in {layer} was {max_norm:.1f}. Large gradients can destabilize training and cause weight explosion.",
                    affected_layers=[layer],
                    step_range=(int(grad_state.steps[0]), int(grad_state.steps[-1])),
                    remediation=[
                        "Add gradient clipping (torch.nn.utils.clip_grad_norm_).",
                        "Reduce the learning rate.",
                    ],
                ))
        return findings
```

The detector will be picked up automatically once the module containing it
is imported before the `Autopsy` context manager is entered.

## License

`torchmortem` is provided under the [MIT License](LICENSE).
