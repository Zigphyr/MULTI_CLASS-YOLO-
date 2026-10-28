# YOLO-Count: Multi-Class Counting with QVI

Reference implementation of the three-model progression described in the
accompanying architecture review and implementation guide.

| Model | What it does                                                   |
|-------|----------------------------------------------------------------|
| **A** | Original YOLO-Count. One text query per forward pass.          |
| **B** | Multi-class extension. All K classes answered in one pass.     |
| **C** | Model B + Query-Visual Interaction (QVI).                      |

## Layout

```
yolocount_mc/
  config.py         Config dataclass
  text_encoder.py   K text queries -> [B, K, 512]
  backbone.py       YOLO backbone + PAFPN (placeholder)
  head.py           shared visual features + similarity einsum (Step 1)
  qvi.py            QVI module, zero-init residual (Step 7)
  models.py         ModelA / ModelB / ModelC
  yolocount.py      class-agnostic mask / predictions (Step 2)
  validation.py     per-class + overall MAE / RMSE (Step 3)
  dataset.py        toy dataset for sanity training
  train.py          train B, then C in stages (Steps 5 and 9)
  benchmark.py      Step 10 latency / memory benchmark
  ablation.py       Step 11 multi-seed ablation
tests/
  test_models.py    Step 4: shapes, K=1 equivalence, QVI zero-init
scripts/
  run_all.py        end-to-end entry point
```

## Install

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Quick start

```bash
# run the pre-training tests (shapes, consistency, QVI zero-init)
pytest -q

# run the end-to-end pipeline (sanity tests -> train B -> train C -> benchmark)
python scripts/run_all.py
```

## Adapting to the real YOLO-Count repo

Two things to swap for your actual repository:

1. Replace `yolocount_mc/backbone.py:YoloBackbonePAFPN` with the real
   backbone + PAFPN so `feat` has shape `[B, C, H, W]`.
2. Point `yolocount_mc/text_encoder.py:TextEncoder` at your real text encoder
   (CLIP / BERT) producing `[B, K, 512]` embeddings.

Everything downstream — `head.py`, `qvi.py`, `yolocount.py`, `validation.py`,
the tests, the benchmark, and the ablation harness — keeps the exact tensor
contract called out in the architecture review:

- maps:   `[B, K, H, W]`
- counts: `[B, K]`

## Configuration

Toggle QVI through `Config`:

```python
from yolocount_mc import Config, ModelC, TextEncoder
cfg = Config(use_qvi=True, qvi_type="modulation", qvi_channels=64)
model = ModelC(TextEncoder(["car", "person", "dog"]), cfg)
```

`qvi_type` accepts `none | gated | modulation | attention`.
