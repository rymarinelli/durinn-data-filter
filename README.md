Here is a **clean, minimal update** to the README that reflects the caching and fixes, without exposing HF internals or adding noise. You can paste this over the existing one.

---

# Dataset Risk Decorator

A lightweight, plug-and-play **risk annotation and filtering utility for Hugging Face datasets**.

It annotates each sample with:

* `risk_score`
* `is_problematic`

while preserving full compatibility with the Hugging Face `datasets` API.

Designed for:

* Dataset auditing
* Safety filtering before fine-tuning
* Preference / DPO preprocessing
* Security research workflows

## Maintainer
<!--
**Durinn Research**
[victor@durinn.ai](mailto:victor@durinn.ai)
[ryan@durinn.ai](mailto:ryan@durinn.ai)
[https://huggingface.co/durinn](https://huggingface.co/durinn)
[https://durinn.ai](https://durinn.ai)
!-->
## Features

* Works with `datasets.Dataset` and `datasets.DatasetDict`
* Operates directly on loaded datasets
* Adds risk metadata without breaking downstream pipelines
* Optional row limiting for fast iteration
* Optional automatic filtering modes
* Built-in dataset and model caching
* Trainer-compatible output

## Installation

```bash
pip install durinn-data-filter
```

For local development:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```python
from datasets import load_dataset
from dataset_risk_decorator.core import risk_guard

ds = risk_guard(
    load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
)

scores = ds["risk_score"]
print("rows:", len(ds))
print("min:", min(scores))
print("mean:", sum(scores) / len(scores))
print("max:", max(scores))

top_risky = ds.sort("risk_score", reverse=True).select(range(10))

for row in top_risky:
    print(row["risk_score"], row["chosen"][:200])
```

Each row now contains:

```text
risk_score
is_problematic
```

## Configuration

All configuration is optional and passed directly to `risk_guard`.

```python
ds = risk_guard(
    load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train"),
    threshold=0.5,
    filter_mode="none",
    max_rows=2000,
    cache="auto",
)
```

### Parameters

| Parameter     | Description                                      |
| ------------- | ------------------------------------------------ |
| `threshold`   | Threshold used to set `is_problematic`           |
| `filter_mode` | `"none"`, `"keep_safe"`, or `"keep_problematic"` |
| `max_rows`    | Limit number of rows for faster iteration        |
| `cache`       | `"auto"`, `"reuse"`, or `"disable"`              |

### Cache behavior

* **Datasets** use Hugging Face’s built-in fingerprinted caching
  Annotation results are reused automatically across reruns when inputs and configuration are unchanged.
* **Models** are downloaded once and reused across calls within the same process.
* No environment variables or cache directories need to be configured by the user.

Cache modes:

* `auto`
  Default behavior. Uses Hugging Face cache when safe.
* `reuse`
  Forces reuse of cached annotation results.
* `disable`
  Forces recomputation.

## Filtering Modes

| `filter_mode`        | Behavior                 |
| -------------------- | ------------------------ |
| `"none"`             | Annotate all rows        |
| `"keep_safe"`        | Keep only low-risk rows  |
| `"keep_problematic"` | Keep only high-risk rows |

Filtering happens after annotation.

## Common Usage Patterns

### Inspect Score Distribution

```python
scores = ds["risk_score"]
print("rows:", len(ds))
print("min:", min(scores))
print("mean:", sum(scores) / len(scores))
print("max:", max(scores))
```

### Train on Safe Samples Only

```python
safe_ds = risk_guard(
    load_dataset("yahma/alpaca-cleaned", split="train"),
    filter_mode="keep_safe",
)

print(len(safe_ds))
```

### Extract High-Risk Samples for Analysis

```python
risky_ds = risk_guard(
    load_dataset("microsoft/Devign", split="train"),
    filter_mode="keep_problematic",
)

for row in risky_ds.select(range(10)):
    print(row["risk_score"], row.get("func", "")[:200])
```

## DatasetDict Support

```python
ds = risk_guard(
    load_dataset("yahma/alpaca-cleaned")
)

train = ds["train"]
test = ds["test"]
```

Each split is annotated independently.

## Project Structure

```text
dataset-risk-decorator/
├── dataset_risk_decorator/
│   ├── __init__.py
│   └── core.py
├── examples/
│   ├── default.py
│   └── with_config.py
├── pyproject.toml
└── README.md
```

## Disclaimer

The current risk scorer is **MVP-grade by design**.
It is intended to validate data pipelines and filtering mechanics prior to deploying a fully calibrated classifier.
