# Signal

English | [简体中文](./README_ZH.md)

Deep-learning workflows for **automatic modulation classification (AMC)** and related inference on raw communication signals (I/Q).

## Overview

The `project/` package provides training and inference entry points for multi-task models (modulation type, confidence, and related targets such as symbol width). The codebase is oriented toward research experiments rather than a packaged commercial product.

## Features / scope

- Inference helper in `project/inference.py`
  - I/Q channel detection and shape normalization (1024-sample windows as used by the scripts)
  - Multi-task predictions (modulation, confidence, symbol-width related outputs)
  - CUDA or CPU execution depending on the installed PyTorch build
- Training orchestration in `project/train.py` (and related scripts such as `train_modulation.py`, `train_binary.py`)
- Tests under `tests/` (including inference preprocessing checks)
- Tooling via `pyproject.toml` (Ruff) and `project/requirements.txt`

## Requirements

- Python 3.8+
- PyTorch 1.9.0+ (or newer compatible with the checked-in code)
- Additional packages from `project/requirements.txt`

## Getting started

```bash
git clone https://github.com/AsaqeLee/Signal.git
cd Signal
pip install -r project/requirements.txt

ruff format . && ruff check .
pytest tests/
```

See `project/run.py`, `Signal_LAB.md`, and training scripts for dataset paths and experiment recipes. Large artifacts under `output/` may be local-only.

## Project layout

```text
.
├── project/
│   ├── inference.py
│   ├── train.py
│   ├── model/
│   ├── data/
│   └── requirements.txt
├── tests/
├── pyproject.toml
├── Signal_LAB.md
└── README_ZH.md
```

## Status / limitations

Research code. Avoid “production-grade” claims unless your own deployment checklist is complete. Model checkpoints and datasets are not assumed to be redistributed with every clone; configure paths locally.

## License

MIT. See [LICENSE](LICENSE).
