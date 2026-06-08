<div align="center">

# Signal Intelligence

**Deep Learning for Signal Modulation Classification and Inference**

[![Model: CNN--Attention](https://img.shields.io/badge/model-cnn--attention-000000.svg?style=flat-square)](https://github.com/AsaqeLee/Signal)
[![Standard: Production--Grade](https://img.shields.io/badge/standard-production--grade-000000.svg?style=flat-square)](https://github.com/AsaqeLee/Signal)
[![Tooling: Ruff](https://img.shields.io/badge/tooling-ruff-000000.svg?style=flat-square)](https://github.com/AsaqeLee/Signal)

English | [简体中文](./README_ZH.md)

</div>

---

## Introduction

**Signal Intelligence** is an advanced framework for automatic modulation classification (AMC). Following a comprehensive refactor, the project now features a production-grade inference pipeline and modern project management tooling for high-integrity signal processing research.

---

## Technical Specifications

<details>
<summary><b>Inference Pipeline</b></summary>

The `project/inference.py` module provides a clean interface for model deployment:
- **Preprocessing:** Automatic I/Q channel detection and shape normalization (1024 samples).
- **Multi-Task Inference:** Simultaneous prediction of modulation types, confidence levels, and symbol widths.
- **Hardware Agnostic:** Optimized for both CUDA and CPU-only environments.
</details>

<details>
<summary><b>Project Standards</b></summary>

```text
project/
├── pyproject.toml      # Ruff & Metadata configuration
├── project/
│   ├── inference.py    # Production inference class
│   └── train.py        # Multi-task training orchestrator
└── tests/
    └── test_inference.py # Preprocessing & Pipeline validation
```
</details>

<details>
<summary><b>Installation & Execution</b></summary>

### Prerequisites
- Python 3.8+
- PyTorch 1.9.0+

### Execution
```bash
# Format and check code quality
ruff format . && ruff check .

# Run validation suite
pytest tests/
```
</details>

---

<div align="center">

&copy; 2026 AsaqeLee. Built for advanced signal processing research.

</div>
