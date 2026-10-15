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

- Python 3.8+ (the submission environment targeted by `project/run.py` is Python 3.8)
- PyTorch 1.9.0+ (or newer compatible with the checked-in code)
- Additional packages from `project/requirements.txt`

## Task setup (from `project/config.py`)

| Item | Value |
|---|---|
| Classes | 10 modulations: BPSK, QPSK, 8PSK, MSK, 8QAM, 16QAM, 32QAM, 8APSK, 16APSK, 32APSK |
| Input | 2 × 1024 I/Q window, sampling rate 20 MHz |
| Targets | modulation type + symbol width (1 µs – 100 µs); loss weights MT 0.4 / SW 0.6 |
| Training | AdamW, lr 1e-3, OneCycle, batch 256 (grad-accum 2), up to 300 epochs, early stopping 20, AMP |
| Augmentation | mixup, cutmix, SpecAugment, random erasing |

## Reproduce

```bash
git clone https://github.com/AsaqeLee/Signal.git
cd Signal
pip install -r project/requirements.txt pytest

# 1) quick check, no dataset needed (random I/Q through an untrained model)
pytest tests/
python -m project.inference

# 2) training
#    Data is NOT included. Put CSV files (two columns: I, Q) under
#    ./train_data_true/<MODULATION>/data_<N>.csv, e.g. train_data_true/QPSK/data_1.csv.
#    Each class needs >= 16200 files (80/20 train/val split by file number).
python -m project.train
```

- Paths are relative to the working directory: outputs go to `./output/` (`checkpoints/`, `logs/`).
- `USE_WANDB = True` by default; run `wandb login` or set it to `False` in `project/config.py`.
- To use a trained model: `SignalInference(model_path="output/checkpoints/best_checkpoint.pth")`.
- `project/run.py` is the competition submission template (reads `<dir>/test/*.csv`, writes `file_name,modulation_type,symbol_width,code_sequence`); its prediction lines are still placeholders.

## Results

No evaluation figures or result tables are checked into this repository. The only recorded run is `output/checkpoints/training_state.json` (2024-11-30): 15 epochs, ~35 min of training, best validation score 0.331 (`total_score` on the validation set as computed in `project/train.py`). Treat it as an early, unfinished run, not a benchmark.

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
