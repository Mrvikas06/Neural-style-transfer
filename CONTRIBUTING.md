# Contributing to Neural Style Transfer

Thank you for your interest in contributing to **Neural Style Transfer — AI Art Generator**! This project aims to deliver a robust, modular, and memory-efficient implementation of classical Neural Style Transfer (NST) using PyTorch, tailored for consumer GPUs (such as the NVIDIA RTX 3050 with 4–6 GB VRAM) as well as CPU environments.

Please read this guide to understand our engineering expectations, branching model, coding standards, and verification workflow.

---

## 1. Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for everyone, regardless of background or experience level. Please maintain professional, constructive, and respectful interactions at all times.

---

## 2. Development Setup

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA drivers installed (recommended: NVIDIA RTX 3050 4GB/6GB) or modern multi-core CPU
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/neural-style-transfer.git
   cd neural-style-transfer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install PyTorch with CUDA support:**
   Follow official instructions from [pytorch.org](https://pytorch.org/get-started/locally/) matching your CUDA version (e.g. CUDA 12.1):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

5. **Verify your CUDA runtime:**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

---

## 3. Branching & Commit Conventions

### Branch Naming
Create focused, descriptively named branches off `main`:
- `feature/<feature-name>`: New functionality or architecture additions (e.g., `feature/lbfgs-closure`).
- `fix/<bug-name>`: Bug fixes or stability patches (e.g., `fix/cuda-oom-handling`).
- `docs/<doc-update>`: Documentation updates or additions (e.g., `docs/add-benchmarks`).
- `refactor/<module>`: Refactoring without modifying external behavior.

### Commit Messages
We follow the [Conventional Commits](https://www.conventionalcommits.org/) standard:
```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

Examples:
- `feat(core): implement Gram matrix computation and normalization`
- `fix(memory): purge optimizer history on cuda out-of-memory error`
- `docs(math): clarify Gatys loss normalization constant`
- `test(losses): add unit tests for total variation loss gradient stability`

---

## 4. Code Quality & Standards

All code submissions must comply with our formatting, typing, and quality standards:

1. **Type Hints**: All function and method signatures must include Python type hints (`from typing import Optional, Tuple, Dict`).
2. **Formatting**: Format all code with [Black](https://github.com/psf/black) with line length `100`:
   ```bash
   black --line-length 100 src/ tests/ app/
   ```
3. **Linting**: Lint with [Ruff](https://github.com/astral-sh/ruff):
   ```bash
   ruff check src/ tests/ app/
   ```
4. **Static Type Checking**: Check type correctness using [Mypy](https://mypy-lang.org/):
   ```bash
   mypy src/
   ```
5. **Clean Tensor Hygiene**:
   - Explicitly document tensor shapes in docstrings: `(B, C, H, W)`.
   - Ensure tensors are moved to appropriate devices dynamically (`device = content_tensor.device`).
   - Freeze feature extractor weights (`param.requires_grad = False`).
   - Separate computational logic from UI code (`src/nst` must never import `streamlit`).

---

## 5. Testing Requirements

Never submit code without accompanying automated tests:
- Place unit tests under `tests/unit/` and integration tests under `tests/integration/`.
- Ensure all tests execute on CPU so CI pipelines without GPUs pass cleanly.
- Verify numerical stability: check that losses are non-negative, finite, and gradients do not produce `NaN` or `Inf`.
- Run the test suite:
  ```bash
  pytest -v --cov=src/nst tests/
  ```
- Any new module must maintain minimum 85% branch and statement test coverage.

---

## 6. Pull Request (PR) Workflow

1. Keep PRs small, isolated, and focused on a single responsibility.
2. Ensure existing tests pass and new tests are written.
3. Update relevant documentation in `docs/` and summarize architectural decisions in `BRAIN.md` if fundamental designs change.
4. Fill out the PR template completely:
   - Summary of changes
   - Motivation and context
   - Test results (actual console outputs)
   - Hardware verified on (e.g., RTX 3050 4GB, CPU)
5. Request review from maintainers and address review comments promptly.

---

## 7. Reporting Issues

When opening an issue, please use the issue tracker and include:
- A clear, concise title.
- Steps to reproduce the issue.
- Operating System, Python version, PyTorch version, and GPU model.
- Complete traceback or error logs.
- Minimal reproducible example or sample images where applicable.
