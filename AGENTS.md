# AGENTS.md — Repository Instructions for AI Coding Assistants

Welcome, AI Agent. This document defines the strict operational protocol, engineering standards, and behavioral constraints for working inside the `neural-style-transfer` codebase.

You must follow these rules without exception on every task.

---

## 1. Prime Directives for AI Agents

1. **Mandatory Documentation Review**: Before proposing or making substantial changes to architecture, loss functions, optimization loops, or data flow, you must read [BRAIN.md](file:///c:/Users/vikas/Documents/learn/Neural%20style/BRAIN.md) and the corresponding file in `docs/` (e.g., [docs/NST_MATHEMATICS.md](file:///c:/Users/vikas/Documents/learn/Neural%20style/docs/NST_MATHEMATICS.md), [docs/ARCHITECTURE.md](file:///c:/Users/vikas/Documents/learn/Neural%20style/docs/ARCHITECTURE.md)).
2. **Inspect Before Modifying**: Never assume file content or directory state. Always inspect existing code using filesystem tools before making modifications. Preserve useful existing implementations, docstrings, and comments.
3. **Plan Before Non-Trivial Execution**: For any multi-step task, refactoring, or new module addition, present a concise implementation plan outlining planned files, design choices, and verification steps.
4. **Strict Truthfulness & Verification**:
   - Never fabricate benchmark numbers, FPS, latency, or VRAM consumption.
   - Never claim tests passed unless you executed them and observed an exit code of 0.
   - If tests cannot be run (e.g. missing dependencies or no CUDA environment), state this explicitly.
5. **Clear Separation of Concerns**: Keep model/NST core logic in `src/nst/` completely decoupled from web UI code (`app/`) and scripts (`scripts/`). Core modules must never import Streamlit or CLI frameworks.

---

## 2. PyTorch & Tensor Engineering Standards

- **Hardware Target**: The target deployment platform includes an NVIDIA GeForce RTX 3050 (4–6 GB VRAM) as well as CPU fallback.
- **Default Resolution**: Always enforce conservative default image dimensions (e.g. 256×256 pixels) to prevent CUDA Out-Of-Memory (OOM) faults. Provide clear dimension clamps.
- **Explicit Tensor Shapes**: Always document expected and returned tensor shapes in function docstrings:
  - Image tensors: `(B, C, H, W)` where `B=1`, `C=3`.
  - Feature maps: `(B, C, H, W)` or flattened `(C, H * W)`.
  - Gram matrices: `(C, C)`.
- **Parameter Freezing**: In classical NST (Gatys et al.), the VGG-19 network is **frozen** (`param.requires_grad = False`). Optimization is applied strictly to the target image tensor (`input_img.requires_grad_(True)`).
- **Device & Type Hygiene**: Do not hardcode `.cuda()` or `torch.float32`. Use dynamic device matching (`tensor.to(device)`) and respect configuration parameters.
- **Numerical Stability**: Guard against divisions by zero, exploding Gram matrix products, and invalid values. When clamping output images back to `[0.0, 1.0]`, do so carefully without breaking gradient flow during the optimization steps.

---

## 3. Optimization Protocol & Optimizer Distinctions

- **L-BFGS**:
  - Requires an explicit closure function: `def closure(): ... optimizer.step(closure)`.
  - Inside the closure: zero gradients, forward pass through VGG-19, compute content/style/TV losses, backward pass (`loss.backward()`), and return the aggregated loss.
  - Retains history buffers; beware of memory overhead on 4GB VRAM.
- **Adam**:
  - Uses standard iteration loops: zero grad, backward, step.
  - Requires lower learning rates (e.g., 0.02 – 0.05) compared to L-BFGS (typically 1.0).

---

## 4. Code Quality & Formatting Rules

- Target Python 3.10+ syntax.
- Use explicit type annotations for all parameters and return types.
- Format code using Black (line length 100).
- Adhere to Ruff linting guidelines; resolve all linting errors immediately.
- Avoid introducing heavy third-party dependencies unless explicitly requested by the user.

---

## 5. Testing & Verification Requirements

- Every behavioral change or new feature must have corresponding unit or integration tests under `tests/`.
- All tests in CI/automated runs must be executable on CPU without assuming an active CUDA GPU.
- Verify gradient existence: assert `input_img.grad is not None` and `not torch.isnan(input_img.grad).any()`.
- Add test fixtures for dummy synthetic tensors (e.g., `torch.randn(1, 3, 64, 64)`) to enable instant test execution without network calls or large weight downloads.

---

## 6. Documentation Maintenance

- When you alter an API signature, module responsibility, or default parameter in `configs/default.yaml`, immediately update the relevant documentation:
  - [docs/API_REFERENCE.md](file:///c:/Users/vikas/Documents/learn/Neural%20style/docs/API_REFERENCE.md)
  - [docs/REQUIREMENTS.md](file:///c:/Users/vikas/Documents/learn/Neural%20style/docs/REQUIREMENTS.md)
  - [BRAIN.md](file:///c:/Users/vikas/Documents/learn/Neural%20style/BRAIN.md)
  - [CHANGELOG.md](file:///c:/Users/vikas/Documents/learn/Neural%20style/CHANGELOG.md)
- Ensure all relative Markdown links remain intact.

---

## 7. Turn Report Contract

At the conclusion of each assistance turn or multi-step execution, the agent must report:
1. **Files Created or Modified**: Relative paths with brief descriptions of rationale.
2. **Tests Executed**: Actual commands run and exact results observed (or explicit notice if not executed).
3. **Current Limitations & Unverified Assumptions**: State what remains planned versus tested.
4. **Recommended Next Steps**: Concrete actions ready for execution.
