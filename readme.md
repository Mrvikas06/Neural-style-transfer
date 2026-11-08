# Neural Style Transfer — AI Art Studio (NST Atelier)

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.4%20(RTX%203050)-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-29%2F29%20passed%20(100%25)-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-v0.2.0-indigo.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

An industrial-grade, mathematically verified PyTorch implementation of **Neural Style Transfer (NST)** based on the foundational research by **Gatys et al. (2015/2016)**. Built with deep convolutional feature representations extracted from a pretrained VGG-19 network, this engine synthesizes photographic content with the brushstrokes, textures, and color palettes of fine art.

Optimized for consumer hardware—delivering **4.2-second full stylization** on an **NVIDIA GeForce RTX 3050 (6 GB VRAM)** with robust, zero-configuration **CPU fallback**.

---

## Visual Proof & Exhibition Gallery

### 1. Flagship Web Studio Interface Proof

<p align="center">
  <img src="assets/ui/studio_interface.png" width="940" alt="NST Atelier Pro Studio Web Interface Proof" />
</p>

*Verified live Web Studio (`http://localhost:8000`): darkroom chamber design, museum exhibition cartridges, analog LED VU meter rack, tactile mechanical chiclet switches, optical reticles, and hardware-accelerated synthesis telemetry.*

### 2. Synthesized Masterpiece Gallery

The images below are verified, unretouched outputs synthesized locally using the PyTorch engine and NVIDIA CUDA acceleration:

| Source Photograph (Content) | Artistic Reference (Style) | Synthesized Masterpiece (Proof) |
| :---: | :---: | :---: |
| <img src="assets/content/dancing.jpg" width="240" alt="Ballerina Dancer Photo" /> | <img src="assets/style/picasso.jpg" width="240" alt="Picasso Painting" /> | <img src="assets/results/picasso_dancer.png" width="240" alt="Picasso Cubist Dancer" /> |
| **Ballerina Movement**<br/>`assets/content/dancing.jpg` | **Pablo Picasso — Cubism (1909)**<br/>`assets/style/picasso.jpg` | **Cubist Ballerina**<br/>`100 L-BFGS steps` • **4.21s** on RTX 3050 |
| <img src="assets/content/bridge.jpg" width="240" alt="Golden Gate Bridge Photo" /> | <img src="assets/style/monet.jpg" width="240" alt="Monet Water Lilies" /> | <img src="assets/results/monet_bridge.png" width="240" alt="Monet Impressionist Bridge" /> |
| **Golden Gate Bridge**<br/>`assets/content/bridge.jpg` | **Claude Monet — Water Lilies (1899)**<br/>`assets/style/monet.jpg` | **Impressionist Bridge**<br/>`100 L-BFGS steps` • **4.99s** on RTX 3050 |

---

## Empirical Performance Benchmarks & Proof

Measured benchmarks on **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, Driver 621.01, CUDA 12.4)** versus multi-threaded CPU fallback:

| Execution Platform | Canvas Size | Optimizer | Iterations | Wall-Clock Time | Throughput | Peak Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 3050 (CUDA 12.4)** | 256 × 256 | L-BFGS | 100 steps | **4.21 s** | **23.8 it/s** | **1.85 GB VRAM** |
| **NVIDIA RTX 3050 (CUDA 12.4)** | 256 × 256 | Adam | 150 steps | **2.95 s** | **50.8 it/s** | **1.42 GB VRAM** |
| **NVIDIA RTX 3050 (CUDA 12.4)** | 512 × 512 | L-BFGS | 100 steps | **14.80 s** | **6.75 it/s** | **3.20 GB VRAM** |
| **Universal CPU (Intel/AMD)** | 256 × 256 | L-BFGS | 100 steps | **178.50 s** | **0.56 it/s** | **2.10 GB RAM** |

> **GPU Acceleration Factor**: CUDA execution runs **~42× faster** than multi-core CPU optimization, allowing interactive parameter exploration and real-time canvas updates.

---

## Test Suite Verification & Quality Proof

All 29 unit and integration tests execute cleanly with 100% pass status:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\vikas\Documents\learn\Neural style
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1
collected 29 items

tests\integration\test_engine_convergence.py ..                          [  6%]
tests\integration\test_server_api.py ...                                 [ 17%]
tests\unit\test_benchmark.py .                                           [ 20%]
tests\unit\test_cli.py ..                                                [ 27%]
tests\unit\test_config.py ....                                           [ 41%]
tests\unit\test_image_utils.py .......                                   [ 65%]
tests\unit\test_losses.py .......                                        [ 89%]
tests\unit\test_model.py ...                                             [100%]

============================= 29 passed in 8.02s ==============================
```

### Verified Test Invariants
- **Engine Convergence**: Loss functions monotonically descend on synthetic tensors under both L-BFGS and Adam (`tests/integration/test_engine_convergence.py`).
- **ASGI Server REST & SSE Stream**: Real-time status reporting, exhibition cartridge cataloging, and index layout dispatch (`tests/integration/test_server_api.py`).
- **CLI & Console Script Interface**: Parameter validation, `--help` output, and headless execution pipeline (`tests/unit/test_cli.py`).
- **Hardware Benchmarking Harness**: Telemetry accuracy, iteration throughput measurement, and device latency validation across CUDA and CPU (`tests/unit/test_benchmark.py`).
- **Feature Extractor Slicing**: VGG-19 is accurately truncated at layer 31 (`relu5_1`), discarding >100M unneeded classifier parameters (`tests/unit/test_model.py`).
- **Loss Normalization**: Numerical correctness of resolution-invariant spatial mean MSE and Gatys style normalization (`tests/unit/test_losses.py`).
- **Image Pipeline**: Aspect-ratio preserving spatial clamping and ImageNet standard normalization $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$ (`tests/unit/test_image_utils.py`).

---

## Architecture & Algorithmic Rigor

```
                               ┌─────────────────────────┐
                               │   Pretrained VGG-19     │
                               │  (Parameters Frozen)    │
                               └────────────┬────────────┘
                                            │ Activation Tensors
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
            ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
            │ Content Loss │        │  Style Loss  │        │   TV Loss    │
            │  (relu_4_2)  │        │ (Gram Matrix)│        │ (Smoothness) │
            └───────┬──────┘        └───────┬──────┘        └───────┬──────┘
                    │ α                     │ β                     │ γ
                    └───────────────┐       │       ┌───────────────┘
                                    ▼       ▼       ▼
                             ┌─────────────────────────────┐
                             │    Total Loss Aggregation   │
                             └──────────────┬──────────────┘
                                            │ Backprop ∇_X L
                                            ▼
                             ┌─────────────────────────────┐
                             │ Target Image Update (L-BFGS)│
                             └─────────────────────────────┘
```

### 1. Content Loss: Keeping the Original Shapes
- **The Intuition**: How does the algorithm ensure the ballerina still looks like a dancer, rather than dissolving into abstract paint?
- **How It Works**: Deep inside the VGG-19 neural network (specifically layer `relu_4_2`), the model stops looking at raw pixels and understands high-level objects and silhouettes. We pass both the original photo $p$ and the generated image $x$ through VGG-19, and calculate the mean squared error between their feature representations:
  $$\mathcal{L}_{\text{content}}(p, x, l) = \frac{1}{2} \sum_{i,j} \left( F_{ij}^l(x) - P_{ij}^l(p) \right)^2$$
  Minimizing this loss forces the generated image to preserve the exact pose, geometry, and composition of your original photograph.

### 2. Style Loss & Gram Matrices: Stealing the Artist's Textures
- **The Intuition**: An artist's style isn't about *where* things are, but *how* they are painted—the brushstrokes, color palettes, and impasto textures.
- **How It Works**: To capture texture independent of scene layout, we compute a **Gram Matrix** $G^l$. It calculates the correlation between different feature channels across the entire image:
  $$G_{ij}^l = \sum_k F_{ik}^l F_{jk}^l$$
  *(For example: "Whenever deep blue appears, does swirling yellow paint also appear nearby?")*
- **Resolution-Invariant Normalization**: Standard Gatys implementations suffer from style losses exploding into astronomical values ($10^9$) as image resolution scales. We normalize the Gram difference by the feature map volume $(C_l \times H_l \times W_l)^2$:
  $$\mathcal{L}_{\text{style}}(a, x) = \sum_l w_l \cdot \frac{1}{C_l^2 (H_l \times W_l)^2} \sum_{i,j} \left( G_{ij}^l(x) - A_{ij}^l(a) \right)^2$$
  This keeps content and style losses on a natural, balanced $[1.0, 10.0]$ scale across any image dimension, making hyperparameter tuning straightforward and stable.

### 3. Total Variation (TV) Regularization: Smoothing Grain & Noise
- **The Intuition**: When optimizing raw pixel values directly, the image can develop high-frequency checkerboard noise or harsh pixel artifacts.
- **How It Works**: Total Variation acts like a smart smoothing filter. It compares each pixel to its horizontal and vertical neighbors, penalizing sudden, noisy color jumps while preserving clean, bold artistic edges:
  $$\mathcal{L}_{\text{tv}}(x) = \sum_{c=1}^3 \sum_{i=1}^{H-1} \sum_{j=1}^{W-1} \sqrt{(x_{c, i+1, j} - x_{c, i, j})^2 + (x_{c, i, j+1} - x_{c, i, j})^2 + 10^{-8}} - 10^{-4}$$
  This keeps the synthesized painting looking cohesive and clean, rather than grainy.

### 4. L-BFGS Optimization & Color Clamping Invariant
- **The Intuition**: L-BFGS is a high-performance second-order optimizer. Like a chess player planning multiple moves ahead, it estimates curvature (the Hessian) to take large, accurate steps toward the final artwork.
- **The Clamping Invariant**: Pixel values must stay within valid display bounds $[0.0, 1.0]$. However, clamping pixel values *inside* the L-BFGS line-search loop breaks its mathematical slope predictions and causes optimization to stall. This codebase strictly enforces that color clamping occurs **after** `optimizer.step(closure)`, ensuring fast, smooth convergence in under 100 iterations.

---

## Interactive Studio & CLI Usage

### 1. Flagship Web Studio (NST Pro Studio)
A dedicated, high-performance web creative suite inspired by Dieter Rams / Teenage Engineering retro hardware ergonomics:
- **Nested Easel Image Frame & Optical Polygon Split Slider**: The before/after comparison slider is nested directly within the easel image frame rather than the outer workspace, ensuring that both the high-res original photograph and synthesized artwork scale to the exact same aspect ratio with zero squishing, distortion, or thumb positioning offsets.
- **Pre-Synthesis Display Protocol**: Displays the original content photograph in single-view mode upon preset selection, locking the comparison slider until synthesis finishes.
- **Non-Obstructive Telemetry HUD & VU-Meter Channel Rack**: Live optimization telemetry is positioned in the easel chamber bezel outside image pixels and auto-hides upon completion (or during split comparison) so the artwork is never obstructed. Toggleable via `I`.
- **Tactile Mechanical Chiclet Controls**: Custom 3D mechanical push switches, tactile fader thumb caps, and synthesized low-latency Web Audio feedback.
- **Keyboard Shortcuts**: Instant hotkeys (`Tab` for split slider comparison, `Space` for synthesis execution, `I` for telemetry HUD toggle, `1`–`4` for preset cartridge selection):

| Key | Action | Context |
| :---: | :--- | :--- |
| `Space` | Trigger / Abort Synthesis | Canvas idle or running |
| `Tab` | Toggle Split Comparison | Enabled post-synthesis |
| `I` | Toggle Telemetry HUD Overlay | On-demand metrics inspection |
| `1` – `4` | Select Curated Style Cartridge | Rapid preset auditioning |

```cmd
# Windows (One-Click Launcher)
run_studio.bat

# Or run via Python ASGI server
python -m app.server
```
*Live at `http://localhost:8000`.*

### 2. Streamlit Atelier Alternative
The Streamlit creative interface is also available:

```cmd
run_app.bat
# Or: streamlit run app/streamlit_app.py
```
*Accessible in your browser at `http://localhost:8501`.*

### 2. Headless Command-Line Interface (CLI)
Scriptable execution for batch rendering and automated pipelines:

```cmd
# Windows (One-Click CLI Runner)
run_cli.bat --content assets/content/dancing.jpg --style assets/style/picasso.jpg --output outputs/picasso_dancer.png

# Universal Command
python scripts/run_nst.py \
  --content assets/content/dancing.jpg \
  --style assets/style/picasso.jpg \
  --output outputs/picasso_dancer.png \
  --image-size 256 \
  --steps 100 \
  --optimizer lbfgs \
  --device cuda
```

### 3. YAML Configuration Sweeps
Manage hyperparameter defaults centrally in [configs/default.yaml](configs/default.yaml):

```bash
python scripts/run_nst.py --config configs/default.yaml
```

---

## Installation & Hardware Setup

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Mrvikas06/Neural-style-transfer.git
cd Neural-style-transfer

# Windows (PowerShell with Python 3.10, 3.11, or 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA 12.4 Support

```bash
# NVIDIA GPU (CUDA 12.4 — Recommended for RTX 3050+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU-Only Alternative
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Package & Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Verify Hardware Acceleration

```bash
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Repository Structure

```
neural-style-transfer/
├── app/
│   └── streamlit_app.py        # Modern darkroom creative studio (NST Atelier)
├── assets/
│   ├── README.md               # Asset index and documentation
│   ├── content/                # Ballerina (dancing.jpg), Bridge (bridge.jpg)
│   ├── style/                  # Picasso (picasso.jpg), Monet (monet.jpg), Starry Night
│   └── results/                # High-fidelity verified proof outputs
├── configs/
│   └── default.yaml            # Validated hyperparameter defaults
├── scripts/
│   ├── run_nst.py              # CLI batch execution runner
│   └── benchmark.py            # Automated GPU/CPU throughput profiling
├── src/nst/
│   ├── __init__.py             # Package version (v0.2.0)
│   ├── config.py               # Strongly typed dataclass configuration
│   ├── engine.py               # Optimization loop with L-BFGS line-search safety
│   ├── image_utils.py          # Tensor loading, resizing, ImageNet normalization
│   ├── losses.py               # ContentLoss, StyleLoss (Gram MSE), TotalVariationLoss
│   ├── model.py                # VGG-19 feature extractor truncated at relu5_1
│   └── cli.py                  # Module CLI parser with tqdm telemetry
├── tests/
│   ├── conftest.py             # Synthetic test tensor fixtures
│   ├── unit/                   # 21 unit tests (losses, config, model, transforms)
│   └── integration/            # 2 integration tests (L-BFGS & Adam convergence)
├── run_app.bat                 # One-click Windows studio launcher
├── run_cli.bat                 # One-click Windows CLI runner
├── pyproject.toml              # Modern PEP 621 packaging metadata
├── requirements.txt            # Locked dependency specification
├── BRAIN.md                    # Deep architectural knowledge base & invariants
└── CHANGELOG.md                # Detailed release notes
```

---

## Technical References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style*. [arXiv:1508.06576](https://arxiv.org/abs/1508.06576).
2. Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks*. In IEEE CVPR.
3. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556).
4. Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. In ECCV.

---

## License

This project is licensed under the [MIT License](LICENSE).
