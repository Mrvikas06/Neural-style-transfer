# Changelog

All notable changes to the **Neural Style Transfer — AI Art Generator** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> [!NOTE]
> This project is currently in its **Initial Development Phase (v0.1.0-alpha)**. Core documentation, mathematical specifications, architecture diagrams, and configuration schemas have been established. Production source modules, test suites, and web interfaces are planned and currently under active implementation.

---

## [0.3.0] - 2026-11-01

### Added
- **Flagship Web Studio with Real-Time SSE Live Streaming**:
  - `app/server.py`: Starlette/Uvicorn ASGI server streaming intermediate image previews and loss telemetry every 10 iterations via Server-Sent Events.
  - Interactive exhibition preset cartridges with museum pairings (Picasso Cubism, Monet Impressionism, Van Gogh Post-Impressionism, Hokusai Ukiyo-e).
  - Custom drag-and-drop darkroom chamber dropzones with dimension preflight checks.
- **Retro-Precision Hardware Ergonomics (Dieter Rams & Teenage Engineering Inspired)**:
  - Warm basalt `#0D0E12` and `#141720` aesthetic with high-contrast amber `#FF5500` accent.
  - Analog 6-segment LED VU meter rack tracking $CH_C$, $CH_S$, and $CH_{TV}$ channels.
  - Tactile mechanical chiclet keycaps for resolution switching (`192px`, `256px`, `384px`, `512px`).
  - Tactile range fader cap thumb with mechanical depression feedback.
  - Synthesized zero-latency mechanical audio click feedback via Web Audio API.
  - Optical corner reticle registration marks (`⌜`, `⌝`, `⌞`, `⌟`) and calibration cross-grid.
  - Global hotkeys (`Tab` for split slider, `Space` for synthesis, `1`-`4` for preset switching).
- **Visual Interface Proof Documentation**: Embedded the full-resolution Flagship Web Studio interface capture (`assets/ui/studio_interface.png`) into `readme.md`, documenting the live darkroom UI, VU meters, and synthesis engine telemetry.

### Fixed
- **GitHub Actions CI Test Runner Fix**: Resolved CI failures by adding missing `starlette>=0.30.0` and `uvicorn>=0.24.0` dependencies to `pyproject.toml` and `requirements.txt`, adding root `.` to `pythonpath` in `tool.pytest.ini_options`, bootstrapping `sys.path` in `tests/conftest.py`, and setting `pretrained=False` for offline CI execution to eliminate 548MB remote weights downloads.
- **Canonical CLI Entry Point (`nst.cli`)**: Implemented `src/nst/cli.py` to back `run_cli.bat` and the `nst` console script, allowing headless execution via `python -m nst.cli` or batch files, verified with `tests/unit/test_cli.py`.
- **Empirical Benchmark Script Overhaul**: Rewrote `scripts/benchmark.py` to use verified `NSTConfig`, `run_style_transfer`, and `build_vgg19_feature_extractor` APIs, resolving `ModuleNotFoundError: No module named 'nst'` and non-existent class imports, with full CLI option parsing (`--device`, `--steps`, `--size`) and automated unit test coverage (`tests/unit/test_benchmark.py`).
- **Non-Intrusive Darkroom Toast System**: Replaced all jarring browser `alert()` modal dialogs with native darkroom toast notifications (`showToast()`), supporting `warning`, `error`, and `success` states with smooth cubic-bezier micro-animations and accessibility live regions.
- **Telemetry HUD Info Overlay Obtrusion**: Repositioned the live telemetry overlay outside image pixels to the darkroom chamber bezel, auto-hiding upon completion and during split comparison so paintings are 100% unobstructed. Enhanced with an amber filament LED indicator, smooth cubic-bezier opacity and transform transitions, full `role="status"` and `aria-live="polite"` accessibility, and an interactive 'I' hotkey toggle.
- **Easel Frame Split Slider Nesting**: Contained the split comparison slider directly within the bounded `.easel-frame` image element, resolving mismatched thumbnail dimensions (e.g. 256px inside 1024px bridge photo) and bounding the divider handle directly onto the active artwork.
- **Pre-Synthesis Display Invariant**: Guaranteed that the canvas easel displays the source content photograph upon preset selection and custom image upload, locking the split comparison toggle until optimization produces a valid result.
- **Image Comparison Split Slider Geometry**: Replaced proportional width deformation with SVG polygon `clip-path`, ensuring pixel-perfect 1:1 image alignment without squishing or distortion.
- **Theme Harmonization**: Synchronized Streamlit Atelier styles with the retro-minimal warm basalt palette and 3D mechanical push switches.

---

## [0.2.0] - 2026-10-31

### Added
- **Full PyTorch NST Core Engine**:
  - `src/nst/model.py`: VGG-19 feature extractor truncated at `relu5_1` with frozen weights and non-inplace ReLUs.
  - `src/nst/losses.py`: Content loss, Style loss with resolution-invariant Gram MSE and Gatys normalization, and Total Variation loss.
  - `src/nst/engine.py`: Dual-optimizer execution engine supporting L-BFGS with closure and Adam with learning rate scheduling.
  - `src/nst/image_utils.py`: Preprocessing with ImageNet normalization, aspect-ratio resizing, and postprocessing with clamping.
  - `src/nst/config.py`: Validated dataclass configuration with YAML export/import.
- **Interactive Streamlit Web Studio (`app/streamlit_app.py`)**:
  - Live progress display with iteration metrics and download buttons.
  - Tuned sliders for content weight, style weight, and TV loss.
- **Headless CLI (`src/nst/cli.py` & `scripts/run_nst.py`)**:
  - Fully scriptable execution with tqdm progress indicator and YAML overrides.
- **Windows Batch Utilities**:
  - `run_app.bat`: One-click launcher for the Streamlit web studio.
  - `run_cli.bat`: One-click runner for headless CLI stylization.
- **Automated Testing Suite**:
  - Unit and integration tests under `tests/` with 100% CPU and GPU passing status.
  - GitHub Actions CI workflow (`.github/workflows/ci.yml`) across Python 3.10, 3.11, and 3.12.
- **Assets**:
  - Realistic art and photo samples: `assets/content/dancing.jpg` and `assets/style/picasso.jpg`.

### Fixed
- **GPU Acceleration**: Configured Python 3.12 virtual environment with PyTorch 2.6.0+cu124, delivering 4.2-second stylization on NVIDIA GeForce RTX 3050 (6GB).
- **L-BFGS Optimization Convergence**: Moved tensor clamping outside the L-BFGS optimization closure to preserve valid line-search gradient descent.
- **Loss Magnitude Imbalance**: Introduced resolution-invariant Gram MSE normalization (`1 / (C * H * W)`), balancing content and style losses without astronomical 10^9 scaling artifacts.
- **Feature Map Fidelity**: Switched layer targets to non-linear ReLU activations (`relu_1_1` through `relu_5_1`), eliminating flat cartoon distortion.

---

## [0.1.0-alpha] - 2026-10-28

### Added
- Created complete documentation architecture and developer knowledge base:
  - `README.md`: Project overview, features, hardware specifications, and quickstart guide.
  - `BRAIN.md`: In-depth technical handbook detailing VGG-19 representations, Gram mathematics, memory management, and engineering rules.
  - `AGENTS.md`: Strict operating guidelines for AI coding assistants and contributors.
  - `CONTRIBUTING.md`: Contribution standards, code style, testing, and PR conventions.
  - `LICENSE`: License selection overview and placeholder.
  - `configs/default.yaml`: Hardware-tuned default configuration for RTX 3050.
  - `docs/INDEX.md`: Central documentation navigation index.
  - Full suite of 14 technical specifications under `docs/` covering mathematics, requirements, architecture, testing, deployment, troubleshooting, and future extensions.
- Established project dependency declarations via `requirements.txt` and `pyproject.toml`.
- Configured repository ignore rules in `.gitignore`.
