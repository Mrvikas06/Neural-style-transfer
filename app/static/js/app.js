/**
 * NST ATELIER — Flagship Interactive Application Logic
 * Implements real-time SSE live rendering, preset switching, split-slider comparison,
 * and custom image uploads.
 */

// Global State
const state = {
  inputMode: 'presets', // 'presets' | 'custom'
  activePresetId: 'picasso_ballerina',
  presets: {},
  customContentBase64: null,
  customStyleBase64: null,
  activeContentUrl: '/assets/content/dancing.jpg',
  activeStyleUrl: '/assets/style/picasso.jpg',
  activeResultUrl: null,
  hasSynthesized: false,
  resolution: 256,
  styleIntensity: 1e5,
  steps: 100,
  alpha: 1.0,
  tvWeight: 1e-4,
  optimizer: 'lbfgs',
  isProcessing: false,
  viewMode: 'single', // 'single' | 'split'
};

const intensityMap = {
  1: { weight: 2e4, label: '2×10⁴ (Subtle)' },
  2: { weight: 1e5, label: '1×10⁵ (Balanced)' },
  3: { weight: 3e5, label: '3×10⁵ (Vivid)' },
  4: { weight: 8e5, label: '8×10⁵ (Abstract)' },
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
  initHardwareStatus();
  await loadPresets();
  initSplitSlider();
  initCustomDropzones();
  initParamControls();
  initKeyboardShortcuts();
});

function initKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    const tag = e.target.tagName ? e.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

    if (e.key === 'Tab') {
      e.preventDefault();
      if (!state.hasSynthesized || !state.activeResultUrl) {
        playTactileClick('click');
        showToast('Synthesize an artwork first to unlock Before/After comparison.', 'warning');
        return;
      }
      toggleComparisonMode();
    } else if (e.code === 'Space') {
      e.preventDefault();
      startSynthesis();
    } else if (e.key === 'i' || e.key === 'I') {
      e.preventDefault();
      toggleTelemetryHUD();
    } else if (['1', '2', '3', '4'].includes(e.key)) {
      const idx = parseInt(e.key, 10) - 1;
      const keys = Object.keys(state.presets);
      if (keys[idx]) {
        selectPreset(keys[idx]);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Hardware Telemetry
// ---------------------------------------------------------------------------
async function initHardwareStatus() {
  const badgeText = document.getElementById('hardware-text');
  const metricDevice = document.getElementById('metric-device');
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.cuda_available) {
      badgeText.textContent = `CUDA 12.4 • ${data.device_name}`;
      if (metricDevice) metricDevice.textContent = data.device_name;
    } else {
      badgeText.textContent = 'CPU MODE (UNIVERSAL)';
      if (metricDevice) metricDevice.textContent = 'Universal CPU';
      const ind = document.querySelector('.status-indicator');
      if (ind) ind.style.background = '#F59E0B';
    }
  } catch (e) {
    badgeText.textContent = 'OFFLINE ENGINE';
  }
}

// ---------------------------------------------------------------------------
// Exhibition Presets Loading
// ---------------------------------------------------------------------------
async function loadPresets() {
  const container = document.getElementById('preset-card-list');
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    state.presets = data.presets;

    container.innerHTML = '';
    Object.values(state.presets).forEach((preset, idx) => {
      const card = document.createElement('div');
      card.className = `preset-card ${preset.id === state.activePresetId ? 'active' : ''}`;
      card.id = `preset-${preset.id}`;
      card.onclick = () => selectPreset(preset.id);

      card.innerHTML = `
        <div class="preset-card-top">
          <span class="preset-card-title"><span style="color: var(--accent-retro, #FF5500); font-family: var(--font-mono); font-size: 0.72rem; margin-right: 0.4rem;">0${idx + 1} //</span>${preset.title}</span>
          <span class="preset-card-movement">${preset.movement}</span>
        </div>
        <p class="preset-card-desc">${preset.note}</p>
      `;
      container.appendChild(card);
    });

    selectPreset(state.activePresetId);
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

// Retro Mechanical Tactile Audio Feedback (Web Audio API)
let audioCtx = null;
function playTactileClick(type = 'click') {
  try {
    if (!audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) audioCtx = new AudioContext();
    }
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    if (!audioCtx) return;

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    const now = audioCtx.currentTime;
    if (type === 'heavy') {
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(160, now);
      osc.frequency.exponentialRampToValueAtTime(45, now + 0.06);
      gain.gain.setValueAtTime(0.09, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);
      osc.start(now);
      osc.stop(now + 0.06);
    } else {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(340, now);
      osc.frequency.exponentialRampToValueAtTime(90, now + 0.03);
      gain.gain.setValueAtTime(0.035, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.03);
      osc.start(now);
      osc.stop(now + 0.03);
    }
  } catch (e) {
    // Fail silently if browser audio autoplay is restricted
  }
}

function selectPreset(presetId) {
  playTactileClick('click');
  const preset = state.presets[presetId];
  if (!preset) return;

  state.activePresetId = presetId;
  state.activeContentUrl = preset.content_url;
  state.activeStyleUrl = preset.style_url;
  state.activeResultUrl = null;
  state.hasSynthesized = false;

  // Revert back to single frame mode if in split comparison
  if (state.viewMode === 'split') {
    setSingleView();
  }

  // Update active UI cards
  document.querySelectorAll('.preset-card').forEach((c) => c.classList.remove('active'));
  const activeEl = document.getElementById(`preset-${presetId}`);
  if (activeEl) activeEl.classList.add('active');

  // Update summary thumbnails
  document.getElementById('summary-content-img').src = preset.content_url;
  document.getElementById('summary-style-img').src = preset.style_url;

  document.getElementById('artwork-title').textContent = preset.title;
  const sub = document.getElementById('artwork-subtitle');
  if (sub) {
    sub.innerHTML = `<span>${preset.artist} · ${preset.movement}</span> <span id="canvas-state-chip" class="canvas-state-chip">Ready to Synthesize</span>`;
  }

  // Always display content image on the easel before synthesis!
  const easelImg = document.getElementById('easel-image');
  easelImg.src = preset.content_url;
  easelImg.style.visibility = 'visible';

  // Prepare split layers
  document.getElementById('split-before-img').src = preset.content_url;
  document.getElementById('split-after-img').src = '';

  // Disable split comparison toggle until synthesis is run
  disableSplitButton();

  // Hide download button until artwork is generated
  const dlBtn = document.getElementById('download-btn');
  if (dlBtn) dlBtn.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Input Mode Switching
// ---------------------------------------------------------------------------
function switchInputMode(mode) {
  state.inputMode = mode;

  const btnPresets = document.getElementById('tab-presets-btn');
  const btnCustom = document.getElementById('tab-custom-btn');
  const presetsContainer = document.getElementById('presets-container');
  const customContainer = document.getElementById('custom-container');

  if (mode === 'presets') {
    btnPresets.classList.add('active');
    btnCustom.classList.remove('active');
    presetsContainer.classList.remove('hidden');
    customContainer.classList.add('hidden');
    selectPreset(state.activePresetId);
  } else {
    btnCustom.classList.add('active');
    btnPresets.classList.remove('active');
    presetsContainer.classList.add('hidden');
    customContainer.classList.remove('hidden');

    document.getElementById('artwork-title').textContent = 'Custom Composition';
    document.getElementById('artwork-subtitle').textContent = 'User Photograph & Style Reference';
  }
}

// ---------------------------------------------------------------------------
// Custom Dropzones
// ---------------------------------------------------------------------------
function initCustomDropzones() {
  setupFileUploader(
    'content-input',
    'content-placeholder',
    'content-preview',
    'content-dims',
    (b64, url) => {
      state.customContentBase64 = b64;
      state.activeContentUrl = url;
      state.activeResultUrl = null;
      state.hasSynthesized = false;
      document.getElementById('summary-content-img').src = url;
      const easelImg = document.getElementById('easel-image');
      easelImg.src = url;
      easelImg.style.visibility = 'visible';
      document.getElementById('split-before-img').src = url;
      document.getElementById('split-after-img').src = '';
      disableSplitButton();
      if (state.viewMode === 'split') setSingleView();
    }
  );

  setupFileUploader(
    'style-input',
    'style-placeholder',
    'style-preview',
    'style-dims',
    (b64, url) => {
      state.customStyleBase64 = b64;
      state.activeStyleUrl = url;
      document.getElementById('summary-style-img').src = url;
    }
  );
}

function setupFileUploader(inputId, placeholderId, previewId, dimsId, onLoaded) {
  const input = document.getElementById(inputId);
  input.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      const b64 = evt.target.result;
      const img = new Image();
      img.onload = () => {
        document.getElementById(dimsId).textContent = `${img.width}×${img.height}`;
        document.getElementById(placeholderId).classList.add('hidden');
        const preview = document.getElementById(previewId);
        preview.src = b64;
        preview.classList.remove('hidden');
        onLoaded(b64, b64);
      };
      img.src = b64;
    };
    reader.readAsDataURL(file);
  });
}

// ---------------------------------------------------------------------------
// Parameter Controls
function syncSliderValue(slider) {
  if (!slider) return;
  slider.style.setProperty('--attr-min', slider.min || 0);
  slider.style.setProperty('--attr-max', slider.max || 100);
  slider.style.setProperty('--control-value', slider.value);
  slider.setAttribute('aria-valuenow', slider.value);
}

function initParamControls() {
  const intensitySlider = document.getElementById('intensity-slider');
  const intensityVal = document.getElementById('intensity-val-label');
  syncSliderValue(intensitySlider);
  intensitySlider.addEventListener('input', (e) => {
    syncSliderValue(e.target);
    const item = intensityMap[e.target.value] || intensityMap[2];
    state.styleIntensity = item.weight;
    intensityVal.textContent = item.label;
    e.target.setAttribute('aria-valuetext', item.label);
  });

  const resButtons = document.querySelectorAll('.pill-choice');
  resButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      playTactileClick('click');
      resButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.resolution = parseInt(btn.dataset.res, 10);
      document.getElementById('res-val-label').textContent = `${state.resolution}px`;
    });
  });

  const stepsInput = document.getElementById('steps-input');
  const stepsLabel = document.getElementById('steps-val-label');
  syncSliderValue(stepsInput);
  stepsInput.addEventListener('input', (e) => {
    syncSliderValue(e.target);
    state.steps = parseInt(e.target.value, 10);
    const text = `${state.steps} ${state.optimizer.toUpperCase()}`;
    stepsLabel.textContent = text;
    e.target.setAttribute('aria-valuetext', text);
  });

  const alphaInput = document.getElementById('alpha-input');
  syncSliderValue(alphaInput);
  alphaInput.addEventListener('input', (e) => {
    syncSliderValue(e.target);
    state.alpha = parseFloat(e.target.value);
    document.getElementById('alpha-label').textContent = state.alpha.toFixed(1);
    e.target.setAttribute('aria-valuetext', state.alpha.toFixed(1));
  });

  document.getElementById('tv-input').addEventListener('change', (e) => {
    state.tvWeight = parseFloat(e.target.value);
    document.getElementById('tv-label').textContent = e.target.value;
  });

  document.getElementById('optimizer-input').addEventListener('change', (e) => {
    state.optimizer = e.target.value;
    stepsLabel.textContent = `${state.steps} ${state.optimizer.toUpperCase()}`;
  });
}

// ---------------------------------------------------------------------------
// Split Comparison Slider Interaction
// ---------------------------------------------------------------------------
function initSplitSlider() {
  const container = document.getElementById('split-comparison-container');
  const divider = document.getElementById('split-divider');
  const afterWrapper = document.getElementById('split-after-wrapper');
  let isDragging = false;

  const setSplitPosition = (clientX) => {
    const rect = container.getBoundingClientRect();
    if (!rect.width) return;
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const pct = (x / rect.width) * 100;
    container.style.setProperty('--split-pos', `${pct}%`);
    divider.style.left = `${pct}%`;
    if (afterWrapper) {
      afterWrapper.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    }
  };

  const onMove = (clientX) => {
    if (!isDragging) return;
    setSplitPosition(clientX);
  };

  divider.addEventListener('mousedown', (e) => {
    e.preventDefault();
    e.stopPropagation();
    isDragging = true;
  });
  container.addEventListener('mousedown', (e) => {
    e.preventDefault();
    isDragging = true;
    setSplitPosition(e.clientX);
  });
  window.addEventListener('mouseup', () => (isDragging = false));
  window.addEventListener('mousemove', (e) => onMove(e.clientX));

  divider.addEventListener('touchstart', (e) => {
    e.stopPropagation();
    isDragging = true;
  }, { passive: true });
  container.addEventListener('touchstart', (e) => {
    isDragging = true;
    setSplitPosition(e.touches[0].clientX);
  }, { passive: true });
  window.addEventListener('touchend', () => (isDragging = false));
  window.addEventListener('touchmove', (e) => onMove(e.touches[0].clientX), { passive: true });

  // Initialize at 50%
  divider.style.left = '50%';
  if (afterWrapper) {
    afterWrapper.style.clipPath = 'polygon(50% 0, 100% 0, 100% 100%, 50% 100%)';
  }
}

function toggleComparisonMode() {
  if (!state.hasSynthesized || !state.activeResultUrl) {
    playTactileClick('click');
    alert('Please synthesize an artwork first before comparing with the original photograph.');
    return;
  }

  playTactileClick('click');
  const splitView = document.getElementById('split-comparison-container');
  const easelImg = document.getElementById('easel-image');
  const toggleBtnText = document.getElementById('view-mode-text');

  if (state.viewMode === 'single') {
    state.viewMode = 'split';
    easelImg.style.visibility = 'hidden';
    splitView.classList.remove('hidden');
    toggleBtnText.textContent = '□ Single Frame';

    // Ensure HUD overlay is hidden in split comparison mode
    const hudBadge = document.getElementById('live-telemetry-badge');
    if (hudBadge) hudBadge.classList.add('hidden');
    const hudBtn = document.getElementById('hud-toggle-btn');
    if (hudBtn) hudBtn.classList.remove('active');

    const divider = document.getElementById('split-divider');
    const afterWrapper = document.getElementById('split-after-wrapper');
    if (divider) divider.style.left = '50%';
    if (afterWrapper) afterWrapper.style.clipPath = 'polygon(50% 0, 100% 0, 100% 100%, 50% 100%)';
  } else {
    setSingleView();
  }
}

function toggleTelemetryHUD() {
  playTactileClick('click');
  const hud = document.getElementById('live-telemetry-badge');
  const btn = document.getElementById('hud-toggle-btn');
  if (!hud) return;

  if (hud.classList.contains('hidden')) {
    hud.classList.remove('hidden');
    if (btn) {
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
    }
  } else {
    hud.classList.add('hidden');
    if (btn) {
      btn.classList.remove('active');
      btn.setAttribute('aria-pressed', 'false');
    }
  }
}

function setSingleView() {
  state.viewMode = 'single';
  const splitView = document.getElementById('split-comparison-container');
  const easelImg = document.getElementById('easel-image');
  const toggleBtnText = document.getElementById('view-mode-text');
  if (splitView) splitView.classList.add('hidden');
  if (easelImg) easelImg.style.visibility = 'visible';
  if (toggleBtnText) toggleBtnText.textContent = '⇄ Split Slider';
}

function enableSplitButton() {
  const toggleBtn = document.getElementById('view-mode-toggle');
  if (toggleBtn) {
    toggleBtn.disabled = false;
    toggleBtn.setAttribute('aria-disabled', 'false');
    toggleBtn.classList.remove('btn-disabled');
    toggleBtn.title = 'Toggle Split Slider Comparison (Hotkey: Tab)';
  }
}

function disableSplitButton() {
  const toggleBtn = document.getElementById('view-mode-toggle');
  if (toggleBtn) {
    toggleBtn.disabled = true;
    toggleBtn.setAttribute('aria-disabled', 'true');
    toggleBtn.classList.add('btn-disabled');
    toggleBtn.title = 'Run synthesis first to compare before & after';
  }
}

function updateVUMeter(cLoss, sLoss, tvLoss) {
  const updateChannel = (stripId, valId, val, maxRef) => {
    const strip = document.getElementById(stripId);
    const readout = document.getElementById(valId);
    if (!strip || !readout) return;
    readout.textContent = val.toFixed(2);
    const leds = strip.querySelectorAll('.vu-led');
    const level = Math.min(6, Math.max(1, Math.round((val / maxRef) * 6)));
    leds.forEach((led, idx) => {
      led.className = 'vu-led';
      if (idx < level) {
        if (idx < 3) led.classList.add('active-green');
        else if (idx < 5) led.classList.add('active-amber');
        else led.classList.add('active-red');
      }
    });
  };
  updateChannel('vu-content-strip', 'vu-content-val', cLoss, 2.0);
  updateChannel('vu-style-strip', 'vu-style-val', sLoss, 50.0);
  updateChannel('vu-tv-strip', 'vu-tv-val', tvLoss, 4.0);
}

// ---------------------------------------------------------------------------
// Synthesis Execution (Real-time SSE Streaming)
// ---------------------------------------------------------------------------
async function startSynthesis() {
  if (state.isProcessing) return;
  playTactileClick('heavy');

  const btn = document.getElementById('synthesize-btn');
  const btnText = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');
  const progressWrap = document.getElementById('progress-container');
  const progressBar = document.getElementById('progress-bar-fill');
  const progressPct = document.getElementById('progress-pct');
  const hud = document.getElementById('live-telemetry-badge');
  const easelImg = document.getElementById('easel-image');

  state.isProcessing = true;
  btn.disabled = true;
  btnText.textContent = 'Synthesizing...';
  btnSpinner.classList.remove('hidden');
  progressWrap.classList.remove('hidden');
  hud.classList.remove('hidden');

  // Switch back to single frame view if split was on
  if (state.viewMode === 'split') toggleComparisonMode();

  const payload = {
    resolution: state.resolution,
    steps: state.steps,
    style_weight: state.styleIntensity,
    content_weight: state.alpha,
    tv_weight: state.tvWeight,
    optimizer: state.optimizer,
  };

  if (state.inputMode === 'presets') {
    payload.preset_id = state.activePresetId;
  } else {
    if (!state.customContentBase64 || !state.customStyleBase64) {
      showToast('Please upload both a content photograph and a style painting.', 'warning');
      resetSynthesisButton();
      return;
    }
    payload.content_base64 = state.customContentBase64;
    payload.style_base64 = state.customStyleBase64;
  }

  try {
    const response = await fetch('/api/stylize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // Keep incomplete chunk

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.replace('data: ', '').trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);

          if (event.type === 'progress') {
            const pct = Math.round((event.step / event.total) * 100);
            progressBar.style.width = `${pct}%`;
            progressPct.textContent = `${pct}%`;

            document.getElementById('hud-step').textContent = `${event.step} / ${event.total}`;
            document.getElementById('hud-loss').textContent = event.loss.toFixed(2);
            document.getElementById('metric-loss').textContent = event.loss.toFixed(2);

            updateVUMeter(event.content_loss || 1.0, event.style_loss || 20.0, event.tv_loss || 2.5);

            if (event.preview) {
              easelImg.src = event.preview;
            }
          } else if (event.type === 'complete') {
            state.hasSynthesized = true;
            state.activeResultUrl = event.result_url;
            easelImg.src = event.result_url;
            easelImg.style.visibility = 'visible';
            document.getElementById('split-before-img').src = state.activeContentUrl;
            document.getElementById('split-after-img').src = event.result_url;

            document.getElementById('metric-duration').textContent = `${event.elapsed}s`;
            document.getElementById('metric-res').textContent = `${event.resolution} px`;
            document.getElementById('hud-speed').textContent = `${event.throughput} it/s`;
            if (event.final_loss !== undefined) {
              const hudLoss = document.getElementById('hud-loss');
              if (hudLoss) hudLoss.textContent = Number(event.final_loss).toFixed(2);
            }
            const hudStep = document.getElementById('hud-step');
            if (hudStep) hudStep.textContent = `${state.steps} / ${state.steps}`;

            updateDownloadButton(event.result_url);
            enableSplitButton();
            const chip = document.getElementById('canvas-state-chip');
            if (chip) {
              chip.textContent = 'Synthesized Masterpiece';
              chip.className = 'canvas-state-chip synthesized';
            }

            // Auto-hide live telemetry overlay so the final artwork is completely unobstructed
            const hudBadge = document.getElementById('live-telemetry-badge');
            if (hudBadge) hudBadge.classList.add('hidden');
            const hudBtn = document.getElementById('hud-toggle-btn');
            if (hudBtn) {
              hudBtn.classList.remove('active');
              hudBtn.setAttribute('aria-pressed', 'false');
            }
          } else if (event.type === 'error') {
            showToast(`Optimization error: ${event.message}`, 'error');
          }
        } catch (e) {
          console.error('Error parsing SSE event:', e);
        }
      }
    }
  } catch (err) {
    console.error('Network or execution error:', err);
    showToast('Failed to connect to the synthesis engine.', 'error');
  } finally {
    resetSynthesisButton();
  }
}

function resetSynthesisButton() {
  state.isProcessing = false;
  const btn = document.getElementById('synthesize-btn');
  const btnText = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');
  btn.disabled = false;
  btnText.textContent = 'Synthesize Artwork';
  btnSpinner.classList.add('hidden');

  const hudBadge = document.getElementById('live-telemetry-badge');
  if (hudBadge) hudBadge.classList.add('hidden');
  const hudBtn = document.getElementById('hud-toggle-btn');
  if (hudBtn) hudBtn.classList.remove('active');
  const progressWrap = document.getElementById('progress-container');
  if (progressWrap) progressWrap.classList.add('hidden');
}

function updateDownloadButton(url) {
  const dlBtn = document.getElementById('download-btn');
  dlBtn.href = url;
  dlBtn.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Non-Intrusive Darkroom Toast Notifications
// ---------------------------------------------------------------------------
function showToast(message, type = 'warning') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `atelier-toast toast-${type}`;
  toast.setAttribute('role', 'alert');

  let icon = '⚡';
  if (type === 'error') icon = '✕';
  else if (type === 'success') icon = '✓';
  else if (type === 'warning') icon = '▲';

  toast.innerHTML = `<span class="toast-icon">${icon}</span><span>${message}</span>`;
  container.appendChild(toast);

  // Trigger smooth enter animation
  requestAnimationFrame(() => {
    toast.classList.add('show');
  });

  // Auto-dismiss after 3.5s
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 280);
  }, 3500);
}
