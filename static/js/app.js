/**
 * OmniDetect AI - Client-Side Studio Controller
 * Handles real-time video streaming, detection overlays, custom model training, and audio alerts.
 */

// Global Application State
const state = {
  activeModel: 'oxford_universe_detector.pt',
  availableModels: [],
  activeClasses: {},
  activeSection: 'all',
  taxonomy: null,
  lastUploadedFile: null,
  isScientificMode: true,
  confThreshold: 0.50,
  iouThreshold: 0.45,
  isCameraRunning: false,
  isNativeCam: false,
  cameraStream: null,
  isAudioAlertEnabled: false,
  isDetectingFrame: false,
  trainingPollingInterval: null,
  lastDetectionsCount: 0,
};

// DOM Elements
const elements = {
  // Tabs
  tabBtns: document.querySelectorAll('.nav-tab'),
  tabContents: document.querySelectorAll('.tab-content'),

  // Model & Class Controls
  modelSelect: document.getElementById('modelSelect'),
  btnUploadModel: document.getElementById('btnUploadModel'),
  modelFileInput: document.getElementById('modelFileInput'),
  classListContainer: document.getElementById('classListContainer'),
  totalClassesCount: document.getElementById('totalClassesCount'),

  // Sessions & Sections
  activeSessionName: document.getElementById('activeSessionName'),
  activeSessionCountBadge: document.getElementById('activeSessionCountBadge'),

  // Oxford Lexicon
  oxfordGrid: document.getElementById('oxfordGrid'),
  oxfordSearchInput: document.getElementById('oxfordSearchInput'),
  btnOxfordSearch: document.getElementById('btnOxfordSearch'),

  // Scientific AI Mode
  btnToggleScientific: document.getElementById('btnToggleScientific'),

  // Audio
  btnToggleAudio: document.getElementById('btnToggleAudio'),
  alertAudio: document.getElementById('alertAudio'),

  // Sliders
  confSlider: document.getElementById('confSlider'),
  confVal: document.getElementById('confVal'),
  iouSlider: document.getElementById('iouSlider'),
  iouVal: document.getElementById('iouVal'),

  // Live Camera
  btnStartCamera: document.getElementById('btnStartCamera'),
  btnStopCamera: document.getElementById('btnStopCamera'),
  btnToggleNativeCam: document.getElementById('btnToggleNativeCam'),
  cameraVideo: document.getElementById('cameraVideo'),
  cameraCanvas: document.getElementById('cameraCanvas'),
  nativeCamFeed: document.getElementById('nativeCamFeed'),
  cameraStatusBadge: document.getElementById('cameraStatusBadge'),
  cameraReticle: document.getElementById('cameraReticle'),
  btnSnapSample: document.getElementById('btnSnapSample'),

  // HUD
  camFpsVal: document.getElementById('camFpsVal'),
  camLatencyVal: document.getElementById('camLatencyVal'),
  camDetectionsVal: document.getElementById('camDetectionsVal'),
  liveDetectionsList: document.getElementById('liveDetectionsList'),
  liveDetCount: document.getElementById('liveDetCount'),

  // Image Studio
  imageDropzone: document.getElementById('imageDropzone'),
  imageFileInput: document.getElementById('imageFileInput'),
  imagePreview: document.getElementById('imagePreview'),
  imageBadge: document.getElementById('imageBadge'),
  btnDownloadAnnotated: document.getElementById('btnDownloadAnnotated'),
  imageStatsSummary: document.getElementById('imageStatsSummary'),
  imageDetectionsFeed: document.getElementById('imageDetectionsFeed'),
  imageBoxesList: document.getElementById('imageBoxesList'),

  // Custom Trainer
  datasetSelect: document.getElementById('datasetSelect'),
  btnGenerateDemoDataset: document.getElementById('btnGenerateDemoDataset'),
  newDatasetName: document.getElementById('newDatasetName'),
  newDatasetClasses: document.getElementById('newDatasetClasses'),
  btnCreateDataset: document.getElementById('btnCreateDataset'),
  trainModelName: document.getElementById('trainModelName'),
  trainEpochs: document.getElementById('trainEpochs'),
  btnStartTraining: document.getElementById('btnStartTraining'),
  trainStatusBadge: document.getElementById('trainStatusBadge'),
  trainEpochCount: document.getElementById('trainEpochCount'),
  trainProgressPercent: document.getElementById('trainProgressPercent'),
  trainProgressBar: document.getElementById('trainProgressBar'),
  trainLossVal: document.getElementById('trainLossVal'),
  trainMapVal: document.getElementById('trainMapVal'),
  trainingTerminal: document.getElementById('trainingTerminal'),
};

/* ==========================================================================
   1. Initialization & Navigation
   ========================================================================== */

document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupSliders();
  setupAudio();
  setupImageDropzone();
  setupCustomTrainer();
  setupSections();
  setupScientificAI();
  setupOxfordLexicon();

  await loadModels();
  await loadClasses();
  await loadDatasets();
});

function setupTabs() {
  elements.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.tabBtns.forEach(b => b.classList.remove('active'));
      elements.tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetTabId = btn.getAttribute('data-tab');
      const targetContent = document.getElementById(targetTabId);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });
}

/* ==========================================================================
   Sessions & Sections (10,000+ Object Universe & High-Precision Subsets)
   ========================================================================== */

async function setupSections() {
  try {
    const res = await fetch('/api/taxonomy');
    const data = await res.json();
    state.taxonomy = data.sections || {};

    const pills = document.querySelectorAll('.session-pill');
    pills.forEach(pill => {
      const secKey = pill.getAttribute('data-section');
      const secData = state.taxonomy[secKey];
      if (secData) {
        const badge = pill.querySelector('.pill-badge');
        if (badge) {
          badge.textContent = secData.count > 999 ? `${secData.count.toLocaleString()}+` : secData.count;
        }
      }

      pill.addEventListener('click', async () => {
        pills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');

        state.activeSection = secKey;
        const info = state.taxonomy[secKey] || { name: pill.querySelector('.pill-text').textContent, count: 0 };

        if (elements.activeSessionName) elements.activeSessionName.textContent = info.name;
        if (elements.activeSessionCountBadge) elements.activeSessionCountBadge.textContent = `${info.count.toLocaleString()} Active`;

        // Update class list in sidebar to reflect active session
        updateSidebarClassList(secKey);

        showNotification(`Activated Session: ${info.name} (${info.count} objects)`, 'info');

        // If an image is currently loaded in Image Studio, immediately re-analyze with this session
        if (state.lastUploadedFile) {
          processUploadedImage(state.lastUploadedFile);
        }
      });
    });
  } catch (err) {
    console.warn('Failed to load taxonomy metadata:', err);
  }
}

function updateSidebarClassList(secKey) {
  if (!state.taxonomy) return;
  const secData = state.taxonomy[secKey];
  if (!secData) return;

  const count = secData.count || 0;
  elements.totalClassesCount.textContent = `${count.toLocaleString()} Classes (${secData.name.split(' ')[0]})`;

  elements.classListContainer.innerHTML = '';
  const classesToShow = secData.classes && secData.classes.length > 0 ? secData.classes : Object.values(state.activeClasses);

  const sample = classesToShow.slice(0, 100);
  sample.forEach(name => {
    const tag = document.createElement('span');
    tag.style.cssText = 'display:inline-block; padding:3px 9px; margin:2px; background:rgba(0, 240, 255, 0.08); border:1px solid rgba(0, 240, 255, 0.2); border-radius:12px; font-size:11px; color:#fff;';
    tag.textContent = name;
    elements.classListContainer.appendChild(tag);
  });

  if (classesToShow.length > 100) {
    const more = document.createElement('span');
    more.style.cssText = 'display:inline-block; padding:3px 8px; margin:2px; color:var(--text-dim); font-size:11px;';
    more.textContent = `+${(classesToShow.length - 100).toLocaleString()} more classes in session...`;
    elements.classListContainer.appendChild(more);
  }
}

function setupScientificAI() {
  if (!elements.btnToggleScientific) return;
  elements.btnToggleScientific.addEventListener('click', () => {
    state.isScientificMode = !state.isScientificMode;
    elements.btnToggleScientific.classList.toggle('active', state.isScientificMode);
    const txt = elements.btnToggleScientific.querySelector('.science-text');
    if (txt) {
      txt.textContent = state.isScientificMode ? 'Scientific AI: ON' : 'Scientific AI: OFF';
    }
    showNotification(
      state.isScientificMode
        ? '🔬 Scientific AI Activated: Optical Morphology & Dual-Stage Multimodal Verification'
        : 'Scientific AI Disabled (Standard Mode)',
      'info'
    );
    if (state.lastUploadedFile) {
      processUploadedImage(state.lastUploadedFile);
    }
  });
}

/* ==========================================================================
   Oxford Dictionary Visual Knowledge Lexicon ("Their Looks")
   ========================================================================== */

async function setupOxfordLexicon() {
  const grid = elements.oxfordGrid;
  if (!grid) return;

  let allOxfordItems = {};
  try {
    const res = await fetch('/models/oxford_visual_knowledge.json');
    const data = await res.json();
    allOxfordItems = data.items || {};
  } catch (err) {
    console.warn('Could not load Oxford visual knowledge file:', err);
  }

  function renderCards(filterText = '') {
    grid.innerHTML = '';
    const q = filterText.toLowerCase().trim();
    let entries = Object.entries(allOxfordItems);
    if (q) {
      entries = entries.filter(([name, info]) => 
        name.includes(q) || 
        (info.looks && info.looks.toLowerCase().includes(q)) || 
        (info.definition && info.definition.toLowerCase().includes(q)) ||
        (info.category && info.category.toLowerCase().includes(q))
      );
    }

    const sample = entries.slice(0, 48); // Render 48 cards for ultra-smooth responsiveness
    sample.forEach(([name, info]) => {
      const card = document.createElement('div');
      card.className = 'dock-card';
      card.style.cssText = 'border:1px solid rgba(0, 240, 255, 0.15); border-radius:12px; padding:16px; background:rgba(18, 24, 42, 0.85); display:flex; flex-direction:column; gap:8px;';
      
      let icon = '📦';
      if (info.category === 'stationery') icon = '✏️';
      else if (info.category === 'electronics') icon = '💻';
      else if (info.category === 'personal_care') icon = '🧴';
      else if (info.category === 'furniture') icon = '🛋️';
      else if (info.category === 'sports') icon = '⚾';
      else if (info.category === 'kitchen_dining') icon = '🍽️';
      else if (info.category === 'vehicles') icon = '🚗';
      else if (info.category === 'tools_hardware') icon = '🛠️';
      else if (info.category === 'clothing') icon = '👕';
      else if (info.category === 'animals_nature') icon = '🐕';

      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:700; font-size:16px; color:#fff; text-transform:capitalize;">${icon} ${name}</span>
          <span style="font-size:10px; padding:2px 8px; border-radius:10px; background:rgba(0, 240, 255, 0.12); color:var(--accent-cyan); text-transform:uppercase; font-weight:600;">${(info.category || 'artifact').replace('_', ' ')}</span>
        </div>
        <div style="font-size:12px; color:var(--text-muted); line-height:1.4;">
          <strong style="color:var(--text-main);">📘 Oxford Definition:</strong> ${info.definition || 'Physical entity'}
        </div>
        <div style="font-size:12px; color:var(--accent-emerald); background:rgba(16, 240, 142, 0.08); padding:8px 10px; border-radius:8px; border:1px solid rgba(16, 240, 142, 0.2); line-height:1.4;">
          <strong style="color:#fff;">👁️ Their Looks:</strong> ${info.looks || 'Distinct physical visual characteristics.'}
        </div>
      `;
      grid.appendChild(card);
    });

    if (entries.length === 0) {
      grid.innerHTML = '<div style="color:var(--text-dim); padding:20px; font-size:13px; grid-column:1/-1;">No matching Oxford items found. Try searching marker, pen, pencil, smartphone, television, perfume bottle, or water bottle.</div>';
    }
  }

  if (elements.btnOxfordSearch) {
    elements.btnOxfordSearch.addEventListener('click', () => {
      renderCards(elements.oxfordSearchInput.value);
    });
  }
  if (elements.oxfordSearchInput) {
    elements.oxfordSearchInput.addEventListener('input', (e) => {
      renderCards(e.target.value);
    });
  }

  renderCards();
}

function setupSliders() {
  elements.confSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    state.confThreshold = val / 100.0;
    elements.confVal.textContent = `${val}%`;
  });

  elements.iouSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    state.iouThreshold = val / 100.0;
    elements.iouVal.textContent = `${val}%`;
  });
}

function setupAudio() {
  // Synthesize a clean modern audio beep using Web Audio API
  let audioCtx = null;

  elements.btnToggleAudio.addEventListener('click', () => {
    state.isAudioAlertEnabled = !state.isAudioAlertEnabled;
    elements.btnToggleAudio.classList.toggle('active-audio', state.isAudioAlertEnabled);
    if (state.isAudioAlertEnabled && !audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
  });

  window.playAlertBeep = function() {
    if (!state.isAudioAlertEnabled) return;
    try {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
      osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.12);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.13);
    } catch (e) {
      console.warn('Audio playback error', e);
    }
  };
}

/* ==========================================================================
   2. Models & Class Selection
   ========================================================================== */

async function loadModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    state.availableModels = data.models || [];
    state.activeModel = data.active_model || 'oxford_universe_detector.pt';

    elements.modelSelect.innerHTML = '';
    state.availableModels.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      let icon = '📦';
      if (m.id.includes('oxford')) icon = '📘';
      else if (m.id.includes('10000')) icon = '🌌';
      else if (m.id.includes('precision')) icon = '🎯';
      else if (m.id.includes('5000')) icon = '🪐';
      else if (m.id.includes('1000')) icon = '⚡';
      else if (m.id.includes('general')) icon = '🌐';

      opt.textContent = `${icon} ${m.name} (${m.size_mb} MB)`;
      if (m.is_active || m.id === state.activeModel) opt.selected = true;
      elements.modelSelect.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to load models:', err);
  }
}

elements.modelSelect.addEventListener('change', async (e) => {
  const selectedModel = e.target.value;
  try {
    const res = await fetch('/api/models/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: selectedModel })
    });
    const data = await res.json();
    if (data.status === 'success') {
      state.activeModel = selectedModel;
      await loadClasses();
      showNotification(`Switched to model: ${selectedModel}`, 'success');
    }
  } catch (err) {
    console.error('Error switching model:', err);
  }
});

// Model Upload (.pt file)
elements.btnUploadModel.addEventListener('click', () => {
  elements.modelFileInput.click();
});

elements.modelFileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    showNotification('Uploading custom model...', 'info');
    const res = await fetch('/api/models/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.status === 'success') {
      showNotification(data.message, 'success');
      await loadModels();
      await loadClasses();
    } else {
      showNotification(data.detail || 'Upload failed', 'error');
    }
  } catch (err) {
    showNotification('Error uploading model', 'error');
  } finally {
    elements.modelFileInput.value = '';
  }
});

async function loadClasses() {
  try {
    const res = await fetch('/api/classes');
    const data = await res.json();
    state.activeClasses = data.classes || {};
    const classCount = Object.keys(state.activeClasses).length;

    if (state.activeSection !== 'all' && state.taxonomy && state.taxonomy[state.activeSection]) {
      updateSidebarClassList(state.activeSection);
    } else {
      elements.totalClassesCount.textContent = `${classCount.toLocaleString()} Classes`;
      elements.classListContainer.innerHTML = '';
      const entries = Object.entries(state.activeClasses);
      entries.slice(0, 100).forEach(([id, name]) => {
        const tag = document.createElement('span');
        tag.style.cssText = 'display:inline-block; padding:2px 8px; margin:2px; background:rgba(255,255,255,0.06); border-radius:12px; font-size:11px;';
        tag.textContent = name;
        elements.classListContainer.appendChild(tag);
      });
      if (entries.length > 100) {
        const more = document.createElement('span');
        more.style.cssText = 'display:inline-block; padding:2px 8px; margin:2px; color:var(--text-dim); font-size:11px;';
        more.textContent = `+${(entries.length - 100).toLocaleString()} more classes...`;
        elements.classListContainer.appendChild(more);
      }
    }
  } catch (err) {
    console.error('Error loading classes:', err);
  }
}

/* ==========================================================================
   3. Real-Time Camera Stream & Inference
   ========================================================================== */

elements.btnStartCamera.addEventListener('click', async () => {
  if (state.isNativeCam) {
    startNativeCamFeed();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });

    elements.cameraVideo.srcObject = stream;
    state.cameraStream = stream;
    state.isCameraRunning = true;

    elements.cameraVideo.onloadedmetadata = () => {
      elements.cameraVideo.play();
      elements.cameraCanvas.width = elements.cameraVideo.videoWidth;
      elements.cameraCanvas.height = elements.cameraVideo.videoHeight;
      elements.cameraStatusBadge.textContent = 'LIVE';
      elements.cameraStatusBadge.style.color = 'var(--accent-emerald)';
      elements.cameraStatusBadge.style.borderColor = 'var(--accent-emerald)';

      elements.btnStartCamera.style.display = 'none';
      elements.btnStopCamera.style.display = 'inline-flex';

      runCameraDetectionLoop();
    };
  } catch (err) {
    console.warn('Browser webcam error:', err);
    showNotification('Could not open browser webcam. Falling back to native OpenCV camera...', 'info');
    elements.btnToggleNativeCam.click();
  }
});

elements.btnStopCamera.addEventListener('click', () => {
  stopCamera();
});

function stopCamera() {
  state.isCameraRunning = false;
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(track => track.stop());
    state.cameraStream = null;
  }
  elements.cameraVideo.srcObject = null;

  // Clear canvas
  const ctx = elements.cameraCanvas.getContext('2d');
  ctx.clearRect(0, 0, elements.cameraCanvas.width, elements.cameraCanvas.height);

  elements.cameraStatusBadge.textContent = 'STANDBY';
  elements.cameraStatusBadge.style.color = 'var(--accent-cyan)';
  elements.cameraStatusBadge.style.borderColor = 'rgba(0, 240, 255, 0.3)';

  elements.btnStartCamera.style.display = 'inline-flex';
  elements.btnStopCamera.style.display = 'none';
  elements.nativeCamFeed.style.display = 'none';
  elements.cameraVideo.style.display = 'block';

  elements.camFpsVal.textContent = '0.0';
  elements.camLatencyVal.textContent = '0 ms';
  elements.camDetectionsVal.textContent = '0';
  elements.liveDetectionsList.innerHTML = '<span style="color:var(--text-dim); font-size:12px;">Camera stopped</span>';
  elements.liveDetCount.textContent = '0 Items';
}

// Toggle between Browser MediaDevices and Direct Native OpenCV MJPEG
elements.btnToggleNativeCam.addEventListener('click', () => {
  state.isNativeCam = !state.isNativeCam;
  if (state.isNativeCam) {
    elements.btnToggleNativeCam.textContent = 'Switch to Browser WebCam';
    elements.btnToggleNativeCam.classList.add('btn-emerald');
    if (state.isCameraRunning) {
      stopCamera();
      startNativeCamFeed();
    }
  } else {
    elements.btnToggleNativeCam.textContent = 'Switch to Native OpenCV Cam';
    elements.btnToggleNativeCam.classList.remove('btn-emerald');
    if (state.isCameraRunning) {
      stopCamera();
      elements.btnStartCamera.click();
    }
  }
});

function startNativeCamFeed() {
  state.isCameraRunning = true;
  elements.cameraVideo.style.display = 'none';
  elements.nativeCamFeed.style.display = 'block';
  elements.nativeCamFeed.src = `/api/stream/native_webcam?conf=${state.confThreshold}&t=${Date.now()}`;
  elements.cameraStatusBadge.textContent = 'NATIVE CAM';
  elements.btnStartCamera.style.display = 'none';
  elements.btnStopCamera.style.display = 'inline-flex';
}

// Capture off-screen video frames and process with FastAPI
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

async function runCameraDetectionLoop() {
  if (!state.isCameraRunning || state.isNativeCam) return;

  if (!state.isDetectingFrame && elements.cameraVideo.readyState === elements.cameraVideo.HAVE_ENOUGH_DATA) {
    state.isDetectingFrame = true;

    // Scale down frame slightly for high throughput (max 640px)
    const scale = Math.min(1.0, 640 / elements.cameraVideo.videoWidth);
    captureCanvas.width = Math.round(elements.cameraVideo.videoWidth * scale);
    captureCanvas.height = Math.round(elements.cameraVideo.videoHeight * scale);

    captureCtx.drawImage(elements.cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);
    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.75);

    try {
      const res = await fetch('/api/detect/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataUrl,
          conf: state.confThreshold,
          iou: state.iouThreshold,
          return_image: false, // client renders boxes directly on canvas for highest FPS
          section: state.activeSection,
          scientific_mode: state.isScientificMode
        })
      });

      const data = await res.json();
      renderLiveDetections(data, scale);
    } catch (err) {
      console.warn('Frame detection frame dropped:', err);
    } finally {
      state.isDetectingFrame = false;
    }
  }

  // Next frame
  if (state.isCameraRunning) {
    requestAnimationFrame(runCameraDetectionLoop);
  }
}

function renderLiveDetections(data, scale) {
  const ctx = elements.cameraCanvas.getContext('2d');
  ctx.clearRect(0, 0, elements.cameraCanvas.width, elements.cameraCanvas.height);

  // Update HUD metrics
  elements.camFpsVal.textContent = data.fps || '0.0';
  elements.camLatencyVal.textContent = `${data.inference_time_ms || 0} ms`;
  elements.camDetectionsVal.textContent = data.total_detections || 0;
  elements.liveDetCount.textContent = `${data.total_detections || 0} Items`;

  // Trigger audio alert if new items entered view
  if (data.total_detections > state.lastDetectionsCount && state.isAudioAlertEnabled) {
    window.playAlertBeep();
  }
  state.lastDetectionsCount = data.total_detections;

  // Render bounding boxes on overlay canvas
  if (data.detections && data.detections.length > 0) {
    data.detections.forEach(det => {
      // Scale coordinates back to full video canvas size
      const [x1, y1, x2, y2] = det.box.map(v => v / scale);
      const w = x2 - x1;
      const h = y2 - y1;
      const color = det.color || '#00f0ff';

      // 1. Box semi-transparent fill
      ctx.fillStyle = hexToRgba(color, 0.12);
      ctx.fillRect(x1, y1, w, h);

      // 2. Crisp border
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, w, h);

      // 3. Modern Corner Accents
      const cLen = Math.min(18, w * 0.2, h * 0.2);
      ctx.lineWidth = 4;
      // Top Left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + cLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cLen, y1);
      ctx.stroke();
      // Top Right
      ctx.beginPath();
      ctx.moveTo(x2 - cLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cLen);
      ctx.stroke();
      // Bottom Left
      ctx.beginPath();
      ctx.moveTo(x1, y2 - cLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cLen, y2);
      ctx.stroke();
      // Bottom Right
      ctx.beginPath();
      ctx.moveTo(x2 - cLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cLen);
      ctx.stroke();

      // 4. Label Badge
      const sciPrefix = det.scientific_verified ? '🔬 ' : '';
      const label = `${sciPrefix}${det.class_name} ${Math.round(det.confidence * 100)}%`;
      ctx.font = '600 13px Outfit, sans-serif';
      const textMetrics = ctx.measureText(label);
      const badgeH = 22;
      const badgeW = textMetrics.width + 12;

      let badgeY = y1 - badgeH - 2;
      if (badgeY < 0) badgeY = y1 + 2;

      ctx.fillStyle = color;
      ctx.fillRect(x1, badgeY, badgeW, badgeH);

      ctx.fillStyle = '#060a12';
      ctx.fillText(label, x1 + 6, badgeY + 15);
    });

    // Populate live detection summary pills
    elements.liveDetectionsList.innerHTML = '';
    Object.entries(data.summary || {}).forEach(([clsName, count]) => {
      const badge = document.createElement('div');
      badge.className = 'det-badge';
      badge.innerHTML = `<span>${clsName}</span><span class="count">${count}</span>`;
      elements.liveDetectionsList.appendChild(badge);
    });
  } else {
    elements.liveDetectionsList.innerHTML = '<span style="color:var(--text-dim); font-size:12px;">No objects detected in frame.</span>';
  }
}

function hexToRgba(hex, alpha) {
  let c;
  if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
    c = hex.substring(1).split('');
    if (c.length === 3) c = [c[0], c[0], c[1], c[1], c[2], c[2]];
    c = '0x' + c.join('');
    return `rgba(${(c >> 16) & 255}, ${(c >> 8) & 255}, ${c & 255}, ${alpha})`;
  }
  return `rgba(0, 240, 255, ${alpha})`;
}

// Snap frame for dataset collector
elements.btnSnapSample.addEventListener('click', async () => {
  const selectedDataset = elements.datasetSelect.value;
  if (!selectedDataset) {
    showNotification('Please select or create a dataset first in the "Custom Trainer" tab!', 'error');
    // Switch to trainer tab
    document.getElementById('tabBtnTrainer').click();
    return;
  }

  if (!state.isCameraRunning) {
    showNotification('Please start the camera to snap samples', 'error');
    return;
  }

  // Snap current frame from video
  captureCanvas.width = elements.cameraVideo.videoWidth;
  captureCanvas.height = elements.cameraVideo.videoHeight;
  captureCtx.drawImage(elements.cameraVideo, 0, 0);

  captureCanvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('dataset_name', selectedDataset);
    formData.append('class_id', 0);
    // Center bounding box (0.5, 0.5, 0.5, 0.5)
    formData.append('box_normalized', '0.5,0.5,0.5,0.5');
    formData.append('image', blob, 'sample.jpg');

    try {
      const res = await fetch('/api/dataset/add_sample', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.status === 'success') {
        showNotification(`Sample captured & saved to ${selectedDataset} (${data.split} set)!`, 'success');
        await loadDatasets();
      }
    } catch (e) {
      showNotification('Failed to save sample', 'error');
    }
  }, 'image/jpeg', 0.95);
});

/* ==========================================================================
   4. Image & Media Studio (Upload & Inspect)
   ========================================================================== */

function setupImageDropzone() {
  const dropzone = elements.imageDropzone;
  const input = elements.imageFileInput;

  dropzone.addEventListener('click', () => input.click());

  ['dragenter', 'dragover'].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0) {
      processUploadedImage(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      processUploadedImage(e.target.files[0]);
    }
  });
}

async function processUploadedImage(file) {
  if (!file.type.startsWith('image/')) {
    showNotification('Please upload an image file (JPG, PNG, WebP)', 'error');
    return;
  }

  state.lastUploadedFile = file;

  elements.imageBadge.textContent = 'ANALYZING...';
  elements.imageBadge.style.color = 'var(--accent-cyan)';

  const formData = new FormData();
  formData.append('file', file);
  formData.append('conf', state.confThreshold);
  formData.append('iou', state.iouThreshold);
  formData.append('section', state.activeSection);
  formData.append('scientific_mode', state.isScientificMode);

  try {
    const res = await fetch('/api/detect/image', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    // Display annotated result image
    elements.imagePreview.src = data.annotated_image;
    elements.imagePreview.style.display = 'block';
    elements.imageDropzone.style.display = 'none';

    elements.imageBadge.textContent = `${data.total_detections} OBJECTS`;
    elements.imageBadge.style.color = 'var(--accent-emerald)';

    elements.btnDownloadAnnotated.style.display = 'inline-flex';
    elements.btnDownloadAnnotated.onclick = () => {
      const a = document.createElement('a');
      a.href = data.annotated_image;
      a.download = `omnidetect_${Date.now()}.jpg`;
      a.click();
    };

    elements.imageStatsSummary.textContent = `${data.image_width}x${data.image_height}px | Latency: ${data.inference_time_ms}ms | Model: ${data.model}`;

    // Summary feed
    elements.imageDetectionsFeed.innerHTML = '';
    Object.entries(data.summary || {}).forEach(([clsName, count]) => {
      const badge = document.createElement('div');
      badge.className = 'det-badge';
      badge.innerHTML = `<span>${clsName}</span><span class="count">${count}</span>`;
      elements.imageDetectionsFeed.appendChild(badge);
    });

    // Box Coordinates List
    elements.imageBoxesList.innerHTML = '';
    if (data.detections && data.detections.length > 0) {
      data.detections.forEach((det, idx) => {
        const item = document.createElement('div');
        item.style.cssText = 'padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);';
        const sciBadge = det.scientific_verified ? `<span style="font-size:10px; padding:1px 6px; background:rgba(16,240,142,0.15); color:var(--accent-emerald); border-radius:8px; margin-left:4px;">🔬 Scientific Verified</span>` : '';
        const oxLooks = det.oxford_looks ? `<div style="color:var(--accent-emerald); font-size:11px; margin-top:3px; background:rgba(16,240,142,0.06); padding:4px 8px; border-radius:6px;"><strong>👁️ Their Looks:</strong> ${det.oxford_looks}</div>` : '';
        const oxDefn = det.oxford_definition ? `<div style="color:var(--text-muted); font-size:10px; margin-top:2px;"><strong>📘 Oxford Definition:</strong> ${det.oxford_definition}</div>` : '';

        item.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="color:${det.color}">${idx + 1}. ${det.class_name}</strong> 
            <span style="color:var(--text-main); font-weight:700;">${Math.round(det.confidence * 100)}%</span>
          </div>
          ${sciBadge}
          ${oxLooks}
          ${oxDefn}
          <div style="color:var(--text-dim); font-size:10px; margin-top:2px;">Coordinates: [x1:${det.box[0]}, y1:${det.box[1]}, x2:${det.box[2]}, y2:${det.box[3]}]</div>
        `;
        elements.imageBoxesList.appendChild(item);
      });
    } else {
      elements.imageBoxesList.innerHTML = '<span style="color:var(--text-dim);">No objects found above confidence threshold.</span>';
    }
  } catch (err) {
    console.error('Image detection failed:', err);
    showNotification('Image detection failed', 'error');
    elements.imageBadge.textContent = 'ERROR';
  }
}

/* ==========================================================================
   5. Custom Model Trainer Studio
   ========================================================================== */

function setupCustomTrainer() {
  // Generate demo dataset
  elements.btnGenerateDemoDataset.addEventListener('click', async () => {
    showNotification('Generating instant demo dataset with 24 annotated images...', 'info');
    try {
      const res = await fetch('/api/dataset/create_sample_demo', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        showNotification(data.message, 'success');
        await loadDatasets();
        elements.datasetSelect.value = data.dataset_name;
      }
    } catch (e) {
      showNotification('Failed to generate demo dataset', 'error');
    }
  });

  // Create new dataset
  elements.btnCreateDataset.addEventListener('click', async () => {
    const name = elements.newDatasetName.value.trim();
    const classes = elements.newDatasetClasses.value.split(',').map(c => c.trim()).filter(Boolean);

    if (!name || classes.length === 0) {
      showNotification('Please enter a dataset name and at least one class', 'error');
      return;
    }

    try {
      const res = await fetch('/api/dataset/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, classes })
      });
      const data = await res.json();
      if (data.status === 'success') {
        showNotification(`Dataset '${data.dataset_name}' created!`, 'success');
        await loadDatasets();
        elements.datasetSelect.value = data.dataset_name;
        elements.newDatasetName.value = '';
        elements.newDatasetClasses.value = '';
      }
    } catch (e) {
      showNotification('Failed to create dataset', 'error');
    }
  });

  // Start Training
  elements.btnStartTraining.addEventListener('click', async () => {
    const datasetName = elements.datasetSelect.value;
    const outputModelName = elements.trainModelName.value.trim() || 'custom_model';
    const epochs = parseInt(elements.trainEpochs.value, 10);

    if (!datasetName) {
      showNotification('Please select a dataset to train on', 'error');
      return;
    }

    try {
      const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: datasetName,
          output_model_name: outputModelName,
          epochs: epochs
        })
      });
      const data = await res.json();

      if (data.status === 'started') {
        showNotification(`Training started on GPU for '${outputModelName}'!`, 'success');
        elements.btnStartTraining.disabled = true;
        startTrainingPolling();
      } else {
        showNotification(data.detail || 'Could not start training', 'error');
      }
    } catch (e) {
      showNotification('Error starting training job', 'error');
    }
  });
}

async function loadDatasets() {
  try {
    const res = await fetch('/api/datasets');
    const data = await res.json();
    const currentVal = elements.datasetSelect.value;
    elements.datasetSelect.innerHTML = '<option value="">-- Select Dataset --</option>';

    (data.datasets || []).forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.name;
      opt.textContent = `${d.name} (${d.total_images} images: [${d.classes.join(', ')}])`;
      if (d.name === currentVal) opt.selected = true;
      elements.datasetSelect.appendChild(opt);
    });
  } catch (e) {
    console.error('Error loading datasets', e);
  }
}

function startTrainingPolling() {
  if (state.trainingPollingInterval) clearInterval(state.trainingPollingInterval);

  elements.trainingTerminal.innerHTML = '<div class="terminal-line info">[Initializing training on NVIDIA GPU...]</div>';

  state.trainingPollingInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/train/status');
      const job = await res.json();

      // Update badge
      elements.trainStatusBadge.textContent = job.status.toUpperCase();
      elements.trainStatusBadge.style.color = job.status === 'completed' ? 'var(--accent-emerald)' : 'var(--accent-cyan)';

      // Update metrics
      elements.trainEpochCount.textContent = `${job.epoch}/${job.total_epochs}`;
      elements.trainProgressPercent.textContent = `${job.progress}%`;
      elements.trainProgressBar.style.width = `${job.progress}%`;
      elements.trainLossVal.textContent = job.loss.toFixed(4);
      elements.trainMapVal.textContent = job.map50.toFixed(4);

      // Terminal logs
      if (job.log && job.log.length > 0) {
        elements.trainingTerminal.innerHTML = job.log.map(line => {
          let cls = 'info';
          if (line.includes('completed') || line.includes('succeeded')) cls = 'success';
          if (line.includes('error') || line.includes('Error')) cls = 'error';
          return `<div class="terminal-line ${cls}">> ${line}</div>`;
        }).join('');
        elements.trainingTerminal.scrollTop = elements.trainingTerminal.scrollHeight;
      }

      if (job.status === 'completed') {
        clearInterval(state.trainingPollingInterval);
        elements.btnStartTraining.disabled = false;
        showNotification(`Training finished! New model '${job.output_model}' is now ready!`, 'success');
        await loadModels();
        // Auto-select newly trained model
        if (job.output_model) {
          elements.modelSelect.value = job.output_model;
          elements.modelSelect.dispatchEvent(new Event('change'));
        }
      } else if (job.status === 'failed') {
        clearInterval(state.trainingPollingInterval);
        elements.btnStartTraining.disabled = false;
        showNotification(`Training failed: ${job.error}`, 'error');
      }
    } catch (e) {
      console.warn('Status poll error', e);
    }
  }, 1000);
}

/* ==========================================================================
   6. UI Notifications
   ========================================================================== */

function showNotification(message, type = 'info') {
  const note = document.createElement('div');
  const colors = {
    info: '#00f0ff',
    success: '#10f08e',
    error: '#ff3c8a'
  };

  note.style.cssText = `
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: rgba(10, 14, 26, 0.92);
    backdrop-filter: blur(14px);
    border: 1px solid ${colors[type] || colors.info};
    color: #fff;
    padding: 12px 20px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
    z-index: 9999;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    display: flex;
    align-items: center;
    gap: 10px;
    animation: fadeIn 0.3s ease;
  `;

  note.innerHTML = `<span>●</span><span>${message}</span>`;
  document.body.appendChild(note);

  setTimeout(() => {
    note.style.opacity = '0';
    note.style.transform = 'translateY(10px)';
    note.style.transition = 'all 0.3s ease';
    setTimeout(() => note.remove(), 300);
  }, 3500);
}
