// Waiting state constants for better UX
const WAITING_STATES = {
  CAMERA: 'Waiting for camera...',
  CAMERA_STARTING: 'Starting camera...',
  CAMERA_PERMISSION: 'Requesting camera permission...',
  CAMERA_BUSY: 'Camera busy, retrying...',
  EMOTION_MODEL: 'Waiting for emotion model...',
  FACE_MODEL: 'Waiting for face detection model...',
  OBJECT_MODEL: 'Waiting for object detection model...',
  FACE_DETECTION: 'Waiting for face detection...',
  ANALYZING: 'Analyzing...',
  NO_FACE: 'No face detected - position face in frame',
  READY: 'Ready'
};

const initialState = {
  threatLevel: 24,
  monitoring: false,
  panicActive: false,
  panicCountdown: 0,
  statusMessage: 'The system is standing by and monitoring your surroundings.',
  contacts: [
    { id: 1, name: 'Maya', phone: '+1 555-0101', role: 'Primary contact' },
    { id: 2, name: 'Alex', phone: '+1 555-0102', role: 'Family' }
  ],
  incidents: [
    { id: 1, title: 'Low-risk movement detected', time: '10 mins ago', level: 'Low' },
    { id: 2, title: 'Emergency alert sent', time: '1 hour ago', level: 'High' }
  ],
  settings: {
    sms: true,
    push: true,
    email: false,
    threshold: 70
  },
  emotion: WAITING_STATES.CAMERA,
  motion: 'Still',
  objects: WAITING_STATES.CAMERA
};

let state = JSON.parse(JSON.stringify(initialState));
let panicTimer = null;
let detectionTimer = null;
let cameraStream = null;
let faceApiReady = false;
let isLoadingModels = false;
let cocoModel = null;
const SMOOTHING_WINDOW = 2;
let expressionHistory = [];
let objectHistory = [];
let lastEmergencyAlertAt = 0;
let cameraRetryTimer = null;
let cameraStartAttempts = 0;
let cameraRestarting = false;
let modelsPreloaded = false;
let firstDetectionDone = false;
let modelLoadPromise = null;
let lastModelRetryAt = 0;
const motionCanvas = document.createElement('canvas');
motionCanvas.width = 64;
motionCanvas.height = 48;
const motionContext = motionCanvas.getContext('2d', { willReadFrequently: true });
let previousMotionFrame = null;
let smoothedMotion = 0;

const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights';
const FACE_DETECTION_MIN_CONFIDENCE = 0.20;
const EMOTION_MIN_CONFIDENCE = 0.15;
const OBJECT_MIN_CONFIDENCE = 0.30;
const DETECTION_INTERVAL = 300;
const tabs = document.querySelectorAll('.tab');
const views = document.querySelectorAll('.view');
const quickCards = document.querySelectorAll('.quick-card');

// --- Text Analysis Keywords (mirrors Android TextAnalysisScreen.kt) ---
const TEXT_EMOTION_KEYWORDS = {
  Happy: ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 'love', 'lovely', 'fantastic', 'blessed', 'grateful', 'pleased', 'delighted', 'joyful', 'content', 'satisfied', 'ecstatic', 'fabulous', 'fun', 'enjoy', 'celebrate', 'beautiful'],
  Sad: ['sad', 'depressed', 'lonely', 'hurt', 'pain', 'miserable', 'empty', 'heartbroken', 'gloomy', 'unhappy', 'hopeless', 'helpless', 'tired', 'exhausted', 'drained', 'melancholy', 'suffering', 'broken', 'lost', 'cry', 'tears', 'worried', 'anxious', 'stressed', 'overwhelmed', 'burdened'],
  Fear: ['fear', 'afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'panic', 'paranoid', 'threatened', 'apprehensive', 'shaky', 'frightened', 'concerned', 'uneasy', 'dread', 'horror', 'terror'],
  Angry: ['angry', 'mad', 'furious', 'rage', 'hate', 'irritated', 'annoyed', 'frustrated', 'pissed', 'resentful', 'hostile', 'violent', 'livid', 'enraged', 'outraged', 'bitter', 'resentment'],
  Disgust: ['disgust', 'disgusted', 'nauseated', 'revolted', 'repulsed', 'sickened', 'gross', 'filthy', 'dirty', 'vile', 'detest'],
  Surprise: ['surprise', 'surprised', 'shocked', 'amazed', 'astonished', 'startled', 'unexpected', 'suddenly', 'wow', 'incredible'],
  Neutral: ['neutral', 'okay', 'fine', 'normal', 'calm', 'peaceful', 'relaxed', 'steady', 'stable', 'balanced', 'clear', 'focused']
};

function render() {
  document.getElementById('status-pill').textContent = state.monitoring ? 'Monitoring' : 'Ready';
  document.getElementById('threat-title').textContent = state.panicActive ? 'Emergency alert triggered' : getThreatTitle(state.threatLevel);
  document.getElementById('threat-message').textContent = state.panicActive ? 'A panic sequence is active. Contacts are being notified.' : state.statusMessage;
  document.getElementById('monitoring-status').textContent = state.monitoring ? 'Active' : 'Inactive';
  document.getElementById('contacts-count').textContent = state.contacts.length;
  document.getElementById('incidents-count').textContent = state.incidents.length;
  document.getElementById('threat-score').textContent = `${state.threatLevel}%`;
  document.getElementById('monitor-badge').textContent = state.monitoring ? 'Live' : 'Standby';
  document.getElementById('emotion-pill').textContent = state.emotion;
  document.getElementById('motion-pill').textContent = state.motion;
  document.getElementById('objects-pill').textContent = state.objects;

  const gauge = document.getElementById('gauge');
  const hue = Math.max(120, 360 - state.threatLevel * 2.8);
  gauge.style.background = `conic-gradient(hsl(${hue} 70% 56%) 0deg, hsl(${hue} 70% 50%) ${state.threatLevel * 3.6}deg, rgba(255,255,255,0.08) ${state.threatLevel * 3.6}deg)`;

  document.getElementById('monitor-toggle').textContent = state.monitoring ? 'Stop monitoring' : 'Start monitoring';
  document.getElementById('panic-btn').textContent = state.panicActive ? `Cancel SOS (${state.panicCountdown}s)` : 'Trigger SOS';
  document.getElementById('sms-toggle').checked = state.settings.sms;
  document.getElementById('push-toggle').checked = state.settings.push;
  document.getElementById('email-toggle').checked = state.settings.email;
  document.getElementById('threshold-range').value = state.settings.threshold;
  document.getElementById('threshold-value').textContent = `Threshold: ${state.settings.threshold}%`;

  const cameraShell = document.getElementById('camera-shell');
  const cameraStatus = document.getElementById('camera-status');
  const cameraFeed = document.getElementById('camera-feed');
  cameraShell.classList.toggle('active', state.monitoring);
  
  // Enhanced camera status messages based on current state
  if (state.monitoring) {
    if (state.emotion === WAITING_STATES.CAMERA_STARTING || state.emotion === WAITING_STATES.CAMERA_PERMISSION) {
      cameraStatus.textContent = state.emotion;
    } else if (state.emotion === WAITING_STATES.FACE_MODEL || state.emotion === WAITING_STATES.EMOTION_MODEL) {
      cameraStatus.textContent = state.emotion;
    } else if (state.emotion === WAITING_STATES.FACE_DETECTION) {
      cameraStatus.textContent = 'Face detected - analyzing emotion...';
    } else if (state.emotion === WAITING_STATES.NO_FACE) {
      cameraStatus.textContent = state.emotion;
    } else if (state.emotion === WAITING_STATES.ANALYZING) {
      cameraStatus.textContent = 'Analyzing frame...';
    } else if (faceApiReady) {
      cameraStatus.textContent = 'Live emotion analysis active';
    } else {
      cameraStatus.textContent = 'Starting camera and face model...';
    }
  } else {
    cameraStatus.textContent = 'Camera will start when monitoring begins.';
  }

  // Show/hide video feed based on monitoring state
  if (cameraFeed) {
    cameraFeed.style.display = state.monitoring ? 'block' : 'none';
  }

  // Add visual indicator classes to detail pills
  updateDetailPillState();

  renderContacts();
  renderHistory();
}

function updateDetailPillState() {
  const emotionPill = document.getElementById('emotion-pill');
  const motionPill = document.getElementById('motion-pill');
  const objectsPill = document.getElementById('objects-pill');
  const cameraStatus = document.getElementById('camera-status');
  
  // Reset classes
  [emotionPill, motionPill, objectsPill, cameraStatus].forEach(el => {
    el.classList.remove('waiting', 'active', 'error', 'loading-pulse');
  });
  
  // Determine state for emotion
  if (state.monitoring) {
    const waitingStates = Object.values(WAITING_STATES);
    const isWaiting = waitingStates.some(ws => state.emotion.includes(ws.replace('...', '').replace('Waiting for ', '').trim()));
    
    if (state.emotion === WAITING_STATES.NO_FACE) {
      emotionPill.classList.add('waiting');
      emotionPill.classList.add('loading-pulse');
    } else if (waitingStates.some(ws => state.emotion === ws || state.emotion.includes(ws.replace('...', '')))) {
      emotionPill.classList.add('waiting');
      emotionPill.classList.add('loading-pulse');
    } else if (state.emotion === 'Analysis paused' || state.emotion === 'Camera unavailable') {
      emotionPill.classList.add('error');
    } else {
      emotionPill.classList.add('active');
    }
    
    // Motion pill
    motionPill.classList.add('active');
    
    // Objects pill
    if (state.objects === WAITING_STATES.OBJECT_MODEL || state.objects === WAITING_STATES.CAMERA || state.objects === 'Loading object model...' || state.objects === 'Waiting for camera') {
      objectsPill.classList.add('waiting');
      objectsPill.classList.add('loading-pulse');
    } else if (state.objects.includes('detected') || state.objects !== 'No high-confidence objects detected') {
      objectsPill.classList.add('active');
    } else {
      objectsPill.classList.add('waiting');
    }
    
    // Camera status
    if (state.emotion === WAITING_STATES.NO_FACE) {
      cameraStatus.classList.add('waiting');
    } else if (waitingStates.some(ws => state.emotion === ws)) {
      cameraStatus.classList.add('waiting');
    } else if (state.emotion === 'Analysis paused' || state.emotion === 'Camera unavailable') {
      cameraStatus.classList.add('error');
    } else {
      cameraStatus.classList.add('active');
    }
  } else {
    // Not monitoring - all in waiting state
    emotionPill.classList.add('waiting');
    motionPill.classList.add('waiting');
    objectsPill.classList.add('waiting');
    cameraStatus.classList.add('waiting');
  }
}

// --- Persistence helpers -----------------------------------------------
function loadState() {
  try {
    const raw = localStorage.getItem('safeguard_state');
    if (raw) {
      const parsed = JSON.parse(raw);
      // merge parsed contacts/incidents/settings into current state
      if (Array.isArray(parsed.contacts)) state.contacts = parsed.contacts.concat(state.contacts.filter(c => !parsed.contacts.some(pc => pc.id === c.id)));
      if (Array.isArray(parsed.incidents)) state.incidents = parsed.incidents.concat(state.incidents.filter(i => !parsed.incidents.some(pi => pi.id === i.id)));
      if (parsed.settings) state.settings = Object.assign({}, state.settings, parsed.settings);
    }
  } catch (e) {
    console.warn('Failed to load saved state', e);
  }
}

function saveState() {
  try {
    const toSave = { contacts: state.contacts, incidents: state.incidents, settings: state.settings };
    localStorage.setItem('safeguard_state', JSON.stringify(toSave));
  } catch (e) {
    console.warn('Failed to save state', e);
  }
}

function getAlertPlan(emotionName, confidencePercent) {
  const normalizedEmotion = (emotionName || '').toLowerCase();
  const isCritical = confidencePercent >= 85 || ['fear', 'fearful', 'angry', 'disgust', 'disgusted'].includes(normalizedEmotion);
  const isWarning = confidencePercent >= 60 || ['sad', 'surprise', 'surprised'].includes(normalizedEmotion);
  const severity = isCritical ? 'critical' : isWarning ? 'warning' : 'info';

  const title = severity === 'critical'
    ? 'Critical safety incident'
    : severity === 'warning'
      ? 'Elevated safety concern'
      : 'Safety alert';

  const body = `${emotionName} detected with ${confidencePercent}% confidence.`;

  return {
    severity,
    title,
    body,
    shouldSendSms: state.settings.sms && severity !== 'info',
    shouldSendPush: state.settings.push,
    shouldSendEmail: state.settings.email && severity === 'critical'
  };
}

function triggerEmergencyAlert(emotionName, confidencePercent) {
  const shouldNotify = state.settings.sms || state.settings.push || state.settings.email;
  if (!shouldNotify) return;

  const now = Date.now();
  if (lastEmergencyAlertAt && now - lastEmergencyAlertAt < 20000) return;
  lastEmergencyAlertAt = now;

  const plan = getAlertPlan(emotionName, confidencePercent);
  const contacts = state.contacts.length
    ? state.contacts.map((contact) => contact.name).join(', ')
    : 'your emergency contacts';
  const message = `${plan.body} Alerting ${contacts}.`;

  state.threatLevel = Math.min(100, Math.max(plan.severity === 'critical' ? 95 : plan.severity === 'warning' ? 80 : 70, confidencePercent));
  state.statusMessage = message;
  state.incidents.unshift({
    id: Date.now(),
    title: plan.severity === 'critical' ? 'Critical alert sent' : 'Distress alert sent',
    time: 'Just now',
    level: plan.severity === 'critical' ? 'High' : 'Medium'
  });
  if (state.incidents.length > 8) {
    state.incidents = state.incidents.slice(0, 8);
  }

  if (plan.shouldSendPush && typeof window !== 'undefined' && 'Notification' in window) {
    if (Notification.permission === 'granted') {
      new Notification(plan.title, {
        body: message,
        tag: 'safeguard-distress',
        renotify: true
      });
    } else if (Notification.permission === 'default') {
      Notification.requestPermission().catch(() => {});
    }
  }

  if (plan.shouldSendSms && state.contacts.length) {
    console.info('Web alert: SMS channel requested for', state.contacts.length, 'contact(s)');
  }

  if (plan.shouldSendEmail) {
    console.info('Web alert: email-style escalation requested');
  }

  saveState();
  render();
}

// Load persisted state at startup
loadState();

// Preload models immediately at startup for faster detection
async function preloadModels() {
  await loadFaceApiModels();
}

// Start preloading immediately
preloadModels();

function getThreatTitle(level) {
  if (level < 40) return 'Environment appears safe';
  if (level < 60) return 'Mild distress signals detected';
  if (level < 80) return 'Elevated threat level';
  return 'High threat level';
}

function showView(viewName) {
  tabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.view === viewName));
  views.forEach((view) => view.classList.toggle('active', view.id === `${viewName}-view`));
}

// Escape user-supplied text before it is interpolated into innerHTML.
// Contact name/role/phone are free-text and are persisted to localStorage, so an
// unescaped value would re-execute on every page load (stored XSS).
function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderContacts() {
  const container = document.getElementById('contacts-list');
  container.innerHTML = state.contacts.map((c) => `
    <div class="list-item" data-id="${escapeHtml(c.id)}">
      <div>
        <strong>${escapeHtml(c.name)}</strong>
        <div class="helper-text">${escapeHtml(c.role)}</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <div class="helper-text" style="margin-right:8px;">${escapeHtml(c.phone)}</div>
        <button class="secondary-btn small contact-edit" data-id="${escapeHtml(c.id)}">Edit</button>
        <button class="secondary-btn small contact-delete" data-id="${escapeHtml(c.id)}">Delete</button>
      </div>
    </div>
  `).join('');

  // wire edit/delete handlers
  container.querySelectorAll('.contact-edit').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const id = Number(btn.dataset.id) || btn.dataset.id;
      startEditContact(id);
    });
  });
  container.querySelectorAll('.contact-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const id = Number(btn.dataset.id) || btn.dataset.id;
      deleteContactFromUI(id);
    });
  });
}

let editingContactId = null;

function startEditContact(id) {
  const contact = state.contacts.find(c => String(c.id) === String(id));
  if (!contact) return;
  editingContactId = contact.id;
  const modal = document.getElementById('add-contact-modal');
  document.getElementById('add-name').value = contact.name;
  document.getElementById('add-phone').value = contact.phone;
  document.getElementById('add-role').value = contact.role || '';
  modal.style.display = 'block';
  modal.setAttribute('aria-hidden', 'false');
  document.getElementById('add-name').focus();
}

function deleteContactFromUI(id) {
  if (!confirm('Delete this contact?')) return;
  state.contacts = state.contacts.filter(c => String(c.id) !== String(id));
  saveState();
  render();
}

function renderHistory() {
  const container = document.getElementById('history-list');
  container.innerHTML = state.incidents.map((item) => `
    <div class="list-item">
      <div>
        <strong>${item.title}</strong>
        <div class="helper-text">${item.time}</div>
      </div>
      <span class="badge">${item.level}</span>
    </div>
  `).join('');
}

// --- Text Analysis (mirrors Android TextAnalysisScreen.kt) ---
function analyzeText(text) {
  const lowerText = text.toLowerCase().trim();
  if (!lowerText) {
    return {
      emotion: 'Neutral',
      confidence: 0,
      threatScore: 0,
      breakdown: {},
      message: 'Please enter some text to analyze.'
    };
  }

  const words = lowerText.split(/[^a-z']+/).filter(w => w.length > 0);
  const wordCount = words.length;

  // Count keyword matches per emotion
  const scores = {};
  for (const [emotion, keywords] of Object.entries(TEXT_EMOTION_KEYWORDS)) {
    const matches = words.filter(w => keywords.includes(w)).length;
    if (matches > 0) {
      const proportion = matches / Math.max(1, wordCount);
      const frequencyFactor = Math.min(1, matches / keywords.length);
      scores[emotion] = Math.min(1, Math.max(0, proportion * 0.7 + frequencyFactor * 0.3));
    }
  }

  if (Object.keys(scores).length === 0) {
    return {
      emotion: 'Neutral',
      confidence: 0.3,
      threatScore: 0,
      breakdown: { Neutral: 1 },
      message: 'No strong emotional keywords detected. Tone appears neutral.'
    };
  }

  // Sort by score and take top emotion
  const sortedEmotions = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const topEmotion = sortedEmotions[0][0];
  const confidence = sortedEmotions[0][1];

  // Normalize breakdown
  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  const breakdown = {};
  for (const [emotion, score] of Object.entries(scores)) {
    breakdown[emotion] = score / Math.max(1e-6, totalScore);
  }

  // Compute threat score (simplified ThreatEngine)
  let threatScore = 0;
  if (topEmotion === 'Fear' || topEmotion === 'Angry') threatScore = Math.round(confidence * 90);
  else if (topEmotion === 'Sad' || topEmotion === 'Disgust') threatScore = Math.round(confidence * 60);
  else if (topEmotion === 'Surprise') threatScore = Math.round(confidence * 30);
  else threatScore = 0;

  const messages = {
    Fear: 'Potential distress detected in text. Consider reaching out for support.',
    Angry: 'Anger detected in text. Take a moment to breathe.',
    Sad: 'Sadness detected in text. Your feelings are valid.',
    Disgust: 'Discomfort detected in text.',
    Surprise: 'Surprise detected in text.',
    Happy: 'Positive tone detected. Keep it up!',
    Neutral: 'Neutral tone detected.'
  };

  return {
    emotion: topEmotion,
    confidence,
    threatScore,
    breakdown,
    message: messages[topEmotion] || messages.Neutral
  };
}

function renderTextAnalysisResult(result) {
  const container = document.getElementById('text-analysis-result');
  if (!container) return;
  
  const color = result.threatScore >= 70 ? '#ff4d4f' : result.threatScore >= 40 ? '#ffa940' : '#52c41a';
  
  container.style.display = 'block';
  container.innerHTML = `
    <div class="card" style="border-left:4px solid ${color};">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <h4 style="margin:0;">Detected Emotion</h4>
        <span style="font-size:1.25rem;font-weight:700;color:${color};">${result.emotion}</span>
      </div>
      <div style="margin-bottom:8px;color:var(--muted);">Confidence: ${Math.round(result.confidence * 100)}%</div>
      <div style="height:8px;background:rgba(0,0,0,0.1);border-radius:4px;overflow:hidden;margin-bottom:12px;">
        <div style="width:${result.confidence * 100}%;height:100%;background:${color};transition:width 0.3s;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <span>Threat Score: ${result.threatScore}%</span>
        <span style="color:${color};font-weight:600;">${result.threatScore >= 70 ? 'HIGH' : result.threatScore >= 40 ? 'MEDIUM' : 'LOW'}</span>
      </div>
      <p style="color:var(--muted);font-size:0.875rem;margin-bottom:12px;">${result.message}</p>
      ${Object.keys(result.breakdown).length > 0 ? `
        <div>
          <div style="font-size:0.75rem;color:var(--muted);margin-bottom:8px;text-transform:uppercase;">Breakdown:</div>
          ${Object.entries(result.breakdown).sort((a,b) => b[1]-a[1]).map(([emotion, score]) => `
            <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:0.8125rem;">
              <span>${emotion}</span>
              <span style="color:var(--muted);">${Math.round(score * 100)}%</span>
            </div>
          `).join('')}
        </div>
      ` : ''}
    </div>
  `;
}

function handleTextAnalysis() {
  const input = document.getElementById('text-input');
  if (!input) return;
  
  const text = input.value;
  const result = analyzeText(text);
  renderTextAnalysisResult(result);
  
  // If threat score is high, optionally trigger alert
  if (result.threatScore >= state.settings.threshold) {
    state.threatLevel = Math.max(state.threatLevel, result.threatScore);
    state.statusMessage = `Text analysis: ${result.emotion} (${Math.round(result.confidence * 100)}%) - ${result.message}`;
    state.incidents.unshift({
      id: Date.now(),
      title: `Text alert: ${result.emotion} detected`,
      time: 'Just now',
      level: result.threatScore >= 70 ? 'High' : 'Medium'
    });
    if (state.incidents.length > 8) state.incidents = state.incidents.slice(0, 8);
    saveState();
    render();
  }
}

function clearTextAnalysis() {
  const input = document.getElementById('text-input');
  const result = document.getElementById('text-analysis-result');
  if (input) input.value = '';
  if (result) {
    result.style.display = 'none';
    result.innerHTML = '';
  }
}

function normalizeEmotion(expressions) {
  if (!expressions) {
    return { label: 'No face detected', confidence: 0.1 };
  }

  const ranked = Object.entries(expressions)
    .filter(([, score]) => Number(score) >= EMOTION_MIN_CONFIDENCE)
    .map(([name, score]) => {
      let label = 'Neutral';
      switch (name) {
        case 'happy':
          label = 'Happy';
          break;
        case 'sad':
          label = 'Sad';
          break;
        case 'fearful':
          label = 'Fear';
          break;
        case 'angry':
          label = 'Angry';
          break;
        case 'disgusted':
          label = 'Disgust';
          break;
        case 'surprised':
          label = 'Surprise';
          break;
        case 'neutral':
          label = 'Neutral';
          break;
      }
      return { label, score: Number(score) };
    })
    .sort((a, b) => b.score - a.score);

  if (!ranked.length) {
    return { label: 'No face detected', confidence: 0.1 };
  }

  const top = ranked[0];
  const second = ranked[1];
  const useSecond = top.label === 'Neutral' && second && second.score >= 0.12 && (top.score - second.score) < 0.18;

  return {
    label: useSecond ? second.label : top.label,
    confidence: Math.min(0.99, Math.max(0.2, (useSecond ? second.score : top.score)))
  };
}

function summarizeObjects(objects) {
  return objects
    .filter((obj) => Number(obj.score) >= OBJECT_MIN_CONFIDENCE)
    .slice(0, 3)
    .map((obj) => ({
      label: String(obj.className || obj.class || 'object').replace(/_/g, ' '),
      score: Number(obj.score)
    }));
}

async function loadFaceApiModels() {
  if (faceApiReady && cocoModel) return;
  if (modelLoadPromise) return modelLoadPromise;

  isLoadingModels = true;
  modelLoadPromise = (async () => {
    if (typeof tf === 'undefined') throw new Error('TensorFlow.js did not load');
    try {
      await tf.setBackend('webgl');
    } catch (backendError) {
      console.warn('WebGL unavailable; TensorFlow will use its fallback backend.', backendError);
    }
    await tf.ready();

    const jobs = [];
    if (!faceApiReady) {
      jobs.push(Promise.all([
        faceapi.nets.ssdMobilenetv1.load(MODEL_URL),
        faceapi.nets.faceLandmark68Net.load(MODEL_URL),
        faceapi.nets.faceExpressionNet.load(MODEL_URL)
      ]).then(() => { faceApiReady = true; }));
    }
    if (!cocoModel) {
      jobs.push(cocoSsd.load({ base: 'lite_mobilenet_v2' }).then(model => { cocoModel = model; }));
    }

    const results = await Promise.allSettled(jobs);
    results.filter(result => result.status === 'rejected').forEach(result => {
      console.warn('Detection model load failed; it will be retried.', result.reason);
    });
    modelsPreloaded = faceApiReady && Boolean(cocoModel);
  })().finally(() => {
    isLoadingModels = false;
    modelLoadPromise = null;
  });
  return modelLoadPromise;
}

function stopDetectionLoop() {
  if (detectionTimer) {
    clearInterval(detectionTimer);
    detectionTimer = null;
  }
}

function detectMotion(video) {
  if (!motionContext || video.readyState < 2) return { label: 'Detecting', intensity: 0 };
  motionContext.drawImage(video, 0, 0, motionCanvas.width, motionCanvas.height);
  const current = motionContext.getImageData(0, 0, motionCanvas.width, motionCanvas.height).data;
  if (!previousMotionFrame) {
    previousMotionFrame = new Uint8ClampedArray(current);
    return { label: 'Still', intensity: 0 };
  }
  let changed = 0;
  const pixels = motionCanvas.width * motionCanvas.height;
  for (let i = 0; i < current.length; i += 4) {
    const delta = (Math.abs(current[i] - previousMotionFrame[i]) + Math.abs(current[i + 1] - previousMotionFrame[i + 1]) + Math.abs(current[i + 2] - previousMotionFrame[i + 2])) / 3;
    if (delta > 24) changed++;
  }
  previousMotionFrame.set(current);
  smoothedMotion = 0.35 * (changed / pixels) + 0.65 * smoothedMotion;
  const label = smoothedMotion >= 0.32 ? 'Possible Struggle' : smoothedMotion >= 0.20 ? 'Rapid Movement' : smoothedMotion >= 0.06 ? 'Normal Movement' : 'Still';
  return { label, intensity: Math.min(1, smoothedMotion / 0.32) };
}

function stopCameraStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  const video = document.getElementById('camera-feed');
  if (video) {
    video.srcObject = null;
  }
}

async function ensureCameraStream() {
  if (!state.monitoring) return false;
  if (cameraRestarting) return false;

  const video = document.getElementById('camera-feed');
  if (!video) return false;

  const hasActiveStream = cameraStream && video.srcObject === cameraStream && video.readyState >= 2 && !video.paused;
  if (hasActiveStream) return true;

  cameraRestarting = true;
  try {
    stopCameraStream();
    await wait(300);
    const stream = await requestCameraStream();
    cameraStream = stream;
    video.srcObject = stream;
    await video.play();
    cameraRestarting = false;
    return true;
  } catch (error) {
    cameraRestarting = false;
    if (error?.name === 'NotReadableError') {
      state.statusMessage = 'Camera is busy. Reconnecting...';
      render();
    }
    return false;
  }
}

function startDetectionLoop() {
  stopDetectionLoop();
  firstDetectionDone = false;
  detectionTimer = window.setInterval(async () => {
    try {
      if (!state.monitoring) return;

      const video = document.getElementById('camera-feed');
      if (!video) return;

      const streamReady = await ensureCameraStream();
      if (!streamReady) {
        state.emotion = WAITING_STATES.CAMERA_BUSY;
        state.objects = WAITING_STATES.CAMERA;
        render();
        return;
      }

      if ((!faceApiReady || !cocoModel) && Date.now() - lastModelRetryAt > 5000) {
        lastModelRetryAt = Date.now();
        await loadFaceApiModels();
      }

      let emotionConfidencePercent = 0;
      let emotionLabel = WAITING_STATES.NO_FACE;
      const motionResult = detectMotion(video);
      state.motion = motionResult.label;

      if (faceApiReady) {
        const detection = await faceapi
          .detectSingleFace(video, new faceapi.SsdMobilenetv1Options({ minConfidence: FACE_DETECTION_MIN_CONFIDENCE }))
          .withFaceLandmarks()
          .withFaceExpressions();

        if (detection && detection.expressions) {
          const rawExpressions = detection.expressions;
          
          if (!firstDetectionDone) {
            // Immediate result on first detection - no smoothing delay
            const emotionResult = normalizeEmotion(rawExpressions);
            emotionConfidencePercent = Math.round(emotionResult.confidence * 100);
            emotionLabel = emotionResult.label;
            state.emotion = emotionLabel;
            firstDetectionDone = true;
          } else {
            // Apply smoothing for subsequent frames
            expressionHistory.push(rawExpressions);
            if (expressionHistory.length > SMOOTHING_WINDOW) expressionHistory.shift();

            const avg = {};
            expressionHistory.forEach(map => {
              Object.entries(map).forEach(([k, v]) => { avg[k] = (avg[k] || 0) + v; });
            });
            Object.keys(avg).forEach(k => { avg[k] = avg[k] / expressionHistory.length; });

            const emotionResult = normalizeEmotion(avg);
            emotionConfidencePercent = Math.round(emotionResult.confidence * 100);
            emotionLabel = emotionResult.label;
            state.emotion = emotionLabel;
          }
        } else {
          expressionHistory = [];
          state.emotion = WAITING_STATES.NO_FACE;
        }
      } else {
        state.emotion = WAITING_STATES.EMOTION_MODEL;
      }

      let detectedObjectLabels = [];
      if (cocoModel) {
        try {
          const objs = await cocoModel.detect(video);
          const highConfidenceObjects = summarizeObjects(objs);
          
          if (!firstDetectionDone || objectHistory.length === 0) {
            // Immediate result on first detection
            state.objects = highConfidenceObjects.length
              ? highConfidenceObjects.map(obj => `${obj.label} (${Math.round(obj.score * 100)}%)`).join(', ')
              : 'No high-confidence objects detected';
            detectedObjectLabels = highConfidenceObjects.map(obj => obj.label.toLowerCase());
            if (objectHistory.length === 0 && highConfidenceObjects.length > 0) {
              firstDetectionDone = true;
            }
          } else {
            // Apply smoothing for subsequent frames
            const frameMap = {};
            highConfidenceObjects.forEach((obj) => {
              frameMap[obj.label] = Math.max(frameMap[obj.label] || 0, obj.score);
            });
            objectHistory.push(frameMap);
            if (objectHistory.length > SMOOTHING_WINDOW) objectHistory.shift();

            const agg = {};
            objectHistory.forEach(m => Object.entries(m).forEach(([k, v]) => { agg[k] = (agg[k] || 0) + v; }));
            Object.keys(agg).forEach(k => { agg[k] = agg[k] / objectHistory.length; });
            const top = Object.entries(agg).sort((a, b) => b[1] - a[1]).slice(0, 2);
            state.objects = top.length
              ? top.map(([label, score]) => `${label} (${Math.round(score * 100)}%)`).join(', ')
              : 'No high-confidence objects detected';
            
            detectedObjectLabels = top.map(([label]) => label.toLowerCase());
          }
        } catch (e) {
          console.warn('Object detection error', e);
        }
      } else {
        state.objects = WAITING_STATES.OBJECT_MODEL;
      }

      let baseScore = 20;
      if (emotionLabel !== WAITING_STATES.NO_FACE) {
         baseScore = Math.max(baseScore, emotionConfidencePercent);
         state.statusMessage = `${emotionLabel} detected with ${emotionConfidencePercent}% confidence.`;
      } else {
         state.statusMessage = 'Please face the camera clearly for emotion analysis.';
      }

      const criticalObjects = ['knife', 'gun', 'pistol', 'firearm', 'sword', 'weapon', 'scissors', 'baseball bat', 'bottle', 'cup'];
      const hasCriticalObject = detectedObjectLabels.some(obj => criticalObjects.some(c => obj.includes(c)));
      
      const isBottleOrCup = detectedObjectLabels.some(obj => obj.includes('bottle') || obj.includes('cup'));
      const criticalLabel = isBottleOrCup ? 'Harmful object' : 'Weapon';
      
      const distressEmotions = ['Sad', 'Fear', 'Angry', 'Disgust'];
      const hasDistress = distressEmotions.includes(emotionLabel) && emotionConfidencePercent >= 35;

      if (hasCriticalObject) {
         baseScore = Math.max(baseScore + 35, 85);
         if (hasDistress) baseScore = Math.min(100, baseScore + 15);
         state.statusMessage = `CRITICAL: ${criticalLabel} detected (${detectedObjectLabels.join(', ')})!`;
      }

      if (motionResult.label === 'Possible Struggle') {
        baseScore = Math.max(baseScore, 85);
        state.statusMessage = 'Possible struggle detected from rapid scene movement.';
      } else if (motionResult.label === 'Rapid Movement') {
        baseScore = Math.max(baseScore, 60);
      }

      state.threatLevel = Math.min(100, baseScore);

      if (state.threatLevel >= state.settings.threshold || (hasDistress && emotionConfidencePercent >= state.settings.threshold)) {
         triggerEmergencyAlert(hasCriticalObject ? criticalLabel : emotionLabel, state.threatLevel);
      }

      render();
    } catch (error) {
      state.emotion = 'Analysis paused';
      state.statusMessage = 'Monitoring paused. Please allow camera access.';
      render();
    }
  }, DETECTION_INTERVAL);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function resetVideoElement(video) {
  try {
    if (video) {
      video.pause();
      video.srcObject = null;
    }
  } catch (error) {
    console.warn('Failed to reset camera video element', error);
  }
}

function getCameraErrorMessage(error) {
  const name = error?.name || 'Error';
  if (name === 'NotReadableError') {
    return 'Camera is busy or not readable right now. Close other apps using the camera and try again.';
  }
  if (name === 'NotAllowedError') {
    return 'Camera permission was denied. Please allow camera access in Chrome and try again.';
  }
  if (name === 'NotFoundError') {
    return 'No camera was found on this device.';
  }
  if (name === 'OverconstrainedError') {
    return 'The selected camera settings are not supported. Trying a simpler camera setup.';
  }
  return `Camera unavailable (${name}). Please allow camera access to start live monitoring.`;
}

async function requestCameraStream() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera access is not supported in this browser.');
  }

  const attempts = [
    {
      video: {
        facingMode: 'user',
        width: { max: 640 },
        height: { max: 480 }
      },
      audio: false
    },
    {
      video: {
        facingMode: 'user'
      },
      audio: false
    },
    {
      video: {
        width: { max: 320 },
        height: { max: 240 }
      },
      audio: false
    },
    {
      video: true,
      audio: false
    }
  ];

  const errors = [];
  for (const constraints of attempts) {
    try {
      stopCameraStream();
      await wait(400);
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      errors.push(error);
      if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') {
        break;
      }
    }
  }

  throw errors[errors.length - 1] || new Error('Camera could not be started.');
}

async function startMonitoring() {
  if (state.monitoring && !cameraRetryTimer) return;

  if (!cameraRetryTimer) {
    cameraStartAttempts = 0;
  }

  state.monitoring = true;
  state.emotion = WAITING_STATES.CAMERA_STARTING;
  state.objects = WAITING_STATES.CAMERA;
  state.statusMessage = 'Opening camera...';
  render();

  try {
    const video = document.getElementById('camera-feed');
    if (!video) {
      throw new Error('Camera element not found.');
    }

    state.emotion = WAITING_STATES.CAMERA_PERMISSION;
    render();

    const stream = await requestCameraStream();
    cameraStream = stream;
    video.srcObject = stream;
    try {
      await video.play();
    } catch (playError) {
      resetVideoElement(video);
      await wait(500);
      video.srcObject = stream;
      await video.play();
    }

    state.emotion = WAITING_STATES.FACE_MODEL;
    state.objects = WAITING_STATES.OBJECT_MODEL;
    state.statusMessage = 'Loading AI models...';
    render();

    // Models are preloaded, just verify they're ready
    await loadFaceApiModels();
    
    if (!faceApiReady) {
      state.emotion = WAITING_STATES.FACE_MODEL;
    } else {
      state.emotion = WAITING_STATES.FACE_DETECTION;
    }
    
    if (!cocoModel) {
      state.objects = WAITING_STATES.OBJECT_MODEL;
    }

    state.statusMessage = 'Camera is live. Detecting emotion...';
    clearCameraRetryTimer();
    cameraStartAttempts = 0;
    render();
    startDetectionLoop();
  } catch (error) {
    stopCameraStream();
    if (scheduleCameraRetry(error)) {
      return;
    }
    state.monitoring = false;
    state.emotion = WAITING_STATES.CAMERA;
    state.objects = WAITING_STATES.CAMERA;
    state.statusMessage = getCameraErrorMessage(error);
    render();
  }
}

function clearCameraRetryTimer() {
  if (cameraRetryTimer) {
    clearTimeout(cameraRetryTimer);
    cameraRetryTimer = null;
  }
}

function scheduleCameraRetry(error) {
  if (error?.name !== 'NotReadableError' || cameraStartAttempts >= 2) {
    return false;
  }

  clearCameraRetryTimer();
  cameraStartAttempts += 1;
  state.monitoring = true;
  state.statusMessage = 'Camera is busy. Retrying in a moment...';
  render();

  cameraRetryTimer = window.setTimeout(() => {
    cameraRetryTimer = null;
    startMonitoring().catch(() => {});
  }, 1500);

  return true;
}

function stopMonitoring() {
  clearCameraRetryTimer();
  stopDetectionLoop();
  stopCameraStream();
  state.monitoring = false;
  state.emotion = WAITING_STATES.CAMERA;
  state.motion = 'Still';
  previousMotionFrame = null;
  smoothedMotion = 0;
  state.objects = WAITING_STATES.CAMERA;
  state.statusMessage = 'Monitoring paused. The dashboard is ready for the next session.';
  render();
}

async function toggleMonitoring() {
  if (state.monitoring) {
    stopMonitoring();
    return;
  }

  await startMonitoring();
}

function startPanic() {
  if (state.panicActive) {
    clearInterval(panicTimer);
    state.panicActive = false;
    state.panicCountdown = 0;
    state.statusMessage = 'SOS sequence cancelled.';
    render();
    return;
  }

  state.panicActive = true;
  state.panicCountdown = 5;
  state.threatLevel = 100;
  state.statusMessage = 'Panic sequence activated. Emergency contacts will be notified.';
  render();

  panicTimer = setInterval(() => {
    state.panicCountdown -= 1;
    if (state.panicCountdown <= 0) {
      clearInterval(panicTimer);
      state.panicActive = false;
      state.panicCountdown = 0;
      state.incidents.unshift({ id: Date.now(), title: 'Emergency alert sent', time: 'Just now', level: 'High' });
      state.statusMessage = 'Emergency alert sent to your trusted contacts.';
      render();
    } else {
      render();
    }
  }, 1000);
}

function addContact() {
  // show modal form
  const modal = document.getElementById('add-contact-modal');
  if (!modal) return;
  modal.style.display = 'block';
  modal.setAttribute('aria-hidden', 'false');
  document.getElementById('add-name').focus();
}

function hideAddContactModal() {
  const modal = document.getElementById('add-contact-modal');
  if (!modal) return;
  modal.style.display = 'none';
  modal.setAttribute('aria-hidden', 'true');
  document.getElementById('add-name').value = '';
  document.getElementById('add-phone').value = '';
  document.getElementById('add-role').value = '';
}

function saveAddContactFromModal() {
  const name = document.getElementById('add-name').value.trim();
  const phone = document.getElementById('add-phone').value.trim();
  const role = document.getElementById('add-role').value.trim() || 'Trusted contact';
  if (!name || !phone) {
    alert('Please provide both name and phone number.');
    return;
  }
  if (editingContactId) {
    // update existing
    state.contacts = state.contacts.map(c => String(c.id) === String(editingContactId) ? { ...c, name, phone, role } : c);
    editingContactId = null;
  } else {
    const contact = { id: Date.now(), name, phone, role };
    state.contacts.unshift(contact);
  }
  saveState();
  hideAddContactModal();
  render();
}

function handleSettings() {
  state.settings.sms = document.getElementById('sms-toggle').checked;
  state.settings.push = document.getElementById('push-toggle').checked;
  state.settings.email = document.getElementById('email-toggle').checked;
  state.settings.threshold = Number(document.getElementById('threshold-range').value);
  state.statusMessage = `Alert threshold updated to ${state.settings.threshold}%`;
  // Persist. Without this the toggles and threshold only lived in memory and were
  // silently lost on reload, unlike contacts/incidents which already call saveState().
  saveState();
  render();
}

tabs.forEach((tab) => tab.addEventListener('click', () => showView(tab.dataset.view)));
quickCards.forEach((card) => card.addEventListener('click', () => showView(card.dataset.view)));
document.getElementById('monitor-toggle').addEventListener('click', () => {
  toggleMonitoring().catch((error) => {
    state.monitoring = false;
    state.statusMessage = `Unable to start monitoring: ${error.message}`;
    render();
  });
});
document.getElementById('panic-btn').addEventListener('click', startPanic);
document.getElementById('add-contact-btn').addEventListener('click', addContact);
document.getElementById('save-add-contact').addEventListener('click', saveAddContactFromModal);
document.getElementById('cancel-add-contact').addEventListener('click', hideAddContactModal);
document.getElementById('sms-toggle').addEventListener('change', handleSettings);
document.getElementById('push-toggle').addEventListener('change', handleSettings);
document.getElementById('email-toggle').addEventListener('change', handleSettings);
document.getElementById('threshold-range').addEventListener('input', handleSettings);

// Text Analysis event handlers
document.getElementById('analyze-text-btn')?.addEventListener('click', handleTextAnalysis);
document.getElementById('clear-text-btn')?.addEventListener('click', clearTextAnalysis);
document.getElementById('text-input')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleTextAnalysis();
});

render();

// Enhance modal edit/delete behavior: show delete button when editing and wire handler
(function() {
  const modalId = 'add-contact-modal';
  const delBtnId = 'delete-add-contact';
  function showDeleteInModal(id) {
    const modal = document.getElementById(modalId);
    const delBtn = document.getElementById(delBtnId);
    if (!modal) return;
    try { modal.dataset.editId = String(id); } catch (e) {}
    if (delBtn) delBtn.style.display = 'inline-block';
  }
  function hideDeleteInModal() {
    const modal = document.getElementById(modalId);
    const delBtn = document.getElementById(delBtnId);
    if (delBtn) delBtn.style.display = 'none';
    if (modal && modal.dataset) delete modal.dataset.editId;
    if (typeof editingContactId !== 'undefined') editingContactId = null;
  }

  // wrap existing startEditContact/hideAddContactModal if present
  try {
    if (typeof startEditContact === 'function') {
      const _orig = startEditContact;
      window.startEditContact = function(id) { _orig(id); showDeleteInModal(id); };
    }
    if (typeof hideAddContactModal === 'function') {
      const _origHide = hideAddContactModal;
      window.hideAddContactModal = function() { _origHide(); hideDeleteInModal(); };
    }
  } catch (e) {
    // ignore
  }

  // wire delete button inside modal
  const delBtn = document.getElementById(delBtnId);
  if (delBtn) {
    delBtn.addEventListener('click', () => {
      const modal = document.getElementById(modalId);
      const editId = modal && modal.dataset && modal.dataset.editId ? Number(modal.dataset.editId) : null;
      if (!editId) return;
      if (!confirm('Delete this contact?')) return;
      state.contacts = state.contacts.filter((c) => c.id !== editId);
      saveState();
      hideAddContactModal();
      render();
    });
  }
})();

// Ensure global function bindings are wrapped (some environments bind function names differently)
try {
  if (typeof startEditContact === 'function') {
    const _origStart = startEditContact;
    startEditContact = function(id) {
      _origStart(id);
      const modal = document.getElementById('add-contact-modal');
      const delBtn = document.getElementById('delete-add-contact');
      try { if (modal) modal.dataset.editId = String(id); } catch (e) {}
      if (delBtn) delBtn.style.display = 'inline-block';
    };
    window.startEditContact = startEditContact;
  }
  if (typeof hideAddContactModal === 'function') {
    const _origHide = hideAddContactModal;
    hideAddContactModal = function() {
      _origHide();
      const delBtn = document.getElementById('delete-add-contact');
      if (delBtn) delBtn.style.display = 'none';
      const modal = document.getElementById('add-contact-modal');
      if (modal && modal.dataset) delete modal.dataset.editId;
      if (typeof editingContactId !== 'undefined') editingContactId = null;
    };
    window.hideAddContactModal = hideAddContactModal;
  }
} catch (e) {
  // ignore
}
