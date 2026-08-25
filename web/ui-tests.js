// Web UI Integration Tests - Run in browser console
// This tests that the UI correctly displays emotion, object, and motion detection results

const UI_TESTS = {
  passed: 0,
  failed: 0,
  
  assertEqual(actual, expected, msg) {
    if (actual !== expected) {
      throw new Error(`${msg}: expected "${expected}", got "${actual}"`);
    }
    this.passed++;
  },
  
  assertContains(text, substr, msg) {
    if (!text.includes(substr)) {
      throw new Error(`${msg}: "${text}" does not contain "${substr}"`);
    }
    this.passed++;
  },
  
  assertTrue(condition, msg) {
    if (!condition) throw new Error(msg || 'Assertion failed');
    this.passed++;
  },
  
  summary() {
    console.log(`\n=== UI Test Summary ===`);
    console.log(`Passed: ${this.passed}`);
    console.log(`Failed: ${this.failed}`);
    console.log(`${this.failed === 0 ? '✅ ALL UI TESTS PASSED' : '❌ SOME TESTS FAILED'}`);
    return this.failed === 0;
  }
};

// Mock video element for testing
function createMockVideo() {
  const video = document.createElement('video');
  video.id = 'camera-feed';
  video.style.display = 'none';
  document.body.appendChild(video);
  return video;
}

// Test 1: Verify UI elements exist
function testUIElementsExist() {
  console.log('\n--- Test: UI Elements Exist ---');
  
  const elements = {
    'emotion-pill': document.getElementById('emotion-pill'),
    'objects-pill': document.getElementById('objects-pill'),
    'motion-pill': document.getElementById('motion-pill'),
    'threat-score': document.getElementById('threat-score'),
    'camera-status': document.getElementById('camera-status'),
    'monitor-badge': document.getElementById('monitor-badge'),
    'camera-feed': document.getElementById('camera-feed')
  };
  
  for (const [id, el] of Object.entries(elements)) {
    UI_TESTS.assertTrue(!!el, `Element #${id} exists`);
  }
  
  console.log('✓ All UI elements exist');
}

// Test 2: Initial state rendering
function testInitialState() {
  console.log('\n--- Test: Initial State ---');
  
  // Reset to initial state
  state.monitoring = false;
  state.emotion = WAITING_STATES.CAMERA;
  state.objects = WAITING_STATES.CAMERA;
  state.motion = 'Still';
  state.threatLevel = 0;
  render();
  
  UI_TESTS.assertEqual(
    document.getElementById('emotion-pill').textContent,
    'Waiting for camera...',
    'Initial emotion shows waiting state'
  );
  
  UI_TESTS.assertEqual(
    document.getElementById('objects-pill').textContent,
    'Waiting for camera...',
    'Initial objects shows waiting state'
  );
  
  UI_TESTS.assertEqual(
    document.getElementById('motion-pill').textContent,
    'Still',
    'Initial motion shows Still'
  );
  
  UI_TESTS.assertEqual(
    document.getElementById('threat-score').textContent,
    '0%',
    'Initial threat level is 0%'
  );
  
  UI_TESTS.assertEqual(
    document.getElementById('monitor-badge').textContent,
    'Standby',
    'Monitor badge shows Standby'
  );
  
  console.log('✓ Initial state renders correctly');
}

// Test 3: Monitoring active state
function testMonitoringActiveState() {
  console.log('\n--- Test: Monitoring Active State ---');
  
  state.monitoring = true;
  state.emotion = 'Happy';
  state.objects = 'person (95%), book (80%)';
  state.motion = 'Normal Movement';
  state.threatLevel = 15;
  render();
  
  UI_TESTS.assertEqual(
    document.getElementById('emotion-pill').textContent,
    'Happy',
    'Shows detected emotion'
  );
  
  UI_TESTS.assertContains(
    document.getElementById('objects-pill').textContent,
    'person',
    'Shows detected objects'
  );
  
  UI_TESTS.assertEqual(
    document.getElementById('motion-pill').textContent,
    'Normal Movement',
    'Shows the detected motion state while monitoring'
  );
  
  UI_TESTS.assertEqual(
    document.getElementById('threat-score').textContent,
    '15%',
    'Shows threat level'
  );
  
  UI_TESTS.assertEqual(
    document.getElementById('monitor-badge').textContent,
    'Live',
    'Monitor badge shows Live'
  );
  
  // Check pill styling
  const emotionPill = document.getElementById('emotion-pill');
  UI_TESTS.assertTrue(
    emotionPill.classList.contains('active'),
    'Emotion pill has active class when monitoring'
  );
  
  console.log('✓ Monitoring active state renders correctly');
}

// Test 4: Threat level gauge updates
function testThreatLevelGauge() {
  console.log('\n--- Test: Threat Level Gauge Updates ---');
  
  const gauge = document.getElementById('gauge');
  const threatScore = document.getElementById('threat-score');
  
  // Test that threat score text updates
  state.threatLevel = 20;
  render();
  UI_TESTS.assertEqual(
    threatScore.textContent,
    '20%',
    'Threat score text shows 20%'
  );
  
  state.threatLevel = 60;
  render();
  UI_TESTS.assertEqual(
    threatScore.textContent,
    '60%',
    'Threat score text shows 60%'
  );
  
  state.threatLevel = 90;
  render();
  UI_TESTS.assertEqual(
    threatScore.textContent,
    '90%',
    'Threat score text shows 90%'
  );
  
  // Verify gauge element exists
  UI_TESTS.assertTrue(
    !!gauge,
    'Gauge element exists'
  );
  
  console.log('✓ Threat gauge text updates correctly');
}

// Test 5: Emotion pill colors based on emotion
function testEmotionPillColors() {
  console.log('\n--- Test: Emotion Pill Styling ---');
  
  state.monitoring = true;
  
  // Test distress emotions get error/active styling
  const distressEmotions = ['Fear', 'Sad', 'Angry', 'Disgust'];
  for (const emotion of distressEmotions) {
    state.emotion = emotion;
    render();
    const pill = document.getElementById('emotion-pill');
    UI_TESTS.assertTrue(
      pill.classList.contains('active'),
      `${emotion} pill has active class`
    );
  }
  
  // Test happy gets active styling
  state.emotion = 'Happy';
  render();
  UI_TESTS.assertTrue(
    document.getElementById('emotion-pill').classList.contains('active'),
    'Happy pill has active class'
  );
  
  // Test waiting state gets waiting class
  state.emotion = WAITING_STATES.NO_FACE;
  render();
  UI_TESTS.assertTrue(
    document.getElementById('emotion-pill').classList.contains('waiting'),
    'No face pill has waiting class'
  );
  UI_TESTS.assertTrue(
    document.getElementById('emotion-pill').classList.contains('loading-pulse'),
    'No face pill has loading pulse'
  );
  
  console.log('✓ Emotion pill styling updates correctly');
}

// Test 6: Camera status messages
function testCameraStatusMessages() {
  console.log('\n--- Test: Camera Status Messages ---');
  
  state.monitoring = true;
  
  const statusMap = {
    [WAITING_STATES.CAMERA_STARTING]: 'Starting camera...',
    [WAITING_STATES.CAMERA_PERMISSION]: 'Requesting camera permission...',
    [WAITING_STATES.FACE_MODEL]: 'Waiting for face detection model...',
    [WAITING_STATES.EMOTION_MODEL]: 'Waiting for emotion model...',
    [WAITING_STATES.FACE_DETECTION]: 'Face detected - analyzing emotion...',
    [WAITING_STATES.NO_FACE]: 'No face detected - position face in frame',
    [WAITING_STATES.ANALYZING]: 'Analyzing frame...'
  };
  
  for (const [emotionState, expectedMsg] of Object.entries(statusMap)) {
    state.emotion = emotionState;
    render();
    const actualMsg = document.getElementById('camera-status').textContent;
    UI_TESTS.assertContains(
      actualMsg,
      expectedMsg.replace('...', ''),
      `Camera status for ${emotionState}`
    );
  }
  
  // Test active state
  state.emotion = 'Happy';
  faceApiReady = true;
  render();
  UI_TESTS.assertContains(
    document.getElementById('camera-status').textContent,
    'Live emotion analysis active',
    'Active camera status'
  );
  
  console.log('✓ Camera status messages update correctly');
}

// Test 7: Object pill shows detected objects
function testObjectPillDisplay() {
  console.log('\n--- Test: Object Pill Display ---');
  
  state.monitoring = true;
  
  // Test with objects
  state.objects = 'knife (95%), person (80%)';
  render();
  UI_TESTS.assertContains(
    document.getElementById('objects-pill').textContent,
    'knife',
    'Shows knife detection'
  );
  UI_TESTS.assertContains(
    document.getElementById('objects-pill').textContent,
    '95%',
    'Shows confidence percentage'
  );
  
  // Test no objects
  state.objects = 'No high-confidence objects detected';
  render();
  UI_TESTS.assertEqual(
    document.getElementById('objects-pill').textContent,
    'No high-confidence objects detected',
    'Shows no objects message'
  );
  
  // Test waiting state
  state.objects = WAITING_STATES.OBJECT_MODEL;
  render();
  const objPill = document.getElementById('objects-pill');
  UI_TESTS.assertTrue(
    objPill.classList.contains('waiting'),
    'Object pill shows waiting state'
  );
  UI_TESTS.assertTrue(
    objPill.classList.contains('loading-pulse'),
    'Object pill shows loading pulse'
  );
  
  console.log('✓ Object pill display updates correctly');
}

// Test 8: Motion pill display
function testMotionPillDisplay() {
  console.log('\n--- Test: Motion Pill Display ---');
  
  state.monitoring = true;
  state.motion = 'Rapid Movement';
  render();
  UI_TESTS.assertEqual(
    document.getElementById('motion-pill').textContent,
    'Rapid Movement',
    'Shows live motion classification when monitoring'
  );
  
  // Test non-monitoring state
  state.monitoring = false;
  state.motion = 'Still';
  render();
  UI_TESTS.assertEqual(
    document.getElementById('motion-pill').textContent,
    'Still',
    'Shows Still when not monitoring'
  );
  
  console.log('✓ Motion pill display updates correctly (web app shows Detecting/Still only)');
}

// Test 9: Threat level title updates
function testThreatTitleUpdates() {
  console.log('\n--- Test: Threat Title Updates ---');
  
  state.panicActive = false;
  
  const threatTests = [
    { level: 20, expected: 'Environment appears safe' },
    { level: 40, expected: 'Mild distress signals detected' },
    { level: 60, expected: 'Elevated threat level' },
    { level: 80, expected: 'High threat level' },
    { level: 95, expected: 'High threat level' }
  ];
  
  for (const { level, expected } of threatTests) {
    state.threatLevel = level;
    render();
    UI_TESTS.assertEqual(
      document.getElementById('threat-title').textContent,
      expected,
      `Threat title at ${level}%`
    );
  }
  
  // Test panic state
  state.panicActive = true;
  render();
  UI_TESTS.assertEqual(
    document.getElementById('threat-title').textContent,
    'Emergency alert triggered',
    'Panic state shows emergency title'
  );
  
  console.log('✓ Threat title updates correctly');
}

// Test 10: Status message updates
function testStatusMessages() {
  console.log('\n--- Test: Status Messages ---');
  
  state.panicActive = false;
  
  // Test object detection message
  state.threatLevel = 90;
  state.statusMessage = '⚠️ High-confidence knife detected';
  render();
  UI_TESTS.assertEqual(
    document.getElementById('threat-message').textContent,
    '⚠️ High-confidence knife detected',
    'Shows knife detection message'
  );
  
  // Test safe message
  state.threatLevel = 20;
  state.statusMessage = '✅ Environment appears safe';
  render();
  UI_TESTS.assertEqual(
    document.getElementById('threat-message').textContent,
    '✅ Environment appears safe',
    'Shows safe message'
  );
  
  // Test panic message
  state.panicActive = true;
  render();
  UI_TESTS.assertContains(
    document.getElementById('threat-message').textContent,
    'panic sequence is active',
    'Shows panic message'
  );
  
  console.log('✓ Status messages update correctly');
}

// Test 11: Panic button countdown
function testPanicButton() {
  console.log('\n--- Test: Panic Button ---');
  
  state.panicActive = false;
  state.panicCountdown = 0;
  render();
  UI_TESTS.assertEqual(
    document.getElementById('panic-btn').textContent,
    'Trigger SOS',
    'Shows Trigger SOS when inactive'
  );
  
  state.panicActive = true;
  state.panicCountdown = 5;
  render();
  UI_TESTS.assertEqual(
    document.getElementById('panic-btn').textContent,
    'Cancel SOS (5s)',
    'Shows countdown when active'
  );
  
  console.log('✓ Panic button updates correctly');
}

// Test 12: Monitoring toggle button
function testMonitoringToggle() {
  console.log('\n--- Test: Monitoring Toggle Button ---');
  
  state.monitoring = false;
  render();
  UI_TESTS.assertEqual(
    document.getElementById('monitor-toggle').textContent,
    'Start monitoring',
    'Shows Start monitoring when inactive'
  );
  
  state.monitoring = true;
  render();
  UI_TESTS.assertEqual(
    document.getElementById('monitor-toggle').textContent,
    'Stop monitoring',
    'Shows Stop monitoring when active'
  );
  
  console.log('✓ Monitoring toggle button updates correctly');
}

// Run all UI tests
function runAllUITests() {
  console.log('╔════════════════════════════════════════╗');
  console.log('║     Web UI Integration Tests          ║');
  console.log('╚════════════════════════════════════════╝');
  
  try {
    testUIElementsExist();
    testInitialState();
    testMonitoringActiveState();
    testThreatLevelGauge();
    testEmotionPillColors();
    testCameraStatusMessages();
    testObjectPillDisplay();
    testMotionPillDisplay();
    testThreatTitleUpdates();
    testStatusMessages();
    testPanicButton();
    testMonitoringToggle();
    
    return UI_TESTS.summary();
  } catch (e) {
    UI_TESTS.failed++;
    console.error('\n❌ Test failed:', e.message);
    return UI_TESTS.summary();
  }
}

// Auto-run if in browser
if (typeof window !== 'undefined' && document.getElementById('emotion-pill')) {
  runAllUITests();
}

// Export for manual running
window.runUITests = runAllUITests;
window.UITests = UI_TESTS;
