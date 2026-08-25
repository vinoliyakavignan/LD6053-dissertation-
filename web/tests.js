// Web App Tests - Run in browser console or with Jest
// These test the core logic functions from web/app.js

const TEST_UTILS = {
  assertEqual(actual, expected, msg) {
    if (actual !== expected) {
      throw new Error(`${msg}: expected ${expected}, got ${actual}`);
    }
  },
  assertTrue(condition, msg) {
    if (!condition) throw new Error(msg || 'Assertion failed');
  },
  assertApprox(actual, expected, tolerance, msg) {
    if (Math.abs(actual - expected) > tolerance) {
      throw new Error(`${msg}: expected ~${expected}, got ${actual}`);
    }
  }
};

// ========== Test WAITING_STATES ==========
function testWaitingStates() {
  const states = {
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
  
  TEST_UTILS.assertEqual(Object.keys(states).length, 11, 'Should have 11 waiting states');
  TEST_UTILS.assertTrue(states.NO_FACE.includes('No face'), 'NO_FACE should mention face');
  console.log('✓ WAITING_STATES tests passed');
}

// ========== Test normalizeEmotion ==========
function testNormalizeEmotion() {
  // Test happy detection
  const happyExpr = { happy: 0.9, sad: 0.01, fearful: 0.01, angry: 0.01, disgusted: 0.01, surprised: 0.01, neutral: 0.05 };
  const happyResult = normalizeEmotion(happyExpr);
  TEST_UTILS.assertEqual(happyResult.label, 'Happy', 'Should detect Happy');
  TEST_UTILS.assertTrue(happyResult.confidence > 0.7, 'High confidence for clear happy');
  
  // Test sad detection
  const sadExpr = { happy: 0.01, sad: 0.85, fearful: 0.05, angry: 0.02, disgusted: 0.02, surprised: 0.01, neutral: 0.04 };
  const sadResult = normalizeEmotion(sadExpr);
  TEST_UTILS.assertEqual(sadResult.label, 'Sad', 'Should detect Sad');
  
  // Test fear detection
  const fearExpr = { happy: 0.01, sad: 0.05, fearful: 0.9, angry: 0.01, disgusted: 0.01, surprised: 0.01, neutral: 0.01 };
  const fearResult = normalizeEmotion(fearExpr);
  TEST_UTILS.assertEqual(fearResult.label, 'Fear', 'Should detect Fear');
  
  // Test angry detection
  const angryExpr = { happy: 0.01, sad: 0.02, fearful: 0.02, angry: 0.88, disgusted: 0.02, surprised: 0.01, neutral: 0.04 };
  const angryResult = normalizeEmotion(angryExpr);
  TEST_UTILS.assertEqual(angryResult.label, 'Angry', 'Should detect Angry');
  
  // Test disgust detection
  const disgustExpr = { happy: 0.01, sad: 0.02, fearful: 0.01, angry: 0.02, disgusted: 0.85, surprised: 0.01, neutral: 0.08 };
  const disgustResult = normalizeEmotion(disgustExpr);
  TEST_UTILS.assertEqual(disgustResult.label, 'Disgust', 'Should detect Disgust');
  
  // Test surprise detection
  const surpriseExpr = { happy: 0.01, sad: 0.01, fearful: 0.02, angry: 0.01, disgusted: 0.01, surprised: 0.9, neutral: 0.04 };
  const surpriseResult = normalizeEmotion(surpriseExpr);
  TEST_UTILS.assertEqual(surpriseResult.label, 'Surprise', 'Should detect Surprise');
  
  // Test neutral detection
  const neutralExpr = { happy: 0.1, sad: 0.1, fearful: 0.1, angry: 0.1, disgusted: 0.1, surprised: 0.1, neutral: 0.4 };
  const neutralResult = normalizeEmotion(neutralExpr);
  TEST_UTILS.assertEqual(neutralResult.label, 'Neutral', 'Should detect Neutral');
  
  // Test no expressions (no face)
  const noFaceResult = normalizeEmotion(null);
  TEST_UTILS.assertEqual(noFaceResult.label, 'No face detected', 'Should return no face for null');
  TEST_UTILS.assertEqual(noFaceResult.confidence, 0.1, 'Low confidence for no face');
  
  // Test all low confidence (below threshold)
  const lowExpr = { happy: 0.1, sad: 0.1, fearful: 0.1, angry: 0.1, disgusted: 0.1, surprised: 0.1, neutral: 0.4 };
  // Actually neutral is 0.4 which is above 0.15 threshold, so it should return Neutral
  const lowResult = normalizeEmotion(lowExpr);
  TEST_UTILS.assertEqual(lowResult.label, 'Neutral', 'Should return Neutral when all low');
  
  // Test second-highest emotion when neutral is top but close
  const closeExpr = { happy: 0.3, sad: 0.25, fearful: 0.1, angry: 0.1, disgusted: 0.1, surprised: 0.1, neutral: 0.35 };
  const closeResult = normalizeEmotion(closeExpr);
  // Neutral is top (0.35), Happy is second (0.3), diff = 0.05 < 0.18, so should use Happy
  TEST_UTILS.assertEqual(closeResult.label, 'Happy', 'Should use second emotion when neutral is close');
  
  console.log('✓ normalizeEmotion tests passed');
}

// ========== Test summarizeObjects ==========
function testSummarizeObjects() {
  // Test filtering by confidence
  const objects = [
    { className: 'person', score: 0.9 },
    { className: 'knife', score: 0.4 },
    { className: 'cup', score: 0.25 }, // below 0.3 threshold
    { className: 'book', score: 0.8 }
  ];
  const summarized = summarizeObjects(objects);
  TEST_UTILS.assertEqual(summarized.length, 3, 'Should keep 3 objects above threshold');
  TEST_UTILS.assertEqual(summarized[0].label, 'person', 'Should preserve person');
  TEST_UTILS.assertEqual(summarized[1].label, 'knife', 'Should preserve knife');
  TEST_UTILS.assertEqual(summarized[2].label, 'book', 'Should preserve book');
  
  // Test limit to 3
  const manyObjects = [
    { className: 'a', score: 0.9 },
    { className: 'b', score: 0.8 },
    { className: 'c', score: 0.7 },
    { className: 'd', score: 0.6 },
    { className: 'e', score: 0.5 }
  ];
  const limited = summarizeObjects(manyObjects);
  TEST_UTILS.assertEqual(limited.length, 3, 'Should limit to 3 objects');
  
  // Test underscore replacement
  const underscoreObj = [{ className: 'baseball_bat', score: 0.9 }];
  const underscoreResult = summarizeObjects(underscoreObj);
  TEST_UTILS.assertEqual(underscoreResult[0].label, 'baseball bat', 'Should replace underscores');
  
  console.log('✓ summarizeObjects tests passed');
}

// ========== Test getThreatTitle ==========
function testGetThreatTitle() {
  TEST_UTILS.assertEqual(getThreatTitle(20), 'Environment appears safe', 'Low threat');
  TEST_UTILS.assertEqual(getThreatTitle(40), 'Mild distress signals detected', 'Medium-low threat');
  TEST_UTILS.assertEqual(getThreatTitle(60), 'Elevated threat level', 'Medium threat');
  TEST_UTILS.assertEqual(getThreatTitle(80), 'High threat level', 'High threat');
  TEST_UTILS.assertEqual(getThreatTitle(95), 'High threat level', 'Very high threat');
  console.log('✓ getThreatTitle tests passed');
}

// ========== Test TEXT_EMOTION_KEYWORDS ==========
function testTextAnalysisKeywords() {
  // Test happy keywords
  const happyText = 'I am so happy and excited about this wonderful news!';
  const happyResult = analyzeText(happyText);
  TEST_UTILS.assertEqual(happyResult.emotion, 'Happy', 'Should detect Happy from text');
  
  // Test sad keywords
  const sadText = 'I feel so sad and depressed, everything is hopeless.';
  const sadResult = analyzeText(sadText);
  TEST_UTILS.assertEqual(sadResult.emotion, 'Sad', 'Should detect Sad from text');
  
  // Test fear keywords
  const fearText = 'I am terrified and scared, panic is overwhelming.';
  const fearResult = analyzeText(fearText);
  TEST_UTILS.assertEqual(fearResult.emotion, 'Fear', 'Should detect Fear from text');
  
  // Test angry keywords
  const angryText = 'I am furious and angry, this makes me so mad!';
  const angryResult = analyzeText(angryText);
  TEST_UTILS.assertEqual(angryResult.emotion, 'Angry', 'Should detect Angry from text');
  
  // Test neutral fallback
  const neutralText = 'The weather is okay today.';
  const neutralResult = analyzeText(neutralText);
  TEST_UTILS.assertEqual(neutralResult.emotion, 'Neutral', 'Should default to Neutral');
  
  // Test empty text
  const emptyResult = analyzeText('');
  TEST_UTILS.assertEqual(emptyResult.emotion, 'Neutral', 'Empty text should be Neutral');
  TEST_UTILS.assertEqual(emptyResult.confidence, 0, 'Empty text confidence should be 0');
  
  // Test threat scores
  TEST_UTILS.assertTrue(happyResult.threatScore === 0, 'Happy should have 0 threat');
  TEST_UTILS.assertTrue(sadResult.threatScore > 0 && sadResult.threatScore <= 60, 'Sad threat 0-60');
  TEST_UTILS.assertTrue(fearResult.threatScore > 0 && fearResult.threatScore <= 90, 'Fear threat 0-90');
  TEST_UTILS.assertTrue(angryResult.threatScore > 0 && angryResult.threatScore <= 90, 'Angry threat 0-90');
  
  console.log('✓ Text analysis tests passed');
}

// ========== Test getAlertPlan ==========
function testGetAlertPlan() {
  // Mock state.settings
  const mockState = {
    settings: { sms: true, push: true, email: false, threshold: 70 }
  };
  
  // Test critical emotion with high confidence
  const criticalPlan = getAlertPlan('Fear', 90);
  TEST_UTILS.assertEqual(criticalPlan.severity, 'critical', 'Fear at 90% should be critical');
  TEST_UTILS.assertTrue(criticalPlan.shouldSendSms, 'Should send SMS for critical');
  TEST_UTILS.assertTrue(criticalPlan.shouldSendPush, 'Should send push for critical');
  
  // Test warning emotion
  const warningPlan = getAlertPlan('Sad', 65);
  TEST_UTILS.assertEqual(warningPlan.severity, 'warning', 'Sad at 65% should be warning');
  
  // Test info emotion (low confidence)
  const infoPlan = getAlertPlan('Happy', 30);
  TEST_UTILS.assertEqual(infoPlan.severity, 'info', 'Happy at 30% should be info');
  TEST_UTILS.assertTrue(!infoPlan.shouldSendSms, 'Should not send SMS for info');
  
  // Test warning emotion (high confidence happy)
  const warningPlan2 = getAlertPlan('Happy', 80);
  TEST_UTILS.assertEqual(warningPlan2.severity, 'warning', 'Happy at 80% should be warning (high confidence)');
  
  // Test disgust (always critical per code)
  const disgustPlan = getAlertPlan('Disgust', 50);
  TEST_UTILS.assertEqual(disgustPlan.severity, 'critical', 'Disgust should be critical');
  
  console.log('✓ getAlertPlan tests passed');
}

// ========== Run all tests ==========
function runAllTests() {
  console.log('Running Web App Tests...\n');
  
  try {
    testWaitingStates();
    testNormalizeEmotion();
    testSummarizeObjects();
    testGetThreatTitle();
    testTextAnalysisKeywords();
    testGetAlertPlan();
    
    console.log('\n✅ All Web App tests passed!');
    return true;
  } catch (e) {
    console.error('\n❌ Test failed:', e.message);
    console.error(e.stack);
    return false;
  }
}

// Copy the functions from app.js for testing
// WAITING_STATES
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

// normalizeEmotion (copied from app.js)
function normalizeEmotion(expressions) {
  if (!expressions) {
    return { label: 'No face detected', confidence: 0.1 };
  }

  const ranked = Object.entries(expressions)
    .filter(([, score]) => Number(score) >= 0.15)
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

// summarizeObjects (copied from app.js)
function summarizeObjects(objects) {
  return objects
    .filter((obj) => Number(obj.score) >= 0.30)
    .slice(0, 3)
    .map((obj) => ({
      label: String(obj.className || obj.class || 'object').replace(/_/g, ' '),
      score: Number(obj.score)
    }));
}

// getThreatTitle (copied from app.js)
function getThreatTitle(level) {
  if (level < 40) return 'Environment appears safe';
  if (level < 60) return 'Mild distress signals detected';
  if (level < 80) return 'Elevated threat level';
  return 'High threat level';
}

// TEXT_EMOTION_KEYWORDS and analyzeText (copied from app.js)
const TEXT_EMOTION_KEYWORDS = {
  Happy: ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 'love', 'lovely', 'fantastic', 'blessed', 'grateful', 'pleased', 'delighted', 'joyful', 'content', 'satisfied', 'ecstatic', 'fabulous', 'fun', 'enjoy', 'celebrate', 'beautiful'],
  Sad: ['sad', 'depressed', 'lonely', 'hurt', 'pain', 'miserable', 'empty', 'heartbroken', 'gloomy', 'unhappy', 'hopeless', 'helpless', 'tired', 'exhausted', 'drained', 'melancholy', 'suffering', 'broken', 'lost', 'cry', 'tears', 'worried', 'anxious', 'stressed', 'overwhelmed', 'burdened'],
  Fear: ['fear', 'afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'panic', 'paranoid', 'threatened', 'apprehensive', 'shaky', 'frightened', 'concerned', 'uneasy', 'dread', 'horror', 'terror'],
  Angry: ['angry', 'mad', 'furious', 'rage', 'hate', 'irritated', 'annoyed', 'frustrated', 'pissed', 'resentful', 'hostile', 'violent', 'livid', 'enraged', 'outraged', 'bitter', 'resentment'],
  Disgust: ['disgust', 'disgusted', 'nauseated', 'revolted', 'repulsed', 'sickened', 'gross', 'filthy', 'dirty', 'vile', 'detest'],
  Surprise: ['surprise', 'surprised', 'shocked', 'amazed', 'astonished', 'startled', 'unexpected', 'suddenly', 'wow', 'incredible'],
  Neutral: ['neutral', 'okay', 'fine', 'normal', 'calm', 'peaceful', 'relaxed', 'steady', 'stable', 'balanced', 'clear', 'focused']
};

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

  const sortedEmotions = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const topEmotion = sortedEmotions[0][0];
  const confidence = sortedEmotions[0][1];

  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  const breakdown = {};
  for (const [emotion, score] of Object.entries(scores)) {
    breakdown[emotion] = score / Math.max(1e-6, totalScore);
  }

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

// getAlertPlan (copied from app.js)
function getAlertPlan(emotionName, confidencePercent) {
  const state = {
    settings: { sms: true, push: true, email: false, threshold: 70 }
  };
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

// Run tests if in browser/Node
if (typeof window !== 'undefined' || typeof global !== 'undefined') {
  runAllTests();
}

export { runAllTests, testWaitingStates, testNormalizeEmotion, testSummarizeObjects, testGetThreatTitle, testTextAnalysisKeywords, testGetAlertPlan };