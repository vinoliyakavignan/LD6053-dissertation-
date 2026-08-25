package com.example.saftymonitoringsystem.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ThreatEngineTest {

    @Test
    fun safeEnvironmentYieldsLowThreatScore() {
        val score = ThreatEngine.calculate(
            emotion = "Neutral",
            emotionConfidence = 0.9f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Still",
            motionIntensity = 0.1f,
            hourOfDay = 12
        )
        // Neutral * 0.1 * 0.35 * 100 = 3.15 -> 3
        assertEquals(3, score)
    }

    @Test
    fun happyEmotionWithNoThreatsYieldsLowScore() {
        val score = ThreatEngine.calculate(
            emotion = "Happy",
            emotionConfidence = 0.8f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Still",
            motionIntensity = 0.1f,
            hourOfDay = 12
        )
        assertTrue(score < 20)
    }

    @Test
    fun fearEmotionIncreasesThreatScore() {
        val score = ThreatEngine.calculate(
            emotion = "Fear",
            emotionConfidence = 0.85f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Still",
            motionIntensity = 0.1f
        )
        assertTrue(score > 25)
    }

    @Test
    fun criticalObjectKnifeTriggersHighThreat() {
        val score = ThreatEngine.calculate(
            emotion = "Neutral",
            emotionConfidence = 0.5f,
            detectedObjects = listOf("knife"),
            objectConfidences = listOf(0.9f),
            motionActivity = "Still",
            motionIntensity = 0.1f
        )
        assertTrue(score >= 85)
    }

    @Test
    fun fearPlusKnifeYieldsCriticalThreat() {
        val score = ThreatEngine.calculate(
            emotion = "Fear",
            emotionConfidence = 0.8f,
            detectedObjects = listOf("knife"),
            objectConfidences = listOf(0.9f),
            motionActivity = "Still",
            motionIntensity = 0.1f
        )
        assertEquals(100, score)
    }

    @Test
    fun possibleStruggleMotionIncreasesThreat() {
        val score = ThreatEngine.calculate(
            emotion = "Neutral",
            emotionConfidence = 0.5f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Possible Struggle",
            motionIntensity = 0.8f
        )
        // Neutral(0.05) * 0.35 + Possible Struggle(0.8) * 0.20 = 0.0175 + 0.16 = 0.1775 * 100 = 17.75 -> 17
        assertTrue(score > 15)
    }

    @Test
    fun distressEmotionPlusStruggleYieldsBonus() {
        val score = ThreatEngine.calculate(
            emotion = "Fear",
            emotionConfidence = 0.7f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Possible Struggle",
            motionIntensity = 0.7f
        )
        assertTrue(score > 40)
    }

    @Test
    fun highObjectConfidenceKnifeAddsBonus() {
        val score = ThreatEngine.calculate(
            emotion = "Neutral",
            emotionConfidence = 0.5f,
            detectedObjects = listOf("knife"),
            objectConfidences = listOf(0.95f),
            motionActivity = "Still",
            motionIntensity = 0.1f,
            hourOfDay = 12
        )
        // Base: 0.35*0.05 + 0.45*0.95 + 0.20*0.01 = 0.0175 + 0.4275 + 0.002 = 0.447 * 100 = 44.7
        // + 45 bonus for critical object > 0.6 = 89.7 -> 89
        assertEquals(89, score)
    }

    @Test
    fun surpriseEmotionYieldsModerateScore() {
        val score = ThreatEngine.calculate(
            emotion = "Surprise",
            emotionConfidence = 0.9f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Still",
            motionIntensity = 0.1f
        )
        assertTrue(score > 0 && score < 30)
    }

    @Test
    fun rapidMovementYieldsModerateThreat() {
        val score = ThreatEngine.calculate(
            emotion = "Neutral",
            emotionConfidence = 0.5f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Rapid Movement",
            motionIntensity = 0.6f
        )
        assertTrue(score > 5 && score < 30)
    }

    @Test
    fun normalMovementYieldsLowThreat() {
        val score = ThreatEngine.calculate(
            emotion = "Neutral",
            emotionConfidence = 0.5f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Normal Movement",
            motionIntensity = 0.3f
        )
        assertTrue(score < 15)
    }

    @Test
    fun angerEmotionContributesToThreat() {
        val score = ThreatEngine.calculate(
            emotion = "Angry",
            emotionConfidence = 0.8f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Still",
            motionIntensity = 0.1f
        )
        assertTrue(score > 20)
    }

    @Test
    fun disgustEmotionContributesToThreat() {
        val score = ThreatEngine.calculate(
            emotion = "Disgust",
            emotionConfidence = 0.7f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Still",
            motionIntensity = 0.1f
        )
        assertTrue(score > 15)
    }

    @Test
    fun sadEmotionContributesToThreat() {
        val score = ThreatEngine.calculate(
            emotion = "Sad",
            emotionConfidence = 0.75f,
            detectedObjects = emptyList(),
            objectConfidences = emptyList(),
            motionActivity = "Still",
            motionIntensity = 0.1f
        )
        assertTrue(score > 15)
    }
}
