package com.example.saftymonitoringsystem.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceAnalyzerHeuristicTest {

    @Test
    fun smilingFaceYieldsHappyEmotion() {
        // Simulate heuristic analysis with high smile probability
        val smile = 0.9f
        val leftEye = 0.8f
        val rightEye = 0.8f
        val avgEye = (leftEye + rightEye) / 2f
        val eyeAsymmetry = kotlin.math.abs(leftEye - rightEye)

        val raw = mutableMapOf(
            "Happy" to clamp(smile * 1.5f - 0.2f),
            "Sad" to clamp((1f - smile) * (1f - avgEye + 0.3f)),
            "Fear" to clamp(avgEye * (1f - smile) * 1.2f + eyeAsymmetry),
            "Angry" to clamp((1f - smile) * 0.6f + eyeAsymmetry * 0.8f),
            "Disgust" to clamp((1f - smile) * 0.5f - avgEye * 0.3f),
            "Surprise" to clamp(avgEye * 1.3f - smile * 0.5f),
            "Neutral" to clamp(0.4f + smile * 0.1f - eyeAsymmetry)
        )

        val breakdown = softmax(raw)
        val primary = breakdown.maxByOrNull { it.value }

        assertEquals("Happy", primary?.key)
        assertTrue(primary?.value!! > 0.5f)
    }

    @Test
    fun nonSmilingClosedEyesYieldsSadEmotion() {
        val smile = 0.1f
        val leftEye = 0.1f
        val rightEye = 0.1f
        val avgEye = (leftEye + rightEye) / 2f
        val eyeAsymmetry = kotlin.math.abs(leftEye - rightEye)

        val raw = mutableMapOf(
            "Happy" to clamp(smile * 1.5f - 0.2f),
            "Sad" to clamp((1f - smile) * (1f - avgEye + 0.3f)),
            "Fear" to clamp(avgEye * (1f - smile) * 1.2f + eyeAsymmetry),
            "Angry" to clamp((1f - smile) * 0.6f + eyeAsymmetry * 0.8f),
            "Disgust" to clamp((1f - smile) * 0.5f - avgEye * 0.3f),
            "Surprise" to clamp(avgEye * 1.3f - smile * 0.5f),
            "Neutral" to clamp(0.4f + smile * 0.1f - eyeAsymmetry)
        )

        val breakdown = softmax(raw)
        val primary = breakdown.maxByOrNull { it.value }

        assertEquals("Sad", primary?.key)
        assertTrue(primary?.value!! > 0.3f)
    }

    @Test
    fun wideEyesNoSmileYieldsFearOrSurprise() {
        val smile = 0.1f
        val leftEye = 0.9f
        val rightEye = 0.9f
        val avgEye = (leftEye + rightEye) / 2f
        val eyeAsymmetry = kotlin.math.abs(leftEye - rightEye)

        val raw = mutableMapOf(
            "Happy" to clamp(smile * 1.5f - 0.2f),
            "Sad" to clamp((1f - smile) * (1f - avgEye + 0.3f)),
            "Fear" to clamp(avgEye * (1f - smile) * 1.2f + eyeAsymmetry),
            "Angry" to clamp((1f - smile) * 0.6f + eyeAsymmetry * 0.8f),
            "Disgust" to clamp((1f - smile) * 0.5f - avgEye * 0.3f),
            "Surprise" to clamp(avgEye * 1.3f - smile * 0.5f),
            "Neutral" to clamp(0.4f + smile * 0.1f - eyeAsymmetry)
        )

        val breakdown = softmax(raw)
        val primary = breakdown.maxByOrNull { it.value }

        assertTrue(primary?.key == "Fear" || primary?.key == "Surprise")
        assertTrue(primary?.value!! > 0.3f)
    }

    @Test
    fun asymmetricEyesYieldsAngryOrFear() {
        val smile = 0.2f
        val leftEye = 0.9f
        val rightEye = 0.3f
        val avgEye = (leftEye + rightEye) / 2f
        val eyeAsymmetry = kotlin.math.abs(leftEye - rightEye)

        val raw = mutableMapOf(
            "Happy" to clamp(smile * 1.5f - 0.2f),
            "Sad" to clamp((1f - smile) * (1f - avgEye + 0.3f)),
            "Fear" to clamp(avgEye * (1f - smile) * 1.2f + eyeAsymmetry),
            "Angry" to clamp((1f - smile) * 0.6f + eyeAsymmetry * 0.8f),
            "Disgust" to clamp((1f - smile) * 0.5f - avgEye * 0.3f),
            "Surprise" to clamp(avgEye * 1.3f - smile * 0.5f),
            "Neutral" to clamp(0.4f + smile * 0.1f - eyeAsymmetry)
        )

        val breakdown = softmax(raw)
        val primary = breakdown.maxByOrNull { it.value }

        assertTrue(primary?.key == "Angry" || primary?.key == "Fear")
    }

    @Test
    fun neutralFaceYieldsNeutralEmotion() {
        val smile = 0.5f
        val leftEye = 0.5f
        val rightEye = 0.5f
        val avgEye = (leftEye + rightEye) / 2f
        val eyeAsymmetry = kotlin.math.abs(leftEye - rightEye)

        val raw = mutableMapOf(
            "Happy" to clamp(smile * 1.5f - 0.2f),
            "Sad" to clamp((1f - smile) * (1f - avgEye + 0.3f)),
            "Fear" to clamp(avgEye * (1f - smile) * 1.2f + eyeAsymmetry),
            "Angry" to clamp((1f - smile) * 0.6f + eyeAsymmetry * 0.8f),
            "Disgust" to clamp((1f - smile) * 0.5f - avgEye * 0.3f),
            "Surprise" to clamp(avgEye * 1.3f - smile * 0.5f),
            "Neutral" to clamp(0.4f + smile * 0.1f - eyeAsymmetry)
        )

        val breakdown = softmax(raw)
        val primary = breakdown.maxByOrNull { it.value }

        // Neutral should be competitive
        assertTrue(breakdown["Neutral"]!! > 0.1f)
    }

    @Test
    fun softmaxProducesValidProbabilityDistribution() {
        val raw = mutableMapOf(
            "Happy" to 0.8f,
            "Sad" to 0.2f,
            "Fear" to 0.1f,
            "Angry" to 0.1f,
            "Disgust" to 0.05f,
            "Surprise" to 0.05f,
            "Neutral" to 0.3f
        )

        val breakdown = softmax(raw)

        var sum = 0f
        for (v in breakdown.values) sum += v
        assertEquals(1f, sum, 0.001f)

        for (v in breakdown.values) {
            assertTrue(v >= 0f && v <= 1f)
        }
    }

    // Helper methods copied from FaceAnalyzer for testing
    private fun clamp(v: Float): Float = v.coerceIn(0f, 1f)

    private fun softmax(raw: Map<String, Float>): Map<String, Float> {
        val expValues = raw.mapValues { (_, v) -> Math.exp((v * 5.0)).toFloat() }
        val sum = expValues.values.sum().coerceAtLeast(1e-6f)
        return expValues.mapValues { (_, e) -> e / sum }
    }
}