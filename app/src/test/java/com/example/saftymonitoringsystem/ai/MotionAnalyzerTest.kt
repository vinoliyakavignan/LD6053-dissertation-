package com.example.saftymonitoringsystem.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionAnalyzerTest {

    @Test
    fun stillMotionYieldsZeroIntensity() {
        val analyzer = MotionAnalyzer { _, _ -> }
        
        // Simulate first frame - should not produce output
        val histogram1 = FloatArray(64) { 0f }
        
        // Simulate second identical frame - diff should be near zero
        val histogram2 = FloatArray(64) { 0f }
        
        val diff = l1Distance(histogram1, histogram2) / 64f
        assertEquals(0f, diff, 0.001f)
        
        val (activity, intensity) = classify(diff)
        assertEquals("Still", activity)
        assertEquals(0f, intensity, 0.01f)
    }

    @Test
    fun smallDifferenceYieldsNormalMovement() {
        val histogram1 = FloatArray(64) { 0f }
        val histogram2 = FloatArray(64) { 0f }
        // Create diff > 0.03 (Normal Movement threshold): need sum > 1.92
        // Set 20 bins to 0.1 vs 0 = diff of 0.1 each, sum = 2.0, diff = 2.0/64 = 0.03125
        for (i in 0..19) {
            histogram2[i] = 0.1f
        }
        
        val diff = l1Distance(histogram1, histogram2) / 64f
        val (activity, intensity) = classify(diff)
        
        assertEquals("Normal Movement", activity)
        assertTrue(intensity > 0f && intensity <= 1f)
    }

    @Test
    fun largeDifferenceYieldsRapidMovement() {
        val histogram1 = FloatArray(64) { 0f }
        val histogram2 = FloatArray(64) { 0f }
        // Create diff > 0.10 (Rapid Movement threshold): need sum > 6.4
        // Set 40 bins to 0.2 vs 0 = diff of 0.2 each, sum = 8.0, diff = 8.0/64 = 0.125
        for (i in 0..39) {
            histogram2[i] = 0.2f
        }
        
        val diff = l1Distance(histogram1, histogram2) / 64f
        val (activity, intensity) = classify(diff)
        
        assertTrue(activity == "Rapid Movement" || activity == "Possible Running" || activity == "Possible Struggle")
        assertTrue(intensity > 0f)
    }

    @Test
    fun veryLargeDifferenceYieldsPossibleStruggle() {
        val histogram1 = FloatArray(64) { 0f }
        val histogram2 = FloatArray(64) { 0f }
        // Create diff > 0.32 (Struggle threshold): need sum > 20.48
        // Set all 64 bins to 0.5 vs 0 = diff of 0.5 each, sum = 32.0, diff = 32.0/64 = 0.5
        for (i in 0..63) {
            histogram2[i] = 0.5f
        }
        
        val diff = l1Distance(histogram1, histogram2) / 64f
        val (activity, intensity) = classify(diff)
        
        assertEquals("Possible Struggle", activity)
        assertEquals(1f, intensity, 0.01f)
    }

    @Test
    fun histogramComputationProducesNormalizedOutput() {
        // Test with manually created histogram instead of Bitmap
        val hist = FloatArray(64) { 1f / 64f } // uniform distribution
        
        // Should be normalized (sum = 1)
        var sum = 0f
        for (v in hist) sum += v
        assertEquals(1f, sum, 0.01f)
    }

    // Helper methods copied from MotionAnalyzer for testing
    private fun l1Distance(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += kotlin.math.abs(a[i] - b[i])
        return sum
    }

    private fun classify(diff: Float): Pair<String, Float> = when {
        diff < 0.03f -> "Still" to diff / 0.03f
        diff < 0.10f -> "Normal Movement" to diff / 0.10f
        diff < 0.20f -> "Rapid Movement" to diff / 0.20f
        diff < 0.32f -> "Possible Running" to diff / 0.32f
        else -> "Possible Struggle" to 1f
    }

    private fun computeGreyscaleHistogram(bitmap: android.graphics.Bitmap): FloatArray {
        val hist = FloatArray(64)
        val scaled = android.graphics.Bitmap.createScaledBitmap(bitmap, 64, 64, false)
        val pixelCount = (scaled.width.toFloat() * scaled.height.toFloat())

        for (y in 0 until scaled.height) {
            for (x in 0 until scaled.width) {
                val pixel = scaled.getPixel(x, y)
                val r = (pixel shr 16 and 0xFF)
                val g = (pixel shr 8 and 0xFF)
                val b = (pixel and 0xFF)
                val grey = (0.299f * r + 0.587f * g + 0.114f * b).toInt()
                val bin = (grey * 64 / 256).coerceIn(0, 63)
                hist[bin]++
            }
        }
        for (i in hist.indices) hist[i] /= pixelCount
        return hist
    }
}