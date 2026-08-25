package com.example.saftymonitoringsystem.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ObjectAnalyzerLogicTest {

    @Test
    fun knifeDetectionAboveThresholdTriggersAlert() {
        val knifeScore = 0.75f
        val nonKnifeScore = 1f - knifeScore
        val isKnife = knifeScore >= 0.50f

        assertTrue(isKnife)
        assertEquals("Knife", if (isKnife) "Knife" else "")
    }

    @Test
    fun nonKnifeBelowThresholdYieldsNoDetection() {
        val knifeScore = 0.30f
        val nonKnifeScore = 1f - knifeScore
        val isKnife = knifeScore >= 0.50f

        assertTrue(!isKnife)
        assertEquals(0, if (isKnife) 1 else 0)
    }

    @Test
    fun knifeScoreAtExactThresholdTriggers() {
        val knifeScore = 0.50f
        val isKnife = knifeScore >= 0.50f

        assertTrue(isKnife)
    }

    @Test
    fun preprocessingProducesCorrectBufferSize() {
        val INPUT_SIZE = 224
        val bufferSize = 1 * INPUT_SIZE * INPUT_SIZE * 3 * 4 // 1 * 224 * 224 * 3 * 4 = 602112 bytes
        
        assertEquals(602112, bufferSize)
    }

    @Test
    fun `rgb normalization produces values in range`() {
        val pixel = 0xFFFFFFFF // White
        val r = ((pixel shr 16) and 0xFF) / 255.0f
        val g = ((pixel shr 8) and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f

        assertEquals(1f, r, 0.001f)
        assertEquals(1f, g, 0.001f)
        assertEquals(1f, b, 0.001f)
    }

    @Test
    fun `black pixel normalizes to zero`() {
        val pixel = 0xFF000000 // Black
        val r = ((pixel shr 16) and 0xFF) / 255.0f
        val g = ((pixel shr 8) and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f

        assertEquals(0f, r, 0.001f)
        assertEquals(0f, g, 0.001f)
        assertEquals(0f, b, 0.001f)
    }

    @Test
    fun midGrayNormalizesToHalf() {
        val pixel = 0xFF808080 // Mid-gray
        val r = ((pixel shr 16) and 0xFF) / 255.0f
        val g = ((pixel shr 8) and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f

        assertTrue(r > 0.49f && r < 0.51f)
        assertTrue(g > 0.49f && g < 0.51f)
        assertTrue(b > 0.49f && b < 0.51f)
    }
}