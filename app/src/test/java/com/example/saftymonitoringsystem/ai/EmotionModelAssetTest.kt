package com.example.saftymonitoringsystem.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class EmotionModelAssetTest {
    @Test
    fun `defaults to the bundled model asset name`() {
        assertEquals("emotion_classifier.tflite", EmotionModelAsset.inferAssetName(null))
    }

    @Test
    fun `normalizes prefixed asset paths`() {
        assertEquals("model.pth", EmotionModelAsset.inferAssetName("/assets/model.pth"))
    }
}
