package com.example.saftymonitoringsystem.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * TensorFlow Lite-based emotion classifier.
 *
 * Loads the trained `emotion_classifier.tflite` model from assets and runs
 * inference on a cropped face Bitmap (224×224 RGB).
 *
 * Emotion classes (8): Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised
 * These are mapped to the app's 7-emotion schema used by ThreatEngine.
 *
 * The model was trained on the bundled facial emotion dataset (19 subjects × 8 emotions).
 */
class TFLiteEmotionClassifier(context: Context) {

    companion object {
        private const val TAG = "TFLiteEmotionClassifier"
        private const val MODEL_NAME = "emotion_classifier.tflite"
        private const val INPUT_SIZE = 224
        private const val NUM_CLASSES = 8

        // Model output labels (order matches training)
        val MODEL_LABELS = listOf(
            "Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"
        )

        // Mapping to app's 7-emotion schema used by ThreatEngine
        val APP_EMOTION_MAP = mapOf(
            "Anger" to "Angry",
            "Contempt" to "Disgust",
            "Disgust" to "Disgust",
            "Fear" to "Fear",
            "Happy" to "Happy",
            "Neutral" to "Neutral",
            "Sad" to "Sad",
            "Surprised" to "Surprise"
        )
    }

    private var interpreter: Interpreter? = null
    private var isModelLoaded = false

    init {
        try {
            val descriptor = context.assets.openFd(MODEL_NAME)
            val modelBuffer = descriptor.createInputStream().channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }.also { descriptor.close() }
            interpreter = Interpreter(modelBuffer)
            isModelLoaded = true
            Log.i(TAG, "TFLite emotion model loaded successfully from $MODEL_NAME")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load TFLite emotion model: ${e.message}")
            Log.i(TAG, "Falling back to heuristic emotion analysis")
            isModelLoaded = false
        }
    }

    /**
     * Classify a face Bitmap and return the primary emotion, confidence, and breakdown.
     *
     * @param faceBitmap Cropped face region (will be resized to 224×224)
     * @return Triple of (emotionLabel, confidence, breakdownMap) or null if model not loaded
     */
    fun classify(faceBitmap: Bitmap): Triple<String, Float, Map<String, Float>>? {
        if (!isModelLoaded || interpreter == null) return null

        val resized = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Preprocess: normalize to [0, 1] and convert to float array
        val input = preprocess(resized)

        // Run inference
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        interpreter?.run(input, output)

        // Apply softmax for probability distribution.
        // The v2 model emits raw LOGITS, so this is the single softmax in the chain.
        // (The v1 model already ended in a softmax layer, so this line was applying a
        // second one and capping the reported confidence at roughly 0.14.)
        val probs = softmax(output[0])

        // Find the top emotion
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0

        // Map the raw model label into the app's 7-emotion schema before returning it.
        // ThreatEngine matches on "Angry" / "Surprise", but MODEL_LABELS emits
        // "Anger" / "Surprised". Returning the raw label meant those two never matched
        // DISTRESS_EMOTIONS and were scored at the lowest risk weight (x0.1).
        val primaryLabel = APP_EMOTION_MAP[MODEL_LABELS[maxIdx]] ?: MODEL_LABELS[maxIdx]
        val confidence = probs[maxIdx]

        // Build breakdown map (mapped to app emotion labels).
        // Most mappings are 1:1, but Contempt and Disgust both map to "Disgust", so the
        // colliding probabilities are summed instead of one silently overwriting the other.
        val breakdown = MODEL_LABELS.mapIndexed { idx, label ->
            (APP_EMOTION_MAP[label] ?: label) to probs[idx]
        }.groupBy({ it.first }, { it.second })
            .mapValues { (_, values) -> values.sum() }

        return Triple(primaryLabel, confidence, breakdown)
    }

    /**
     * Preprocess a Bitmap into a normalized float array for TFLite input.
     */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixelValue in pixels) {
            // Extract R, G, B and normalize to [0, 1]
            val r = (pixelValue shr 16 and 0xFF) / 255.0f
            val g = (pixelValue shr 8 and 0xFF) / 255.0f
            val b = (pixelValue and 0xFF) / 255.0f
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }
        return byteBuffer
    }

    /**
     * Apply softmax normalization to raw model outputs.
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sum = expValues.sum().coerceAtLeast(1e-6f)
        return expValues.map { it / sum }.toFloatArray()
    }

    /**
     * Crop a face region from a full-frame Bitmap using ML Kit's bounding box.
     */
    fun cropFace(fullFrame: Bitmap, left: Int, top: Int, right: Int, bottom: Int): Bitmap {
        val safeLeft = left.coerceIn(0, fullFrame.width - 1)
        val safeTop = top.coerceIn(0, fullFrame.height - 1)
        val safeRight = right.coerceIn(safeLeft + 1, fullFrame.width)
        val safeBottom = bottom.coerceIn(safeTop + 1, fullFrame.height)

        val width = safeRight - safeLeft
        val height = safeBottom - safeTop

        return if (width > 0 && height > 0) {
            Bitmap.createBitmap(fullFrame, safeLeft, safeTop, width, height)
        } else {
            // Fallback: return center crop
            val size = minOf(fullFrame.width, fullFrame.height) / 2
            val x = (fullFrame.width - size) / 2
            val y = (fullFrame.height - size) / 2
            Bitmap.createBitmap(fullFrame, x, y, size, size)
        }
    }

    fun isModelAvailable(): Boolean = isModelLoaded

    fun close() {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
    }
}
