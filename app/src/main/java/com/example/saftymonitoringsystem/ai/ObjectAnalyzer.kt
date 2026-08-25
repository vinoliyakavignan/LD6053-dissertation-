package com.example.saftymonitoringsystem.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Detects common COCO objects and preserves the safety-specific knife classifier.
 *
 * EfficientDet-Lite0 supplies localized classes such as person, bottle, cup,
 * chair, phone, and scissors. The separate binary classifier continues to
 * detect knives because knife is not one of COCO's standard 80 classes.
 */
class ObjectAnalyzer(
    context: Context,
    private val onObjectsDetected: (
        labels: List<String>,
        confidences: List<Float>
    ) -> Unit
) : ImageAnalysis.Analyzer {

    companion object {
        private const val TAG = "ObjectAnalyzer"
        private const val DETECTOR_MODEL_NAME = "efficientdet_lite0.tflite"
        private const val KNIFE_MODEL_NAME = "knife_classifier.tflite"
        private const val LABELS_NAME = "labels.txt"
        private const val INPUT_SIZE = 224
        private const val KNIFE_THRESHOLD = 0.50f
        private const val OBJECT_THRESHOLD = 0.30f
        private const val MAX_RESULTS = 5
    }

    private var interpreter: Interpreter? = null
    private var objectDetector: ObjectDetector? = null
    private var labels: List<String> = listOf("knife", "non_knife")
    private var isModelLoaded = false

    init {
        try {
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(DETECTOR_MODEL_NAME)
                        .build()
                )
                .setRunningMode(RunningMode.IMAGE)
                .setScoreThreshold(OBJECT_THRESHOLD)
                .setMaxResults(MAX_RESULTS)
                .build()
            objectDetector = ObjectDetector.createFromOptions(context, options)
            Log.i(TAG, "Loaded EfficientDet-Lite0 COCO object detector")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load general object detector", e)
        }

        try {
            val descriptor = context.assets.openFd(KNIFE_MODEL_NAME)
            val modelBuffer = descriptor.createInputStream().channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }.also { descriptor.close() }
            interpreter = Interpreter(modelBuffer)
            labels = context.assets.open(LABELS_NAME).bufferedReader().useLines { lines ->
                lines.map(String::trim).filter(String::isNotEmpty).toList()
            }
            isModelLoaded = true
            Log.i(TAG, "Loaded knife classifier model successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load knife classifier model", e)
            isModelLoaded = false
        }
    }

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        if (objectDetector == null && (!isModelLoaded || interpreter == null)) {
            onObjectsDetected(emptyList(), emptyList())
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toRotatedBitmap()
            val detections = mutableListOf<Pair<String, Float>>()

            objectDetector?.let { detector ->
                // MPImage.close() also recycles its Bitmap. Give MediaPipe its own
                // copy so the shared camera bitmap remains valid for the knife model.
                val mediaPipeBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false)
                val mpImage = BitmapImageBuilder(mediaPipeBitmap).build()
                try {
                    detector.detect(mpImage).detections().forEach { detection ->
                        detection.categories().maxByOrNull { it.score() }?.let { category ->
                            val label = category.displayName().takeIf { it.isNotBlank() }
                                ?: category.categoryName()
                            if (label.isNotBlank() && category.score() >= OBJECT_THRESHOLD) {
                                detections += label.replaceFirstChar { it.uppercase() } to category.score()
                            }
                        }
                    }
                } finally {
                    mpImage.close()
                }
            }

            interpreter?.let { knifeInterpreter ->
                val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
                try {
                    val output = Array(1) { FloatArray(1) }
                    knifeInterpreter.run(preprocess(resized), output)
                    val knifeScore = 1f - output[0][0].coerceIn(0f, 1f)
                    if (knifeScore >= KNIFE_THRESHOLD) detections += "Knife" to knifeScore
                } finally {
                    if (resized !== bitmap) resized.recycle()
                }
            }

            bitmap.recycle()
            val ranked = detections
                .groupBy { it.first.lowercase() }
                .map { (_, matches) -> matches.maxBy { it.second } }
                .sortedByDescending { it.second }
                .take(MAX_RESULTS)
            onObjectsDetected(ranked.map { it.first }, ranked.map { it.second })
        } catch (e: Exception) {
            Log.e(TAG, "Object detection failed", e)
            onObjectsDetected(emptyList(), emptyList())
        } finally {
            imageProxy.close()
        }
    }

    fun close() {
        objectDetector?.close()
        objectDetector = null
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val imageBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        imageBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            imageBuffer.putFloat(r)
            imageBuffer.putFloat(g)
            imageBuffer.putFloat(b)
        }
        imageBuffer.rewind()
        return imageBuffer
    }
}
