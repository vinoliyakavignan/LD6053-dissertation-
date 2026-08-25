package com.example.saftymonitoringsystem.ai

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

/**
 * Extension function to convert an [ImageProxy] to a [Bitmap].
 *
 * CameraX ImageAnalysis delivers frames in YUV_420_888 format.
 * This function converts the YUV data to NV21 byte array and then
 * to a Bitmap suitable for downstream processing (motion analysis,
 * face/emotion classification, and knife classification).
 *
 * The resulting bitmap is rotated according to the image's rotation
 * degrees so it matches the display orientation.
 *
 * NAMING: this was previously called `toBitmap()`, which collided with the
 * member function `ImageProxy.toBitmap()` added in camera-core 1.3.0. In Kotlin a
 * member always wins over an extension, so every call site silently resolved to
 * CameraX's version — which does NOT apply rotation. That broke FaceAnalyzer,
 * because ML Kit reports `Face.boundingBox` in the ROTATED coordinate space while
 * the bitmap being cropped was still in sensor orientation. Renamed so the
 * rotation-applying implementation is actually reachable.
 */
fun ImageProxy.toRotatedBitmap(): Bitmap {
    // CameraX handles YUV row/pixel strides correctly. The previous manual NV21
    // concatenation assumed tightly packed planes and produced corrupt frames on
    // many devices whose camera buffers contain padding.
    val bitmap = toBitmap()

    // Rotate according to the image's rotation degrees
    val rotationDegrees = this.imageInfo.rotationDegrees
    return if (rotationDegrees != 0) {
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true).also {
            if (it !== bitmap) bitmap.recycle()
        }
    } else {
        bitmap
    }
}
