package com.example.saftymonitoringsystem.ai

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

@RunWith(AndroidJUnit4::class)
class KnifeModelLatencyTest {
    @Test
    fun benchmarkDeployedKnifeModel() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val descriptor = context.assets.openFd("knife_classifier.tflite")
        val model = descriptor.createInputStream().channel.use { channel ->
            channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
        }.also { descriptor.close() }

        Interpreter(model, Interpreter.Options().apply { setNumThreads(1) }).use { interpreter ->
            val input = ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder())
            repeat(224 * 224 * 3) { input.putFloat(0.5f) }
            input.rewind()
            val output = Array(1) { FloatArray(1) }

            repeat(20) {
                input.rewind()
                interpreter.run(input, output)
            }

            val samples = DoubleArray(200)
            repeat(samples.size) { index ->
                input.rewind()
                val started = System.nanoTime()
                interpreter.run(input, output)
                samples[index] = (System.nanoTime() - started) / 1_000_000.0
            }
            samples.sort()
            val mean = samples.average()
            val median = percentile(samples, 50.0)
            val p90 = percentile(samples, 90.0)
            val p95 = percentile(samples, 95.0)
            Log.i("KnifeLatency", "runs=200 threads=1 mean_ms=$mean median_ms=$median p90_ms=$p90 p95_ms=$p95 min_ms=${samples.first()} max_ms=${samples.last()}")
        }
    }

    private fun percentile(sorted: DoubleArray, percentile: Double): Double {
        val position = (sorted.size - 1) * percentile / 100.0
        val lower = position.toInt()
        val upper = kotlin.math.ceil(position).toInt()
        if (lower == upper) return sorted[lower]
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
    }
}
