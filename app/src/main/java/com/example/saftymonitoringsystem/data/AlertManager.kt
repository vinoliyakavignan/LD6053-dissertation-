package com.example.saftymonitoringsystem.data

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.telephony.SmsManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.example.saftymonitoringsystem.data.model.SafetyIncident

/**
 * Manages sending emergency alerts via SMS and local push notifications.
 *
 * Future extension: add email via JavaMail / SendGrid API, or Firebase Cloud Messaging.
 */
class AlertManager(private val context: Context) {

    companion object {
        private const val TAG              = "AlertManager"
        private const val CHANNEL_ID       = "safety_alerts"
        private const val CHANNEL_NAME     = "Safety Alerts"
        private const val NOTIF_ID_BASE    = 1000
        private val VIBRATION_PATTERN     = longArrayOf(0, 300, 200, 300, 200, 300)
    }

    private val notifManager =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    private val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator

    init { createNotificationChannel() }

    // ── SMS Alert ─────────────────────────────────────────────────────────────

    /**
     * Sends a multi-part SMS to [phoneNumber] with [message].
     * SMS is split automatically if over 160 characters.
     */
    fun sendSmsAlert(phoneNumber: String, message: String) {
        if (phoneNumber.isBlank()) return
        try {
            @Suppress("Deprecation")
            val smsManager: SmsManager = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                context.getSystemService(SmsManager::class.java) ?: SmsManager.getDefault()
            } else {
                SmsManager.getDefault()
            }
            val parts = smsManager.divideMessage(message)
            if (parts.size == 1) {
                smsManager.sendTextMessage(phoneNumber, null, message, null, null)
            } else {
                smsManager.sendMultipartTextMessage(phoneNumber, null, parts, null, null)
            }
            Log.d(TAG, "SMS sent to $phoneNumber")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send SMS to $phoneNumber", e)
        }
    }

    // ── Push Notification ─────────────────────────────────────────────────────

    /**
     * Posts a high-priority local notification summarising the incident.
     */
    fun sendPushNotification(incident: SafetyIncident) {
        sendPushNotification(incident, "⚠️ Safety Alert — Threat: ${incident.threatLevel}%", buildNotificationBody(incident))
    }

    fun sendPushNotification(incident: SafetyIncident, title: String, body: String) {
        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(
                NotificationCompat.BigTextStyle()
                    .bigText(buildNotificationBody(incident, body))
            )
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setDefaults(NotificationCompat.DEFAULT_ALL)
            .setVibrate(VIBRATION_PATTERN)
            .setAutoCancel(true)
            .build()

        notifManager.notify(NOTIF_ID_BASE + incident.id.hashCode(), notification)
        vibrateAlert()
    }

    /**
     * Placeholder email hook for critical incident escalation.
     * Keeps the app functional even when no email provider is configured.
     */
    fun sendEmailAlert(incident: SafetyIncident, title: String, body: String) {
        try {
            Log.d(TAG, "Email escalation requested: $title | $body")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to dispatch email escalation", e)
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Builds the full SMS message body for an emergency alert.
     */
    fun buildSmsBody(incident: SafetyIncident): String {
        val sb = StringBuilder()
        sb.appendLine("🚨 SAFETY ALERT 🚨")
        sb.appendLine("Potential emotional distress detected.")
        sb.appendLine("Emotion : ${incident.emotion} (${(incident.emotionConf * 100).toInt()}%)")
        if (incident.detectedObjects.isNotEmpty())
            sb.appendLine("Objects : ${incident.detectedObjects.joinToString()}")
        sb.appendLine("Activity: ${incident.motionActivity}")
        sb.appendLine("Threat  : ${incident.threatLevel}%")
        if (incident.mapsUrl.isNotBlank())
            sb.appendLine("Location: ${incident.mapsUrl}")
        else if (incident.location.isNotBlank())
            sb.appendLine("Location: ${incident.location}")
        sb.appendLine("Time    : ${java.text.SimpleDateFormat("dd MMM yyyy HH:mm:ss",
            java.util.Locale.getDefault()).format(java.util.Date(incident.timestamp))}")
        return sb.toString().trim()
    }

    private fun buildNotificationBody(incident: SafetyIncident): String {
        return buildNotificationBody(incident, "Emotion: ${incident.emotion} | Objects: ${
            incident.detectedObjects.ifEmpty { listOf("None") }.joinToString()
        } | Motion: ${incident.motionActivity} | Threat: ${incident.threatLevel}%")
    }

    private fun buildNotificationBody(incident: SafetyIncident, fallbackBody: String): String {
        return if (fallbackBody.isBlank()) {
            "Emotion: ${incident.emotion} | Objects: ${
                incident.detectedObjects.ifEmpty { listOf("None") }.joinToString()
            } | Motion: ${incident.motionActivity} | Threat: ${incident.threatLevel}%"
        } else {
            fallbackBody
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, CHANNEL_NAME, NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Emergency safety alerts from AI monitoring"
                enableVibration(true)
                vibrationPattern = VIBRATION_PATTERN
                enableLights(true)
            }
            notifManager.createNotificationChannel(channel)
        }
    }

    private fun vibrateAlert() {
        vibrator?.let { v ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                v.vibrate(VibrationEffect.createWaveform(VIBRATION_PATTERN, -1))
            } else {
                @Suppress("DEPRECATION")
                v.vibrate(VIBRATION_PATTERN, -1)
            }
        }
    }
}
