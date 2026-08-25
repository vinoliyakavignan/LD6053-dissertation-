package com.example.saftymonitoringsystem.data

import com.example.saftymonitoringsystem.data.model.SafetyIncident
import com.example.saftymonitoringsystem.data.model.UserSettings

/**
 * Severity classification for safety alerts.
 */
enum class AlertSeverity {
    INFO,
    WARNING,
    CRITICAL
}

/**
 * Describes how an alert should be dispatched.
 */
data class AlertDispatchPlan(
    val severity: AlertSeverity,
    val shouldSendPush: Boolean,
    val shouldSendSms: Boolean,
    val shouldSendEmail: Boolean,
    val escalationDelayMs: Long,
    val title: String,
    val body: String
)

/**
 * Evaluates a detected incident and decides which channels should be used.
 * The logic is intentionally additive so the existing threshold-based trigger remains intact.
 */
object AlertSeverityEvaluator {

    private val criticalKeywords = setOf("knife", "gun", "weapon", "fire", "blood", "attack")
    private val warningKeywords = setOf("fear", "angry", "panic", "fall", "injury", "danger")

    fun evaluate(incident: SafetyIncident, settings: UserSettings): AlertDispatchPlan {
        val normalizedObjects = incident.detectedObjects.map { it.lowercase() }
        val emotion = incident.emotion.lowercase()

        val isCritical = incident.threatLevel >= 90 ||
            normalizedObjects.any { label -> criticalKeywords.any { keyword -> keyword in label } } ||
            emotion in setOf("fear", "angry", "panic")

        val isWarning = incident.threatLevel >= 70 ||
            normalizedObjects.any { label -> warningKeywords.any { keyword -> keyword in label } } ||
            emotion in setOf("sad", "disgust", "surprise")

        val severity = when {
            isCritical -> AlertSeverity.CRITICAL
            isWarning -> AlertSeverity.WARNING
            else -> AlertSeverity.INFO
        }

        val shouldSendSms = settings.notifySms && severity != AlertSeverity.INFO
        val shouldSendPush = settings.notifyPush
        val shouldSendEmail = settings.notifyEmail && severity == AlertSeverity.CRITICAL

        val escalationDelayMs = when (severity) {
            AlertSeverity.CRITICAL -> 0L
            AlertSeverity.WARNING -> 2_000L
            AlertSeverity.INFO -> 1_000L
        }

        val title = when (severity) {
            AlertSeverity.CRITICAL -> "Critical safety incident"
            AlertSeverity.WARNING -> "Elevated safety concern"
            AlertSeverity.INFO -> "Safety alert"
        }

        val body = "Threat ${incident.threatLevel}% • ${incident.emotion} • ${incident.motionActivity}"

        return AlertDispatchPlan(
            severity = severity,
            shouldSendPush = shouldSendPush,
            shouldSendSms = shouldSendSms,
            shouldSendEmail = shouldSendEmail,
            escalationDelayMs = escalationDelayMs,
            title = title,
            body = body
        )
    }
}
