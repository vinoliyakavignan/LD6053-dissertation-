package com.example.saftymonitoringsystem.data

import com.example.saftymonitoringsystem.data.model.SafetyIncident
import com.example.saftymonitoringsystem.data.model.UserSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AlertSeverityEvaluatorTest {

    @Test
    fun criticalSeverityIsAssignedForHighThreatIncidents() {
        val incident = SafetyIncident(
            emotion = "Fear",
            threatLevel = 95,
            detectedObjects = listOf("Knife")
        )

        val dispatch = AlertSeverityEvaluator.evaluate(
            incident,
            UserSettings(notifySms = true, notifyPush = true, notifyEmail = true)
        )

        assertEquals(AlertSeverity.CRITICAL, dispatch.severity)
        assertTrue(dispatch.shouldSendPush)
        assertTrue(dispatch.shouldSendSms)
    }

    @Test
    fun warningSeverityUsesEscalationDelay() {
        val incident = SafetyIncident(
            emotion = "Neutral",
            threatLevel = 75,
            detectedObjects = emptyList()
        )

        val dispatch = AlertSeverityEvaluator.evaluate(
            incident,
            UserSettings(notifySms = true, notifyPush = true, notifyEmail = false)
        )

        assertEquals(AlertSeverity.WARNING, dispatch.severity)
        assertTrue(dispatch.escalationDelayMs > 0L)
    }
}
