package com.example.saftymonitoringsystem.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.saftymonitoringsystem.ai.ThreatEngine
import com.example.saftymonitoringsystem.data.model.UserSettings
import com.example.saftymonitoringsystem.ui.theme.AccentRed
import com.example.saftymonitoringsystem.ui.theme.AccentGreen
import com.example.saftymonitoringsystem.ui.theme.DarkBackground
import com.example.saftymonitoringsystem.ui.theme.DarkSurface
import com.example.saftymonitoringsystem.ui.theme.DarkSurface2
import com.example.saftymonitoringsystem.ui.theme.OnDark
import com.example.saftymonitoringsystem.ui.theme.OnDarkSecondary
import com.example.saftymonitoringsystem.ui.theme.ThreatHigh
import com.example.saftymonitoringsystem.ui.theme.ThreatLow
import com.example.saftymonitoringsystem.ui.theme.ThreatMedium
import com.example.saftymonitoringsystem.ui.theme.ThreatSafe
import com.example.saftymonitoringsystem.ui.theme.threatColor

/**
 * Text-based emotion analysis screen.
 *
 * Allows the user to type free-text (e.g. a journal entry or message) and
 * receive a heuristic emotion classification based on keyword matching
 * against the emotion vocabulary bundled in `emotion_text_model_config.json`.
 *
 * This complements the camera-based FaceAnalyzer by providing an alternative
 * input modality for users who prefer typing.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextAnalysisScreen(onBack: () -> Unit) {
    var text by remember { mutableStateOf("") }
    var result by remember { mutableStateOf<TextAnalysisResult?>(null) }
    val keyboardController = LocalSoftwareKeyboardController.current

    Scaffold(
        containerColor = DarkBackground,
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Text Emotion Analysis",
                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                        color = OnDark
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = OnDark)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = DarkSurface)
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp)
        ) {
            Text(
                "Type a message or journal entry to analyse emotional tone.",
                style = MaterialTheme.typography.bodyMedium,
                color = OnDarkSecondary,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            OutlinedTextField(
                value = text,
                onValueChange = { if (it.length <= 500) text = it },
                label = { Text("Enter text...") },
                placeholder = { Text("I feel anxious about tomorrow...") },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(140.dp),
                maxLines = 6,
                singleLine = false,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(
                    onSend = {
                        keyboardController?.hide()
                        result = analyzeText(text)
                    }
                ),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color(0xFF5A9EFF),
                    cursorColor = Color(0xFF5A9EFF),
                )
            )

            Spacer(modifier = Modifier.height(12.dp))

            Button(
                onClick = { result = analyzeText(text) },
                modifier = Modifier.fillMaxWidth(),
                enabled = text.isNotBlank(),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF5A9EFF))
            ) {
                Icon(Icons.Default.Send, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("Analyse Emotion")
            }

            result?.let { res ->
                Spacer(modifier = Modifier.height(20.dp))
                ResultCard(res)
            }
        }
    }
}

// ─── Data ─────────────────────────────────────────────────────────────────────

data class TextAnalysisResult(
    val emotion: String,
    val confidence: Float,
    val threatScore: Int,
    val breakdown: Map<String, Float>,
    val message: String
)

// ─── Analysis Logic ───────────────────────────────────────────────────────────

/**
 * Heuristic text-based emotion analysis using keyword matching.
 *
 * Maps detected emotions to the app's 7-emotion schema and computes
 * a threat score via ThreatEngine for consistency with camera-based analysis.
 */
private fun analyzeText(text: String): TextAnalysisResult {
    val lowerText = text.lowercase().trim()
    if (lowerText.isEmpty()) {
        return TextAnalysisResult(
            emotion = "Neutral",
            confidence = 0f,
            threatScore = 0,
            breakdown = emptyMap(),
            message = "Please enter some text to analyse."
        )
    }

    val words = lowerText.split(Regex("[^a-z']+")).filter { it.isNotBlank() }
    val wordCount = words.size

    // Emotion keyword dictionaries (sourced from the bundled emotion vocabulary)
    val emotionKeywords = mapOf(
        "Happy" to setOf("happy", "joy", "excited", "wonderful", "amazing", "great", "love",
            "lovely", "fantastic", "blessed", "grateful", "pleased", "delighted", "joyful",
            "content", "satisfied", "ecstatic", "fabulous", "fun", "enjoy", "celebrate", "beautiful"),
        "Sad" to setOf("sad", "depressed", "lonely", "hurt", "pain", "miserable", "empty",
            "heartbroken", "gloomy", "unhappy", "hopeless", "helpless", "tired", "exhausted",
            "drained", "melancholy", "suffering", "broken", "lost", "cry", "tears", "worried",
            "anxious", "stressed", "overwhelmed", "burdened"),
        "Fear" to setOf("fear", "afraid", "scared", "terrified", "anxious", "worried",
            "nervous", "panic", "paranoid", "threatened", "apprehensive", "shaky",
            "frightened", "concerned", "uneasy", "dread", "horror", "terror"),
        "Angry" to setOf("angry", "mad", "furious", "rage", "hate", "irritated",
            "annoyed", "frustrated", "pissed", "resentful", "hostile", "violent",
            "livid", "enraged", "outraged", "bitter", "resentment"),
        "Disgust" to setOf("disgust", "disgusted", "nauseated", "revolted", "repulsed",
            "sickened", "gross", "filthy", "dirty", "vile", "detest"),
        "Surprise" to setOf("surprise", "surprised", "shocked", "amazed", "astonished",
            "startled", "unexpected", "suddenly", "wow", "incredible"),
        "Neutral" to setOf("neutral", "okay", "fine", "normal", "calm", "peaceful",
            "relaxed", "steady", "stable", "balanced", "clear", "focused")
    )

    // Count keyword matches per emotion
    val scores = mutableMapOf<String, Float>()
    for ((emotion, keywords) in emotionKeywords) {
        val matches = words.count { keywords.contains(it) }
        if (matches > 0) {
            // Score: proportion of words matching × 0.7 + frequency factor × 0.3
            val proportion = matches.toFloat() / wordCount.coerceAtLeast(1)
            val frequencyFactor = (matches.toFloat() / keywords.size).coerceAtMost(1f)
            scores[emotion] = (proportion * 0.7f + frequencyFactor * 0.3f).coerceIn(0f, 1f)
        }
    }

    // If no emotion keywords matched, default to Neutral
    if (scores.isEmpty()) {
        return TextAnalysisResult(
            emotion = "Neutral",
            confidence = 0.3f,
            threatScore = 0,
            breakdown = mapOf("Neutral" to 0.3f),
            message = "No strong emotional keywords detected. Tone appears neutral."
        )
    }

    // Sort by score and take the top emotion
    val sortedEmotions = scores.entries.sortedByDescending { it.value }
    val topEmotion = sortedEmotions.first().key
    val confidence = sortedEmotions.first().value

    // Build breakdown (normalise so they sum to ~1)
    val totalScore = scores.values.sum()
    val breakdown = scores.mapValues { it.value / totalScore.coerceAtLeast(1e-6f) }

    // Compute threat score via ThreatEngine for consistency
    val threatScore = ThreatEngine.calculate(
        emotion = topEmotion,
        emotionConfidence = confidence,
        detectedObjects = emptyList(),
        objectConfidences = emptyList(),
        motionActivity = "Still",
        motionIntensity = 0f
    )

    val message = when (topEmotion) {
        "Fear" -> "Potential distress detected in text. Consider reaching out for support."
        "Angry" -> "Anger detected in text. Take a moment to breathe."
        "Sad" -> "Sadness detected in text. Your feelings are valid."
        "Disgust" -> "Discomfort detected in text."
        "Surprise" -> "Surprise detected in text."
        "Happy" -> "Positive tone detected. Keep it up!"
        else -> "Neutral tone detected."
    }

    return TextAnalysisResult(
        emotion = topEmotion,
        confidence = confidence,
        threatScore = threatScore,
        breakdown = breakdown,
        message = message
    )
}

// ─── UI ───────────────────────────────────────────────────────────────────────

@Composable
private fun ResultCard(result: TextAnalysisResult) {
    val color = threatColor(result.threatScore)

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface2),
        shape = androidx.compose.foundation.shape.RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(8.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "Detected Emotion",
                    style = MaterialTheme.typography.titleMedium,
                    color = OnDark
                )
                Text(
                    result.emotion,
                    style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                    color = color
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                "Confidence: ${(result.confidence * 100).toInt()}%",
                style = MaterialTheme.typography.bodyMedium,
                color = OnDarkSecondary
            )

            Spacer(modifier = Modifier.height(8.dp))

            LinearProgressIndicator(
                progress = { result.confidence },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .padding(vertical = 4.dp),
                color = color,
                trackColor = DarkBackground
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                "Threat Score: ${result.threatScore}%",
                style = MaterialTheme.typography.bodyMedium,
                color = if (result.threatScore >= 70) AccentRed else if (result.threatScore >= 40) ThreatMedium else ThreatSafe
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                result.message,
                style = MaterialTheme.typography.bodySmall,
                color = OnDarkSecondary
            )

            if (result.breakdown.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                Text("Breakdown:", style = MaterialTheme.typography.labelMedium, color = OnDarkSecondary)
                Column {
                    result.breakdown.entries.sortedByDescending { it.value }.forEach { (emotion, score) ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 2.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(emotion, style = MaterialTheme.typography.bodySmall, color = OnDark)
                            Text("${(score * 100).toInt()}%", style = MaterialTheme.typography.bodySmall, color = OnDarkSecondary)
                        }
                    }
                }
            }
        }
    }
}
