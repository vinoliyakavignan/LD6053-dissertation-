package com.example.saftymonitoringsystem

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.saftymonitoringsystem.ui.SafetyViewModel
import com.example.saftymonitoringsystem.ui.screens.MonitoringScreen
import com.example.saftymonitoringsystem.ui.theme.SaftyMonitoringSystemTheme

import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.saftymonitoringsystem.ui.screens.ContactsScreen
import com.example.saftymonitoringsystem.ui.screens.DashboardScreen
import com.example.saftymonitoringsystem.ui.screens.HistoryScreen
import com.example.saftymonitoringsystem.ui.screens.TextAnalysisScreen
import com.example.saftymonitoringsystem.ui.screens.WebScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SaftyMonitoringSystemTheme {
                val navController = rememberNavController()
                val viewModel: SafetyViewModel = viewModel()

                val requiredPermissions = remember {
                    mutableStateListOf<String>().apply {
                        add(Manifest.permission.CAMERA)
                        add(Manifest.permission.ACCESS_FINE_LOCATION)
                        add(Manifest.permission.SEND_SMS)
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            add(Manifest.permission.POST_NOTIFICATIONS)
                        }
                    }
                }

                var permissionsGranted by remember {
                    mutableStateOf(requiredPermissions.all {
                        ContextCompat.checkSelfPermission(this@MainActivity, it) == PackageManager.PERMISSION_GRANTED
                    })
                }

                val launcher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.RequestMultiplePermissions(),
                    onResult = { permissions ->
                        permissionsGranted = permissions.values.all { it }
                    }
                )

                LaunchedEffect(Unit) {
                    if (!permissionsGranted) {
                        launcher.launch(requiredPermissions.toTypedArray())
                    }
                }

                NavHost(navController = navController, startDestination = "dashboard") {
                    composable("dashboard") {
                        DashboardScreen(
                            viewModel = viewModel,
                            onStartMonitoring = {
                                if (permissionsGranted) {
                                    navController.navigate("monitoring")
                                } else {
                                    launcher.launch(requiredPermissions.toTypedArray())
                                }
                            },
                            onNavigateToContacts = {
                                navController.navigate("contacts")
                            },
                            onNavigateToHistory = {
                                navController.navigate("history")
                            },
                            onNavigateToWeb = {
                                navController.navigate("web")
                            },
                            onNavigateToSettings = {},
                            onNavigateToProfile = {}
                        )
                    }
                    composable("monitoring") {
                        MonitoringScreen(viewModel, onBack = { navController.popBackStack() })
                    }
                    composable("contacts") {
                        ContactsScreen(viewModel, onBack = { navController.popBackStack() })
                    }
                    composable("history") {
                        HistoryScreen(viewModel, onBack = { navController.popBackStack() })
                    }
                    composable("textAnalysis") {
                        TextAnalysisScreen(onBack = { navController.popBackStack() })
                    }
                    composable("web") {
                        WebScreen(onBack = { navController.popBackStack() })
                    }
                }
            }
        }
    }
}
