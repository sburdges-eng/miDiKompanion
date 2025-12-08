use serde::{Deserialize, Serialize};
use tauri::command;

#[derive(Debug, Serialize, Deserialize)]
pub struct EmotionalIntent {
    pub core_wound: Option<String>,
    pub core_desire: Option<String>,
    #[serde(default)]
    pub emotional_intent: Option<String>,  // Legacy field
    pub technical: Option<serde_json::Value>,
    // New format: base_emotion, intensity, specific_emotion
    pub base_emotion: Option<String>,
    pub intensity: Option<String>,
    pub specific_emotion: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub intent: EmotionalIntent,
    pub output_format: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InterrogateRequest {
    pub message: String,
    pub session_id: Option<String>,
    pub context: Option<serde_json::Value>,
}

#[command]
pub async fn generate_music(
    request: GenerateRequest,
    state: tauri::State<'_, crate::AppState>,
) -> Result<serde_json::Value, String> {
    // Ensure Python server is running
    if !crate::python_server::check_server_health().await {
        // Try to start it
        if let Err(e) = crate::python_server::start_server(
            state.python_server.clone()
        ).await {
            return Err(format!("Python server not available: {}", e));
        }
        // Wait a bit more
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    }
    
    crate::bridge::musicbrain::generate(request)
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn interrogate(
    request: InterrogateRequest,
    state: tauri::State<'_, crate::AppState>,
) -> Result<serde_json::Value, String> {
    // Ensure Python server is running
    if !crate::python_server::check_server_health().await {
        if let Err(e) = crate::python_server::start_server(
            state.python_server.clone()
        ).await {
            return Err(format!("Python server not available: {}", e));
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    }
    
    crate::bridge::musicbrain::interrogate(request)
        .await
        .map_err(|e| e.to_string())
}

#[command]
pub async fn get_emotions(
    state: tauri::State<'_, crate::AppState>,
) -> Result<serde_json::Value, String> {
    // Ensure Python server is running
    if !crate::python_server::check_server_health().await {
        if let Err(e) = crate::python_server::start_server(
            state.python_server.clone()
        ).await {
            return Err(format!("Python server not available: {}", e));
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    }
    
    crate::bridge::musicbrain::get_emotions()
        .await
        .map_err(|e| e.to_string())
}
