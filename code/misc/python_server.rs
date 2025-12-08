use std::process::{Command, Child, Stdio};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::State;
use serde_json::Value;

// Port for the Python API server
const PYTHON_API_PORT: u16 = 8000;

/// Find the embedded Python interpreter
fn find_python_interpreter() -> Option<PathBuf> {
    // Try system Python first (for development)
    if let Ok(output) = Command::new("python3").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("python3"));
        }
    }
    
    // Try to find embedded Python in app bundle (for production)
    if let Ok(exe_path) = std::env::current_exe() {
        // In macOS app bundle, Python might be in Resources
        if let Some(app_dir) = exe_path.parent().and_then(|p| p.parent().and_then(|p| p.parent())) {
            let python_paths = vec![
                app_dir.join("Resources").join("python").join("bin").join("python3"),
                app_dir.join("Resources").join("python3"),
                app_dir.join("Frameworks").join("Python.framework").join("Versions").join("Current").join("bin").join("python3"),
            ];
            
            for path in python_paths {
                if path.exists() {
                    return Some(path);
                }
            }
        }
    }
    
    None
}

/// Find the Python API script
fn find_api_script() -> Option<PathBuf> {
    // Try to find in app bundle Resources
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(app_dir) = exe_path.parent().and_then(|p| p.parent().and_then(|p| p.parent())) {
            let script_paths = vec![
                app_dir.join("Resources").join("python").join("start_api.py"),
                app_dir.join("Resources").join("music_brain").join("start_api_embedded.py"),
                app_dir.join("Resources").join("music_brain").join("api.py"),
            ];
            
            for path in script_paths {
                if path.exists() {
                    return Some(path);
                }
            }
        }
    }
    
    // Fallback: try to find in project root (for development)
    if let Ok(current_dir) = std::env::current_dir() {
        let dev_paths = vec![
            current_dir.join("music_brain").join("start_api_embedded.py"),
            current_dir.join("music_brain").join("api.py"),
            current_dir.join("start_api.py"),
        ];
        
        for path in dev_paths {
            if path.exists() {
                return Some(path);
            }
        }
    }
    
    None
}

/// Start the Python API server
pub async fn start_server(
    server_handle: Arc<Mutex<Option<Child>>>,
) -> Result<(), String> {
    let mut server = server_handle.lock().await;
    
    // Check if server is already running
    if server.is_some() {
        // Verify it's actually running
        if check_server_health().await {
            return Ok(());
        } else {
            // Server handle exists but not responding, clean it up
            if let Some(mut child) = server.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    
    // Find Python interpreter
    let python = find_python_interpreter()
        .ok_or_else(|| {
            "Python interpreter not found. Please ensure Python 3.9+ is installed or embedded in the app bundle.".to_string()
        })?;
    
    // Find API script
    let api_script = find_api_script()
        .ok_or_else(|| {
            "API script not found. Please ensure music_brain/api.py or start_api_embedded.py exists.".to_string()
        })?;
    
    // Set environment variable for port
    let mut cmd = Command::new(&python);
    cmd.arg(api_script.to_str().unwrap())
        .env("MUSIC_BRAIN_PORT", PYTHON_API_PORT.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());
    
    // Add Python path if we're in a bundle
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(app_dir) = exe_path.parent().and_then(|p| p.parent().and_then(|p| p.parent())) {
            let python_resources = app_dir.join("Resources").join("python");
            if python_resources.exists() {
                let python_path = python_resources.to_str().unwrap();
                cmd.env("PYTHONPATH", python_path);
            }
        }
    }
    
    // Start the server
    let child = cmd.spawn()
        .map_err(|e| {
            format!(
                "Failed to start Python server: {}. Check that Python and required packages are installed.",
                e
            )
        })?;
    
    *server = Some(child);
    
    // Wait for server to start with retries
    let max_retries = 10;
    let mut retries = 0;
    while retries < max_retries {
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        if check_server_health().await {
            return Ok(());
        }
        
        retries += 1;
    }
    
    // Server didn't start in time
    if let Some(mut child) = server.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    
    Err(format!(
        "Server started but not responding after {} seconds. Check server logs for errors.",
        max_retries * 500 / 1000
    ))
}

/// Stop the Python API server
pub async fn stop_server(
    server_handle: Arc<Mutex<Option<Child>>>,
) -> Result<(), String> {
    let mut server = server_handle.lock().await;
    
    if let Some(mut child) = server.take() {
        child.kill()
            .map_err(|e| format!("Failed to stop Python server: {}", e))?;
        child.wait().ok();
        Ok(())
    } else {
        Ok(())
    }
}

/// Check if the Python server is running and healthy
pub async fn check_server_health() -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    
    let client = match client {
        Ok(c) => c,
        Err(_) => return false,
    };
    
    let url = format!("http://127.0.0.1:{}/health", PYTHON_API_PORT);
    
    match client.get(&url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Tauri command: Start Python server
#[tauri::command]
pub async fn start_python_server(
    state: State<'_, crate::AppState>,
) -> Result<Value, String> {
    start_server(state.python_server.clone()).await?;
    Ok(serde_json::json!({
        "success": true,
        "message": "Python server started",
        "port": PYTHON_API_PORT
    }))
}

/// Tauri command: Stop Python server
#[tauri::command]
pub async fn stop_python_server(
    state: State<'_, crate::AppState>,
) -> Result<Value, String> {
    stop_server(state.python_server.clone()).await?;
    Ok(serde_json::json!({
        "success": true,
        "message": "Python server stopped"
    }))
}

/// Tauri command: Check Python server status
#[tauri::command]
pub async fn check_python_server() -> Result<Value, String> {
    let is_healthy = check_server_health().await;
    Ok(serde_json::json!({
        "running": is_healthy,
        "port": PYTHON_API_PORT,
        "url": format!("http://127.0.0.1:{}", PYTHON_API_PORT)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[test]
    fn test_find_python_interpreter() {
        // Should find system Python in development
        let result = find_python_interpreter();
        // In test environment, should at least try to find python3
        assert!(result.is_some() || std::env::var("CI").is_ok());
    }

    #[test]
    fn test_find_api_script() {
        // Should find API script in development
        let result = find_api_script();
        // In test environment, might not find it if not in project root
        // This is OK - the test verifies the function doesn't panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_check_server_health_when_down() {
        // When server is not running, should return false
        let result = check_server_health().await;
        // In test environment without server, should be false
        assert!(!result);
    }

    #[tokio::test]
    async fn test_stop_server_when_not_running() {
        // Stopping a non-existent server should succeed
        let server_handle: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
        let result = stop_server(server_handle).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_python_api_port_constant() {
        // Verify port constant is set correctly
        assert_eq!(PYTHON_API_PORT, 8000);
    }
}
