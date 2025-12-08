pub mod commands;
pub mod bridge;
pub mod python_server;

use std::sync::Arc;
use tokio::sync::Mutex;
use std::process::Child;

/// Application state shared across Tauri commands
pub struct AppState {
    pub python_server: Arc<Mutex<Option<Child>>>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
