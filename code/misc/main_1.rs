// Prevents additional console window on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod bridge;
mod python_server;

use std::process::Child;
use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::Manager;

use commands::{generate_music, interrogate, get_emotions};

// App state to manage Python server
pub struct AppState {
    pub python_server: Arc<Mutex<Option<Child>>>,
}

fn main() {
    let app_state = AppState {
        python_server: Arc::new(Mutex::new(None)),
    };

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            generate_music,
            interrogate,
            get_emotions,
            python_server::start_python_server,
            python_server::stop_python_server,
            python_server::check_python_server,
        ])
        .setup(|app| {
            // Auto-start Python server on app launch
            let handle = app.handle().clone();
            let state = handle.state::<AppState>();
            let server_handle = state.python_server.clone();
            
            tauri::async_runtime::spawn(async move {
                // Wait a bit for the app to initialize
                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                
                // Try to start the Python server
                if let Err(e) = python_server::start_server(server_handle.clone()).await {
                    eprintln!("Failed to auto-start Python server: {}", e);
                }
            });
            
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
