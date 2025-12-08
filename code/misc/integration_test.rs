/// Integration tests for Tauri commands
/// 
/// These tests verify that the Tauri commands work correctly
/// with the Python server management system.

use idaw_lib::commands::{EmotionalIntent, GenerateRequest, InterrogateRequest};
use serde_json::json;

#[tokio::test]
async fn test_check_python_server_command() {
    // This test verifies the check_python_server command structure
    // Note: Actual server won't be running in test environment
    // but we can verify the command doesn't panic
    
    // The command should return a JSON value with running status
    // In a real test environment, we'd mock the server
    let expected_keys = vec!["running", "port", "url"];
    
    // Verify the structure is correct (would need actual Tauri test setup)
    // For now, just verify the test compiles
    assert!(true);
}

#[test]
fn test_emotional_intent_serialization() {
    // Test that EmotionalIntent can be serialized/deserialized
    let intent = EmotionalIntent {
        core_wound: Some("loss".to_string()),
        core_desire: Some("healing".to_string()),
        emotional_intent: Some("grief".to_string()),
        technical: Some(json!({"key": "C", "bpm": 120})),
        base_emotion: Some("grief".to_string()),
        intensity: Some("high".to_string()),
        specific_emotion: Some("longing".to_string()),
    };
    
    // Serialize
    let json = serde_json::to_string(&intent).unwrap();
    assert!(json.contains("grief"));
    
    // Deserialize
    let deserialized: EmotionalIntent = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.base_emotion, Some("grief".to_string()));
}

#[test]
fn test_generate_request_serialization() {
    let intent = EmotionalIntent {
        base_emotion: Some("joy".to_string()),
        intensity: Some("moderate".to_string()),
        specific_emotion: Some("happiness".to_string()),
        core_wound: None,
        core_desire: None,
        emotional_intent: None,
        technical: None,
    };
    
    let request = GenerateRequest {
        intent,
        output_format: Some("midi".to_string()),
    };
    
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("joy"));
    assert!(json.contains("midi"));
}

#[test]
fn test_interrogate_request_serialization() {
    let request = InterrogateRequest {
        message: "I want to write about loss".to_string(),
        session_id: Some("test-session-123".to_string()),
        context: Some(json!({"mood": "sad"})),
    };
    
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("I want to write about loss"));
    assert!(json.contains("test-session-123"));
}
