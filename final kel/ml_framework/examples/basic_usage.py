"""
Basic Usage Example

Demonstrates how to use the CIF/LAS/QEF ML Framework.
"""

from cif_las_qef import UnifiedFramework, FrameworkConfig

def main():
    # Create framework configuration
    config = FrameworkConfig(
        enable_cif=True,
        enable_las=True,
        enable_ethics=True,
        enable_qef=True,
        ethics_strict_mode=True
    )
    
    # Initialize framework
    framework = UnifiedFramework(config)
    
    # Example: Human emotional input
    human_input = {
        "biofeedback": {
            "heart_rate": 75.0,
            "eeg_alpha": 0.6,
            "gsr": 0.4
        },
        "voice": {
            "tone": 0.3,
            "intensity": 0.5
        },
        "text": {
            "sentiment": 0.4,
            "energy": 0.6
        },
        "intent": {
            "type": "creation",
            "clarity": 0.7
        }
    }
    
    # Create with consent
    print("Creating with ethical consent protocol...")
    result = framework.create_with_consent(
        human_emotional_input=human_input,
        creative_goal={
            "style": "ambient",
            "emotion": "calm",
            "min_length": 16,
            "max_length": 32
        },
        require_consent=True
    )
    
    print("\n=== Creation Result ===")
    print(f"Created: {result.get('created', False)}")
    print(f"Consent Granted: {result.get('consent_granted', False)}")
    print(f"Overall Ethics Score: {result.get('overall_ethics', 'N/A')}")
    
    if result.get('las_output'):
        print(f"\nLAS Output: {result['las_output'].get('output', {}).get('content_type', 'N/A')}")
    
    # Provide feedback
    print("\n=== Providing Feedback ===")
    feedback = {
        "aesthetic_rating": 0.8,
        "emotional_resonance": 0.7,
        "engagement": 0.75
    }
    
    evolution = framework.evolve_from_feedback(feedback)
    print(f"Evolution Result: {evolution}")
    
    # Get collective resonance (if QEF enabled)
    if config.enable_qef:
        print("\n=== Collective Resonance ===")
        collective = framework.get_collective_resonance()
        print(f"Resonance Level: {collective.get('resonance_level', 'N/A')}")
        print(f"Active Nodes: {collective.get('active_nodes', 'N/A')}")
    
    # Get framework status
    print("\n=== Framework Status ===")
    status = framework.get_status()
    print(f"Session Count: {status.get('session_count', 0)}")
    print(f"CIF Enabled: {status['config']['cif_enabled']}")
    print(f"LAS Enabled: {status['config']['las_enabled']}")
    print(f"Ethics Enabled: {status['config']['ethics_enabled']}")
    print(f"QEF Enabled: {status['config']['qef_enabled']}")

if __name__ == "__main__":
    main()
