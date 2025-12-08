#!/usr/bin/env python3
"""
Example: Using Therapy Prompts with DAiW Therapy Session

Demonstrates how to integrate evidence-based therapy prompts
with the existing TherapySession for enhanced feeling extraction.
"""

from music_brain.structure.comprehensive_engine import TherapySession
from music_brain.session.therapy_prompts import (
    TherapyPromptBank,
    TherapyApproach,
    suggest_next_prompt,
    create_casual_therapy_prompt,
    extract_core_elements_from_response,
    identify_emotional_granularity,
)


def interactive_therapy_session():
    """Interactive therapy session using prompts."""
    session = TherapySession()
    
    print("=" * 60)
    print("DAiW Therapy Session - Enhanced with Therapy Prompts")
    print("=" * 60)
    print()
    
    # Start with initial prompt
    current_prompt = TherapyPromptBank.get_initial_prompt()
    session_depth = 0
    collected_elements = {}
    
    print(f"[THERAPIST]: {current_prompt.question}")
    print()
    
    while session_depth < 3:  # Limit to 3 prompts for example
        # Get user response
        response = input("[YOU]: ").strip()
        
        if not response:
            print("[THERAPIST]: I hear you. Sometimes silence is an answer.")
            continue
        
        # Analyze response
        affect = session.process_core_input(response)
        
        # Extract core elements
        elements = extract_core_elements_from_response(response, current_prompt)
        for key, value in elements.items():
            if value:
                collected_elements[key] = value
        
        # Check emotional granularity
        granularity = identify_emotional_granularity(response)
        
        # Show analysis
        print()
        print(f"[ANALYSIS]: Detected affect: {affect}")
        if session.state.affect_result:
            print(f"          Intensity: {session.state.affect_result.intensity:.2f}")
        print(f"          Emotional granularity: {granularity['emotional_granularity']}")
        if granularity['body_references']:
            print("          ✓ Body awareness detected")
        if granularity['metaphor_use']:
            print("          ✓ Metaphorical language detected")
        print()
        
        # Suggest next prompt
        next_prompt = suggest_next_prompt(response, current_prompt, session_depth)
        
        if next_prompt and session_depth < 2:
            print(f"[THERAPIST]: {next_prompt.question}")
            print()
            current_prompt = next_prompt
            session_depth += 1
        else:
            # End session
            print("[THERAPIST]: Thank you for sharing. Let me reflect what I'm hearing...")
            print()
            break
    
    # Set scales
    print("[THERAPIST]: A couple more questions to help me understand your needs:")
    print()
    
    try:
        motivation = int(input("[THERAPIST]: On a scale of 1-10, how much do you need this song right now? >> "))
        chaos_input = int(input("[THERAPIST]: On a scale of 1-10, how much control do you need? (1 = total control, 10 = surprise me) >> "))
        chaos = 1.0 - (chaos_input / 10.0)  # Invert: high input = low control = high chaos tolerance
        session.set_scales(motivation, chaos)
    except ValueError:
        print("[SYSTEM]: Using default values.")
        session.set_scales(5, 0.3)
    
    # Generate plan
    plan = session.generate_plan()
    
    # Show collected elements
    print()
    print("=" * 60)
    print("COLLECTED ELEMENTS")
    print("=" * 60)
    for key, value in collected_elements.items():
        if value and key != "feeling_keywords":
            print(f"{key}: {value}")
    if collected_elements.get("feeling_keywords"):
        print(f"feeling_keywords: {', '.join(collected_elements['feeling_keywords'])}")
    print()
    
    # Show generation directive
    print("=" * 60)
    print("GENERATION DIRECTIVE")
    print("=" * 60)
    print(f"Mode: {plan.root_note} {plan.mode}")
    print(f"Tempo: {plan.tempo_bpm} BPM")
    print(f"Length: {plan.length_bars} bars")
    print(f"Progression: {' - '.join(plan.chord_symbols)}")
    print(f"Complexity: {plan.complexity:.2f}")
    print()
    
    return plan, collected_elements


def demonstrate_prompt_selection():
    """Demonstrate different prompt selection strategies."""
    print("=" * 60)
    print("THERAPY PROMPT DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Show prompts by approach
    print("1. PROMPTS BY APPROACH:")
    print("-" * 60)
    for approach in TherapyApproach:
        prompts = TherapyPromptBank.get_prompts_by_approach(approach)
        print(f"\n{approach.value.upper()}:")
        for i, prompt in enumerate(prompts[:2], 1):  # Show first 2
            print(f"  {i}. {prompt.question}")
            print(f"     Purpose: {prompt.purpose}")
    print()
    
    # Show casual prompts
    print("2. CASUAL PROMPTS:")
    print("-" * 60)
    for _ in range(3):
        casual = create_casual_therapy_prompt()
        print(f"  • {casual}")
    print()
    
    # Show prompt flow
    print("3. PROMPT FLOW EXAMPLE:")
    print("-" * 60)
    initial = TherapyPromptBank.get_initial_prompt()
    print(f"Initial: {initial.question}")
    
    # Simulate responses and next prompts
    responses = [
        "I feel so lost and empty",
        "I want to feel peace and acceptance",
    ]
    
    current = initial
    for i, response in enumerate(responses):
        next_p = suggest_next_prompt(response, current, session_depth=i)
        if next_p:
            print(f"Response: '{response}'")
            print(f"Next: {next_p.question}")
            print()
            current = next_p
    print()


def demonstrate_feeling_extraction():
    """Demonstrate feeling extraction methods."""
    print("=" * 60)
    print("FEELING EXTRACTION DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Example responses
    responses = [
        ("I feel bad", "low granularity"),
        ("I feel a tight, heavy sensation in my chest, like a dark gray cloud", "high granularity"),
        ("I'm afraid that if I let myself feel this grief, I'll never stop crying", "core resistance"),
        ("I want to feel at peace, like I can finally breathe again", "core longing"),
    ]
    
    prompt = TherapyPromptBank.EMOTION_FOCUSED_PROMPTS[0]
    
    for response, label in responses:
        print(f"Response ({label}): '{response}'")
        print("-" * 60)
        
        # Granularity
        granularity = identify_emotional_granularity(response)
        print(f"Granularity: {granularity['emotional_granularity']}")
        print(f"  - Specificity: {granularity['specificity_score']:.2f}")
        print(f"  - Body references: {granularity['body_references']}")
        print(f"  - Metaphor use: {granularity['metaphor_use']}")
        
        # Core elements
        elements = extract_core_elements_from_response(response, prompt)
        if elements.get("core_resistance"):
            print(f"  - Core resistance: {elements['core_resistance']}")
        if elements.get("core_longing"):
            print(f"  - Core longing: {elements['core_longing']}")
        if elements.get("feeling_keywords"):
            print(f"  - Feeling keywords: {', '.join(elements['feeling_keywords'])}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demonstrations
        demonstrate_prompt_selection()
        demonstrate_feeling_extraction()
    else:
        # Run interactive session
        plan, elements = interactive_therapy_session()
        print("\n[SYSTEM]: Session complete. Use render_plan_to_midi() to generate MIDI.")

