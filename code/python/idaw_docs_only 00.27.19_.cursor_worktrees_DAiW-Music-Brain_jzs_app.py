# app.py
"""
DAiW Streamlit UI - Enhanced desktop-facing product interface.

Multi-page application with:
- Therapy Session (emotion-to-MIDI)
- Intent-Based Generation (three-phase schema)
- Harmony Generator (interactive chord generation)
- MIDI Analysis (file upload and analysis)
- Groove Tools (extraction and application)
"""
import os
import tempfile
import json
from pathlib import Path
from typing import Optional

import streamlit as st

# Core imports
from music_brain.structure.comprehensive_engine import (
    TherapySession,
    render_plan_to_midi,
)
from music_brain.harmony import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongRoot,
    SongIntent,
    TechnicalConstraints,
    SystemDirective,
    suggest_rule_break,
    list_all_rules,
    validate_intent,
)
from music_brain.session.intent_processor import process_intent
from music_brain.structure import analyze_chords, detect_sections
from music_brain.groove import extract_groove, apply_groove
from music_brain.text.lyrical_mirror import generate_lyrical_fragments


# =================================================================
# PAGE CONFIGURATION
# =================================================================

st.set_page_config(
    page_title="DAiW - Digital Audio Intimate Workstation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =================================================================
# SIDEBAR NAVIGATION
# =================================================================

def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.title("🎵 DAiW")
    st.sidebar.caption("Creative companion, not a factory.")
    
    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Home",
            "💭 Therapy Session",
            "📝 Intent Generator",
            "🎹 Harmony Generator",
            "📊 MIDI Analysis",
            "🥁 Groove Tools",
        ],
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**DAiW** translates emotional states into MIDI. "
        "Philosophy: 'Interrogate Before Generate' — "
        "Make musicians braver, not lazier."
    )
    
    return page


# =================================================================
# HOME PAGE
# =================================================================

def render_home():
    """Render home page with overview."""
    st.title("DAiW - Digital Audio Intimate Workstation")
    st.caption("Creative companion, not a factory.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💭 Therapy Session")
        st.markdown(
            "Express your emotional state and generate MIDI "
            "that reflects your inner world."
        )
        if st.button("Go to Therapy Session", use_container_width=True):
            st.session_state.page = "💭 Therapy Session"
            st.rerun()
    
    with col2:
        st.markdown("### 📝 Intent Generator")
        st.markdown(
            "Use the three-phase intent schema to deeply "
            "interrogate your song's emotional core."
        )
        if st.button("Go to Intent Generator", use_container_width=True):
            st.session_state.page = "📝 Intent Generator"
            st.rerun()
    
    with col3:
        st.markdown("### 🎹 Harmony Generator")
        st.markdown(
            "Generate chord progressions with intentional "
            "rule-breaking for emotional impact."
        )
        if st.button("Go to Harmony Generator", use_container_width=True):
            st.session_state.page = "🎹 Harmony Generator"
            st.rerun()
    
    st.markdown("---")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("### 📊 MIDI Analysis")
        st.markdown(
            "Upload MIDI files to analyze chord progressions, "
            "sections, and musical structure."
        )
        if st.button("Go to MIDI Analysis", use_container_width=True):
            st.session_state.page = "📊 MIDI Analysis"
            st.rerun()
    
    with col5:
        st.markdown("### 🥁 Groove Tools")
        st.markdown(
            "Extract groove patterns from MIDI or apply genre "
            "templates to humanize your tracks."
        )
        if st.button("Go to Groove Tools", use_container_width=True):
            st.session_state.page = "🥁 Groove Tools"
            st.rerun()


# =================================================================
# THERAPY SESSION PAGE
# =================================================================

def render_therapy_session():
    """Render therapy session page (original functionality)."""
    st.title("💭 Therapy Session")
    st.markdown("Express what hurts, and we'll translate it into music.")
    
    st.markdown("### 1. Tell me what hurts")
    
    default_text = "I feel dead inside because I chose safety over freedom."
    user_text = st.text_area(
        "What is hurting you right now?",
        value=default_text,
        height=140,
        key="therapy_input",
    )
    
    st.markdown("### 2. Set your state")
    
    col1, col2 = st.columns(2)
    
    with col1:
        motivation = st.slider(
            "Motivation (1 = sketch, 10 = full piece)",
            min_value=1,
            max_value=10,
            value=7,
            key="therapy_motivation",
        )
    
    with col2:
        chaos_1_10 = st.slider(
            "Chaos tolerance (1-10)",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher = more unstable tempo and structure.",
            key="therapy_chaos",
        )
    
    st.markdown("### 3. Generate session")
    
    if st.button("Generate MIDI session", type="primary", use_container_width=True):
        if not user_text.strip():
            st.error("I need at least one sentence to work with.")
            return
        
        with st.spinner("Processing your emotional state..."):
            session = TherapySession()
            affect = session.process_core_input(user_text)
            session.set_scales(motivation, chaos_1_10 / 10.0)
            plan = session.generate_plan()
        
        st.success("Analysis complete!")
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎭 Emotional Analysis")
            if session.state.affect_result:
                st.metric("Primary Affect", affect)
                if session.state.affect_result.secondary:
                    st.metric(
                        "Secondary Undertone",
                        session.state.affect_result.secondary
                    )
                st.metric(
                    "Intensity",
                    f"{session.state.affect_result.intensity:.2f}",
                )
        
        with col2:
            st.subheader("🎵 Generation Directive")
            st.write(f"**Mode:** {plan.root_note} {plan.mode}")
            st.write(f"**Tempo:** {plan.tempo_bpm} BPM")
            st.write(f"**Length:** {plan.length_bars} bars")
            st.write(f"**Progression:** `{' - '.join(plan.chord_symbols)}`")
            st.write(f"**Complexity:** {plan.complexity:.2f}")
        
        with st.spinner("Rendering MIDI..."):
            tmpdir = tempfile.mkdtemp(prefix="daiw_")
            midi_path = os.path.join(tmpdir, "daiw_therapy_session.mid")
            midi_path = render_plan_to_midi(plan, midi_path)
        
        st.success("MIDI generated!")
        
        try:
            with open(midi_path, "rb") as f:
                st.download_button(
                    label="📥 Download MIDI",
                    data=f.read(),
                    file_name="daiw_therapy_session.mid",
                    mime="audio/midi",
                    type="primary",
                    use_container_width=True,
                )
        except OSError:
            st.error("MIDI file could not be read back from disk.")


# =================================================================
# INTENT GENERATOR PAGE
# =================================================================

def render_intent_generator():
    """Render intent-based song generation page."""
    st.title("📝 Intent-Based Song Generator")
    st.markdown(
        "Use the three-phase intent schema to deeply interrogate "
        "your song's emotional core before generating."
    )
    
    tab1, tab2, tab3 = st.tabs(["Phase 0: Core Wound", "Phase 1: Emotional Intent", "Phase 2: Technical"])
    
    with tab1:
        st.markdown("### Phase 0: Core Wound/Desire")
        st.caption("Deep interrogation — what happened? What do you need to say?")
        
        core_event = st.text_area(
            "What happened? (Core Event)",
            placeholder="The specific moment, memory, or realization that sparked this song...",
            height=100,
        )
        core_resistance = st.text_area(
            "What holds you back? (Core Resistance)",
            placeholder="What makes it hard to say this? Fear? Shame? Uncertainty?",
            height=100,
        )
        core_longing = st.text_area(
            "What do you want to feel? (Core Longing)",
            placeholder="What emotional state are you seeking?",
            height=100,
        )
        core_stakes = st.selectbox(
            "What's at risk? (Core Stakes)",
            ["Personal", "Relational", "Existential", "Creative", "Spiritual"],
        )
        core_transformation = st.text_area(
            "How should you feel when done? (Core Transformation)",
            placeholder="What transformation are you seeking through this song?",
            height=100,
        )
    
    with tab2:
        st.markdown("### Phase 1: Emotional Intent")
        st.caption("Validated by Phase 0 — what emotion drives this?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mood_primary = st.text_input(
                "Primary Emotion",
                placeholder="grief, anger, nostalgia, defiance, hope...",
            )
            mood_secondary_tension = st.slider(
                "Secondary Tension (0.0-1.0)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Internal conflict level",
            )
            vulnerability_scale = st.selectbox(
                "Vulnerability Scale",
                ["Low", "Medium", "High"],
            )
        
        with col2:
            imagery_texture = st.text_input(
                "Imagery/Texture",
                placeholder="Visual or tactile quality (e.g., 'gritty', 'ethereal')",
            )
            narrative_arc = st.selectbox(
                "Narrative Arc",
                [
                    "Climb-to-Climax",
                    "Slow Reveal",
                    "Repetitive Despair",
                    "Cathartic Release",
                    "Circular Return",
                ],
            )
    
    with tab3:
        st.markdown("### Phase 2: Technical Constraints")
        st.caption("Implementation — how will this sound?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            technical_genre = st.text_input("Genre", placeholder="bedroom emo, lo-fi, jazz...")
            technical_key = st.selectbox(
                "Key",
                ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
            )
            technical_mode = st.selectbox(
                "Mode",
                ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian"],
            )
        
        with col2:
            tempo_min, tempo_max = st.slider(
                "Tempo Range (BPM)",
                min_value=60,
                max_value=180,
                value=(80, 120),
            )
            technical_groove_feel = st.selectbox(
                "Groove Feel",
                [
                    "Tight/Grid",
                    "Organic/Breathing",
                    "Loose/Laid-back",
                    "Anxious/Unstable",
                ],
            )
        
        st.markdown("#### Rule Breaking")
        
        emotion_for_suggestions = st.text_input(
            "Emotion for rule suggestions",
            placeholder="Enter emotion to get rule-breaking suggestions",
        )
        
        if emotion_for_suggestions:
            suggestions = suggest_rule_break(emotion_for_suggestions)
            if suggestions:
                st.info("💡 Suggested rules to break:")
                for sug in suggestions[:3]:  # Show top 3
                    st.write(f"**{sug['rule']}**: {sug['description']}")
                    st.caption(f"Effect: {sug['effect']}")
        
        technical_rule_to_break = st.text_input(
            "Rule to Break (optional)",
            placeholder="e.g., HARMONY_AvoidTonicResolution",
            help="Leave empty for no rule-breaking",
        )
        rule_breaking_justification = st.text_area(
            "Why break this rule?",
            placeholder="Emotional justification for breaking this rule...",
            height=80,
        )
    
    st.markdown("---")
    
    if st.button("Generate from Intent", type="primary", use_container_width=True):
        if not core_event or not mood_primary:
            st.error("Please fill in at least Core Event and Primary Emotion.")
            return
        
        with st.spinner("Processing intent and generating musical elements..."):
            try:
                intent = CompleteSongIntent(
                    title="Generated Song",
                    song_root=SongRoot(
                        core_event=core_event,
                        core_resistance=core_resistance or "[Not specified]",
                        core_longing=core_longing or "[Not specified]",
                        core_stakes=core_stakes,
                        core_transformation=core_transformation or "[Not specified]",
                    ),
                    song_intent=SongIntent(
                        mood_primary=mood_primary,
                        mood_secondary_tension=mood_secondary_tension,
                        imagery_texture=imagery_texture or "[Not specified]",
                        vulnerability_scale=vulnerability_scale,
                        narrative_arc=narrative_arc,
                    ),
                    technical_constraints=TechnicalConstraints(
                        technical_genre=technical_genre or "[Not specified]",
                        technical_tempo_range=(tempo_min, tempo_max),
                        technical_key=technical_key,
                        technical_mode=technical_mode,
                        technical_groove_feel=technical_groove_feel,
                        technical_rule_to_break=technical_rule_to_break or "",
                        rule_breaking_justification=rule_breaking_justification or "",
                    ),
                    system_directive=SystemDirective(
                        output_target="Chord progression",
                        output_feedback_loop="Harmony",
                    ),
                )
                
                # Validate
                issues = validate_intent(intent)
                if issues:
                    st.warning("⚠️ Validation issues found:")
                    for issue in issues:
                        st.write(f"- {issue}")
                
                # Process
                result = process_intent(intent)
                
                st.success("Generation complete!")
                
                # Display results
                st.subheader("🎵 Generated Elements")
                
                # Harmony
                harmony = result['harmony']
                with st.expander("📌 Harmony", expanded=True):
                    st.write(f"**Progression:** {' - '.join(harmony.chords)}")
                    st.write(f"**Roman Numerals:** {' - '.join(harmony.roman_numerals)}")
                    if harmony.rule_broken:
                        st.write(f"**Rule Broken:** {harmony.rule_broken}")
                        st.write(f"**Effect:** {harmony.rule_effect}")
                
                # Groove
                groove = result['groove']
                with st.expander("🥁 Groove"):
                    st.write(f"**Pattern:** {groove.pattern_name}")
                    st.write(f"**Tempo:** {groove.tempo_bpm} BPM")
                    if groove.rule_broken:
                        st.write(f"**Rule Broken:** {groove.rule_broken}")
                
                # Arrangement
                arr = result['arrangement']
                with st.expander("🎼 Arrangement"):
                    for section in arr.sections:
                        st.write(f"**{section['name']}:** {section['bars']} bars @ {section['energy']:.0%} energy")
                
                # Production
                prod = result['production']
                with st.expander("🎚️ Production"):
                    st.write(f"**Vocal Treatment:** {prod.vocal_treatment}")
                    if prod.eq_notes:
                        st.write("**EQ Notes:**")
                        for note in prod.eq_notes[:3]:
                            st.write(f"- {note}")
                
                # Lyrical fragments
                with st.spinner("Generating lyrical fragments..."):
                    fragments = generate_lyrical_fragments(intent)
                    if fragments:
                        with st.expander("📝 Lyrical Fragments"):
                            for fragment in fragments[:5]:
                                st.write(f"*{fragment}*")
                
            except Exception as e:
                st.error(f"Error generating from intent: {str(e)}")
                st.exception(e)


# =================================================================
# HARMONY GENERATOR PAGE
# =================================================================

def render_harmony_generator():
    """Render harmony generator page."""
    st.title("🎹 Harmony Generator")
    st.markdown("Generate chord progressions with intentional rule-breaking.")
    
    tab1, tab2 = st.tabs(["Basic Generation", "From Intent File"])
    
    with tab1:
        st.markdown("### Basic Harmony Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            key = st.selectbox(
                "Key",
                ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
                index=5,  # Default to F
            )
            mode = st.selectbox(
                "Mode",
                ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian"],
            )
        
        with col2:
            pattern = st.text_input(
                "Roman Numeral Pattern",
                value="I-V-vi-IV",
                help="e.g., I-V-vi-IV, I-vi-IV-V, ii-V-I",
            )
            tempo = st.number_input(
                "Tempo (BPM)",
                min_value=60,
                max_value=180,
                value=82,
            )
        
        if st.button("Generate Progression", type="primary"):
            with st.spinner("Generating harmony..."):
                try:
                    generator = HarmonyGenerator()
                    harmony = generator.generate_basic_progression(
                        key=key,
                        mode=mode,
                        pattern=pattern,
                    )
                    
                    st.success("Harmony generated!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Key", f"{harmony.key} {harmony.mode}")
                        st.metric("Progression", " - ".join(harmony.chords))
                    
                    with col2:
                        if harmony.rule_break_applied:
                            st.info(f"**Rule Break:** {harmony.rule_break_applied}")
                            if harmony.emotional_justification:
                                st.caption(harmony.emotional_justification)
                    
                    # Generate MIDI
                    with st.spinner("Generating MIDI..."):
                        tmpdir = tempfile.mkdtemp(prefix="daiw_")
                        midi_path = os.path.join(tmpdir, "harmony.mid")
                        generate_midi_from_harmony(harmony, midi_path, tempo_bpm=tempo)
                        
                        with open(midi_path, "rb") as f:
                            st.download_button(
                                label="📥 Download MIDI",
                                data=f.read(),
                                file_name="harmony.mid",
                                mime="audio/midi",
                                type="primary",
                                use_container_width=True,
                            )
                
                except Exception as e:
                    st.error(f"Error generating harmony: {str(e)}")
                    st.exception(e)
    
    with tab2:
        st.markdown("### Generate from Intent File")
        
        uploaded_file = st.file_uploader(
            "Upload Intent JSON File",
            type=["json"],
            help="Upload a CompleteSongIntent JSON file",
        )
        
        if uploaded_file:
            try:
                intent_data = json.load(uploaded_file)
                intent = CompleteSongIntent.from_dict(intent_data)
                
                st.success("Intent file loaded!")
                
                with st.expander("View Intent Summary"):
                    st.json(intent_data)
                
                if st.button("Generate Harmony from Intent", type="primary"):
                    with st.spinner("Generating harmony from intent..."):
                        try:
                            generator = HarmonyGenerator()
                            harmony = generator.generate_from_intent(intent)
                            
                            st.success("Harmony generated!")
                            
                            st.write(f"**Key:** {harmony.key} {harmony.mode}")
                            st.write(f"**Progression:** {' - '.join(harmony.chords)}")
                            
                            if harmony.rule_break_applied:
                                st.info(f"**Rule Break:** {harmony.rule_break_applied}")
                            
                            # Generate MIDI
                            tempo = intent.technical_constraints.technical_tempo_range[0]
                            tmpdir = tempfile.mkdtemp(prefix="daiw_")
                            midi_path = os.path.join(tmpdir, "harmony_from_intent.mid")
                            generate_midi_from_harmony(harmony, midi_path, tempo_bpm=tempo)
                            
                            with open(midi_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download MIDI",
                                    data=f.read(),
                                    file_name="harmony_from_intent.mid",
                                    mime="audio/midi",
                                    type="primary",
                                    use_container_width=True,
                                )
                        
                        except Exception as e:
                            st.error(f"Error generating harmony: {str(e)}")
                            st.exception(e)
            
            except Exception as e:
                st.error(f"Error loading intent file: {str(e)}")


# =================================================================
# MIDI ANALYSIS PAGE
# =================================================================

def render_midi_analysis():
    """Render MIDI analysis page."""
    st.title("📊 MIDI Analysis")
    st.markdown("Upload MIDI files to analyze chord progressions and structure.")
    
    uploaded_file = st.file_uploader(
        "Upload MIDI File",
        type=["mid", "midi"],
        help="Upload a MIDI file for analysis",
    )
    
    if uploaded_file:
        # Save uploaded file temporarily
        tmpdir = tempfile.mkdtemp(prefix="daiw_")
        midi_path = os.path.join(tmpdir, uploaded_file.name)
        
        with open(midi_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        tab1, tab2 = st.tabs(["Chord Analysis", "Section Detection"])
        
        with tab1:
            st.markdown("### Chord Progression Analysis")
            
            if st.button("Analyze Chords", type="primary"):
                with st.spinner("Analyzing chord progression..."):
                    try:
                        progression = analyze_chords(midi_path)
                        
                        st.success("Analysis complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Key", progression.key)
                            st.metric("Progression", " - ".join(progression.chords))
                        
                        with col2:
                            st.metric("Roman Numerals", " - ".join(progression.roman_numerals))
                            if progression.borrowed_chords:
                                st.write("**Borrowed Chords:**")
                                for chord, source in progression.borrowed_chords.items():
                                    st.write(f"- {chord} ← from {source}")
                    
                    except Exception as e:
                        st.error(f"Error analyzing chords: {str(e)}")
                        st.exception(e)
        
        with tab2:
            st.markdown("### Section Detection")
            
            if st.button("Detect Sections", type="primary"):
                with st.spinner("Detecting sections..."):
                    try:
                        sections = detect_sections(midi_path)
                        
                        st.success(f"Found {len(sections)} sections!")
                        
                        for section in sections:
                            with st.expander(f"{section.name} (bars {section.start_bar}-{section.end_bar})"):
                                st.metric("Energy", f"{section.energy:.2f}")
                                if hasattr(section, 'tempo'):
                                    st.metric("Tempo", f"{section.tempo} BPM")
                    
                    except Exception as e:
                        st.error(f"Error detecting sections: {str(e)}")
                        st.exception(e)


# =================================================================
# GROOVE TOOLS PAGE
# =================================================================

def render_groove_tools():
    """Render groove tools page."""
    st.title("🥁 Groove Tools")
    st.markdown("Extract groove patterns or apply genre templates.")
    
    tab1, tab2 = st.tabs(["Extract Groove", "Apply Groove Template"])
    
    with tab1:
        st.markdown("### Extract Groove from MIDI")
        
        uploaded_file = st.file_uploader(
            "Upload MIDI File",
            type=["mid", "midi"],
            help="Upload a MIDI file to extract groove from",
            key="groove_extract_upload",
        )
        
        if uploaded_file:
            tmpdir = tempfile.mkdtemp(prefix="daiw_")
            midi_path = os.path.join(tmpdir, uploaded_file.name)
            
            with open(midi_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Extract Groove", type="primary"):
                with st.spinner("Extracting groove..."):
                    try:
                        groove = extract_groove(midi_path)
                        
                        st.success("Groove extracted!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Timing Deviation",
                                f"{groove.timing_stats['mean_deviation_ms']:.1f} ms",
                            )
                            st.metric("Swing Factor", f"{groove.swing_factor:.2f}")
                        
                        with col2:
                            st.metric(
                                "Velocity Range",
                                f"{groove.velocity_stats['min']}-{groove.velocity_stats['max']}",
                            )
                        
                        # Save groove JSON
                        groove_dict = groove.to_dict()
                        groove_json = json.dumps(groove_dict, indent=2)
                        
                        st.download_button(
                            label="📥 Download Groove JSON",
                            data=groove_json,
                            file_name="groove.json",
                            mime="application/json",
                            type="primary",
                        )
                    
                    except Exception as e:
                        st.error(f"Error extracting groove: {str(e)}")
                        st.exception(e)
    
    with tab2:
        st.markdown("### Apply Groove Template")
        
        uploaded_file = st.file_uploader(
            "Upload MIDI File",
            type=["mid", "midi"],
            help="Upload a MIDI file to apply groove to",
            key="groove_apply_upload",
        )
        
        if uploaded_file:
            tmpdir = tempfile.mkdtemp(prefix="daiw_")
            midi_path = os.path.join(tmpdir, uploaded_file.name)
            
            with open(midi_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            col1, col2 = st.columns(2)
            
            with col1:
                genre = st.selectbox(
                    "Genre Template",
                    ["funk", "jazz", "rock", "hiphop", "edm", "latin"],
                )
            
            with col2:
                intensity = st.slider(
                    "Groove Intensity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            
            if st.button("Apply Groove", type="primary"):
                with st.spinner(f"Applying {genre} groove..."):
                    try:
                        input_name = Path(uploaded_file.name).stem
                        output_path = os.path.join(tmpdir, f"{input_name}_grooved.mid")
                        apply_groove(midi_path, genre=genre, output=output_path, intensity=intensity)
                        
                        st.success("Groove applied!")
                        
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Grooved MIDI",
                                data=f.read(),
                                file_name=f"{input_name}_grooved.mid",
                                mime="audio/midi",
                                type="primary",
                                use_container_width=True,
                            )
                    
                    except Exception as e:
                        st.error(f"Error applying groove: {str(e)}")
                        st.exception(e)


# =================================================================
# MAIN APP
# =================================================================

def main() -> None:
    """Main application entry point."""
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Override with session state if set
    if "page" in st.session_state:
        selected_page = st.session_state.page
    
    # Route to appropriate page
    if selected_page == "🏠 Home":
        render_home()
    elif selected_page == "💭 Therapy Session":
        render_therapy_session()
    elif selected_page == "📝 Intent Generator":
        render_intent_generator()
    elif selected_page == "🎹 Harmony Generator":
        render_harmony_generator()
    elif selected_page == "📊 MIDI Analysis":
        render_midi_analysis()
    elif selected_page == "🥁 Groove Tools":
        render_groove_tools()


if __name__ == "__main__":
    main()
