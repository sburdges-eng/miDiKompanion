"""
iDAW Streamlit UI
=================

Run with: streamlit run idaw_ui.py

Requirements:
    pip install streamlit music21 mido numpy
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from idaw_complete_pipeline import (
    InterrogationEngine, 
    get_parameters_for_state,
    StructureGenerator,
    HarmonyEngine,
    MelodyEngine,
    GrooveEngine,
    MIDIBuilder,
    PostProcessor,
    EMOTIONAL_PRESETS,
    EmotionalState,
    TimingFeel,
    RuleBreakCode,
)
from datetime import datetime

# Page config
st.set_page_config(
    page_title="iDAW", 
    page_icon="🎵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #888;
        font-style: italic;
    }
    .metric-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎵 iDAW</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">intelligent Digital Audio Workspace — Interrogate Before Generate</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR - Controls
# ============================================================================
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Mode selection
    mode = st.radio(
        "Mode",
        ["Generate", "Critique", "Analyze"],
        help="Generate: Create new music | Critique: Analyze your ideas | Analyze: Break down existing songs"
    )
    
    st.markdown("---")
    
    # Preset selection
    st.subheader("Emotion Presets")
    preset = st.selectbox(
        "Quick Start",
        ["(custom)"] + list(EMOTIONAL_PRESETS.keys()),
        help="Start from an emotional preset"
    )
    
    st.markdown("---")
    
    # Manual overrides
    st.subheader("Manual Overrides")
    
    tempo = st.slider("Tempo (BPM)", 40, 180, 82, help="Beats per minute")
    
    key_sig = st.selectbox(
        "Key",
        ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        index=5,  # Default to F
        help="Root key"
    )
    
    humanize = st.slider(
        "Humanize", 
        0.0, 0.5, 0.2,
        help="Timing variation (0 = robotic, 0.5 = very loose)"
    )
    
    timing_feel = st.select_slider(
        "Pocket",
        options=["ahead", "on", "behind"],
        value="behind",
        help="Where notes sit relative to beat"
    )
    
    dissonance = st.slider(
        "Dissonance",
        0.0, 1.0, 0.3,
        help="Harmonic tension level"
    )
    
    imperfection = st.slider(
        "Lo-Fi Level",
        0.0, 0.6, 0.3,
        help="Imperfection aesthetic (0 = clean, 0.6 = very lo-fi)"
    )
    
    st.markdown("---")
    
    # Rule breaks
    st.subheader("Rule Breaks")
    
    rule_breaks = st.multiselect(
        "Intentional Theory Violations",
        [
            "Non-Resolution (grief, longing)",
            "Modal Interchange (bittersweet)",
            "Parallel Motion (power)",
            "Tempo Fluctuation (intimacy)",
            "Pitch Imperfection (vulnerability)",
        ],
        help="Which rules to break for emotional effect"
    )
    
    st.markdown("---")
    
    # Output settings
    st.subheader("Output")
    output_dir = st.text_input(
        "Output Directory",
        str(Path.home() / "Music" / "iDAW_Output"),
        help="Where to save generated files"
    )

# ============================================================================
# MAIN AREA - Based on Mode
# ============================================================================

if mode == "Generate":
    st.header("1️⃣ Interrogation")
    
    # Two-column layout for interrogation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phase 0: Core Wound")
        st.markdown("*The event, not the feeling*")
        
        core_event = st.text_area(
            "What happened?",
            placeholder="e.g., 'I found my friend after she took her own life'",
            height=80,
            help="The raw event - be as specific as you can"
        )
        
        core_resistance = st.text_input(
            "What's hardest to say?",
            placeholder="e.g., 'I couldn't save her'",
            help="The thing you resist admitting"
        )
        
        core_longing = st.text_input(
            "What do you wish you could feel?",
            placeholder="e.g., 'Peace, or permission to let go'",
            help="The transformation you're seeking"
        )
    
    with col2:
        st.subheader("Phase 1: Emotional Intent")
        st.markdown("*How should it feel?*")
        
        user_input = st.text_area(
            "Describe the song (vernacular welcome):",
            placeholder="e.g., 'slow grief song, acoustic, laid back, lo-fi bedroom feel, sounds like a love song but it's about death'",
            height=100,
            help="Use casual terms - 'fat', 'crispy', 'laid back' all work"
        )
        
        vulnerability = st.slider(
            "Vulnerability Level",
            1, 10, 7,
            help="How exposed should this feel? (10 = completely raw)"
        )
        
        misdirection = st.checkbox(
            "Use Misdirection",
            help="Surface emotion differs from undertow (sounds like X but is really Y)"
        )
        
        if misdirection:
            col_a, col_b = st.columns(2)
            with col_a:
                surface_emotion = st.selectbox("Surface (what it sounds like)", list(EMOTIONAL_PRESETS.keys()))
            with col_b:
                undertow_emotion = st.selectbox("Undertow (what it really is)", list(EMOTIONAL_PRESETS.keys()), index=0)

    st.markdown("---")
    
    # Generate button
    if st.button("🎵 Generate Song", type="primary", use_container_width=True):
        
        # Progress container
        progress = st.progress(0)
        status = st.empty()
        
        # 1. INTERROGATION
        status.text("1/7 Interrogating emotional state...")
        progress.progress(14)
        
        engine = InterrogationEngine()
        
        # Build input from all fields
        full_input = user_input or ""
        if core_event:
            full_input += f" about {core_event}"
        if preset != "(custom)":
            full_input = f"{preset} {full_input}"
            
        state = engine.quick_interrogate(full_input)
        
        # Apply manual settings
        if preset != "(custom)":
            state.primary_emotion = preset
        state.vulnerability = vulnerability / 10.0
        if misdirection:
            state.misdirection_intensity = 0.8
            state.surface_emotion = surface_emotion
            state.undertow_emotion = undertow_emotion
            
        # 2. PARAMETERS
        status.text("2/7 Translating to musical parameters...")
        progress.progress(28)
        
        params = get_parameters_for_state(state)
        
        # Apply overrides from sidebar
        params.tempo_suggested = tempo
        params.key_signature = key_sig
        params.humanize = humanize
        params.dissonance = dissonance
        params.imperfection_level = imperfection
        params.timing_feel = {
            "ahead": TimingFeel.AHEAD,
            "on": TimingFeel.ON,
            "behind": TimingFeel.BEHIND
        }[timing_feel]
        
        # Apply rule breaks
        rule_break_map = {
            "Non-Resolution (grief, longing)": RuleBreakCode.STRUCTURE_NonResolution,
            "Modal Interchange (bittersweet)": RuleBreakCode.HARMONY_ModalInterchange,
            "Parallel Motion (power)": RuleBreakCode.HARMONY_ParallelMotion,
            "Tempo Fluctuation (intimacy)": RuleBreakCode.RHYTHM_TempoFluctuation,
            "Pitch Imperfection (vulnerability)": RuleBreakCode.PRODUCTION_PitchImperfection,
        }
        params.rule_breaks = [rule_break_map[r] for r in rule_breaks if r in rule_break_map]
        
        # 3. STRUCTURE
        status.text("3/7 Generating song structure...")
        progress.progress(42)
        
        struct_gen = StructureGenerator()
        structure = struct_gen.generate(params, state)
        total_bars = sum(s.bars for s in structure)
        
        # 4. HARMONY
        status.text("4/7 Generating chord progression...")
        progress.progress(56)
        
        harmony = HarmonyEngine(params)
        progression = harmony.generate_progression(total_bars, state.primary_emotion)
        
        # 5. MELODY
        status.text("5/7 Generating melody...")
        progress.progress(70)
        
        melody_engine = MelodyEngine(params, harmony)
        melody = melody_engine.generate(progression, total_bars)
        
        # 6. GROOVE
        status.text("6/7 Generating groove...")
        progress.progress(84)
        
        groove = GrooveEngine(params)
        style = "sparse" if params.density_suggested < 0.8 else "basic"
        drums = groove.generate_drums(total_bars, style)
        
        # 7. MIDI
        status.text("7/7 Building MIDI file...")
        progress.progress(100)
        
        builder = MIDIBuilder(bpm=params.tempo_suggested)
        builder.add_track("melody", melody)
        builder.add_track("drums", drums)
        
        # Save
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_emotion = state.primary_emotion.replace(" ", "_")
        midi_path = out_path / f"{timestamp}_{safe_emotion}.mid"
        builder.save(midi_path)
        
        status.empty()
        progress.empty()
        
        # Success!
        st.success(f"✓ Generated: {midi_path.name}")
        
        # Display results
        st.markdown("---")
        st.header("2️⃣ Results")
        
        # Parameters display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tempo", f"{params.tempo_suggested} BPM")
        with col2:
            st.metric("Key", f"{params.key_signature} {list(params.mode_weights.keys())[0].value if params.mode_weights else 'major'}")
        with col3:
            st.metric("Bars", total_bars)
        with col4:
            st.metric("Feel", params.timing_feel.value)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Notes", len(melody))
        with col2:
            st.metric("Drum Hits", len(drums))
        with col3:
            st.metric("Humanize", f"{params.humanize:.0%}")
        with col4:
            st.metric("Lo-Fi", f"{params.imperfection_level:.0%}")
        
        # Structure display
        st.subheader("Structure")
        structure_str = " → ".join([f"**{s.name}** ({s.bars})" for s in structure])
        st.markdown(structure_str)
        
        # Rule breaks
        if params.rule_breaks:
            st.subheader("Rule Breaks Applied")
            for rb in params.rule_breaks:
                st.markdown(f"- {rb.name}")
        
        # Download
        with open(midi_path, "rb") as f:
            st.download_button(
                "📥 Download MIDI",
                f,
                file_name=midi_path.name,
                mime="audio/midi",
                use_container_width=True
            )
        
        st.info(f"File saved to: {midi_path}")

elif mode == "Critique":
    st.header("🎯 Three-Way Critique")
    st.markdown("*Quality Checker + Interpretation Critic + Arbiter*")
    
    critique_input = st.text_area(
        "Describe your song idea or paste MIDI analysis:",
        placeholder="e.g., 'grief song at 130 BPM in C major with busy drums'",
        height=150
    )
    
    if st.button("Analyze", type="primary"):
        with st.spinner("Running three-way critique..."):
            # Parse input
            engine = InterrogationEngine()
            parsed = engine.parse_vernacular(critique_input)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🔍 Quality Checker")
                st.markdown("*Technical Assessment*")
                
                issues = []
                if "130" in critique_input and any(e in critique_input.lower() for e in ["grief", "sad", "slow"]):
                    issues.append("⚠️ Tempo mismatch: 130 BPM is too fast for grief")
                if "major" in critique_input.lower() and "grief" in critique_input.lower():
                    issues.append("⚠️ Key mismatch: Major key for grief is unusual")
                if "busy" in critique_input.lower() and any(e in critique_input.lower() for e in ["intimate", "sparse", "grief"]):
                    issues.append("⚠️ Density mismatch: Busy drums conflict with intimate feel")
                    
                if issues:
                    for issue in issues:
                        st.warning(issue)
                    st.metric("Score", f"{100 - len(issues)*20}/100")
                else:
                    st.success("No technical issues detected")
                    st.metric("Score", "95/100")
            
            with col2:
                st.subheader("🎭 Interpretation Critic")
                st.markdown("*Intent Matching*")
                
                detected = parsed.get("detected_emotions", [])
                st.write(f"Detected emotions: {', '.join(detected) or 'none'}")
                
                if "grief" in critique_input.lower():
                    st.write("Expected: slow tempo, minor/dorian, behind-beat, sparse")
                    
                cliches = []
                if any(p in critique_input.lower() for p in ["i v vi iv", "1 5 6 4", "four chord"]):
                    cliches.append("Axis progression (I-V-vi-IV) - used in 10,000 songs")
                    
                if cliches:
                    st.warning(f"Cliché detected: {cliches[0]}")
                else:
                    st.success("No obvious clichés")
            
            with col3:
                st.subheader("⚖️ Arbiter")
                st.markdown("*Final Judgment*")
                
                if len(issues) >= 2:
                    st.error("**RETHINK**")
                    st.write("Fundamental mismatch between emotion and parameters. Consider:")
                    st.write("- Slow to 70-85 BPM")
                    st.write("- Switch to minor or dorian")
                    st.write("- Reduce drum density")
                elif len(issues) == 1:
                    st.warning("**REVISE**")
                    st.write("Minor adjustments needed. The core idea works.")
                else:
                    st.success("**PASS**")
                    st.write("Parameters align with emotional intent. Ship it.")

else:  # Analyze
    st.header("🔬 Song Analysis")
    st.markdown("*Break down existing music*")
    
    analysis_input = st.text_area(
        "Describe a song to analyze:",
        placeholder="e.g., 'Radiohead - Creep: G B C Cm progression, dynamics from quiet to loud'",
        height=100
    )
    
    if st.button("Analyze Song", type="primary"):
        with st.spinner("Analyzing..."):
            st.subheader("Analysis Results")
            
            # Detect known progressions
            if any(p in analysis_input.lower() for p in ["g b c cm", "creep", "i iii iv iv"]):
                st.markdown("### Chord Progression")
                st.markdown("**I - III - IV - iv** (The 'Creep' Progression)")
                st.markdown("- I to III: Borrowed from parallel major")
                st.markdown("- IV to iv: Modal interchange (major to minor)")
                st.markdown("- **Rule Break:** `HARMONY_ModalInterchange`")
                st.markdown("- **Emotional Effect:** Bittersweet, yearning")
            
            if "radiohead" in analysis_input.lower():
                st.markdown("### Style Markers")
                st.markdown("- Double-tonic complex (Am and C equally weighted)")
                st.markdown("- Meter ambiguity")
                st.markdown("- Dynamic contrast as structure")
            
            st.markdown("### Suggested Sample Mapping")
            st.markdown("From your library:")
            st.markdown("- Drums: Drum Tornado 2023 (acoustic kit)")
            st.markdown("- Pads: Pads & Strings")

# Footer
st.markdown("---")
st.markdown("*iDAW: Making musicians braver since 2025*")
st.markdown("*'The audience doesn't hear borrowed from Dorian. They hear that part made me cry.'*")
