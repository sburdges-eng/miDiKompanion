#!/usr/bin/env python3
"""
Complete Musical Variables Visualization

Creates comprehensive 3D visualizations including ALL musical parameters:
- Timbre/Instrumentation
- Melodic Contour
- Texture (mono/homo/polyphonic)
- Key Center/Root (actual pitch)
- Chord Progression Patterns
- Velocity Range (min/max spread)
- Time Signature
- Swing/Groove factor

Also includes Trust and Anticipation emotions from Plutchik's wheel.

Usage:
    python visualize_complete_musical_variables.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Error: plotly is required. Install with: pip install plotly")
    sys.exit(1)


# Mapping functions
MODE_MAP = {
    "major": 1.0, "lydian": 0.9, "mixolydian": 0.7,
    "dorian": 0.5, "aeolian": 0.3, "minor": 0.3,
    "harmonic_minor": 0.2, "phrygian": 0.1, "locrian": 0.0
}

ARTICULATION_MAP = {
    "legato": 0.0, "tenuto": 0.33, "marcato": 0.67, "staccato": 1.0
}

REGISTER_MAP = {
    "low": 0.0, "mid": 0.5, "high": 1.0, "full": 0.75
}

CONTOUR_MAP = {
    "ascending": 1.0, "spiral_up": 0.9, "arch": 0.7,
    "wave": 0.5, "static": 0.3, "inverse_arch": 0.2,
    "descending": 0.0, "spiral_down": 0.1, "jagged": 0.4, "collapse": 0.0
}

TEXTURE_MAP = {
    "monophonic": 0.0, "homophonic": 0.5, "polyphonic": 1.0
}

KEY_CENTER_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}

CATEGORY_COLORS = {
    "joy": "#FFD700", "sadness": "#4169E1", "anger": "#DC143C",
    "fear": "#8B008B", "surprise": "#FF8C00", "disgust": "#228B22",
    "trust": "#00CED1", "anticipation": "#FF69B4"
}


def derive_musical_parameters(emotion: Dict) -> Dict:
    """Derive additional musical parameters from emotion data."""
    valence = emotion.get("emotion_valence", 0.0)
    arousal = emotion.get("emotion_arousal", 0.0)
    intensity = emotion.get("emotion_intensity", 0.5)
    category = emotion.get("category", "neutral")
    
    # Melodic Contour based on valence and arousal
    if valence > 0.5 and arousal > 0.5:
        contour = "ascending"
    elif valence < -0.3 and arousal < 0.5:
        contour = "descending"
    elif arousal > 0.7:
        contour = "wave"
    elif valence > 0:
        contour = "arch"
    else:
        contour = "inverse_arch"
    
    # Texture based on intensity and arousal
    if intensity > 0.7 and arousal > 0.6:
        texture = "polyphonic"
    elif intensity < 0.3 or arousal < 0.3:
        texture = "monophonic"
    else:
        texture = "homophonic"
    
    # Key Center based on valence
    # Negative valence -> flat keys (Eb, Bb, F)
    # Positive valence -> sharp keys (G, D, A)
    # Neutral -> C
    if valence < -0.5:
        key_center = "Eb"  # Darker
    elif valence < -0.2:
        key_center = "Bb"
    elif valence < 0.2:
        key_center = "C"  # Neutral
    elif valence < 0.5:
        key_center = "G"
    else:
        key_center = "D"  # Brighter
    
    # Time Signature based on arousal
    if arousal > 0.7:
        time_sig_numerator = 4
        time_sig_denominator = 4
    elif arousal > 0.4:
        time_sig_numerator = 4
        time_sig_denominator = 4
    else:
        time_sig_numerator = 3
        time_sig_denominator = 4  # Waltz feel for calm
    
    # Swing/Groove factor based on category
    if category in ["joy", "surprise"]:
        swing_factor = 0.3 + arousal * 0.3  # Some swing
    elif category in ["sadness", "fear"]:
        swing_factor = 0.0  # Straight
    elif category == "anger":
        swing_factor = 0.1  # Minimal
    else:
        swing_factor = arousal * 0.2
    
    # Instrument brightness based on valence and category
    if category == "joy":
        instrument_brightness = 0.8
    elif category == "sadness":
        instrument_brightness = 0.3
    elif category == "anger":
        instrument_brightness = 0.6
    elif category == "fear":
        instrument_brightness = 0.4
    else:
        instrument_brightness = 0.5 + valence * 0.3
    
    # Chord progression complexity (simplified as numeric)
    if intensity > 0.7:
        progression_complexity = 0.8
    elif intensity > 0.4:
        progression_complexity = 0.5
    else:
        progression_complexity = 0.3
    
    return {
        "contour": contour,
        "contour_numeric": CONTOUR_MAP.get(contour, 0.5),
        "texture": texture,
        "texture_numeric": TEXTURE_MAP.get(texture, 0.5),
        "key_center": key_center,
        "key_center_numeric": KEY_CENTER_MAP.get(key_center, 0) / 11.0,  # Normalize 0-1
        "time_sig_numerator": time_sig_numerator,
        "time_sig_denominator": time_sig_denominator,
        "time_sig_numeric": time_sig_numerator / 8.0,  # Normalize (assuming max 8/4)
        "swing_factor": swing_factor,
        "instrument_brightness": instrument_brightness,
        "progression_complexity": progression_complexity
    }


def load_emotions_with_complete_musical_data(data_dir: Path) -> List[Dict]:
    """Load emotions with complete musical mappings including derived parameters."""
    try:
        base_path = Path(__file__).parent
        possible_paths = [
            base_path / "reference" / "python_kelly" / "core",
            base_path / "reference" / "python_kelly",
            base_path,
        ]
        
        original_path = sys.path[:]
        for path in possible_paths:
            if (path / "kelly.thesaurus.py").exists():
                sys.path.insert(0, str(path))
                break
        
        try:
            from kelly.thesaurus import EmotionThesaurus
            
            thesaurus = EmotionThesaurus()
            emotions = []
            
            for node_id, node in thesaurus.nodes.items():
                mapping = node.musical_mapping
                
                emotion_data = {
                    "id": node_id,
                    "name": node.name,
                    "category": node.category.value,
                    "emotion_valence": node.valence,
                    "emotion_arousal": node.arousal,
                    "emotion_intensity": node.intensity,  # 3rd dimension!
                    "tempo_modifier": mapping.tempo_modifier,
                    "mode": mapping.mode,
                    "mode_numeric": MODE_MAP.get(mapping.mode, 0.5),
                    "dynamic_min": mapping.dynamic_range[0],
                    "dynamic_max": mapping.dynamic_range[1],
                    "dynamic_avg": (mapping.dynamic_range[0] + mapping.dynamic_range[1]) / 2.0,
                    "dynamic_spread": mapping.dynamic_range[1] - mapping.dynamic_range[0],  # NEW
                    "harmonic_complexity": mapping.harmonic_complexity,
                    "dissonance_tolerance": mapping.dissonance_tolerance,
                    "rhythm_regularity": mapping.rhythm_regularity,
                    "articulation": mapping.articulation,
                    "articulation_numeric": ARTICULATION_MAP.get(mapping.articulation, 0.5),
                    "register": mapping.register_preference,
                    "register_numeric": REGISTER_MAP.get(mapping.register_preference, 0.5),
                    "space_density": mapping.space_density,
                }
                
                # Add derived parameters
                derived = derive_musical_parameters(emotion_data)
                emotion_data.update(derived)
                
                emotions.append(emotion_data)
            
            sys.path[:] = original_path
            return emotions
        finally:
            sys.path[:] = original_path
            
    except Exception as e:
        print(f"Error loading emotions: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_timbre_contour_texture_viz(emotions: List[Dict], output_file: str = "musical_timbre_contour_texture.html"):
    """3D: Instrument Brightness × Melodic Contour × Texture."""
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    fig = go.Figure()
    
    for category, cat_emotions in categories.items():
        x = [e["instrument_brightness"] for e in cat_emotions]
        y = [e["contour_numeric"] for e in cat_emotions]
        z = [e["texture_numeric"] for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            name=category.capitalize(),
            marker=dict(size=sizes, color=color, opacity=0.7,
                       line=dict(width=1, color='rgba(0,0,0,0.3)')),
            text=names,
            hovertemplate='<b>%{text}</b><br>' +
                         'Brightness: %{x:.2f}<br>' +
                         'Contour: %{y:.2f}<br>' +
                         'Texture: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': 'Timbre × Contour × Texture', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        scene=dict(
            xaxis_title='Instrument Brightness<br>(0 = Dark, 1 = Bright)',
            yaxis_title='Melodic Contour<br>(0 = Descending, 1 = Ascending)',
            zaxis_title='Texture<br>(0 = Mono, 1 = Poly)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            zaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            bgcolor='rgba(250,250,250,1)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)),
            aspectmode='cube'
        ),
        width=1400, height=900,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.3)', borderwidth=1)
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Saved: {output_file}")
    return fig


def create_key_swing_progression_viz(emotions: List[Dict], output_file: str = "musical_key_swing_progression.html"):
    """3D: Key Center × Swing Factor × Progression Complexity."""
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    fig = go.Figure()
    
    for category, cat_emotions in categories.items():
        x = [e["key_center_numeric"] for e in cat_emotions]
        y = [e["swing_factor"] for e in cat_emotions]
        z = [e["progression_complexity"] for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        keys = [e["key_center"] for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            name=category.capitalize(),
            marker=dict(size=sizes, color=color, opacity=0.7,
                       line=dict(width=1, color='rgba(0,0,0,0.3)')),
            text=[f"{n}<br>Key: {k}" for n, k in zip(names, keys)],
            hovertemplate='<b>%{text}</b><br>' +
                         'Key Center: %{x:.2f}<br>' +
                         'Swing: %{y:.2f}<br>' +
                         'Progression: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': 'Key Center × Swing × Progression', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        scene=dict(
            xaxis_title='Key Center<br>(0 = C, 1 = B)',
            yaxis_title='Swing/Groove Factor<br>(0 = Straight, 1 = Heavy Swing)',
            zaxis_title='Progression Complexity<br>(0 = Simple, 1 = Complex)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            zaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            bgcolor='rgba(250,250,250,1)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)),
            aspectmode='cube'
        ),
        width=1400, height=900,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.3)', borderwidth=1)
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Saved: {output_file}")
    return fig


def create_velocity_spread_time_viz(emotions: List[Dict], output_file: str = "musical_velocity_time.html"):
    """3D: Velocity Spread × Time Signature × Intensity (3rd dimension!)."""
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    fig = go.Figure()
    
    for category, cat_emotions in categories.items():
        x = [e["dynamic_spread"] for e in cat_emotions]
        y = [e["time_sig_numeric"] for e in cat_emotions]
        z = [e["emotion_intensity"] for e in cat_emotions]  # 3RD DIMENSION!
        names = [e["name"] for e in cat_emotions]
        time_sigs = [f"{e['time_sig_numerator']}/{e['time_sig_denominator']}" for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            name=category.capitalize(),
            marker=dict(size=sizes, color=color, opacity=0.7,
                       line=dict(width=1, color='rgba(0,0,0,0.3)')),
            text=[f"{n}<br>{ts}" for n, ts in zip(names, time_sigs)],
            hovertemplate='<b>%{text}</b><br>' +
                         'Velocity Spread: %{x:.0f}<br>' +
                         'Time Sig: %{y:.2f}<br>' +
                         'Intensity: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': 'Velocity Spread × Time Signature × Intensity (3rd Dimension)', 
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        scene=dict(
            xaxis_title='Velocity Spread<br>(Dynamic Range)',
            yaxis_title='Time Signature<br>(Normalized)',
            zaxis_title='Intensity<br>(0 = Subtle, 1 = Extreme)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            zaxis=dict(gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            bgcolor='rgba(250,250,250,1)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)),
            aspectmode='cube'
        ),
        width=1400, height=900,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.3)', borderwidth=1)
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Saved: {output_file}")
    return fig


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    print("Loading emotions with complete musical data...")
    emotions = load_emotions_with_complete_musical_data(data_dir)
    
    if not emotions or len(emotions) == 0:
        print("Error: No emotions loaded.")
        return
    
    print(f"Loaded {len(emotions)} emotions with complete musical mappings")
    categories = set(e['category'] for e in emotions)
    print(f"Categories: {categories}")
    print(f"Includes Trust: {'trust' in categories}")
    print(f"Includes Anticipation: {'anticipation' in categories}")
    
    print("\nCreating complete musical variable visualizations...")
    create_timbre_contour_texture_viz(emotions)
    create_key_swing_progression_viz(emotions)
    create_velocity_spread_time_viz(emotions)
    
    print("\nAll complete musical variable visualizations created!")


if __name__ == "__main__":
    main()
