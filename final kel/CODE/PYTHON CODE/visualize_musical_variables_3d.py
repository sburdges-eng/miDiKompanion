#!/usr/bin/env python3
"""
3D Musical Variables Visualization

Creates interactive 3D visualizations of musical parameters derived from emotions.
Shows relationships between tempo, harmonic complexity, articulation, and other musical variables.

Usage:
    python visualize_musical_variables_3d.py
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


# Mapping functions for categorical to numeric
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


def load_emotions_with_musical_data(data_dir: Path) -> List[Dict]:
    """Load emotions with their musical mappings from Python thesaurus."""
    try:
        base_path = Path(__file__).parent
        possible_paths = [
            base_path / "reference" / "python_kelly" / "core",
            base_path / "reference" / "python_kelly",
            base_path,
        ]
        
        original_path = sys.path[:]
        for path in possible_paths:
            if (path / "emotion_thesaurus.py").exists():
                sys.path.insert(0, str(path))
                break
        
        try:
            from emotion_thesaurus import EmotionThesaurus
            
            thesaurus = EmotionThesaurus()
            emotions = []
            
            for node_id, node in thesaurus.nodes.items():
                mapping = node.musical_mapping
                emotions.append({
                    "id": node_id,
                    "name": node.name,
                    "category": node.category.value,
                    "emotion_valence": node.valence,
                    "emotion_arousal": node.arousal,
                    "emotion_intensity": node.intensity,
                    # Musical variables
                    "tempo_modifier": mapping.tempo_modifier,
                    "mode": mapping.mode,
                    "mode_numeric": MODE_MAP.get(mapping.mode, 0.5),
                    "dynamic_min": mapping.dynamic_range[0],
                    "dynamic_max": mapping.dynamic_range[1],
                    "dynamic_avg": (mapping.dynamic_range[0] + mapping.dynamic_range[1]) / 2.0,
                    "harmonic_complexity": mapping.harmonic_complexity,
                    "dissonance_tolerance": mapping.dissonance_tolerance,
                    "rhythm_regularity": mapping.rhythm_regularity,
                    "articulation": mapping.articulation,
                    "articulation_numeric": ARTICULATION_MAP.get(mapping.articulation, 0.5),
                    "register": mapping.register_preference,
                    "register_numeric": REGISTER_MAP.get(mapping.register_preference, 0.5),
                    "space_density": mapping.space_density,
                    "rule_breaks_count": len(mapping.rule_breaks)
                })
            
            sys.path[:] = original_path
            return emotions
        finally:
            sys.path[:] = original_path
            
    except Exception as e:
        print(f"Could not load from Python thesaurus: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_tempo_harmony_articulation_viz(emotions: List[Dict], output_file: str = "musical_variables_tempo_harmony_articulation.html"):
    """Create 3D visualization: Tempo Modifier, Harmonic Complexity, Articulation."""
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    fig = go.Figure()
    
    CATEGORY_COLORS = {
        "joy": "#FFD700", "sadness": "#4169E1", "anger": "#DC143C",
        "fear": "#8B008B", "surprise": "#FF8C00", "disgust": "#228B22"
    }
    
    for category, cat_emotions in categories.items():
        x = [e["tempo_modifier"] for e in cat_emotions]
        y = [e["harmonic_complexity"] for e in cat_emotions]
        z = [e["articulation_numeric"] for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=category.capitalize(),
            marker=dict(size=sizes, color=color, opacity=0.7,
                       line=dict(width=1, color='rgba(0,0,0,0.3)')),
            text=names,
            hovertemplate='<b>%{text}</b><br>' +
                         'Tempo Modifier: %{x:.2f}x<br>' +
                         'Harmonic Complexity: %{y:.2f}<br>' +
                         'Articulation: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': 'Musical Variables: Tempo × Harmony × Articulation', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        scene=dict(
            xaxis_title='Tempo Modifier<br>(0.7x - 2.0x)',
            yaxis_title='Harmonic Complexity<br>(0.0 - 1.0)',
            zaxis_title='Articulation<br>(0=Legato, 1=Staccato)',
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


def create_dynamics_rhythm_space_viz(emotions: List[Dict], output_file: str = "musical_variables_dynamics_rhythm_space.html"):
    """Create 3D visualization: Dynamic Range, Rhythm Regularity, Space Density."""
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    fig = go.Figure()
    
    CATEGORY_COLORS = {
        "joy": "#FFD700", "sadness": "#4169E1", "anger": "#DC143C",
        "fear": "#8B008B", "surprise": "#FF8C00", "disgust": "#228B22"
    }
    
    for category, cat_emotions in categories.items():
        x = [e["dynamic_avg"] for e in cat_emotions]
        y = [e["rhythm_regularity"] for e in cat_emotions]
        z = [e["space_density"] for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=category.capitalize(),
            marker=dict(size=sizes, color=color, opacity=0.7,
                       line=dict(width=1, color='rgba(0,0,0,0.3)')),
            text=names,
            hovertemplate='<b>%{text}</b><br>' +
                         'Avg Velocity: %{x:.0f}<br>' +
                         'Rhythm Regularity: %{y:.2f}<br>' +
                         'Space Density: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': 'Musical Variables: Dynamics × Rhythm × Space', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        scene=dict(
            xaxis_title='Average Velocity<br>(30 - 127)',
            yaxis_title='Rhythm Regularity<br>(0.0 = Free, 1.0 = Strict)',
            zaxis_title='Space Density<br>(0.0 = Dense, 1.0 = Sparse)',
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


def create_dissonance_mode_register_viz(emotions: List[Dict], output_file: str = "musical_variables_dissonance_mode_register.html"):
    """Create 3D visualization: Dissonance Tolerance, Mode, Register."""
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    fig = go.Figure()
    
    CATEGORY_COLORS = {
        "joy": "#FFD700", "sadness": "#4169E1", "anger": "#DC143C",
        "fear": "#8B008B", "surprise": "#FF8C00", "disgust": "#228B22"
    }
    
    for category, cat_emotions in categories.items():
        x = [e["dissonance_tolerance"] for e in cat_emotions]
        y = [e["mode_numeric"] for e in cat_emotions]
        z = [e["register_numeric"] for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        modes = [e["mode"] for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=category.capitalize(),
            marker=dict(size=sizes, color=color, opacity=0.7,
                       line=dict(width=1, color='rgba(0,0,0,0.3)')),
            text=[f"{n}<br>Mode: {m}" for n, m in zip(names, modes)],
            hovertemplate='<b>%{text}</b><br>' +
                         'Dissonance Tolerance: %{x:.2f}<br>' +
                         'Mode (numeric): %{y:.2f}<br>' +
                         'Register: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': 'Musical Variables: Dissonance × Mode × Register', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        scene=dict(
            xaxis_title='Dissonance Tolerance<br>(0.0 = None, 1.0 = High)',
            yaxis_title='Mode<br>(0.0 = Locrian, 1.0 = Major)',
            zaxis_title='Register<br>(0.0 = Low, 1.0 = High)',
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
    
    print("Loading emotions with musical data...")
    emotions = load_emotions_with_musical_data(data_dir)
    
    if not emotions or len(emotions) == 0:
        print("Error: No emotions loaded. Check data directory.")
        return
    
    print(f"Loaded {len(emotions)} emotions with musical mappings")
    print(f"Categories: {set(e['category'] for e in emotions)}")
    
    print("\nCreating musical variable visualizations...")
    create_tempo_harmony_articulation_viz(emotions)
    create_dynamics_rhythm_space_viz(emotions)
    create_dissonance_mode_register_viz(emotions)
    
    print("\nAll musical variable visualizations created!")


if __name__ == "__main__":
    main()
