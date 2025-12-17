#!/usr/bin/env python3
"""
Comprehensive Overlap Visualization

Creates comprehensive visualizations showing overlaps and commonalities between
emotions in both emotional space and musical parameter space.

Usage:
    python visualize_comprehensive_overlap.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from scipy.spatial.distance import cdist
    from scipy.stats import gaussian_kde
    PLOTLY_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    if not PLOTLY_AVAILABLE:
        print("Error: plotly is required. Install with: pip install plotly")
    try:
        from scipy.stats import gaussian_kde
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        print("Warning: scipy not available. Some features will be limited.")
    sys.exit(1) if not PLOTLY_AVAILABLE else None


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

CATEGORY_COLORS = {
    "joy": "#FFD700", "sadness": "#4169E1", "anger": "#DC143C",
    "fear": "#8B008B", "surprise": "#FF8C00", "disgust": "#228B22",
    "trust": "#00CED1", "anticipation": "#FF69B4"
}


def load_emotions_with_musical_data(data_dir: Path) -> List[Dict]:
    """Load emotions with their musical mappings."""
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
                emotions.append({
                    "id": node_id,
                    "name": node.name,
                    "category": node.category.value,
                    "emotion_valence": node.valence,
                    "emotion_arousal": node.arousal,
                    "emotion_intensity": node.intensity,
                    "tempo_modifier": mapping.tempo_modifier,
                    "mode": mapping.mode,
                    "mode_numeric": MODE_MAP.get(mapping.mode, 0.5),
                    "dynamic_avg": (mapping.dynamic_range[0] + mapping.dynamic_range[1]) / 2.0,
                    "harmonic_complexity": mapping.harmonic_complexity,
                    "dissonance_tolerance": mapping.dissonance_tolerance,
                    "rhythm_regularity": mapping.rhythm_regularity,
                    "articulation_numeric": ARTICULATION_MAP.get(mapping.articulation, 0.5),
                    "register_numeric": REGISTER_MAP.get(mapping.register_preference, 0.5),
                    "space_density": mapping.space_density,
                })
            
            sys.path[:] = original_path
            return emotions
        finally:
            sys.path[:] = original_path
            
    except Exception as e:
        print(f"Error loading emotions: {e}")
        import traceback
        traceback.print_exc()
        return []


def calculate_musical_distance(e1: Dict, e2: Dict) -> float:
    """Calculate distance in musical parameter space."""
    features = [
        e1["tempo_modifier"], e1["harmonic_complexity"], e1["dissonance_tolerance"],
        e1["rhythm_regularity"], e1["articulation_numeric"], e1["register_numeric"],
        e1["space_density"], e1["dynamic_avg"] / 127.0  # Normalize
    ]
    features2 = [
        e2["tempo_modifier"], e2["harmonic_complexity"], e2["dissonance_tolerance"],
        e2["rhythm_regularity"], e2["articulation_numeric"], e2["register_numeric"],
        e2["space_density"], e2["dynamic_avg"] / 127.0
    ]
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(features, features2)))


def find_musical_clusters(emotions: List[Dict], threshold: float = 0.3) -> Dict[int, List[int]]:
    """Find emotions that cluster together in musical space."""
    clusters = {}
    cluster_id = 0
    
    for i, e1 in enumerate(emotions):
        if i in clusters:
            continue
        
        cluster = [i]
        for j, e2 in enumerate(emotions[i+1:], start=i+1):
            if j in clusters:
                continue
            dist = calculate_musical_distance(e1, e2)
            if dist < threshold:
                cluster.append(j)
                clusters[j] = cluster_id
        
        if len(cluster) > 1:
            clusters[i] = cluster_id
            cluster_id += 1
    
    # Group by cluster
    cluster_groups = defaultdict(list)
    for idx, cid in clusters.items():
        cluster_groups[cid].append(idx)
    
    return dict(cluster_groups)


def derive_musical_parameters(emotion: Dict) -> Dict:
    """Derive additional musical parameters from emotion data."""
    valence = emotion.get("emotion_valence", 0.0)
    arousal = emotion.get("emotion_arousal", 0.0)
    intensity = emotion.get("emotion_intensity", 0.5)
    category = emotion.get("category", "neutral")
    
    # Melodic Contour
    CONTOUR_MAP = {
        "ascending": 1.0, "spiral_up": 0.9, "arch": 0.7,
        "wave": 0.5, "static": 0.3, "inverse_arch": 0.2,
        "descending": 0.0, "spiral_down": 0.1, "jagged": 0.4, "collapse": 0.0
    }
    
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
    
    # Texture
    TEXTURE_MAP = {"monophonic": 0.0, "homophonic": 0.5, "polyphonic": 1.0}
    if intensity > 0.7 and arousal > 0.6:
        texture = "polyphonic"
    elif intensity < 0.3 or arousal < 0.3:
        texture = "monophonic"
    else:
        texture = "homophonic"
    
    # Key Center
    KEY_CENTER_MAP = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
    }
    if valence < -0.5:
        key_center = "Eb"
    elif valence < -0.2:
        key_center = "Bb"
    elif valence < 0.2:
        key_center = "C"
    elif valence < 0.5:
        key_center = "G"
    else:
        key_center = "D"
    
    # Swing factor
    if category in ["joy", "surprise"]:
        swing_factor = 0.3 + arousal * 0.3
    elif category in ["sadness", "fear"]:
        swing_factor = 0.0
    elif category == "anger":
        swing_factor = 0.1
    else:
        swing_factor = arousal * 0.2
    
    # Instrument brightness
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
    
    return {
        "contour_numeric": CONTOUR_MAP.get(contour, 0.5),
        "texture_numeric": TEXTURE_MAP.get(texture, 0.5),
        "key_center_numeric": KEY_CENTER_MAP.get(key_center, 0) / 11.0,
        "swing_factor": swing_factor,
        "instrument_brightness": instrument_brightness,
        "dynamic_spread": emotion.get("dynamic_max", 100) - emotion.get("dynamic_min", 50)
    }


def create_comprehensive_overlap_viz(emotions: List[Dict], output_file: str = "comprehensive_overlap.html"):
    """Create comprehensive visualization showing all overlaps."""
    
    # Add derived parameters to emotions
    for emotion in emotions:
        derived = derive_musical_parameters(emotion)
        emotion.update(derived)
    
    # Create subplots: 2x3 grid for more views
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=(
            "Emotion Space: Valence × Arousal × Intensity",
            "Musical Space: Tempo × Harmony × Articulation",
            "Timbre × Contour × Texture",
            "Emotion → Musical (Valence vs Tempo)",
            "Musical Clusters (Harmony vs Dissonance)",
            "Intensity × Swing × Key Center"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Organize by category
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    # 1. Emotion Space 3D - WITH INTENSITY AS 3RD DIMENSION (top left)
    for category, cat_emotions in categories.items():
        x = [e["emotion_valence"] for e in cat_emotions]
        y = [e["emotion_arousal"] for e in cat_emotions]
        z = [e["emotion_intensity"] for e in cat_emotions]  # 3RD DIMENSION!
        names = [e["name"] for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                name=category.capitalize(),
                marker=dict(size=sizes, color=color, opacity=0.6,
                           line=dict(width=1, color='rgba(0,0,0,0.3)')),
                text=names,
                hovertemplate='<b>%{text}</b><br>Valence: %{x:.2f}<br>Arousal: %{y:.2f}<br>Intensity: %{z:.2f}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Musical Space 3D (top middle)
    for category, cat_emotions in categories.items():
        x = [e["tempo_modifier"] for e in cat_emotions]
        y = [e["harmonic_complexity"] for e in cat_emotions]
        z = [e["articulation_numeric"] for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                name=category.capitalize() + " (Musical)",
                marker=dict(size=sizes, color=color, opacity=0.6,
                           line=dict(width=1, color='rgba(0,0,0,0.3)')),
                text=names,
                hovertemplate='<b>%{text}</b><br>Tempo: %{x:.2f}x<br>Harmony: %{y:.2f}<br>Articulation: %{z:.2f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Timbre × Contour × Texture 3D (top right)
    for category, cat_emotions in categories.items():
        x = [e.get("instrument_brightness", 0.5) for e in cat_emotions]
        y = [e.get("contour_numeric", 0.5) for e in cat_emotions]
        z = [e.get("texture_numeric", 0.5) for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        sizes = [4 + e["emotion_intensity"] * 8 for e in cat_emotions]
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                name=category.capitalize() + " (Timbre)",
                marker=dict(size=sizes, color=color, opacity=0.6,
                           line=dict(width=1, color='rgba(0,0,0,0.3)')),
                text=names,
                hovertemplate='<b>%{text}</b><br>Brightness: %{x:.2f}<br>Contour: %{y:.2f}<br>Texture: %{z:.2f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=3
        )
    
    # 4. Emotion → Musical Mapping 2D (bottom left)
    for category, cat_emotions in categories.items():
        x = [e["emotion_valence"] for e in cat_emotions]
        y = [e["tempo_modifier"] for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='markers',
                name=category.capitalize(),
                marker=dict(size=8, color=color, opacity=0.7,
                           line=dict(width=1, color='rgba(0,0,0,0.3)')),
                text=names,
                hovertemplate='<b>%{text}</b><br>Valence: %{x:.2f}<br>Tempo: %{y:.2f}x<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Add density contours if scipy available
    if SCIPY_AVAILABLE:
        try:
            all_valence = [e["emotion_valence"] for e in emotions]
            all_tempo = [e["tempo_modifier"] for e in emotions]
            
            if len(all_valence) > 2:
                # Create density plot
                xy = np.vstack([all_valence, all_tempo])
                kde = gaussian_kde(xy)
                
                x_range = np.linspace(min(all_valence), max(all_valence), 50)
                y_range = np.linspace(min(all_tempo), max(all_tempo), 50)
                X, Y = np.meshgrid(x_range, y_range)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde(positions).T, X.shape)
                
                fig.add_trace(
                    go.Contour(
                        x=x_range, y=y_range, z=Z,
                        colorscale='Blues', showscale=False,
                        opacity=0.3, contours=dict(showlines=False),
                        hoverinfo='skip'
                    ),
                    row=2, col=1
                )
        except:
            pass
    
    # 4. Musical Clusters 2D (bottom right)
    clusters = find_musical_clusters(emotions, threshold=0.4)
    
    # Plot all emotions
    all_harmony = [e["harmonic_complexity"] for e in emotions]
    all_dissonance = [e["dissonance_tolerance"] for e in emotions]
    all_names = [e["name"] for e in emotions]
    all_categories = [e["category"] for e in emotions]
    
    # Color by cluster membership
    cluster_colors = {}
    cluster_id = 0
    for cluster_indices in clusters.values():
        if len(cluster_indices) > 1:
            cluster_colors[cluster_id] = px.colors.qualitative.Set3[cluster_id % len(px.colors.qualitative.Set3)]
            cluster_id += 1
    
    colors = []
    cluster_labels = []
    for i, emotion in enumerate(emotions):
        in_cluster = False
        for cid, cluster_indices in clusters.items():
            if i in cluster_indices and len(cluster_indices) > 1:
                colors.append(cluster_colors.get(cid, "#CCCCCC"))
                cluster_labels.append(f"Cluster {cid}")
                in_cluster = True
                break
        if not in_cluster:
            colors.append(CATEGORY_COLORS.get(emotion["category"], "#808080"))
            cluster_labels.append("No cluster")
    
    fig.add_trace(
        go.Scatter(
            x=all_harmony, y=all_dissonance, mode='markers',
            name="Musical Clusters",
            marker=dict(size=10, color=colors, opacity=0.7,
                       line=dict(width=1, color='rgba(0,0,0,0.3)')),
            text=[f"{n}<br>{l}" for n, l in zip(all_names, cluster_labels)],
            hovertemplate='<b>%{text}</b><br>Harmony: %{x:.2f}<br>Dissonance: %{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Highlight clusters with circles
    for cid, cluster_indices in clusters.items():
        if len(cluster_indices) > 1:
            cluster_emotions = [emotions[i] for i in cluster_indices]
            center_harmony = np.mean([e["harmonic_complexity"] for e in cluster_emotions])
            center_dissonance = np.mean([e["dissonance_tolerance"] for e in cluster_emotions])
            max_dist = max([np.sqrt((e["harmonic_complexity"] - center_harmony)**2 + 
                                   (e["dissonance_tolerance"] - center_dissonance)**2) 
                           for e in cluster_emotions])
            
            fig.add_trace(
                go.Scatter(
                    x=[center_harmony], y=[center_dissonance],
                    mode='markers',
                    marker=dict(size=max_dist*200, color=cluster_colors.get(cid, "#CCCCCC"),
                               opacity=0.2, line=dict(width=2, color=cluster_colors.get(cid, "#000000"))),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=2
            )
    
    # 5. Intensity × Swing × Key Center 2D (bottom right) - NEW!
    for category, cat_emotions in categories.items():
        x = [e["emotion_intensity"] for e in cat_emotions]
        y = [e.get("swing_factor", 0.0) for e in cat_emotions]
        z = [e.get("key_center_numeric", 0.5) for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='markers',
                name=category.capitalize() + " (Intensity)",
                marker=dict(size=10, color=color, opacity=0.7,
                           line=dict(width=1, color='rgba(0,0,0,0.3)')),
                text=names,
                hovertemplate='<b>%{text}</b><br>Intensity: %{x:.2f}<br>Swing: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=3
        )
    
    # Update axes
    fig.update_xaxes(title_text="Valence", row=2, col=1)
    fig.update_yaxes(title_text="Tempo Modifier", row=2, col=1)
    fig.update_xaxes(title_text="Harmonic Complexity", row=2, col=2)
    fig.update_yaxes(title_text="Dissonance Tolerance", row=2, col=2)
    fig.update_xaxes(title_text="Intensity", row=2, col=3)
    fig.update_yaxes(title_text="Swing Factor", row=2, col=3)
    
    # Update 3D scenes
    fig.update_scenes(
        xaxis_title="Valence", yaxis_title="Arousal", zaxis_title="Intensity (3rd Dimension!)",
        row=1, col=1
    )
    fig.update_scenes(
        xaxis_title="Tempo Modifier", yaxis_title="Harmonic Complexity", zaxis_title="Articulation",
        row=1, col=2
    )
    fig.update_scenes(
        xaxis_title="Instrument Brightness", yaxis_title="Melodic Contour", zaxis_title="Texture",
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title={'text': 'Comprehensive Emotion-Musical Overlap Analysis (All Variables + Intensity as 3rd Dimension)', 
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 24}},
        height=1600,
        width=2400,
        margin=dict(l=20, r=20, t=120, b=20),
        legend=dict(x=1.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.3)', borderwidth=1)
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Comprehensive overlap visualization saved to {output_file}")
    return fig


def create_parallel_coordinates_viz(emotions: List[Dict], output_file: str = "parallel_coordinates.html"):
    """Create parallel coordinates plot showing all variables."""
    
    # Prepare data
    categories = [e["category"] for e in emotions]
    names = [e["name"] for e in emotions]
    
    dimensions = [
        dict(label="Valence", values=[e["emotion_valence"] for e in emotions],
             range=[-1.1, 1.1]),
        dict(label="Arousal", values=[e["emotion_arousal"] for e in emotions],
             range=[-0.1, 1.1]),
        dict(label="Intensity", values=[e["emotion_intensity"] for e in emotions],
             range=[-0.1, 1.1]),
        dict(label="Tempo Mod", values=[e["tempo_modifier"] for e in emotions],
             range=[0.5, 2.0]),
        dict(label="Harmony", values=[e["harmonic_complexity"] for e in emotions],
             range=[0, 1]),
        dict(label="Dissonance", values=[e["dissonance_tolerance"] for e in emotions],
             range=[0, 1]),
        dict(label="Rhythm Reg", values=[e["rhythm_regularity"] for e in emotions],
             range=[0, 1]),
        dict(label="Articulation", values=[e["articulation_numeric"] for e in emotions],
             range=[0, 1]),
        dict(label="Space Density", values=[e["space_density"] for e in emotions],
             range=[0, 1]),
    ]
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=[CATEGORY_COLORS.get(cat, "#808080") for cat in categories],
            colorscale=None,
            showscale=False,
            cmin=0,
            cmax=1
        ),
        dimensions=dimensions,
        labelangle=-45,
        labelside="bottom"
    ))
    
    fig.update_layout(
        title={'text': 'Parallel Coordinates: All Emotion & Musical Variables', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        height=600,
        width=1600,
        margin=dict(l=20, r=20, t=60, b=100)
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Parallel coordinates visualization saved to {output_file}")
    return fig


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    print("Loading emotions with musical data...")
    emotions = load_emotions_with_musical_data(data_dir)
    
    if not emotions or len(emotions) == 0:
        print("Error: No emotions loaded.")
        return
    
    print(f"Loaded {len(emotions)} emotions")
    
    print("\nCreating comprehensive overlap visualization...")
    create_comprehensive_overlap_viz(emotions)
    
    print("\nCreating parallel coordinates visualization...")
    create_parallel_coordinates_viz(emotions)
    
    print("\nAll comprehensive visualizations created!")


if __name__ == "__main__":
    main()
