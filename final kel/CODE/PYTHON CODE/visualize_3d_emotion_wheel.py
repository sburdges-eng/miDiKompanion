#!/usr/bin/env python3
"""
3D Emotion Wheel Visualization

Creates an interactive 3D visualization of the emotion thesaurus using
three dimensions: Valence, Arousal, and Intensity.

Usage:
    python visualize_3d_emotion_wheel.py
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
    print("Warning: plotly not available. Install with: pip install plotly")
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        print("Error: Neither plotly nor matplotlib available.")
        sys.exit(1)


# Color mapping for emotion categories
CATEGORY_COLORS = {
    "joy": "#FFD700",      # Gold
    "sadness": "#4169E1",  # Royal Blue
    "anger": "#DC143C",    # Crimson
    "fear": "#8B008B",     # Dark Magenta
    "surprise": "#FF8C00", # Dark Orange
    "disgust": "#228B22",  # Forest Green
    "trust": "#00CED1",    # Dark Turquoise
    "anticipation": "#FF69B4"  # Hot Pink
}


def load_emotions_from_json(data_dir: Path) -> List[Dict]:
    """Load all emotions from JSON files in the data directory."""
    emotions = []
    emotion_files = [
        "joy.json", "sad.json", "anger.json", 
        "fear.json", "surprise.json", "disgust.json"
    ]
    
    # Also check emotions subdirectory
    emotions_dir = data_dir / "emotions"
    if emotions_dir.exists():
        emotion_files = [f.name for f in emotions_dir.glob("*.json")]
        data_dir = emotions_dir
    
    emotion_id = 0
    
    for filename in emotion_files:
        json_path = data_dir / filename
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            category = data.get("category", filename.replace(".json", ""))
            valence_str = data.get("valence", "positive" if category == "joy" else "negative")
            
            # Convert valence string to float
            if valence_str == "positive":
                base_valence = 0.5
            elif valence_str == "negative":
                base_valence = -0.5
            else:
                base_valence = 0.0
            
            # Process sub_emotions
            sub_emotions = data.get("sub_emotions", {})
            for sub_name, sub_data in sub_emotions.items():
                sub_sub_emotions = sub_data.get("sub_sub_emotions", {})
                for sub_sub_name, sub_sub_data in sub_sub_emotions.items():
                    intensity_tiers = sub_sub_data.get("intensity_tiers", {})
                    
                    for tier_name, emotion_words in intensity_tiers.items():
                        # Extract intensity from tier name (e.g., "1_subtle" -> 0.2)
                        tier_num = tier_name.split("_")[0]
                        try:
                            tier_int = int(tier_num)
                            intensity = (tier_int - 1) / 4.0  # Map 1-5 to 0.0-1.0
                        except:
                            intensity = 0.5
                        
                        # Calculate arousal based on intensity and category
                        arousal = 0.3 + intensity * 0.5
                        if category in ["anger", "fear"]:
                            arousal += 0.2
                        elif category in ["sadness"]:
                            arousal -= 0.2
                        
                        # Adjust valence based on category
                        valence = base_valence
                        if category == "joy":
                            valence = 0.3 + intensity * 0.5
                        elif category == "sadness":
                            valence = -0.3 - intensity * 0.4
                        elif category == "anger":
                            valence = -0.4 - intensity * 0.3
                        elif category == "fear":
                            valence = -0.2 - intensity * 0.3
                        elif category == "surprise":
                            valence = 0.0 + (intensity - 0.5) * 0.4
                        elif category == "disgust":
                            valence = -0.3 - intensity * 0.3
                        
                        # Clamp values
                        valence = max(-1.0, min(1.0, valence))
                        arousal = max(0.0, min(1.0, arousal))
                        intensity = max(0.0, min(1.0, intensity))
                        
                        # Add each emotion word
                        for word in emotion_words:
                            emotions.append({
                                "id": emotion_id,
                                "name": word,
                                "category": category,
                                "valence": valence,
                                "arousal": arousal,
                                "intensity": intensity
                            })
                            emotion_id += 1
                            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return emotions


def load_emotions_from_python_thesaurus() -> Optional[List[Dict]]:
    """Try to load emotions from the Python thesaurus module."""
    try:
        # Try multiple paths
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
                emotions.append({
                    "id": node_id,
                    "name": node.name,
                    "category": node.category.value,
                    "valence": node.valence,
                    "arousal": node.arousal,
                    "intensity": node.intensity
                })
            
            sys.path[:] = original_path
            return emotions
        finally:
            sys.path[:] = original_path
            
    except Exception as e:
        print(f"Could not load from Python thesaurus: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_plotly_visualization(emotions: List[Dict], output_file: str = "emotion_wheel_3d.html"):
    """Create interactive 3D visualization using Plotly."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for this visualization")
    
    # Organize by category
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter for each category with size based on intensity
    for category, cat_emotions in categories.items():
        x = [e["valence"] for e in cat_emotions]
        y = [e["arousal"] for e in cat_emotions]
        z = [e["intensity"] for e in cat_emotions]
        names = [e["name"] for e in cat_emotions]
        sizes = [4 + e["intensity"] * 8 for e in cat_emotions]  # Size based on intensity
        
        color = CATEGORY_COLORS.get(category, "#808080")
        
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            name=category.capitalize(),
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.7,
                line=dict(width=1, color='rgba(0,0,0,0.3)'),
                sizemode='diameter'
            ),
            text=names,
            hovertemplate='<b>%{text}</b><br>' +
                         'Category: ' + category.capitalize() + '<br>' +
                         'Valence: %{x:.2f}<br>' +
                         'Arousal: %{y:.2f}<br>' +
                         'Intensity: %{z:.2f}<extra></extra>',
            customdata=[[e["id"], e["category"]] for e in cat_emotions]
        ))
    
    # Add axis lines for reference
    # X-axis (valence)
    fig.add_trace(go.Scatter3d(
        x=[-1, 1], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='rgba(128,128,128,0.5)', width=2, dash='dash'),
        name='Valence Axis',
        showlegend=False,
        hoverinfo='skip'
    ))
    # Y-axis (arousal)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 1], z=[0, 0],
        mode='lines',
        line=dict(color='rgba(128,128,128,0.5)', width=2, dash='dash'),
        name='Arousal Axis',
        showlegend=False,
        hoverinfo='skip'
    ))
    # Z-axis (intensity)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, 1],
        mode='lines',
        line=dict(color='rgba(128,128,128,0.5)', width=2, dash='dash'),
        name='Intensity Axis',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': '3D Emotion Wheel - Valence, Arousal, Intensity',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='Valence<br>(-1 = Negative, +1 = Positive)',
            yaxis_title='Arousal<br>(0 = Calm, 1 = Excited)',
            zaxis_title='Intensity<br>(0 = Subtle, 1 = Extreme)',
            xaxis=dict(range=[-1.1, 1.1], gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            yaxis=dict(range=[-0.1, 1.1], gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            zaxis=dict(range=[-0.1, 1.1], gridcolor='rgba(128,128,128,0.3)', showbackground=True, backgroundcolor='rgba(230,230,230,0.1)'),
            bgcolor='rgba(250,250,250,1)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='cube'
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1,
            font=dict(size=11)
        ),
        paper_bgcolor='white'
    )
    
    # Save to HTML
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"3D visualization saved to {output_file}")
    print(f"Open it in your browser to interact with the visualization!")
    print(f"  - Rotate: Click and drag")
    print(f"  - Zoom: Scroll wheel")
    print(f"  - Pan: Shift + Click and drag")
    print(f"  - Hover: See emotion details")
    
    # Also show in browser if possible
    try:
        fig.show()
    except:
        pass
    
    return fig


def create_matplotlib_visualization(emotions: List[Dict], output_file: str = "emotion_wheel_3d.png"):
    """Create 3D visualization using Matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this visualization")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Organize by category
    categories = {}
    for emotion in emotions:
        cat = emotion["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(emotion)
    
    # Plot each category
    for category, cat_emotions in categories.items():
        x = [e["valence"] for e in cat_emotions]
        y = [e["arousal"] for e in cat_emotions]
        z = [e["intensity"] for e in cat_emotions]
        
        color = CATEGORY_COLORS.get(category, "#808080")
        
        ax.scatter(x, y, z, c=color, label=category.capitalize(), 
                  alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Labels and title
    ax.set_xlabel('Valence (-1 = Negative, +1 = Positive)', fontsize=12)
    ax.set_ylabel('Arousal (0 = Calm, 1 = Excited)', fontsize=12)
    ax.set_zlabel('Intensity (0 = Subtle, 1 = Extreme)', fontsize=12)
    ax.set_title('3D Emotion Wheel - Valence, Arousal, Intensity', fontsize=16, pad=20)
    
    # Set axis limits
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"3D visualization saved to {output_file}")
    
    # Show
    plt.show()
    
    return fig


def main():
    """Main function to create the visualization."""
    # Determine data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    print("Loading emotion data...")
    
    # Try loading from Python thesaurus first (more accurate)
    emotions = load_emotions_from_python_thesaurus()
    
    # Fallback to JSON files
    if not emotions or len(emotions) == 0:
        print("Loading from JSON files...")
        emotions = load_emotions_from_json(data_dir)
    
    if not emotions or len(emotions) == 0:
        print("Error: No emotions loaded. Check data directory.")
        return
    
    print(f"Loaded {len(emotions)} emotions")
    print(f"Categories: {set(e['category'] for e in emotions)}")
    
    # Create visualization
    if PLOTLY_AVAILABLE:
        print("\nCreating interactive 3D visualization with Plotly...")
        create_plotly_visualization(emotions)
    elif MATPLOTLIB_AVAILABLE:
        print("\nCreating 3D visualization with Matplotlib...")
        create_matplotlib_visualization(emotions)
    else:
        print("Error: No visualization library available")
        return


if __name__ == "__main__":
    main()
