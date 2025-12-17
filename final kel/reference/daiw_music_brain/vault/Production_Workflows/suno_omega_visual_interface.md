# Omega CEFE Visual Interface Plan: Hybrid Mode

**Tags:** `#omega-visual-interface` `#matplotlib` `#dash-dashboard` `#real-time-visualization` `#eeg-visualization` `#ai-architecture` `#ai-priority`

**Last Updated:** 2025-01-27

**Related:** [[suno_omega_simulation_prototype]] | [[suno_omega_synthesis_v5]] | [[suno_omega_resonance_protocol]]

---

## Overview

The Omega CEFE Visual Interface provides real-time visualization of emotional states, EEG data, and system coherence through two complementary modes:

1. **Matplotlib Live Window** - Lightweight embedded graph for local performance and live debugging
2. **Dash Web Dashboard** - Full interactive UI served via localhost (or network) for remote or immersive viewing

---

## Visual Interface Modes

| Mode | Description | Target |
|------|-------------|--------|
| **1️⃣ Matplotlib Live Window** | Lightweight embedded graph showing Valence, Arousal, Coherence | Local performance + live debug |
| **2️⃣ Dash Web Dashboard** | Full interactive UI served via localhost (or network) | Remote or immersive viewing, integrates with DAWs or visual software |

---

## 1. Local Window (Matplotlib)

### Features

**Real-time 3-line graph:**
- 💙 **Valence** (Emotional Polarity) - Blue line
- ❤️ **Arousal** (Energy/Excitement) - Red line
- 💛 **Coherence** (System Stability) - Yellow line

**Additional Features:**
- Auto-updates every 0.5 seconds
- Frame color reflects dominant EEG band (Alpha/Beta/Theta/Gamma)
- Minimal CPU overhead for real-time performance
- Embedded in main process for low-latency updates

### Implementation Example

```python
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class MatplotlibVisualizer:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.history_v = deque(maxlen=max_points)
        self.history_a = deque(maxlen=max_points)
        self.history_c = deque(maxlen=max_points)
        self.eeg_bands = {'alpha': 0, 'beta': 0, 'theta': 0, 'gamma': 0}
        
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle('OMEGA CEFE - Emotional Resonance Monitor', fontsize=14)
        
    def update(self, valence, arousal, coherence, eeg_data=None):
        self.history_v.append(valence)
        self.history_a.append(arousal)
        self.history_c.append(coherence)
        
        if eeg_data:
            self.eeg_bands = eeg_data
        
        # Determine dominant EEG band for frame color
        dominant_band = max(self.eeg_bands, key=self.eeg_bands.get)
        color_map = {
            'alpha': '#4A90E2',  # Blue
            'beta': '#E24A4A',   # Red
            'theta': '#E2B84A',  # Yellow
            'gamma': '#4AE2B8'   # Cyan
        }
        frame_color = color_map.get(dominant_band, '#FFFFFF')
        
        self.ax.clear()
        self.ax.set_facecolor(frame_color)
        self.ax.set_alpha(0.1)
        
        x = np.arange(len(self.history_v))
        self.ax.plot(x, list(self.history_v), 'b-', label='Valence', linewidth=2)
        self.ax.plot(x, list(self.history_a), 'r-', label='Arousal', linewidth=2)
        self.ax.plot(x, list(self.history_c), 'y-', label='Coherence', linewidth=2)
        
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlabel('Time Steps')
        self.ax.set_ylabel('Emotional State')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        # Update frame color
        self.fig.patch.set_facecolor(frame_color)
        self.fig.patch.set_alpha(0.2)
        
        plt.draw()
        plt.pause(0.05)
```

---

## 2. Web Dashboard (Dash/Plotly)

### Features

**Interactive Panel Components:**

1. **Live EEG Spectral Bands**
   - Real-time frequency domain visualization
   - Alpha, Beta, Theta, Gamma power levels
   - Spectral waterfall or bar chart

2. **Dynamic Emotion Flower (3D Radar Visualization)**
   - Valence, Arousal, Dominance plotted on radial axes
   - Real-time updates with smooth transitions
   - Color-coded by emotional state

3. **MIDI/OSC Output Monitors**
   - Current tempo, mode, key
   - OSC message log
   - MIDI note/CC activity

4. **Optional Ambient Visualizer Canvas**
   - Music + mood fusion visualization
   - Particle effects or waveform displays
   - Synchronized with audio output

5. **Control Toggles**
   - Sim Mode on/off
   - Pause/Resume
   - Visual Theme selector
   - Sensitivity adjustments

### Implementation Example

```python
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
from collections import deque
import numpy as np

class DashDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.history_v = deque(maxlen=100)
        self.history_a = deque(maxlen=100)
        self.history_c = deque(maxlen=100)
        self.eeg_data = {'alpha': 0, 'beta': 0, 'theta': 0, 'gamma': 0}
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("OMEGA CEFE - Conscious Emotional Feedback Engine", 
                   style={'textAlign': 'center', 'color': '#4A90E2'}),
            
            # Control Panel
            html.Div([
                html.Button('Pause', id='pause-btn', n_clicks=0),
                html.Button('Sim Mode', id='sim-btn', n_clicks=0),
                dcc.Dropdown(
                    id='theme-selector',
                    options=[
                        {'label': 'Dark', 'value': 'dark'},
                        {'label': 'Light', 'value': 'light'},
                        {'label': 'Neon', 'value': 'neon'}
                    ],
                    value='dark'
                )
            ], style={'padding': '20px'}),
            
            # Main Graphs Row
            html.Div([
                # Emotion Timeline
                dcc.Graph(id='emotion-timeline'),
                
                # EEG Spectral Bands
                dcc.Graph(id='eeg-spectrum'),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            
            # Emotion Flower (Radar Chart)
            dcc.Graph(id='emotion-flower'),
            
            # MIDI/OSC Monitor
            html.Div([
                html.H3("MIDI/OSC Output Monitor"),
                html.Div(id='midi-osc-log', style={
                    'background': '#1e1e1e',
                    'color': '#fff',
                    'padding': '10px',
                    'height': '200px',
                    'overflow-y': 'scroll',
                    'font-family': 'monospace'
                })
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=500,  # Update every 500ms
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('emotion-timeline', 'figure'),
             Output('eeg-spectrum', 'figure'),
             Output('emotion-flower', 'figure'),
             Output('midi-osc-log', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Emotion Timeline
            timeline_fig = go.Figure()
            timeline_fig.add_trace(go.Scatter(
                y=list(self.history_v),
                mode='lines',
                name='Valence',
                line=dict(color='#4A90E2', width=2)
            ))
            timeline_fig.add_trace(go.Scatter(
                y=list(self.history_a),
                mode='lines',
                name='Arousal',
                line=dict(color='#E24A4A', width=2)
            ))
            timeline_fig.add_trace(go.Scatter(
                y=list(self.history_c),
                mode='lines',
                name='Coherence',
                line=dict(color='#E2B84A', width=2)
            ))
            timeline_fig.update_layout(
                title='Emotional State Timeline',
                yaxis=dict(range=[-1, 1]),
                template='plotly_dark'
            )
            
            # EEG Spectrum
            spectrum_fig = go.Figure(data=[
                go.Bar(x=list(self.eeg_data.keys()),
                       y=list(self.eeg_data.values()),
                       marker_color=['#4A90E2', '#E24A4A', '#E2B84A', '#4AE2B8'])
            ])
            spectrum_fig.update_layout(
                title='EEG Spectral Bands',
                template='plotly_dark'
            )
            
            # Emotion Flower (Radar Chart)
            if len(self.history_v) > 0:
                current_v = self.history_v[-1]
                current_a = self.history_a[-1]
                current_c = self.history_c[-1]
            else:
                current_v, current_a, current_c = 0, 0, 0
            
            flower_fig = go.Figure(data=go.Scatterpolar(
                r=[current_v, current_a, current_c],
                theta=['Valence', 'Arousal', 'Coherence'],
                fill='toself',
                name='Current State'
            ))
            flower_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
                showlegend=True,
                title='Emotion Flower (3D Radar)',
                template='plotly_dark'
            )
            
            # MIDI/OSC Log (example)
            log_content = html.Div([
                html.P(f"Tempo: {int(60 + current_a * 80)} BPM"),
                html.P(f"Mode: {'Major' if current_v > 0 else 'Minor'}"),
                html.P(f"OSC Hue: {int(180 + current_v * 80)}"),
            ])
            
            return timeline_fig, spectrum_fig, flower_fig, log_content
    
    def update_data(self, valence, arousal, coherence, eeg_data):
        self.history_v.append(valence)
        self.history_a.append(arousal)
        self.history_c.append(coherence)
        self.eeg_data = eeg_data
    
    def run(self, port=8050, debug=True):
        self.app.run_server(port=port, debug=debug)
```

---

## 3. Updated Build Structure

| File | Description |
|------|-------------|
| **omega_live.py** | Main engine (EEG sim → Fusion → Agents → Outputs) |
| **omega_config.json** | Device routing, thresholds, color & tempo mappings |
| **omega_emotion_core.py** | EEG + Biometric → Valence/Arousal/Dominance |
| **omega_midi_osc.py** | Unified output to MIDI + OSC targets |
| **omega_visuals.py** | Matplotlib window + Dash hybrid system |
| **omega_protocol.json** | ORP-compatible live data schema |
| **omega_utils.py** | Logging, smoothing, normalization |
| **/dash_assets/** | (auto-generated) CSS, visuals, icons for dashboard |

---

## 4. Workflow Example

### Complete Execution Flow

```
1. omega_live.py runs EEG & biometric simulators
   ↓
2. Fusion core calculates emotion vectors (VAD)
   ↓
3. MIDI + OSC Agents respond in realtime
   ↓
4. Visualization layers reflect current system state:
     ├── Local: Matplotlib graph window
     └── Remote: Dash dashboard (http://localhost:8050)
```

### Integration Code

```python
# omega_live.py integration
from omega_visuals import MatplotlibVisualizer, DashDashboard

# Initialize visualizers
matplotlib_viz = MatplotlibVisualizer()
dash_viz = DashDashboard()

# Run Dash in separate thread
import threading
dash_thread = threading.Thread(target=dash_viz.run, args=(8050, False))
dash_thread.daemon = True
dash_thread.start()

# Main loop
while True:
    # ... emotion calculation ...
    valence, arousal, coherence = emotion_core.compute()
    eeg_data = eeg_stream.get_bands()
    
    # Update visualizations
    matplotlib_viz.update(valence, arousal, coherence, eeg_data)
    dash_viz.update_data(valence, arousal, coherence, eeg_data)
    
    time.sleep(0.5)
```

---

## 5. Configuration (omega_config.json)

### Example Configuration

```json
{
  "visualization": {
    "matplotlib": {
      "enabled": true,
      "update_interval": 0.5,
      "max_points": 100,
      "window_size": [10, 6]
    },
    "dash": {
      "enabled": true,
      "port": 8050,
      "host": "127.0.0.1",
      "theme": "dark",
      "auto_refresh_interval": 500
    }
  },
  "eeg_bands": {
    "alpha": {"range": [8, 12], "color": "#4A90E2"},
    "beta": {"range": [13, 30], "color": "#E24A4A"},
    "theta": {"range": [4, 7], "color": "#E2B84A"},
    "gamma": {"range": [30, 50], "color": "#4AE2B8"}
  },
  "emotion_mapping": {
    "valence": {"color": "#4A90E2", "label": "Valence"},
    "arousal": {"color": "#E24A4A", "label": "Arousal"},
    "coherence": {"color": "#E2B84A", "label": "Coherence"}
  }
}
```

---

## 6. Optional Enhancements (Post-Prototyping)

### Future Features

#### 🧬 EEG-Driven Visual Modulation

- Color hue & waveform blend based on dominant EEG band
- Real-time visual effects synchronized with brain activity
- Adaptive color palettes responding to emotional state

#### 🕹️ User Control Nodes

- Adjust responsiveness live (sensitivity sliders)
- Manual override for emotion vectors
- Custom mapping presets

#### 🔊 Audio Reactivity

- Spectrum sync via pyaudio or DAW OSC
- Visual waveforms matching audio output
- Frequency-domain visualization

#### 🧭 Geo-Time Layer

- Local time display
- Circadian pattern visualization
- Sleep-cycle awareness indicators
- Location-based environmental context

---

## 7. Performance Considerations

### Matplotlib Optimization

- Use `plt.ion()` for interactive mode
- Limit history deque size (default: 100 points)
- Update only visible regions
- Use `plt.pause(0.05)` for smooth updates

### Dash Optimization

- Use `dcc.Interval` for auto-refresh
- Limit data history to prevent memory bloat
- Use Plotly's built-in optimizations
- Consider WebSocket for real-time updates (future)

### Resource Usage

| Component | CPU Usage | Memory | Network |
|-----------|-----------|--------|---------|
| **Matplotlib** | Low (~2-5%) | ~50MB | None |
| **Dash** | Medium (~5-10%) | ~100MB | Local (127.0.0.1) |
| **Combined** | Medium (~7-15%) | ~150MB | Minimal |

---

## 8. Integration with DAWs and Visual Software

### TouchDesigner Integration

- OSC messages sent to TouchDesigner
- Real-time parameter control
- Visual effects synchronized with emotion

### Resolume Integration

- OSC color/brightness control
- Layer opacity based on arousal
- Effect intensity based on valence

### Ableton Live Integration

- MIDI tempo sync
- Key/mode changes
- CC parameter automation

---

## 9. Deployment Options

### Local Development

```bash
python omega_live.py
# Opens Matplotlib window + starts Dash on localhost:8050
```

### Network Deployment

```python
# In omega_config.json
"dash": {
  "host": "0.0.0.0",  # Allow network access
  "port": 8050
}
```

### Production Mode

- Disable Matplotlib for headless servers
- Use Dash only for remote access
- Optional: Embed in web application

---

## 10. Troubleshooting

### Common Issues

**Matplotlib window not updating:**
- Ensure `plt.ion()` is called
- Check `plt.pause()` is in update loop
- Verify figure is not closed

**Dash dashboard not loading:**
- Check port 8050 is available
- Verify firewall settings
- Check browser console for errors

**Performance issues:**
- Reduce update frequency
- Limit history size
- Disable unused visualizations

---

## Related Documents

- [[suno_omega_simulation_prototype]] - Complete simulation prototype
- [[suno_omega_synthesis_v5]] - Unified multi-agent engine
- [[suno_omega_resonance_protocol]] - ORP communication standard
- [[suno_complete_system_reference]] - Master index
