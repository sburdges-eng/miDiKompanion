```
Title: Real-Time MIDI Engine Plan (Phase 3 Kickoff)
Last Updated: 2025-11-28
Owner: DAiW Core Team
```

## 1. Why this exists
- Current flow renders full MIDI files (`generate_midi_from_harmony`, `render_plan_to_midi`) before playback.
- Phase 3 requires **live** response to emotional inputs, DAW plugin control, and mobile capture.
- Need a reusable streaming core that any client (CLI, JUCE, MCP, mobile) can subscribe to.

## 2. Goals & guardrails
1. **Latency**: <15 ms scheduling jitter at 120 BPM on macOS/Win.
2. **Determinism**: Same intent + seed → identical note stream (record vs live).
3. **Interoperability**: Use existing `NoteEvent` + Groove/Harmony modules.
4. **Safety**: Back-pressure + drop protection (never overflow MIDI out).
5. **Observability**: Health events + perf counters consumable by UI/CLI.

## 3. Architecture Overview
```
Intent (CompleteSongIntent) / HarmonyPlan
        │
        ▼
[Generator Adapter Layer]
  - HarmonyAdapter (wraps HarmonyGenerator)
  - GrooveAdapter (wraps GrooveSettings/humanize_drums)
        │
        ▼
[Event Graph]
  ├─ PatternTrack (sequence of chord/voicing NoteEvents)
  ├─ GrooveTrack (timing offsets, velocity curves)
  └─ ControlTrack (tempo ramps, rule-break cues)
        │
        ▼
[RealtimeEngine]
  ├─ Clock (high-res, tempo-aware)
  ├─ Scheduler (ring buffer of upcoming NoteEvents)
  ├─ Transport (MIDI port / OSC / WebSocket)
  └─ State Store (play/pause, seek, loop, scene)
```

## 4. Module Plan
| Module | Path | Responsibility |
| --- | --- | --- |
| `music_brain/realtime/engine.py` | Core orchestrator, exposes high-level API (`start`, `queue_intent`, `subscribe`). |
| `music_brain/realtime/clock.py` | Tempo-aware clock (PPQ <-> ms) w/ drift correction + tap tempo hooks. |
| `music_brain/realtime/scheduler.py` | Priority queue of `ScheduledEvent` objects (wrapping `NoteEvent` + metadata). |
| `music_brain/realtime/transport.py` | Abstract base for outputs (MIDI out via `mido`, OSC, WebSocket). |
| `music_brain/realtime/session.py` | Holds active `HarmonyPlan`, groove state, rule-break automation. |
| `music_brain/realtime/events.py` | Data classes for streaming (`ScheduledEvent`, `ControlEvent`, `MetricEvent`). |

## 5. API Surface (draft)
```python
from music_brain.realtime.engine import RealtimeEngine

engine = RealtimeEngine(ppq=960)
engine.add_output(MidiPort(name="DAiW Virtual"))
engine.load_intent(intent)             # or load_plan(HarmonyPlan)
engine.set_groove(settings)            # optional humanization
engine.start()

# runtime control
engine.push_control(ControlEvent.type("rule_break_toggle", value=True))
engine.seek(bars=8)
engine.stop()
```

### Event ingestion
- `queue_pattern(note_events: List[NoteEvent], channel: int, loop: bool)`
- `register_scene(name, builder_fn)` for live scene recalls.
- `subscribe_metrics(callback)` (latency, dropped events, CPU).

### Transport plug-ins
- `MidiPort`: wraps `mido.open_output`; batches note on/off. Add throttle/back-pressure.
- `OscPort`: sends JSON note events to OSC/UDP (DAW companion).
- `WebSocketPort`: for Streamlit/mobile preview with latency stats.
- `PluginBridge`: shared-memory bridge so JUCE/C++ plugin can request scenes.

## 6. Data Flow
1. Intent / HarmonyPlan enters Engine.
2. Engine converts to `NoteEvent` lists (reuse `HarmonyGenerator`, `render_plan_to_midi` logic).
3. Groove adapter mutates timing/velocity -> `NoteEvent`.
4. Scheduler splits events into 1-bar chunks, stores in ring buffer (configurable lookahead, default 2 bars).
5. Clock ticks -> scheduler dequeues events whose start time <= now + safety window.
6. Transport writes NoteOn/Off + control metadata.
7. Metrics service records ms offset, dropped events, CPU load.

## 7. Phase 3 Milestones for Real-Time Engine
| Milestone | Deliverable |
| --- | --- |
| **RT-1 (Week 1)** | Stub modules + simulation (no MIDI). Unit tests verifying clock + scheduler accuracy. |
| **RT-2 (Week 2)** | MIDI port transport working; CLI command `daiw stream --intent path.json`. |
| **RT-3 (Week 3)** | Groove integration + latency profiler CLI. |
| **RT-4 (Week 4)** | WebSocket/OSC transport for JUCE/mobile prototypes. |
| **RT-5 (Week 5)** | Scene management, play/pause/seek/loop, health metrics. |
| **RT-6 (Week 6)** | C++/JUCE Plugin Bridge demo + pybind11 bindings for EMIDI.

## 8. Open Questions / Risks
1. **Python GIL**: may require `asyncio` + `mido` threading model; consider C extension for scheduler if jitter is high.
2. **Groove humanization**: deterministic random seeds per loop vs continuous drift?
3. **Multi-track support**: initial focus on harmony; extend to bass/drums once scheduler proven.
4. **State serialization**: need snapshot format for engine to resume after crash.
5. **Testing harness**: virtual MIDI loopback (CoreMIDI IAC, Windows loopMIDI) required in CI?

## 9. Immediate Next Steps
1. Create `music_brain/realtime/` package with placeholder modules per table.
2. Port `NoteEvent` / `HarmonyPlan` conversions into adapters.
3. Implement deterministic clock + scheduler with simulation tests.
4. Expose experimental CLI command `daiw stream` for developer testing.

> “The pocket is where life happens.” → Real-time engine should feel alive while staying interrogative-first.

