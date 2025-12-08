# iDAWi Master TODO

## 🔴 CRITICAL - Build & Runtime (Do First)

### Build Setup
- [ ] Run `npm install` - ensure all dependencies installed
- [ ] Run `npm run type-check` - fix all TypeScript errors
- [ ] Run `npm run lint` - fix all ESLint errors
- [ ] Run `npm run build` - ensure production build works
- [ ] Run `npm run dev` - ensure dev server starts on localhost:3000

### Tauri/Rust Setup
- [ ] Run `cd src-tauri && cargo build` - ensure Rust compiles
- [ ] Run `cargo clippy` - fix all Rust warnings
- [ ] Test `npm run tauri dev` - ensure native app launches

### Python Music Brain Setup
- [ ] Create venv: `cd music-brain && python3 -m venv venv`
- [ ] Install deps: `source venv/bin/activate && pip install -r requirements.txt`
- [ ] Test bridge: `echo '{"command":"get_emotions","args":{}}' | python bridge.py`

---

## 🟡 TYPESCRIPT ERRORS - Fix All

### src/App.tsx
- [ ] Verify useStore imports match store exports
- [ ] Check keyboard event handlers have proper types
- [ ] Ensure toggleSide function exists in store

### src/store/useStore.ts
- [ ] All Track interface properties typed
- [ ] All Clip interface properties typed
- [ ] All SongIntent properties typed
- [ ] All actions properly typed
- [ ] No `any` types

### src/components/SideA/
- [ ] Timeline.tsx - canvas typing, track/clip types
- [ ] Mixer.tsx - level state typing, track props
- [ ] VUMeter.tsx - level/peak number types
- [ ] Knob.tsx - mouse event types, ref types
- [ ] Transport.tsx - all handler types
- [ ] Toolbar.tsx - button handler types

### src/components/SideB/
- [ ] EmotionWheel.tsx - emotion interface types
- [ ] Interrogator.tsx - phase/answer types
- [ ] GhostWriter.tsx - suggestion interface types
- [ ] RuleBreaker.tsx - rule option types
- [ ] SideBToolbar.tsx - handler types

### src/hooks/
- [ ] useMusicBrain.ts - all return types, async types
- [ ] useTauriAudio.ts - Tauri invoke types

---

## 🟢 ESLINT WARNINGS - Fix All

- [ ] Remove unused imports across all files
- [ ] Fix missing dependencies in useEffect hooks
- [ ] Remove unused variables
- [ ] Fix any React hooks rules violations
- [ ] Ensure consistent quote style
- [ ] Fix any accessibility warnings (a11y)

---

## 🔵 FEATURE COMPLETION

### Side A - DAW Interface
- [ ] Timeline: Click to set playhead position
- [ ] Timeline: Drag clips to move them
- [ ] Timeline: Resize clips by dragging edges
- [ ] Timeline: Add new clips via double-click
- [ ] Mixer: Connect VU meters to actual audio levels
- [ ] Mixer: Master channel level summing
- [ ] Transport: Loop region visualization
- [ ] Transport: Keyboard shortcuts (Space=play, etc.)

### Side B - Emotion Interface
- [ ] EmotionWheel: All 6 categories with sub-emotions
- [ ] EmotionWheel: Visual selection feedback
- [ ] Interrogator: Save answers to store
- [ ] Interrogator: Progress persistence
- [ ] GhostWriter: Connect to Python Music Brain
- [ ] GhostWriter: Apply suggestions to DAW state
- [ ] RuleBreaker: Dynamic suggestions based on emotion

### Tauri Integration
- [ ] audio_play command working
- [ ] audio_stop command working
- [ ] audio_set_tempo command working
- [ ] music_brain_command IPC working
- [ ] File save/load dialogs
- [ ] MIDI export functionality

### Python Music Brain
- [ ] All emotions in EMOTIONS_DATABASE
- [ ] All rule breaks in RULE_BREAKING_EFFECTS
- [ ] process_intent returns valid musical params
- [ ] suggest_rule_break returns contextual suggestions

---

## 🟣 CODE QUALITY

### Refactoring
- [ ] Extract repeated styles to Tailwind components
- [ ] Create shared Button component
- [ ] Create shared Panel component
- [ ] Consolidate color definitions
- [ ] Remove duplicate type definitions

### Documentation
- [ ] JSDoc comments on all exported functions
- [ ] README quick start verified working
- [ ] Component prop documentation
- [ ] Hook usage examples

### Testing
- [ ] Add Jest/Vitest setup
- [ ] Test useStore actions
- [ ] Test useMusicBrain hooks
- [ ] Test Python bridge responses

---

## ⚪ NICE TO HAVE

- [ ] Dark/light theme toggle
- [ ] Keyboard shortcuts modal (show on ?)
- [ ] Undo/redo for DAW actions
- [ ] Project save/load to JSON
- [ ] Audio file import
- [ ] Waveform visualization
- [ ] MIDI piano roll editor
- [ ] Plugin slot UI (placeholder)

---

## 📋 CURRENT SESSION FOCUS

Agent: Start with 🔴 CRITICAL section. 
Move down the list in order.
Skip to next section if stuck on any item for 3+ attempts.

---

## ✅ COMPLETED
(Move items here when done)

---

## 🚫 STUCK (See STUCK_LOG.md)
(Reference items that couldn't be resolved)
