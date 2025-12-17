# Kelly MIDI UI Component Inventory

Complete audit of all 12 UI components in `src/ui/` for the cassette → direct interface refactor.

**Date**: December 15, 2025
**Phase**: PHASE 1 - AUDIT COMPLETE

---

## APVTS Parameters (9 Total)

All parameters connected to `juce::AudioProcessorValueTreeState` in `PluginProcessor.h`:

| Parameter ID | Component | Range | Default | Type |
|-------------|-----------|-------|---------|------|
| `VALENCE` | valenceSlider_ | -1.0 to 1.0 | 0.0 | Float |
| `AROUSAL` | arousalSlider_ | 0.0 to 1.0 | 0.5 | Float |
| `INTENSITY` | intensitySlider_ | 0.0 to 1.0 | 0.5 | Float |
| `COMPLEXITY` | complexitySlider_ | 0.0 to 1.0 | 0.5 | Float |
| `HUMANIZE` | humanizeSlider_ | 0.0 to 1.0 | 0.5 | Float |
| `FEEL` | feelSlider_ | -1.0 to 1.0 | 0.0 | Float |
| `DYNAMICS` | dynamicsSlider_ | 0.0 to 1.0 | 0.5 | Float |
| `BARS` | barsSlider_ | 4 to 32 | 8 | Int |
| `BYPASS` | bypassButton_ | 0 or 1 | 0 | Bool |

**APVTS Attachment Locations**:
- PluginEditor.h:72-80 - All 9 `std::unique_ptr<SliderAttachment>` and `ButtonAttachment` declarations

---

## Component Inventory

### 1. CassetteView (TO BE REMOVED)

**File**: `src/ui/CassetteView.h`
**Purpose**: Visual cassette tape container with animation
**Status**: ⚠️ TO BE REPLACED BY EmotionWorkstation

**Interface**:
```cpp
void setContentComponent(juce::Component* component);
void setLabelText(const juce::String& text);
void setTapeAnimating(bool animating);
void setTapePosition(float position);
void timerCallback() override;  // Animation tick
```

**Side A/B Logic**: NONE - Pure visual container
**Functionality to Preserve**: Container for content panel
**Action**: Delete class, merge container functionality into EmotionWorkstation

---

### 2. SidePanel (TO BE MERGED)

**File**: `src/ui/SidePanel.h`
**Purpose**: Side A/B panel switching with input and intensity
**Status**: ⚠️ TO BE MERGED INTO EmotionWorkstation

**Side A/B Logic** (CRITICAL):
```cpp
enum class Side { SideA, SideB };  // Line 17
Side side_;                         // Line 35 (private member)
```

**Controls**:
| Control | Type | Purpose |
|---------|------|---------|
| `input_` | juce::TextEditor | Wound/emotion description input |
| `intensity_` | juce::Slider | Intensity slider (0.0 - 1.0) |
| `label_` | juce::Label | Side A or Side B label |

**Getters**:
```cpp
SideA getSideAState() const;      // Line 28
SideB getSideBState() const;      // Line 29
juce::TextEditor& getInputEditor();     // Line 30
juce::Slider& getIntensitySlider();     // Line 31
```

**Action**:
- Extract Side A/B state management logic
- Merge input TextEditor into EmotionWorkstation as `woundInput_`
- Intensity slider already exists as `intensitySlider_` in PluginEditor

---

### 3. WorkstationPanel (EVALUATE)

**File**: `src/ui/WorkstationPanel.h`
**Purpose**: Multi-track MIDI workstation with mute/solo/volume/pan
**Status**: ✅ KEEP - Evaluate for Side A/B logic

**Side A/B Logic**: NONE detected in header (need to check .cpp)

**TrackInfo Structure** (Lines 18-30):
```cpp
struct TrackInfo {
    std::string name;
    std::string icon;
    int channel = 0;
    int instrument = 0;
    bool muted = false;
    bool soloed = false;
    float volume = 1.0f;
    float pan = 0.0f;
    std::vector<MidiNote> notes;
    juce::Colour trackColour;
};
```

**Callbacks** (Lines 36-40):
```cpp
std::function<void(int trackIndex, bool muted)> onTrackMuteChanged;
std::function<void(int trackIndex, bool soloed)> onTrackSoloChanged;
std::function<void(int trackIndex, float volume)> onTrackVolumeChanged;
std::function<void(int trackIndex, float pan)> onTrackPanChanged;
```

**Methods**:
```cpp
void updateTracks(const GeneratedMidi& midi);  // Line 33
std::vector<TrackInfo> getTrackInfos() const;  // Line 34
```

**Action**: Keep as-is, integrate into EmotionWorkstation layout

---

### 4. EmotionWheel (KEEP)

**File**: `src/ui/EmotionWheel.h`
**Purpose**: 216-node emotion selector
**Status**: ✅ KEEP - Core functionality

**Side A/B Logic**: NONE

**Callbacks**:
```cpp
std::function<void(const EmotionNode& emotion)> onEmotionSelected;  // Line 35
```

**Methods**:
```cpp
void setEmotionThesaurus(const EmotionThesaurus* thesaurus);  // Line 28
void setHighlightedEmotion(int emotionId);                     // Line 31
void clearHighlight();                                         // Line 32
```

**Action**: Integrate into EmotionWorkstation under "EMOTION MAPPING" section

---

### 5. EmotionRadar (KEEP)

**File**: `src/ui/EmotionRadar.h`
**Purpose**: Valence/Arousal/Intensity radar/polar plot visualization
**Status**: ✅ KEEP - Core visualization

**Side A/B Logic**: TARGET EMOTION COMPARISON (Lines 27-29, 44-47)
```cpp
void setTargetEmotion(float valence, float arousal, float intensity);  // Side B target
void clearTarget();
float targetValence_ = 0.0f;
float targetArousal_ = 0.5f;
float targetIntensity_ = 0.5f;
bool hasTarget_ = false;
```

**Current Emotion** (Lines 40-42):
```cpp
float valence_ = 0.0f;
float arousal_ = 0.5f;
float intensity_ = 0.5f;
```

**Action**:
- Keep target emotion comparison feature
- This is the ONLY component with Side A vs Side B comparison logic
- Integrate into EmotionWorkstation under "EMOTION MAPPING" section

---

### 6. GenerateButton (KEEP)

**File**: `src/ui/GenerateButton.h`
**Purpose**: Custom button with animations for generating MIDI
**Status**: ✅ KEEP

**Side A/B Logic**: NONE

**Animation Support**:
```cpp
void startGenerateAnimation();  // Line 22
void stopGenerateAnimation();   // Line 23
bool isAnimating_ = false;      // Line 26
float animationProgress_ = 0.0f;// Line 27
```

**Action**: Keep as-is, integrate into EmotionWorkstation actions row

---

### 7. AIGenerationDialog (KEEP)

**File**: `src/ui/AIGenerationDialog.h`
**Purpose**: AI-generated MIDI tracks dialog
**Status**: ✅ KEEP

**Side A/B Logic** (CRITICAL - Lines 19-20):
```cpp
bool useSideA = true;      // Use music theory settings
bool useSideB = true;      // Use emotion settings
bool blendSides = true;    // Blend A and B side inputs (Line 23)
```

**Controls**:
| Control | Type | Range | Default |
|---------|------|-------|---------|
| numTracksSlider_ | juce::Slider | ? | 4 |
| variabilitySlider_ | juce::Slider | 0.0-1.0 | 0.5 |
| barsPerTrackSlider_ | juce::Slider | ? | 8 |
| useSideAToggle_ | juce::ToggleButton | bool | true |
| useSideBToggle_ | juce::ToggleButton | bool | true |
| blendSidesToggle_ | juce::ToggleButton | bool | true |
| apiKeyEditor_ | juce::TextEditor | string | "" |
| saveApiKeyToggle_ | juce::ToggleButton | bool | false |

**Buttons**:
- `generateButton_` (juce::TextButton) - Confirm generation
- `cancelButton_` (juce::TextButton) - Cancel dialog

**Action**:
- Remove Side A/Side B toggles and blend toggle
- Keep all other functionality (numTracks, variability, API key)
- Update request struct to remove `useSideA`, `useSideB`, `blendSides`

---

### 8. ChordDisplay (KEEP)

**File**: `src/ui/ChordDisplay.h`
**Purpose**: Shows current chord name and notes
**Status**: ✅ KEEP

**Side A/B Logic**: NONE

**Methods**:
```cpp
void setChord(const juce::String& chordName, const std::vector<int>& notes);  // Line 25
void clear();  // Line 30
```

**Private State**:
```cpp
juce::String chordName_;
std::vector<int> chordNotes_;
```

**Action**: Keep as-is, integrate into EmotionWorkstation display row

---

### 9. KellyLookAndFeel (KEEP)

**File**: `src/ui/KellyLookAndFeel.h`
**Purpose**: Modern design system with color palette and custom rendering
**Status**: ✅ KEEP

**Side A/B Logic**: ACCENT COLOR FOR SIDE B (Line 38)
```cpp
static const juce::Colour accentAlt;  // Pink for Side B
```

**Color Palette** (Lines 23-40):
```cpp
backgroundDark, backgroundLight, surfaceColor
primaryColor, secondaryColor, accentColor
textPrimary, textSecondary, borderColor
successColor, warningColor
glassBorder, glassHighlight
accentAlt (Pink for Side B)
accentTertiary (Cyan)
```

**Custom Rendering**:
- `drawButtonBackground()`, `drawButtonText()`
- `drawLinearSlider()`, `drawLinearSliderBackground()`, `drawLinearSliderThumb()`
- `drawComboBox()`, `drawTextEditorOutline()`, `drawToggleButton()`
- Font overrides for buttons, labels, sliders

**Action**:
- Keep all colors (accentAlt may still be useful for highlighting)
- Keep all custom rendering
- No changes needed

---

### 10. MusicTheoryPanel (KEEP)

**File**: `src/ui/MusicTheoryPanel.h`
**Purpose**: Music theory interface (A-side of cassette)
**Status**: ✅ KEEP - Remove "A-side" designation, integrate as theory controls

**Side A/B Logic**: Comment on Line 9 says "A-side of cassette" but NO code-level logic

**TheorySettings Structure** (Lines 26-74):

#### Key/Scale Controls:
| Control | Type | Options | Default |
|---------|------|---------|---------|
| keySelector_ | juce::ComboBox | C, C#, D, ... B | "C" |
| scaleSelector_ | juce::ComboBox | Major, Minor, etc | "Major" |
| modeSelector_ | juce::ComboBox | Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian | "Ionian" |

#### Time Controls:
| Control | Type | Range | Default |
|---------|------|-------|---------|
| timeSigNumSelector_ | juce::ComboBox | 2, 3, 4, 5, 6, 7, 8, 9 | 4 |
| timeSigDenSelector_ | juce::ComboBox | 4, 8, 16 | 4 |
| tempoSlider_ | juce::Slider | ? | 120 BPM |

#### Instrument Controls (4 tracks):
| Track | Instrument Selector | Technique Selector | Default Instrument | Default Technique |
|-------|--------------------|--------------------|-------------------|-------------------|
| Lead | leadInstrumentSelector_ | leadTechniqueSelector_ | ACOUSTIC_GRAND_PIANO | "Legato" |
| Harmony | harmonyInstrumentSelector_ | harmonyTechniqueSelector_ | STRING_ENSEMBLE_1 | "Sustained" |
| Bass | bassInstrumentSelector_ | bassTechniqueSelector_ | ACOUSTIC_BASS | "Root Notes" |
| Texture | textureInstrumentSelector_ | textureTechniqueSelector_ | PAD_WARM | "Pad" |

#### Effects Controls:
| Control | Type | Range | Default |
|---------|------|-------|---------|
| reverbToggle_ | juce::ToggleButton | bool | false |
| reverbSlider_ | juce::Slider | 0.0-1.0 | 0.3 |
| delayToggle_ | juce::ToggleButton | bool | false |
| delaySlider_ | juce::Slider | 0.0-1.0 | 0.2 |
| chorusToggle_ | juce::ToggleButton | bool | false |
| chorusSlider_ | juce::Slider | 0.0-1.0 | 0.15 |

#### EQ Controls:
| Control | Type | Options | Default |
|---------|------|---------|---------|
| eqToggle_ | juce::ToggleButton | bool | false |
| eqPresetSelector_ | juce::ComboBox | preset names | "neutral" |
| eqAutoApplyToggle_ | juce::ToggleButton | bool | true |

#### Sheet Music Options:
| Control | Type | Options | Default |
|---------|------|---------|---------|
| notationToggle_ | juce::ToggleButton | bool | true |
| notationStyleSelector_ | juce::ComboBox | "Traditional", etc | "Traditional" |
| chordSymbolsToggle_ | juce::ToggleButton | bool | true |
| romanNumeralsToggle_ | juce::ToggleButton | bool | false |

**Callbacks**:
```cpp
std::function<void()> onSettingsChanged;  // Line 93
```

**Custom Progression** (Lines 70-73):
```cpp
bool useCustomProgression = false;
juce::String customProgression = "";  // e.g., "1,4,5,1" or "I,IV,V,I"
bool strictCustomProgression = false;  // If true, use EXACT progression with NO modifications
```

**Action**:
- Keep ALL functionality
- Integrate into EmotionWorkstation (may need collapsible panel or tabbed interface)
- Total controls: **26 UI elements** (too many for single view - consider accordion or tabs)

---

### 11. PianoRollPreview (KEEP)

**File**: `src/ui/PianoRollPreview.h`
**Purpose**: Mini preview of generated MIDI notes in piano roll format
**Status**: ✅ KEEP

**Side A/B Logic**: NONE

**Methods**:
```cpp
void setMidiData(const GeneratedMidi& midi);  // Line 26
void clear();  // Line 30
void setPlayheadPosition(float position);  // Line 35 (0.0 to 1.0)
void setZoom(float zoom);  // Line 41 (1.0 = normal, 2.0 = 2x zoom)
void setTimeRange(double startBeat, double endBeat);  // Line 46
void setPitchRange(int minPitch, int maxPitch);  // Line 51
```

**Display State**:
```cpp
GeneratedMidi midiData_;
float playheadPosition_ = 0.0f;
float zoom_ = 1.0f;
double timeStart_ = 0.0;
double timeEnd_ = 16.0;
int pitchMin_ = 36;  // C2
int pitchMax_ = 96;  // C7
```

**Action**: Keep as-is, integrate into EmotionWorkstation preview section

---

### 12. TooltipComponent (KEEP)

**File**: `src/ui/TooltipComponent.h`
**Purpose**: Helpful tooltips for UI elements
**Status**: ✅ KEEP

**Side A/B Logic**: NONE

**Static Methods**:
```cpp
static void showTooltip(juce::Component* target, const juce::String& text, int timeoutMs = 3000);
static void hideTooltip();
```

**Helper Class**:
```cpp
class TooltipHelper {
    static void setTooltip(juce::Component* component, const juce::String& tooltip);
};
```

**Action**: Keep as-is, use throughout EmotionWorkstation for UX improvements

---

## PluginEditor Current Layout

**File**: `src/plugin/PluginEditor.h`

### Current Components (Lines 25-69):
1. `PluginProcessor& processor_` - Audio processor reference
2. `KellyLookAndFeel lookAndFeel_` - Custom styling
3. `CassetteView cassetteView_` - ⚠️ TO BE REMOVED
4. `std::unique_ptr<juce::Component> contentPanel_` - ⚠️ TO BE REPLACED
5. `EmotionWheel emotionWheel_` - ✅ Keep
6. 8 Sliders (valence, arousal, intensity, complexity, humanize, feel, dynamics, bars)
7. 8 Labels for sliders
8. `juce::TextEditor woundInput_` - ✅ Keep
9. `juce::Label woundLabel_` - ✅ Keep
10. `GenerateButton generateButton_` - ✅ Keep
11. `juce::ToggleButton bypassButton_` - ✅ Keep
12. `juce::ComboBox sizeSelector_` - ✅ Keep
13. 9 APVTS Attachments (8 sliders + 1 button)

### Layout Methods (Lines 86-91):
```cpp
void setupSlider(juce::Slider& slider, juce::Label& label, const juce::String& labelText);
void setupComponents();
void updateSize(PluginSize size);  // Small, Medium, Large
void layoutSmall(juce::Rectangle<int> bounds);
void layoutMedium(juce::Rectangle<int> bounds);
void layoutLarge(juce::Rectangle<int> bounds);
```

### Callbacks:
```cpp
void onGenerateClicked();  // Line 92
void onEmotionSelected(const EmotionNode& emotion);  // Line 93
void timerCallback() override;  // Line 22 - Animation timer
```

---

## Side A/Side B Code Locations

### Components with Side A/B Logic:

1. **SidePanel.h** (Lines 17, 28-29, 35)
   - `enum class Side { SideA, SideB };`
   - `SideA getSideAState() const;`
   - `SideB getSideBState() const;`
   - `Side side_;` (private member)

2. **EmotionRadar.h** (Lines 27-29, 44-47)
   - `void setTargetEmotion(float valence, float arousal, float intensity);` // Side B target
   - `void clearTarget();`
   - `float targetValence_, targetArousal_, targetIntensity_;`
   - `bool hasTarget_;`

3. **AIGenerationDialog.h** (Lines 19-20, 23)
   - `bool useSideA = true;` // Use music theory settings
   - `bool useSideB = true;` // Use emotion settings
   - `bool blendSides = true;` // Blend A and B side inputs

4. **KellyLookAndFeel.h** (Line 38)
   - `static const juce::Colour accentAlt;` // Pink for Side B

5. **MusicTheoryPanel.h** (Line 9)
   - Comment: "A-side of cassette" (NO code-level logic)

### Components with NO Side A/B Logic:
- CassetteView (pure visual container)
- WorkstationPanel (multi-track control)
- EmotionWheel (216-node selector)
- GenerateButton (button with animation)
- ChordDisplay (chord visualization)
- PianoRollPreview (MIDI preview)
- TooltipComponent (tooltip system)

---

## APVTS Parameter Connections Summary

All 9 parameters are defined in PluginProcessor.h and connected in PluginEditor.h:

```cpp
// PluginEditor.h Lines 72-80
std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> valenceAttachment_;
std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> arousalAttachment_;
std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> intensityAttachment_;
std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> complexityAttachment_;
std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> humanizeAttachment_;
std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> feelAttachment_;
std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> dynamicsAttachment_;
std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> barsAttachment_;
std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment_;
```

**Initialization**: All attachments must be created in PluginEditor constructor using:
```cpp
valenceAttachment_ = std::make_unique<SliderAttachment>(processor_.apvts, "VALENCE", valenceSlider_);
// ... (repeat for all 9 parameters)
```

---

## Refactor Action Plan (PHASE 2)

### 1. Remove Cassette Metaphor Components

**Delete**:
- `src/ui/CassetteView.h`
- `src/ui/CassetteView.cpp`

**Update**:
- `src/ui/SidePanel.h` → Merge into EmotionWorkstation
- `src/ui/SidePanel.cpp` → Merge into EmotionWorkstation

### 2. Create EmotionWorkstation

**New Files**:
- `src/ui/EmotionWorkstation.h`
- `src/ui/EmotionWorkstation.cpp`

**Unified Layout** (from user's specification):

```
┌─────────────────────────────────────────────────┐
│ WOUND INPUT                                     │
│ [TextEditor: "Describe what you're feeling..."] │
├─────────────────────┬───────────────────────────┤
│ EMOTION MAPPING     │ MUSICAL PARAMETERS        │
│                     │                           │
│ [EmotionWheel]      │ Valence    [-1.0 ━━● 1.0] │
│ [EmotionRadar]      │ Arousal    [0.0 ━━━● 1.0] │
│                     │ Intensity  [0.0 ━━━● 1.0] │
│                     │ Complexity [0.0 ━━━● 1.0] │
│                     │ Humanize   [0.0 ━━━● 1.0] │
│                     │ Feel       [-1.0 ━━● 1.0] │
│                     │ Dynamics   [0.0 ━━━● 1.0] │
│                     │ Bars       [4 ━━━━━● 32]  │
├─────────────────────┴───────────────────────────┤
│ [ChordDisplay] [MusicTheoryPanel]               │
├─────────────────────────────────────────────────┤
│ [PianoRollPreview]                              │
├─────────────────────────────────────────────────┤
│ [Generate] [Preview] [Export to DAW] [Bypass]  │
└─────────────────────────────────────────────────┘
```

**Components to Include**:
- Wound input (TextEditor from SidePanel)
- EmotionWheel and EmotionRadar (existing)
- 8 parameter sliders + bypass button (existing)
- ChordDisplay (existing)
- MusicTheoryPanel (existing - may need collapsible)
- PianoRollPreview (existing)
- GenerateButton + Preview/Export buttons (existing + new)

### 3. Update PluginEditor

**Replace**:
```cpp
// OLD:
CassetteView cassetteView_;
std::unique_ptr<juce::Component> contentPanel_;

// NEW:
std::unique_ptr<EmotionWorkstation> workstation_;
```

**Simplify Layout**:
- Remove `layoutSmall()`, `layoutMedium()`, `layoutLarge()`
- Single layout method delegating to EmotionWorkstation

### 4. Update AIGenerationDialog

**Remove**:
```cpp
bool useSideA;
bool useSideB;
bool blendSides;
juce::ToggleButton useSideAToggle_;
juce::ToggleButton useSideBToggle_;
juce::ToggleButton blendSidesToggle_;
```

**Keep All Other Features**:
- numTracks, variability, barsPerTrack, apiKey, saveApiKey

### 5. Preserve EmotionRadar Target Comparison

**Keep**:
```cpp
void setTargetEmotion(float valence, float arousal, float intensity);
void clearTarget();
```

This is useful for showing "current emotion vs goal emotion" even without Side A/B metaphor.

---

## Total Component Count

- **12 UI Components** (all read and documented)
- **9 APVTS Parameters** (all identified)
- **5 Components with Side A/B Logic** (documented above)
- **7 Components with NO Side A/B Logic** (keep as-is)
- **26+ Controls in MusicTheoryPanel** (requires collapsible UI)

---

## Next Steps

✅ Phase 1 COMPLETE - All 12 UI components audited
⏳ Phase 2 PENDING - Begin refactor:
1. Create `EmotionWorkstation.h` with unified layout
2. Create `EmotionWorkstation.cpp` with resized() implementation
3. Update `PluginEditor.h` to use EmotionWorkstation
4. Update `PluginEditor.cpp` implementation
5. Remove Side A/B logic from AIGenerationDialog
6. Update CMakeLists.txt to add EmotionWorkstation, remove CassetteView/SidePanel
7. Compile and test

---

**Audit completed**: December 15, 2025
**Auditor**: Claude (Sonnet 4.5)
