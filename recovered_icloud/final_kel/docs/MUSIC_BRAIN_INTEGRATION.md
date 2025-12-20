# Music Brain Integration Guide

## Overview

Integration of ML model outputs with Music Brain (DAiW-Music-Brain) for music theory validation and rule-breaking.

## Integration Flow

```
ML Model Outputs:
  • MIDI notes (128-dim probabilities)
  • Chords (64-dim probabilities)
  • Groove (32-dim parameters)
  • Expression (16-dim parameters)
    ↓
Music Brain Validation:
  • Intent-driven composition (3-phase schema)
  • Rule-breaking system
  • Groove extraction/application
  • Chord progression analysis
    ↓
Validated/Refined MIDI Output
```

## Components

### 1. Intent-Driven Composition

**3-Phase Schema**: Why → What → How

**Phase 0 (Why)**: Emotional intent
- Core wound/desire
- Emotional goal
- Creative intent

**Phase 1 (What)**: Musical structure
- Form and structure
- Section layout
- Narrative arc

**Phase 2 (How)**: Technical implementation
- Note choices
- Harmony
- Rhythm

### 2. Rule-Breaking System

**Purpose**: Intentional "wrongness" for emotional expression

**Categories**:
- **Harmony**: Avoid resolution, wrong inversions, modal mixture
- **Rhythm**: Constant displacement, polyrhythm, metric modulation
- **Arrangement**: Buried vocals, inverted mix, frequency gaps
- **Production**: Pitch imperfection, controlled distortion, silence

**Location**: `python/penta_core/rules/`

### 3. Groove Extraction and Application

**Extract**: Groove patterns from reference audio  
**Apply**: Groove to generated rhythm  
**Location**: `music_brain/groove/`

### 4. Chord Progression Analysis

**Validate**: Chord progressions  
**Refine**: Apply rule-breaking  
**Location**: `music_brain/harmony/`

## Integration Code

```python
from music_brain import IntentProcessor, RuleBreaker, GrooveExtractor

# ML model outputs
ml_outputs = {
    "notes": melody_output,      # 128-dim probabilities
    "chords": harmony_output,    # 64-dim probabilities
    "groove": groove_output,     # 32-dim parameters
    "expression": dynamics_output # 16-dim parameters
}

# Intent-driven composition
intent_processor = IntentProcessor()
validated_structure = intent_processor.validate(
    ml_outputs,
    emotional_intent=emotion_embedding
)

# Rule-breaking
rule_breaker = RuleBreaker()
refined_output = rule_breaker.apply(
    validated_structure,
    rule_breaking_level=0.3  # 0.0 = no breaking, 1.0 = maximum
)

# Groove application
groove_extractor = GrooveExtractor()
if reference_audio:
    groove = groove_extractor.extract(reference_audio)
    refined_output["groove"] = groove_extractor.apply(
        refined_output["groove"],
        extracted_groove=groove
    )

# Final validated MIDI
final_midi = convert_to_midi(refined_output)
```

## Status

✅ **Music Brain components exist**  
✅ **Integration architecture documented**  
⚠️ **Complete integration code implementation needed**

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-18
