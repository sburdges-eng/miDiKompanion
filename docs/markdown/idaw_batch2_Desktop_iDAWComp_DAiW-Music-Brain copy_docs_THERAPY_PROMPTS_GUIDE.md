# Therapy Prompts Guide

## Overview

The `therapy_prompts.py` module provides evidence-based therapy questions and feeling extraction methods for the DAiW therapy session. It incorporates techniques from:

- **Person-Centered Therapy** (Carl Rogers) - Open, non-directive exploration
- **Solution-Focused Therapy** - Future-oriented, miracle questions
- **Narrative Therapy** - Story reframing, externalization
- **CBT** (Cognitive Behavioral Therapy) - Thought-feeling-behavior links
- **Gestalt Therapy** - Present-moment, body awareness
- **Emotion-Focused Therapy** - Deep feeling exploration

## Common Questions by Category

### 1. Open-Ended Questions (Person-Centered)

**Purpose:** Initial exploration, surface-level feeling identification

- "What brings you here today?"
- "How does this situation make you feel?"
- "What emotion shows up most often for you right now?"
- "Can you describe a time when you felt differently about this?"
- "What's the hardest part about this for you?"

**When to use:** Start of session, general exploration

### 2. Miracle Question (Solution-Focused)

**Purpose:** Envision desired future, extract longing/desire

- "Imagine that tonight, while you sleep, a miracle occurs and your problem is solved. When you wake up tomorrow, what's the first thing you'll notice that's different?"
- "If you woke up tomorrow and everything felt right, what would that look like?"
- "What do you want to feel instead of what you're feeling now?"

**When to use:** When user is stuck in problem, need to shift to solution

### 3. Narrative Therapy Questions

**Purpose:** Externalize problem, reframe story

- "How would you title the story of what you're going through right now?"
- "When did this problem first enter your story?"
- "Are there moments when this problem doesn't have as much power over you?"
- "If this feeling had a voice, what would it say?"

**When to use:** When problem feels overwhelming, need to separate person from problem

### 4. Body Awareness Questions (Gestalt)

**Purpose:** Somatic feeling identification

- "Where do you feel this in your body?"
- "If this feeling had a shape or color, what would it be?"
- "What's happening in your body right now as you talk about this?"

**When to use:** When emotions are vague, need concrete somatic awareness

### 5. Emotion-Focused Questions

**Purpose:** Deep feeling exploration, access primary emotions

- "What's underneath this feeling?"
- "What does this emotion need from you?"
- "If you could give this feeling a name, what would it be?"
- "What's the most vulnerable thing you could say about this?"

**When to use:** When surface emotions are present, need to go deeper

### 6. CBT Questions

**Purpose:** Link thoughts, feelings, and behaviors

- "What thoughts go through your mind when you feel this way?"
- "What happens in your body when you have that thought?"
- "What do you do when you feel this way?"

**When to use:** When patterns need to be identified, cognitive connections

### 7. Core Wound Questions

**Purpose:** Access deepest truth and resistance

- "What's the hardest thing to say about this?"
- "What are you most afraid of feeling?"
- "What's at stake if you don't change this?"
- "What's holding you back from feeling what you want to feel?"

**When to use:** When resistance is present, need to access core wound

### 8. Transformation Questions

**Purpose:** Identify desired state and change

- "How do you want to feel when this is resolved?"
- "What would need to happen for you to feel at peace with this?"
- "What would it mean to you if you could feel differently about this?"

**When to use:** When problem is clear, need to identify desired outcome

## Feeling Extraction Methods

### 1. Keyword Extraction

```python
from music_brain.session.therapy_prompts import extract_feeling_keywords, TherapyPromptBank

prompt = TherapyPromptBank.OPEN_ENDED_PROMPTS[0]
response = "I feel so lost and empty, like there's a weight in my chest"

keywords = extract_feeling_keywords(response, prompt)
# Returns: ["feel", "lost", "empty", "weight"]
```

### 2. Emotional Granularity Analysis

```python
from music_brain.session.therapy_prompts import identify_emotional_granularity

response = "I feel a tight, heavy sensation in my chest, like a dark gray cloud"

granularity = identify_emotional_granularity(response)
# Returns: {
#   "specificity_score": 0.8,
#   "body_references": True,
#   "metaphor_use": True,
#   "emotional_granularity": "high"
# }
```

### 3. Core Element Extraction

```python
from music_brain.session.therapy_prompts import extract_core_elements_from_response

prompt = TherapyPromptBank.CORE_WOUND_PROMPTS[0]
response = "I'm afraid that if I let myself feel this grief, I'll never stop crying"

elements = extract_core_elements_from_response(response, prompt)
# Returns: {
#   "core_resistance": "I'm afraid that if I let myself feel this grief...",
#   "feeling_keywords": ["afraid", "feel", "grief"]
# }
```

## Usage in DAiW

### Basic Usage

```python
from music_brain.session.therapy_prompts import (
    TherapyPromptBank,
    suggest_next_prompt,
    create_casual_therapy_prompt
)

# Get initial prompt
initial = TherapyPromptBank.get_initial_prompt()
print(initial.question)  # "What brings you here today?"

# Get casual version
casual = create_casual_therapy_prompt()
print(casual)  # "I'm curious... what brings you here today?"

# Suggest next prompt based on response
response = "I feel dead inside because I chose safety over freedom"
next_prompt = suggest_next_prompt(response, initial, session_depth=0)
print(next_prompt.question)  # "What's the hardest thing to say about this?"
```

### Integration with TherapySession

```python
from music_brain.structure.comprehensive_engine import TherapySession
from music_brain.session.therapy_prompts import TherapyPromptBank

session = TherapySession()

# Use therapy prompts to guide conversation
prompt = TherapyPromptBank.get_initial_prompt()
user_response = input(prompt.question)

# Process response
affect = session.process_core_input(user_response)

# Get follow-up prompt
next_prompt = TherapyPromptBank.get_follow_up_prompt(prompt)
```

## Prompt Selection Strategy

### Session Flow

1. **Initial (Depth 0):** Open-ended or core wound prompt
   - "What brings you here today?"
   - "What's the hardest thing to say about this?"

2. **Exploration (Depth 1):** Feeling identification
   - Body awareness if vague
   - Emotion-focused if specific

3. **Transformation (Depth 2+):** Solution-focused
   - Miracle question
   - Transformation questions

### By Approach

- **Person-Centered:** Use when user needs space to explore
- **Solution-Focused:** Use when stuck in problem, need forward movement
- **Narrative:** Use when problem feels overwhelming
- **CBT:** Use when patterns need identification
- **Gestalt:** Use when emotions are vague, need body awareness
- **Emotion-Focused:** Use when need to go deeper than surface

## Best Practices

1. **Start Open:** Begin with open-ended questions
2. **Follow the Feeling:** Let responses guide next prompt
3. **Go Deeper Gradually:** Don't jump to core wound too early
4. **Use Body Awareness:** When emotions are vague
5. **Shift to Solution:** When problem is clear, move to transformation
6. **Make it Casual:** Use `create_casual_therapy_prompt()` for natural flow

## Research Basis

These prompts are based on:

- **Person-Centered Therapy:** Rogers, C. (1951). *Client-Centered Therapy*
- **Solution-Focused Therapy:** de Shazer, S. (1985). *Keys to Solution in Brief Therapy*
- **Narrative Therapy:** White, M. & Epston, D. (1990). *Narrative Means to Therapeutic Ends*
- **CBT:** Beck, A. (1979). *Cognitive Therapy of Depression*
- **Gestalt Therapy:** Perls, F. (1973). *The Gestalt Approach*
- **Emotion-Focused Therapy:** Greenberg, L. (2002). *Emotion-Focused Therapy*

## Integration with Intent Schema

Therapy prompts map to DAiW's three-phase intent schema:

- **Phase 0 (Core Wound):** Core wound prompts extract `core_event`, `core_resistance`
- **Phase 1 (Emotional Intent):** Emotion-focused prompts extract `mood_primary`, `vulnerability_scale`
- **Phase 2 (Technical Constraints):** Transformation prompts inform `core_transformation`

Use `extract_core_elements_from_response()` to bridge therapy responses to intent schema.

