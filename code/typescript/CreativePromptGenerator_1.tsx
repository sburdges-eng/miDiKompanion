/**
 * CreativePromptGenerator - Auto-generates contextual creative prompts for B-side
 * Uses emotion, intent, and music theory context to generate meaningful questions
 */

import React, { useState, useEffect, useCallback } from 'react';

interface EmotionContext {
  base?: string;
  intensity?: string;
  specific?: string;
}

interface MusicContext {
  key?: string;
  mode?: string;
  tempo?: number;
  progression?: string[];
}

interface CreativePromptGeneratorProps {
  emotion?: EmotionContext;
  musicContext?: MusicContext;
  phase?: number;
  onPromptSelect?: (prompt: string, category: string) => void;
  autoAdvance?: boolean;
  showAllPrompts?: boolean;
}

interface PromptCategory {
  name: string;
  icon: string;
  color: string;
  prompts: string[];
}

// Emotion-based prompt templates
const EMOTION_PROMPTS: Record<string, string[]> = {
  grief: [
    "What would you say if you could speak to them one more time?",
    "Describe the weight you carry - is it sharp or dull?",
    "What small thing triggers the memory most unexpectedly?",
    "If grief had a texture, what would yours feel like?",
    "What are you afraid to forget?",
    "Where in your body do you feel the absence?",
  ],
  joy: [
    "What made you laugh until you couldn't breathe?",
    "Describe the exact moment everything felt perfect",
    "What color is your happiness right now?",
    "If this feeling had a sound, what would it be?",
    "What do you want to celebrate but haven't yet?",
    "Who do you wish was here to share this moment?",
  ],
  anger: [
    "What words have you been swallowing?",
    "Describe the heat - where does it start, where does it go?",
    "What boundary was crossed that still burns?",
    "If you could shout into a void, what would echo back?",
    "What truth needs to be spoken that you've been protecting others from?",
    "What would justice look like to you?",
  ],
  fear: [
    "What keeps you awake when the room goes dark?",
    "Describe the space between knowing and not knowing",
    "What's the shape of what you're running from?",
    "If courage had a voice, what would it whisper to you?",
    "What's the worst thing that could happen? What's the most likely?",
    "What safety feels like something you haven't earned?",
  ],
  love: [
    "What's the smallest gesture that means everything?",
    "Describe how they smell when they first wake up",
    "What vulnerability have you never shown anyone else?",
    "If love had a taste, what would yours be?",
    "What are you afraid to need from them?",
    "What does 'home' feel like with them?",
  ],
  loneliness: [
    "What silence is the loudest?",
    "Describe the version of yourself that only exists when no one's watching",
    "What connection do you miss that you've never actually had?",
    "If your isolation had walls, what color would they be?",
    "What do you pretend not to need?",
    "Who do you become in a crowded room?",
  ],
};

// Phase-based prompts (Core Wound -> Emotional Intent -> Technical)
const PHASE_PROMPTS: PromptCategory[] = [
  {
    name: "Core Wound",
    icon: "💔",
    color: "#ef4444",
    prompts: [
      "What's the moment that changed everything?",
      "What were you before this happened?",
      "What do you wish you could unknow?",
      "Describe the shape of what you lost",
      "What innocence can't you get back?",
      "What truth did this reveal about the world?",
    ],
  },
  {
    name: "Resistance",
    icon: "🛡️",
    color: "#f59e0b",
    prompts: [
      "What are you protecting by not feeling this?",
      "What would happen if you let go completely?",
      "What mask do you wear that's starting to crack?",
      "What would the people who love you say if they really knew?",
      "What part of you is screaming while you smile?",
      "What comfort are you afraid to accept?",
    ],
  },
  {
    name: "Longing",
    icon: "✨",
    color: "#6366f1",
    prompts: [
      "What do you want so badly it hurts to name?",
      "Describe the life you imagine when you're brave enough to dream",
      "What would healing look like? What would it cost?",
      "What forgiveness are you not ready to give?",
      "What version of yourself is waiting on the other side?",
      "What small hope keeps you moving forward?",
    ],
  },
];

// Music theory prompts
const MUSIC_THEORY_PROMPTS: string[] = [
  "Should this resolve, or stay suspended in wanting?",
  "Does the melody need to soar or stay grounded?",
  "What instrument carries the weight of the story?",
  "Is this a whisper or a scream?",
  "Should the listener feel safe or uncomfortable?",
  "What should the silence between notes hold?",
  "Does this need a bridge or should we stay in one place?",
  "What key would match how your body feels right now?",
];

// Production prompts
const PRODUCTION_PROMPTS: string[] = [
  "Should we hear the room or feel isolated?",
  "What texture should surround the voice?",
  "Should this feel polished or raw?",
  "What should we hear that normally stays hidden?",
  "How close should the listener feel?",
  "What imperfection would make this feel real?",
];

export const CreativePromptGenerator: React.FC<CreativePromptGeneratorProps> = ({
  emotion,
  musicContext,
  phase = 0,
  onPromptSelect,
  autoAdvance = true,
  showAllPrompts = false,
}) => {
  const [currentPrompt, setCurrentPrompt] = useState<string>("");
  const [currentCategory, setCurrentCategory] = useState<string>("");
  const [promptHistory, setPromptHistory] = useState<string[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [displayedText, setDisplayedText] = useState("");
  const [autoMode, setAutoMode] = useState(autoAdvance);

  // Generate a contextual prompt based on current state
  const generatePrompt = useCallback((): { prompt: string; category: string } => {
    let prompts: string[] = [];
    let category = "General";

    // Priority 1: Phase-based prompts
    if (phase >= 0 && phase < PHASE_PROMPTS.length) {
      const phaseCategory = PHASE_PROMPTS[phase];
      prompts = [...phaseCategory.prompts];
      category = phaseCategory.name;
    }

    // Priority 2: Emotion-based prompts
    if (emotion?.base && EMOTION_PROMPTS[emotion.base.toLowerCase()]) {
      const emotionPrompts = EMOTION_PROMPTS[emotion.base.toLowerCase()];
      prompts = [...prompts, ...emotionPrompts];
      if (!category || category === "General") {
        category = `${emotion.base} ${emotion.intensity || ""}`.trim();
      }
    }

    // Priority 3: Music theory prompts (if we have music context)
    if (musicContext?.key || musicContext?.progression) {
      prompts = [...prompts, ...MUSIC_THEORY_PROMPTS];
    }

    // Priority 4: Production prompts (later phases)
    if (phase >= 2) {
      prompts = [...prompts, ...PRODUCTION_PROMPTS];
    }

    // Filter out prompts we've already shown
    const availablePrompts = prompts.filter((p) => !promptHistory.includes(p));

    // If we've used all prompts, reset
    if (availablePrompts.length === 0) {
      setPromptHistory([]);
      return generatePrompt();
    }

    // Select random prompt
    const selectedPrompt = availablePrompts[Math.floor(Math.random() * availablePrompts.length)];

    return { prompt: selectedPrompt, category };
  }, [emotion, musicContext, phase, promptHistory]);

  // Typewriter effect
  useEffect(() => {
    if (!currentPrompt || !isTyping) return;

    let index = 0;
    setDisplayedText("");

    const typeInterval = setInterval(() => {
      if (index < currentPrompt.length) {
        setDisplayedText(currentPrompt.substring(0, index + 1));
        index++;
      } else {
        clearInterval(typeInterval);
        setIsTyping(false);
      }
    }, 30);

    return () => clearInterval(typeInterval);
  }, [currentPrompt, isTyping]);

  // Auto-advance prompts
  useEffect(() => {
    if (!autoMode) return;

    const advanceInterval = setInterval(() => {
      const { prompt, category } = generatePrompt();
      setCurrentPrompt(prompt);
      setCurrentCategory(category);
      setPromptHistory((prev) => [...prev, prompt]);
      setIsTyping(true);
    }, 15000); // New prompt every 15 seconds

    // Initial prompt
    const { prompt, category } = generatePrompt();
    setCurrentPrompt(prompt);
    setCurrentCategory(category);
    setPromptHistory([prompt]);
    setIsTyping(true);

    return () => clearInterval(advanceInterval);
  }, [autoMode, generatePrompt]);

  // Manual next prompt
  const handleNext = () => {
    const { prompt, category } = generatePrompt();
    setCurrentPrompt(prompt);
    setCurrentCategory(category);
    setPromptHistory((prev) => [...prev, prompt]);
    setIsTyping(true);
  };

  // Handle prompt selection
  const handleSelect = () => {
    if (currentPrompt) {
      onPromptSelect?.(currentPrompt, currentCategory);
    }
  };

  return (
    <div
      style={{
        backgroundColor: 'rgba(0, 0, 0, 0.4)',
        borderRadius: '12px',
        padding: '20px',
        color: '#fff',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '15px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '1.2em' }}>
            {PHASE_PROMPTS[phase]?.icon || "💭"}
          </span>
          <span
            style={{
              fontSize: '0.85em',
              color: PHASE_PROMPTS[phase]?.color || '#6366f1',
              fontWeight: 'bold',
            }}
          >
            {currentCategory}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setAutoMode(!autoMode)}
            style={{
              padding: '4px 10px',
              backgroundColor: autoMode ? '#22c55e' : '#333',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.75em',
              cursor: 'pointer',
            }}
          >
            {autoMode ? '⏸ Auto' : '▶ Auto'}
          </button>
          <button
            onClick={handleNext}
            style={{
              padding: '4px 10px',
              backgroundColor: '#333',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.75em',
              cursor: 'pointer',
            }}
          >
            Next →
          </button>
        </div>
      </div>

      {/* Main prompt display */}
      <div
        style={{
          minHeight: '80px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '8px',
          marginBottom: '15px',
        }}
      >
        <p
          style={{
            fontSize: '1.3em',
            fontStyle: 'italic',
            textAlign: 'center',
            lineHeight: '1.5',
            color: '#e0e0e0',
            margin: 0,
          }}
        >
          "{displayedText}"
          {isTyping && (
            <span
              style={{
                display: 'inline-block',
                width: '2px',
                height: '1em',
                backgroundColor: '#6366f1',
                marginLeft: '2px',
                animation: 'blink 1s infinite',
              }}
            />
          )}
        </p>
      </div>

      {/* Action buttons */}
      <div
        style={{
          display: 'flex',
          gap: '10px',
          justifyContent: 'center',
        }}
      >
        <button
          onClick={handleSelect}
          style={{
            padding: '10px 24px',
            backgroundColor: '#6366f1',
            border: 'none',
            borderRadius: '6px',
            color: '#fff',
            fontSize: '0.9em',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'transform 0.1s',
          }}
          onMouseOver={(e) => (e.currentTarget.style.transform = 'scale(1.05)')}
          onMouseOut={(e) => (e.currentTarget.style.transform = 'scale(1)')}
        >
          Use This Prompt
        </button>
      </div>

      {/* Show all prompts toggle */}
      {showAllPrompts && (
        <div
          style={{
            marginTop: '20px',
            padding: '15px',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '8px',
          }}
        >
          <div style={{ fontSize: '0.85em', color: '#888', marginBottom: '10px' }}>
            All prompts for current phase:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {(phase >= 0 && phase < PHASE_PROMPTS.length
              ? PHASE_PROMPTS[phase].prompts
              : []
            ).map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setCurrentPrompt(prompt);
                  setCurrentCategory(PHASE_PROMPTS[phase]?.name || "");
                  setIsTyping(true);
                }}
                style={{
                  padding: '6px 12px',
                  backgroundColor: currentPrompt === prompt ? '#6366f1' : '#222',
                  border: '1px solid #333',
                  borderRadius: '4px',
                  color: '#ddd',
                  fontSize: '0.75em',
                  cursor: 'pointer',
                  maxWidth: '200px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {prompt.substring(0, 40)}...
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Phase indicator */}
      <div
        style={{
          marginTop: '15px',
          display: 'flex',
          justifyContent: 'center',
          gap: '8px',
        }}
      >
        {PHASE_PROMPTS.map((p, idx) => (
          <div
            key={idx}
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: idx <= phase ? p.color : '#333',
              transition: 'background-color 0.3s',
            }}
          />
        ))}
      </div>

      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </div>
  );
};
