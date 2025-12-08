import React, { useState, useEffect, useRef } from 'react';

interface AutoPromptGeneratorProps {
  selectedEmotion?: {
    base: string;
    intensity: string;
    sub: string;
  } | null;
  onPromptGenerated?: (prompt: string) => void;
  autoGenerate?: boolean;
}

// Prompt templates based on emotions
const PROMPT_TEMPLATES: Record<string, string[]> = {
  grief: [
    "I'm writing about the weight of loss",
    "This song is about what I couldn't say",
    "I need to express the ache of missing someone",
    "I want to capture the silence after goodbye",
    "This is about the memories that won't fade",
  ],
  joy: [
    "I'm writing about that moment everything clicked",
    "This song captures pure happiness",
    "I want to express overwhelming gratitude",
    "This is about finding light in darkness",
    "I'm celebrating a breakthrough moment",
  ],
  anger: [
    "I'm writing about being pushed too far",
    "This song is about standing up for myself",
    "I need to express this burning frustration",
    "I want to capture the fire of injustice",
    "This is about breaking free from control",
  ],
  fear: [
    "I'm writing about the anxiety that won't leave",
    "This song is about facing the unknown",
    "I need to express this overwhelming worry",
    "I want to capture the feeling of being trapped",
    "This is about the fear of losing everything",
  ],
  love: [
    "I'm writing about that first moment I knew",
    "This song captures deep connection",
    "I want to express how they make me feel",
    "This is about finding home in someone",
    "I'm celebrating this beautiful bond",
  ],
  longing: [
    "I'm writing about what I can't have",
    "This song is about waiting for something",
    "I need to express this deep yearning",
    "I want to capture the ache of distance",
    "This is about hoping for what's missing",
  ],
};

// Intensity modifiers
const INTENSITY_MODIFIERS: Record<string, string[]> = {
  low: ["gently", "softly", "quietly", "subtly", "lightly"],
  moderate: ["clearly", "directly", "honestly", "openly", "sincerely"],
  high: ["intensely", "powerfully", "deeply", "urgently", "desperately"],
};

// Generate contextual prompt
const generatePrompt = (
  baseEmotion: string,
  intensity: string,
  subEmotion: string
): string => {
  const templates = PROMPT_TEMPLATES[baseEmotion.toLowerCase()] || [
    `I'm writing about ${baseEmotion.toLowerCase()}`,
  ];
  const modifiers = INTENSITY_MODIFIERS[intensity.toLowerCase()] || ["honestly"];

  const template = templates[Math.floor(Math.random() * templates.length)];
  const modifier = modifiers[Math.floor(Math.random() * modifiers.length)];

  // Combine into natural prompt
  const variations = [
    `${template}, ${modifier} expressing ${subEmotion.toLowerCase()}.`,
    `I want to ${modifier} write about ${subEmotion.toLowerCase()}.`,
    `This song ${modifier} explores ${subEmotion.toLowerCase()}.`,
    `I'm ${modifier} trying to capture ${subEmotion.toLowerCase()}.`,
  ];

  return variations[Math.floor(Math.random() * variations.length)];
};

export const AutoPromptGenerator: React.FC<AutoPromptGeneratorProps> = ({
  selectedEmotion,
  onPromptGenerated,
  autoGenerate = true,
}) => {
  const [currentPrompt, setCurrentPrompt] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [promptHistory, setPromptHistory] = useState<string[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (autoGenerate && selectedEmotion) {
      // Generate initial prompt
      generateNewPrompt();

      // Auto-regenerate every 30 seconds if emotion changes
      intervalRef.current = setInterval(() => {
        generateNewPrompt();
      }, 30000);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [selectedEmotion, autoGenerate]);

  const generateNewPrompt = () => {
    if (!selectedEmotion) return;

    setIsGenerating(true);

    // Simulate generation delay for realism
    setTimeout(() => {
      const prompt = generatePrompt(
        selectedEmotion.base,
        selectedEmotion.intensity,
        selectedEmotion.sub
      );
      setCurrentPrompt(prompt);
      setPromptHistory((prev) => [prompt, ...prev.slice(0, 4)]);
      onPromptGenerated?.(prompt);
      setIsGenerating(false);
    }, 500);
  };

  if (!selectedEmotion) {
    return (
      <div
        style={{
          padding: '20px',
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '8px',
          textAlign: 'center',
          color: '#888',
          fontStyle: 'italic',
        }}
      >
        Select an emotion to generate prompts
      </div>
    );
  }

  return (
    <div
      style={{
        padding: '20px',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        borderRadius: '8px',
        border: '1px solid rgba(99, 102, 241, 0.3)',
      }}
    >
      <div style={{ marginBottom: '15px' }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#fff' }}>
          Auto-Generated Prompt
        </h4>
        <div
          style={{
            padding: '15px',
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            borderRadius: '6px',
            minHeight: '60px',
            display: 'flex',
            alignItems: 'center',
            position: 'relative',
          }}
        >
          {isGenerating ? (
            <div
              style={{
                color: '#888',
                fontStyle: 'italic',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}
            >
              <span
                style={{
                  display: 'inline-block',
                  width: '20px',
                  height: '20px',
                  border: '2px solid #6366f1',
                  borderTopColor: 'transparent',
                  borderRadius: '50%',
                  animation: 'spin 0.8s linear infinite',
                }}
              />
              Generating...
            </div>
          ) : (
            <p
              style={{
                margin: 0,
                color: '#fff',
                fontSize: '1em',
                lineHeight: '1.5',
              }}
            >
              {currentPrompt || 'Click generate to create a prompt'}
            </p>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
        <button
          onClick={generateNewPrompt}
          disabled={isGenerating}
          style={{
            flex: 1,
            padding: '10px',
            backgroundColor: '#6366f1',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isGenerating ? 'not-allowed' : 'pointer',
            fontWeight: 'bold',
            opacity: isGenerating ? 0.6 : 1,
          }}
        >
          {isGenerating ? 'Generating...' : '🔄 Generate New'}
        </button>
        {currentPrompt && (
          <button
            onClick={() => {
              // Copy to clipboard
              navigator.clipboard.writeText(currentPrompt);
            }}
            style={{
              padding: '10px 15px',
              backgroundColor: '#333',
              color: 'white',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
            title="Copy prompt"
          >
            📋
          </button>
        )}
      </div>

      {promptHistory.length > 0 && (
        <div>
          <div
            style={{
              fontSize: '0.85em',
              color: '#888',
              marginBottom: '8px',
            }}
          >
            Recent prompts:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {promptHistory.slice(0, 3).map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setCurrentPrompt(prompt);
                  onPromptGenerated?.(prompt);
                }}
                style={{
                  padding: '8px 12px',
                  backgroundColor: 'rgba(0, 0, 0, 0.2)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '4px',
                  color: '#ccc',
                  fontSize: '0.85em',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(0, 0, 0, 0.2)';
                }}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};
