import React, { useState } from 'react';

interface VoiceProfile {
  name: string;
  pitch: number;
  formant: number;
  breathiness: number;
  vibrato: number;
  warmth: number;
}

interface VocalSynthProps {
  onVoiceChange?: (profile: VoiceProfile) => void;
  onGenerate?: (text: string, profile: VoiceProfile) => void;
}

const VOICE_PRESETS: { [key: string]: VoiceProfile } = {
  natural: {
    name: 'Natural',
    pitch: 0,
    formant: 0,
    breathiness: 20,
    vibrato: 30,
    warmth: 50,
  },
  intimate: {
    name: 'Intimate',
    pitch: -2,
    formant: -1,
    breathiness: 40,
    vibrato: 15,
    warmth: 70,
  },
  powerful: {
    name: 'Powerful',
    pitch: 2,
    formant: 1,
    breathiness: 10,
    vibrato: 40,
    warmth: 40,
  },
  ethereal: {
    name: 'Ethereal',
    pitch: 5,
    formant: 3,
    breathiness: 50,
    vibrato: 60,
    warmth: 60,
  },
  raspy: {
    name: 'Raspy',
    pitch: -3,
    formant: -2,
    breathiness: 60,
    vibrato: 20,
    warmth: 30,
  },
  robotic: {
    name: 'Robotic',
    pitch: 0,
    formant: 0,
    breathiness: 0,
    vibrato: 0,
    warmth: 20,
  },
};

const EMOTION_STYLES: { [key: string]: { description: string; adjustments: Partial<VoiceProfile> } } = {
  grief: {
    description: 'Fragile, breaking voice with catch in throat',
    adjustments: { breathiness: 50, vibrato: 15, warmth: 60, pitch: -1 },
  },
  anger: {
    description: 'Tight, controlled tension with sharp edges',
    adjustments: { breathiness: 10, vibrato: 5, warmth: 20, pitch: 2 },
  },
  joy: {
    description: 'Bright, lifted tone with natural energy',
    adjustments: { breathiness: 20, vibrato: 40, warmth: 70, pitch: 3 },
  },
  longing: {
    description: 'Distant, reaching quality with ache',
    adjustments: { breathiness: 35, vibrato: 25, warmth: 55, pitch: 0 },
  },
  peace: {
    description: 'Settled, grounded with gentle flow',
    adjustments: { breathiness: 30, vibrato: 20, warmth: 65, pitch: -1 },
  },
};

export const VocalSynth: React.FC<VocalSynthProps> = ({
  onVoiceChange,
  onGenerate,
}) => {
  const [profile, setProfile] = useState<VoiceProfile>(VOICE_PRESETS.natural);
  const [lyrics, setLyrics] = useState('');
  const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);

  const updateProfile = (changes: Partial<VoiceProfile>) => {
    const newProfile = { ...profile, ...changes };
    setProfile(newProfile);
    onVoiceChange?.(newProfile);
  };

  const applyPreset = (presetName: string) => {
    const preset = VOICE_PRESETS[presetName];
    if (preset) {
      setProfile(preset);
      onVoiceChange?.(preset);
    }
  };

  const applyEmotionStyle = (emotion: string) => {
    const style = EMOTION_STYLES[emotion];
    if (style) {
      setSelectedEmotion(emotion);
      updateProfile(style.adjustments);
    }
  };

  const handleGenerate = async () => {
    if (!lyrics.trim()) return;

    setIsGenerating(true);
    try {
      // Call API for vocal synthesis
      const response = await fetch('http://localhost:8000/voice/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: lyrics,
          profile: profile,
          emotion: selectedEmotion,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setGeneratedAudio(data.audio_url || null);
        onGenerate?.(lyrics, profile);
      }
    } catch (error) {
      console.error('Vocal synthesis error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      borderRadius: '8px',
      padding: '20px',
      color: '#fff',
    }}>
      <h3 style={{ margin: '0 0 15px 0', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span>Vocal Synth</span>
        <span style={{ fontSize: '0.6em', color: '#888' }}>AI Voice Generation</span>
      </h3>

      {/* Voice Presets */}
      <div style={{ marginBottom: '20px' }}>
        <label style={{ fontSize: '0.85em', color: '#888', marginBottom: '8px', display: 'block' }}>
          Voice Character
        </label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {Object.entries(VOICE_PRESETS).map(([key, preset]) => (
            <button
              key={key}
              onClick={() => applyPreset(key)}
              style={{
                padding: '8px 16px',
                backgroundColor: profile.name === preset.name ? '#6366f1' : '#333',
                border: 'none',
                borderRadius: '20px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.85em',
                transition: 'all 0.2s',
              }}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* Emotion Styles */}
      <div style={{ marginBottom: '20px' }}>
        <label style={{ fontSize: '0.85em', color: '#888', marginBottom: '8px', display: 'block' }}>
          Emotional Style
        </label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {Object.entries(EMOTION_STYLES).map(([key, style]) => (
            <button
              key={key}
              onClick={() => applyEmotionStyle(key)}
              title={style.description}
              style={{
                padding: '8px 16px',
                backgroundColor: selectedEmotion === key ? '#ec4899' : '#333',
                border: 'none',
                borderRadius: '20px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.85em',
                transition: 'all 0.2s',
              }}
            >
              {key.charAt(0).toUpperCase() + key.slice(1)}
            </button>
          ))}
        </div>
        {selectedEmotion && (
          <div style={{
            marginTop: '8px',
            padding: '8px 12px',
            backgroundColor: 'rgba(236, 72, 153, 0.1)',
            borderRadius: '4px',
            fontSize: '0.8em',
            color: '#ec4899',
          }}>
            {EMOTION_STYLES[selectedEmotion].description}
          </div>
        )}
      </div>

      {/* Voice Parameters */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '15px',
        marginBottom: '20px',
      }}>
        <div>
          <label style={{ fontSize: '0.8em', color: '#888' }}>Pitch</label>
          <input
            type="range"
            min="-12"
            max="12"
            value={profile.pitch}
            onChange={(e) => updateProfile({ pitch: Number(e.target.value) })}
            style={{ width: '100%' }}
          />
          <div style={{ fontSize: '0.75em', textAlign: 'center' }}>
            {profile.pitch > 0 ? '+' : ''}{profile.pitch} semitones
          </div>
        </div>

        <div>
          <label style={{ fontSize: '0.8em', color: '#888' }}>Formant</label>
          <input
            type="range"
            min="-5"
            max="5"
            value={profile.formant}
            onChange={(e) => updateProfile({ formant: Number(e.target.value) })}
            style={{ width: '100%' }}
          />
          <div style={{ fontSize: '0.75em', textAlign: 'center' }}>
            {profile.formant > 0 ? '+' : ''}{profile.formant}
          </div>
        </div>

        <div>
          <label style={{ fontSize: '0.8em', color: '#888' }}>Breathiness</label>
          <input
            type="range"
            min="0"
            max="100"
            value={profile.breathiness}
            onChange={(e) => updateProfile({ breathiness: Number(e.target.value) })}
            style={{ width: '100%' }}
          />
          <div style={{ fontSize: '0.75em', textAlign: 'center' }}>
            {profile.breathiness}%
          </div>
        </div>

        <div>
          <label style={{ fontSize: '0.8em', color: '#888' }}>Vibrato</label>
          <input
            type="range"
            min="0"
            max="100"
            value={profile.vibrato}
            onChange={(e) => updateProfile({ vibrato: Number(e.target.value) })}
            style={{ width: '100%' }}
          />
          <div style={{ fontSize: '0.75em', textAlign: 'center' }}>
            {profile.vibrato}%
          </div>
        </div>

        <div>
          <label style={{ fontSize: '0.8em', color: '#888' }}>Warmth</label>
          <input
            type="range"
            min="0"
            max="100"
            value={profile.warmth}
            onChange={(e) => updateProfile({ warmth: Number(e.target.value) })}
            style={{ width: '100%' }}
          />
          <div style={{ fontSize: '0.75em', textAlign: 'center' }}>
            {profile.warmth}%
          </div>
        </div>
      </div>

      {/* Lyrics Input */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ fontSize: '0.85em', color: '#888', marginBottom: '8px', display: 'block' }}>
          Lyrics / Text to Sing
        </label>
        <textarea
          value={lyrics}
          onChange={(e) => setLyrics(e.target.value)}
          placeholder="Enter lyrics or phrases to synthesize..."
          style={{
            width: '100%',
            height: '100px',
            padding: '12px',
            backgroundColor: '#222',
            border: '1px solid #444',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '0.9em',
            resize: 'vertical',
          }}
        />
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={isGenerating || !lyrics.trim()}
        style={{
          width: '100%',
          padding: '12px',
          backgroundColor: isGenerating ? '#666' : '#6366f1',
          border: 'none',
          borderRadius: '8px',
          color: '#fff',
          fontSize: '1em',
          fontWeight: 'bold',
          cursor: isGenerating || !lyrics.trim() ? 'not-allowed' : 'pointer',
          opacity: !lyrics.trim() ? 0.5 : 1,
        }}
      >
        {isGenerating ? 'Generating...' : 'Generate Vocal'}
      </button>

      {/* Generated Audio Preview */}
      {generatedAudio && (
        <div style={{
          marginTop: '15px',
          padding: '15px',
          backgroundColor: '#222',
          borderRadius: '8px',
        }}>
          <div style={{ marginBottom: '10px', fontWeight: 'bold' }}>Generated Audio</div>
          <audio controls style={{ width: '100%' }}>
            <source src={generatedAudio} type="audio/wav" />
          </audio>
        </div>
      )}

      {/* Voice Visualization */}
      <div style={{
        marginTop: '20px',
        padding: '15px',
        backgroundColor: '#0f0f0f',
        borderRadius: '8px',
      }}>
        <div style={{ fontSize: '0.85em', color: '#888', marginBottom: '10px' }}>
          Voice Character Preview
        </div>
        <div style={{
          display: 'flex',
          height: '60px',
          alignItems: 'flex-end',
          gap: '4px',
        }}>
          {[
            { label: 'Pitch', value: (profile.pitch + 12) / 24, color: '#6366f1' },
            { label: 'Formant', value: (profile.formant + 5) / 10, color: '#ec4899' },
            { label: 'Breath', value: profile.breathiness / 100, color: '#10b981' },
            { label: 'Vibrato', value: profile.vibrato / 100, color: '#f59e0b' },
            { label: 'Warmth', value: profile.warmth / 100, color: '#ef4444' },
          ].map((param, idx) => (
            <div key={idx} style={{ flex: 1, textAlign: 'center' }}>
              <div
                style={{
                  height: `${Math.max(5, param.value * 50)}px`,
                  backgroundColor: param.color,
                  borderRadius: '2px 2px 0 0',
                  margin: '0 auto',
                  width: '80%',
                  transition: 'height 0.2s',
                }}
              />
              <div style={{ fontSize: '0.65em', color: '#666', marginTop: '4px' }}>
                {param.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
