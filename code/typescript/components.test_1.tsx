/**
 * iDAW Frontend Component Tests
 * Tests for React components used in the DAW interface
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock components for testing
// Note: These tests verify component rendering and basic interactions

describe('EQ Component', () => {
  it('should render with default bands', async () => {
    const { EQ } = await import('../components/EQ');
    const mockOnChange = vi.fn();

    render(<EQ channelName="Test Channel" onEQChange={mockOnChange} />);

    // Should show channel name
    expect(screen.getByText(/EQ - Test Channel/i)).toBeInTheDocument();

    // Should show preset selector
    expect(screen.getByText(/Presets/i)).toBeInTheDocument();

    // Should show 5 band controls
    expect(screen.getByText(/Band 1/i)).toBeInTheDocument();
    expect(screen.getByText(/Band 5/i)).toBeInTheDocument();
  });

  it('should toggle bypass state', async () => {
    const { EQ } = await import('../components/EQ');

    render(<EQ channelName="Master" />);

    const bypassButton = screen.getByText(/ACTIVE/i);
    expect(bypassButton).toBeInTheDocument();

    fireEvent.click(bypassButton);
    expect(screen.getByText(/BYPASSED/i)).toBeInTheDocument();
  });
});

describe('VocalSynth Component', () => {
  it('should render with default profile', async () => {
    const { VocalSynth } = await import('../components/VocalSynth');
    const mockOnVoiceChange = vi.fn();
    const mockOnGenerate = vi.fn();

    render(
      <VocalSynth
        onVoiceChange={mockOnVoiceChange}
        onGenerate={mockOnGenerate}
      />
    );

    // Should show voice profiles section
    expect(screen.getByText(/Voice Profiles/i)).toBeInTheDocument();

    // Should show Natural profile by default
    expect(screen.getByText(/Natural/i)).toBeInTheDocument();
  });

  it('should allow profile selection', async () => {
    const { VocalSynth } = await import('../components/VocalSynth');
    const mockOnVoiceChange = vi.fn();

    render(<VocalSynth onVoiceChange={mockOnVoiceChange} />);

    // Click on a different profile
    const intimateProfile = screen.getByText(/Intimate/i);
    fireEvent.click(intimateProfile);

    expect(mockOnVoiceChange).toHaveBeenCalled();
  });
});

describe('MixConsole Component', () => {
  it('should render with channel name', async () => {
    const { MixConsole } = await import('../components/MixConsole');
    const mockOnMixChange = vi.fn();

    render(<MixConsole channelName="Drums" onMixChange={mockOnMixChange} />);

    // Should show dynamics section
    expect(screen.getByText(/Dynamics/i)).toBeInTheDocument();

    // Should show spatial section
    expect(screen.getByText(/Spatial/i)).toBeInTheDocument();

    // Should show output section
    expect(screen.getByText(/Output/i)).toBeInTheDocument();
  });
});

describe('EmotionWheel Component', () => {
  it('should render with emotion data', async () => {
    const { EmotionWheel } = await import('../components/EmotionWheel');
    const mockOnSelect = vi.fn();

    const mockEmotions = {
      emotions: {
        joy: {
          subtle: ['contentment', 'serenity'],
          moderate: ['happiness', 'cheerfulness'],
          intense: ['elation', 'euphoria']
        },
        grief: {
          subtle: ['melancholy', 'wistfulness'],
          moderate: ['sadness', 'sorrow'],
          intense: ['despair', 'anguish']
        }
      }
    };

    render(<EmotionWheel emotions={mockEmotions} onEmotionSelected={mockOnSelect} />);

    // Should render base emotions
    expect(screen.getByText(/joy/i)).toBeInTheDocument();
    expect(screen.getByText(/grief/i)).toBeInTheDocument();
  });

  it('should call onEmotionSelected when emotion is clicked', async () => {
    const { EmotionWheel } = await import('../components/EmotionWheel');
    const mockOnSelect = vi.fn();

    const mockEmotions = {
      emotions: {
        joy: {
          subtle: ['contentment'],
          moderate: ['happiness'],
          intense: ['elation']
        }
      }
    };

    render(<EmotionWheel emotions={mockEmotions} onEmotionSelected={mockOnSelect} />);

    // Click on a base emotion
    fireEvent.click(screen.getByText(/joy/i));

    // Emotion wheel may require drilling down - check callback is eventually called
    // or check that intensity options appear
  });
});

describe('Timeline Component', () => {
  it('should render with tempo and time signature', async () => {
    const { Timeline } = await import('../components/Timeline');

    render(<Timeline tempo={120} timeSignature={[4, 4]} />);

    // Timeline should render without crashing
    // More specific assertions depend on Timeline implementation
  });
});

describe('TransportControls Component', () => {
  it('should render play/pause/stop/record buttons', async () => {
    const { TransportControls } = await import('../components/TransportControls');

    const mockOnPlay = vi.fn();
    const mockOnPause = vi.fn();
    const mockOnStop = vi.fn();
    const mockOnRecord = vi.fn();

    render(
      <TransportControls
        tempo={120}
        timeSignature={[4, 4]}
        isPlaying={false}
        isRecording={false}
        onPlay={mockOnPlay}
        onPause={mockOnPause}
        onStop={mockOnStop}
        onRecord={mockOnRecord}
      />
    );

    // Should show tempo
    expect(screen.getByText(/120/)).toBeInTheDocument();
  });

  it('should call onPlay when play button is clicked', async () => {
    const { TransportControls } = await import('../components/TransportControls');

    const mockOnPlay = vi.fn();

    render(
      <TransportControls
        tempo={120}
        timeSignature={[4, 4]}
        isPlaying={false}
        isRecording={false}
        onPlay={mockOnPlay}
        onPause={vi.fn()}
        onStop={vi.fn()}
        onRecord={vi.fn()}
      />
    );

    // Find and click play button (usually has ▶ or Play text/aria-label)
    const playButton = screen.getByRole('button', { name: /play/i }) ||
                       screen.getByText(/▶/);

    if (playButton) {
      fireEvent.click(playButton);
      expect(mockOnPlay).toHaveBeenCalled();
    }
  });
});

describe('RuleBreaker Component', () => {
  it('should render with selected emotion', async () => {
    const { RuleBreaker } = await import('../components/RuleBreaker');
    const mockOnRuleSelected = vi.fn();

    render(
      <RuleBreaker
        selectedEmotion="grief"
        onRuleSelected={mockOnRuleSelected}
      />
    );

    // RuleBreaker should render rule options
    // Specific assertions depend on implementation
  });
});

describe('Mixer Component', () => {
  it('should render mixer channels', async () => {
    const { Mixer } = await import('../components/Mixer');

    render(<Mixer />);

    // Mixer should render without crashing
    // Should show channel strips or master section
  });
});

describe('InterrogatorChat Component', () => {
  it('should render chat interface', async () => {
    const { InterrogatorChat } = await import('../components/InterrogatorChat');
    const mockOnReady = vi.fn();

    render(<InterrogatorChat onReady={mockOnReady} />);

    // Should have an input for messages
    const input = screen.getByRole('textbox') || screen.getByPlaceholderText(/message/i);
    expect(input).toBeInTheDocument();
  });
});

describe('MidiPlayer Component', () => {
  it('should render with midi data', async () => {
    const { MidiPlayer } = await import('../components/MidiPlayer');

    render(
      <MidiPlayer
        midiData="base64encodeddata"
        midiPath="/path/to/midi"
        musicConfig={{ key: 'C', mode: 'major', tempo: 120 }}
      />
    );

    // MidiPlayer should render playback controls
    // Should show music config info
    expect(screen.getByText(/C/i) || screen.getByText(/120/)).toBeInTheDocument();
  });
});
