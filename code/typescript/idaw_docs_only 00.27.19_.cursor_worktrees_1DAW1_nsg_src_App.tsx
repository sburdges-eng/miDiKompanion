import { useState, useEffect } from "react";
import { useMusicBrain } from "./hooks/useMusicBrain";
import { EmotionWheel, SelectedEmotion } from "./components/EmotionWheel";
import "./App.css";

function App() {
  console.log("App: Component rendering");

  const [sideA, setSideA] = useState(true);
  const [emotions, setEmotions] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [selectedEmotion, setSelectedEmotion] = useState<SelectedEmotion | null>(null);
  const [generatedMidiPath, setGeneratedMidiPath] = useState<string | null>(null);
  const [generatedMidiData, setGeneratedMidiData] = useState<string | null>(null);

  // Always call hooks unconditionally
  const musicBrain = useMusicBrain();
  const { getEmotions, generateMusic, interrogate } = musicBrain;
  console.log("App: useMusicBrain hook initialized");

  // Initialize Tauri API check and Music Brain API status
  useEffect(() => {
    console.log("App: useEffect running, checking Tauri API");
    try {
      // Check if we're in Tauri environment
      if (typeof window !== 'undefined' && (window as any).__TAURI__) {
        console.log("App: Tauri API detected");
      } else {
        console.warn("App: Tauri API not detected - running in browser mode");
      }
    } catch (e) {
      console.error("App: Error checking Tauri API:", e);
    }

    // Check Music Brain API status
    const checkApiStatus = async () => {
      try {
        await getEmotions();
        setApiStatus('online');
        console.log("App: Music Brain API is online");
      } catch (error) {
        setApiStatus('offline');
        console.warn("App: Music Brain API is offline:", error);
      }
    };

    checkApiStatus();
    // Check API status every 30 seconds
    const interval = setInterval(checkApiStatus, 30000);
    return () => clearInterval(interval);
  }, [getEmotions]);

  const toggleSide = () => {
    setSideA(!sideA);
  };

  const handleGetEmotions = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getEmotions();
      setEmotions(result);
      setApiStatus('online');
      console.log('Emotions loaded:', result);
    } catch (error) {
      console.error('Error loading emotions:', error);
      setApiStatus('offline');
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (errorMsg.includes('Connection') || errorMsg.includes('Failed to fetch') || errorMsg.includes('network')) {
        setError('Music Brain API is not running. Start it with: python -m music_brain.api');
      } else {
        setError(`Failed to load emotions: ${errorMsg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateMusic = async () => {
    setLoading(true);
    setError(null);
    try {
      if (!selectedEmotion) {
        setError('Please select an emotion first');
        setLoading(false);
        return;
      }

      // Use new format: base_emotion, intensity, specific_emotion
      const result = await generateMusic({
        intent: {
          base_emotion: selectedEmotion.base,
          intensity: selectedEmotion.intensity,
          specific_emotion: selectedEmotion.sub,
          // Keep technical for overrides if needed
          technical: {
            // Let emotion_mapper determine these, but allow overrides
          }
        }
      });
      setApiStatus('online');
      console.log('Music generated:', result);

      // Extract MIDI data and path from result
      const midiData = (result as any)?.midi_data || null;
      const midiPath = (result as any)?.midi_path || null;
      const musicConfig = (result as any)?.music_config || null;

      setGeneratedMidiPath(midiPath);
      setGeneratedMidiData(midiData);

      if (musicConfig) {
        console.log('Music config:', musicConfig);
      }

      if (midiData || midiPath) {
        // Success - MIDI is available for download
        console.log('MIDI generated successfully');
      } else {
        console.warn('MIDI generation completed but no MIDI data/path returned');
      }
    } catch (error) {
      console.error('Error generating music:', error);
      setApiStatus('offline');
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (errorMsg.includes('Connection') || errorMsg.includes('Failed to fetch') || errorMsg.includes('network')) {
        setError('Music Brain API is not running. Start it with: python -m music_brain.api');
      } else {
        setError(`Failed to generate music: ${errorMsg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadMidi = () => {
    if (!generatedMidiData) {
      setError('No MIDI data available to download');
      return;
    }

    try {
      // Convert base64 to blob
      const binaryString = atob(generatedMidiData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'audio/midi' });

      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `generated_music_${Date.now()}.mid`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading MIDI:', error);
      setError('Failed to download MIDI file');
    }
  };

  const handleInterrogate = async () => {
    setLoading(true);
    setError(null);
    try {
      // Include selected emotion in interrogation message if available
      let message = "I want to write a song about loss";
      if (selectedEmotion) {
        message = `I want to write a song about ${selectedEmotion.base} (${selectedEmotion.intensity}): ${selectedEmotion.sub}`;
      }

      const result = await interrogate({
        message: message
      });
      setApiStatus('online');
      console.log('Interrogation response:', result);
      alert('Interrogation response received! Check console.');
    } catch (error) {
      console.error('Error interrogating:', error);
      setApiStatus('offline');
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (errorMsg.includes('Connection') || errorMsg.includes('Failed to fetch') || errorMsg.includes('network')) {
        setError('Music Brain API is not running. Start it with: python -m music_brain.api');
      } else {
        setError(`Failed to interrogate: ${errorMsg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {error && (
        <div style={{ padding: '10px', background: '#ffebee', border: '1px solid #f44336', borderRadius: '4px', margin: '10px' }}>
          <strong>Error:</strong> {error}
          <button onClick={() => setError(null)} style={{ marginLeft: '10px' }}>Dismiss</button>
        </div>
      )}
      <div className="cassette-header">
        <h1>iDAW - Kelly Project</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '5px 10px',
            borderRadius: '4px',
            backgroundColor: apiStatus === 'online' ? '#e8f5e9' : apiStatus === 'offline' ? '#ffebee' : '#fff3e0',
            border: `1px solid ${apiStatus === 'online' ? '#4caf50' : apiStatus === 'offline' ? '#f44336' : '#ff9800'}`
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: apiStatus === 'online' ? '#4caf50' : apiStatus === 'offline' ? '#f44336' : '#ff9800',
              animation: apiStatus === 'checking' ? 'pulse 2s infinite' : 'none'
            }}></span>
            <span style={{ fontSize: '0.9em' }}>
              API: {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}
            </span>
          </div>
          <button onClick={toggleSide} className="toggle-btn">
            {sideA ? "⏭ Side B" : "⏮ Side A"}
          </button>
        </div>
      </div>

      {sideA ? (
        <div className="side-a">
          <h2>Side A: Professional DAW</h2>
          <p>Mixer, Timeline, Transport controls coming soon...</p>
          <div className="test-buttons">
            <button
              onClick={handleGenerateMusic}
              disabled={loading || !selectedEmotion}
              title={!selectedEmotion ? "Please select an emotion first" : ""}
            >
              {loading ? "Generating..." : selectedEmotion ? `Test Generate Music (${selectedEmotion.sub})` : "Test Generate Music (Select Emotion First)"}
            </button>
          </div>
        </div>
      ) : (
        <div className="side-b">
          <h2>Side B: Therapeutic Interface</h2>

          <div className="emotion-section">
            <h3>Emotion Wheel (6×6×6)</h3>
            <button onClick={handleGetEmotions} disabled={loading}>
              {loading ? "Loading..." : "Load Emotions"}
            </button>
            {emotions && (
              <div style={{ marginTop: '20px' }}>
                <EmotionWheel emotions={emotions} onEmotionSelected={setSelectedEmotion} />
              </div>
            )}
          </div>

          <div className="ghostwriter-section">
            <h3>GhostWriter</h3>
            {selectedEmotion && (
              <div style={{ marginBottom: '10px', padding: '10px', backgroundColor: 'rgba(99, 102, 241, 0.1)', borderRadius: '4px' }}>
                <strong>Selected:</strong> {selectedEmotion.base} → {selectedEmotion.intensity} → {selectedEmotion.sub}
              </div>
            )}
            <button
              onClick={handleGenerateMusic}
              disabled={loading || !selectedEmotion}
              title={!selectedEmotion ? "Please select an emotion first" : ""}
            >
              {loading ? "Generating..." : selectedEmotion ? `Generate Music (${selectedEmotion.sub})` : "Generate Music (Select Emotion First)"}
            </button>
            {(generatedMidiPath || generatedMidiData) && (
              <div style={{ marginTop: '10px', padding: '10px', backgroundColor: 'rgba(76, 175, 80, 0.1)', borderRadius: '4px', border: '1px solid #4caf50' }}>
                <strong>✅ MIDI Generated:</strong>
                {generatedMidiData && (
                  <div style={{ marginTop: '10px' }}>
                    <button
                      onClick={handleDownloadMidi}
                      style={{
                        padding: '8px 16px',
                        backgroundColor: '#4caf50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.9em',
                        fontWeight: 'bold'
                      }}
                    >
                      📥 Download MIDI
                    </button>
                  </div>
                )}
                {generatedMidiPath && (
                  <div style={{ marginTop: '5px', fontSize: '0.8em', fontFamily: 'monospace', wordBreak: 'break-all', color: '#666' }}>
                    Path: {generatedMidiPath}
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="interrogator-section">
            <h3>Interrogator</h3>
            <button onClick={handleInterrogate} disabled={loading}>
              {loading ? "Interrogating..." : "Start Interrogation"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
