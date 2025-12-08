import { useState } from "react";
import { useMusicBrain } from "./hooks/useMusicBrain";
import "./App.css";

function App() {
  const [sideA, setSideA] = useState(true);
  const [emotions, setEmotions] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const { getEmotions, generateMusic, interrogate } = useMusicBrain();

  const toggleSide = () => {
    setSideA(!sideA);
  };

  const handleGetEmotions = async () => {
    setLoading(true);
    try {
      const result = await getEmotions();
      setEmotions(result);
      console.log('Emotions loaded:', result);
    } catch (error) {
      console.error('Error loading emotions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateMusic = async () => {
    setLoading(true);
    try {
      const result = await generateMusic({
        intent: {
          emotional_intent: "grief hidden as love",
          technical: {
            key: "F major",
            bpm: 82,
            progression: ["F", "C", "Am", "Dm"],
            genre: "lo-fi bedroom emo"
          }
        }
      });
      console.log('Music generated:', result);
      alert('Music generated! Check console for details.');
    } catch (error) {
      console.error('Error generating music:', error);
      alert('Failed to generate music. Check console.');
    } finally {
      setLoading(false);
    }
  };

  const handleInterrogate = async () => {
    setLoading(true);
    try {
      const result = await interrogate({
        message: "I want to write a song about loss"
      });
      console.log('Interrogation response:', result);
      alert('Interrogation response received! Check console.');
    } catch (error) {
      console.error('Error interrogating:', error);
      alert('Failed to interrogate. Check console.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="cassette-header">
        <h1>iDAW - Kelly Project</h1>
        <button onClick={toggleSide} className="toggle-btn">
          {sideA ? "⏭ Side B" : "⏮ Side A"}
        </button>
      </div>

      {sideA ? (
        <div className="side-a">
          <h2>Side A: Professional DAW</h2>
          <p>Mixer, Timeline, Transport controls coming soon...</p>
          <div className="test-buttons">
            <button onClick={handleGenerateMusic} disabled={loading}>
              {loading ? "Generating..." : "Test Generate Music"}
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
              <div className="emotion-preview">
                <p>Total emotion nodes: {emotions.total_nodes}</p>
                <p>Base emotions: {Object.keys(emotions.emotions).length}</p>
              </div>
            )}
          </div>

          <div className="ghostwriter-section">
            <h3>GhostWriter</h3>
            <button onClick={handleGenerateMusic} disabled={loading}>
              {loading ? "Generating..." : "Generate Music"}
            </button>
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
