// VirtualInstrumentsPanel - UI for Virtual Instruments Engine

import React, { useState, useEffect } from 'react';
import {
  VirtualInstrumentsEngine,
  SamplerInstrument,
  Synthesizer,
  DrumMachine,
  OrchestralInstrument,
} from './VirtualInstrumentsEngine';

interface VirtualInstrumentsPanelProps {
  engine: VirtualInstrumentsEngine;
  onInstrumentChange?: () => void;
}

type InstrumentTab = 'synth' | 'sampler' | 'drums' | 'orchestral';

export const VirtualInstrumentsPanel: React.FC<VirtualInstrumentsPanelProps> = ({
  engine,
  onInstrumentChange,
}) => {
  const [activeTab, setActiveTab] = useState<InstrumentTab>('synth');
  const [synths, setSynths] = useState<Synthesizer[]>([]);
  const [samplers, setSamplers] = useState<SamplerInstrument[]>([]);
  const [drumMachines, setDrumMachines] = useState<DrumMachine[]>([]);
  const [orchestral, setOrchestral] = useState<OrchestralInstrument[]>([]);
  const [selectedInstrument, setSelectedInstrument] = useState<string | null>(null);

  useEffect(() => {
    refreshInstruments();
  }, [engine]);

  const refreshInstruments = () => {
    setSynths(engine.getAllSynthesizers());
    setSamplers(engine.getAllSamplers());
    setDrumMachines(engine.getAllDrumMachines());
    setOrchestral(engine.getAllOrchestralInstruments());
  };

  const handleCreateSynth = (type: 'subtractive' | 'fm' | 'wavetable' | 'granular') => {
    let synth: Synthesizer;
    switch (type) {
      case 'fm':
        synth = engine.createFMSynth(`FM Synth ${synths.length + 1}`);
        break;
      case 'wavetable':
        synth = engine.createWavetableSynth(`Wavetable ${synths.length + 1}`);
        break;
      case 'granular':
        synth = engine.createGranularSynth(`Granular ${synths.length + 1}`);
        break;
      default:
        synth = engine.createSynthesizer(`Synth ${synths.length + 1}`, type);
    }
    setSelectedInstrument(synth.id);
    refreshInstruments();
    onInstrumentChange?.();
  };

  const handleCreateSampler = (type: 'multi-sample' | 'wavetable' | 'granular') => {
    const sampler = engine.createSampler(`Sampler ${samplers.length + 1}`, type);
    setSelectedInstrument(sampler.id);
    refreshInstruments();
    onInstrumentChange?.();
  };

  const handleCreateDrumMachine = (kit: string) => {
    const drum = engine.createDrumMachine(`Drums ${drumMachines.length + 1}`, kit);
    setSelectedInstrument(drum.id);
    refreshInstruments();
    onInstrumentChange?.();
  };

  const handleCreateOrchestral = (category: OrchestralInstrument['category']) => {
    const categoryNames: Record<string, string> = {
      strings: 'Strings',
      woodwinds: 'Woodwinds',
      brass: 'Brass',
      percussion: 'Percussion',
      keyboards: 'Keys',
      choir: 'Choir',
    };
    const inst = engine.createOrchestralInstrument(
      `${categoryNames[category]} ${orchestral.length + 1}`,
      category
    );
    setSelectedInstrument(inst.id);
    refreshInstruments();
    onInstrumentChange?.();
  };

  const renderSynthPanel = (synth: Synthesizer) => (
    <div style={styles.instrumentPanel}>
      <h4 style={styles.panelTitle}>{synth.name}</h4>
      <div style={styles.paramSection}>
        <h5 style={styles.sectionTitle}>Oscillators</h5>
        {synth.voices.oscillators.map((osc, index) => (
          <div key={osc.id} style={styles.oscRow}>
            <span style={styles.label}>OSC {index + 1}</span>
            <select
              value={osc.waveform}
              onChange={(e) => {
                engine.setOscillatorWaveform(synth.id, osc.id, e.target.value as any);
                refreshInstruments();
              }}
              style={styles.select}
            >
              <option value="sine">Sine</option>
              <option value="triangle">Triangle</option>
              <option value="sawtooth">Sawtooth</option>
              <option value="square">Square</option>
              <option value="pulse">Pulse</option>
            </select>
            <input
              type="range"
              min="-100"
              max="100"
              value={osc.detune}
              onChange={(e) => {
                engine.setOscillatorDetune(synth.id, osc.id, parseInt(e.target.value));
                refreshInstruments();
              }}
              style={styles.slider}
            />
            <span style={styles.value}>{osc.detune}ct</span>
          </div>
        ))}
        <button
          onClick={() => {
            engine.addOscillator(synth.id);
            refreshInstruments();
          }}
          style={styles.addButton}
          disabled={synth.voices.oscillators.length >= 8}
        >
          + Add Oscillator
        </button>
      </div>

      <div style={styles.paramSection}>
        <h5 style={styles.sectionTitle}>Filter</h5>
        <div style={styles.paramRow}>
          <span style={styles.label}>Type</span>
          <select
            value={synth.voices.filter.type}
            onChange={(e) => {
              engine.setSynthFilter(synth.id, { type: e.target.value as any });
              refreshInstruments();
            }}
            style={styles.select}
          >
            <option value="lowpass">Lowpass</option>
            <option value="highpass">Highpass</option>
            <option value="bandpass">Bandpass</option>
            <option value="notch">Notch</option>
          </select>
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Cutoff</span>
          <input
            type="range"
            min="20"
            max="20000"
            value={synth.voices.filter.frequency}
            onChange={(e) => {
              engine.setSynthFilter(synth.id, { frequency: parseInt(e.target.value) });
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{synth.voices.filter.frequency}Hz</span>
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Resonance</span>
          <input
            type="range"
            min="0.1"
            max="20"
            step="0.1"
            value={synth.voices.filter.Q}
            onChange={(e) => {
              engine.setSynthFilter(synth.id, { Q: parseFloat(e.target.value) });
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{synth.voices.filter.Q.toFixed(1)}</span>
        </div>
      </div>

      <div style={styles.paramSection}>
        <h5 style={styles.sectionTitle}>Amp Envelope</h5>
        {synth.voices.envelopes.filter(e => e.destination === 'amplitude').map(env => (
          <div key={env.id} style={styles.envelopeGrid}>
            <div>
              <span style={styles.label}>A</span>
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={env.attack}
                onChange={(e) => {
                  engine.setSynthEnvelope(synth.id, env.id, { attack: parseFloat(e.target.value) });
                  refreshInstruments();
                }}
                style={styles.sliderVertical}
              />
              <span style={styles.valueSmall}>{env.attack.toFixed(2)}</span>
            </div>
            <div>
              <span style={styles.label}>D</span>
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={env.decay}
                onChange={(e) => {
                  engine.setSynthEnvelope(synth.id, env.id, { decay: parseFloat(e.target.value) });
                  refreshInstruments();
                }}
                style={styles.sliderVertical}
              />
              <span style={styles.valueSmall}>{env.decay.toFixed(2)}</span>
            </div>
            <div>
              <span style={styles.label}>S</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={env.sustain}
                onChange={(e) => {
                  engine.setSynthEnvelope(synth.id, env.id, { sustain: parseFloat(e.target.value) });
                  refreshInstruments();
                }}
                style={styles.sliderVertical}
              />
              <span style={styles.valueSmall}>{env.sustain.toFixed(2)}</span>
            </div>
            <div>
              <span style={styles.label}>R</span>
              <input
                type="range"
                min="0"
                max="5"
                step="0.01"
                value={env.release}
                onChange={(e) => {
                  engine.setSynthEnvelope(synth.id, env.id, { release: parseFloat(e.target.value) });
                  refreshInstruments();
                }}
                style={styles.sliderVertical}
              />
              <span style={styles.valueSmall}>{env.release.toFixed(2)}</span>
            </div>
          </div>
        ))}
      </div>

      <div style={styles.paramSection}>
        <h5 style={styles.sectionTitle}>Unison</h5>
        <div style={styles.paramRow}>
          <span style={styles.label}>Voices</span>
          <input
            type="range"
            min="1"
            max="16"
            value={synth.unisonVoices}
            onChange={(e) => {
              engine.setUnison(synth.id, parseInt(e.target.value), synth.unisonDetune, synth.unisonSpread);
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{synth.unisonVoices}</span>
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Detune</span>
          <input
            type="range"
            min="0"
            max="100"
            value={synth.unisonDetune}
            onChange={(e) => {
              engine.setUnison(synth.id, synth.unisonVoices, parseInt(e.target.value), synth.unisonSpread);
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{synth.unisonDetune}ct</span>
        </div>
      </div>
    </div>
  );

  const renderDrumMachinePanel = (drum: DrumMachine) => (
    <div style={styles.instrumentPanel}>
      <h4 style={styles.panelTitle}>{drum.name} ({drum.kit})</h4>
      <div style={styles.drumGrid}>
        {drum.pads.slice(0, 16).map((pad, index) => (
          <div
            key={pad.id}
            style={{
              ...styles.drumPad,
              backgroundColor: pad.muted ? '#333' : pad.soloed ? '#f59e0b' : '#6366f1',
            }}
            onClick={() => engine.triggerPad(drum.id, pad.id, 1.0)}
          >
            <span style={styles.padLabel}>{pad.name}</span>
            <span style={styles.padNote}>C{Math.floor((36 + index) / 12) - 1}</span>
          </div>
        ))}
      </div>
      <div style={styles.paramSection}>
        <div style={styles.paramRow}>
          <span style={styles.label}>Swing</span>
          <input
            type="range"
            min="0"
            max="100"
            value={drum.swing}
            onChange={(e) => {
              engine.setSwing(drum.id, parseInt(e.target.value));
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{drum.swing}%</span>
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Master</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={drum.masterVolume}
            onChange={() => refreshInstruments()}
            style={styles.slider}
          />
          <span style={styles.value}>{Math.round(drum.masterVolume * 100)}%</span>
        </div>
      </div>
      <div style={styles.transportRow}>
        <button
          onClick={() => {
            if (drum.playing) {
              engine.stopPattern(drum.id);
            } else {
              engine.startPattern(drum.id);
            }
            refreshInstruments();
          }}
          style={drum.playing ? styles.stopButton : styles.playButton}
        >
          {drum.playing ? 'Stop' : 'Play'}
        </button>
      </div>
    </div>
  );

  const renderOrchestralPanel = (inst: OrchestralInstrument) => (
    <div style={styles.instrumentPanel}>
      <h4 style={styles.panelTitle}>{inst.name}</h4>
      <div style={styles.paramSection}>
        <h5 style={styles.sectionTitle}>Articulations</h5>
        <div style={styles.articulationGrid}>
          {inst.articulations.map(art => (
            <button
              key={art.id}
              onClick={() => {
                engine.setArticulation(inst.id, art.id);
                refreshInstruments();
              }}
              style={{
                ...styles.articulationButton,
                backgroundColor: inst.currentArticulation === art.id ? '#6366f1' : '#333',
              }}
            >
              {art.name}
            </button>
          ))}
        </div>
      </div>
      <div style={styles.paramSection}>
        <div style={styles.paramRow}>
          <span style={styles.label}>Legato</span>
          <input
            type="checkbox"
            checked={inst.legato}
            onChange={(e) => {
              engine.setOrchestralLegato(inst.id, e.target.checked);
              refreshInstruments();
            }}
          />
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Portamento</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={inst.portamento}
            onChange={(e) => {
              engine.setOrchestralPortamento(inst.id, parseFloat(e.target.value));
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{inst.portamento.toFixed(2)}s</span>
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Divisi</span>
          <input
            type="checkbox"
            checked={inst.divisi}
            onChange={(e) => {
              engine.enableDivisi(inst.id, e.target.checked);
              refreshInstruments();
            }}
          />
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Section Size</span>
          <input
            type="range"
            min="1"
            max="32"
            value={inst.sectionSize}
            onChange={(e) => {
              engine.setSectionSize(inst.id, parseInt(e.target.value));
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{inst.sectionSize}</span>
        </div>
      </div>
    </div>
  );

  const renderSamplerPanel = (sampler: SamplerInstrument) => (
    <div style={styles.instrumentPanel}>
      <h4 style={styles.panelTitle}>{sampler.name}</h4>
      <div style={styles.paramSection}>
        <h5 style={styles.sectionTitle}>ADSR</h5>
        <div style={styles.envelopeGrid}>
          <div>
            <span style={styles.label}>A</span>
            <input
              type="range"
              min="0"
              max="2"
              step="0.01"
              value={sampler.adsr.attack}
              onChange={(e) => {
                engine.setSamplerADSR(sampler.id, { ...sampler.adsr, attack: parseFloat(e.target.value) });
                refreshInstruments();
              }}
              style={styles.sliderVertical}
            />
            <span style={styles.valueSmall}>{sampler.adsr.attack.toFixed(2)}</span>
          </div>
          <div>
            <span style={styles.label}>D</span>
            <input
              type="range"
              min="0"
              max="2"
              step="0.01"
              value={sampler.adsr.decay}
              onChange={(e) => {
                engine.setSamplerADSR(sampler.id, { ...sampler.adsr, decay: parseFloat(e.target.value) });
                refreshInstruments();
              }}
              style={styles.sliderVertical}
            />
            <span style={styles.valueSmall}>{sampler.adsr.decay.toFixed(2)}</span>
          </div>
          <div>
            <span style={styles.label}>S</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={sampler.adsr.sustain}
              onChange={(e) => {
                engine.setSamplerADSR(sampler.id, { ...sampler.adsr, sustain: parseFloat(e.target.value) });
                refreshInstruments();
              }}
              style={styles.sliderVertical}
            />
            <span style={styles.valueSmall}>{sampler.adsr.sustain.toFixed(2)}</span>
          </div>
          <div>
            <span style={styles.label}>R</span>
            <input
              type="range"
              min="0"
              max="5"
              step="0.01"
              value={sampler.adsr.release}
              onChange={(e) => {
                engine.setSamplerADSR(sampler.id, { ...sampler.adsr, release: parseFloat(e.target.value) });
                refreshInstruments();
              }}
              style={styles.sliderVertical}
            />
            <span style={styles.valueSmall}>{sampler.adsr.release.toFixed(2)}</span>
          </div>
        </div>
      </div>
      <div style={styles.paramSection}>
        <h5 style={styles.sectionTitle}>Settings</h5>
        <div style={styles.paramRow}>
          <span style={styles.label}>Polyphony</span>
          <input
            type="range"
            min="1"
            max="128"
            value={sampler.polyphony}
            onChange={(e) => {
              engine.setSamplerPolyphony(sampler.id, parseInt(e.target.value));
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{sampler.polyphony}</span>
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Legato</span>
          <input
            type="checkbox"
            checked={sampler.legatoMode}
            onChange={(e) => {
              engine.enableLegatoMode(sampler.id, e.target.checked);
              refreshInstruments();
            }}
          />
        </div>
        <div style={styles.paramRow}>
          <span style={styles.label}>Portamento</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={sampler.portamentoTime}
            onChange={(e) => {
              engine.setPortamentoTime(sampler.id, parseFloat(e.target.value));
              refreshInstruments();
            }}
            style={styles.slider}
          />
          <span style={styles.value}>{sampler.portamentoTime.toFixed(2)}s</span>
        </div>
      </div>
    </div>
  );

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Virtual Instruments</h3>

      <div style={styles.tabs}>
        <button
          onClick={() => setActiveTab('synth')}
          style={activeTab === 'synth' ? styles.activeTab : styles.tab}
        >
          Synthesizers ({synths.length})
        </button>
        <button
          onClick={() => setActiveTab('sampler')}
          style={activeTab === 'sampler' ? styles.activeTab : styles.tab}
        >
          Samplers ({samplers.length})
        </button>
        <button
          onClick={() => setActiveTab('drums')}
          style={activeTab === 'drums' ? styles.activeTab : styles.tab}
        >
          Drums ({drumMachines.length})
        </button>
        <button
          onClick={() => setActiveTab('orchestral')}
          style={activeTab === 'orchestral' ? styles.activeTab : styles.tab}
        >
          Orchestral ({orchestral.length})
        </button>
      </div>

      <div style={styles.content}>
        {activeTab === 'synth' && (
          <>
            <div style={styles.createSection}>
              <span style={styles.createLabel}>Create:</span>
              <button onClick={() => handleCreateSynth('subtractive')} style={styles.createButton}>Subtractive</button>
              <button onClick={() => handleCreateSynth('fm')} style={styles.createButton}>FM</button>
              <button onClick={() => handleCreateSynth('wavetable')} style={styles.createButton}>Wavetable</button>
              <button onClick={() => handleCreateSynth('granular')} style={styles.createButton}>Granular</button>
            </div>
            <div style={styles.instrumentList}>
              {synths.map(synth => (
                <div key={synth.id}>
                  <button
                    onClick={() => setSelectedInstrument(selectedInstrument === synth.id ? null : synth.id)}
                    style={selectedInstrument === synth.id ? styles.selectedInstrument : styles.instrumentButton}
                  >
                    {synth.name} ({synth.type})
                  </button>
                  {selectedInstrument === synth.id && renderSynthPanel(synth)}
                </div>
              ))}
            </div>
          </>
        )}

        {activeTab === 'sampler' && (
          <>
            <div style={styles.createSection}>
              <span style={styles.createLabel}>Create:</span>
              <button onClick={() => handleCreateSampler('multi-sample')} style={styles.createButton}>Multi-Sample</button>
              <button onClick={() => handleCreateSampler('wavetable')} style={styles.createButton}>Wavetable</button>
              <button onClick={() => handleCreateSampler('granular')} style={styles.createButton}>Granular</button>
            </div>
            <div style={styles.instrumentList}>
              {samplers.map(sampler => (
                <div key={sampler.id}>
                  <button
                    onClick={() => setSelectedInstrument(selectedInstrument === sampler.id ? null : sampler.id)}
                    style={selectedInstrument === sampler.id ? styles.selectedInstrument : styles.instrumentButton}
                  >
                    {sampler.name} ({sampler.type})
                  </button>
                  {selectedInstrument === sampler.id && renderSamplerPanel(sampler)}
                </div>
              ))}
            </div>
          </>
        )}

        {activeTab === 'drums' && (
          <>
            <div style={styles.createSection}>
              <span style={styles.createLabel}>Create Kit:</span>
              {['808', '909', 'Acoustic', 'Electronic', 'Lo-Fi', 'Trap'].map(kit => (
                <button key={kit} onClick={() => handleCreateDrumMachine(kit)} style={styles.createButton}>{kit}</button>
              ))}
            </div>
            <div style={styles.instrumentList}>
              {drumMachines.map(drum => (
                <div key={drum.id}>
                  <button
                    onClick={() => setSelectedInstrument(selectedInstrument === drum.id ? null : drum.id)}
                    style={selectedInstrument === drum.id ? styles.selectedInstrument : styles.instrumentButton}
                  >
                    {drum.name}
                  </button>
                  {selectedInstrument === drum.id && renderDrumMachinePanel(drum)}
                </div>
              ))}
            </div>
          </>
        )}

        {activeTab === 'orchestral' && (
          <>
            <div style={styles.createSection}>
              <span style={styles.createLabel}>Create:</span>
              {(['strings', 'woodwinds', 'brass', 'percussion', 'choir'] as const).map(cat => (
                <button key={cat} onClick={() => handleCreateOrchestral(cat)} style={styles.createButton}>
                  {cat.charAt(0).toUpperCase() + cat.slice(1)}
                </button>
              ))}
            </div>
            <div style={styles.instrumentList}>
              {orchestral.map(inst => (
                <div key={inst.id}>
                  <button
                    onClick={() => setSelectedInstrument(selectedInstrument === inst.id ? null : inst.id)}
                    style={selectedInstrument === inst.id ? styles.selectedInstrument : styles.instrumentButton}
                  >
                    {inst.name} ({inst.category})
                  </button>
                  {selectedInstrument === inst.id && renderOrchestralPanel(inst)}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
    padding: '15px',
    color: '#fff',
  },
  title: {
    margin: '0 0 15px 0',
    fontSize: '1.2em',
    color: '#6366f1',
  },
  tabs: {
    display: 'flex',
    gap: '5px',
    marginBottom: '15px',
  },
  tab: {
    padding: '8px 16px',
    backgroundColor: '#333',
    border: 'none',
    borderRadius: '4px',
    color: '#aaa',
    cursor: 'pointer',
    fontSize: '0.9em',
  },
  activeTab: {
    padding: '8px 16px',
    backgroundColor: '#6366f1',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '0.9em',
  },
  content: {
    minHeight: '200px',
  },
  createSection: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '15px',
    flexWrap: 'wrap',
  },
  createLabel: {
    color: '#aaa',
    fontSize: '0.9em',
  },
  createButton: {
    padding: '6px 12px',
    backgroundColor: '#333',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '0.85em',
  },
  instrumentList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  instrumentButton: {
    width: '100%',
    padding: '10px',
    backgroundColor: '#222',
    border: '1px solid #333',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    textAlign: 'left',
  },
  selectedInstrument: {
    width: '100%',
    padding: '10px',
    backgroundColor: '#333',
    border: '1px solid #6366f1',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    textAlign: 'left',
  },
  instrumentPanel: {
    backgroundColor: '#222',
    borderRadius: '0 0 4px 4px',
    padding: '15px',
    marginTop: '-1px',
  },
  panelTitle: {
    margin: '0 0 15px 0',
    color: '#6366f1',
  },
  paramSection: {
    marginBottom: '15px',
  },
  sectionTitle: {
    margin: '0 0 10px 0',
    color: '#aaa',
    fontSize: '0.9em',
  },
  oscRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '8px',
  },
  paramRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '8px',
  },
  label: {
    width: '80px',
    color: '#aaa',
    fontSize: '0.85em',
  },
  select: {
    padding: '4px 8px',
    backgroundColor: '#333',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '0.85em',
  },
  slider: {
    flex: 1,
    height: '4px',
  },
  sliderVertical: {
    width: '100%',
    height: '4px',
  },
  value: {
    width: '60px',
    color: '#6366f1',
    fontSize: '0.85em',
    textAlign: 'right',
  },
  valueSmall: {
    color: '#6366f1',
    fontSize: '0.75em',
    textAlign: 'center',
    display: 'block',
  },
  envelopeGrid: {
    display: 'flex',
    gap: '20px',
    justifyContent: 'center',
  },
  addButton: {
    marginTop: '10px',
    padding: '6px 12px',
    backgroundColor: '#333',
    border: '1px dashed #555',
    borderRadius: '4px',
    color: '#aaa',
    cursor: 'pointer',
    fontSize: '0.85em',
  },
  drumGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '8px',
    marginBottom: '15px',
  },
  drumPad: {
    aspectRatio: '1',
    borderRadius: '8px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    border: '2px solid rgba(255,255,255,0.1)',
  },
  padLabel: {
    fontSize: '0.7em',
    fontWeight: 'bold',
  },
  padNote: {
    fontSize: '0.6em',
    color: 'rgba(255,255,255,0.6)',
  },
  transportRow: {
    display: 'flex',
    justifyContent: 'center',
    gap: '10px',
  },
  playButton: {
    padding: '10px 30px',
    backgroundColor: '#22c55e',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontWeight: 'bold',
  },
  stopButton: {
    padding: '10px 30px',
    backgroundColor: '#ef4444',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontWeight: 'bold',
  },
  articulationGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '5px',
  },
  articulationButton: {
    padding: '6px 12px',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '0.8em',
  },
};
