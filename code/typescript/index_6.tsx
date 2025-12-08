import React, { useState } from 'react';
import { EmotionWheel } from './EmotionWheel';
import { Interrogator } from './Interrogator';
import { GhostWriter } from './GhostWriter';
import { RuleBreaker } from './RuleBreaker';
import { SideBToolbar } from './SideBToolbar';
import { useStore } from '../../store/useStore';

export const SideB: React.FC = () => {
  const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null);
  const { songIntent, updateSongIntent } = useStore();

  const handleEmotionSelect = (emotion: string) => {
    setSelectedEmotion(emotion);
  };

  const handleIntentComplete = (intent: Record<string, unknown>) => {
    updateSongIntent(intent);
  };

  return (
    <div className="w-full h-full flex flex-col bg-ableton-bg">
      {/* Toolbar */}
      <SideBToolbar />

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden p-4 gap-4">
        {/* Left Panel - Emotion Selection */}
        <div className="w-80 flex flex-col gap-4">
          <div className="panel flex-1 overflow-hidden flex flex-col">
            <div className="panel-header">Core Emotion</div>
            <div className="flex-1 p-4 overflow-auto">
              <EmotionWheel onSelectEmotion={handleEmotionSelect} />
            </div>
          </div>
        </div>

        {/* Center Panel - Interrogator */}
        <div className="flex-1 flex flex-col gap-4">
          <div className="panel flex-1 overflow-hidden flex flex-col">
            <div className="panel-header">Interrogator</div>
            <div className="flex-1 overflow-auto">
              <Interrogator 
                emotion={selectedEmotion} 
                onComplete={handleIntentComplete}
              />
            </div>
          </div>

          {/* Rule Breaker */}
          <div className="panel h-48 overflow-hidden flex flex-col">
            <div className="panel-header">Rule to Break</div>
            <div className="flex-1 overflow-auto">
              <RuleBreaker />
            </div>
          </div>
        </div>

        {/* Right Panel - Ghost Writer */}
        <div className="w-96 flex flex-col">
          <div className="panel flex-1 overflow-hidden flex flex-col">
            <div className="panel-header">Ghost Writer</div>
            <div className="flex-1 overflow-auto">
              <GhostWriter 
                emotion={selectedEmotion}
                intent={songIntent}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
