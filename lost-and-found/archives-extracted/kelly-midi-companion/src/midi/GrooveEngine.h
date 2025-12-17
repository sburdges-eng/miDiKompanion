#pragma once

#include "common/Types.h"
#include <vector>
#include <random>

namespace kelly {

/**
 * Applies groove, swing, and humanization to MIDI data.
 * 
 * Transforms rigid grid-locked notes into living, breathing rhythms.
 * Humanization is emotion-driven: grief drags, anger rushes.
 */
class GrooveEngine {
public:
    GrooveEngine();
    
    /** Apply groove template to notes */
    void applyGroove(std::vector<MidiNote>& notes, const GrooveTemplate& groove);
    
    /** Apply humanization based on emotional intent */
    void humanize(std::vector<MidiNote>& notes, const IntentResult& intent);
    
    /** Convert chords to rhythmic pattern */
    std::vector<MidiNote> chordsToRhythm(
        const std::vector<Chord>& chords,
        const GrooveTemplate& groove,
        int velocityBase = 80
    );
    
    /** Get a groove template by type */
    GrooveTemplate getGroove(GrooveType type) const;
    
    /** Get groove based on emotion */
    GrooveTemplate suggestGroove(const EmotionNode& emotion) const;
    
private:
    std::mt19937 rng_;
    std::vector<GrooveTemplate> templates_;
    
    void initializeTemplates();
    
    // Timing humanization
    float getTimingOffset(float intensity, float valence, float arousal);
    
    // Velocity humanization  
    int getVelocityOffset(float intensity, float baseVelocity);
};

} // namespace kelly
