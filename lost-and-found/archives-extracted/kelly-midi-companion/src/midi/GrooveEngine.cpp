#include "midi/GrooveEngine.h"
#include <cmath>
#include <algorithm>

namespace kelly {

GrooveEngine::GrooveEngine() 
    : rng_(std::random_device{}())
{
    initializeTemplates();
}

void GrooveEngine::initializeTemplates() {
    // =========================================================================
    // STRAIGHT - Even 8ths, no swing
    // =========================================================================
    templates_.push_back({
        GrooveType::Straight,
        "Straight 8ths",
        4, 4,  // time signature
        0.0f,  // no swing
        {
            {0.0f, 100},    // beat 1 - strong
            {0.5f, 70},     // beat 1& - weak
            {1.0f, 90},     // beat 2 - medium
            {1.5f, 65},     // beat 2&
            {2.0f, 95},     // beat 3 - strong
            {2.5f, 70},     // beat 3&
            {3.0f, 85},     // beat 4 - medium
            {3.5f, 60},     // beat 4&
        }
    });
    
    // =========================================================================
    // SWING - Jazz/Blues feel
    // =========================================================================
    templates_.push_back({
        GrooveType::Swing,
        "Swing",
        4, 4,
        0.66f,  // triplet swing
        {
            {0.0f, 100},
            {0.66f, 60},    // swung 8th
            {1.0f, 85},
            {1.66f, 55},
            {2.0f, 95},
            {2.66f, 60},
            {3.0f, 80},
            {3.66f, 50},
        }
    });
    
    // =========================================================================
    // SYNCOPATED - Emphasis on off-beats
    // =========================================================================
    templates_.push_back({
        GrooveType::Syncopated,
        "Syncopated",
        4, 4,
        0.0f,
        {
            {0.0f, 85},     // beat 1 - pulled back
            {0.5f, 95},     // beat 1& - STRONG (syncopation)
            {1.0f, 70},     // beat 2 - weak
            {1.5f, 100},    // beat 2& - STRONG
            {2.0f, 75},
            {2.5f, 90},
            {3.0f, 65},
            {3.5f, 95},     // beat 4& - STRONG
        }
    });
    
    // =========================================================================
    // HALFTIME - Sparse, heavy
    // =========================================================================
    templates_.push_back({
        GrooveType::Halftime,
        "Halftime",
        4, 4,
        0.0f,
        {
            {0.0f, 110},    // beat 1 - HEAVY
            {2.0f, 100},    // beat 3 - accent
        }
    });
    
    // =========================================================================
    // SHUFFLE - Blues shuffle feel
    // =========================================================================
    templates_.push_back({
        GrooveType::Shuffle,
        "Shuffle",
        4, 4,
        0.5f,  // lighter swing than jazz
        {
            {0.0f, 100},
            {0.5f, 50},     // ghost note
            {0.75f, 80},    // shuffle accent
            {1.0f, 90},
            {1.5f, 45},
            {1.75f, 75},
            {2.0f, 95},
            {2.5f, 50},
            {2.75f, 80},
            {3.0f, 85},
            {3.5f, 45},
            {3.75f, 70},
        }
    });
}

GrooveTemplate GrooveEngine::getGroove(GrooveType type) const {
    for (const auto& tmpl : templates_) {
        if (tmpl.type == type) {
            return tmpl;
        }
    }
    return templates_[0];  // Default to straight
}

GrooveTemplate GrooveEngine::suggestGroove(const EmotionNode& emotion) const {
    // Low arousal, negative valence → Halftime (heavy, slow)
    if (emotion.arousal < 0.3f && emotion.valence < -0.3f) {
        return getGroove(GrooveType::Halftime);
    }
    
    // High arousal, negative valence → Syncopated (agitated)
    if (emotion.arousal > 0.7f && emotion.valence < -0.3f) {
        return getGroove(GrooveType::Syncopated);
    }
    
    // Mid arousal, positive valence → Swing (warm, jazzy)
    if (emotion.arousal > 0.4f && emotion.arousal < 0.7f && emotion.valence > 0.3f) {
        return getGroove(GrooveType::Swing);
    }
    
    // High arousal, positive valence → Shuffle (energetic)
    if (emotion.arousal > 0.7f && emotion.valence > 0.3f) {
        return getGroove(GrooveType::Shuffle);
    }
    
    // Default → Straight
    return getGroove(GrooveType::Straight);
}

void GrooveEngine::applyGroove(std::vector<MidiNote>& notes, const GrooveTemplate& groove) {
    if (groove.swingAmount <= 0.0f) return;
    
    for (auto& note : notes) {
        // Find position within beat
        double beatPosition = std::fmod(note.startBeat, 1.0);
        
        // Swing affects off-beat 8ths (0.5 position)
        if (std::abs(beatPosition - 0.5) < 0.1) {
            // Push the off-beat later based on swing amount
            double swingOffset = groove.swingAmount * 0.5;  // Max swing = triplet feel
            note.startBeat += swingOffset - 0.5;
            
            // Slightly shorten the note to maintain groove pocket
            note.duration *= 0.9;
        }
    }
}

float GrooveEngine::getTimingOffset(float intensity, float valence, float arousal) {
    // Grief/sadness → drag behind the beat
    // Anger/excitement → push ahead of the beat
    // High intensity → more extreme timing
    
    float baseOffset = 0.0f;
    
    if (valence < -0.5f && arousal < 0.4f) {
        // Sad: drag
        baseOffset = 0.02f;  // 20ms late
    } else if (valence < -0.3f && arousal > 0.6f) {
        // Angry: rush
        baseOffset = -0.015f;  // 15ms early
    } else if (valence > 0.5f && arousal > 0.7f) {
        // Excited: slight push
        baseOffset = -0.01f;
    }
    
    return baseOffset * intensity;
}

int GrooveEngine::getVelocityOffset(float intensity, float baseVelocity) {
    // Higher intensity = more velocity variance
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float variance = dist(rng_) * intensity * 20.0f;  // ±20 at max intensity
    
    return static_cast<int>(std::clamp(variance, -25.0f, 25.0f));
}

void GrooveEngine::humanize(std::vector<MidiNote>& notes, const IntentResult& intent) {
    float humanization = intent.humanization;
    if (humanization <= 0.0f) return;
    
    std::uniform_real_distribution<float> timingDist(-1.0f, 1.0f);
    
    float emotionOffset = getTimingOffset(
        intent.emotion.intensity,
        intent.emotion.valence,
        intent.emotion.arousal
    );
    
    for (auto& note : notes) {
        // Timing humanization
        float timingVariance = timingDist(rng_) * humanization * 0.03;  // ±30ms max
        note.startBeat += timingVariance + emotionOffset;
        
        // Velocity humanization
        int velOffset = getVelocityOffset(humanization, static_cast<float>(note.velocity));
        note.velocity = std::clamp(note.velocity + velOffset, 1, 127);
        
        // Duration humanization (slight variations in note length)
        float durationVariance = 1.0f + (timingDist(rng_) * humanization * 0.1f);
        note.duration *= durationVariance;
    }
}

std::vector<MidiNote> GrooveEngine::chordsToRhythm(
    const std::vector<Chord>& chords,
    const GrooveTemplate& groove,
    int velocityBase
) {
    std::vector<MidiNote> notes;
    
    for (const auto& chord : chords) {
        double chordStart = chord.startBeat;
        double chordEnd = chord.startBeat + chord.duration;
        
        // Find which groove hits fall within this chord
        int barsInChord = static_cast<int>(std::ceil(chord.duration / 4.0));
        
        for (int bar = 0; bar < barsInChord; ++bar) {
            for (const auto& [beatPos, velocity] : groove.pattern) {
                double absoluteBeat = chordStart + (bar * 4.0) + beatPos;
                
                if (absoluteBeat >= chordStart && absoluteBeat < chordEnd) {
                    // Scale velocity
                    int scaledVel = static_cast<int>((velocity / 100.0f) * velocityBase);
                    scaledVel = std::clamp(scaledVel, 1, 127);
                    
                    // Add each note of the chord
                    for (int pitch : chord.pitches) {
                        MidiNote note;
                        note.pitch = pitch;
                        note.velocity = scaledVel;
                        note.startBeat = absoluteBeat;
                        note.duration = 0.4;  // Default 8th note length
                        
                        notes.push_back(note);
                    }
                }
            }
        }
    }
    
    return notes;
}

} // namespace kelly
