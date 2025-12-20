#pragma once
/*
 * chord_diagnostics.h - Chord Analysis Utilities
 * ==============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Core Layer: Chord analysis utilities
 * - MIDI Layer: Used by ChordGenerator (chord analysis)
 * - Engine Layer: Provides chord identification and dissonance calculation
 *
 * Purpose: Chord analysis utilities for identifying chords and calculating dissonance.
 *          Provides diagnostic functions for chord analysis.
 *
 * Features:
 * - Chord identification
 * - Dissonance calculation
 * - Consonance checking
 * - Interval analysis
 */

#include <vector>
#include <string>
#include <cstdint>

namespace kelly {

struct Chord {
    std::vector<uint8_t> notes;
    std::string name;
    float dissonanceLevel = 0.0f;
};

class ChordDiagnostics {
public:
    ChordDiagnostics() = default;
    ~ChordDiagnostics() = default;

    float calculateDissonance(const Chord& chord) const;
    std::string identifyChord(const std::vector<uint8_t>& notes) const;
    bool isConsonant(const Chord& chord, float threshold = 0.3f) const;

private:
    float intervalDissonance(int interval) const;
};

} // namespace kelly
