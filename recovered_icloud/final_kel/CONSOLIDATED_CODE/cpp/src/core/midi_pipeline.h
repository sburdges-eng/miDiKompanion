#pragma once
/*
 * midi_pipeline.h - Legacy MIDI Pipeline
 * ======================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Core Layer: Legacy MIDI pipeline (may be superseded by midi/MidiGenerator.h)
 * - Type System: Defines basic MidiNote structure
 * - MIDI Layer: Basic MIDI note storage and tempo management
 *
 * Purpose: Legacy MIDI pipeline providing basic MIDI note storage.
 *          Note: May be superseded by midi/MidiGenerator.h which provides
 *          complete MIDI generation with all engines.
 *
 * Features:
 * - MIDI note storage
 * - Tempo management
 * - Basic MIDI operations
 */

#include <string>
#include <vector>
#include <cstdint>

namespace kelly {

struct MidiNote {
    uint8_t note;
    uint8_t velocity;
    uint32_t time;  // in ticks
    uint32_t duration;  // in ticks
};

class MidiPipeline {
public:
    MidiPipeline();
    ~MidiPipeline() = default;

    void setTempo(int bpm);
    void addNote(const MidiNote& note);
    void clear();

    const std::vector<MidiNote>& getNotes() const { return notes_; }
    int getTempo() const { return tempo_; }

private:
    std::vector<MidiNote> notes_;
    int tempo_ = 120;
};

} // namespace kelly
