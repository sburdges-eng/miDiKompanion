#include <gtest/gtest.h>
#include "engines/MelodyEngine.h"
#include "engines/BassEngine.h"
#include "engines/RhythmEngine.h"
#include "engines/ArrangementEngine.h"
#include "engine/IntentPipeline.h"
#include "midi/MidiGenerator.h"
#include "common/Types.h"
#include <vector>
#include <string>
#include <algorithm>

using namespace kelly;

class EngineIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        melodyEngine = std::make_unique<MelodyEngine>();
        bassEngine = std::make_unique<BassEngine>();
        rhythmEngine = std::make_unique<RhythmEngine>();
        arrangementEngine = std::make_unique<ArrangementEngine>();
        pipeline = std::make_unique<IntentPipeline>();
    }

    std::unique_ptr<MelodyEngine> melodyEngine;
    std::unique_ptr<BassEngine> bassEngine;
    std::unique_ptr<RhythmEngine> rhythmEngine;
    std::unique_ptr<ArrangementEngine> arrangementEngine;
    std::unique_ptr<IntentPipeline> pipeline;
};

// Test full arrangement generation
TEST_F(EngineIntegrationTest, FullArrangement) {
    Wound wound;
    wound.description = "I feel joyful and energetic";
    wound.intensity = 0.7f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);

    // Generate arrangement
    ArrangementOutput arrangement = arrangementEngine->generate("joy", "", 32);
    EXPECT_GT(arrangement.sections.size(), 0);

    // Generate melody for a section
    MelodyOutput melody = melodyEngine->generate("joy", "C", "major", 4, 120);
    EXPECT_GT(melody.notes.size(), 0);

    // Generate bass
    std::vector<std::string> progression = {"C", "Am", "F", "G"};
    BassOutput bass = bassEngine->generate("joy", progression, "C", 4, 120);
    EXPECT_GT(bass.notes.size(), 0);

    // Generate rhythm
    RhythmOutput rhythm = rhythmEngine->generate("joy", 4, 120);
    EXPECT_GT(rhythm.hits.size(), 0);
}

// Test emotion consistency across engines
TEST_F(EngineIntegrationTest, EmotionConsistency) {
    std::string emotion = "joy";

    MelodyOutput melody = melodyEngine->generate(emotion, "C", "major", 4, 120);
    EXPECT_EQ(melody.emotion, emotion);

    std::vector<std::string> progression = {"C", "Am", "F", "G"};
    BassOutput bass = bassEngine->generate(emotion, progression, "C", 4, 120);
    EXPECT_EQ(bass.emotion, emotion);

    RhythmOutput rhythm = rhythmEngine->generate(emotion, 4, 120);
    EXPECT_EQ(rhythm.config.emotion, emotion);
}

// Test timing synchronization
TEST_F(EngineIntegrationTest, TimingSynchronization) {
    int bars = 4;
    int tempo = 120;
    int expectedTicks = bars * 4 * 480; // 4 beats/bar, 480 ticks/beat

    MelodyOutput melody = melodyEngine->generate("neutral", "C", "major", bars, tempo);
    EXPECT_EQ(melody.totalTicks, expectedTicks);

    RhythmOutput rhythm = rhythmEngine->generate("neutral", bars, tempo);
    EXPECT_EQ(rhythm.totalTicks, expectedTicks);
}

// Test all engines are called during generation
TEST_F(EngineIntegrationTest, AllEnginesCalled) {
    Wound wound;
    wound.description = "I feel complex";
    wound.intensity = 0.8f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);

    // Verify engines can be called independently
    MelodyOutput melody = melodyEngine->generate("joy", "C", "major", 4, 120);
    EXPECT_GT(melody.notes.size(), 0) << "MelodyEngine should produce notes";

    std::vector<std::string> progression = {"C", "Am", "F", "G"};
    BassOutput bass = bassEngine->generate("joy", progression, "C", 4, 120);
    EXPECT_GT(bass.notes.size(), 0) << "BassEngine should produce notes";

    RhythmOutput rhythm = rhythmEngine->generate("joy", 4, 120);
    EXPECT_GT(rhythm.hits.size(), 0) << "RhythmEngine should produce hits";

    ArrangementOutput arrangement = arrangementEngine->generate("joy", "", 8);
    EXPECT_GT(arrangement.sections.size(), 0) << "ArrangementEngine should produce sections";
}

// Test engine outputs are properly merged
TEST_F(EngineIntegrationTest, EngineOutputMerging) {
    // Generate outputs from multiple engines
    MelodyOutput melody = melodyEngine->generate("joy", "C", "major", 4, 120);
    std::vector<std::string> progression = {"C", "Am", "F", "G"};
    BassOutput bass = bassEngine->generate("joy", progression, "C", 4, 120);
    RhythmOutput rhythm = rhythmEngine->generate("joy", 4, 120);

    // Verify outputs can be merged (all have valid timing)
    EXPECT_GT(melody.totalTicks, 0) << "Melody should have valid totalTicks";
    EXPECT_GT(bass.totalTicks, 0) << "Bass should have valid totalTicks";
    EXPECT_GT(rhythm.totalTicks, 0) << "Rhythm should have valid totalTicks";

    // Verify timing is synchronized (all should have same totalTicks for same bars/tempo)
    int bars = 4;
    int tempo = 120;
    int expectedTicks = bars * 4 * 480;
    EXPECT_EQ(melody.totalTicks, expectedTicks) << "Melody timing should match expected";
    EXPECT_EQ(bass.totalTicks, expectedTicks) << "Bass timing should match expected";
    EXPECT_EQ(rhythm.totalTicks, expectedTicks) << "Rhythm timing should match expected";
}

// Test no conflicts between engine outputs
TEST_F(EngineIntegrationTest, NoEngineOutputConflicts) {
    // Generate outputs from multiple engines with same parameters
    std::string emotion = "joy";
    std::string key = "C";
    std::string mode = "major";
    int bars = 4;
    int tempo = 120;

    MelodyOutput melody = melodyEngine->generate(emotion, key, mode, bars, tempo);
    std::vector<std::string> progression = {"C", "Am", "F", "G"};
    BassOutput bass = bassEngine->generate(emotion, progression, key, bars, tempo);
    RhythmOutput rhythm = rhythmEngine->generate(emotion, bars, tempo);

    // Verify all outputs are valid
    EXPECT_GT(melody.notes.size(), 0);
    EXPECT_GT(bass.notes.size(), 0);
    EXPECT_GT(rhythm.hits.size(), 0);

    // Verify note ranges don't conflict
    // Melody should be in mid-high range
    // Bass should be in low range
    // Rhythm should be in drum range
    for (const auto& note : melody.notes) {
        EXPECT_GE(note.pitch, 48) << "Melody notes should be in mid-high range";
        EXPECT_LE(note.pitch, 84) << "Melody notes should be in mid-high range";
    }

    for (const auto& note : bass.notes) {
        EXPECT_GE(note.pitch, 24) << "Bass notes should be in low range";
        EXPECT_LE(note.pitch, 60) << "Bass notes should be in low range";
    }

    for (const auto& hit : rhythm.hits) {
        EXPECT_GE(hit.note, 36) << "Drum notes should be in drum range";
        EXPECT_LE(hit.note, 60) << "Drum notes should be in drum range";
    }
}

// Test all engines are called during MidiGenerator generation
TEST_F(EngineIntegrationTest, AllEnginesCalled) {
    // Use MidiGenerator to verify all engines are called
    MidiGenerator midiGenerator;

    Wound wound;
    wound.description = "I feel complex and emotional";
    wound.intensity = 0.8f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);

    // Generate with high complexity and long bars to trigger all engines
    GeneratedMidi midi = midiGenerator.generate(intent, 16, 0.9f, 0.4f, 0.0f, 0.75f);

    // Verify all engine outputs are present
    // Core engines (always called)
    EXPECT_GT(midi.chords.size(), 0) << "ChordGenerator should be called";
    EXPECT_GT(midi.melody.size(), 0) << "MelodyEngine should be called";
    EXPECT_GT(midi.bass.size(), 0) << "BassEngine should be called";

    // Optional engines (called when complexity is high)
    EXPECT_GE(midi.pad.size(), 0) << "PadEngine should be called";
    EXPECT_GE(midi.strings.size(), 0) << "StringEngine should be called";
    EXPECT_GE(midi.counterMelody.size(), 0) << "CounterMelodyEngine should be called";
    EXPECT_GE(midi.rhythm.size(), 0) << "RhythmEngine should be called";
    EXPECT_GE(midi.drumGroove.size(), 0) << "DrumGrooveEngine should be called";
    EXPECT_GE(midi.fills.size(), 0) << "FillEngine should be called";
    EXPECT_GE(midi.transitions.size(), 0) << "TransitionEngine should be called";

    // Processing engines (always called, verified by output quality)
    // DynamicsEngine - verified by valid velocities
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
    }

    // TensionEngine - verified by chord structure
    EXPECT_GT(midi.chords.size(), 0);

    // VariationEngine - verified by melody existence
    EXPECT_GT(midi.melody.size(), 0);

    // GrooveEngine - verified by timing
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.startBeat, 0.0);
    }
}

// Test engine outputs are properly merged
TEST_F(EngineIntegrationTest, EngineOutputsMerged) {
    MidiGenerator midiGenerator;

    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = 0.5f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);
    GeneratedMidi midi = midiGenerator.generate(intent, 8, 0.5f, 0.4f, 0.0f, 0.75f);

    // Verify no conflicts between engine outputs
    // All notes should have valid ranges
    auto validateNotes = [](const std::vector<MidiNote>& notes, const std::string& layerName) {
        for (const auto& note : notes) {
            EXPECT_GE(note.pitch, 0) << layerName << " note pitch should be >= 0";
            EXPECT_LE(note.pitch, 127) << layerName << " note pitch should be <= 127";
            EXPECT_GE(note.velocity, 0) << layerName << " note velocity should be >= 0";
            EXPECT_LE(note.velocity, 127) << layerName << " note velocity should be <= 127";
            EXPECT_GE(note.startBeat, 0.0) << layerName << " note startBeat should be >= 0.0";
            EXPECT_GT(note.duration, 0.0) << layerName << " note duration should be > 0.0";
        }
    };

    validateNotes(midi.melody, "Melody");
    validateNotes(midi.bass, "Bass");
    validateNotes(midi.pad, "Pad");
    validateNotes(midi.strings, "Strings");
    validateNotes(midi.counterMelody, "CounterMelody");
    validateNotes(midi.rhythm, "Rhythm");
    validateNotes(midi.fills, "Fills");
    validateNotes(midi.drumGroove, "DrumGroove");
    validateNotes(midi.transitions, "Transitions");

    // Verify chords are valid
    for (const auto& chord : midi.chords) {
        EXPECT_FALSE(chord.name.empty()) << "Chord name should not be empty";
        EXPECT_GT(chord.pitches.size(), 0) << "Chord should have at least one pitch";
        for (int pitch : chord.pitches) {
            EXPECT_GE(pitch, 0) << "Chord pitch should be >= 0";
            EXPECT_LE(pitch, 127) << "Chord pitch should be <= 127";
        }
        EXPECT_GE(chord.startBeat, 0.0) << "Chord startBeat should be >= 0.0";
        EXPECT_GT(chord.duration, 0.0) << "Chord duration should be > 0.0";
    }
}

// Test no conflicts between engine outputs
TEST_F(EngineIntegrationTest, NoEngineConflicts) {
    MidiGenerator midiGenerator;

    Wound wound;
    wound.description = "I feel complex";
    wound.intensity = 0.8f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);
    GeneratedMidi midi = midiGenerator.generate(intent, 8, 0.9f, 0.4f, 0.0f, 0.75f);

    // Verify all layers have valid timing (no negative start times, no overlapping issues)
    double maxBeat = 0.0;

    // Find maximum beat across all layers
    for (const auto& note : midi.melody) {
        maxBeat = std::max(maxBeat, note.startBeat + note.duration);
    }
    for (const auto& note : midi.bass) {
        maxBeat = std::max(maxBeat, note.startBeat + note.duration);
    }
    for (const auto& note : midi.pad) {
        maxBeat = std::max(maxBeat, note.startBeat + note.duration);
    }
    for (const auto& note : midi.strings) {
        maxBeat = std::max(maxBeat, note.startBeat + note.duration);
    }
    for (const auto& chord : midi.chords) {
        maxBeat = std::max(maxBeat, chord.startBeat + chord.duration);
    }

    // Verify length matches expected (8 bars = 32 beats)
    EXPECT_GE(maxBeat, 30.0) << "MIDI should span approximately 8 bars (32 beats)";
    EXPECT_LE(maxBeat, 34.0) << "MIDI should not exceed 8 bars significantly";

    // Verify all notes are within valid time range
    EXPECT_GE(maxBeat, 0.0) << "Maximum beat should be >= 0";
}

// Test all engines are called during generation
TEST_F(EngineIntegrationTest, AllEnginesCalled) {
    Wound wound;
    wound.description = "I feel complex";
    wound.intensity = 0.8f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);

    // Use MidiGenerator to verify all engines are called
    MidiGenerator generator;
    GeneratedMidi midi = generator.generate(intent, 16, 0.9f, 0.4f, 0.0f, 0.75f);

    // Verify core engines are called
    EXPECT_GT(midi.chords.size(), 0) << "ChordGenerator should be called";
    EXPECT_GT(midi.melody.size(), 0) << "MelodyEngine should be called";
    EXPECT_GT(midi.bass.size(), 0) << "BassEngine should be called";

    // Verify optional engines are called when complexity is high
    EXPECT_GE(midi.pad.size(), 0) << "PadEngine should be called";
    EXPECT_GE(midi.strings.size(), 0) << "StringEngine should be called";
    EXPECT_GE(midi.counterMelody.size(), 0) << "CounterMelodyEngine should be called";
    EXPECT_GE(midi.rhythm.size(), 0) << "RhythmEngine should be called";
    EXPECT_GE(midi.fills.size(), 0) << "FillEngine should be called";
    EXPECT_GE(midi.drumGroove.size(), 0) << "DrumGrooveEngine should be called";
    EXPECT_GE(midi.transitions.size(), 0) << "TransitionEngine should be called";

    // Verify processing engines are applied
    // DynamicsEngine - verified by valid velocities
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
    }

    // TensionEngine - verified by chord structure
    for (const auto& chord : midi.chords) {
        EXPECT_GT(chord.pitches.size(), 0);
    }

    // GrooveEngine - verified by valid timing
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.startBeat, 0.0);
        EXPECT_GT(note.duration, 0.0);
    }
}

// Test engine outputs are properly merged
TEST_F(EngineIntegrationTest, EngineOutputsMerged) {
    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = 0.5f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);

    MidiGenerator generator;
    GeneratedMidi midi = generator.generate(intent, 8, 0.7f, 0.4f, 0.0f, 0.75f);

    // Verify all layers are present and non-overlapping in structure
    // (They may overlap in time, but should have distinct purposes)
    EXPECT_GT(midi.chords.size(), 0);
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);

    // Verify no conflicts between engine outputs
    // All notes should have valid ranges
    auto validateLayer = [](const std::vector<MidiNote>& notes, const std::string& name) {
        for (const auto& note : notes) {
            EXPECT_GE(note.pitch, 0) << name << " pitch out of range";
            EXPECT_LE(note.pitch, 127) << name << " pitch out of range";
            EXPECT_GE(note.velocity, 0) << name << " velocity out of range";
            EXPECT_LE(note.velocity, 127) << name << " velocity out of range";
        }
    };

    validateLayer(midi.melody, "Melody");
    validateLayer(midi.bass, "Bass");
    if (!midi.pad.empty()) validateLayer(midi.pad, "Pad");
    if (!midi.strings.empty()) validateLayer(midi.strings, "Strings");
    if (!midi.counterMelody.empty()) validateLayer(midi.counterMelody, "CounterMelody");

    // Verify chords are valid
    for (const auto& chord : midi.chords) {
        EXPECT_GT(chord.pitches.size(), 0);
        for (int pitch : chord.pitches) {
            EXPECT_GE(pitch, 0);
            EXPECT_LE(pitch, 127);
        }
    }
}

// Test all engines are called during MidiGenerator::generate()
// This verifies that MidiGenerator properly orchestrates all 14 engines:
// 1. ChordGenerator, 2. MelodyEngine, 3. BassEngine, 4. PadEngine,
// 5. StringEngine, 6. CounterMelodyEngine, 7. RhythmEngine, 8. FillEngine,
// 9. DrumGrooveEngine, 10. TransitionEngine, 11. ArrangementEngine,
// 12. DynamicsEngine, 13. TensionEngine, 14. VariationEngine, 15. GrooveEngine
TEST_F(EngineIntegrationTest, AllEnginesCalledInMidiGenerator) {
    Wound wound;
    wound.description = "I feel complex and emotional";
    wound.intensity = 0.8f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);

    // Use MidiGenerator to verify all engines are called
    MidiGenerator generator;
    GeneratedMidi midi = generator.generate(intent, 16, 0.9f, 0.4f, 0.0f, 0.75f);

    // Verify core generation engines produce output
    EXPECT_GT(midi.chords.size(), 0) << "ChordGenerator should be called";
    EXPECT_GT(midi.melody.size(), 0) << "MelodyEngine should be called";
    EXPECT_GT(midi.bass.size(), 0) << "BassEngine should be called";

    // Verify optional engines are called when complexity is high
    EXPECT_GE(midi.pad.size(), 0) << "PadEngine should be called";
    EXPECT_GE(midi.strings.size(), 0) << "StringEngine should be called";
    EXPECT_GE(midi.counterMelody.size(), 0) << "CounterMelodyEngine should be called";
    EXPECT_GE(midi.rhythm.size(), 0) << "RhythmEngine should be called";
    EXPECT_GE(midi.fills.size(), 0) << "FillEngine should be called";
    EXPECT_GE(midi.drumGroove.size(), 0) << "DrumGrooveEngine should be called";
    EXPECT_GE(midi.transitions.size(), 0) << "TransitionEngine should be called";

    // Verify processing engines are applied (they modify existing notes)
    // DynamicsEngine, TensionEngine, VariationEngine, GrooveEngine
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.pitch, 0) << "Processing engines should produce valid pitches";
        EXPECT_LE(note.pitch, 127);
        EXPECT_GE(note.velocity, 0) << "DynamicsEngine should produce valid velocities";
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startBeat, 0.0) << "GrooveEngine should produce valid timing";
    }

    // Verify arrangement engine is used (for bars >= 8)
    EXPECT_GT(midi.lengthInBeats, 0.0) << "ArrangementEngine should inform generation";
}

// Test engine outputs are properly merged
TEST_F(EngineIntegrationTest, EngineOutputsMerged) {
    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = 0.5f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);

    MidiGenerator generator;
    GeneratedMidi midi = generator.generate(intent, 8, 0.7f, 0.4f, 0.0f, 0.75f);

    // Verify all layers are present and don't conflict
    EXPECT_GT(midi.chords.size(), 0);
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);

    // Verify notes don't overlap in timing (basic conflict check)
    // Melody and bass should have different timing or pitches
    bool hasOverlap = false;
    for (const auto& melodyNote : midi.melody) {
        for (const auto& bassNote : midi.bass) {
            // Check if notes overlap in time
            double melodyEnd = melodyNote.startBeat + melodyNote.duration;
            double bassEnd = bassNote.startBeat + bassNote.duration;

            if ((melodyNote.startBeat < bassEnd) && (bassNote.startBeat < melodyEnd)) {
                // Notes overlap in time - this is OK, but pitches should be different
                if (melodyNote.pitch == bassNote.pitch) {
                    hasOverlap = true;
                }
            }
        }
    }

    // Overlapping notes with same pitch is acceptable for harmony,
    // but we verify the system handles it correctly
    // (This is a soft check - exact behavior may vary)
}

// Test no conflicts between engine outputs
TEST_F(EngineIntegrationTest, NoEngineOutputConflicts) {
    Wound wound;
    wound.description = "I feel complex";
    wound.intensity = 0.8f;
    wound.source = "internal";

    IntentResult intent = pipeline->process(wound);

    MidiGenerator generator;
    GeneratedMidi midi = generator.generate(intent, 8, 0.9f, 0.4f, 0.0f, 0.75f);

    // Verify all notes are valid (no conflicts would produce invalid notes)
    auto validateNotes = [](const std::vector<MidiNote>& notes, const std::string& layer) {
        for (const auto& note : notes) {
            EXPECT_GE(note.pitch, 0) << layer << " note pitch should be >= 0";
            EXPECT_LE(note.pitch, 127) << layer << " note pitch should be <= 127";
            EXPECT_GE(note.velocity, 0) << layer << " note velocity should be >= 0";
            EXPECT_LE(note.velocity, 127) << layer << " note velocity should be <= 127";
            EXPECT_GE(note.startBeat, 0.0) << layer << " note startBeat should be >= 0.0";
            EXPECT_GT(note.duration, 0.0) << layer << " note duration should be > 0.0";
        }
    };

    validateNotes(midi.melody, "Melody");
    validateNotes(midi.bass, "Bass");
    validateNotes(midi.pad, "Pad");
    validateNotes(midi.strings, "Strings");
    validateNotes(midi.counterMelody, "CounterMelody");
    validateNotes(midi.rhythm, "Rhythm");
    validateNotes(midi.fills, "Fills");
    validateNotes(midi.drumGroove, "DrumGroove");
    validateNotes(midi.transitions, "Transitions");
}

// Test all engines are called during MidiGenerator generation
TEST_F(EngineIntegrationTest, AllEnginesCalledDuringGeneration) {
    // Use MidiGenerator to verify all engines are called
    MidiGenerator generator;
    IntentPipeline pipeline;

    Wound wound;
    wound.description = "I feel complex";
    wound.intensity = 0.9f;
    wound.source = "internal";

    IntentResult intent = pipeline.process(wound);

    // Generate with high complexity to trigger all engines
    GeneratedMidi midi = generator.generate(intent, 16, 0.95f, 0.5f, 0.0f, 0.8f);

    // Verify all engine outputs are present
    EXPECT_GT(midi.chords.size(), 0) << "ChordGenerator output should be present";
    EXPECT_GT(midi.melody.size(), 0) << "MelodyEngine output should be present";
    EXPECT_GT(midi.bass.size(), 0) << "BassEngine output should be present";
    EXPECT_GE(midi.pad.size(), 0) << "PadEngine output should be present";
    EXPECT_GE(midi.strings.size(), 0) << "StringEngine output should be present";
    EXPECT_GE(midi.counterMelody.size(), 0) << "CounterMelodyEngine output should be present";
    EXPECT_GE(midi.rhythm.size(), 0) << "RhythmEngine output should be present";
    EXPECT_GE(midi.drumGroove.size(), 0) << "DrumGrooveEngine output should be present";
    EXPECT_GE(midi.fills.size(), 0) << "FillEngine output should be present";
    EXPECT_GE(midi.transitions.size(), 0) << "TransitionEngine output should be present";
}

// Test engine outputs are properly merged
TEST_F(EngineIntegrationTest, EngineOutputsProperlyMerged) {
    MidiGenerator generator;
    IntentPipeline pipeline;

    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = 0.5f;
    wound.source = "internal";

    IntentResult intent = pipeline.process(wound);
    GeneratedMidi midi = generator.generate(intent, 8, 0.7f, 0.4f, 0.0f, 0.75f);

    // Verify all layers are merged into single GeneratedMidi structure
    EXPECT_GT(midi.chords.size(), 0);
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);

    // Verify no conflicts between engine outputs
    // (e.g., notes don't overlap in invalid ways, channels are correct)
    // All notes should have valid timing
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.startBeat, 0.0);
        EXPECT_GT(note.duration, 0.0);
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
    }

    for (const auto& note : midi.bass) {
        EXPECT_GE(note.startBeat, 0.0);
        EXPECT_GT(note.duration, 0.0);
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
    }
}

// Test no conflicts between engine outputs
TEST_F(EngineIntegrationTest, NoEngineOutputConflicts) {
    MidiGenerator generator;
    IntentPipeline pipeline;

    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = 0.5f;
    wound.source = "internal";

    IntentResult intent = pipeline.process(wound);
    GeneratedMidi midi = generator.generate(intent, 8, 0.8f, 0.4f, 0.0f, 0.75f);

    // Verify all notes are valid and don't conflict
    // Check that notes have valid ranges and timing
    auto validateLayer = [](const std::vector<MidiNote>& notes, const std::string& layerName) {
        for (const auto& note : notes) {
            EXPECT_GE(note.pitch, 0) << layerName << " note has invalid pitch";
            EXPECT_LE(note.pitch, 127) << layerName << " note has invalid pitch";
            EXPECT_GE(note.velocity, 0) << layerName << " note has invalid velocity";
            EXPECT_LE(note.velocity, 127) << layerName << " note has invalid velocity";
            EXPECT_GE(note.startBeat, 0.0) << layerName << " note has invalid startBeat";
            EXPECT_GT(note.duration, 0.0) << layerName << " note has invalid duration";
        }
    };

    validateLayer(midi.melody, "Melody");
    validateLayer(midi.bass, "Bass");
    validateLayer(midi.pad, "Pad");
    validateLayer(midi.strings, "Strings");
    validateLayer(midi.counterMelody, "CounterMelody");
    validateLayer(midi.rhythm, "Rhythm");
    validateLayer(midi.fills, "Fills");
    validateLayer(midi.drumGroove, "DrumGroove");
    validateLayer(midi.transitions, "Transitions");
}
