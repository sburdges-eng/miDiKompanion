#include <gtest/gtest.h>
#include "engines/CounterMelodyEngine.h"
#include <vector>
#include <string>
#include "common/Types.h"

using namespace kelly;

class CounterMelodyEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<CounterMelodyEngine>();
        
        // Create test primary melody
        primaryMelody.clear();
        for (int i = 0; i < 8; ++i) {
            MidiNote note;
            note.pitch = 60 + i;
            note.velocity = 100;
            note.startBeat = i * 0.5;
            note.duration = 0.5;
            primaryMelody.push_back(note);
        }
    }
    
    std::unique_ptr<CounterMelodyEngine> engine;
    std::vector<MidiNote> primaryMelody;
};

// Test basic generation
TEST_F(CounterMelodyEngineTest, BasicGeneration) {
    CounterMelodyOutput output = engine->generate("neutral", primaryMelody, "C", "major");
    
    EXPECT_GT(output.notes.size(), 0);
}

// Test different emotions
TEST_F(CounterMelodyEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        CounterMelodyOutput output = engine->generate(emotion, primaryMelody, "C", "major");
        EXPECT_GT(output.notes.size(), 0) << "Failed for emotion: " << emotion;
    }
}

// Test note validity
TEST_F(CounterMelodyEngineTest, NoteValidity) {
    CounterMelodyOutput output = engine->generate("neutral", primaryMelody, "C", "major");
    
    for (const auto& note : output.notes) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GT(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startTick, 0);
        EXPECT_GT(note.durationTicks, 0);
    }
}

// Test config-based generation
TEST_F(CounterMelodyEngineTest, ConfigGeneration) {
    CounterMelodyConfig config;
    config.emotion = "neutral";
    config.primaryMelody = primaryMelody;
    config.key = "C";
    config.mode = "major";
    config.type = CounterMelodyType::Contrary;
    config.relation = CounterMelodyRelation::Third;
    
    CounterMelodyOutput output = engine->generate(config);
    EXPECT_EQ(output.typeUsed, CounterMelodyType::Contrary);
    EXPECT_EQ(output.relationUsed, CounterMelodyRelation::Third);
}

// Test voice leading rules
TEST_F(CounterMelodyEngineTest, VoiceLeadingRules) {
    CounterMelodyOutput output = engine->generate("neutral", primaryMelody, "C", "major");
    
    // Counter-melody should follow voice leading principles
    // Check for reasonable intervals between consecutive notes
    for (size_t i = 1; i < output.notes.size(); ++i) {
        int interval = abs(output.notes[i].pitch - output.notes[i-1].pitch);
        // Most intervals should be reasonable (not huge jumps)
        EXPECT_LE(interval, 12) << "Interval too large at note " << i;
    }
}
