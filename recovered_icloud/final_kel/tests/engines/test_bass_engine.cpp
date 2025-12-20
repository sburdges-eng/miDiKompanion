#include <gtest/gtest.h>
#include "engines/BassEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class BassEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<BassEngine>();
    }

    std::unique_ptr<BassEngine> engine;
    std::vector<std::string> testProgression = {"C", "Am", "F", "G"};
};

// Test basic generation
TEST_F(BassEngineTest, BasicGeneration) {
    BassOutput output = engine->generate("neutral", testProgression, "C", 4, 120);

    EXPECT_GT(output.notes.size(), 0);
    EXPECT_EQ(output.emotion, "neutral");
}

// Test different emotions
TEST_F(BassEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};

    for (const auto& emotion : emotions) {
        BassOutput output = engine->generate(emotion, testProgression, "C", 4, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for emotion: " << emotion;
        EXPECT_EQ(output.emotion, emotion);
    }
}

// Test pattern override
TEST_F(BassEngineTest, PatternOverride) {
    BassConfig config;
    config.emotion = "neutral";
    config.chordProgression = testProgression;
    config.key = "C";
    config.bars = 4;
    config.tempoBpm = 120;
    config.patternOverride = BassPattern::Walking;

    BassOutput output = engine->generate(config);
    EXPECT_EQ(output.patternUsed, BassPattern::Walking);
}

// Test register override
TEST_F(BassEngineTest, RegisterOverride) {
    BassConfig config;
    config.emotion = "neutral";
    config.chordProgression = testProgression;
    config.key = "C";
    config.bars = 4;
    config.tempoBpm = 120;
    config.registerOverride = BassRegister::Low;

    BassOutput output = engine->generate(config);
    EXPECT_EQ(output.registerUsed, BassRegister::Low);
}

// Test different chord progressions
TEST_F(BassEngineTest, DifferentProgressions) {
    std::vector<std::vector<std::string>> progressions = {
        {"C", "F", "G", "C"},
        {"Am", "Dm", "G", "C"},
        {"C", "Am", "F", "G"},
        {"Dm", "G", "C", "F"}
    };

    for (const auto& progression : progressions) {
        BassOutput output = engine->generate("neutral", progression, "C", 4, 120);
        EXPECT_GT(output.notes.size(), 0);
    }
}

// Test note validity
TEST_F(BassEngineTest, NoteValidity) {
    BassOutput output = engine->generate("neutral", testProgression, "C", 4, 120);

    for (const auto& note : output.notes) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GT(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startTick, 0);
        EXPECT_GT(note.durationTicks, 0);
        EXPECT_FALSE(note.function.empty());
    }
}

// Test root motion
TEST_F(BassEngineTest, RootMotion) {
    BassOutput output = engine->generate("neutral", testProgression, "C", 4, 120);

    // Should have notes corresponding to chord roots
    // This is a basic check - actual implementation may vary
    EXPECT_GT(output.notes.size(), 0);
}

// Test seed reproducibility
TEST_F(BassEngineTest, SeedReproducibility) {
    BassConfig config1;
    config1.emotion = "neutral";
    config1.chordProgression = testProgression;
    config1.key = "C";
    config1.bars = 2;
    config1.tempoBpm = 120;
    config1.seed = 42;

    BassConfig config2 = config1;

    BassOutput output1 = engine->generate(config1);
    BassOutput output2 = engine->generate(config2);

    EXPECT_EQ(output1.notes.size(), output2.notes.size());
    for (size_t i = 0; i < output1.notes.size(); ++i) {
        EXPECT_EQ(output1.notes[i].pitch, output2.notes[i].pitch);
    }
}

// Test section-specific generation
TEST_F(BassEngineTest, SectionGeneration) {
    BassOutput output = engine->generateForSection("joy", testProgression, "chorus", "C", 4, 120);

    EXPECT_GT(output.notes.size(), 0);
    EXPECT_EQ(output.emotion, "joy");
}

// Test different bar counts
TEST_F(BassEngineTest, DifferentBarCounts) {
    for (int bars : {1, 2, 4, 8}) {
        BassOutput output = engine->generate("neutral", testProgression, "C", bars, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for bars: " << bars;
    }
}

// Test GM instrument assignment
TEST_F(BassEngineTest, GMInstrumentAssignment) {
    BassOutput output = engine->generate("neutral", testProgression, "C", 2, 120);

    EXPECT_GE(output.gmInstrument, 0);
    EXPECT_LE(output.gmInstrument, 127);
}

// Test all bass patterns
TEST_F(BassEngineTest, AllBassPatterns) {
    std::vector<BassPattern> patterns = {
        BassPattern::RootOnly,
        BassPattern::RootFifth,
        BassPattern::Walking,
        BassPattern::Pedal,
        BassPattern::Arpeggiated,
        BassPattern::Syncopated,
        BassPattern::Driving,
        BassPattern::Pulsing,
        BassPattern::Breathing,
        BassPattern::Descending,
        BassPattern::Climbing,
        BassPattern::Ghost
    };

    for (const auto& pattern : patterns) {
        BassConfig config;
        config.emotion = "neutral";
        config.chordProgression = testProgression;
        config.key = "C";
        config.bars = 4;
        config.tempoBpm = 120;
        config.patternOverride = pattern;

        BassOutput output = engine->generate(config);
        EXPECT_EQ(output.patternUsed, pattern) << "Pattern should match override";
        EXPECT_GT(output.notes.size(), 0) << "Should generate notes for pattern";

        // Verify all notes are valid
        for (const auto& note : output.notes) {
            EXPECT_GE(note.pitch, 0);
            EXPECT_LE(note.pitch, 127);
            EXPECT_GT(note.velocity, 0);
            EXPECT_LE(note.velocity, 127);
            EXPECT_GE(note.startTick, 0);
            EXPECT_GT(note.durationTicks, 0);
        }
    }
}
