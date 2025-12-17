#include <gtest/gtest.h>
#include "midi/ChordGenerator.h"
#include "engine/IntentPipeline.h"
#include <vector>
#include <string>

using namespace kelly;

class ChordGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        generator = std::make_unique<ChordGenerator>();
        pipeline = std::make_unique<IntentPipeline>();
    }
    
    std::unique_ptr<ChordGenerator> generator;
    std::unique_ptr<IntentPipeline> pipeline;
};

// Test basic chord generation
TEST_F(ChordGeneratorTest, BasicGeneration) {
    Wound wound;
    wound.description = "I feel joyful";
    wound.intensity = 0.6f;
    wound.source = "internal";
    
    IntentResult intent = pipeline->process(wound);
    std::vector<Chord> chords = generator->generate(intent, 4);
    
    EXPECT_GT(chords.size(), 0);
    EXPECT_LE(chords.size(), 8); // Reasonable number of chords
}

// Test different bar counts
TEST_F(ChordGeneratorTest, DifferentBarCounts) {
    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = 0.5f;
    wound.source = "internal";
    
    IntentResult intent = pipeline->process(wound);
    
    for (int bars : {2, 4, 8, 16}) {
        std::vector<Chord> chords = generator->generate(intent, bars);
        EXPECT_GT(chords.size(), 0) << "Failed for bars: " << bars;
    }
}

// Test progression generation
TEST_F(ChordGeneratorTest, ProgressionGeneration) {
    std::vector<Chord> chords = generator->generateProgression(
        "major",
        60, // C
        4,
        false, // no dissonance
        0.5f
    );
    
    EXPECT_GT(chords.size(), 0);
    
    for (const auto& chord : chords) {
        EXPECT_GT(chord.pitches.size(), 0);
        EXPECT_FALSE(chord.name.empty());
        EXPECT_GE(chord.startBeat, 0.0);
        EXPECT_GT(chord.duration, 0.0);
    }
}

// Test different modes
TEST_F(ChordGeneratorTest, DifferentModes) {
    std::vector<std::string> modes = {"major", "minor", "dorian", "mixolydian"};
    
    for (const auto& mode : modes) {
        std::vector<Chord> chords = generator->generateProgression(
            mode,
            60,
            4,
            false,
            0.5f
        );
        EXPECT_GT(chords.size(), 0) << "Failed for mode: " << mode;
    }
}

// Test dissonance parameter
TEST_F(ChordGeneratorTest, DissonanceParameter) {
    Wound wound;
    wound.description = "I feel tense";
    wound.intensity = 0.8f;
    wound.source = "internal";
    
    IntentResult intent = pipeline->process(wound);
    intent.allowDissonance = true;
    
    std::vector<Chord> chords = generator->generate(intent, 4);
    EXPECT_GT(chords.size(), 0);
}

// Test chord validity
TEST_F(ChordGeneratorTest, ChordValidity) {
    std::vector<Chord> chords = generator->generateProgression("major", 60, 4, false, 0.5f);
    
    for (const auto& chord : chords) {
        EXPECT_GT(chord.pitches.size(), 0);
        for (int pitch : chord.pitches) {
            EXPECT_GE(pitch, 0);
            EXPECT_LE(pitch, 127);
        }
    }
}
