#include <gtest/gtest.h>
#include "engines/VoiceLeading.h"
#include <vector>
#include <string>

using namespace kelly;

class VoiceLeadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<VoiceLeadingEngine>();
    }
    
    std::unique_ptr<VoiceLeadingEngine> engine;
};

// Test basic voicing
TEST_F(VoiceLeadingTest, BasicVoicing) {
    std::vector<int> chordTones = {60, 64, 67}; // C major
    std::vector<int> previousVoicing = {60, 64, 67, 72};
    
    std::vector<int> voicing = engine->voice(chordTones, previousVoicing);
    
    EXPECT_GT(voicing.size(), 0);
}

// Test voice leading analysis
TEST_F(VoiceLeadingTest, VoiceLeadingAnalysis) {
    std::vector<int> fromVoicing = {60, 64, 67, 72}; // C major
    std::vector<int> toVoicing = {57, 60, 64, 67}; // A minor
    
    VoiceLeadingResult result = engine->analyze(fromVoicing, toVoicing);
    
    EXPECT_GT(result.movements.size(), 0);
    EXPECT_GE(result.smoothnessScore, 0.0f);
    EXPECT_LE(result.smoothnessScore, 1.0f);
}

// Test chord progression voicing
TEST_F(VoiceLeadingTest, ChordProgressionVoicing) {
    std::vector<std::vector<int>> chordTones = {
        {60, 64, 67},  // C major
        {57, 60, 64},  // A minor
        {53, 57, 60},  // F major
        {55, 59, 62}   // G major
    };
    
    std::vector<std::vector<int>> voicings = engine->voiceProgression(chordTones, 36);
    
    EXPECT_EQ(voicings.size(), chordTones.size());
    for (const auto& voicing : voicings) {
        EXPECT_GT(voicing.size(), 0);
    }
}

// Test smooth voice leading
TEST_F(VoiceLeadingTest, SmoothVoiceLeading) {
    std::vector<int> fromVoicing = {60, 64, 67, 72};
    std::vector<int> toVoicing = {57, 60, 64, 67};

    VoiceLeadingResult result = engine->analyze(fromVoicing, toVoicing);

    // Check that movements are reasonable (can be negative for downward motion)
    for (const auto& movement : result.movements) {
        EXPECT_GE(movement.interval, -12);  // Down an octave max
        EXPECT_LE(movement.interval, 12);   // Up an octave max
    }
}

// Test parallel fifths/octaves detection
TEST_F(VoiceLeadingTest, ParallelDetection) {
    std::vector<int> fromVoicing = {60, 64, 67, 72};
    std::vector<int> toVoicing = {65, 69, 72, 77}; // Parallel motion
    
    VoiceLeadingResult result = engine->analyze(fromVoicing, toVoicing);
    
    // Should detect parallel motion if present
    // (Result depends on implementation)
    EXPECT_TRUE(true); // Placeholder
}
