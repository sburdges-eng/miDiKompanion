#include <gtest/gtest.h>
#include "engines/VariationEngine.h"
#include <vector>
#include "common/Types.h"

using namespace kelly;

class VariationEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<VariationEngine>();
        
        // Create test notes
        testNotes.clear();
        for (int i = 0; i < 8; ++i) {
            MidiNote note;
            note.pitch = 60 + i;
            note.velocity = 100;
            note.startBeat = i * 0.5;
            note.duration = 0.5;
            testNotes.push_back(note);
        }
    }
    
    std::unique_ptr<VariationEngine> engine;
    std::vector<MidiNote> testNotes;
};

// Test basic variation
TEST_F(VariationEngineTest, BasicVariation) {
    VariationOutput output = engine->generate(testNotes, "neutral", 0.5f);
    
    EXPECT_GT(output.notes.size(), 0);
    EXPECT_EQ(output.emotion, "neutral");
}

// Test different emotions
TEST_F(VariationEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        VariationOutput output = engine->generate(testNotes, emotion, 0.5f);
        EXPECT_GT(output.notes.size(), 0) << "Failed for emotion: " << emotion;
    }
}

// Test config-based variation
TEST_F(VariationEngineTest, ConfigVariation) {
    VariationConfig config;
    config.emotion = "neutral";
    config.source = testNotes;
    config.key = "C";
    config.mode = "major";
    config.intensity = 0.5f;
    config.typeOverride = VariationType::Ornamentation;
    
    VariationOutput output = engine->generate(config);
    EXPECT_EQ(output.typeUsed, VariationType::Ornamentation);
}

// Test variation amount
TEST_F(VariationEngineTest, VariationAmount) {
    VariationOutput lowVar = engine->generate(testNotes, "neutral", 0.1f);
    VariationOutput highVar = engine->generate(testNotes, "neutral", 0.9f);
    
    EXPECT_GT(lowVar.notes.size(), 0);
    EXPECT_GT(highVar.notes.size(), 0);
}

// Test note validity
TEST_F(VariationEngineTest, NoteValidity) {
    VariationOutput output = engine->generate(testNotes, "neutral", 0.5f);
    
    for (const auto& note : output.notes) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GT(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
    }
}

// Test similarity score
TEST_F(VariationEngineTest, SimilarityScore) {
    VariationOutput output = engine->generate(testNotes, "neutral", 0.5f);
    
    EXPECT_GE(output.similarityScore, 0.0f);
    EXPECT_LE(output.similarityScore, 1.0f);
}
