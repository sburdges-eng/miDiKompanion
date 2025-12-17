#include <gtest/gtest.h>
#include "engines/TransitionEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class TransitionEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<TransitionEngine>();
    }
    
    std::unique_ptr<TransitionEngine> engine;
};

// Test basic generation
TEST_F(TransitionEngineTest, BasicGeneration) {
    TransitionOutput output = engine->generate("neutral", TransitionType::Crossfade, 2, 120);
    
    EXPECT_GT(output.notes.size(), 0);
    EXPECT_EQ(output.typeUsed, TransitionType::Crossfade);
}

// Test different transition types
TEST_F(TransitionEngineTest, DifferentTransitionTypes) {
    std::vector<TransitionType> types = {
        TransitionType::Crossfade,
        TransitionType::Build,
        TransitionType::Breakdown,
        TransitionType::Riser
    };
    
    for (const auto& type : types) {
        TransitionOutput output = engine->generate("neutral", type, 2, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for type";
        EXPECT_EQ(output.typeUsed, type);
    }
}

// Test config-based generation
TEST_F(TransitionEngineTest, ConfigGeneration) {
    TransitionConfig config;
    config.emotion = "neutral";
    config.type = TransitionType::Build;
    config.durationBars = 2;
    config.tempoBpm = 120;
    config.fromSection = "verse";
    config.toSection = "chorus";
    
    TransitionOutput output = engine->generate(config);
    EXPECT_EQ(output.typeUsed, TransitionType::Build);
}

// Test note validity
TEST_F(TransitionEngineTest, NoteValidity) {
    TransitionOutput output = engine->generate("neutral", TransitionType::Crossfade, 2, 120);
    
    for (const auto& note : output.notes) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GT(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startTick, 0);
        EXPECT_GT(note.durationTicks, 0);
        EXPECT_FALSE(note.type.empty());
    }
}

// Test helper methods
TEST_F(TransitionEngineTest, HelperMethods) {
    TransitionOutput build = engine->createBuild("neutral", 2, 120);
    TransitionOutput breakdown = engine->createBreakdown("neutral", 2, 120);
    TransitionOutput drop = engine->createDrop("neutral", 2, 120);
    
    EXPECT_GT(build.notes.size(), 0);
    EXPECT_GT(breakdown.notes.size(), 0);
    EXPECT_GT(drop.notes.size(), 0);
}
