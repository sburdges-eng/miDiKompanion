#include <gtest/gtest.h>
#include "engines/StringEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class StringEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<StringEngine>();
    }
    
    std::unique_ptr<StringEngine> engine;
    std::vector<std::string> testProgression = {"C", "Am", "F", "G"};
};

// Test basic generation
TEST_F(StringEngineTest, BasicGeneration) {
    StringOutput output = engine->generate("neutral", testProgression, "C", 4, 120);
    
    EXPECT_GT(output.notes.size(), 0);
    EXPECT_EQ(output.emotion, "neutral");
}

// Test different emotions
TEST_F(StringEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        StringOutput output = engine->generate(emotion, testProgression, "C", 4, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for emotion: " << emotion;
    }
}

// Test note validity
TEST_F(StringEngineTest, NoteValidity) {
    StringOutput output = engine->generate("neutral", testProgression, "C", 4, 120);
    
    for (const auto& note : output.notes) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GT(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startTick, 0);
        EXPECT_GT(note.durationTicks, 0);
    }
}
