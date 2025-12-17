#include <gtest/gtest.h>
#include "engines/PadEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class PadEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<PadEngine>();
    }
    
    std::unique_ptr<PadEngine> engine;
    std::vector<std::string> testProgression = {"C", "Am", "F", "G"};
};

// Test basic generation
TEST_F(PadEngineTest, BasicGeneration) {
    PadOutput output = engine->generate("neutral", testProgression, "C", 4, 120);
    
    EXPECT_GT(output.notes.size(), 0);
    EXPECT_EQ(output.emotion, "neutral");
}

// Test different emotions
TEST_F(PadEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        PadOutput output = engine->generate(emotion, testProgression, "C", 4, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for emotion: " << emotion;
    }
}

// Test texture override
TEST_F(PadEngineTest, TextureOverride) {
    PadConfig config;
    config.emotion = "neutral";
    config.chordProgression = testProgression;
    config.key = "C";
    config.bars = 4;
    config.tempoBpm = 120;
    config.textureOverride = PadTexture::Warm;
    
    PadOutput output = engine->generate(config);
    EXPECT_EQ(output.textureUsed, PadTexture::Warm);
}

// Test movement override
TEST_F(PadEngineTest, MovementOverride) {
    PadConfig config;
    config.emotion = "neutral";
    config.chordProgression = testProgression;
    config.key = "C";
    config.bars = 4;
    config.tempoBpm = 120;
    config.movementOverride = PadMovement::Breathing;
    
    PadOutput output = engine->generate(config);
    EXPECT_EQ(output.movementUsed, PadMovement::Breathing);
}

// Test voicing override
TEST_F(PadEngineTest, VoicingOverride) {
    PadConfig config;
    config.emotion = "neutral";
    config.chordProgression = testProgression;
    config.key = "C";
    config.bars = 4;
    config.tempoBpm = 120;
    config.voicingOverride = PadVoicing::Open;
    
    PadOutput output = engine->generate(config);
    EXPECT_EQ(output.voicingUsed, PadVoicing::Open);
}

// Test note validity
TEST_F(PadEngineTest, NoteValidity) {
    PadOutput output = engine->generate("neutral", testProgression, "C", 4, 120);
    
    for (const auto& note : output.notes) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GT(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startTick, 0);
        EXPECT_GT(note.durationTicks, 0);
        EXPECT_GE(note.channel, 0);
        EXPECT_LE(note.channel, 15);
    }
}

// Test sustained notes (pads should have longer durations)
TEST_F(PadEngineTest, SustainedNotes) {
    PadOutput output = engine->generate("neutral", testProgression, "C", 4, 120);
    
    // Pads typically have longer durations
    int barTicks = 1920; // 4 beats * 480 ticks/beat
    for (const auto& note : output.notes) {
        // Most pad notes should be at least a beat long
        EXPECT_GE(note.durationTicks, 240); // Quarter note minimum
    }
}

// Test seed reproducibility
TEST_F(PadEngineTest, SeedReproducibility) {
    PadConfig config1;
    config1.emotion = "neutral";
    config1.chordProgression = testProgression;
    config1.key = "C";
    config1.bars = 2;
    config1.tempoBpm = 120;
    config1.seed = 42;
    
    PadConfig config2 = config1;
    
    PadOutput output1 = engine->generate(config1);
    PadOutput output2 = engine->generate(config2);
    
    EXPECT_EQ(output1.notes.size(), output2.notes.size());
    for (size_t i = 0; i < output1.notes.size(); ++i) {
        EXPECT_EQ(output1.notes[i].pitch, output2.notes[i].pitch);
    }
}

// Test different bar counts
TEST_F(PadEngineTest, DifferentBarCounts) {
    for (int bars : {1, 2, 4, 8}) {
        PadOutput output = engine->generate("neutral", testProgression, "C", bars, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for bars: " << bars;
    }
}

// Test GM instrument assignment
TEST_F(PadEngineTest, GMInstrumentAssignment) {
    PadOutput output = engine->generate("neutral", testProgression, "C", 2, 120);
    
    EXPECT_GE(output.gmInstrument, 0);
    EXPECT_LE(output.gmInstrument, 127);
}
