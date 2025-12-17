#include <gtest/gtest.h>
#include "engines/RhythmEngine.h"
#include <vector>
#include <string>
#include <set>

using namespace kelly;

class RhythmEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<RhythmEngine>();
    }
    
    std::unique_ptr<RhythmEngine> engine;
};

// Test basic generation
TEST_F(RhythmEngineTest, BasicGeneration) {
    RhythmOutput output = engine->generate("neutral", 4, 120);
    
    EXPECT_GT(output.hits.size(), 0);
    EXPECT_EQ(output.config.emotion, "neutral");
    EXPECT_EQ(output.totalTicks, 7680); // 4 bars * 4 beats * 480 ticks/beat
}

// Test different emotions
TEST_F(RhythmEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear", "neutral"};
    
    for (const auto& emotion : emotions) {
        RhythmOutput output = engine->generate(emotion, 4, 120);
        EXPECT_GT(output.hits.size(), 0) << "Failed for emotion: " << emotion;
        EXPECT_EQ(output.config.emotion, emotion);
    }
}

// Test groove override
TEST_F(RhythmEngineTest, GrooveOverride) {
    RhythmConfig config;
    config.emotion = "neutral";
    config.bars = 4;
    config.tempoBpm = 120;
    config.grooveOverride = GrooveType::Swing;
    
    RhythmOutput output = engine->generate(config);
    EXPECT_EQ(output.grooveUsed, GrooveType::Swing);
}

// Test density override
TEST_F(RhythmEngineTest, DensityOverride) {
    RhythmConfig config;
    config.emotion = "neutral";
    config.bars = 4;
    config.tempoBpm = 120;
    config.densityOverride = PatternDensity::Busy;
    
    RhythmOutput output = engine->generate(config);
    EXPECT_EQ(output.densityUsed, PatternDensity::Busy);
}

// Test genre parameter
TEST_F(RhythmEngineTest, GenreParameter) {
    std::vector<std::string> genres = {"rock", "jazz", "hiphop", "electronic", ""};
    
    for (const auto& genre : genres) {
        RhythmOutput output = engine->generate("neutral", 4, 120, genre);
        EXPECT_GT(output.hits.size(), 0) << "Failed for genre: " << genre;
    }
}

// Test hit validity
TEST_F(RhythmEngineTest, HitValidity) {
    RhythmOutput output = engine->generate("neutral", 4, 120);
    
    for (const auto& hit : output.hits) {
        EXPECT_GE(hit.note, 0);
        EXPECT_LE(hit.note, 127);
        EXPECT_GT(hit.velocity, 0);
        EXPECT_LE(hit.velocity, 127);
        EXPECT_GE(hit.startTick, 0);
        EXPECT_GT(hit.duration, 0);
        EXPECT_FALSE(hit.type.empty());
    }
}

// Test instruments used
TEST_F(RhythmEngineTest, InstrumentsUsed) {
    RhythmOutput output = engine->generate("neutral", 4, 120);
    
    EXPECT_GT(output.instrumentsUsed.size(), 0);
    // Common drum instruments should be present
    bool hasKick = false, hasSnare = false, hasHat = false;
    for (const auto& hit : output.hits) {
        if (hit.note == GMDrum::KICK) hasKick = true;
        if (hit.note == GMDrum::SNARE) hasSnare = true;
        if (hit.note == GMDrum::CLOSED_HAT || hit.note == GMDrum::OPEN_HAT) hasHat = true;
    }
    // At least one should be present in a typical pattern
    EXPECT_TRUE(hasKick || hasSnare || hasHat);
}

// Test intro generation
TEST_F(RhythmEngineTest, IntroGeneration) {
    RhythmOutput output = engine->generateIntro("neutral", 2, 120);
    
    EXPECT_GT(output.hits.size(), 0);
    EXPECT_EQ(output.config.emotion, "neutral");
}

// Test buildup generation
TEST_F(RhythmEngineTest, BuildupGeneration) {
    RhythmOutput output = engine->generateBuildup("neutral", 2, 120);
    
    EXPECT_GT(output.hits.size(), 0);
    EXPECT_EQ(output.config.emotion, "neutral");
}

// Test fill inclusion
TEST_F(RhythmEngineTest, FillInclusion) {
    RhythmConfig config;
    config.emotion = "neutral";
    config.bars = 4;
    config.tempoBpm = 120;
    config.includeFills = true;
    config.fillOnBar = {3, 7}; // Fills on bars 3 and 7
    
    RhythmOutput output = engine->generate(config);
    EXPECT_GT(output.hits.size(), 0);
}

// Test seed reproducibility
TEST_F(RhythmEngineTest, SeedReproducibility) {
    RhythmConfig config1;
    config1.emotion = "neutral";
    config1.bars = 2;
    config1.tempoBpm = 120;
    config1.seed = 42;
    
    RhythmConfig config2 = config1;
    
    RhythmOutput output1 = engine->generate(config1);
    RhythmOutput output2 = engine->generate(config2);
    
    EXPECT_EQ(output1.hits.size(), output2.hits.size());
    for (size_t i = 0; i < output1.hits.size(); ++i) {
        EXPECT_EQ(output1.hits[i].note, output2.hits[i].note);
        EXPECT_EQ(output1.hits[i].startTick, output2.hits[i].startTick);
    }
}

// Test different bar counts
TEST_F(RhythmEngineTest, DifferentBarCounts) {
    for (int bars : {1, 2, 4, 8, 16}) {
        RhythmOutput output = engine->generate("neutral", bars, 120);
        EXPECT_GT(output.hits.size(), 0) << "Failed for bars: " << bars;
        EXPECT_EQ(output.totalTicks, bars * 4 * 480);
    }
}

// Test different tempos
TEST_F(RhythmEngineTest, DifferentTempos) {
    std::vector<int> tempos = {60, 90, 120, 150, 180};
    
    for (int tempo : tempos) {
        RhythmOutput output = engine->generate("neutral", 4, tempo);
        EXPECT_GT(output.hits.size(), 0) << "Failed for tempo: " << tempo;
    }
}

// Test time signature
TEST_F(RhythmEngineTest, TimeSignature) {
    RhythmConfig config;
    config.emotion = "neutral";
    config.bars = 4;
    config.tempoBpm = 120;
    config.timeSignature = {3, 4}; // 3/4 time
    
    RhythmOutput output = engine->generate(config);
    EXPECT_GT(output.hits.size(), 0);
}
