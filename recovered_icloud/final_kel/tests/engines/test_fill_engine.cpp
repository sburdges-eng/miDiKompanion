#include <gtest/gtest.h>
#include "engines/FillEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class FillEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<FillEngine>();
    }
    
    std::unique_ptr<FillEngine> engine;
};

// Test basic generation
TEST_F(FillEngineTest, BasicGeneration) {
    FillOutput output = engine->generate("neutral", FillLength::Full, 0, 120);
    
    EXPECT_GT(output.hits.size(), 0);
    EXPECT_EQ(output.length, FillLength::Full);
}

// Test different emotions
TEST_F(FillEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        FillOutput output = engine->generate(emotion, FillLength::Full, 0, 120);
        EXPECT_GT(output.hits.size(), 0) << "Failed for emotion: " << emotion;
    }
}

// Test different fill lengths
TEST_F(FillEngineTest, DifferentFillLengths) {
    std::vector<FillLength> lengths = {FillLength::Quarter, FillLength::Half, FillLength::Full, FillLength::Double};
    
    for (const auto& length : lengths) {
        FillOutput output = engine->generate("neutral", length, 0, 120);
        EXPECT_GT(output.hits.size(), 0) << "Failed for length";
        EXPECT_EQ(output.length, length);
    }
}

// Test hit validity
TEST_F(FillEngineTest, HitValidity) {
    FillOutput output = engine->generate("neutral", FillLength::Full, 0, 120);
    
    for (const auto& hit : output.hits) {
        EXPECT_GE(hit.note, 0);
        EXPECT_LE(hit.note, 127);
        EXPECT_GT(hit.velocity, 0);
        EXPECT_LE(hit.velocity, 127);
        EXPECT_GE(hit.startTick, 0);
        EXPECT_GT(hit.duration, 0);
    }
}

// Test config-based generation
TEST_F(FillEngineTest, ConfigGeneration) {
    FillConfig config;
    config.emotion = "neutral";
    config.length = FillLength::Full;
    config.startTick = 0;
    config.tempoBpm = 120;
    config.typeOverride = FillType::TomRoll;
    config.intensityOverride = FillIntensity::Intense;
    
    FillOutput output = engine->generate(config);
    EXPECT_EQ(output.typeUsed, FillType::TomRoll);
    EXPECT_EQ(output.intensityUsed, FillIntensity::Intense);
}
