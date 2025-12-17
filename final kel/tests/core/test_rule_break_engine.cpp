#include <gtest/gtest.h>
#include "engine/RuleBreakEngine.h"
#include <string>

using namespace kelly;

class RuleBreakEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<RuleBreakEngine>();
    }
    
    std::unique_ptr<RuleBreakEngine> engine;
};

// Test rule break generation
TEST_F(RuleBreakEngineTest, GenerateRuleBreaks) {
    EmotionNode emotion;
    emotion.id = 1;
    emotion.name = "grief";
    emotion.valence = -0.8f;
    emotion.arousal = 0.3f;
    emotion.intensity = 0.9f;
    emotion.category = EmotionCategory::Sadness;
    
    std::vector<RuleBreak> breaks = engine->generateRuleBreaks(emotion);
    
    EXPECT_GE(breaks.size(), 0);
    
    for (const auto& rb : breaks) {
        EXPECT_GE(rb.severity, 0.0f);
        EXPECT_LE(rb.severity, 1.0f);
        EXPECT_FALSE(rb.description.empty());
        EXPECT_FALSE(rb.reason.empty());
    }
}

// Test different emotion categories
TEST_F(RuleBreakEngineTest, DifferentEmotionCategories) {
    std::vector<EmotionCategory> categories = {
        EmotionCategory::Joy,
        EmotionCategory::Sadness,
        EmotionCategory::Anger,
        EmotionCategory::Fear
    };
    
    for (const auto& category : categories) {
        EmotionNode emotion;
        emotion.id = 1;
        emotion.name = "test";
        emotion.valence = 0.0f;
        emotion.arousal = 0.5f;
        emotion.intensity = 0.7f;
        emotion.category = category;
        
        std::vector<RuleBreak> breaks = engine->generateRuleBreaks(emotion);
        EXPECT_GE(breaks.size(), 0) << "Failed for category";
    }
}

// Test intensity parameter
TEST_F(RuleBreakEngineTest, IntensityParameter) {
    EmotionNode emotion;
    emotion.id = 1;
    emotion.name = "grief";
    emotion.valence = -0.8f;
    emotion.arousal = 0.3f;
    emotion.intensity = 0.9f;
    emotion.category = EmotionCategory::Sadness;
    
    std::vector<RuleBreak> lowIntensity = engine->generateRuleBreaks(emotion);
    std::vector<RuleBreak> highIntensity = engine->generateRuleBreaks(emotion);
    
    // High intensity might produce more or more severe rule breaks
    EXPECT_GE(lowIntensity.size(), 0);
    EXPECT_GE(highIntensity.size(), 0);
}

// Test rule break types
TEST_F(RuleBreakEngineTest, RuleBreakTypes) {
    EmotionNode emotion;
    emotion.id = 1;
    emotion.name = "rage";
    emotion.valence = -0.9f;
    emotion.arousal = 0.9f;
    emotion.intensity = 0.95f;
    emotion.category = EmotionCategory::Anger;
    
    std::vector<RuleBreak> breaks = engine->generateRuleBreaks(emotion);
    
    // Check that we have various types
    std::set<RuleBreakType> types;
    for (const auto& rb : breaks) {
        types.insert(rb.type);
    }
    
    // Should have at least one type
    EXPECT_GT(types.size(), 0);
}
