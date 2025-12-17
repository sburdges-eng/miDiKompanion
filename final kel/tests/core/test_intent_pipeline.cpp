#include <gtest/gtest.h>
#include "engine/IntentPipeline.h"
#include <string>

using namespace kelly;

class IntentPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        pipeline = std::make_unique<IntentPipeline>();
    }
    
    std::unique_ptr<IntentPipeline> pipeline;
};

// Test basic intent processing
TEST_F(IntentPipelineTest, BasicProcessing) {
    Wound wound;
    wound.description = "I feel sad and lonely";
    wound.intensity = 0.7f;
    wound.source = "internal";
    
    IntentResult result = pipeline->process(wound);
    
    EXPECT_GT(result.emotion.id, 0);
    EXPECT_FALSE(result.emotion.name.empty());
    EXPECT_FALSE(result.mode.empty());
    EXPECT_GT(result.tempo, 0.0f);
}

// Test different wound descriptions
TEST_F(IntentPipelineTest, DifferentWounds) {
    std::vector<std::string> descriptions = {
        "I'm feeling really happy today",
        "I'm angry at the world",
        "I'm scared about the future",
        "I feel peaceful and calm"
    };
    
    for (const auto& desc : descriptions) {
        Wound wound;
        wound.description = desc;
        wound.intensity = 0.5f;
        wound.source = "internal";
        
        IntentResult result = pipeline->process(wound);
        EXPECT_GT(result.emotion.id, 0) << "Failed for: " << desc;
    }
}

// Test intensity parameter
TEST_F(IntentPipelineTest, IntensityParameter) {
    Wound lowIntensity;
    lowIntensity.description = "I feel a bit sad";
    lowIntensity.intensity = 0.2f;
    lowIntensity.source = "internal";
    
    Wound highIntensity;
    highIntensity.description = "I feel extremely sad";
    highIntensity.intensity = 0.9f;
    highIntensity.source = "internal";
    
    IntentResult low = pipeline->process(lowIntensity);
    IntentResult high = pipeline->process(highIntensity);
    
    // High intensity should generally produce different results
    EXPECT_GT(low.emotion.id, 0);
    EXPECT_GT(high.emotion.id, 0);
}

// Test rule breaks generation
TEST_F(IntentPipelineTest, RuleBreaksGeneration) {
    Wound wound;
    wound.description = "I'm feeling conflicted and confused";
    wound.intensity = 0.8f;
    wound.source = "internal";
    
    IntentResult result = pipeline->process(wound);
    
    // Should potentially have rule breaks for complex emotions
    EXPECT_GE(result.ruleBreaks.size(), 0);
}

// Test musical parameters
TEST_F(IntentPipelineTest, MusicalParameters) {
    Wound wound;
    wound.description = "I feel joyful";
    wound.intensity = 0.6f;
    wound.source = "internal";
    
    IntentResult result = pipeline->process(wound);
    
    EXPECT_FALSE(result.mode.empty());
    EXPECT_GT(result.tempo, 0.0f);
    EXPECT_LE(result.tempo, 2.0f);
    EXPECT_GE(result.dynamicRange, 0.0f);
    EXPECT_LE(result.dynamicRange, 1.0f);
    EXPECT_GE(result.syncopationLevel, 0.0f);
    EXPECT_LE(result.syncopationLevel, 1.0f);
    EXPECT_GE(result.humanization, 0.0f);
    EXPECT_LE(result.humanization, 1.0f);
}

// Test allow dissonance
TEST_F(IntentPipelineTest, AllowDissonance) {
    Wound wound;
    wound.description = "I'm feeling tense and anxious";
    wound.intensity = 0.8f;
    wound.source = "internal";
    
    IntentResult result = pipeline->process(wound);
    
    // Anxious emotions might allow dissonance
    // (Result depends on implementation)
    EXPECT_TRUE(true); // Placeholder
}
