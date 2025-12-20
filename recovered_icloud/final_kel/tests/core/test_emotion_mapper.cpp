#include <gtest/gtest.h>
#include "engine/EmotionMapper.h"
#include <string>

using namespace kelly;

class EmotionMapperTest : public ::testing::Test {
protected:
    void SetUp() override {
        mapper = std::make_unique<EmotionMapper>();
    }
    
    std::unique_ptr<EmotionMapper> mapper;
};

// Test basic emotion mapping
TEST_F(EmotionMapperTest, BasicMapping) {
    EmotionalState state;
    state.valence = 0.7f;
    state.arousal = 0.6f;
    state.intensity = 0.5f;
    state.primaryEmotion = "joy";
    
    MusicalParameters params = mapper->mapToParameters(state);
    
    EXPECT_GT(params.tempoSuggested, 0);
    EXPECT_FALSE(params.keySuggested.empty());
    EXPECT_FALSE(params.modeSuggested.empty());
    EXPECT_GE(params.dissonance, 0.0f);
    EXPECT_LE(params.dissonance, 1.0f);
    EXPECT_GE(params.velocityMin, 0);
    EXPECT_LE(params.velocityMin, 127);
    EXPECT_GE(params.velocityMax, params.velocityMin);
    EXPECT_LE(params.velocityMax, 127);
}

// Test different valence values
TEST_F(EmotionMapperTest, DifferentValence) {
    EmotionalState positive;
    positive.valence = 0.8f;
    positive.arousal = 0.5f;
    positive.intensity = 0.5f;
    positive.primaryEmotion = "joy";
    
    EmotionalState negative;
    negative.valence = -0.8f;
    negative.arousal = 0.5f;
    negative.intensity = 0.5f;
    negative.primaryEmotion = "sad";
    
    MusicalParameters posParams = mapper->mapToParameters(positive);
    MusicalParameters negParams = mapper->mapToParameters(negative);
    
    // Positive valence might suggest major mode
    // Negative valence might suggest minor mode
    EXPECT_FALSE(posParams.modeSuggested.empty());
    EXPECT_FALSE(negParams.modeSuggested.empty());
}

// Test different arousal values
TEST_F(EmotionMapperTest, DifferentArousal) {
    EmotionalState lowArousal;
    lowArousal.valence = 0.0f;
    lowArousal.arousal = 0.2f;
    lowArousal.intensity = 0.5f;
    lowArousal.primaryEmotion = "calm";
    
    EmotionalState highArousal;
    highArousal.valence = 0.0f;
    highArousal.arousal = 0.9f;
    highArousal.intensity = 0.5f;
    highArousal.primaryEmotion = "excited";
    
    MusicalParameters lowParams = mapper->mapToParameters(lowArousal);
    MusicalParameters highParams = mapper->mapToParameters(highArousal);
    
    // High arousal should suggest higher tempo
    EXPECT_LE(lowParams.tempoSuggested, highParams.tempoSuggested);
}

// Test intensity parameter
TEST_F(EmotionMapperTest, IntensityParameter) {
    EmotionalState lowIntensity;
    lowIntensity.valence = 0.5f;
    lowIntensity.arousal = 0.5f;
    lowIntensity.intensity = 0.2f;
    lowIntensity.primaryEmotion = "content";
    
    EmotionalState highIntensity;
    highIntensity.valence = 0.5f;
    highIntensity.arousal = 0.5f;
    highIntensity.intensity = 0.9f;
    highIntensity.primaryEmotion = "ecstatic";
    
    MusicalParameters lowParams = mapper->mapToParameters(lowIntensity);
    MusicalParameters highParams = mapper->mapToParameters(highIntensity);
    
    // High intensity might affect dynamics or tempo
    EXPECT_GT(lowParams.velocityMin, 0);
    EXPECT_GT(highParams.velocityMin, 0);
}
