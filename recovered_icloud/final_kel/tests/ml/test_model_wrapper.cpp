#include <gtest/gtest.h>
#include "ml/MultiModelProcessor.h"
#include <juce_core/juce_core.h>
#include <vector>
#include <cmath>

using namespace Kelly::ML;

class ModelWrapperTest : public ::testing::Test {
protected:
    void SetUp() override {
        // ModelWrapper is tested through MultiModelProcessor
        // Individual ModelWrapper instances are private
    }
};

TEST_F(ModelWrapperTest, ModelSpecDefinitions) {
    // Test that MODEL_SPECS are correctly defined
    // This is compile-time, but we can verify at runtime

    EXPECT_EQ(MODEL_SPECS[0].inputSize, 128);   // EmotionRecognizer
    EXPECT_EQ(MODEL_SPECS[0].outputSize, 64);
    EXPECT_EQ(MODEL_SPECS[0].estimatedParams, 497664);

    EXPECT_EQ(MODEL_SPECS[1].inputSize, 64);    // MelodyTransformer
    EXPECT_EQ(MODEL_SPECS[1].outputSize, 128);
    EXPECT_EQ(MODEL_SPECS[1].estimatedParams, 412672);

    EXPECT_EQ(MODEL_SPECS[2].inputSize, 128);   // HarmonyPredictor
    EXPECT_EQ(MODEL_SPECS[2].outputSize, 64);
    EXPECT_EQ(MODEL_SPECS[2].estimatedParams, 74048);

    EXPECT_EQ(MODEL_SPECS[3].inputSize, 32);    // DynamicsEngine
    EXPECT_EQ(MODEL_SPECS[3].outputSize, 16);
    EXPECT_EQ(MODEL_SPECS[3].estimatedParams, 13456);

    EXPECT_EQ(MODEL_SPECS[4].inputSize, 64);    // GroovePredictor
    EXPECT_EQ(MODEL_SPECS[4].outputSize, 32);
    EXPECT_EQ(MODEL_SPECS[4].estimatedParams, 19040);
}

TEST_F(ModelWrapperTest, ModelTypeEnum) {
    // Test ModelType enum values
    EXPECT_EQ(static_cast<size_t>(ModelType::EmotionRecognizer), 0);
    EXPECT_EQ(static_cast<size_t>(ModelType::MelodyTransformer), 1);
    EXPECT_EQ(static_cast<size_t>(ModelType::HarmonyPredictor), 2);
    EXPECT_EQ(static_cast<size_t>(ModelType::DynamicsEngine), 3);
    EXPECT_EQ(static_cast<size_t>(ModelType::GroovePredictor), 4);
    EXPECT_EQ(static_cast<size_t>(ModelType::COUNT), 5);
}

TEST_F(ModelWrapperTest, InferenceResultStructure) {
    // Test InferenceResult structure
    InferenceResult result;

    // Check sizes
    EXPECT_EQ(result.emotionEmbedding.size(), 64);
    EXPECT_EQ(result.melodyProbabilities.size(), 128);
    EXPECT_EQ(result.harmonyPrediction.size(), 64);
    EXPECT_EQ(result.dynamicsOutput.size(), 16);
    EXPECT_EQ(result.grooveParameters.size(), 32);

    // Check initial state
    EXPECT_FALSE(result.valid);

    // Initialize with zeros
    result.emotionEmbedding.fill(0.0f);
    result.melodyProbabilities.fill(0.0f);
    result.harmonyPrediction.fill(0.0f);
    result.dynamicsOutput.fill(0.0f);
    result.grooveParameters.fill(0.0f);
    result.valid = true;

    EXPECT_TRUE(result.valid);
}
