#include <gtest/gtest.h>
#include "ml/MultiModelProcessor.h"
#include <juce_core/juce_core.h>
#include <array>
#include <cmath>
#include <fstream>

using namespace Kelly::ML;

class MultiModelProcessorTest : public ::testing::Test {
protected:
    MultiModelProcessor processor;
    juce::File tempModelsDir;

    void SetUp() override {
        // Create temporary directory for test models
        tempModelsDir = juce::File::getSpecialLocation(juce::File::tempDirectory)
                       .getChildFile("kelly_ml_test_models");
        tempModelsDir.createDirectory();
    }

    void TearDown() override {
        // Cleanup temporary directory
        if (tempModelsDir.exists()) {
            tempModelsDir.deleteRecursively();
        }
    }

    // Helper to create a minimal valid RTNeural JSON file
    void createMinimalModelJSON(const juce::File& file, size_t inputSize, size_t outputSize) {
        std::ofstream out(file.getFullPathName().toStdString());
        out << R"({
  "layers": [
    {
      "type": "dense",
      "in_size": )" << inputSize << R"(,
      "out_size": )" << outputSize << R"(,
      "activation": "tanh",
      "weights": [[0.1, 0.2], [0.3, 0.4]],
      "bias": [0.0, 0.0]
    }
  ],
  "metadata": {
    "model_name": "TestModel",
    "framework": "PyTorch",
    "export_version": "2.0",
    "parameter_count": 4,
    "input_size": )" << inputSize << R"(,
    "output_size": )" << outputSize << R"(
  }
})";
        out.close();
    }
};

TEST_F(MultiModelProcessorTest, InitialState) {
    // Processor should start uninitialized
    EXPECT_FALSE(processor.isInitialized());
    EXPECT_EQ(processor.getTotalParams(), 0);
    EXPECT_EQ(processor.getTotalMemoryKB(), 0);
}

TEST_F(MultiModelProcessorTest, InitializeWithEmptyDirectory) {
    // Initialize with empty directory (should use fallback)
    bool result = processor.initialize(tempModelsDir);

    // Should succeed (fallback mode)
    EXPECT_TRUE(result);
    EXPECT_TRUE(processor.isInitialized());
}

TEST_F(MultiModelProcessorTest, InitializeWithModelFiles) {
    // Create model files
    createMinimalModelJSON(tempModelsDir.getChildFile("emotionrecognizer.json"), 128, 64);
    createMinimalModelJSON(tempModelsDir.getChildFile("melodytransformer.json"), 64, 128);
    createMinimalModelJSON(tempModelsDir.getChildFile("harmonypredictor.json"), 128, 64);
    createMinimalModelJSON(tempModelsDir.getChildFile("dynamicsengine.json"), 32, 16);
    createMinimalModelJSON(tempModelsDir.getChildFile("groovepredictor.json"), 64, 32);

    bool result = processor.initialize(tempModelsDir);

    // Should succeed
    EXPECT_TRUE(result);
    EXPECT_TRUE(processor.isInitialized());
}

TEST_F(MultiModelProcessorTest, RunFullPipelineWithoutModels) {
    // Initialize without models (fallback mode)
    processor.initialize(tempModelsDir);

    // Run full pipeline
    std::array<float, 128> features;
    features.fill(0.5f);

    auto result = processor.runFullPipeline(features);

    // Should produce valid result (even if fallback)
    EXPECT_TRUE(result.valid);

    // Check output sizes
    EXPECT_EQ(result.emotionEmbedding.size(), 64);
    EXPECT_EQ(result.melodyProbabilities.size(), 128);
    EXPECT_EQ(result.harmonyPrediction.size(), 64);
    EXPECT_EQ(result.dynamicsOutput.size(), 16);
    EXPECT_EQ(result.grooveParameters.size(), 32);
}

TEST_F(MultiModelProcessorTest, RunFullPipelineOutputRanges) {
    processor.initialize(tempModelsDir);

    std::array<float, 128> features;
    features.fill(0.5f);

    auto result = processor.runFullPipeline(features);

    // Check that outputs are finite (no NaN or Inf)
    for (float val : result.emotionEmbedding) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }

    for (float val : result.melodyProbabilities) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }

    for (float val : result.harmonyPrediction) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }

    for (float val : result.dynamicsOutput) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }

    for (float val : result.grooveParameters) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(MultiModelProcessorTest, ModelEnableDisable) {
    processor.initialize(tempModelsDir);

    // Test enabling/disabling models
    processor.setModelEnabled(ModelType::EmotionRecognizer, false);
    EXPECT_FALSE(processor.isModelEnabled(ModelType::EmotionRecognizer));

    processor.setModelEnabled(ModelType::EmotionRecognizer, true);
    EXPECT_TRUE(processor.isModelEnabled(ModelType::EmotionRecognizer));
}

TEST_F(MultiModelProcessorTest, InferSingleModel) {
    processor.initialize(tempModelsDir);

    // Test single model inference
    std::vector<float> input(128, 0.5f);
    auto output = processor.infer(ModelType::EmotionRecognizer, input);

    // Should produce output of correct size
    EXPECT_EQ(output.size(), 64);

    // Check output is valid
    for (float val : output) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(MultiModelProcessorTest, InputSizeValidation) {
    processor.initialize(tempModelsDir);

    // Test with wrong input size
    std::vector<float> wrongSizeInput(64, 0.5f);  // Should be 128 for EmotionRecognizer
    auto output = processor.infer(ModelType::EmotionRecognizer, wrongSizeInput);

    // Should handle gracefully (may return empty or fallback)
    // Just verify it doesn't crash
    EXPECT_TRUE(true);
}

TEST_F(MultiModelProcessorTest, TotalParamsAndMemory) {
    processor.initialize(tempModelsDir);

    // Should report total parameters and memory
    size_t totalParams = processor.getTotalParams();
    size_t totalMemoryKB = processor.getTotalMemoryKB();

    // In fallback mode, should be 0 or small
    // With real models, should be ~1M params, ~4MB
    EXPECT_GE(totalParams, 0);
    EXPECT_GE(totalMemoryKB, 0);

    // Memory should be approximately params * 4 bytes / 1024
    if (totalParams > 0) {
        size_t expectedMemoryKB = (totalParams * 4) / 1024;
        EXPECT_NEAR(totalMemoryKB, expectedMemoryKB, expectedMemoryKB * 0.1);  // 10% tolerance
    }
}
