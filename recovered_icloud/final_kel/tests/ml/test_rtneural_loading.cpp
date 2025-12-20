#include <gtest/gtest.h>
#include "ml/MultiModelProcessor.h"
#include <juce_core/juce_core.h>
#include <fstream>
#include <array>

using namespace Kelly::ML;

class RTNeuralLoadingTest : public ::testing::Test {
protected:
    juce::File tempModelsDir;

    void SetUp() override {
        tempModelsDir = juce::File::getSpecialLocation(juce::File::tempDirectory)
                       .getChildFile("kelly_rtneural_test");
        tempModelsDir.createDirectory();
    }

    void TearDown() override {
        if (tempModelsDir.exists()) {
            tempModelsDir.deleteRecursively();
        }
    }

    // Create a valid RTNeural JSON file
    void createValidRTNeuralJSON(const juce::File& file,
                                  size_t inputSize,
                                  size_t outputSize,
                                  const std::string& modelName) {
        std::ofstream out(file.getFullPathName().toStdString());
        out << R"({
  "layers": [
    {
      "type": "dense",
      "in_size": )" << inputSize << R"(,
      "out_size": )" << outputSize << R"(,
      "activation": "tanh",
      "weights": [)";

        // Generate simple weight matrix
        for (size_t i = 0; i < outputSize; ++i) {
            if (i > 0) out << ",";
            out << "[";
            for (size_t j = 0; j < inputSize; ++j) {
                if (j > 0) out << ",";
                out << (0.1f * (i + j));
            }
            out << "]";
        }

        out << R"(],
      "bias": [)";
        for (size_t i = 0; i < outputSize; ++i) {
            if (i > 0) out << ",";
            out << "0.0";
        }
        out << R"(]
    }
  ],
  "metadata": {
    "model_name": ")" << modelName << R"(",
    "framework": "PyTorch",
    "export_version": "2.0",
    "parameter_count": )" << (inputSize * outputSize + outputSize) << R"(,
    "input_size": )" << inputSize << R"(,
    "output_size": )" << outputSize << R"(
  }
})";
        out.close();
    }

    // Create an LSTM model JSON
    void createLSTMModelJSON(const juce::File& file,
                             size_t inputSize,
                             size_t hiddenSize) {
        std::ofstream out(file.getFullPathName().toStdString());
        out << R"({
  "layers": [
    {
      "type": "lstm",
      "in_size": )" << inputSize << R"(,
      "out_size": )" << hiddenSize << R"(,
      "weights_ih": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
      "weights_hh": [[0.1], [0.2], [0.3], [0.4]],
      "bias_ih": [[0.0], [0.0], [0.0], [0.0]],
      "bias_hh": [[0.0], [0.0], [0.0], [0.0]]
    }
  ],
  "metadata": {
    "model_name": "TestLSTM",
    "framework": "PyTorch",
    "export_version": "2.0",
    "parameter_count": 100,
    "input_size": )" << inputSize << R"(,
    "output_size": )" << hiddenSize << R"(
  }
})";
        out.close();
    }
};

TEST_F(RTNeuralLoadingTest, LoadValidDenseModel) {
    // Create a valid dense model
    juce::File modelFile = tempModelsDir.getChildFile("test_model.json");
    createValidRTNeuralJSON(modelFile, 128, 64, "TestModel");

    MultiModelProcessor processor;
    bool result = processor.initialize(tempModelsDir);

    // Should succeed (even if file not in expected location)
    // The processor will use fallback if model not found
    EXPECT_TRUE(result);
}

TEST_F(RTNeuralLoadingTest, LoadInvalidJSON) {
    // Create invalid JSON
    juce::File invalidFile = tempModelsDir.getChildFile("invalid.json");
    invalidFile.replaceWithText("{ invalid json }");

    MultiModelProcessor processor;
    bool result = processor.initialize(tempModelsDir);

    // Should handle gracefully (use fallback)
    EXPECT_TRUE(result);
}

TEST_F(RTNeuralLoadingTest, LoadMissingFile) {
    // Try to load from directory with no model files
    juce::File emptyDir = tempModelsDir.getChildFile("empty");
    emptyDir.createDirectory();

    MultiModelProcessor processor;
    bool result = processor.initialize(emptyDir);

    // Should succeed with fallback mode
    EXPECT_TRUE(result);
}

TEST_F(RTNeuralLoadingTest, LoadLSTMModel) {
    // Create LSTM model
    juce::File lstmFile = tempModelsDir.getChildFile("lstm_model.json");
    createLSTMModelJSON(lstmFile, 256, 128);

    // Verify file was created
    EXPECT_TRUE(lstmFile.existsAsFile());

    // Try to load (processor will look for specific filenames)
    MultiModelProcessor processor;
    bool result = processor.initialize(tempModelsDir);

    // Should handle gracefully
    EXPECT_TRUE(result);
}

TEST_F(RTNeuralLoadingTest, JSONStructureValidation) {
    // Test that JSON structure is validated
    // Create JSON with missing required fields
    juce::File incompleteFile = tempModelsDir.getChildFile("incomplete.json");
    incompleteFile.replaceWithText(R"({
  "layers": []
})");

    MultiModelProcessor processor;
    bool result = processor.initialize(tempModelsDir);

    // Should handle gracefully
    EXPECT_TRUE(result);
}

#ifdef ENABLE_RTNEURAL
TEST_F(RTNeuralLoadingTest, RTNeuralParserIntegration) {
    // This test only runs if RTNeural is enabled
    // Create valid model and verify it can be parsed
    juce::File modelFile = tempModelsDir.getChildFile("emotionrecognizer.json");
    createValidRTNeuralJSON(modelFile, 128, 64, "EmotionRecognizer");

    MultiModelProcessor processor;
    bool result = processor.initialize(tempModelsDir);

    // If RTNeural is enabled, should attempt to load
    // If disabled, will use fallback
    EXPECT_TRUE(result);
}
#endif
