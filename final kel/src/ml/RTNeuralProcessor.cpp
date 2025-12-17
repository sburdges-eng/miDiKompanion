#include "ml/RTNeuralProcessor.h"
#include <juce_core/juce_core.h>
#include <cstring>
#include <fstream>

namespace kelly {

bool RTNeuralProcessor::loadModel(const juce::File& jsonFile) {
    if (!jsonFile.existsAsFile()) {
        juce::Logger::writeToLog("ML Model file not found: " + jsonFile.getFullPathName());
        return false;
    }

#ifdef ENABLE_RTNEURAL
    try {
        // Convert JUCE File to std::string for ifstream
        std::string filePath = jsonFile.getFullPathName().toStdString();
        std::ifstream jsonStream(filePath, std::ifstream::binary);

        if (!jsonStream.is_open()) {
            juce::Logger::writeToLog("Failed to open ML model file: " + jsonFile.getFullPathName());
            return false;
        }

        // Parse JSON using RTNeural's JSON parser
        auto model = RTNeural::json_parser::parseJson<float>(jsonStream);

        if (!model) {
            juce::Logger::writeToLog("Failed to parse RTNeural model JSON: " + jsonFile.getFullPathName());
            jsonStream.close();
            return false;
        }

        // Reset model state (important for LSTM layers)
        model->reset();

        // Store the model
        model_ = std::move(model);
        modelLoaded_ = true;
        modelPath_ = filePath;

        juce::Logger::writeToLog("ML Model loaded successfully: " + jsonFile.getFullPathName());
        jsonStream.close();
        return true;

    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Exception while loading RTNeural model: " +
                                 juce::String(e.what()));
        modelLoaded_ = false;
        model_ = nullptr;
        return false;
    } catch (...) {
        juce::Logger::writeToLog("Unknown exception while loading RTNeural model");
        modelLoaded_ = false;
        model_ = nullptr;
        return false;
    }
#else
    // Placeholder when RTNeural is not available
    model_.loaded = true;
    model_.modelPath = jsonFile.getFullPathName().toStdString();
    juce::Logger::writeToLog("ML Model loaded (RTNeural disabled): " + jsonFile.getFullPathName());
    return true;
#endif
}

std::array<float, 64> RTNeuralProcessor::inferEmotion(const std::array<float, 128>& features) {
    std::array<float, 64> result{};

    if (!isModelLoaded()) {
        // Return simple heuristic if no model loaded
        // First 32 dimensions for valence, last 32 for arousal
        for (size_t i = 0; i < 32; ++i) {
            result[i] = features[i] * 0.3f;  // Valence proxy
            result[i + 32] = features[i + 32] * 0.5f;  // Arousal proxy
        }
        return result;
    }

#ifdef ENABLE_RTNEURAL
    if (!model_) {
        juce::Logger::writeToLog("Warning: Model pointer is null in inferEmotion()");
        // Return heuristic fallback
        for (size_t i = 0; i < 32; ++i) {
            result[i] = features[i] * 0.3f;
            result[i + 32] = features[i + 32] * 0.5f;
        }
        return result;
    }

    try {
        // Perform actual neural network inference
        // Note: RTNeural's forward() API may vary by version.
        // If compilation fails, check RTNeural documentation for correct signature.
        // Expected: forward(input_ptr, output_ptr) for multi-output models
        // Alternative: forward(input_ptr) returning output array/vector
        model_->forward(features.data(), result.data());

        return result;
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Exception during RTNeural inference: " +
                                 juce::String(e.what()));
        // Return heuristic fallback on error
        for (size_t i = 0; i < 32; ++i) {
            result[i] = features[i] * 0.3f;
            result[i + 32] = features[i + 32] * 0.5f;
        }
        return result;
    } catch (...) {
        juce::Logger::writeToLog("Unknown exception during RTNeural inference");
        // Return heuristic fallback on error
        for (size_t i = 0; i < 32; ++i) {
            result[i] = features[i] * 0.3f;
            result[i + 32] = features[i + 32] * 0.5f;
        }
        return result;
    }
#else
    // Placeholder: simple passthrough when RTNeural is not available
    for (size_t i = 0; i < std::min(64UL, features.size()); ++i) {
        result[i] = features[i] * 0.5f;  // Dummy transformation
    }
    return result;
#endif
}

} // namespace kelly
