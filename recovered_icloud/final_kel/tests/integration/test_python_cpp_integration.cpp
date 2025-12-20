/**
 * Integration Tests - Python-C++ Communication
 *
 * Tests the complete integration between C++ engines and Python intelligence modules.
 */

#include <gtest/gtest.h>
#include "bridge/EngineIntelligenceBridge.h"
#include "bridge/OrchestratorBridge.h"
#include "bridge/IntentBridge.h"
#include "bridge/ContextBridge.h"
#include "bridge/StateBridge.h"
#include "bridge/SuggestionBridge.h"
#include "engine/IntentPipeline.h"
#include "midi/MidiGenerator.h"
#include "common/KellyTypes.h"
#include <string>
#include <thread>
#include <chrono>

using namespace kelly;

class PythonBridgeIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize bridges
        engineBridge_ = std::make_unique<EngineIntelligenceBridge>();
        orchestratorBridge_ = std::make_unique<OrchestratorBridge>();
        intentBridge_ = std::make_unique<IntentBridge>();
        contextBridge_ = std::make_unique<ContextBridge>();
        stateBridge_ = std::make_unique<StateBridge>();
        suggestionBridge_ = std::make_unique<SuggestionBridge>();

        if (stateBridge_) {
            stateBridge_->initialize();
        }
    }

    void TearDown() override {
        if (stateBridge_) {
            stateBridge_->shutdown();
        }
    }

    std::unique_ptr<EngineIntelligenceBridge> engineBridge_;
    std::unique_ptr<OrchestratorBridge> orchestratorBridge_;
    std::unique_ptr<IntentBridge> intentBridge_;
    std::unique_ptr<ContextBridge> contextBridge_;
    std::unique_ptr<StateBridge> stateBridge_;
    std::unique_ptr<SuggestionBridge> suggestionBridge_;
};

// Test EngineIntelligenceBridge
TEST_F(PythonBridgeIntegrationTest, EngineIntelligenceBridgeBasic) {
    if (!engineBridge_->isAvailable()) {
        GTEST_SKIP() << "Python not available, skipping test";
    }

    std::string stateJson = R"({"emotion": "grief", "key": "C", "mode": "minor"})";
    std::string suggestions = engineBridge_->getEngineSuggestions("melody", stateJson);

    // Should return JSON (even if empty)
    EXPECT_FALSE(suggestions.empty());
}

// Test ContextBridge
TEST_F(PythonBridgeIntegrationTest, ContextBridgeBasic) {
    if (!contextBridge_->isAvailable()) {
        GTEST_SKIP() << "Python not available, skipping test";
    }

    std::string stateJson = R"({"emotion": "grief", "parameters": {"valence": -0.5, "arousal": 0.4}})";
    std::string context = contextBridge_->analyzeContext(stateJson);

    // Should return JSON context analysis
    EXPECT_FALSE(context.empty());
}

// Test IntentBridge
TEST_F(PythonBridgeIntegrationTest, IntentBridgeBasic) {
    if (!intentBridge_->isAvailable()) {
        GTEST_SKIP() << "Python not available, skipping test";
    }

    std::string intentJson = R"({"phase_1": {"mood_primary": "grief"}, "phase_2": {"technical_key": "C"}})";
    std::string result = intentBridge_->processIntent(intentJson);

    // Should return processed intent result
    EXPECT_FALSE(result.empty());
}

// Test StateBridge
TEST_F(PythonBridgeIntegrationTest, StateBridgeBasic) {
    if (!stateBridge_->isAvailable()) {
        GTEST_SKIP() << "Python not available, skipping test";
    }

    // Emit state update
    std::string stateJson = R"({"chords": ["Am", "Dm"], "notes": []})";
    stateBridge_->emitStateUpdate("melody", stateJson);

    // Flush to process
    stateBridge_->flush();

    // Get current state
    std::string currentState = stateBridge_->getCurrentState();
    EXPECT_FALSE(currentState.empty());
}

// Test SuggestionBridge
TEST_F(PythonBridgeIntegrationTest, SuggestionBridgeBasic) {
    if (!suggestionBridge_->isAvailable()) {
        GTEST_SKIP() << "Python not available, skipping test";
    }

    std::string stateJson = R"({"emotion": "grief", "parameters": {"valence": -0.5}})";
    std::string suggestions = suggestionBridge_->getSuggestions(stateJson, 5);

    // Should return JSON suggestions array
    EXPECT_FALSE(suggestions.empty());
}

// Test IntentPipeline with Python integration
TEST_F(PythonBridgeIntegrationTest, IntentPipelinePythonIntegration) {
    IntentPipeline pipeline;

    // Create complex wound (should trigger Python processing)
    Wound wound;
    wound.description = "I feel lost and alone, searching for meaning in a world that seems empty and cold";
    wound.intensity = 0.9f;
    wound.urgency = 0.9f;
    wound.source = "user_input";
    wound.desire = "To find connection and purpose";
    wound.expression = "A melancholic journey through uncertainty";

    // Process intent (may use Python if available)
    IntentResult result = pipeline.process(wound);

    // Should have valid result
    EXPECT_FALSE(result.mode.empty());
    EXPECT_GT(result.tempoBpm, 0);
}

// Test MidiGenerator with Python integration
TEST_F(PythonBridgeIntegrationTest, MidiGeneratorPythonIntegration) {
    MidiGenerator generator;

    // Create intent
    Wound wound;
    wound.description = "Feeling hopeful";
    wound.intensity = 0.7f;
    wound.source = "test";

    IntentPipeline pipeline;
    IntentResult intent = pipeline.process(wound);

    // Generate MIDI (should use Python bridges if available)
    GeneratedMidi midi = generator.generate(intent, 4, 0.5f, 0.4f, 0.0f, 0.75f);

    // Should have valid MIDI
    EXPECT_GT(midi.bpm, 0);
    EXPECT_GT(midi.bars, 0);
}

// Test cache effectiveness
TEST_F(PythonBridgeIntegrationTest, BridgeCaching) {
    if (!engineBridge_->isAvailable()) {
        GTEST_SKIP() << "Python not available, skipping test";
    }

    std::string stateJson = R"({"emotion": "grief", "key": "C"})";

    // First call (should hit Python)
    auto start1 = std::chrono::high_resolution_clock::now();
    std::string result1 = engineBridge_->getEngineSuggestions("melody", stateJson);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    // Second call (should hit cache)
    auto start2 = std::chrono::high_resolution_clock::now();
    std::string result2 = engineBridge_->getEngineSuggestions("melody", stateJson);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

    // Cached call should be faster
    EXPECT_LE(duration2.count(), duration1.count());
    EXPECT_EQ(result1, result2);
}

// Test thread safety (simulated audio thread)
TEST_F(PythonBridgeIntegrationTest, ThreadSafety) {
    if (!stateBridge_->isAvailable()) {
        GTEST_SKIP() << "Python not available, skipping test";
    }

    // Simulate audio thread calls
    std::vector<std::thread> threads;
    const int numThreads = 10;
    const int callsPerThread = 100;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, callsPerThread]() {
            for (int j = 0; j < callsPerThread; ++j) {
                std::string stateJson = R"({"engine": "melody", "note_count": )" +
                                       std::to_string(i * callsPerThread + j) + "}";
                stateBridge_->emitStateUpdate("melody", stateJson);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Flush and verify no crashes
    stateBridge_->flush();

    // Test passed if no crashes occurred
    EXPECT_TRUE(true);
}

// Test end-to-end generation flow
TEST_F(PythonBridgeIntegrationTest, EndToEndGeneration) {
    // Create intent
    Wound wound;
    wound.description = "I feel a deep sense of longing";
    wound.intensity = 0.8f;
    wound.source = "test";

    IntentPipeline pipeline;
    IntentResult intent = pipeline.process(wound);

    // Generate MIDI
    MidiGenerator generator;
    GeneratedMidi midi = generator.generate(intent, 8, 0.6f, 0.5f, 0.0f, 0.8f);

    // Verify result
    EXPECT_GT(midi.bpm, 0);
    EXPECT_GT(midi.bars, 0);
    EXPECT_FALSE(midi.key.empty());
    EXPECT_FALSE(midi.mode.empty());
}
