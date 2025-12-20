#include <gtest/gtest.h>
#include "common/EQPresetManager.h"
#include <string>

using namespace kelly;

class EQPresetManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager = std::make_unique<EQPresetManager>();
    }
    
    std::unique_ptr<EQPresetManager> manager;
};

// Test preset loading
TEST_F(EQPresetManagerTest, PresetLoading) {
    std::vector<std::string> presets = manager->getPresetNames();
    EXPECT_GT(presets.size(), 0);
}

// Test preset retrieval
TEST_F(EQPresetManagerTest, PresetRetrieval) {
    std::vector<std::string> presets = manager->getPresetNames();
    
    if (!presets.empty()) {
        EQPreset preset = manager->getPreset(presets[0]);
        EXPECT_FALSE(preset.name.empty());
    }
}

// Test preset by emotion
TEST_F(EQPresetManagerTest, PresetByEmotion) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        EQPreset preset = manager->getPresetForEmotion(emotion);
        EXPECT_FALSE(preset.name.empty());
    }
}
