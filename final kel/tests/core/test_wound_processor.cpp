#include <gtest/gtest.h>
#include "engine/WoundProcessor.h"
#include "engine/EmotionThesaurus.h"
#include <string>
#include <memory>

using namespace kelly;

class WoundProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        thesaurus = std::make_unique<EmotionThesaurus>();
        processor = std::make_unique<WoundProcessor>(*thesaurus);
    }

    std::unique_ptr<EmotionThesaurus> thesaurus;
    std::unique_ptr<WoundProcessor> processor;
};

// Test basic wound processing
TEST_F(WoundProcessorTest, BasicProcessing) {
    Wound wound;
    wound.description = "I feel sad";
    wound.intensity = 0.5f;
    wound.source = "internal";

    EmotionNode result = processor->processWound(wound);

    EXPECT_FALSE(result.name.empty());
    EXPECT_GE(result.valence, -1.0f);
    EXPECT_LE(result.valence, 1.0f);
    EXPECT_GE(result.arousal, 0.0f);
    EXPECT_LE(result.arousal, 1.0f);
    EXPECT_GE(result.intensity, 0.0f);
    EXPECT_LE(result.intensity, 1.0f);
}

// Test different wound descriptions
TEST_F(WoundProcessorTest, DifferentDescriptions) {
    std::vector<std::string> descriptions = {
        "I'm happy",
        "I'm angry",
        "I'm scared",
        "I'm peaceful"
    };

    for (const auto& desc : descriptions) {
        Wound wound;
        wound.description = desc;
        wound.intensity = 0.5f;
        wound.source = "internal";

        EmotionNode result = processor->processWound(wound);
        EXPECT_FALSE(result.name.empty()) << "Failed for: " << desc;
    }
}

// Test intensity mapping
TEST_F(WoundProcessorTest, IntensityMapping) {
    Wound low;
    low.description = "I feel a bit sad";
    low.intensity = 0.2f;
    low.source = "internal";

    Wound high;
    high.description = "I feel extremely sad";
    high.intensity = 0.9f;
    high.source = "internal";

    EmotionNode lowResult = processor->processWound(low);
    EmotionNode highResult = processor->processWound(high);

    // High intensity should map to higher intensity
    EXPECT_LE(lowResult.intensity, highResult.intensity);
}
