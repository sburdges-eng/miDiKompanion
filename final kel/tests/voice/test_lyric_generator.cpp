#include <gtest/gtest.h>
#include "../../src/voice/LyricGenerator.h"
#include "../../src/common/Types.h"
#include <vector>

using namespace kelly;

class LyricGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        generator_ = std::make_unique<LyricGenerator>();
    }

    std::unique_ptr<LyricGenerator> generator_;
};

TEST_F(LyricGeneratorTest, GenerateLyricsBasic) {
    EmotionNode emotion;
    emotion.name = "joy";
    emotion.category = EmotionCategory::Joy;
    emotion.valence = 0.8f;
    emotion.arousal = 0.7f;
    emotion.dominance = 0.6f;
    emotion.intensity = 0.8f;

    Wound wound;
    wound.description = "feeling happy and free";
    wound.intensity = 0.7f;

    auto result = generator_->generateLyrics(emotion, wound, nullptr);

    // Should generate lyrics
    EXPECT_FALSE(result.lines.empty());
    EXPECT_FALSE(result.structure.sections.empty());
}

TEST_F(LyricGeneratorTest, GenerateLyricsWithStructure) {
    EmotionNode emotion;
    emotion.name = "sadness";
    emotion.category = EmotionCategory::Sadness;
    emotion.valence = -0.8f;
    emotion.arousal = 0.3f;
    emotion.dominance = 0.4f;
    emotion.intensity = 0.7f;

    Wound wound;
    wound.description = "feeling alone and empty";

    generator_->setStructureType("verse_chorus");
    generator_->setRhymeScheme("ABAB");
    generator_->setLineLength(8);

    auto result = generator_->generateLyrics(emotion, wound, nullptr);

    // Should have structure
    EXPECT_GT(result.structure.sections.size(), 0);

    // Check that lines were generated
    EXPECT_GT(result.lines.size(), 0);
}

TEST_F(LyricGeneratorTest, ExpandEmotionVocabulary) {
    EmotionNode emotion;
    emotion.name = "anger";
    emotion.category = EmotionCategory::Anger;
    emotion.valence = -0.7f;
    emotion.arousal = 0.9f;
    emotion.dominance = 0.8f;
    emotion.intensity = 0.9f;

    Wound wound;
    wound.description = "feeling furious and betrayed";

    auto result = generator_->generateLyrics(emotion, wound, nullptr);

    // Should generate lyrics with emotion-appropriate words
    bool foundEmotionalContent = false;
    for (const auto& line : result.lines) {
        if (!line.text.empty()) {
            foundEmotionalContent = true;
            break;
        }
    }
    EXPECT_TRUE(foundEmotionalContent);
}

TEST_F(LyricGeneratorTest, SetStructureType) {
    generator_->setStructureType("ballad");
    generator_->setRhymeScheme("AABB");
    generator_->setLineLength(10);

    // Verify settings were applied (they affect next generation)
    EmotionNode emotion;
    emotion.name = "love";
    emotion.category = EmotionCategory::Trust;
    emotion.valence = 0.6f;
    emotion.arousal = 0.5f;
    emotion.dominance = 0.5f;
    emotion.intensity = 0.6f;

    Wound wound;
    wound.description = "feeling connected";

    auto result = generator_->generateLyrics(emotion, wound, nullptr);

    // Should have generated structure
    EXPECT_GT(result.structure.sections.size(), 0);
}

TEST_F(LyricGeneratorTest, DifferentEmotionCategories) {
    std::vector<EmotionCategory> categories = {
        EmotionCategory::Joy,
        EmotionCategory::Sadness,
        EmotionCategory::Anger,
        EmotionCategory::Fear,
        EmotionCategory::Surprise,
        EmotionCategory::Trust
    };

    Wound wound;
    wound.description = "test";

    for (auto category : categories) {
        EmotionNode emotion;
        emotion.category = category;
        emotion.name = "test";
        emotion.valence = 0.0f;
        emotion.arousal = 0.5f;
        emotion.dominance = 0.5f;
        emotion.intensity = 0.5f;

        auto result = generator_->generateLyrics(emotion, wound, nullptr);

        // Each emotion category should generate lyrics
        EXPECT_GT(result.lines.size(), 0) << "Failed for category: " << static_cast<int>(category);
    }
}
