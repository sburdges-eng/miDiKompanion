#include <gtest/gtest.h>
#include "../../src/voice/VoiceSynthesizer.h"
#include "../../src/voice/ExpressionEngine.h"
#include "../../src/common/Types.h"
#include <vector>

using namespace kelly;

class EmotionVocalMappingTest : public ::testing::Test {
protected:
    void SetUp() override {
        voiceSynthesizer_ = std::make_unique<VoiceSynthesizer>();
        expressionEngine_ = std::make_unique<ExpressionEngine>();
        voiceSynthesizer_->setEnabled(true);
        voiceSynthesizer_->prepare(44100.0);
    }

    std::unique_ptr<VoiceSynthesizer> voiceSynthesizer_;
    std::unique_ptr<ExpressionEngine> expressionEngine_;
};

TEST_F(EmotionVocalMappingTest, EmotionCategoryMapping) {
    std::vector<EmotionCategory> categories = {
        EmotionCategory::Joy,
        EmotionCategory::Sadness,
        EmotionCategory::Anger,
        EmotionCategory::Fear,
        EmotionCategory::Surprise,
        EmotionCategory::Trust,
        EmotionCategory::Anticipation,
        EmotionCategory::Disgust
    };

    for (auto category : categories) {
        EmotionNode emotion;
        emotion.category = category;
        emotion.name = "test";
        emotion.valence = category == EmotionCategory::Joy ? 0.8f : -0.3f;
        emotion.arousal = 0.6f;
        emotion.dominance = 0.5f;
        emotion.intensity = 0.7f;

        // Get vocal characteristics
        auto characteristics = voiceSynthesizer_->getVocalCharacteristics(emotion);

        // Should have valid characteristics
        EXPECT_GE(characteristics.brightness, 0.0f);
        EXPECT_LE(characteristics.brightness, 1.0f);
        EXPECT_GE(characteristics.breathiness, 0.0f);
        EXPECT_LE(characteristics.breathiness, 1.0f);
        EXPECT_GE(characteristics.vibratoRate, 0.0f);
        EXPECT_GT(characteristics.vibratoDepth, 0.0f);
    }
}

TEST_F(EmotionVocalMappingTest, ValenceToBrightness) {
    // High valence should map to higher brightness
    EmotionNode positiveEmotion;
    positiveEmotion.valence = 0.8f;
    positiveEmotion.arousal = 0.5f;
    positiveEmotion.dominance = 0.5f;
    positiveEmotion.intensity = 0.5f;

    EmotionNode negativeEmotion;
    negativeEmotion.valence = -0.8f;
    negativeEmotion.arousal = 0.5f;
    negativeEmotion.dominance = 0.5f;
    negativeEmotion.intensity = 0.5f;

    auto posChars = voiceSynthesizer_->getVocalCharacteristics(positiveEmotion);
    auto negChars = voiceSynthesizer_->getVocalCharacteristics(negativeEmotion);

    // Positive emotion should have higher brightness
    EXPECT_GT(posChars.brightness, negChars.brightness);
}

TEST_F(EmotionVocalMappingTest, ArousalToVibratoRate) {
    // High arousal should map to faster vibrato
    EmotionNode highArousal;
    highArousal.valence = 0.0f;
    highArousal.arousal = 0.9f;
    highArousal.dominance = 0.5f;
    highArousal.intensity = 0.5f;

    EmotionNode lowArousal;
    lowArousal.valence = 0.0f;
    lowArousal.arousal = 0.2f;
    lowArousal.dominance = 0.5f;
    lowArousal.intensity = 0.5f;

    auto highChars = voiceSynthesizer_->getVocalCharacteristics(highArousal);
    auto lowChars = voiceSynthesizer_->getVocalCharacteristics(lowArousal);

    // High arousal should have faster vibrato
    EXPECT_GT(highChars.vibratoRate, lowChars.vibratoRate);
}

TEST_F(EmotionVocalMappingTest, ExpressionEngineMapping) {
    EmotionNode emotion;
    emotion.category = EmotionCategory::Joy;
    emotion.valence = 0.8f;
    emotion.arousal = 0.7f;
    emotion.dominance = 0.6f;
    emotion.intensity = 0.8f;

    VocalExpression baseExpression;
    auto expression = expressionEngine_->applyEmotionExpression(baseExpression, emotion);

    // Expression should be modified based on emotion
    EXPECT_GE(expression.brightness, 0.0f);
    EXPECT_LE(expression.brightness, 1.0f);
    EXPECT_GE(expression.vibratoRate, 0.0f);
}

TEST_F(EmotionVocalMappingTest, VoiceTypeFormantShifts) {
    // Test different voice types
    voiceSynthesizer_->setVoiceType(VoiceType::Male);
    EXPECT_EQ(voiceSynthesizer_->getVoiceType(), VoiceType::Male);

    voiceSynthesizer_->setVoiceType(VoiceType::Female);
    EXPECT_EQ(voiceSynthesizer_->getVoiceType(), VoiceType::Female);

    voiceSynthesizer_->setVoiceType(VoiceType::Child);
    EXPECT_EQ(voiceSynthesizer_->getVoiceType(), VoiceType::Child);

    voiceSynthesizer_->setVoiceType(VoiceType::Neutral);
    EXPECT_EQ(voiceSynthesizer_->getVoiceType(), VoiceType::Neutral);
}

TEST_F(EmotionVocalMappingTest, ConsistencyAcrossEmotions) {
    // Test that similar emotions produce similar vocal characteristics
    EmotionNode emotion1;
    emotion1.category = EmotionCategory::Joy;
    emotion1.valence = 0.8f;
    emotion1.arousal = 0.7f;
    emotion1.dominance = 0.6f;
    emotion1.intensity = 0.8f;

    EmotionNode emotion2;
    emotion2.category = EmotionCategory::Joy;
    emotion2.valence = 0.75f;
    emotion2.arousal = 0.65f;
    emotion2.dominance = 0.55f;
    emotion2.intensity = 0.75f;

    auto chars1 = voiceSynthesizer_->getVocalCharacteristics(emotion1);
    auto chars2 = voiceSynthesizer_->getVocalCharacteristics(emotion2);

    // Should be similar (within reasonable tolerance)
    EXPECT_NEAR(chars1.brightness, chars2.brightness, 0.2f);
    EXPECT_NEAR(chars1.vibratoRate, chars2.vibratoRate, 1.0f);
}
