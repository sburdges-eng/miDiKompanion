#include <gtest/gtest.h>
#include "engine/EmotionThesaurus.h"
#include <string>
#include <vector>
#include <set>
#include <cmath>

using namespace kelly;

class EmotionIdMatchingTest : public ::testing::Test {
protected:
    void SetUp() override {
        thesaurus = std::make_unique<EmotionThesaurus>();
    }

    std::unique_ptr<EmotionThesaurus> thesaurus;
};

// Test findById with valid IDs (1-216)
TEST_F(EmotionIdMatchingTest, FindById_ValidIds) {
    size_t totalSize = thesaurus->size();
    EXPECT_GE(totalSize, 200);  // Should have at least 200 emotions

    // Test range of valid IDs
    for (int id = 1; id <= static_cast<int>(totalSize) && id <= 216; ++id) {
        auto emotion = thesaurus->findById(id);
        EXPECT_TRUE(emotion.has_value()) << "ID " << id << " should be valid";
        if (emotion.has_value()) {
            EXPECT_EQ(emotion->id, id) << "Returned emotion ID should match requested ID";
            EXPECT_FALSE(emotion->name.empty()) << "Emotion name should not be empty";

            // Verify VAD coordinates are in valid ranges
            EXPECT_GE(emotion->valence, -1.0f);
            EXPECT_LE(emotion->valence, 1.0f);
            EXPECT_GE(emotion->arousal, 0.0f);
            EXPECT_LE(emotion->arousal, 1.0f);
            EXPECT_GE(emotion->intensity, 0.0f);
            EXPECT_LE(emotion->intensity, 1.0f);
        }
    }
}

// Test findById with invalid IDs (0, 217, negative)
TEST_F(EmotionIdMatchingTest, FindById_InvalidIds) {
    // Test with ID 0
    auto emotion0 = thesaurus->findById(0);
    EXPECT_FALSE(emotion0.has_value()) << "ID 0 should be invalid";

    // Test with negative ID
    auto emotionNeg = thesaurus->findById(-1);
    EXPECT_FALSE(emotionNeg.has_value()) << "Negative ID should be invalid";
    auto emotionNeg2 = thesaurus->findById(-100);
    EXPECT_FALSE(emotionNeg2.has_value()) << "Large negative ID should be invalid";

    // Test with ID beyond range (217, 300, etc.)
    size_t totalSize = thesaurus->size();
    auto emotion217 = thesaurus->findById(217);
    if (totalSize < 217) {
        EXPECT_FALSE(emotion217.has_value()) << "ID 217 should be invalid if thesaurus has < 217 emotions";
    }

    auto emotion300 = thesaurus->findById(300);
    EXPECT_FALSE(emotion300.has_value()) << "ID 300 should be invalid";

    auto emotion999 = thesaurus->findById(999);
    EXPECT_FALSE(emotion999.has_value()) << "Very large ID should be invalid";
}

// Test findNearest with various VAD coordinates
TEST_F(EmotionIdMatchingTest, FindNearest_VariousVADCoordinates) {
    // Test with positive valence, high arousal (joy-like)
    EmotionNode joy = thesaurus->findNearest(0.8f, 0.8f, 0.7f);
    EXPECT_GT(joy.id, 0) << "Should find an emotion for positive valence, high arousal";
    EXPECT_GE(joy.valence, 0.0f);
    EXPECT_GE(joy.arousal, 0.5f);

    // Test with negative valence, low arousal (sad-like)
    EmotionNode sad = thesaurus->findNearest(-0.7f, 0.2f, 0.5f);
    EXPECT_GT(sad.id, 0) << "Should find an emotion for negative valence, low arousal";
    EXPECT_LE(sad.valence, 0.0f);
    EXPECT_LE(sad.arousal, 0.5f);

    // Test with neutral coordinates
    EmotionNode neutral = thesaurus->findNearest(0.0f, 0.5f, 0.5f);
    EXPECT_GT(neutral.id, 0) << "Should find an emotion for neutral coordinates";

    // Test with extreme coordinates
    EmotionNode extreme = thesaurus->findNearest(1.0f, 1.0f, 1.0f);
    EXPECT_GT(extreme.id, 0) << "Should find an emotion for extreme coordinates";
    EXPECT_GE(extreme.valence, 0.0f);
    EXPECT_GE(extreme.arousal, 0.5f);

    // Test with negative extreme
    EmotionNode extremeNeg = thesaurus->findNearest(-1.0f, 0.0f, 0.0f);
    EXPECT_GT(extremeNeg.id, 0) << "Should find an emotion for negative extreme coordinates";
    EXPECT_LE(extremeNeg.valence, 0.0f);
}

// Test emotion name consistency
TEST_F(EmotionIdMatchingTest, EmotionNameConsistency) {
    size_t totalSize = thesaurus->size();

    // Get all emotions
    auto all = thesaurus->all();
    EXPECT_EQ(all.size(), totalSize);

    // Verify that findById returns the same emotion as from all()
    for (const auto& [id, expectedEmotion] : all) {
        auto foundEmotion = thesaurus->findById(id);
        EXPECT_TRUE(foundEmotion.has_value()) << "ID " << id << " should be found";
        if (foundEmotion.has_value()) {
            EXPECT_EQ(foundEmotion->id, expectedEmotion.id);
            EXPECT_EQ(foundEmotion->name, expectedEmotion.name)
                << "Emotion name should be consistent between findById and all()";
            EXPECT_FLOAT_EQ(foundEmotion->valence, expectedEmotion.valence);
            EXPECT_FLOAT_EQ(foundEmotion->arousal, expectedEmotion.arousal);
            EXPECT_FLOAT_EQ(foundEmotion->intensity, expectedEmotion.intensity);
        }
    }
}

// Test emotion coordinate mapping
TEST_F(EmotionIdMatchingTest, EmotionCoordinateMapping) {
    // Test that findNearest returns emotions with coordinates close to requested
    std::vector<std::tuple<float, float, float>> testCoords = {
        {0.8f, 0.8f, 0.7f},   // High positive
        {-0.7f, 0.2f, 0.5f},  // Low negative
        {0.0f, 0.5f, 0.5f},   // Neutral
        {0.5f, 0.3f, 0.6f},   // Moderate positive, low arousal
        {-0.3f, 0.9f, 0.8f},  // Negative valence, high arousal
    };

    for (const auto& [v, a, i] : testCoords) {
        EmotionNode found = thesaurus->findNearest(v, a, i);
        EXPECT_GT(found.id, 0) << "Should find emotion for coordinates ("
                               << v << ", " << a << ", " << i << ")";

        // Verify found emotion has valid coordinates
        EXPECT_GE(found.valence, -1.0f);
        EXPECT_LE(found.valence, 1.0f);
        EXPECT_GE(found.arousal, 0.0f);
        EXPECT_LE(found.arousal, 1.0f);
        EXPECT_GE(found.intensity, 0.0f);
        EXPECT_LE(found.intensity, 1.0f);
    }
}

// Test that all emotions have unique IDs
TEST_F(EmotionIdMatchingTest, UniqueEmotionIds) {
    auto all = thesaurus->all();
    std::set<int> ids;

    for (const auto& [id, emotion] : all) {
        EXPECT_EQ(ids.find(id), ids.end()) << "ID " << id << " should be unique";
        ids.insert(id);
    }

    EXPECT_EQ(ids.size(), all.size()) << "All emotions should have unique IDs";
}

// Test findNearest returns closest emotion by distance
TEST_F(EmotionIdMatchingTest, FindNearest_ClosestEmotion) {
    // Find emotion for specific coordinates
    float targetValence = 0.7f;
    float targetArousal = 0.6f;
    float targetIntensity = 0.65f;

    EmotionNode nearest = thesaurus->findNearest(targetValence, targetArousal, targetIntensity);
    EXPECT_GT(nearest.id, 0);

    // Calculate distance to nearest
    float nearestDistance = std::sqrt(
        std::pow(nearest.valence - targetValence, 2) +
        std::pow(nearest.arousal - targetArousal, 2) +
        std::pow(nearest.intensity - targetIntensity, 2)
    );

    // Verify that no other emotion is closer
    auto all = thesaurus->all();
    for (const auto& [id, emotion] : all) {
        if (emotion.id == nearest.id) continue;

        float distance = std::sqrt(
            std::pow(emotion.valence - targetValence, 2) +
            std::pow(emotion.arousal - targetArousal, 2) +
            std::pow(emotion.intensity - targetIntensity, 2)
        );

        EXPECT_GE(distance, nearestDistance * 0.99f)
            << "Nearest emotion should be closest to target coordinates";
    }
}
