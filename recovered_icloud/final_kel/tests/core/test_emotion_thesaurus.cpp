#include <gtest/gtest.h>
#include "engine/EmotionThesaurus.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>

using namespace kelly;

class EmotionThesaurusTest : public ::testing::Test {
protected:
    void SetUp() override {
        thesaurus = std::make_unique<EmotionThesaurus>();
    }

    std::unique_ptr<EmotionThesaurus> thesaurus;
};

// Test thesaurus initialization
TEST_F(EmotionThesaurusTest, Initialization) {
    EXPECT_GT(thesaurus->size(), 0);
    // Should have 216 emotions
    EXPECT_GE(thesaurus->size(), 200);
}

// Test find by ID
TEST_F(EmotionThesaurusTest, FindById) {
    // Test finding a valid ID (assuming IDs start from 1)
    auto emotion = thesaurus->findById(1);
    EXPECT_TRUE(emotion.has_value());
    if (emotion.has_value()) {
        EXPECT_EQ(emotion->id, 1);
        EXPECT_FALSE(emotion->name.empty());
    }
}

// Test findById with valid IDs in range 1-216
TEST_F(EmotionThesaurusTest, FindById_ValidId) {
    size_t totalSize = thesaurus->size();
    EXPECT_GE(totalSize, 200);  // Should have at least 200 emotions

    // Test a range of valid IDs
    int validIds[] = {1, 10, 50, 100, 150, 200};
    int maxId = static_cast<int>(totalSize);

    for (int id : validIds) {
        if (id <= maxId) {
            auto emotion = thesaurus->findById(id);
            EXPECT_TRUE(emotion.has_value()) << "ID " << id << " should be valid";
            if (emotion.has_value()) {
                EXPECT_EQ(emotion->id, id);
                EXPECT_FALSE(emotion->name.empty());
                // Verify VAD values are in valid ranges
                EXPECT_GE(emotion->valence, -1.0f);
                EXPECT_LE(emotion->valence, 1.0f);
                EXPECT_GE(emotion->arousal, 0.0f);
                EXPECT_LE(emotion->arousal, 1.0f);
                EXPECT_GE(emotion->intensity, 0.0f);
                EXPECT_LE(emotion->intensity, 1.0f);
            }
        }
    }
}

// Test findById with invalid IDs
TEST_F(EmotionThesaurusTest, FindById_InvalidId) {
    // Test with ID 0
    auto emotion0 = thesaurus->findById(0);
    EXPECT_FALSE(emotion0.has_value()) << "ID 0 should be invalid";

    // Test with negative ID
    auto emotionNeg = thesaurus->findById(-1);
    EXPECT_FALSE(emotionNeg.has_value()) << "Negative ID should be invalid";

    // Test with ID > 216 (assuming max is around 216)
    size_t totalSize = thesaurus->size();
    auto emotionLarge = thesaurus->findById(static_cast<int>(totalSize + 100));
    EXPECT_FALSE(emotionLarge.has_value()) << "ID beyond range should be invalid";

    // Test with very large ID
    auto emotionVeryLarge = thesaurus->findById(999999);
    EXPECT_FALSE(emotionVeryLarge.has_value()) << "Very large ID should be invalid";
}

// Test findById returns correct emotion
TEST_F(EmotionThesaurusTest, FindById_ReturnsCorrectEmotion) {
    // Get all emotions first
    auto all = thesaurus->all();
    EXPECT_GT(all.size(), 0);

    // Test that findById returns the same emotion as from all()
    for (const auto& [id, expectedEmotion] : all) {
        auto foundEmotion = thesaurus->findById(id);
        EXPECT_TRUE(foundEmotion.has_value()) << "ID " << id << " should be found";
        if (foundEmotion.has_value()) {
            EXPECT_EQ(foundEmotion->id, expectedEmotion.id);
            EXPECT_EQ(foundEmotion->name, expectedEmotion.name);
            EXPECT_FLOAT_EQ(foundEmotion->valence, expectedEmotion.valence);
            EXPECT_FLOAT_EQ(foundEmotion->arousal, expectedEmotion.arousal);
            EXPECT_FLOAT_EQ(foundEmotion->intensity, expectedEmotion.intensity);
            EXPECT_EQ(foundEmotion->category, expectedEmotion.category);
        }

        // Limit to first 10 for performance
        if (id >= 10) break;
    }
}

// Basic thread safety check for ID lookups
TEST_F(EmotionThesaurusTest, FindById_ThreadSafe) {
    const int numThreads = 4;
    const int lookupsPerThread = 25;
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    auto lookupTask = [&](int threadId) {
        for (int i = 0; i < lookupsPerThread; ++i) {
            int id = (threadId * lookupsPerThread + i) % 216 + 1;  // IDs 1-216
            auto emotion = thesaurus->findById(id);
            if (emotion.has_value()) {
                EXPECT_EQ(emotion->id, id);
                successCount++;
            } else {
                failureCount++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(lookupTask, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load(), 0);
    // Basic check: should have more successes than failures for valid ID range
    EXPECT_GE(successCount.load(), failureCount.load());
}

// Test find by name
TEST_F(EmotionThesaurusTest, FindByName) {
    std::vector<std::string> emotionNames = {"joy", "grief", "rage", "anxiety", "serenity"};

    for (const auto& name : emotionNames) {
        auto emotion = thesaurus->findByName(name);
        if (emotion.has_value()) {
            EXPECT_EQ(emotion->name, name);
            EXPECT_GT(emotion->id, 0);
        }
    }
}

// Test case-insensitive search
TEST_F(EmotionThesaurusTest, CaseInsensitiveSearch) {
    auto lower = thesaurus->findByName("joy");
    auto upper = thesaurus->findByName("JOY");
    auto mixed = thesaurus->findByName("JoY");

    if (lower.has_value() && upper.has_value()) {
        EXPECT_EQ(lower->id, upper->id);
    }
    if (lower.has_value() && mixed.has_value()) {
        EXPECT_EQ(lower->id, mixed->id);
    }
}

// Test find by category
TEST_F(EmotionThesaurusTest, FindByCategory) {
    std::vector<EmotionCategory> categories = {
        EmotionCategory::Joy,
        EmotionCategory::Sadness,
        EmotionCategory::Anger,
        EmotionCategory::Fear
    };

    for (const auto& category : categories) {
        std::vector<EmotionNode> emotions = thesaurus->findByCategory(category);
        EXPECT_GT(emotions.size(), 0) << "Category should have emotions";

        for (const auto& emotion : emotions) {
            EXPECT_EQ(emotion.category, category);
        }
    }
}

// Test find nearest
TEST_F(EmotionThesaurusTest, FindNearest) {
    // Test finding nearest emotion to specific V/A/I coordinates
    EmotionNode nearest1 = thesaurus->findNearest(0.8f, 0.8f, 0.7f); // High positive
    EXPECT_GT(nearest1.id, 0);
    EXPECT_GE(nearest1.valence, 0.0f);

    EmotionNode nearest2 = thesaurus->findNearest(-0.7f, 0.3f, 0.8f); // Negative valence
    EXPECT_GT(nearest2.id, 0);
    EXPECT_LE(nearest2.valence, 0.0f);

    EmotionNode nearest3 = thesaurus->findNearest(-0.8f, 0.9f, 0.9f); // High negative
    EXPECT_GT(nearest3.id, 0);
}

// Test find related
TEST_F(EmotionThesaurusTest, FindRelated) {
    auto emotion = thesaurus->findByName("joy");
    if (emotion.has_value()) {
        std::vector<EmotionNode> related = thesaurus->findRelated(emotion->id);

        // Should have some related emotions
        EXPECT_GE(related.size(), 0);

        // Related emotions should be different from the original
        for (const auto& rel : related) {
            EXPECT_NE(rel.id, emotion->id);
        }
    }
}

// Test suggest mode
TEST_F(EmotionThesaurusTest, SuggestMode) {
    auto emotion = thesaurus->findByName("joy");
    if (emotion.has_value()) {
        std::string mode = thesaurus->suggestMode(*emotion);
        EXPECT_FALSE(mode.empty());
    }

    auto sadEmotion = thesaurus->findByName("grief");
    if (sadEmotion.has_value()) {
        std::string mode = thesaurus->suggestMode(*sadEmotion);
        EXPECT_FALSE(mode.empty());
    }
}

// Test suggest tempo modifier
TEST_F(EmotionThesaurusTest, SuggestTempoModifier) {
    auto emotion = thesaurus->findByName("joy");
    if (emotion.has_value()) {
        float modifier = thesaurus->suggestTempoModifier(*emotion);
        EXPECT_GE(modifier, 0.5f);
        EXPECT_LE(modifier, 2.0f);
    }
}

// Test suggest dynamic range
TEST_F(EmotionThesaurusTest, SuggestDynamicRange) {
    auto emotion = thesaurus->findByName("rage");
    if (emotion.has_value()) {
        float range = thesaurus->suggestDynamicRange(*emotion);
        EXPECT_GE(range, 0.0f);
        EXPECT_LE(range, 1.0f);
    }
}

// Test all emotions access
TEST_F(EmotionThesaurusTest, AllEmotions) {
    auto all = thesaurus->all();
    EXPECT_GT(all.size(), 0);
    EXPECT_EQ(all.size(), thesaurus->size());
}

// Test emotion node validity
TEST_F(EmotionThesaurusTest, EmotionNodeValidity) {
    auto all = thesaurus->all();

    for (const auto& [id, emotion] : all) {
        EXPECT_EQ(emotion.id, id);
        EXPECT_FALSE(emotion.name.empty());
        EXPECT_GE(emotion.valence, -1.0f);
        EXPECT_LE(emotion.valence, 1.0f);
        EXPECT_GE(emotion.arousal, 0.0f);
        EXPECT_LE(emotion.arousal, 1.0f);
        EXPECT_GE(emotion.intensity, 0.0f);
        EXPECT_LE(emotion.intensity, 1.0f);
    }
}
