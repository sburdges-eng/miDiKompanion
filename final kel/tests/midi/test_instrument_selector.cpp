#include <gtest/gtest.h>
#include "midi/InstrumentSelector.h"
#include <string>

using namespace kelly;

class InstrumentSelectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        selector = std::make_unique<InstrumentSelector>();
    }
    
    std::unique_ptr<InstrumentSelector> selector;
};

// Test instrument selection by emotion
TEST_F(InstrumentSelectorTest, SelectByEmotion) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        int instrument = selector->selectForEmotion(emotion);
        EXPECT_GE(instrument, 0);
        EXPECT_LE(instrument, 127);
    }
}

// Test instrument selection by category
TEST_F(InstrumentSelectorTest, SelectByCategory) {
    std::vector<EmotionCategory> categories = {
        EmotionCategory::Joy,
        EmotionCategory::Sadness,
        EmotionCategory::Anger
    };
    
    for (const auto& category : categories) {
        int instrument = selector->selectForCategory(category);
        EXPECT_GE(instrument, 0);
        EXPECT_LE(instrument, 127);
    }
}

// Test instrument selection by section
TEST_F(InstrumentSelectorTest, SelectBySection) {
    std::vector<std::string> sections = {"intro", "verse", "chorus", "bridge"};
    
    for (const auto& section : sections) {
        int instrument = selector->selectForSection(section);
        EXPECT_GE(instrument, 0);
        EXPECT_LE(instrument, 127);
    }
}
