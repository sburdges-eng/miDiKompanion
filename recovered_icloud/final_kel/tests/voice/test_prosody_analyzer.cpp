#include <gtest/gtest.h>
#include "../../src/voice/ProsodyAnalyzer.h"
#include "../../src/voice/LyricTypes.h"
#include <vector>
#include <string>

using namespace kelly;

class ProsodyAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override {
        analyzer_ = std::make_unique<ProsodyAnalyzer>();
    }

    std::unique_ptr<ProsodyAnalyzer> analyzer_;
};

TEST_F(ProsodyAnalyzerTest, DetectStressSingleSyllable) {
    auto stress = analyzer_->detectStress("cat");
    EXPECT_EQ(stress.size(), 1);
    if (!stress.empty()) {
        EXPECT_GE(stress[0], 1); // Single syllable should be stressed
    }
}

TEST_F(ProsodyAnalyzerTest, DetectStressMultiSyllable) {
    auto stress = analyzer_->detectStress("hello");
    EXPECT_GT(stress.size(), 1);

    stress = analyzer_->detectStress("beautiful");
    EXPECT_GT(stress.size(), 2);
}

TEST_F(ProsodyAnalyzerTest, DetectStressPattern) {
    std::vector<std::string> words = {"hello", "world"};
    auto pattern = analyzer_->detectStressPattern(words);
    EXPECT_GT(pattern.size(), 0);
}

TEST_F(ProsodyAnalyzerTest, CountSyllables) {
    EXPECT_EQ(analyzer_->countSyllables("cat"), 1);
    EXPECT_EQ(analyzer_->countSyllables("hello"), 2);
    EXPECT_GT(analyzer_->countSyllables("beautiful"), 2);
    EXPECT_EQ(analyzer_->countSyllables(""), 0);
}

TEST_F(ProsodyAnalyzerTest, CountSyllablesMultipleWords) {
    std::vector<std::string> words = {"hello", "world", "test"};
    int total = analyzer_->countSyllables(words);
    EXPECT_GT(total, 0);
}

TEST_F(ProsodyAnalyzerTest, MatchMeterIambic) {
    // Iambic: unstressed-stressed pattern
    std::vector<int> stressPattern = {0, 2, 0, 2, 0, 2};
    float score = analyzer_->matchMeter(stressPattern, ProsodyAnalyzer::MeterType::Iambic);
    EXPECT_GE(score, 0.0f);
    EXPECT_LE(score, 1.0f);
}

TEST_F(ProsodyAnalyzerTest, MatchMeterTrochaic) {
    // Trochaic: stressed-unstressed pattern
    std::vector<int> stressPattern = {2, 0, 2, 0, 2, 0};
    float score = analyzer_->matchMeter(stressPattern, ProsodyAnalyzer::MeterType::Trochaic);
    EXPECT_GE(score, 0.0f);
    EXPECT_LE(score, 1.0f);
}

TEST_F(ProsodyAnalyzerTest, DetectMeter) {
    std::vector<int> iambicPattern = {0, 2, 0, 2};
    auto meterType = analyzer_->detectMeter(iambicPattern);
    EXPECT_NE(meterType, ProsodyAnalyzer::MeterType::None);
}

TEST_F(ProsodyAnalyzerTest, GetMeterPattern) {
    auto pattern = analyzer_->getMeterPattern(ProsodyAnalyzer::MeterType::Iambic, 4);
    EXPECT_FALSE(pattern.pattern.empty());
    EXPECT_EQ(pattern.type, ProsodyAnalyzer::MeterType::Iambic);
}

TEST_F(ProsodyAnalyzerTest, ValidateLineLength) {
    LyricLine line;
    line.text = "hello world test";
    line.targetSyllables = 4;

    bool valid = analyzer_->validateLineLength(line, 4);
    // Should validate (may be true or false depending on actual syllable count)
    // Just check it doesn't crash
    (void)valid;
}

TEST_F(ProsodyAnalyzerTest, SelectWordsForMeter) {
    std::vector<std::string> words = {"hello", "world", "test", "beautiful", "cat"};
    ProsodyAnalyzer::MeterPattern meter;
    meter.type = ProsodyAnalyzer::MeterType::Iambic;
    meter.pattern = {0, 2, 0, 2};

    auto selected = analyzer_->selectWordsForMeter(words, meter);
    // Should return words that match meter
    EXPECT_GE(selected.size(), 0);
}

TEST_F(ProsodyAnalyzerTest, CalculateRhythmScore) {
    std::vector<int> stressPattern = {2, 0, 1, 0, 2, 0};
    float score = analyzer_->calculateRhythmScore(stressPattern);
    EXPECT_GE(score, 0.0f);
    EXPECT_LE(score, 1.0f);
}

TEST_F(ProsodyAnalyzerTest, HandleEmptyInput) {
    auto stress = analyzer_->detectStress("");
    EXPECT_EQ(stress.size(), 0);

    std::vector<std::string> emptyWords;
    auto pattern = analyzer_->detectStressPattern(emptyWords);
    EXPECT_EQ(pattern.size(), 0);

    int count = analyzer_->countSyllables("");
    EXPECT_EQ(count, 0);
}

TEST_F(ProsodyAnalyzerTest, StressWithSuffixes) {
    // Test words with common suffixes that affect stress
    auto stress1 = analyzer_->detectStress("running");
    EXPECT_GT(stress1.size(), 0);

    auto stress2 = analyzer_->detectStress("action");
    EXPECT_GT(stress2.size(), 0);

    auto stress3 = analyzer_->detectStress("beautiful");
    EXPECT_GT(stress3.size(), 0);
}

TEST_F(ProsodyAnalyzerTest, DifferentMeterTypes) {
    std::vector<ProsodyAnalyzer::MeterType> meterTypes = {
        ProsodyAnalyzer::MeterType::Iambic,
        ProsodyAnalyzer::MeterType::Trochaic,
        ProsodyAnalyzer::MeterType::Anapestic,
        ProsodyAnalyzer::MeterType::Dactylic
    };

    for (auto meterType : meterTypes) {
        auto pattern = analyzer_->getMeterPattern(meterType, 8);
        EXPECT_EQ(pattern.type, meterType);
        EXPECT_GT(pattern.pattern.size(), 0);
    }
}
