#include <gtest/gtest.h>
#include "../../src/voice/PhonemeConverter.h"
#include <vector>
#include <string>

using namespace kelly;

class PhonemeConverterTest : public ::testing::Test {
protected:
    void SetUp() override {
        converter_ = std::make_unique<PhonemeConverter>();
    }

    std::unique_ptr<PhonemeConverter> converter_;
};

TEST_F(PhonemeConverterTest, WordToPhonemesCommonWord) {
    // Test common words that should be in dictionary
    auto phonemes = converter_->wordToPhonemes("the");
    EXPECT_GT(phonemes.size(), 0);

    phonemes = converter_->wordToPhonemes("love");
    EXPECT_GT(phonemes.size(), 0);

    phonemes = converter_->wordToPhonemes("heart");
    EXPECT_GT(phonemes.size(), 0);
}

TEST_F(PhonemeConverterTest, TextToPhonemes) {
    std::string text = "hello world";
    auto phonemes = converter_->textToPhonemes(text);

    // Should convert to phonemes
    EXPECT_GT(phonemes.size(), 0);
}

TEST_F(PhonemeConverterTest, SplitIntoSyllables) {
    auto syllables = converter_->splitIntoSyllables("hello");
    EXPECT_GT(syllables.size(), 0);

    syllables = converter_->splitIntoSyllables("beautiful");
    EXPECT_GT(syllables.size(), 0);

    // Single syllable words
    syllables = converter_->splitIntoSyllables("cat");
    EXPECT_GE(syllables.size(), 1);
}

TEST_F(PhonemeConverterTest, DetectStress) {
    auto stress = converter_->detectStress("hello");
    EXPECT_GT(stress.size(), 0);

    stress = converter_->detectStress("beautiful");
    EXPECT_GT(stress.size(), 0);
}

TEST_F(PhonemeConverterTest, CountSyllables) {
    EXPECT_EQ(converter_->countSyllables("cat"), 1);
    EXPECT_EQ(converter_->countSyllables("hello"), 2);
    EXPECT_GT(converter_->countSyllables("beautiful"), 2);
}

TEST_F(PhonemeConverterTest, GetPhonemeFromIPA) {
    // Test getting phoneme data
    Phoneme p = converter_->getPhonemeFromIPA("/i/");
    EXPECT_FALSE(p.ipa.empty());
    EXPECT_EQ(p.ipa, "/i/");

    p = converter_->getPhonemeFromIPA("/p/");
    EXPECT_FALSE(p.ipa.empty());

    // Test non-existent phoneme
    p = converter_->getPhonemeFromIPA("/xyz/");
    EXPECT_TRUE(p.ipa.empty());
}

TEST_F(PhonemeConverterTest, HandlePunctuation) {
    std::string text = "Hello, world!";
    auto phonemes = converter_->textToPhonemes(text);

    // Should handle punctuation gracefully
    EXPECT_GT(phonemes.size(), 0);
}

TEST_F(PhonemeConverterTest, HandleEmptyString) {
    auto phonemes = converter_->textToPhonemes("");
    EXPECT_EQ(phonemes.size(), 0);

    auto syllables = converter_->splitIntoSyllables("");
    EXPECT_EQ(syllables.size(), 0);

    auto stress = converter_->detectStress("");
    EXPECT_EQ(stress.size(), 0);
}

TEST_F(PhonemeConverterTest, WordToPhonemesRuleBased) {
    // Test words not in dictionary (should use rule-based G2P)
    auto phonemes = converter_->wordToPhonemes("testword");
    EXPECT_GT(phonemes.size(), 0);

    phonemes = converter_->wordToPhonemes("xyzabc");
    EXPECT_GT(phonemes.size(), 0);
}
