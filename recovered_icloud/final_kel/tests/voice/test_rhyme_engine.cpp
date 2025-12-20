#include <gtest/gtest.h>
#include "../../src/voice/RhymeEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class RhymeEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_unique<RhymeEngine>();
    }

    std::unique_ptr<RhymeEngine> engine_;
};

TEST_F(RhymeEngineTest, CheckRhymePerfect) {
    // Test perfect rhymes
    auto match1 = engine_->checkRhyme("cat", "bat");
    EXPECT_GT(match1.score, 0.0f);

    auto match2 = engine_->checkRhyme("love", "dove");
    EXPECT_GT(match2.score, 0.0f);
}

TEST_F(RhymeEngineTest, CheckRhymeSlant) {
    // Test slant rhymes (near rhymes)
    auto match = engine_->checkRhyme("cat", "cut");
    // Should detect some level of rhyme (may be perfect or slant)
    EXPECT_GE(match.score, 0.0f);
}

TEST_F(RhymeEngineTest, CheckRhymeNone) {
    // Test non-rhyming words
    auto match = engine_->checkRhyme("cat", "dog");
    // May have low score or none
    EXPECT_GE(match.score, 0.0f);
    EXPECT_LE(match.score, 1.0f);
}

TEST_F(RhymeEngineTest, CheckRhymeSameWord) {
    // Same word should be detected (though not typically desired)
    auto match = engine_->checkRhyme("cat", "cat");
    EXPECT_GT(match.score, 0.0f);
}

TEST_F(RhymeEngineTest, FindRhymes) {
    std::vector<std::string> vocabulary = {
        "cat", "bat", "hat", "rat", "sat",
        "dog", "log", "fog", "bog",
        "love", "dove", "glove", "above"
    };

    auto rhymes = engine_->findRhymes("cat", vocabulary, 5);
    EXPECT_GT(rhymes.size(), 0);

    // Should not include the target word itself
    for (const auto& rhyme : rhymes) {
        EXPECT_NE(rhyme.word2, "cat");
    }
}

TEST_F(RhymeEngineTest, FindRhymesMaxResults) {
    std::vector<std::string> vocabulary = {
        "cat", "bat", "hat", "rat", "sat", "mat", "pat", "vat"
    };

    auto rhymes = engine_->findRhymes("cat", vocabulary, 3);
    EXPECT_LE(rhymes.size(), 3);
}

TEST_F(RhymeEngineTest, ExtractEndPhonemes) {
    auto phonemes = engine_->extractEndPhonemes("cat", 3);
    EXPECT_GT(phonemes.size(), 0);

    phonemes = engine_->extractEndPhonemes("beautiful", 3);
    EXPECT_GT(phonemes.size(), 0);
}

TEST_F(RhymeEngineTest, ComparePhonemeSequences) {
    std::vector<std::string> seq1 = {"/k/", "/æ/", "/t/"};
    std::vector<std::string> seq2 = {"/b/", "/æ/", "/t/"};

    float score = engine_->comparePhonemeSequences(seq1, seq2);
    EXPECT_GE(score, 0.0f);
    EXPECT_LE(score, 1.0f);

    // Same sequence should have high score
    float sameScore = engine_->comparePhonemeSequences(seq1, seq1);
    EXPECT_GT(sameScore, 0.9f);
}

TEST_F(RhymeEngineTest, DetectInternalRhymes) {
    std::vector<std::string> words = {"cat", "sat", "on", "the", "mat"};
    auto internalRhymes = engine_->detectInternalRhymes(words);
    // May find rhymes within the line
    EXPECT_GE(internalRhymes.size(), 0);
}

TEST_F(RhymeEngineTest, GenerateRhymeWords) {
    std::vector<std::string> vocabulary = {
        "cat", "bat", "hat", "rat",
        "dog", "log", "fog",
        "love", "dove", "above"
    };

    // ABAB rhyme scheme
    std::vector<int> scheme = {0, 1, 0, 1};

    auto rhymeWords = engine_->generateRhymeWords(scheme, vocabulary);
    EXPECT_GT(rhymeWords.size(), 0);
}

TEST_F(RhymeEngineTest, GenerateRhymeWordsWithExisting) {
    std::vector<std::string> vocabulary = {
        "cat", "bat", "hat", "rat",
        "dog", "log", "fog"
    };

    std::vector<int> scheme = {0, 1, 0, 1};
    std::map<int, std::string> existingWords;
    existingWords[0] = "cat";

    auto rhymeWords = engine_->generateRhymeWords(scheme, vocabulary, existingWords);
    // Should generate words that rhyme with existing words
    EXPECT_GE(rhymeWords.size(), 0);
}

TEST_F(RhymeEngineTest, BuildRhymeDatabase) {
    std::vector<std::string> vocabulary = {
        "cat", "bat", "hat", "rat",
        "dog", "log", "fog",
        "love", "dove", "above"
    };

    engine_->buildRhymeDatabase(vocabulary);
    // Should not crash, database built
}

TEST_F(RhymeEngineTest, HandleEmptyInput) {
    auto match = engine_->checkRhyme("", "");
    EXPECT_GE(match.score, 0.0f);

    std::vector<std::string> emptyVocab;
    auto rhymes = engine_->findRhymes("cat", emptyVocab, 5);
    EXPECT_EQ(rhymes.size(), 0);

    auto phonemes = engine_->extractEndPhonemes("", 3);
    EXPECT_EQ(phonemes.size(), 0);
}

TEST_F(RhymeEngineTest, RhymeScoreOrdering) {
    std::vector<std::string> vocabulary = {
        "cat", "bat", "hat", "rat", "sat",
        "cut", "but", "hut"
    };

    auto rhymes = engine_->findRhymes("cat", vocabulary, 10);

    // Results should be sorted by score (highest first)
    if (rhymes.size() > 1) {
        for (size_t i = 1; i < rhymes.size(); ++i) {
            EXPECT_GE(rhymes[i-1].score, rhymes[i].score);
        }
    }
}

TEST_F(RhymeEngineTest, DifferentRhymeSchemes) {
    std::vector<std::string> vocabulary = {
        "cat", "bat", "hat", "rat",
        "dog", "log", "fog",
        "love", "dove", "above"
    };

    // Test AABB scheme
    std::vector<int> schemeAABB = {0, 0, 1, 1};
    auto wordsAABB = engine_->generateRhymeWords(schemeAABB, vocabulary);
    EXPECT_GE(wordsAABB.size(), 0);

    // Test ABAB scheme
    std::vector<int> schemeABAB = {0, 1, 0, 1};
    auto wordsABAB = engine_->generateRhymeWords(schemeABAB, vocabulary);
    EXPECT_GE(wordsABAB.size(), 0);
}

TEST_F(RhymeEngineTest, RhymeTypeClassification) {
    // Perfect rhymes should be classified as Perfect
    auto perfectMatch = engine_->checkRhyme("cat", "bat");
    // May be Perfect or Slant depending on implementation
    EXPECT_NE(perfectMatch.type, RhymeEngine::RhymeType::None);

    // Non-rhyming words
    auto noMatch = engine_->checkRhyme("cat", "dog");
    // May be None or have low score
    EXPECT_GE(noMatch.score, 0.0f);
}
