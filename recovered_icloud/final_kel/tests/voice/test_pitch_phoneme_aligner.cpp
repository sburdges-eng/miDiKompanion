#include <gtest/gtest.h>
#include "../../src/voice/PitchPhonemeAligner.h"
#include "../../src/voice/VoiceSynthesizer.h"
#include "../../src/common/Types.h"
#include <vector>

using namespace kelly;

class PitchPhonemeAlignerTest : public ::testing::Test {
protected:
    void SetUp() override {
        aligner_ = std::make_unique<PitchPhonemeAligner>();
        aligner_->setBPM(120.0f);
    }

    std::unique_ptr<PitchPhonemeAligner> aligner_;
};

TEST_F(PitchPhonemeAlignerTest, AlignPhonemesToNote) {
    std::vector<Phoneme> phonemes;

    // Create test phonemes
    Phoneme p1, p2;
    p1.ipa = "/h/";
    p1.duration_ms = 50.0f;
    p2.ipa = "/i/";
    p2.duration_ms = 120.0f;
    phonemes.push_back(p1);
    phonemes.push_back(p2);

    VoiceSynthesizer::VocalNote note;
    note.pitch = 60;
    note.startBeat = 0.0;
    note.duration = 1.0;

    auto aligned = aligner_->alignPhonemesToNote(phonemes, note, 0.0);

    EXPECT_EQ(aligned.size(), 2);
    if (!aligned.empty()) {
        EXPECT_EQ(aligned[0].phoneme.ipa, "/h/");
        EXPECT_EQ(aligned[0].midiPitch, 60);
        EXPECT_EQ(aligned[0].isStartOfSyllable, true);
    }
    if (aligned.size() > 1) {
        EXPECT_EQ(aligned[1].phoneme.ipa, "/i/");
        EXPECT_EQ(aligned[1].isEndOfSyllable, true);
    }
}

TEST_F(PitchPhonemeAlignerTest, CalculatePhonemeDurations) {
    std::vector<Phoneme> phonemes;

    Phoneme p1, p2, p3;
    p1.duration_ms = 100.0f;
    p2.duration_ms = 150.0f;
    p3.duration_ms = 120.0f;
    phonemes.push_back(p1);
    phonemes.push_back(p2);
    phonemes.push_back(p3);

    double noteDuration = 1.0; // 1 beat
    auto durations = aligner_->calculatePhonemeDurations(noteDuration, phonemes);

    EXPECT_EQ(durations.size(), 3);

    // Durations should sum to approximately noteDuration
    double totalDuration = 0.0;
    for (double d : durations) {
        totalDuration += d;
        EXPECT_GT(d, 0.0);
    }
    EXPECT_NEAR(totalDuration, noteDuration, 0.1);
}

TEST_F(PitchPhonemeAlignerTest, AlignLyricsToMelody) {
    LyricStructure lyrics;

    // Create a simple lyric structure
    LyricSection verse;
    verse.type = SectionType::Verse;
    verse.sectionNumber = 1;

    LyricLine line1, line2;
    line1.text = "hello world";
    line1.targetSyllables = 4;
    line2.text = "test phrase";
    line2.targetSyllables = 4;

    verse.lines.push_back(line1);
    verse.lines.push_back(line2);
    lyrics.sections.push_back(verse);

    // Create vocal notes
    std::vector<VoiceSynthesizer::VocalNote> notes;
    VoiceSynthesizer::VocalNote note1, note2;
    note1.pitch = 60;
    note1.startBeat = 0.0;
    note1.duration = 1.0;
    note2.pitch = 62;
    note2.startBeat = 1.0;
    note2.duration = 1.0;
    notes.push_back(note1);
    notes.push_back(note2);

    auto result = aligner_->alignLyricsToMelody(lyrics, notes, nullptr);

    // Should align phonemes to notes
    EXPECT_GT(result.alignedPhonemes.size(), 0);
    EXPECT_EQ(result.vocalNotes.size(), notes.size());
}

TEST_F(PitchPhonemeAlignerTest, SetBPM) {
    aligner_->setBPM(140.0f);
    // BPM should be set (no direct getter, but it affects timing)
    // This is verified indirectly through duration calculations
}

TEST_F(PitchPhonemeAlignerTest, SetAllowMelisma) {
    aligner_->setAllowMelisma(true);
    aligner_->setAllowMelisma(false);
    // Should not crash
}

TEST_F(PitchPhonemeAlignerTest, SetPortamentoTime) {
    aligner_->setPortamentoTime(0.1);
    // Should not crash
}

TEST_F(PitchPhonemeAlignerTest, HandleEmptyInput) {
    LyricStructure emptyLyrics;
    std::vector<VoiceSynthesizer::VocalNote> emptyNotes;

    auto result = aligner_->alignLyricsToMelody(emptyLyrics, emptyNotes, nullptr);

    EXPECT_EQ(result.alignedPhonemes.size(), 0);
    EXPECT_EQ(result.vocalNotes.size(), 0);
}
