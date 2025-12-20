#include <gtest/gtest.h>
#include "../../src/voice/LyricGenerator.h"
#include "../../src/voice/PhonemeConverter.h"
#include "../../src/voice/PitchPhonemeAligner.h"
#include "../../src/voice/VoiceSynthesizer.h"
#include "../../src/common/Types.h"
#include <vector>

using namespace kelly;

class LyricVocalIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        lyricGenerator_ = std::make_unique<LyricGenerator>();
        phonemeConverter_ = std::make_unique<PhonemeConverter>();
        aligner_ = std::make_unique<PitchPhonemeAligner>();
        voiceSynthesizer_ = std::make_unique<VoiceSynthesizer>();

        voiceSynthesizer_->setEnabled(true);
        voiceSynthesizer_->prepare(44100.0);
        voiceSynthesizer_->setBPM(120.0f);
        aligner_->setBPM(120.0f);
    }

    std::unique_ptr<LyricGenerator> lyricGenerator_;
    std::unique_ptr<PhonemeConverter> phonemeConverter_;
    std::unique_ptr<PitchPhonemeAligner> aligner_;
    std::unique_ptr<VoiceSynthesizer> voiceSynthesizer_;
};

TEST_F(LyricVocalIntegrationTest, FullPipeline) {
    // Generate lyrics
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

    auto lyricResult = lyricGenerator_->generateLyrics(emotion, wound, nullptr);
    EXPECT_FALSE(lyricResult.lines.empty());

    // Create MIDI context
    GeneratedMidi midiContext;
    midiContext.lengthInBeats = 16.0;
    midiContext.bpm = 120.0f;

    // Generate vocal melody
    auto vocalNotes = voiceSynthesizer_->generateVocalMelody(emotion, midiContext, &lyricResult.structure);
    EXPECT_FALSE(vocalNotes.empty());

    // Align lyrics to melody
    auto alignmentResult = aligner_->alignLyricsToMelody(lyricResult.structure, vocalNotes, &midiContext);
    EXPECT_GT(alignmentResult.alignedPhonemes.size(), 0);
}

TEST_F(LyricVocalIntegrationTest, ConvertLyricsToPhonemes) {
    LyricStructure lyrics;

    LyricSection verse;
    verse.type = SectionType::Verse;
    LyricLine line;
    line.text = "hello world";
    verse.lines.push_back(line);
    lyrics.sections.push_back(verse);

    // Convert lyric lines to phonemes
    for (const auto& section : lyrics.sections) {
        for (const auto& lyricLine : section.lines) {
            auto phonemes = phonemeConverter_->textToPhonemes(lyricLine.text);
            EXPECT_GT(phonemes.size(), 0);
        }
    }
}

TEST_F(LyricVocalIntegrationTest, SynthesizeVocalAudio) {
    EmotionNode emotion;
    emotion.name = "joy";
    emotion.category = EmotionCategory::Joy;
    emotion.valence = 0.8f;
    emotion.arousal = 0.7f;
    emotion.dominance = 0.6f;
    emotion.intensity = 0.8f;

    GeneratedMidi midiContext;
    midiContext.lengthInBeats = 4.0;
    midiContext.bpm = 120.0f;

    // Generate vocal melody
    auto vocalNotes = voiceSynthesizer_->generateVocalMelody(emotion, midiContext);
    EXPECT_FALSE(vocalNotes.empty());

    // Synthesize audio
    auto audio = voiceSynthesizer_->synthesizeAudio(vocalNotes, 44100.0, &emotion);

    // Should generate audio samples
    EXPECT_GT(audio.size(), 0);
}

TEST_F(LyricVocalIntegrationTest, TimingAlignment) {
    LyricStructure lyrics;

    LyricSection verse;
    verse.type = SectionType::Verse;
    LyricLine line;
    line.text = "test line";
    line.targetSyllables = 4;
    verse.lines.push_back(line);
    lyrics.sections.push_back(verse);

    std::vector<VoiceSynthesizer::VocalNote> notes;
    VoiceSynthesizer::VocalNote note;
    note.pitch = 60;
    note.startBeat = 0.0;
    note.duration = 2.0;
    notes.push_back(note);

    auto alignmentResult = aligner_->alignLyricsToMelody(lyrics, notes, nullptr);

    // Check timing alignment
    EXPECT_GT(alignmentResult.alignedPhonemes.size(), 0);
    if (!alignmentResult.alignedPhonemes.empty()) {
        EXPECT_GE(alignmentResult.alignedPhonemes[0].startBeat, 0.0);
        EXPECT_GT(alignmentResult.alignedPhonemes[0].duration, 0.0);
    }
}
