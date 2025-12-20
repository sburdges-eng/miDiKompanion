#include <gtest/gtest.h>
#include "ml/MIDITokenizer.h"
#include "common/Types.h"
#include <vector>
#include <algorithm>

using namespace kelly;

class MIDITokenizerTest : public ::testing::Test {
protected:
    MIDITokenizer tokenizer;
    static constexpr int TICKS_PER_BEAT = 480;
};

TEST_F(MIDITokenizerTest, EncodeDecodeSingleNote) {
    // Create a test MIDI note
    MidiNote originalNote;
    originalNote.pitch = 60;  // C4
    originalNote.velocity = 100;
    originalNote.startBeat = 0.0;
    originalNote.duration = 1.0;  // 1 beat

    // Encode
    uint32_t token = tokenizer.encodeNote(originalNote, TICKS_PER_BEAT);

    // Decode
    MidiNote decodedNote = tokenizer.decodeNote(token, TICKS_PER_BEAT, 0.0);

    // Check pitch (should be exact)
    EXPECT_EQ(decodedNote.pitch, 60);

    // Check velocity (may be quantized, but should be close)
    EXPECT_GE(decodedNote.velocity, 90);
    EXPECT_LE(decodedNote.velocity, 110);

    // Check duration (should be close to 1.0 beats)
    EXPECT_GT(decodedNote.duration, 0.8);
    EXPECT_LT(decodedNote.duration, 1.2);
}

TEST_F(MIDITokenizerTest, EncodeDecodeSequence) {
    // Create a sequence of notes
    std::vector<MidiNote> originalNotes;
    for (int i = 0; i < 4; ++i) {
        MidiNote note;
        note.pitch = 60 + i * 2;  // C4, D4, E4, F#4
        note.velocity = 80 + i * 10;
        note.startBeat = i * 1.0;
        note.duration = 0.5;
        originalNotes.push_back(note);
    }

    // Encode sequence
    std::vector<uint32_t> tokens = tokenizer.encodeSequence(originalNotes, TICKS_PER_BEAT);

    // Should have START, 4 notes, END
    EXPECT_GE(tokens.size(), 6);
    EXPECT_EQ(tokens[0], MIDITokenizer::TOKEN_START);
    EXPECT_EQ(tokens.back(), MIDITokenizer::TOKEN_END);

    // Decode sequence
    std::vector<MidiNote> decodedNotes = tokenizer.decodeSequence(tokens, TICKS_PER_BEAT);

    // Should have 4 notes
    EXPECT_EQ(decodedNotes.size(), 4);

    // Check pitches (should match)
    for (size_t i = 0; i < decodedNotes.size(); ++i) {
        EXPECT_EQ(decodedNotes[i].pitch, originalNotes[i].pitch);
    }
}

TEST_F(MIDITokenizerTest, VelocityQuantization) {
    // Test velocity quantization
    int velocity = 64;  // Middle velocity
    int bin = MIDITokenizer::quantizeVelocity(velocity);
    int dequantized = MIDITokenizer::dequantizeVelocity(bin);

    // Dequantized should be close to original
    EXPECT_GE(dequantized, velocity - 4);
    EXPECT_LE(dequantized, velocity + 4);
}

TEST_F(MIDITokenizerTest, DurationQuantization) {
    // Test duration quantization
    double duration = 1.0;  // 1 beat
    int bin = MIDITokenizer::quantizeDuration(duration, TICKS_PER_BEAT);
    double dequantized = MIDITokenizer::dequantizeDuration(bin, TICKS_PER_BEAT);

    // Dequantized should be close to original
    EXPECT_GT(dequantized, 0.8);
    EXPECT_LT(dequantized, 1.2);
}

TEST_F(MIDITokenizerTest, EmotionConditioning) {
    // Create a token sequence
    std::vector<MidiNote> notes;
    MidiNote note;
    note.pitch = 60;
    note.velocity = 100;
    note.duration = 1.0;
    notes.push_back(note);

    std::vector<uint32_t> tokens = tokenizer.encodeSequence(notes, TICKS_PER_BEAT);

    // Add emotion conditioning
    float valence = 0.5f;
    float arousal = 0.7f;
    tokenizer.addEmotionConditioning(tokens, valence, arousal);

    // Extract emotion conditioning
    float extractedValence = 0.0f;
    float extractedArousal = 0.0f;
    bool extracted = tokenizer.extractEmotionConditioning(tokens, extractedValence, extractedArousal);

    EXPECT_TRUE(extracted);
    EXPECT_NEAR(extractedValence, valence, 0.1f);
    EXPECT_NEAR(extractedArousal, arousal, 0.1f);
}

TEST_F(MIDITokenizerTest, SpecialTokens) {
    // Test special tokens
    MidiNote padNote = tokenizer.decodeNote(MIDITokenizer::TOKEN_PAD, TICKS_PER_BEAT, 0.0);
    EXPECT_EQ(padNote.pitch, 0);
    EXPECT_EQ(padNote.velocity, 0);

    MidiNote restNote = tokenizer.decodeNote(MIDITokenizer::TOKEN_REST, TICKS_PER_BEAT, 0.0);
    EXPECT_EQ(restNote.pitch, 0);
    EXPECT_EQ(restNote.velocity, 0);
}

TEST_F(MIDITokenizerTest, PitchRange) {
    // Test full MIDI pitch range
    for (int pitch = 0; pitch <= 127; pitch += 10) {
        MidiNote note;
        note.pitch = pitch;
        note.velocity = 100;
        note.duration = 1.0;

        uint32_t token = tokenizer.encodeNote(note, TICKS_PER_BEAT);
        MidiNote decoded = tokenizer.decodeNote(token, TICKS_PER_BEAT, 0.0);

        EXPECT_EQ(decoded.pitch, pitch);
    }
}

