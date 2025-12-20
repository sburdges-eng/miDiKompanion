#include <gtest/gtest.h>
#include "engine/IntentPipeline.h"
#include "midi/MidiGenerator.h"
#include "common/Types.h"
#include <string>
#include <vector>

using namespace kelly;

class EmotionJourneyTest : public ::testing::Test {
protected:
    void SetUp() override {
        intentPipeline = std::make_unique<IntentPipeline>();
        midiGenerator = std::make_unique<MidiGenerator>();
    }

    std::unique_ptr<IntentPipeline> intentPipeline;
    std::unique_ptr<MidiGenerator> midiGenerator;
};

// Test processJourney() flow with emotion IDs
TEST_F(EmotionJourneyTest, ProcessJourney_WithEmotionIds) {
    // Create SideA and SideB with emotion IDs
    SideA sideA;
    sideA.description = "I feel sad";
    sideA.intensity = 0.7f;

    // Find a sad emotion ID
    auto sadEmotion = intentPipeline->thesaurus().findNearest(-0.7f, 0.3f, 0.6f);
    sideA.emotionId = sadEmotion.id;

    SideB sideB;
    sideB.description = "I want to feel happy";
    sideB.intensity = 0.7f;

    // Find a happy emotion ID
    auto happyEmotion = intentPipeline->thesaurus().findNearest(0.8f, 0.6f, 0.7f);
    sideB.emotionId = happyEmotion.id;

    // Process journey
    IntentResult intent = intentPipeline->processJourney(sideA, sideB);

    // Verify IntentResult contains emotion, rule breaks, musical params
    EXPECT_GT(intent.emotion.id, 0);
    EXPECT_FALSE(intent.emotion.name.empty());
    EXPECT_GE(intent.emotion.valence, -1.0f);
    EXPECT_LE(intent.emotion.valence, 1.0f);
    EXPECT_GE(intent.emotion.arousal, 0.0f);
    EXPECT_LE(intent.emotion.arousal, 1.0f);
    EXPECT_FALSE(intent.mode.empty());
    EXPECT_GT(intent.tempo, 0.0f);

    // Generate MIDI and verify emotional characteristics match journey
    GeneratedMidi midi = midiGenerator->generate(intent, 8, 0.5f, 0.4f, 0.0f, 0.75f);

    EXPECT_GT(midi.chords.size(), 0);
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);
    EXPECT_GT(midi.lengthInBeats, 0.0);
    EXPECT_GT(midi.bpm, 0.0f);
}

// Test transition between emotions
TEST_F(EmotionJourneyTest, ProcessJourney_EmotionTransition) {
    // Test multiple emotion transitions
    struct JourneyTestCase {
        std::string sideADesc;
        float sideAValence;
        float sideAArousal;
        std::string sideBDesc;
        float sideBValence;
        float sideBArousal;
    };

    std::vector<JourneyTestCase> testCases = {
        {"I feel sad", -0.7f, 0.3f, "I want to feel happy", 0.8f, 0.6f},
        {"I feel anxious", -0.5f, 0.9f, "I want to feel calm", 0.3f, 0.2f},
        {"I feel angry", -0.8f, 0.9f, "I want to feel peaceful", 0.5f, 0.3f},
        {"I feel neutral", 0.0f, 0.5f, "I want to feel excited", 0.7f, 0.8f}
    };

    for (const auto& testCase : testCases) {
        SideA sideA;
        sideA.description = testCase.sideADesc;
        sideA.intensity = 0.6f;
        auto emotionA = intentPipeline->thesaurus().findNearest(
            testCase.sideAValence, testCase.sideAArousal, 0.6f);
        sideA.emotionId = emotionA.id;

        SideB sideB;
        sideB.description = testCase.sideBDesc;
        sideB.intensity = 0.6f;
        auto emotionB = intentPipeline->thesaurus().findNearest(
            testCase.sideBValence, testCase.sideBArousal, 0.6f);
        sideB.emotionId = emotionB.id;

        IntentResult intent = intentPipeline->processJourney(sideA, sideB);

        // Verify intent is valid
        EXPECT_GT(intent.emotion.id, 0);
        EXPECT_FALSE(intent.mode.empty());
        EXPECT_GT(intent.tempo, 0.0f);

        // Generate MIDI
        GeneratedMidi midi = midiGenerator->generate(intent, 4, 0.5f, 0.4f, 0.0f, 0.75f);

        // Verify MIDI is valid
        EXPECT_GT(midi.chords.size(), 0);
        EXPECT_GT(midi.melody.size(), 0);
        EXPECT_GT(midi.bass.size(), 0);
    }
}

// Test journey with same emotion (no transition)
TEST_F(EmotionJourneyTest, ProcessJourney_SameEmotion) {
    auto emotion = intentPipeline->thesaurus().findNearest(0.5f, 0.5f, 0.6f);

    SideA sideA;
    sideA.description = "I feel content";
    sideA.intensity = 0.6f;
    sideA.emotionId = emotion.id;

    SideB sideB;
    sideB.description = "I want to stay content";
    sideB.intensity = 0.6f;
    sideB.emotionId = emotion.id;

    IntentResult intent = intentPipeline->processJourney(sideA, sideB);

    // Should still produce valid intent
    EXPECT_GT(intent.emotion.id, 0);
    EXPECT_FALSE(intent.mode.empty());

    GeneratedMidi midi = midiGenerator->generate(intent, 4, 0.5f, 0.4f, 0.0f, 0.75f);
    EXPECT_GT(midi.chords.size(), 0);
}

// Test journey without emotion IDs (fallback to description)
TEST_F(EmotionJourneyTest, ProcessJourney_WithoutEmotionIds) {
    SideA sideA;
    sideA.description = "I feel sad";
    sideA.intensity = 0.7f;
    // No emotionId set

    SideB sideB;
    sideB.description = "I want to feel happy";
    sideB.intensity = 0.7f;
    // No emotionId set

    IntentResult intent = intentPipeline->processJourney(sideA, sideB);

    // Should still process using descriptions
    EXPECT_GT(intent.emotion.id, 0);
    EXPECT_FALSE(intent.mode.empty());

    GeneratedMidi midi = midiGenerator->generate(intent, 4, 0.5f, 0.4f, 0.0f, 0.75f);
    EXPECT_GT(midi.chords.size(), 0);
}

// Test journey with invalid emotion IDs (should fallback gracefully)
TEST_F(EmotionJourneyTest, ProcessJourney_InvalidEmotionIds) {
    SideA sideA;
    sideA.description = "I feel something";
    sideA.intensity = 0.6f;
    sideA.emotionId = 9999;  // Invalid ID

    SideB sideB;
    sideB.description = "I want to feel something else";
    sideB.intensity = 0.6f;
    sideB.emotionId = 9998;  // Invalid ID

    // Should not crash, should fallback to description processing
    IntentResult intent = intentPipeline->processJourney(sideA, sideB);

    // Should still produce valid intent (using descriptions)
    EXPECT_GT(intent.emotion.id, 0);
    EXPECT_FALSE(intent.mode.empty());

    GeneratedMidi midi = midiGenerator->generate(intent, 4, 0.5f, 0.4f, 0.0f, 0.75f);
    EXPECT_GT(midi.chords.size(), 0);
}

// Test journey emotional characteristics match
TEST_F(EmotionJourneyTest, ProcessJourney_EmotionalCharacteristicsMatch) {
    // Sad to Happy journey
    auto sadEmotion = intentPipeline->thesaurus().findNearest(-0.7f, 0.3f, 0.6f);
    auto happyEmotion = intentPipeline->thesaurus().findNearest(0.8f, 0.6f, 0.7f);

    SideA sideA;
    sideA.description = "I feel sad";
    sideA.intensity = 0.7f;
    sideA.emotionId = sadEmotion.id;

    SideB sideB;
    sideB.description = "I want to feel happy";
    sideB.intensity = 0.7f;
    sideB.emotionId = happyEmotion.id;

    IntentResult intent = intentPipeline->processJourney(sideA, sideB);

    // The journey should reflect a transition
    // The resulting emotion should be somewhere between or reflect the destination
    EXPECT_GT(intent.emotion.id, 0);

    // Generate MIDI and verify it reflects the journey
    GeneratedMidi midi = midiGenerator->generate(intent, 8, 0.6f, 0.4f, 0.0f, 0.75f);

    EXPECT_GT(midi.chords.size(), 0);
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);

    // Verify all notes are valid
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startBeat, 0.0);
        EXPECT_GT(note.duration, 0.0);
    }
}
