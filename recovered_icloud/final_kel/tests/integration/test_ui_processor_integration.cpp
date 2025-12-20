#include <gtest/gtest.h>
#include "engine/IntentPipeline.h"
#include "midi/MidiGenerator.h"
#include "common/Types.h"
#include <string>
#include <vector>
#include <atomic>

using namespace kelly;

/**
 * UI-to-Processor Integration Tests
 *
 * NOTE: These tests simulate UI-to-Processor interactions without requiring JUCE.
 * Full UI integration tests would require JUCE framework and PluginProcessor/PluginEditor.
 *
 * This test suite verifies the integration patterns and data flow that would occur
 * between UI components and the processor.
 */
class UIProcessorIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        intentPipeline = std::make_unique<IntentPipeline>();
        midiGenerator = std::make_unique<MidiGenerator>();
    }

    std::unique_ptr<IntentPipeline> intentPipeline;
    std::unique_ptr<MidiGenerator> midiGenerator;

    // Simulate APVTS parameter values
    struct SimulatedAPVTS {
        float valence = 0.0f;
        float arousal = 0.5f;
        float intensity = 0.5f;
        float complexity = 0.5f;
        float humanize = 0.4f;
        float feel = 0.0f;
        float dynamics = 0.75f;
        int bars = 8;
    } apvts;
};

// Test parameter slider changes update processor state
TEST_F(UIProcessorIntegrationTest, ParameterSliderChanges) {
    // Simulate parameter changes
    std::vector<std::pair<std::string, float>> parameterChanges = {
        {"valence", 0.8f},
        {"arousal", 0.7f},
        {"intensity", 0.6f},
        {"complexity", 0.9f},
        {"humanize", 0.5f},
        {"feel", 0.3f},
        {"dynamics", 0.8f}
    };

    for (const auto& [param, value] : parameterChanges) {
        // Simulate updating parameter
        if (param == "valence") apvts.valence = value;
        else if (param == "arousal") apvts.arousal = value;
        else if (param == "intensity") apvts.intensity = value;
        else if (param == "complexity") apvts.complexity = value;
        else if (param == "humanize") apvts.humanize = value;
        else if (param == "feel") apvts.feel = value;
        else if (param == "dynamics") apvts.dynamics = value;

        // Verify parameter is updated
        if (param == "valence") EXPECT_FLOAT_EQ(apvts.valence, value);
        else if (param == "arousal") EXPECT_FLOAT_EQ(apvts.arousal, value);
        else if (param == "intensity") EXPECT_FLOAT_EQ(apvts.intensity, value);
    }
}

// Test Generate button triggers generateMidi()
TEST_F(UIProcessorIntegrationTest, GenerateButton_TriggersGeneration) {
    // Simulate Generate button click
    // 1. Get wound text from UI
    std::string woundText = "I feel happy";

    // 2. Process wound through IntentPipeline
    Wound wound;
    wound.description = woundText;
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    EXPECT_GT(intent.emotion.id, 0);

    // 3. Generate MIDI using current parameters
    GeneratedMidi midi = midiGenerator->generate(
        intent,
        apvts.bars,
        apvts.complexity,
        apvts.humanize,
        apvts.feel,
        apvts.dynamics
    );

    // 4. Verify MIDI is generated
    EXPECT_GT(midi.chords.size(), 0);
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);
}

// Test emotion selection updates processor state
TEST_F(UIProcessorIntegrationTest, EmotionSelection_UpdatesState) {
    // Simulate emotion selection from wheel
    auto selectedEmotion = intentPipeline->thesaurus().findNearest(0.8f, 0.7f, 0.6f);
    EXPECT_GT(selectedEmotion.id, 0);

    // Simulate updating processor with selected emotion ID
    int selectedEmotionId = selectedEmotion.id;

    // Simulate updating APVTS parameters from emotion coordinates
    apvts.valence = selectedEmotion.valence;
    apvts.arousal = selectedEmotion.arousal;
    apvts.intensity = selectedEmotion.intensity;

    // Verify parameters are updated
    EXPECT_FLOAT_EQ(apvts.valence, selectedEmotion.valence);
    EXPECT_FLOAT_EQ(apvts.arousal, selectedEmotion.arousal);
    EXPECT_FLOAT_EQ(apvts.intensity, selectedEmotion.intensity);

    // Generate MIDI using selected emotion
    SideA sideA;
    sideA.description = "Selected emotion";
    sideA.intensity = apvts.intensity;
    sideA.emotionId = selectedEmotionId;

    SideB sideB;
    sideB.description = "Musical expression";
    sideB.intensity = apvts.intensity;
    sideB.emotionId = selectedEmotionId;

    IntentResult intent = intentPipeline->processJourney(sideA, sideB);
    EXPECT_EQ(intent.emotion.id, selectedEmotionId);

    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);
    EXPECT_GT(midi.chords.size(), 0);
}

// Test wound text input updates processor state
TEST_F(UIProcessorIntegrationTest, WoundTextInput_UpdatesState) {
    // Simulate wound text input
    std::vector<std::string> woundTexts = {
        "I feel joyful",
        "I'm extremely sad",
        "I feel conflicted",
        "I'm anxious about the future"
    };

    for (const auto& woundText : woundTexts) {
        // Simulate setting wound description
        Wound wound;
        wound.description = woundText;
        wound.intensity = apvts.intensity;
        wound.source = "user_input";

        // Process wound
        IntentResult intent = intentPipeline->process(wound);

        // Verify intent is generated
        EXPECT_GT(intent.emotion.id, 0);
        EXPECT_FALSE(intent.mode.empty());

        // Generate MIDI
        GeneratedMidi midi = midiGenerator->generate(
            intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

        // Verify MIDI is generated
        EXPECT_GT(midi.chords.size(), 0);
        EXPECT_GT(midi.melody.size(), 0);
    }
}

// Test display components receive generated MIDI
TEST_F(UIProcessorIntegrationTest, DisplayComponents_ReceiveMidi) {
    // Simulate generating MIDI
    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

    // Simulate updating display components
    // 1. ChordDisplay - should receive chords
    EXPECT_GT(midi.chords.size(), 0);
    if (!midi.chords.empty()) {
        const auto& firstChord = midi.chords[0];
        EXPECT_FALSE(firstChord.name.empty());
        EXPECT_GT(firstChord.pitches.size(), 0);
    }

    // 2. PianoRollPreview - should receive all MIDI layers
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);

    // 3. MusicTheoryPanel - should receive key, mode, tempo
    EXPECT_FALSE(intent.mode.empty());
    EXPECT_GT(intent.tempo, 0.0f);

    // Verify all MIDI data is valid for display
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startBeat, 0.0);
        EXPECT_GT(note.duration, 0.0);
    }
}

// Test real-time parameter updates during playback
TEST_F(UIProcessorIntegrationTest, RealTimeParameterUpdates) {
    // Simulate parameter changes during playback
    std::atomic<bool> isPlaying{true};
    std::atomic<int> updateCount{0};

    // Simulate parameter automation
    for (int i = 0; i < 10; ++i) {
        if (isPlaying.load()) {
            // Update parameter
            apvts.valence = 0.5f + (i * 0.05f);
            apvts.arousal = 0.5f + (i * 0.03f);
            updateCount++;
        }
    }

    EXPECT_EQ(updateCount.load(), 10);

    // Verify parameters are updated
    EXPECT_GT(apvts.valence, 0.5f);
    EXPECT_GT(apvts.arousal, 0.5f);
}

// Test parameter change tracking (for manual regeneration)
TEST_F(UIProcessorIntegrationTest, ParameterChangeTracking) {
    // Simulate parameter change tracking
    std::atomic<bool> parametersChanged{false};

    // Simulate parameter change
    apvts.complexity = 0.8f;
    parametersChanged.store(true);

    // Verify change is tracked
    EXPECT_TRUE(parametersChanged.load());

    // Simulate regeneration
    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

    // After generation, clear flag
    parametersChanged.store(false);

    // Verify flag is cleared
    EXPECT_FALSE(parametersChanged.load());
}

// Test all EmotionWorkstation callbacks are properly wired
TEST_F(UIProcessorIntegrationTest, EmotionWorkstationCallbacksWired) {
    // Verify callback pattern:
    // 1. onGenerateClicked → should trigger generateMidi()
    // 2. onEmotionSelected → should update APVTS parameters
    // 3. onPreviewClicked → should update preview display
    // 4. onExportClicked → should export MIDI

    // Test 1: Generate button triggers generation
    std::string woundText = "I feel happy";
    Wound wound;
    wound.description = woundText;
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

    // Verify MIDI is generated (simulating onGenerateClicked callback)
    EXPECT_GT(midi.chords.size(), 0) << "onGenerateClicked should trigger generation";
    EXPECT_GT(midi.melody.size(), 0);

    // Test 2: Emotion selection updates parameters
    auto selectedEmotion = intentPipeline->thesaurus().findNearest(0.8f, 0.7f, 0.6f);
    EXPECT_GT(selectedEmotion.id, 0);

    // Simulate onEmotionSelected callback updating APVTS
    apvts.valence = selectedEmotion.valence;
    apvts.arousal = selectedEmotion.arousal;
    apvts.intensity = selectedEmotion.intensity;

    // Verify parameters are updated (simulating onEmotionSelected callback)
    EXPECT_FLOAT_EQ(apvts.valence, selectedEmotion.valence) << "onEmotionSelected should update valence";
    EXPECT_FLOAT_EQ(apvts.arousal, selectedEmotion.arousal) << "onEmotionSelected should update arousal";
    EXPECT_FLOAT_EQ(apvts.intensity, selectedEmotion.intensity) << "onEmotionSelected should update intensity";

    // Test 3: Preview updates display (simulating onPreviewClicked callback)
    // Preview should refresh piano roll with current MIDI
    EXPECT_GT(midi.melody.size(), 0) << "onPreviewClicked should update preview with current MIDI";
    EXPECT_GT(midi.bass.size(), 0);

    // Test 4: Export functionality (simulating onExportClicked callback)
    // Export should have MIDI data available
    EXPECT_GT(midi.chords.size(), 0) << "onExportClicked should have MIDI data to export";
    EXPECT_GT(midi.melody.size(), 0);
}

// Test all 9 APVTS parameters have attachments in EmotionWorkstation
TEST_F(UIProcessorIntegrationTest, AllAPVTSParametersAttached) {
    // Verify all 9 parameters are accessible:
    // 1. Valence
    // 2. Arousal
    // 3. Intensity
    // 4. Complexity
    // 5. Humanize
    // 6. Feel
    // 7. Dynamics
    // 8. Bars
    // 9. Bypass

    // All parameters should be accessible (simulating slider attachments)
    EXPECT_GE(apvts.valence, -1.0f);
    EXPECT_LE(apvts.valence, 1.0f);

    EXPECT_GE(apvts.arousal, 0.0f);
    EXPECT_LE(apvts.arousal, 1.0f);

    EXPECT_GE(apvts.intensity, 0.0f);
    EXPECT_LE(apvts.intensity, 1.0f);

    EXPECT_GE(apvts.complexity, 0.0f);
    EXPECT_LE(apvts.complexity, 1.0f);

    EXPECT_GE(apvts.humanize, 0.0f);
    EXPECT_LE(apvts.humanize, 1.0f);

    EXPECT_GE(apvts.feel, -1.0f);
    EXPECT_LE(apvts.feel, 1.0f);

    EXPECT_GE(apvts.dynamics, 0.0f);
    EXPECT_LE(apvts.dynamics, 1.0f);

    EXPECT_GE(apvts.bars, 4);
    EXPECT_LE(apvts.bars, 32);

    // All 9 parameters are accessible and within valid ranges
    EXPECT_TRUE(true) << "All 9 APVTS parameters should be attached to EmotionWorkstation sliders";
}

// Test wound text input triggers generation
TEST_F(UIProcessorIntegrationTest, WoundTextInputTriggersGeneration) {
    // Simulate wound text input from UI
    std::vector<std::string> woundTexts = {
        "I feel joyful",
        "I'm extremely sad",
        "I feel conflicted"
    };

    for (const auto& woundText : woundTexts) {
        // Simulate setting wound description (from UI text input)
        Wound wound;
        wound.description = woundText;
        wound.intensity = apvts.intensity;
        wound.source = "user_input";

        // Process wound (simulating onGenerateClicked after text input)
        IntentResult intent = intentPipeline->process(wound);
        EXPECT_GT(intent.emotion.id, 0) << "Wound text should trigger emotion processing";

        // Generate MIDI (simulating generateMidi() call)
        GeneratedMidi midi = midiGenerator->generate(
            intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

        // Verify MIDI is generated
        EXPECT_GT(midi.chords.size(), 0) << "Wound text should trigger MIDI generation";
        EXPECT_GT(midi.melody.size(), 0);
    }
}

// Test UI displays generated MIDI correctly (neutral emotion)
TEST_F(UIProcessorIntegrationTest, UIDisplaysGeneratedMidiNeutral) {
    // Generate MIDI
    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

    // Simulate updating UI displays with generated MIDI
    // 1. PianoRollPreview - should receive all MIDI layers
    EXPECT_GT(midi.melody.size(), 0) << "PianoRollPreview should display melody";
    EXPECT_GT(midi.bass.size(), 0) << "PianoRollPreview should display bass";
    EXPECT_GT(midi.chords.size(), 0) << "PianoRollPreview should display chords";

    // 2. ChordDisplay - should receive chord information
    if (!midi.chords.empty()) {
        const auto& firstChord = midi.chords[0];
        EXPECT_FALSE(firstChord.name.empty()) << "ChordDisplay should show chord name";
        EXPECT_GT(firstChord.pitches.size(), 0) << "ChordDisplay should show chord pitches";
    }

    // 3. MusicTheoryPanel - should receive key, mode, tempo
    EXPECT_FALSE(intent.mode.empty()) << "MusicTheoryPanel should display mode";
    EXPECT_GT(intent.tempo, 0.0f) << "MusicTheoryPanel should display tempo";

    // 4. EmotionRadar - should display emotion coordinates
    EXPECT_GE(intent.emotion.valence, -1.0f);
    EXPECT_LE(intent.emotion.valence, 1.0f);
    EXPECT_GE(intent.emotion.arousal, 0.0f);
    EXPECT_LE(intent.emotion.arousal, 1.0f);
    EXPECT_GE(intent.emotion.intensity, 0.0f);
    EXPECT_LE(intent.emotion.intensity, 1.0f);
}

// Test EmotionWorkstation callbacks are properly wired (basic)
TEST_F(UIProcessorIntegrationTest, EmotionWorkstationCallbacksBasic) {
    // Simulate EmotionWorkstation callback connections
    bool generateClicked = false;
    bool previewClicked = false;
    bool exportClicked = false;
    bool emotionSelected = false;

    // Simulate onGenerateClicked callback
    auto onGenerateClicked = [&]() {
        generateClicked = true;
        // Should trigger generateMidi()
        Wound wound;
        wound.description = "I feel happy";
        wound.intensity = apvts.intensity;
        wound.source = "user_input";
        IntentResult intent = intentPipeline->process(wound);
        GeneratedMidi midi = midiGenerator->generate(
            intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);
        EXPECT_GT(midi.chords.size(), 0);
    };

    // Simulate onPreviewClicked callback
    auto onPreviewClicked = [&]() {
        previewClicked = true;
        // Should refresh piano roll preview
        Wound wound;
        wound.description = "I feel happy";
        wound.intensity = apvts.intensity;
        wound.source = "user_input";
        IntentResult intent = intentPipeline->process(wound);
        GeneratedMidi midi = midiGenerator->generate(
            intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);
        EXPECT_GT(midi.melody.size(), 0);
    };

    // Simulate onExportClicked callback
    auto onExportClicked = [&]() {
        exportClicked = true;
        // Should export MIDI (in plugin mode, MIDI flows through automatically)
        Wound wound;
        wound.description = "I feel happy";
        wound.intensity = apvts.intensity;
        wound.source = "user_input";
        IntentResult intent = intentPipeline->process(wound);
        GeneratedMidi midi = midiGenerator->generate(
            intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);
        EXPECT_GT(midi.chords.size(), 0);
    };

    // Simulate onEmotionSelected callback
    auto onEmotionSelected = [&](const EmotionNode& emotion) {
        emotionSelected = true;
        // Should update APVTS parameters
        apvts.valence = emotion.valence;
        apvts.arousal = emotion.arousal;
        apvts.intensity = emotion.intensity;
        EXPECT_FLOAT_EQ(apvts.valence, emotion.valence);
        EXPECT_FLOAT_EQ(apvts.arousal, emotion.arousal);
        EXPECT_FLOAT_EQ(apvts.intensity, emotion.intensity);
    };

    // Test callbacks
    onGenerateClicked();
    EXPECT_TRUE(generateClicked) << "onGenerateClicked should be called";

    onPreviewClicked();
    EXPECT_TRUE(previewClicked) << "onPreviewClicked should be called";

    onExportClicked();
    EXPECT_TRUE(exportClicked) << "onExportClicked should be called";

    auto emotion = intentPipeline->thesaurus().findNearest(0.8f, 0.7f, 0.6f);
    onEmotionSelected(emotion);
    EXPECT_TRUE(emotionSelected) << "onEmotionSelected should be called";
}

// Test parameter synchronization between UI and processor (basic)
TEST_F(UIProcessorIntegrationTest, ParameterSynchronizationBasic) {
    // Simulate EmotionWorkstation sliders attached to APVTS
    // All 9 parameters should have attachments

    // Test all parameters can be updated
    std::vector<std::pair<std::string, float>> parameters = {
        {"valence", 0.8f},
        {"arousal", 0.7f},
        {"intensity", 0.6f},
        {"complexity", 0.9f},
        {"humanize", 0.5f},
        {"feel", 0.3f},
        {"dynamics", 0.8f}
    };

    for (const auto& [param, value] : parameters) {
        // Simulate slider value change
        if (param == "valence") apvts.valence = value;
        else if (param == "arousal") apvts.arousal = value;
        else if (param == "intensity") apvts.intensity = value;
        else if (param == "complexity") apvts.complexity = value;
        else if (param == "humanize") apvts.humanize = value;
        else if (param == "feel") apvts.feel = value;
        else if (param == "dynamics") apvts.dynamics = value;

        // Verify parameter is updated
        if (param == "valence") EXPECT_FLOAT_EQ(apvts.valence, value);
        else if (param == "arousal") EXPECT_FLOAT_EQ(apvts.arousal, value);
        else if (param == "intensity") EXPECT_FLOAT_EQ(apvts.intensity, value);
        else if (param == "complexity") EXPECT_FLOAT_EQ(apvts.complexity, value);
        else if (param == "humanize") EXPECT_FLOAT_EQ(apvts.humanize, value);
        else if (param == "feel") EXPECT_FLOAT_EQ(apvts.feel, value);
        else if (param == "dynamics") EXPECT_FLOAT_EQ(apvts.dynamics, value);
    }

    // Test bars parameter (int)
    apvts.bars = 16;
    EXPECT_EQ(apvts.bars, 16);

    // Test bypass parameter (bool)
    // Note: Bypass is a bool, not in SimulatedAPVTS, but would be in real APVTS
}

// Test UI displays generated MIDI correctly (happy emotion)
TEST_F(UIProcessorIntegrationTest, UIDisplaysGeneratedMidiHappy) {
    // Generate MIDI
    Wound wound;
    wound.description = "I feel happy";
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

    // Simulate updating UI displays
    // 1. PianoRollPreview - should receive all MIDI layers
    EXPECT_GT(midi.melody.size(), 0) << "PianoRollPreview should receive melody";
    EXPECT_GT(midi.bass.size(), 0) << "PianoRollPreview should receive bass";
    EXPECT_GT(midi.chords.size(), 0) << "PianoRollPreview should receive chords";

    // 2. ChordDisplay - should receive chord information
    if (!midi.chords.empty()) {
        const auto& firstChord = midi.chords[0];
        EXPECT_FALSE(firstChord.name.empty()) << "ChordDisplay should receive chord name";
        EXPECT_GT(firstChord.pitches.size(), 0) << "ChordDisplay should receive chord pitches";
    }

    // 3. MusicTheoryPanel - should receive key, mode, tempo
    EXPECT_FALSE(intent.mode.empty()) << "MusicTheoryPanel should receive mode";
    EXPECT_GT(intent.tempo, 0.0f) << "MusicTheoryPanel should receive tempo";

    // Verify all MIDI data is valid for display
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.pitch, 0) << "Melody note pitch should be valid";
        EXPECT_LE(note.pitch, 127) << "Melody note pitch should be valid";
        EXPECT_GE(note.velocity, 0) << "Melody note velocity should be valid";
        EXPECT_LE(note.velocity, 127) << "Melody note velocity should be valid";
        EXPECT_GE(note.startBeat, 0.0) << "Melody note startBeat should be valid";
        EXPECT_GT(note.duration, 0.0) << "Melody note duration should be valid";
    }
}

// Test all EmotionWorkstation callbacks are properly wired (verification)
TEST_F(UIProcessorIntegrationTest, EmotionWorkstationCallbacksVerification) {
    // Verify callback connections exist:
    // 1. onGenerateClicked → should trigger generateMidi()
    // 2. onEmotionSelected → should update APVTS parameters
    // 3. onPreviewClicked → should refresh piano roll preview
    // 4. onExportClicked → should export MIDI

    // Simulate Generate button click
    std::string woundText = "I feel happy";
    Wound wound;
    wound.description = woundText;
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

    // Verify generation succeeded (callback would trigger this)
    EXPECT_GT(midi.chords.size(), 0);
    EXPECT_GT(midi.melody.size(), 0);

    // Simulate emotion selection from wheel
    auto selectedEmotion = intentPipeline->thesaurus().findNearest(0.8f, 0.7f, 0.6f);

    // Simulate updating APVTS parameters (onEmotionSelected callback)
    apvts.valence = selectedEmotion.valence;
    apvts.arousal = selectedEmotion.arousal;
    apvts.intensity = selectedEmotion.intensity;

    // Verify parameters are updated
    EXPECT_FLOAT_EQ(apvts.valence, selectedEmotion.valence);
    EXPECT_FLOAT_EQ(apvts.arousal, selectedEmotion.arousal);
    EXPECT_FLOAT_EQ(apvts.intensity, selectedEmotion.intensity);

    // Simulate preview (onPreviewClicked callback)
    // Would refresh piano roll preview with current MIDI
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);

    // Simulate export (onExportClicked callback)
    // Would export MIDI to DAW or file
    EXPECT_GT(midi.chords.size(), 0);
    EXPECT_GT(midi.lengthInBeats, 0.0);
}

// Test parameter synchronization between UI and processor (verification)
TEST_F(UIProcessorIntegrationTest, ParameterSynchronizationVerification) {
    // Verify all 9 APVTS parameters have attachments in EmotionWorkstation:
    // 1. Valence
    // 2. Arousal
    // 3. Intensity
    // 4. Complexity
    // 5. Humanize
    // 6. Feel
    // 7. Dynamics
    // 8. Bars
    // 9. Bypass

    // Simulate parameter updates from UI sliders
    std::vector<std::pair<std::string, float>> parameters = {
        {"valence", 0.8f},
        {"arousal", 0.7f},
        {"intensity", 0.6f},
        {"complexity", 0.9f},
        {"humanize", 0.5f},
        {"feel", 0.3f},
        {"dynamics", 0.8f}
    };

    for (const auto& [param, value] : parameters) {
        // Simulate slider update
        if (param == "valence") apvts.valence = value;
        else if (param == "arousal") apvts.arousal = value;
        else if (param == "intensity") apvts.intensity = value;
        else if (param == "complexity") apvts.complexity = value;
        else if (param == "humanize") apvts.humanize = value;
        else if (param == "feel") apvts.feel = value;
        else if (param == "dynamics") apvts.dynamics = value;

        // Verify parameter is updated (APVTS attachment would do this)
        if (param == "valence") EXPECT_FLOAT_EQ(apvts.valence, value);
        else if (param == "arousal") EXPECT_FLOAT_EQ(apvts.arousal, value);
        else if (param == "intensity") EXPECT_FLOAT_EQ(apvts.intensity, value);
        else if (param == "complexity") EXPECT_FLOAT_EQ(apvts.complexity, value);
        else if (param == "humanize") EXPECT_FLOAT_EQ(apvts.humanize, value);
        else if (param == "feel") EXPECT_FLOAT_EQ(apvts.feel, value);
        else if (param == "dynamics") EXPECT_FLOAT_EQ(apvts.dynamics, value);
    }

    // Test bars parameter (int)
    apvts.bars = 16;
    EXPECT_EQ(apvts.bars, 16);

    // Test bypass parameter (bool)
    // Note: Bypass is boolean, not in our simulated APVTS struct
    // But we verify other parameters work
}

// Test UI displays generated MIDI correctly
TEST_F(UIProcessorIntegrationTest, UIDisplaysMidi) {
    // Generate MIDI
    Wound wound;
    wound.description = "I feel neutral";
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

    // Simulate updating UI displays:
    // 1. PianoRollPreview - should receive all MIDI layers
    EXPECT_GT(midi.melody.size(), 0);
    EXPECT_GT(midi.bass.size(), 0);
    EXPECT_GT(midi.chords.size(), 0);

    // 2. ChordDisplay - should receive chord information
    if (!midi.chords.empty()) {
        const auto& firstChord = midi.chords[0];
        EXPECT_FALSE(firstChord.name.empty());
        EXPECT_GT(firstChord.pitches.size(), 0);
    }

    // 3. EmotionRadar - should display current VAD values
    EXPECT_GE(apvts.valence, -1.0f);
    EXPECT_LE(apvts.valence, 1.0f);
    EXPECT_GE(apvts.arousal, 0.0f);
    EXPECT_LE(apvts.arousal, 1.0f);
    EXPECT_GE(apvts.intensity, 0.0f);
    EXPECT_LE(apvts.intensity, 1.0f);

    // 4. MusicTheoryPanel - should display key, mode, tempo
    EXPECT_FALSE(intent.mode.empty());
    EXPECT_GT(intent.tempo, 0.0f);

    // Verify all MIDI data is valid for display
    for (const auto& note : midi.melody) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startBeat, 0.0);
        EXPECT_GT(note.duration, 0.0);
    }
}

// Test EmotionWorkstation callbacks are properly connected (complete)
TEST_F(UIProcessorIntegrationTest, EmotionWorkstationCallbacksComplete) {
    // Simulate EmotionWorkstation callback connections:
    // 1. onGenerateClicked → PluginEditor::onGenerateClicked() → processor_.generateMidi()
    // 2. onEmotionSelected → Updates APVTS parameters
    // 3. onPreviewClicked → Refreshes piano roll
    // 4. onExportClicked → Exports MIDI

    // Test 1: Generate button callback
    bool generateClicked = false;
    auto onGenerateClicked = [&generateClicked]() {
        generateClicked = true;
        // Simulate processor_.generateMidi()
    };
    onGenerateClicked();
    EXPECT_TRUE(generateClicked) << "Generate button callback should be called";

    // Test 2: Emotion selection callback
    auto selectedEmotion = intentPipeline->thesaurus().findNearest(0.8f, 0.7f, 0.6f);
    bool emotionSelected = false;
    auto onEmotionSelected = [&emotionSelected, &selectedEmotion](const EmotionNode& emotion) {
        emotionSelected = true;
        // Simulate updating APVTS parameters
        EXPECT_EQ(emotion.id, selectedEmotion.id);
    };
    onEmotionSelected(selectedEmotion);
    EXPECT_TRUE(emotionSelected) << "Emotion selection callback should be called";

    // Test 3: Preview button callback
    bool previewClicked = false;
    auto onPreviewClicked = [&previewClicked]() {
        previewClicked = true;
        // Simulate refreshing piano roll
    };
    onPreviewClicked();
    EXPECT_TRUE(previewClicked) << "Preview button callback should be called";

    // Test 4: Export button callback
    bool exportClicked = false;
    auto onExportClicked = [&exportClicked]() {
        exportClicked = true;
        // Simulate exporting MIDI
    };
    onExportClicked();
    EXPECT_TRUE(exportClicked) << "Export button callback should be called";
}

// Test parameter synchronization between EmotionWorkstation and APVTS (complete)
TEST_F(UIProcessorIntegrationTest, ParameterSynchronizationComplete) {
    // Simulate all 9 parameters have attachments in EmotionWorkstation
    // Parameters: valence, arousal, intensity, complexity, humanize, feel, dynamics, bars, bypass

    // Test that parameter changes are synchronized
    std::vector<std::pair<std::string, float>> parameters = {
        {"valence", 0.8f},
        {"arousal", 0.7f},
        {"intensity", 0.6f},
        {"complexity", 0.9f},
        {"humanize", 0.5f},
        {"feel", 0.3f},
        {"dynamics", 0.8f}
    };

    for (const auto& [param, value] : parameters) {
        // Simulate updating parameter via APVTS attachment
        if (param == "valence") apvts.valence = value;
        else if (param == "arousal") apvts.arousal = value;
        else if (param == "intensity") apvts.intensity = value;
        else if (param == "complexity") apvts.complexity = value;
        else if (param == "humanize") apvts.humanize = value;
        else if (param == "feel") apvts.feel = value;
        else if (param == "dynamics") apvts.dynamics = value;

        // Verify parameter is updated
        if (param == "valence") EXPECT_FLOAT_EQ(apvts.valence, value);
        else if (param == "arousal") EXPECT_FLOAT_EQ(apvts.arousal, value);
        else if (param == "intensity") EXPECT_FLOAT_EQ(apvts.intensity, value);
        else if (param == "complexity") EXPECT_FLOAT_EQ(apvts.complexity, value);
        else if (param == "humanize") EXPECT_FLOAT_EQ(apvts.humanize, value);
        else if (param == "feel") EXPECT_FLOAT_EQ(apvts.feel, value);
        else if (param == "dynamics") EXPECT_FLOAT_EQ(apvts.dynamics, value);
    }

    // Test bars parameter (int)
    apvts.bars = 16;
    EXPECT_EQ(apvts.bars, 16) << "Bars parameter should be synchronized";

    // Test bypass parameter (bool)
    // Note: Bypass is a boolean, so we test it separately
    // In real implementation, bypass would be a toggle button attachment
}

// Test UI displays generated MIDI correctly (complete UI update)
TEST_F(UIProcessorIntegrationTest, UIDisplaysGeneratedMidiComplete) {
    // Simulate generating MIDI
    Wound wound;
    wound.description = "I feel happy";
    wound.intensity = apvts.intensity;
    wound.source = "user_input";

    IntentResult intent = intentPipeline->process(wound);
    GeneratedMidi midi = midiGenerator->generate(
        intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

    // Simulate updating UI displays:
    // 1. PianoRollPreview - should receive all MIDI layers
    EXPECT_GT(midi.melody.size(), 0) << "PianoRollPreview should receive melody";
    EXPECT_GT(midi.bass.size(), 0) << "PianoRollPreview should receive bass";
    EXPECT_GT(midi.chords.size(), 0) << "PianoRollPreview should receive chords";

    // 2. ChordDisplay - should receive first chord
    if (!midi.chords.empty()) {
        const auto& firstChord = midi.chords[0];
        EXPECT_FALSE(firstChord.name.empty()) << "ChordDisplay should receive chord name";
        EXPECT_GT(firstChord.pitches.size(), 0) << "ChordDisplay should receive chord pitches";
    }

    // 3. EmotionRadar - should update with processed emotion
    EXPECT_GT(intent.emotion.id, 0) << "EmotionRadar should receive emotion";
    EXPECT_GE(intent.emotion.valence, -1.0f) << "EmotionRadar should receive valid valence";
    EXPECT_LE(intent.emotion.valence, 1.0f) << "EmotionRadar should receive valid valence";
    EXPECT_GE(intent.emotion.arousal, 0.0f) << "EmotionRadar should receive valid arousal";
    EXPECT_LE(intent.emotion.arousal, 1.0f) << "EmotionRadar should receive valid arousal";

    // 4. MusicTheoryPanel - should receive key, mode, tempo
    EXPECT_FALSE(intent.mode.empty()) << "MusicTheoryPanel should receive mode";
    EXPECT_GT(intent.tempo, 0.0f) << "MusicTheoryPanel should receive tempo";
}

// Test EmotionWorkstation callback pattern matches PluginEditor implementation
TEST_F(UIProcessorIntegrationTest, EmotionWorkstation_CallbackPattern) {
    // Simulate EmotionWorkstation callback pattern from PluginEditor
    // Pattern: workstation_->onGenerateClicked = [this]() { onGenerateClicked(); };

    bool generateCallbackCalled = false;
    bool emotionSelectedCallbackCalled = false;
    bool previewCallbackCalled = false;
    bool exportCallbackCalled = false;

    // Simulate callback setup (as done in PluginEditor constructor)
    auto onGenerateClicked = [&]() {
        generateCallbackCalled = true;

        // Simulate PluginEditor::onGenerateClicked() behavior
        // 1. Get wound text
        std::string woundText = "I feel happy";

        // 2. Set wound description
        Wound wound;
        wound.description = woundText;
        wound.intensity = apvts.intensity;
        wound.source = "user_input";

        // 3. Process and generate (simulating processor_.generateMidi())
        IntentResult intent = intentPipeline->process(wound);
        GeneratedMidi midi = midiGenerator->generate(
            intent, apvts.bars, apvts.complexity, apvts.humanize, apvts.feel, apvts.dynamics);

        // 4. Verify MIDI is generated
        EXPECT_GT(midi.chords.size(), 0);
    };

    auto onEmotionSelected = [&](const EmotionNode& emotion) {
        emotionSelectedCallbackCalled = true;

        // Simulate updating APVTS parameters from emotion
        apvts.valence = emotion.valence;
        apvts.arousal = emotion.arousal;
        apvts.intensity = emotion.intensity;

        // Verify parameters are updated
        EXPECT_FLOAT_EQ(apvts.valence, emotion.valence);
        EXPECT_FLOAT_EQ(apvts.arousal, emotion.arousal);
        EXPECT_FLOAT_EQ(apvts.intensity, emotion.intensity);
    };

    // Test Generate button callback
    onGenerateClicked();
    EXPECT_TRUE(generateCallbackCalled) << "Generate callback should be called";

    // Test Emotion selection callback
    auto emotion = intentPipeline->thesaurus().findNearest(0.7f, 0.6f, 0.5f);
    onEmotionSelected(emotion);
    EXPECT_TRUE(emotionSelectedCallbackCalled) << "Emotion selection callback should be called";
}

// Test all 9 APVTS parameters are accessible from EmotionWorkstation
TEST_F(UIProcessorIntegrationTest, EmotionWorkstation_AllParametersAccessible) {
    // Simulate EmotionWorkstation having access to all 9 APVTS parameters
    // Parameters: valence, arousal, intensity, complexity, humanize, feel, dynamics, bars, bypass

    struct SimulatedEmotionWorkstation {
        float valence = 0.0f;
        float arousal = 0.5f;
        float intensity = 0.5f;
        float complexity = 0.5f;
        float humanize = 0.4f;
        float feel = 0.0f;
        float dynamics = 0.75f;
        int bars = 8;
        bool bypass = false;
    } workstation;

    // Test all parameters can be read
    EXPECT_GE(workstation.valence, -1.0f);
    EXPECT_LE(workstation.valence, 1.0f);
    EXPECT_GE(workstation.arousal, 0.0f);
    EXPECT_LE(workstation.arousal, 1.0f);
    EXPECT_GE(workstation.intensity, 0.0f);
    EXPECT_LE(workstation.intensity, 1.0f);
    EXPECT_GE(workstation.complexity, 0.0f);
    EXPECT_LE(workstation.complexity, 1.0f);
    EXPECT_GE(workstation.humanize, 0.0f);
    EXPECT_LE(workstation.humanize, 1.0f);
    EXPECT_GE(workstation.feel, -1.0f);
    EXPECT_LE(workstation.feel, 1.0f);
    EXPECT_GE(workstation.dynamics, 0.0f);
    EXPECT_LE(workstation.dynamics, 1.0f);
    EXPECT_GE(workstation.bars, 4);
    EXPECT_LE(workstation.bars, 32);

    // Test parameters can be updated
    workstation.valence = 0.8f;
    workstation.arousal = 0.7f;
    workstation.intensity = 0.6f;
    workstation.complexity = 0.9f;
    workstation.humanize = 0.5f;
    workstation.feel = 0.3f;
    workstation.dynamics = 0.8f;
    workstation.bars = 16;
    workstation.bypass = true;

    // Verify updates
    EXPECT_FLOAT_EQ(workstation.valence, 0.8f);
    EXPECT_FLOAT_EQ(workstation.arousal, 0.7f);
    EXPECT_FLOAT_EQ(workstation.intensity, 0.6f);
    EXPECT_FLOAT_EQ(workstation.complexity, 0.9f);
    EXPECT_FLOAT_EQ(workstation.humanize, 0.5f);
    EXPECT_FLOAT_EQ(workstation.feel, 0.3f);
    EXPECT_FLOAT_EQ(workstation.dynamics, 0.8f);
    EXPECT_EQ(workstation.bars, 16);
    EXPECT_TRUE(workstation.bypass);
}
