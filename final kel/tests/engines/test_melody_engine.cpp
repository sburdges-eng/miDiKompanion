#include <gtest/gtest.h>
#include "engines/MelodyEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class MelodyEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<MelodyEngine>();
    }

    std::unique_ptr<MelodyEngine> engine;
};

// Test basic generation
TEST_F(MelodyEngineTest, BasicGeneration) {
    MelodyOutput output = engine->generate("joy", "C", "major", 4, 120);

    EXPECT_GT(output.notes.size(), 0);
    EXPECT_EQ(output.emotion, "joy");
    EXPECT_EQ(output.totalTicks, 7680); // 4 bars * 4 beats * 480 ticks/beat at 120 BPM
}

// Test different emotions
TEST_F(MelodyEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear", "neutral"};

    for (const auto& emotion : emotions) {
        MelodyOutput output = engine->generate(emotion, "C", "major", 2, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for emotion: " << emotion;
        EXPECT_EQ(output.emotion, emotion);
    }
}

// Test different keys
TEST_F(MelodyEngineTest, DifferentKeys) {
    std::vector<std::string> keys = {"C", "D", "E", "F", "G", "A", "B"};

    for (const auto& key : keys) {
        MelodyOutput output = engine->generate("neutral", key, "major", 2, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for key: " << key;
    }
}

// Test different modes
TEST_F(MelodyEngineTest, DifferentModes) {
    std::vector<std::string> modes = {"major", "minor", "dorian", "mixolydian", "aeolian"};

    for (const auto& mode : modes) {
        MelodyOutput output = engine->generate("neutral", "C", mode, 2, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for mode: " << mode;
    }
}

// Test contour override
TEST_F(MelodyEngineTest, ContourOverride) {
    MelodyConfig config;
    config.emotion = "joy";
    config.key = "C";
    config.mode = "major";
    config.bars = 4;
    config.tempoBpm = 120;
    config.contourOverride = ContourType::Ascending;

    MelodyOutput output = engine->generate(config);
    EXPECT_EQ(output.contourUsed, ContourType::Ascending);
}

// Test density override
TEST_F(MelodyEngineTest, DensityOverride) {
    MelodyConfig config;
    config.emotion = "joy";
    config.key = "C";
    config.mode = "major";
    config.bars = 4;
    config.tempoBpm = 120;
    config.densityOverride = RhythmDensity::Dense;

    MelodyOutput output = engine->generate(config);
    EXPECT_EQ(output.densityUsed, RhythmDensity::Dense);
}

// Test seed reproducibility
TEST_F(MelodyEngineTest, SeedReproducibility) {
    MelodyConfig config1;
    config1.emotion = "neutral";
    config1.key = "C";
    config1.mode = "major";
    config1.bars = 2;
    config1.tempoBpm = 120;
    config1.seed = 42;

    MelodyConfig config2 = config1;

    MelodyOutput output1 = engine->generate(config1);
    MelodyOutput output2 = engine->generate(config2);

    EXPECT_EQ(output1.notes.size(), output2.notes.size());
    // With same seed, should produce same results
    for (size_t i = 0; i < output1.notes.size(); ++i) {
        EXPECT_EQ(output1.notes[i].pitch, output2.notes[i].pitch);
        EXPECT_EQ(output1.notes[i].startTick, output2.notes[i].startTick);
    }
}

// Test section-specific generation
TEST_F(MelodyEngineTest, SectionGeneration) {
    MelodyOutput output = engine->generateForSection("joy", "chorus", "C", 4, 120);

    EXPECT_GT(output.notes.size(), 0);
    EXPECT_EQ(output.emotion, "joy");
}

// Test note pitch validity
TEST_F(MelodyEngineTest, NotePitchValidity) {
    MelodyOutput output = engine->generate("neutral", "C", "major", 4, 120);

    for (const auto& note : output.notes) {
        EXPECT_GE(note.pitch, 0);
        EXPECT_LE(note.pitch, 127);
        EXPECT_GT(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startTick, 0);
        EXPECT_GT(note.durationTicks, 0);
    }
}

// Test note ordering
TEST_F(MelodyEngineTest, NoteOrdering) {
    MelodyOutput output = engine->generate("neutral", "C", "major", 4, 120);

    for (size_t i = 1; i < output.notes.size(); ++i) {
        EXPECT_GE(output.notes[i].startTick, output.notes[i-1].startTick);
    }
}

// Test different bar counts
TEST_F(MelodyEngineTest, DifferentBarCounts) {
    for (int bars : {1, 2, 4, 8, 16}) {
        MelodyOutput output = engine->generate("neutral", "C", "major", bars, 120);
        EXPECT_GT(output.notes.size(), 0) << "Failed for bars: " << bars;
        EXPECT_EQ(output.totalTicks, bars * 4 * 480); // 4 beats/bar, 480 ticks/beat
    }
}

// Test different tempos
TEST_F(MelodyEngineTest, DifferentTempos) {
    std::vector<int> tempos = {60, 90, 120, 150, 180};

    for (int tempo : tempos) {
        MelodyOutput output = engine->generate("neutral", "C", "major", 4, tempo);
        EXPECT_GT(output.notes.size(), 0) << "Failed for tempo: " << tempo;
    }
}

// Test GM instrument assignment
TEST_F(MelodyEngineTest, GMInstrumentAssignment) {
    MelodyOutput output = engine->generate("joy", "C", "major", 2, 120);

    EXPECT_GE(output.gmInstrument, 0);
    EXPECT_LE(output.gmInstrument, 127);
}

// Test all contour types
TEST_F(MelodyEngineTest, AllContourTypes) {
    std::vector<ContourType> contours = {
        ContourType::Ascending,
        ContourType::Descending,
        ContourType::Arch,
        ContourType::InverseArch,
        ContourType::Static,
        ContourType::Wave,
        ContourType::SpiralDown,
        ContourType::SpiralUp,
        ContourType::Jagged,
        ContourType::Collapse
    };

    for (const auto& contour : contours) {
        MelodyConfig config;
        config.emotion = "neutral";
        config.key = "C";
        config.mode = "major";
        config.bars = 4;
        config.tempoBpm = 120;
        config.contourOverride = contour;

        MelodyOutput output = engine->generate(config);
        EXPECT_EQ(output.contourUsed, contour) << "Contour should match override";
        EXPECT_GT(output.notes.size(), 0) << "Should generate notes for contour";

        // Verify all notes are valid
        for (const auto& note : output.notes) {
            EXPECT_GE(note.pitch, 0);
            EXPECT_LE(note.pitch, 127);
            EXPECT_GT(note.velocity, 0);
            EXPECT_LE(note.velocity, 127);
            EXPECT_GE(note.startTick, 0);
            EXPECT_GT(note.durationTicks, 0);
        }
    }
}

// Test all rhythm density types
TEST_F(MelodyEngineTest, AllRhythmDensityTypes) {
    std::vector<RhythmDensity> densities = {
        RhythmDensity::Sparse,
        RhythmDensity::Moderate,
        RhythmDensity::Dense,
        RhythmDensity::Frantic
    };

    for (const auto& density : densities) {
        MelodyConfig config;
        config.emotion = "neutral";
        config.key = "C";
        config.mode = "major";
        config.bars = 4;
        config.tempoBpm = 120;
        config.densityOverride = density;

        MelodyOutput output = engine->generate(config);
        EXPECT_EQ(output.densityUsed, density) << "Density should match override";
        EXPECT_GT(output.notes.size(), 0) << "Should generate notes for density";
    }
}
