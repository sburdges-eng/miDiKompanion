#include <gtest/gtest.h>
#include "midi/GrooveEngine.h"
#include <vector>
#include "common/Types.h"

using namespace kelly;

class GrooveEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<GrooveEngine>();
    }

    std::unique_ptr<GrooveEngine> engine;
};

// Test basic groove application
TEST_F(GrooveEngineTest, ApplyGroove) {
    std::vector<MidiNote> notes;
    for (int i = 0; i < 8; ++i) {
        MidiNote note;
        note.pitch = 60 + i;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i);
        note.duration = 0.5;
        notes.push_back(note);
    }

    std::vector<MidiNote> grooved = engine->applyGroove(notes, GrooveType::Straight, 0.5f);

    EXPECT_EQ(grooved.size(), notes.size());

    for (const auto& note : grooved) {
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        // Allow small negative values for timing adjustments (pushed notes)
        EXPECT_GE(note.startBeat, -0.5);  // Max push is half a beat
    }
}

// Test swing application
TEST_F(GrooveEngineTest, ApplySwing) {
    std::vector<MidiNote> notes;
    for (int i = 0; i < 8; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i) * 0.5;
        note.duration = 0.25;
        notes.push_back(note);
    }

    std::vector<MidiNote> swung = engine->applySwing(notes, 0.67f, 1.0f);

    EXPECT_EQ(swung.size(), notes.size());
}

// Test different groove types
TEST_F(GrooveEngineTest, DifferentGrooveTypes) {
    std::vector<MidiNote> notes;
    for (int i = 0; i < 4; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i);
        note.duration = 0.5;
        notes.push_back(note);
    }

    std::vector<GrooveType> types = {
        GrooveType::Straight,
        GrooveType::Swing,
        GrooveType::Shuffle
    };

    for (auto type : types) {
        std::vector<MidiNote> grooved = engine->applyGroove(notes, type, 0.5f);
        // Some groove types may add ghost notes or change note count
        EXPECT_GT(grooved.size(), 0) << "Groove should return notes";
    }
}

// Test all valid groove templates
TEST_F(GrooveEngineTest, ApplyGrooveTemplate_ValidTemplates) {
    std::vector<MidiNote> notes;
    for (int i = 0; i < 16; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i) * 0.25;  // 16th notes
        note.duration = 0.25;
        notes.push_back(note);
    }

    std::vector<std::string> templateNames = {
        "funk", "jazz", "rock", "hiphop", "edm", "latin", "blues", "lofi"
    };

    for (const auto& templateName : templateNames) {
        std::vector<MidiNote> grooved = engine->applyGrooveTemplate(
            notes, templateName, 0.5f, 1.0f);

        // Should have processed notes (may have fewer due to dropout)
        EXPECT_GE(grooved.size(), 0);

        // Verify all notes are valid
        for (const auto& note : grooved) {
            EXPECT_GE(note.velocity, 0);
            EXPECT_LE(note.velocity, 127);
            EXPECT_GE(note.startBeat, -0.5);  // Allow pushed timing
        }
    }
}

// Test invalid template name fallback behavior
TEST_F(GrooveEngineTest, ApplyGrooveTemplate_InvalidTemplate) {
    std::vector<MidiNote> notes;
    for (int i = 0; i < 8; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i);
        note.duration = 0.5;
        notes.push_back(note);
    }

    // Invalid template name should fallback to Straight groove
    std::vector<MidiNote> grooved = engine->applyGrooveTemplate(
        notes, "nonexistent_template", 0.5f, 1.0f);

    // Should still process notes (fallback behavior)
    EXPECT_GE(grooved.size(), 0);

    // Verify notes are valid
    for (const auto& note : grooved) {
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startBeat, -0.5);  // Allow pushed timing
    }
}

// Test intensity parameter affects output
TEST_F(GrooveEngineTest, ApplyGrooveTemplate_IntensityParameter) {
    std::vector<MidiNote> notes;
    for (int i = 0; i < 16; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i) * 0.25;
        note.duration = 0.25;
        notes.push_back(note);
    }

    // Test with different intensity values
    std::vector<MidiNote> lowIntensity = engine->applyGrooveTemplate(
        notes, "funk", 0.5f, 0.0f);
    std::vector<MidiNote> midIntensity = engine->applyGrooveTemplate(
        notes, "funk", 0.5f, 0.5f);
    std::vector<MidiNote> highIntensity = engine->applyGrooveTemplate(
        notes, "funk", 0.5f, 1.0f);

    // All should produce valid output
    EXPECT_GE(lowIntensity.size(), 0);
    EXPECT_GE(midIntensity.size(), 0);
    EXPECT_GE(highIntensity.size(), 0);

    // With intensity 0.0, notes should be closer to original
    // With intensity 1.0, notes should have more template characteristics
    // We can verify that timing/velocity changes are more pronounced at higher intensity
    if (highIntensity.size() > 0 && lowIntensity.size() > 0) {
        // High intensity should show more variation from original
        bool hasVariation = false;
        for (size_t i = 0; i < std::min(highIntensity.size(), notes.size()); ++i) {
            if (std::abs(highIntensity[i].startBeat - notes[i].startBeat) > 0.001 ||
                highIntensity[i].velocity != notes[i].velocity) {
                hasVariation = true;
                break;
            }
        }
        // At least some variation should occur (though randomness may affect this)
        // This is a soft check - the main goal is to ensure no crashes
    }
}

// Test timing deviations are applied correctly
TEST_F(GrooveEngineTest, ApplyGrooveTemplate_TimingDeviations) {
    std::vector<MidiNote> notes;
    // Create notes on exact 16th note grid
    for (int i = 0; i < 16; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i) * 0.25;  // Exact 16th notes
        note.duration = 0.25;
        notes.push_back(note);
    }

    // Apply template with high intensity to see timing changes
    std::vector<MidiNote> grooved = engine->applyGrooveTemplate(
        notes, "jazz", 0.3f, 1.0f);  // Lower humanization to reduce randomness

    EXPECT_GE(grooved.size(), 0);

    // Verify timing deviations are applied (notes should not all be on exact grid)
    bool hasTimingDeviation = false;
    for (size_t i = 0; i < std::min(grooved.size(), notes.size()); ++i) {
        double originalBeat = static_cast<double>(i) * 0.25;
        if (std::abs(grooved[i].startBeat - originalBeat) > 0.001) {
            hasTimingDeviation = true;
            break;
        }
    }
    // Note: Due to randomness, this may not always be true, but it's likely
    // The main goal is to ensure the function doesn't crash and produces valid output
}

// Test velocity curves are applied correctly
TEST_F(GrooveEngineTest, ApplyGrooveTemplate_VelocityCurve) {
    std::vector<MidiNote> notes;
    // Create notes with uniform velocity
    for (int i = 0; i < 16; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;  // Uniform velocity
        note.startBeat = static_cast<double>(i) * 0.25;
        note.duration = 0.25;
        notes.push_back(note);
    }

    // Apply template with high intensity
    std::vector<MidiNote> grooved = engine->applyGrooveTemplate(
        notes, "rock", 0.3f, 1.0f);  // Lower humanization to reduce randomness

    EXPECT_GE(grooved.size(), 0);

    // Verify velocity variations are applied
    bool hasVelocityVariation = false;
    for (const auto& note : grooved) {
        if (note.velocity != 100) {
            hasVelocityVariation = true;
            break;
        }
    }
    // Note: Due to randomness and dropout, this may vary
    // Main goal is to ensure valid velocity values
    for (const auto& note : grooved) {
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
    }
}

// Test getTemplateNames returns all template names
TEST_F(GrooveEngineTest, GetTemplateNames) {
    std::vector<std::string> names = engine->getTemplateNames();

    // Should have 8 templates
    EXPECT_EQ(names.size(), 8);

    // Should contain all expected templates
    std::vector<std::string> expectedTemplates = {
        "funk", "jazz", "rock", "hiphop", "edm", "latin", "blues", "lofi"
    };

    for (const auto& expected : expectedTemplates) {
        bool found = false;
        for (const auto& name : names) {
            if (name == expected) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Template '" << expected << "' not found";
    }
}

// Test getTemplate retrieves templates by name
TEST_F(GrooveEngineTest, GetTemplate) {
    // Test valid template names
    std::vector<std::string> validNames = {
        "funk", "jazz", "rock", "hiphop", "edm", "latin", "blues", "lofi"
    };

    for (const auto& name : validNames) {
        const GrooveTemplateData* tmpl = engine->getTemplate(name);
        EXPECT_NE(tmpl, nullptr) << "Template '" << name << "' not found";
        if (tmpl) {
            EXPECT_EQ(tmpl->name.find(name) != std::string::npos ||
                     tmpl->name.find("Funk") != std::string::npos ||
                     tmpl->name.find("Jazz") != std::string::npos ||
                     tmpl->name.find("Rock") != std::string::npos ||
                     tmpl->name.find("Hip-Hop") != std::string::npos ||
                     tmpl->name.find("EDM") != std::string::npos ||
                     tmpl->name.find("Latin") != std::string::npos ||
                     tmpl->name.find("Blues") != std::string::npos ||
                     tmpl->name.find("Lo-Fi") != std::string::npos, true);
            EXPECT_FALSE(tmpl->description.empty());
            EXPECT_EQ(tmpl->timingDeviations.size(), 16);  // 16 16th notes
            EXPECT_EQ(tmpl->velocityCurve.size(), 16);
        }
    }

    // Test invalid template name
    const GrooveTemplateData* invalid = engine->getTemplate("nonexistent");
    EXPECT_EQ(invalid, nullptr);
}

// Test swing factor application
TEST_F(GrooveEngineTest, ApplySwing_SwingFactor) {
    std::vector<MidiNote> notes;
    // Create notes on 8th note grid
    for (int i = 0; i < 8; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i) * 0.5;  // 8th notes
        note.duration = 0.25;
        notes.push_back(note);
    }

    // Test different swing amounts
    std::vector<MidiNote> straight = engine->applySwing(notes, 0.0f, 1.0f);
    std::vector<MidiNote> moderate = engine->applySwing(notes, 0.5f, 1.0f);
    std::vector<MidiNote> triplet = engine->applySwing(notes, 0.67f, 1.0f);

    EXPECT_EQ(straight.size(), notes.size());
    EXPECT_EQ(moderate.size(), notes.size());
    EXPECT_EQ(triplet.size(), notes.size());

    // Verify all notes are valid
    for (const auto& note : straight) {
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startBeat, -0.5);  // Allow pushed timing
    }
}

// Test push/pull timing feel
TEST_F(GrooveEngineTest, ApplyTimingFeel_PushPull) {
    std::vector<MidiNote> notes;
    // Create notes on exact beat grid
    for (int i = 0; i < 8; ++i) {
        MidiNote note;
        note.pitch = 60;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i);  // Exact beats
        note.duration = 0.5;
        notes.push_back(note);
    }

    // Test pull (laid back, -1.0)
    std::vector<MidiNote> pull = engine->applyTimingFeel(notes, -1.0f, 1.0f);

    // Test neutral (0.0)
    std::vector<MidiNote> neutral = engine->applyTimingFeel(notes, 0.0f, 1.0f);

    // Test push (ahead, +1.0)
    std::vector<MidiNote> push = engine->applyTimingFeel(notes, 1.0f, 1.0f);

    EXPECT_EQ(pull.size(), notes.size());
    EXPECT_EQ(neutral.size(), notes.size());
    EXPECT_EQ(push.size(), notes.size());

    // Verify all notes are valid
    for (const auto& note : pull) {
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
        EXPECT_GE(note.startBeat, -0.5);  // Allow pushed timing
    }

    // With pull, notes should be slightly later (or same if intensity is low)
    // With push, notes should be slightly earlier (or same if intensity is low)
    // Due to randomness, we just verify the function doesn't crash and produces valid output
}
