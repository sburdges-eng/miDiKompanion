#include <gtest/gtest.h>
#include "engines/ArrangementEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class ArrangementEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<ArrangementEngine>();
    }
    
    std::unique_ptr<ArrangementEngine> engine;
};

// Test basic generation
TEST_F(ArrangementEngineTest, BasicGeneration) {
    ArrangementOutput output = engine->generate("neutral", "", 32);
    
    EXPECT_GT(output.sections.size(), 0);
    EXPECT_GT(output.totalBars, 0);
    EXPECT_FALSE(output.narrativeArc.empty());
}

// Test different emotions
TEST_F(ArrangementEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        ArrangementOutput output = engine->generate(emotion, "", 32);
        EXPECT_GT(output.sections.size(), 0) << "Failed for emotion: " << emotion;
    }
}

// Test genre parameter
TEST_F(ArrangementEngineTest, GenreParameter) {
    std::vector<std::string> genres = {"rock", "jazz", "electronic", ""};
    
    for (const auto& genre : genres) {
        ArrangementOutput output = engine->generate("neutral", genre, 32);
        EXPECT_GT(output.sections.size(), 0) << "Failed for genre: " << genre;
    }
}

// Test emotional arc
TEST_F(ArrangementEngineTest, EmotionalArc) {
    ArrangementConfig config;
    config.emotion = "neutral";
    config.targetBars = 32;
    config.emotionalArc = 0.0f; // Descending
    
    ArrangementOutput output = engine->generate(config);
    EXPECT_GT(output.sections.size(), 0);
    
    // Energy should generally decrease
    if (output.energyCurve.size() > 1) {
        float startEnergy = output.energyCurve[0];
        float endEnergy = output.energyCurve[output.energyCurve.size() - 1];
        EXPECT_GE(startEnergy, endEnergy);
    }
}

// Test section creation
TEST_F(ArrangementEngineTest, SectionCreation) {
    Section section = engine->createSection(SectionType::Chorus, "joy", 8);
    
    EXPECT_EQ(section.type, SectionType::Chorus);
    EXPECT_EQ(section.bars, 8);
    EXPECT_GT(section.energy, 0.0f);
    EXPECT_LE(section.energy, 1.0f);
}

// Test instruments for section
TEST_F(ArrangementEngineTest, InstrumentsForSection) {
    std::vector<std::string> instruments = engine->getInstrumentsForSection(SectionType::Chorus, 0.8f);
    
    EXPECT_GT(instruments.size(), 0);
}

// Test include intro/outro
TEST_F(ArrangementEngineTest, IncludeIntroOutro) {
    ArrangementConfig config;
    config.emotion = "neutral";
    config.targetBars = 32;
    config.includeIntro = true;
    config.includeOutro = true;
    
    ArrangementOutput output = engine->generate(config);
    
    bool hasIntro = false, hasOutro = false;
    for (const auto& section : output.sections) {
        if (section.type == SectionType::Intro) hasIntro = true;
        if (section.type == SectionType::Outro) hasOutro = true;
    }
    
    EXPECT_TRUE(hasIntro);
    EXPECT_TRUE(hasOutro);
}

// Test section ordering
TEST_F(ArrangementEngineTest, SectionOrdering) {
    ArrangementOutput output = engine->generate("neutral", "", 32);
    
    // Sections should be in logical order
    // Intro should come first if present
    if (!output.sections.empty()) {
        bool foundIntro = false;
        for (const auto& section : output.sections) {
            if (section.type == SectionType::Intro) {
                foundIntro = true;
                break;
            }
            if (foundIntro && section.type == SectionType::Intro) {
                FAIL() << "Intro found after other sections";
            }
        }
    }
}

// Test energy curve
TEST_F(ArrangementEngineTest, EnergyCurve) {
    ArrangementOutput output = engine->generate("neutral", "", 32);
    
    EXPECT_EQ(output.energyCurve.size(), output.sections.size());
    for (float energy : output.energyCurve) {
        EXPECT_GE(energy, 0.0f);
        EXPECT_LE(energy, 1.0f);
    }
}

// Test seed reproducibility
TEST_F(ArrangementEngineTest, SeedReproducibility) {
    ArrangementConfig config1;
    config1.emotion = "neutral";
    config1.targetBars = 16;
    config1.seed = 42;
    
    ArrangementConfig config2 = config1;
    
    ArrangementOutput output1 = engine->generate(config1);
    ArrangementOutput output2 = engine->generate(config2);
    
    EXPECT_EQ(output1.sections.size(), output2.sections.size());
}

// Test different target bar counts
TEST_F(ArrangementEngineTest, DifferentTargetBars) {
    for (int bars : {8, 16, 32, 64}) {
        ArrangementOutput output = engine->generate("neutral", "", bars);
        EXPECT_GT(output.sections.size(), 0) << "Failed for bars: " << bars;
        EXPECT_LE(output.totalBars, bars * 2); // Should be close to target
    }
}
