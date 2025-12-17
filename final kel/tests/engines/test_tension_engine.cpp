#include <gtest/gtest.h>
#include "engines/TensionEngine.h"
#include <vector>
#include <string>

using namespace kelly;

class TensionEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<TensionEngine>();
    }
    
    std::unique_ptr<TensionEngine> engine;
    std::vector<std::string> testProgression = {"C", "Am", "F", "G"};
};

// Test basic generation
TEST_F(TensionEngineTest, BasicGeneration) {
    TensionOutput output = engine->generate("neutral", testProgression, 4, 120);
    
    EXPECT_GT(output.tensionPoints.size(), 0);
    EXPECT_GT(output.tensionCurve.size(), 0);
}

// Test different emotions
TEST_F(TensionEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        TensionOutput output = engine->generate(emotion, testProgression, 4, 120);
        EXPECT_GT(output.tensionPoints.size(), 0) << "Failed for emotion: " << emotion;
    }
}

// Test tension curve
TEST_F(TensionEngineTest, TensionCurve) {
    TensionConfig config;
    config.emotion = "neutral";
    config.chordProgression = testProgression;
    config.bars = 4;
    config.tempoBpm = 120;
    config.curve = TensionCurve::Building;
    
    TensionOutput output = engine->generate(config);
    EXPECT_EQ(output.curveUsed, TensionCurve::Building);
    
    // Building curve should increase
    if (output.tensionCurve.size() > 1) {
        EXPECT_LE(output.tensionCurve[0], output.tensionCurve[output.tensionCurve.size() - 1]);
    }
}

// Test tension calculation
TEST_F(TensionEngineTest, TensionCalculation) {
    std::vector<int> chordPitches = {60, 64, 67}; // C major
    
    float tension = engine->calculateTension(chordPitches, "C");
    
    EXPECT_GE(tension, 0.0f);
    EXPECT_LE(tension, 1.0f);
}

// Test add tension notes
TEST_F(TensionEngineTest, AddTensionNotes) {
    std::vector<int> chordPitches = {60, 64, 67}; // C major
    
    std::vector<int> withTension = engine->addTensionNotes(
        chordPitches,
        TensionTechnique::Suspension,
        0.5f
    );
    
    EXPECT_GT(withTension.size(), chordPitches.size());
}

// Test peak tension
TEST_F(TensionEngineTest, PeakTension) {
    TensionOutput output = engine->generate("neutral", testProgression, 4, 120);
    
    EXPECT_GE(output.peakTension, 0.0f);
    EXPECT_LE(output.peakTension, 1.0f);
    EXPECT_GE(output.peakTick, 0);
}

// Test different tension curves
TEST_F(TensionEngineTest, DifferentTensionCurves) {
    std::vector<TensionCurve> curves = {
        TensionCurve::Building,
        TensionCurve::Releasing,
        TensionCurve::Plateau,
        TensionCurve::Wave
    };
    
    for (const auto& curve : curves) {
        TensionConfig config;
        config.emotion = "neutral";
        config.chordProgression = testProgression;
        config.bars = 4;
        config.tempoBpm = 120;
        config.curve = curve;
        
        TensionOutput output = engine->generate(config);
        EXPECT_EQ(output.curveUsed, curve);
    }
}

// Test max tension parameter
TEST_F(TensionEngineTest, MaxTensionParameter) {
    TensionConfig config;
    config.emotion = "neutral";
    config.chordProgression = testProgression;
    config.bars = 4;
    config.tempoBpm = 120;
    config.maxTension = 0.5f;
    
    TensionOutput output = engine->generate(config);
    
    for (float tension : output.tensionCurve) {
        EXPECT_LE(tension, 0.5f);
    }
}

// Test tension point validity
TEST_F(TensionEngineTest, TensionPointValidity) {
    TensionOutput output = engine->generate("neutral", testProgression, 4, 120);
    
    for (const auto& point : output.tensionPoints) {
        EXPECT_GE(point.tensionLevel, 0.0f);
        EXPECT_LE(point.tensionLevel, 1.0f);
        EXPECT_GE(point.tick, 0);
    }
}

// Test seed reproducibility
TEST_F(TensionEngineTest, SeedReproducibility) {
    TensionConfig config1;
    config1.emotion = "neutral";
    config1.chordProgression = testProgression;
    config1.bars = 2;
    config1.tempoBpm = 120;
    config1.seed = 42;
    
    TensionConfig config2 = config1;
    
    TensionOutput output1 = engine->generate(config1);
    TensionOutput output2 = engine->generate(config2);
    
    EXPECT_EQ(output1.tensionPoints.size(), output2.tensionPoints.size());
}
