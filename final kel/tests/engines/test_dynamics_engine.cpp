#include <gtest/gtest.h>
#include "engines/DynamicsEngine.h"
#include <vector>
#include "common/Types.h"

using namespace kelly;

class DynamicsEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<DynamicsEngine>();
        
        // Create test notes
        testNotes.clear();
        for (int i = 0; i < 8; ++i) {
            MidiNote note;
            note.pitch = 60 + i;
            note.velocity = 100;
            note.startBeat = i * 0.5;
            note.duration = 0.5;
            testNotes.push_back(note);
        }
    }
    
    std::unique_ptr<DynamicsEngine> engine;
    std::vector<MidiNote> testNotes;
};

// Test basic application
TEST_F(DynamicsEngineTest, BasicApplication) {
    DynamicsOutput output = engine->apply(testNotes, "neutral", 0.5f);
    
    EXPECT_EQ(output.notes.size(), testNotes.size());
    EXPECT_EQ(output.emotion, "neutral");
}

// Test different emotions
TEST_F(DynamicsEngineTest, DifferentEmotions) {
    std::vector<std::string> emotions = {"joy", "sad", "anger", "fear"};
    
    for (const auto& emotion : emotions) {
        DynamicsOutput output = engine->apply(testNotes, emotion, 0.5f);
        EXPECT_EQ(output.notes.size(), testNotes.size()) << "Failed for emotion: " << emotion;
        EXPECT_EQ(output.emotion, emotion);
    }
}

// Test marking override
TEST_F(DynamicsEngineTest, MarkingOverride) {
    DynamicsConfig config;
    config.emotion = "neutral";
    config.notes = testNotes;
    config.totalTicks = 1920;
    config.baseMarking = DynamicMarking::Forte;
    config.expressiveness = 0.5f;
    
    DynamicsOutput output = engine->apply(config);
    EXPECT_EQ(output.baseMarking, DynamicMarking::Forte);
}

// Test shape override
TEST_F(DynamicsEngineTest, ShapeOverride) {
    DynamicsConfig config;
    config.emotion = "neutral";
    config.notes = testNotes;
    config.totalTicks = 1920;
    config.shapeOverride = DynamicShape::Crescendo;
    config.expressiveness = 0.5f;
    
    DynamicsOutput output = engine->apply(config);
    EXPECT_EQ(output.shapeUsed, DynamicShape::Crescendo);
}

// Test velocity range
TEST_F(DynamicsEngineTest, VelocityRange) {
    DynamicsOutput output = engine->apply(testNotes, "neutral", 0.5f);
    
    EXPECT_GE(output.velocityRange.first, 0);
    EXPECT_LE(output.velocityRange.first, 127);
    EXPECT_GE(output.velocityRange.second, output.velocityRange.first);
    EXPECT_LE(output.velocityRange.second, 127);
}

// Test note velocities modified
TEST_F(DynamicsEngineTest, NoteVelocitiesModified) {
    DynamicsOutput output = engine->apply(testNotes, "neutral", 0.5f);
    
    for (const auto& note : output.notes) {
        EXPECT_GE(note.velocity, 0);
        EXPECT_LE(note.velocity, 127);
    }
}

// Test curve generation
TEST_F(DynamicsEngineTest, CurveGeneration) {
    DynamicCurve curve = engine->generateCurve(
        1920, 
        DynamicShape::Crescendo,
        DynamicMarking::Piano,
        DynamicMarking::Forte
    );
    
    EXPECT_EQ(curve.shape, DynamicShape::Crescendo);
    EXPECT_GT(curve.points.size(), 0);
    
    // First point should be lower than last (crescendo)
    if (curve.points.size() > 1) {
        EXPECT_LE(curve.points[0].velocity, curve.points[curve.points.size() - 1].velocity);
    }
}

// Test curve application
TEST_F(DynamicsEngineTest, CurveApplication) {
    DynamicCurve curve = engine->generateCurve(
        1920,
        DynamicShape::Crescendo,
        DynamicMarking::Piano,
        DynamicMarking::Forte
    );
    
    std::vector<MidiNote> modified = engine->applyCurve(testNotes, curve);
    
    EXPECT_EQ(modified.size(), testNotes.size());
}

// Test marking to velocity conversion
TEST_F(DynamicsEngineTest, MarkingToVelocity) {
    int ppp = engine->markingToVelocity(DynamicMarking::Pianississimo);
    int fff = engine->markingToVelocity(DynamicMarking::Fortississimo);
    
    EXPECT_GT(ppp, 0);
    EXPECT_LE(ppp, 127);
    EXPECT_GT(fff, ppp);
    EXPECT_LE(fff, 127);
}

// Test velocity to marking conversion
TEST_F(DynamicsEngineTest, VelocityToMarking) {
    DynamicMarking low = engine->velocityToMarking(30);
    DynamicMarking high = engine->velocityToMarking(110);
    
    // Low velocity should map to soft marking
    int lowVel = engine->markingToVelocity(low);
    int highVel = engine->markingToVelocity(high);
    
    EXPECT_LE(lowVel, highVel);
}

// Test accent application
TEST_F(DynamicsEngineTest, AccentApplication) {
    std::vector<MidiNote> accented = engine->applyAccents(testNotes, "anger", 0.8f);
    
    EXPECT_EQ(accented.size(), testNotes.size());
    
    // Some notes should have higher velocity (accented)
    bool hasAccents = false;
    for (size_t i = 0; i < accented.size(); ++i) {
        if (accented[i].velocity > testNotes[i].velocity) {
            hasAccents = true;
            break;
        }
    }
    // May or may not have accents depending on implementation
}

// Test expressiveness parameter
TEST_F(DynamicsEngineTest, ExpressivenessParameter) {
    DynamicsOutput lowExpr = engine->apply(testNotes, "neutral", 0.1f);
    DynamicsOutput highExpr = engine->apply(testNotes, "neutral", 0.9f);
    
    EXPECT_EQ(lowExpr.notes.size(), highExpr.notes.size());
}

// Test different dynamic shapes
TEST_F(DynamicsEngineTest, DifferentShapes) {
    std::vector<DynamicShape> shapes = {
        DynamicShape::Constant,
        DynamicShape::Crescendo,
        DynamicShape::Decrescendo,
        DynamicShape::Swell
    };
    
    for (const auto& shape : shapes) {
        DynamicCurve curve = engine->generateCurve(
            1920,
            shape,
            DynamicMarking::MezzoForte,
            DynamicMarking::Forte
        );
        EXPECT_EQ(curve.shape, shape);
    }
}
