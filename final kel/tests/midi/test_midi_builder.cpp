#include <gtest/gtest.h>
#include "midi/MidiBuilder.h"
#include <vector>
#include "common/Types.h"

using namespace kelly;

class MidiBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        builder = std::make_unique<MidiBuilder>();
    }
    
    std::unique_ptr<MidiBuilder> builder;
};

// Test building from notes
TEST_F(MidiBuilderTest, BuildFromNotes) {
    std::vector<MidiNote> notes;
    for (int i = 0; i < 8; ++i) {
        MidiNote note;
        note.pitch = 60 + i;
        note.velocity = 100;
        note.startBeat = i * 0.5;
        note.duration = 0.5;
        notes.push_back(note);
    }
    
    MidiFile file = builder->build(notes, 120);
    
    EXPECT_GT(file.notes.size(), 0);
    EXPECT_GT(file.tempo, 0.0f);
}

// Test building from chords
TEST_F(MidiBuilderTest, BuildFromChords) {
    std::vector<Chord> chords;
    Chord chord;
    chord.pitches = {60, 64, 67};
    chord.name = "C";
    chord.startBeat = 0.0;
    chord.duration = 2.0;
    chords.push_back(chord);
    
    MidiFile file = builder->buildFromChords(chords, 120);
    
    EXPECT_GT(file.notes.size(), 0);
}

// Test tempo setting
TEST_F(MidiBuilderTest, TempoSetting) {
    std::vector<MidiNote> notes;
    MidiNote note;
    note.pitch = 60;
    note.velocity = 100;
    note.startBeat = 0.0;
    note.duration = 1.0;
    notes.push_back(note);
    
    MidiFile file = builder->build(notes, 120);
    EXPECT_EQ(file.tempo, 120.0f);
    
    MidiFile file2 = builder->build(notes, 140);
    EXPECT_EQ(file2.tempo, 140.0f);
}
