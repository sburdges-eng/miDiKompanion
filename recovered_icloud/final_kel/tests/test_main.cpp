#include <gtest/gtest.h>
#include <iostream>

int main(int argc, char **argv) {
    std::cout << "Running Kelly MIDI Companion Unit Tests\n";
    std::cout << "========================================\n\n";
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
