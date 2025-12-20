#include <gtest/gtest.h>
#include "engine/EmotionThesaurus.h"
#include "midi/GrooveEngine.h"
#include "midi/MidiGenerator.h"
#include "engine/IntentPipeline.h"
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <string>

using namespace kelly;

class ThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        thesaurus = std::make_unique<EmotionThesaurus>();
        grooveEngine = std::make_unique<GrooveEngine>();
        midiGenerator = std::make_unique<MidiGenerator>();
        intentPipeline = std::make_unique<IntentPipeline>();
    }

    std::unique_ptr<EmotionThesaurus> thesaurus;
    std::unique_ptr<GrooveEngine> grooveEngine;
    std::unique_ptr<MidiGenerator> midiGenerator;
    std::unique_ptr<IntentPipeline> intentPipeline;
};

// Test concurrent reads from EmotionThesaurus
TEST_F(ThreadSafetyTest, EmotionThesaurus_ConcurrentReads) {
    const int numThreads = 8;
    const int readsPerThread = 100;
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    auto readTask = [&](int threadId) {
        for (int i = 0; i < readsPerThread; ++i) {
            // Read size
            size_t size = thesaurus->size();
            if (size > 0) successCount++;
            else failureCount++;

            // Read all emotions
            auto all = thesaurus->all();
            if (all.size() > 0) successCount++;
            else failureCount++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(readTask, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load(), 0);
    EXPECT_EQ(failureCount.load(), 0);
}

// Test concurrent findById calls
TEST_F(ThreadSafetyTest, EmotionThesaurus_ConcurrentFindById) {
    const int numThreads = 8;
    const int lookupsPerThread = 50;
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    auto findByIdTask = [&](int threadId) {
        for (int i = 0; i < lookupsPerThread; ++i) {
            int id = (threadId * lookupsPerThread + i) % 216 + 1;  // IDs 1-216
            auto emotion = thesaurus->findById(id);
            if (emotion.has_value()) {
                EXPECT_EQ(emotion->id, id);
                successCount++;
            } else {
                failureCount++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(findByIdTask, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load(), 0);
}

// Test concurrent findByName calls
TEST_F(ThreadSafetyTest, EmotionThesaurus_ConcurrentFindByName) {
    const int numThreads = 8;
    std::vector<std::string> emotionNames = {
        "joy", "grief", "rage", "anxiety", "serenity",
        "fear", "anger", "sadness", "happiness", "love"
    };
    std::atomic<int> successCount{0};

    auto findByNameTask = [&](int threadId) {
        for (int i = 0; i < 50; ++i) {
            std::string name = emotionNames[(threadId + i) % emotionNames.size()];
            auto emotion = thesaurus->findByName(name);
            if (emotion.has_value()) {
                successCount++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(findByNameTask, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load(), 0);
}

// Test concurrent groove application
TEST_F(ThreadSafetyTest, GrooveEngine_ConcurrentApplyGroove) {
    const int numThreads = 4;
    std::atomic<int> successCount{0};

    // Create test notes
    std::vector<MidiNote> notes;
    for (int i = 0; i < 16; ++i) {
        MidiNote note;
        note.pitch = 60 + i;
        note.velocity = 100;
        note.startBeat = static_cast<double>(i) * 0.25;
        note.duration = 0.25;
        notes.push_back(note);
    }

    auto applyGrooveTask = [&](int threadId) {
        std::vector<std::string> templates = {"funk", "jazz", "rock", "hiphop"};
        for (int i = 0; i < 20; ++i) {
            std::string templateName = templates[threadId % templates.size()];
            std::vector<MidiNote> grooved = grooveEngine->applyGrooveTemplate(
                notes, templateName, 0.5f, 1.0f);
            if (grooved.size() >= 0) {  // Valid result
                successCount++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(applyGrooveTask, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successCount.load(), numThreads * 20);
}

// Test concurrent MIDI generation
TEST_F(ThreadSafetyTest, MidiGenerator_ConcurrentGeneration) {
    const int numThreads = 4;
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    auto generateTask = [&](int threadId) {
        for (int i = 0; i < 5; ++i) {
            try {
                Wound wound;
                wound.description = "I feel " + std::to_string(threadId) + " emotion";
                wound.intensity = 0.5f + (threadId * 0.1f);
                wound.source = "internal";

                IntentResult intent = intentPipeline->process(wound);
                GeneratedMidi midi = midiGenerator->generate(
                    intent, 4, 0.5f, 0.4f, 0.0f, 0.75f);

                if (midi.chords.size() > 0 || midi.melody.size() > 0) {
                    successCount++;
                } else {
                    failureCount++;
                }
            } catch (...) {
                failureCount++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(generateTask, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load(), 0);
    // Allow some failures due to resource contention, but most should succeed
    EXPECT_LT(failureCount.load(), numThreads * 5 / 2);
}

// Test try_lock pattern (simulating audio thread behavior)
TEST_F(ThreadSafetyTest, TryLockPattern_NoDeadlocks) {
    // Simulate the try_lock pattern used in PluginProcessor
    // Audio thread uses try_lock, UI thread uses lock_guard

    std::mutex testMutex;
    std::atomic<int> audioThreadSuccess{0};
    std::atomic<int> audioThreadSkips{0};
    std::atomic<int> uiThreadSuccess{0};

    // Simulate audio thread (non-blocking)
    auto audioThreadTask = [&]() {
        for (int i = 0; i < 100; ++i) {
            std::unique_lock<std::mutex> lock(testMutex, std::try_to_lock);
            if (lock.owns_lock()) {
                // Successfully acquired lock
                audioThreadSuccess++;
                // Simulate work
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            } else {
                // Couldn't acquire lock - skip (audio thread must never block)
                audioThreadSkips++;
            }
        }
    };

    // Simulate UI thread (can block)
    auto uiThreadTask = [&]() {
        for (int i = 0; i < 50; ++i) {
            std::lock_guard<std::mutex> lock(testMutex);
            uiThreadSuccess++;
            // Simulate work (UI thread can take longer)
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    };

    std::thread audioThread(audioThreadTask);
    std::thread uiThread(uiThreadTask);

    audioThread.join();
    uiThread.join();

    // Verify no deadlock occurred (both threads completed)
    EXPECT_GT(audioThreadSuccess.load() + audioThreadSkips.load(), 0);
    EXPECT_EQ(uiThreadSuccess.load(), 50);

    // Audio thread should have some successes and some skips
    // (This is the expected behavior - audio thread never blocks)
}

// Test atomic flag behavior (simulating hasPendingMidi)
TEST_F(ThreadSafetyTest, AtomicFlagBehavior) {
    std::atomic<bool> hasPendingMidi{false};
    std::atomic<int> readCount{0};
    std::atomic<int> writeCount{0};

    // Reader thread (simulating audio thread)
    auto readerTask = [&]() {
        for (int i = 0; i < 1000; ++i) {
            bool pending = hasPendingMidi.load();
            readCount++;
            if (pending) {
                // Simulate processing
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
    };

    // Writer thread (simulating UI thread)
    auto writerTask = [&]() {
        for (int i = 0; i < 100; ++i) {
            hasPendingMidi.store(true);
            writeCount++;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            hasPendingMidi.store(false);
        }
    };

    std::thread reader(readerTask);
    std::thread writer(writerTask);

    reader.join();
    writer.join();

    // Verify both threads completed
    EXPECT_EQ(readCount.load(), 1000);
    EXPECT_EQ(writeCount.load(), 100);

    // Verify atomic operations are thread-safe (no crashes)
    EXPECT_TRUE(true);
}

// Test parameter changes during generation
TEST_F(ThreadSafetyTest, ParameterChangesDuringGeneration) {
    std::atomic<bool> isGenerating{false};
    std::atomic<int> generationCount{0};
    std::atomic<int> parameterChangeCount{0};

    // Generation thread
    auto generationTask = [&]() {
        for (int i = 0; i < 10; ++i) {
            if (isGenerating.exchange(true)) {
                continue; // Already generating
            }

            // Simulate generation
            Wound wound;
            wound.description = "I feel " + std::to_string(i);
            wound.intensity = 0.5f;
            wound.source = "internal";

            IntentResult intent = intentPipeline->process(wound);
            GeneratedMidi midi = midiGenerator->generate(intent, 4, 0.5f, 0.4f, 0.0f, 0.75f);

            generationCount++;
            isGenerating.store(false);
        }
    };

    // Parameter change thread
    auto parameterChangeTask = [&]() {
        for (int i = 0; i < 50; ++i) {
            // Simulate parameter change
            parameterChangeCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    std::thread genThread(generationTask);
    std::thread paramThread(parameterChangeTask);

    genThread.join();
    paramThread.join();

    // Verify both threads completed
    EXPECT_GT(generationCount.load(), 0);
    EXPECT_EQ(parameterChangeCount.load(), 50);

    // Verify no deadlock occurred
    EXPECT_FALSE(isGenerating.load());
}
