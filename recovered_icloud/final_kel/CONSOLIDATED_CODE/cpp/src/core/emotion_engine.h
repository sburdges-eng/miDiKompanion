#pragma once
/*
 * emotion_engine.h - Legacy Emotion Engine
 * ========================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Core Layer: Legacy emotion engine (may be superseded by engine/EmotionThesaurus.h)
 * - Type System: Defines EmotionNode, EmotionCategory structures
 * - Engine Layer: Basic emotion lookup functionality
 *
 * Purpose: Legacy emotion engine providing basic emotion lookup.
 *          Note: May be superseded by engine/EmotionThesaurus.h which provides
 *          the full 216-node emotion thesaurus with VAD coordinates.
 *
 * Features:
 * - Emotion lookup by ID
 * - Emotion lookup by name
 * - Nearby emotion search
 * - Basic emotion structure
 */

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace kelly {

enum class EmotionCategory {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Trust,
    Anticipation
};

struct MusicalAttributes {
    float tempoModifier = 1.0f;
    std::string mode = "minor";
    float dynamics = 0.5f;
};

struct EmotionNode {
    int id;
    std::string name;
    EmotionCategory category;
    float intensity;  // 0.0 to 1.0
    float valence;    // -1.0 to 1.0
    float arousal;    // 0.0 to 1.0
    std::vector<int> relatedEmotions;
    MusicalAttributes musicalAttributes;
};

class EmotionEngine {
public:
    EmotionEngine();
    ~EmotionEngine() = default;

    const EmotionNode* getEmotion(int emotionId) const;
    const EmotionNode* findEmotionByName(const std::string& name) const;
    std::vector<const EmotionNode*> getNearbyEmotions(int emotionId, float threshold = 0.3f) const;

    size_t getEmotionCount() const { return nodes_.size(); }

private:
    void initializeEmotions();
    float calculateDistance(const EmotionNode& a, const EmotionNode& b) const;

    std::map<int, EmotionNode> nodes_;
};

} // namespace kelly
