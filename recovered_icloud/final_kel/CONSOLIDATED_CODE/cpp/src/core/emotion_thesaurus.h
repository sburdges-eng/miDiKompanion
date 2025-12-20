#pragma once
/*
 * emotion_thesaurus.h - Legacy Emotion Thesaurus
 * ==============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Core Layer: Legacy emotion thesaurus (may be superseded by engine/EmotionThesaurus.h)
 * - Type System: Defines basic EmotionThesaurusNode structure
 * - Emotion System: Basic emotion node storage
 *
 * Purpose: Legacy emotion thesaurus providing basic emotion node storage.
 *          Note: May be superseded by engine/EmotionThesaurus.h which provides
 *          the full 216-node emotion thesaurus with VAD coordinates, musical attributes,
 *          and comprehensive lookup methods.
 *
 * Features:
 * - Basic emotion node storage
 * - Node lookup by ID
 * - 216-node structure (mentioned in comments)
 */

#include <string>

namespace kelly {

struct EmotionThesaurusNode {
    int id;
    std::string name;
    // Additional fields would be here
};

class EmotionThesaurus {
public:
    EmotionThesaurus();
    ~EmotionThesaurus() = default;

    const EmotionThesaurusNode* getNode(int id) const;
    // Additional methods for the 216-node thesaurus

private:
    void initializeThesaurus();
    // Storage for 216 nodes
};

} // namespace kelly
