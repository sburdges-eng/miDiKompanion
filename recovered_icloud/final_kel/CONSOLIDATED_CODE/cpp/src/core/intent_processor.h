#pragma once
/*
 * intent_processor.h - Legacy Intent Processor
 * ===========================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Core Layer: Legacy intent processor (may be superseded by engine/IntentProcessor.h)
 * - Type System: Defines basic Wound and IntentPhase structures
 * - Engine Layer: Basic wound processing functionality
 *
 * Purpose: Legacy intent processor providing basic wound processing.
 *          Note: May be superseded by engine/IntentProcessor.h which provides
 *          complete three-phase intent processing with emotion mapping and rule breaks.
 *
 * Features:
 * - Basic wound processing
 * - Intent phase enumeration
 * - Simple wound structure
 */

#include <string>

namespace kelly {

enum class IntentPhase {
    Wound,
    Emotion,
    RuleBreak
};

struct Wound {
    std::string description;
    float intensity;
    std::string source;
};

class IntentProcessor {
public:
    IntentProcessor() = default;
    ~IntentProcessor() = default;

    int processWound(const Wound& wound);
    // Additional methods would be implemented

private:
    // Implementation details
};

} // namespace kelly
