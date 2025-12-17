#pragma once

#include "common/Types.h"
#include "engine/EmotionThesaurus.h"
#include <memory>

namespace kelly {

/**
 * The three-phase intent processing pipeline.
 * 
 * Phase 0: Core Wound/Desire - "What hurts?"
 * Phase 1: Emotional Intent - Map to 216-node thesaurus
 * Phase 2: Technical Constraints - Which rules to break and why
 */
class IntentPipeline {
public:
    IntentPipeline();
    
    /**
     * Process a complete intent from wound to musical parameters.
     * This is the main entry point for the emotion engine.
     */
    IntentResult process(const Wound& wound);
    
    /**
     * Process Side A (current state) and Side B (desired state)
     * to create a musical journey between emotions.
     */
    IntentResult processJourney(const SideA& current, const SideB& desired);
    
    /**
     * Get direct access to the thesaurus for UI emotion selection
     */
    const EmotionThesaurus& thesaurus() const { return thesaurus_; }
    
private:
    EmotionThesaurus thesaurus_;
    
    // Phase 1: Wound → Emotion
    EmotionNode processWound(const Wound& wound);
    
    // Phase 2: Emotion → Rule breaks
    std::vector<RuleBreak> generateRuleBreaks(const EmotionNode& emotion);
    
    // Phase 3: Compile musical parameters
    IntentResult compileMusicalParams(
        const Wound& wound,
        const EmotionNode& emotion,
        const std::vector<RuleBreak>& ruleBreaks
    );
    
    // Keyword matching for wound processing
    float matchKeywords(const std::string& text, 
                        const std::vector<std::string>& keywords) const;
};

} // namespace kelly
