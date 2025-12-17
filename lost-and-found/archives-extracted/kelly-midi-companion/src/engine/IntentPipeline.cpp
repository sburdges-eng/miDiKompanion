#include "engine/IntentPipeline.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace kelly {

IntentPipeline::IntentPipeline() = default;

IntentResult IntentPipeline::process(const Wound& wound) {
    // Phase 1: Process wound to emotion
    EmotionNode emotion = processWound(wound);
    
    // Phase 2: Generate rule breaks
    std::vector<RuleBreak> ruleBreaks = generateRuleBreaks(emotion);
    
    // Phase 3: Compile everything into musical parameters
    return compileMusicalParams(wound, emotion, ruleBreaks);
}

IntentResult IntentPipeline::processJourney(const SideA& current, const SideB& desired) {
    // Get emotions for both sides
    EmotionNode emotionA = current.emotionId 
        ? thesaurus_.findById(*current.emotionId).value_or(
            processWound({current.description, current.intensity, "sideA"}))
        : processWound({current.description, current.intensity, "sideA"});
    
    EmotionNode emotionB = desired.emotionId
        ? thesaurus_.findById(*desired.emotionId).value_or(
            processWound({desired.description, desired.intensity, "sideB"}))
        : processWound({desired.description, desired.intensity, "sideB"});
    
    // The journey emotion is a blend - starting from A, leaning toward B
    // This creates musical tension that resolves
    float blendedValence = emotionA.valence * 0.4f + emotionB.valence * 0.6f;
    float blendedArousal = emotionA.arousal * 0.4f + emotionB.arousal * 0.6f;
    float blendedIntensity = std::max(emotionA.intensity, emotionB.intensity);
    
    EmotionNode journeyEmotion = thesaurus_.findNearest(
        blendedValence, blendedArousal, blendedIntensity);
    
    // Create wound representing the journey
    Wound journeyWound{
        "Journey from " + emotionA.name + " toward " + emotionB.name,
        blendedIntensity,
        "cassette_journey"
    };
    
    // Generate rule breaks that serve the transition
    std::vector<RuleBreak> ruleBreaks = generateRuleBreaks(journeyEmotion);
    
    // Add journey-specific rule break if emotions are far apart
    float emotionalDistance = std::abs(emotionA.valence - emotionB.valence) +
                              std::abs(emotionA.arousal - emotionB.arousal);
    
    if (emotionalDistance > 1.0f) {
        ruleBreaks.push_back({
            RuleBreakType::Form,
            emotionalDistance / 2.0f,
            "Dramatic structural shift",
            "The emotional journey requires breaking conventional song structure"
        });
    }
    
    return compileMusicalParams(journeyWound, journeyEmotion, ruleBreaks);
}

EmotionNode IntentPipeline::processWound(const Wound& wound) {
    std::string text = wound.description;
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    
    // Keyword clusters for emotion detection
    // Each cluster maps to primary emotion candidates
    
    struct EmotionClue {
        std::vector<std::string> keywords;
        int primaryEmotionId;
        float confidence;
    };
    
    std::vector<EmotionClue> clues = {
        // Grief/Loss cluster
        {{"loss", "lost", "death", "died", "gone", "missing", "grief", "mourn"}, 1, 0.9f},
        {{"miss", "absence", "empty", "void", "hollow"}, 8, 0.7f},
        
        // Anger cluster  
        {{"angry", "anger", "rage", "furious", "mad", "pissed"}, 20, 0.9f},
        {{"frustrated", "frustration", "annoyed", "irritated"}, 22, 0.8f},
        {{"resentment", "bitter", "betrayed"}, 25, 0.85f},
        
        // Fear/Anxiety cluster
        {{"afraid", "fear", "scared", "terrified", "terror"}, 40, 0.9f},
        {{"anxious", "anxiety", "worried", "nervous", "panic"}, 42, 0.85f},
        {{"dread", "dreading"}, 45, 0.8f},
        
        // Sadness cluster
        {{"sad", "sadness", "unhappy", "depressed", "down"}, 4, 0.8f},
        {{"melancholy", "melancholic", "blue", "gloomy"}, 2, 0.85f},
        {{"lonely", "loneliness", "alone", "isolated"}, 8, 0.8f},
        {{"longing", "yearn", "yearning", "want", "wish"}, 6, 0.75f},
        
        // Joy cluster
        {{"happy", "happiness", "joy", "joyful", "elated"}, 62, 0.85f},
        {{"excited", "excitement", "thrilled"}, 90, 0.8f},
        {{"peaceful", "peace", "calm", "serene", "tranquil"}, 65, 0.8f},
        {{"grateful", "gratitude", "thankful"}, 68, 0.85f},
        {{"hope", "hopeful", "optimistic"}, 69, 0.8f},
        
        // Love/Trust cluster
        {{"love", "loving", "adore", "cherish"}, 80, 0.9f},
        {{"tender", "tenderness", "gentle", "soft"}, 83, 0.75f},
        
        // Complex emotions
        {{"bittersweet", "mixed", "complicated"}, 200, 0.85f},
        {{"nostalgic", "nostalgia", "remember", "memories"}, 201, 0.8f},
        {{"cathartic", "catharsis", "release", "letting go"}, 202, 0.8f},
    };
    
    // Score each emotion cluster
    int bestEmotionId = 2;  // Default to melancholy
    float bestScore = 0.0f;
    
    for (const auto& clue : clues) {
        float match = matchKeywords(text, clue.keywords);
        float score = match * clue.confidence * wound.intensity;
        
        if (score > bestScore) {
            bestScore = score;
            bestEmotionId = clue.primaryEmotionId;
        }
    }
    
    // If no strong match, use valence estimation from word sentiment
    if (bestScore < 0.2f) {
        // Very simple sentiment: count positive vs negative words
        std::vector<std::string> negWords = {"not", "never", "no", "bad", "wrong", "hurt", "pain"};
        std::vector<std::string> posWords = {"good", "well", "better", "beautiful", "light"};
        
        float negScore = matchKeywords(text, negWords);
        float posScore = matchKeywords(text, posWords);
        
        float estimatedValence = (posScore - negScore) * 2.0f;  // -1 to 1 roughly
        estimatedValence = std::clamp(estimatedValence, -1.0f, 1.0f);
        
        return thesaurus_.findNearest(estimatedValence, wound.intensity * 0.7f, wound.intensity);
    }
    
    return thesaurus_.findById(bestEmotionId).value_or(
        thesaurus_.findNearest(-0.3f, 0.3f, wound.intensity)
    );
}

float IntentPipeline::matchKeywords(const std::string& text, 
                                     const std::vector<std::string>& keywords) const {
    float matches = 0.0f;
    for (const auto& keyword : keywords) {
        if (text.find(keyword) != std::string::npos) {
            matches += 1.0f;
        }
    }
    return matches / static_cast<float>(keywords.size());
}

std::vector<RuleBreak> IntentPipeline::generateRuleBreaks(const EmotionNode& emotion) {
    std::vector<RuleBreak> breaks;
    
    // HIGH INTENSITY → Dynamics rule-breaks
    if (emotion.intensity > 0.7f) {
        breaks.push_back({
            RuleBreakType::Dynamics,
            emotion.intensity,
            "Extreme dynamic contrasts",
            "Intense emotions demand breaking the 'smooth dynamics' convention"
        });
    }
    
    // NEGATIVE VALENCE + HIGH AROUSAL → Harmony rule-breaks
    if (emotion.valence < -0.5f) {
        float severity = std::abs(emotion.valence);
        breaks.push_back({
            RuleBreakType::Harmony,
            severity,
            "Dissonant intervals and unresolved tensions",
            "Negative emotions reject the comfort of conventional resolution"
        });
        
        if (emotion.arousal > 0.6f) {
            breaks.push_back({
                RuleBreakType::Harmony,
                severity * 0.8f,
                "Chromatic voice leading",
                "Agitation demands chromatic tension"
            });
        }
    }
    
    // HIGH AROUSAL → Rhythm rule-breaks
    if (emotion.arousal > 0.7f) {
        breaks.push_back({
            RuleBreakType::Rhythm,
            emotion.arousal,
            "Syncopation and displaced accents",
            "High arousal breaks the predictable downbeat pattern"
        });
    }
    
    // LOW AROUSAL + NEGATIVE VALENCE → Tempo rule-breaks
    if (emotion.arousal < 0.3f && emotion.valence < -0.3f) {
        breaks.push_back({
            RuleBreakType::Rhythm,
            0.6f,
            "Rubato and tempo drift",
            "Sadness and numbness reject metronomic rigidity"
        });
    }
    
    // COMPLEX EMOTIONS (high intensity + mid valence) → Melody rule-breaks
    if (emotion.intensity > 0.6f && std::abs(emotion.valence) < 0.4f) {
        breaks.push_back({
            RuleBreakType::Melody,
            emotion.intensity * 0.7f,
            "Wide intervallic leaps",
            "Emotional complexity requires melodic unpredictability"
        });
    }
    
    return breaks;
}

IntentResult IntentPipeline::compileMusicalParams(
    const Wound& wound,
    const EmotionNode& emotion,
    const std::vector<RuleBreak>& ruleBreaks
) {
    IntentResult result;
    result.wound = wound;
    result.emotion = emotion;
    result.ruleBreaks = ruleBreaks;
    
    // Base parameters from emotion
    result.mode = thesaurus_.suggestMode(emotion);
    result.tempo = thesaurus_.suggestTempoModifier(emotion);
    result.dynamicRange = thesaurus_.suggestDynamicRange(emotion);
    
    // Default safe values
    result.allowDissonance = false;
    result.syncopationLevel = 0.3f;
    result.humanization = 0.4f;
    
    // Apply rule breaks to override parameters
    for (const auto& rb : ruleBreaks) {
        switch (rb.type) {
            case RuleBreakType::Harmony:
                result.allowDissonance = true;
                break;
                
            case RuleBreakType::Rhythm:
                result.syncopationLevel = std::max(result.syncopationLevel, rb.severity);
                result.humanization = std::max(result.humanization, rb.severity * 0.8f);
                break;
                
            case RuleBreakType::Dynamics:
                result.dynamicRange = std::max(result.dynamicRange, rb.severity);
                break;
                
            case RuleBreakType::Melody:
            case RuleBreakType::Form:
                // These affect generation algorithms, not direct parameters
                break;
        }
    }
    
    return result;
}

} // namespace kelly
