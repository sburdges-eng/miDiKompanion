#pragma once

/**
 * Kelly Vocal Synthesis Engine
 * C++ header for JUCE integration
 * Mirrors Python implementation in kelly_vocal_system.py
 */

#include <vector>
#include <string>
#include <map>
#include <complex>
#include <cmath>
#include <optional>

namespace kelly {
namespace vocal {

// =============================================================================
// VAD SYSTEM
// =============================================================================

struct VADVector {
    float valence = 0.0f;    // -1 to +1
    float arousal = 0.5f;    // 0 to 1
    float dominance = 0.5f;  // -1 to +1
    
    float energy() const {
        return arousal * (1.0f + std::abs(valence));
    }
    
    float tension() const {
        return std::abs(valence) * (1.0f - dominance);
    }
    
    float stability() const {
        float magnitude = std::sqrt(valence*valence + arousal*arousal + dominance*dominance);
        return 1.0f - magnitude / std::sqrt(3.0f);
    }
    
    float distance(const VADVector& other) const {
        return std::sqrt(
            std::pow(valence - other.valence, 2) +
            std::pow(arousal - other.arousal, 2) +
            std::pow(dominance - other.dominance, 2)
        );
    }
    
    VADVector blend(const VADVector& other, float weight) const {
        return {
            valence * (1-weight) + other.valence * weight,
            arousal * (1-weight) + other.arousal * weight,
            dominance * (1-weight) + other.dominance * weight
        };
    }
};

// =============================================================================
// BIOMETRIC INPUT
// =============================================================================

struct BiometricData {
    float heartRate = 75.0f;           // BPM
    float heartRateVariability = 50.0f; // ms
    float skinConductance = 5.0f;       // microsiemens
    float temperature = 36.5f;          // Celsius
    double timestamp = 0.0;
};

inline VADVector biometricToVAD(const BiometricData& bio) {
    float hrNorm = (bio.heartRate - 60.0f) / 60.0f;
    float arousal = std::clamp(0.5f + hrNorm * 0.5f, 0.0f, 1.0f);
    
    float hrvNorm = (bio.heartRateVariability - 30.0f) / 40.0f;
    float dominance = std::clamp(hrvNorm, -1.0f, 1.0f);
    
    float edaStress = (bio.skinConductance - 5.0f) / 10.0f;
    float valence = std::clamp(bio.heartRateVariability / 50.0f - edaStress * 0.5f, -1.0f, 1.0f);
    
    return {valence, arousal, dominance};
}

// =============================================================================
// VOICE PARAMETERS
// =============================================================================

struct VoiceParameters {
    float f0Base = 200.0f;
    float f0Modulated = 200.0f;
    float vibratoRate = 5.0f;      // Hz
    float vibratoDepth = 2.0f;     // semitones
    float formantF1 = 500.0f;
    float formantF2 = 1500.0f;
    float formantF3 = 2500.0f;
    float spectralTilt = -6.0f;    // dB/octave
    float brightness = 0.5f;
    float amplitude = 0.7f;
    float jitter = 0.01f;
    float shimmer = 0.02f;
    float speechRate = 1.0f;
};

inline VoiceParameters vadToVoice(const VADVector& vad, float fBase = 200.0f) {
    VoiceParameters params;
    params.f0Base = fBase;
    
    float V = vad.valence;
    float A = vad.arousal;
    float D = vad.dominance;
    
    // Pitch
    params.f0Modulated = fBase * (1.0f + 0.5f * A + 0.3f * V);
    
    // Amplitude
    params.amplitude = std::clamp(0.5f + 0.3f * D + 0.2f * A, 0.1f, 1.0f);
    
    // Vibrato
    params.vibratoRate = 4.5f + 2.5f * A;
    params.vibratoDepth = std::clamp(1.5f + V + 0.5f * A, 0.5f, 4.0f);
    
    // Formants
    params.formantF1 = 500.0f * (1.0f + 0.2f * V - 0.1f * D);
    params.formantF2 = 1500.0f * (1.0f + 0.15f * V + 0.1f * A);
    params.formantF3 = 2500.0f * (1.0f + 0.1f * V);
    
    // Timbre
    params.spectralTilt = -6.0f + 4.0f * V - 3.0f * A;
    params.brightness = 0.5f + 0.3f * V + 0.2f * A;
    
    // Speech rate
    params.speechRate = (V < 0) ? 1.0f + 0.5f * A - 0.3f * (1.0f - V) : 1.0f + 0.3f * A;
    
    // Micro-perturbations
    params.jitter = 0.01f + 0.02f * std::max(0.0f, -V) + 0.01f * A;
    params.shimmer = 0.02f + 0.03f * std::max(0.0f, -V) + 0.02f * A;
    
    return params;
}

// =============================================================================
// PHONEME DATA
// =============================================================================

struct Phoneme {
    std::string ipa;
    std::string arpabet;
    float durationMs = 80.0f;
    bool isVoiced = true;
    bool isVowel = false;
    float formantF1 = 500.0f;
    float formantF2 = 1500.0f;
    float formantF3 = 2500.0f;
};

struct Syllable {
    std::string text;
    std::vector<Phoneme> phonemes;
    int stress = 0;  // 0=none, 1=secondary, 2=primary
    float durationBeats = 0.5f;
    float startBeat = 0.0f;
};

// =============================================================================
// VOCAL EXPRESSION
// =============================================================================

struct VocalExpression {
    float breathiness = 0.3f;
    float vibratoRate = 5.5f;
    float vibratoDepth = 20.0f;    // cents
    float vibratoDelay = 0.15f;    // seconds
    float pitchDrift = 0.0f;       // cents
    float dynamics = 0.7f;
    float attackTime = 0.02f;
    float releaseTime = 0.1f;
};

inline VocalExpression emotionToVocalExpression(float v, float a, float d) {
    return {
        0.3f + 0.4f * (1.0f - d) + 0.2f * (1.0f - a),  // breathiness
        4.5f + 2.0f * a,                                 // vibratoRate
        15.0f + 25.0f * (1.0f - v) * (1.0f - d),        // vibratoDepth
        0.1f + 0.15f * (1.0f - a),                       // vibratoDelay
        5.0f * (1.0f - v) - 5.0f * v,                   // pitchDrift
        0.4f + 0.5f * a + 0.1f * d,                     // dynamics
        0.01f + 0.03f * (1.0f - a),                     // attackTime
        0.05f + 0.15f * (1.0f - a)                      // releaseTime
    };
}

// =============================================================================
// VOCAL NOTE
// =============================================================================

struct VocalNote {
    int pitchMidi = 60;
    float startTime = 0.0f;     // seconds
    float duration = 0.5f;      // seconds
    VocalExpression expression;
    float pitchBendCents = 0.0f;
    float portamentoTime = 0.0f;
    std::optional<Syllable> syllable;
    
    float endTime() const { return startTime + duration; }
};

struct VocalPhrase {
    std::vector<VocalNote> notes;
    int phraseId = 0;
    bool breathBefore = true;
    float breathDuration = 0.3f;
};

// =============================================================================
// RESONANCE & TRENDS
// =============================================================================

struct ResonanceMetrics {
    float coherence = 0.0f;
    float emotionBiometricMatch = 0.0f;
    float temporalStability = 0.0f;
    float quantumCoherence = 0.0f;
};

struct TrendMetrics {
    float valenceTrend = 0.0f;
    float arousalTrend = 0.0f;
    float dominanceTrend = 0.0f;
    float confidence = 0.0f;
    std::optional<VADVector> prediction;
};

// =============================================================================
// QUANTUM EMOTION STATE
// =============================================================================

enum class EmotionBasis {
    Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation
};

class QuantumEmotionState {
public:
    std::map<EmotionBasis, std::complex<float>> amplitudes;
    
    QuantumEmotionState() {
        float amp = 1.0f / std::sqrt(8.0f);
        for (int i = 0; i < 8; ++i) {
            amplitudes[static_cast<EmotionBasis>(i)] = {amp, 0.0f};
        }
    }
    
    void normalize() {
        float total = 0.0f;
        for (const auto& [e, a] : amplitudes) {
            total += std::norm(a);
        }
        if (total > 0) {
            float factor = 1.0f / std::sqrt(total);
            for (auto& [e, a] : amplitudes) {
                a *= factor;
            }
        }
    }
    
    std::map<EmotionBasis, float> probabilities() const {
        std::map<EmotionBasis, float> probs;
        for (const auto& [e, a] : amplitudes) {
            probs[e] = std::norm(a);
        }
        return probs;
    }
    
    float coherence() const {
        std::complex<float> total{0, 0};
        for (const auto& [e, a] : amplitudes) {
            total += a;
        }
        return std::abs(total) / std::sqrt(static_cast<float>(amplitudes.size()));
    }
    
    float entropy() const {
        float s = 0.0f;
        for (const auto& [e, a] : amplitudes) {
            float p = std::norm(a);
            if (p > 1e-10f) {
                s -= p * std::log(p);
            }
        }
        return s;
    }
};

// =============================================================================
// CIRCADIAN ADJUSTMENT
// =============================================================================

inline VADVector circadianAdjustment(int hour, int dayOfWeek = 0) {
    float arousalAdj = 0.0f, valenceAdj = 0.0f, dominanceAdj = 0.0f;
    
    if (hour >= 4 && hour < 6) {
        arousalAdj = -0.3f; valenceAdj = -0.1f; dominanceAdj = -0.2f;
    } else if (hour >= 6 && hour < 10) {
        arousalAdj = -0.1f + (hour - 6) * 0.05f;
        valenceAdj = 0.1f;
        dominanceAdj = -0.1f + (hour - 6) * 0.05f;
    } else if (hour >= 10 && hour < 14) {
        arousalAdj = 0.1f; valenceAdj = 0.1f; dominanceAdj = 0.1f;
    } else if (hour >= 14 && hour < 16) {
        arousalAdj = 0.2f; valenceAdj = 0.1f; dominanceAdj = 0.15f;
    } else if (hour >= 18 && hour < 22) {
        arousalAdj = -0.1f; valenceAdj = 0.05f; dominanceAdj = -0.05f;
    } else if (hour >= 22 || hour < 4) {
        arousalAdj = -0.2f; valenceAdj = -0.05f; dominanceAdj = -0.1f;
    }
    
    // Day of week
    if (dayOfWeek == 0) valenceAdj -= 0.1f;       // Monday
    else if (dayOfWeek == 4) valenceAdj += 0.1f;  // Friday
    else if (dayOfWeek >= 5) valenceAdj += 0.05f; // Weekend
    
    return {valenceAdj, arousalAdj, dominanceAdj};
}

// =============================================================================
// PITCH CONTOUR
// =============================================================================

struct PitchTarget {
    float pitchMidi;
    float timeOffset;    // seconds from note start
    float bendCents = 0.0f;
};

inline std::vector<PitchTarget> generateNoteContour(
    int basePitch,
    float duration,
    const VocalExpression& expr,
    bool isVowel = true
) {
    std::vector<PitchTarget> targets;
    
    // Attack scoop
    if (isVowel) {
        float scoopCents = -20.0f + 40.0f * expr.breathiness;
        targets.push_back({static_cast<float>(basePitch), 0.0f, scoopCents});
        targets.push_back({static_cast<float>(basePitch), expr.attackTime, 0.0f});
    } else {
        targets.push_back({static_cast<float>(basePitch), 0.0f, 0.0f});
    }
    
    // Vibrato
    if (duration > expr.vibratoDelay) {
        float vibratoStart = expr.vibratoDelay;
        float vibratoEnd = duration - expr.releaseTime;
        
        if (vibratoEnd > vibratoStart) {
            int samples = std::max(4, static_cast<int>((vibratoEnd - vibratoStart) * expr.vibratoRate * 2));
            for (int i = 0; i < samples; ++i) {
                float t = vibratoStart + (vibratoEnd - vibratoStart) * i / samples;
                float phase = 2.0f * M_PI * expr.vibratoRate * (t - vibratoStart);
                float bend = expr.vibratoDepth * std::sin(phase) + expr.pitchDrift;
                targets.push_back({static_cast<float>(basePitch), t, bend});
            }
        }
    }
    
    // Release
    if (duration > 0.1f) {
        targets.push_back({
            static_cast<float>(basePitch),
            duration - 0.02f,
            -10.0f - 10.0f * (1.0f - expr.dynamics)
        });
    }
    
    return targets;
}

// =============================================================================
// MAIN VOCAL ENGINE CLASS
// =============================================================================

class VocalEngine {
public:
    VocalEngine(float baseFreq = 200.0f) : m_baseFreq(baseFreq) {}
    
    void setVAD(float v, float a, float d) {
        m_currentVAD = {v, a, d};
        m_voiceParams = vadToVoice(m_currentVAD, m_baseFreq);
    }
    
    void setFromBiometrics(const BiometricData& bio) {
        m_currentVAD = biometricToVAD(bio);
        m_voiceParams = vadToVoice(m_currentVAD, m_baseFreq);
    }
    
    void applyCircadian(int hour, int dayOfWeek = 0) {
        auto adj = circadianAdjustment(hour, dayOfWeek);
        m_currentVAD.valence = std::clamp(m_currentVAD.valence + adj.valence, -1.0f, 1.0f);
        m_currentVAD.arousal = std::clamp(m_currentVAD.arousal + adj.arousal, 0.0f, 1.0f);
        m_currentVAD.dominance = std::clamp(m_currentVAD.dominance + adj.dominance, -1.0f, 1.0f);
        m_voiceParams = vadToVoice(m_currentVAD, m_baseFreq);
    }
    
    const VADVector& getVAD() const { return m_currentVAD; }
    const VoiceParameters& getVoiceParams() const { return m_voiceParams; }
    
    VocalExpression getExpression() const {
        return emotionToVocalExpression(
            m_currentVAD.valence,
            m_currentVAD.arousal,
            m_currentVAD.dominance
        );
    }
    
    std::vector<PitchTarget> getContour(int pitch, float duration) const {
        return generateNoteContour(pitch, duration, getExpression());
    }

private:
    float m_baseFreq = 200.0f;
    VADVector m_currentVAD;
    VoiceParameters m_voiceParams;
};

} // namespace vocal
} // namespace kelly
