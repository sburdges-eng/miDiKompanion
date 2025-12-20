#pragma once
/*
 * groove_templates.h - Groove Template Definitions
 * ===============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Core Layer: Groove template storage and lookup
 * - MIDI Layer: Used by GrooveEngine, DrumGrooveEngine (groove pattern templates)
 * - Engine Layer: Provides predefined groove patterns
 *
 * Purpose: Groove template definitions and lookup system.
 *          Provides predefined groove patterns for rhythm generation.
 *
 * Features:
 * - Groove template storage
 * - Template lookup by name
 * - Pattern definitions (time, velocity)
 * - Swing amount configuration
 */

#include <string>
#include <vector>
#include <map>

namespace kelly {

struct GrooveTemplate {
    std::string name;
    int numerator;
    int denominator;
    std::vector<std::pair<float, int>> pattern;  // time, velocity
    float swing = 0.0f;
};

class GrooveTemplates {
public:
    GrooveTemplates();
    ~GrooveTemplates() = default;

    const GrooveTemplate* getTemplate(const std::string& name) const;
    std::vector<std::string> getTemplateNames() const;

private:
    void initializeTemplates();
    std::map<std::string, GrooveTemplate> templates_;
};

} // namespace kelly
