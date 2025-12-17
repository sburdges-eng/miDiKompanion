#!/usr/bin/env python3
"""
Penta Core - Music Theory Rule-Breaking Teacher Demo
=====================================================

Demonstrates:
1. Access comprehensive voice leading, harmony, and counterpoint rules
2. Filter rules by musical context (classical, jazz, contemporary)
3. Filter rules by severity level
4. Use the teacher to demonstrate rule violations
5. Access specific species counterpoint rules

Usage:
    python demo.py
"""

from penta_core.teachers import (
    RuleBreakingTeacher,
    CounterpointTeacher,
)
from penta_core.rules import (
    VoiceLeadingRules,
    HarmonyRules,
    CounterpointRules,
    RhythmRules,
    RuleSeverity,
    Species,
)


def main():
    print("=" * 70)
    print("Penta Core - Music Theory Rule-Breaking Teacher")
    print("=" * 70)
    print()
    
    # Initialize the teacher
    teacher = RuleBreakingTeacher()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 1: Access voice leading rules by context
    # ═══════════════════════════════════════════════════════════════════════════
    print("1. VOICE LEADING RULES - CLASSICAL CONTEXT")
    print("-" * 70)
    classical_rules = VoiceLeadingRules.get_rules_by_context("classical")
    
    for category, rules in classical_rules.items():
        if rules:  # Only show categories with rules
            print(f"\n{category.replace('_', ' ').title()}:")
            for rule_name, rule_data in list(rules.items())[:2]:  # Show first 2
                print(f"  • {rule_data['name']}")
                print(f"    {rule_data['description']}")
                print(f"    Severity: {rule_data['severity'].value}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 2: Filter rules by severity
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n2. STRICT RULES ONLY")
    print("-" * 70)
    strict_rules = VoiceLeadingRules.get_rules_by_severity(RuleSeverity.STRICT)
    
    for category, rules in strict_rules.items():
        if rules:
            print(f"\n{category.replace('_', ' ').title()}:")
            for rule_name, rule_data in rules.items():
                print(f"  • {rule_data['name']}")
                print(f"    Reason: {rule_data['reason']}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 3: Jazz-specific rules
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n3. JAZZ VOICE LEADING RULES")
    print("-" * 70)
    jazz_rules = VoiceLeadingRules.get_rules_by_context("jazz")
    
    if "jazz" in jazz_rules and jazz_rules["jazz"]:
        for rule_name, rule_data in jazz_rules["jazz"].items():
            print(f"\n• {rule_data['name']}")
            print(f"  {rule_data['description']}")
            print(f"  Severity in jazz: {rule_data['severity'].value}")
    else:
        print("  Jazz-specific rules are in the 'jazz' category:")
        # Show jazz-related rules from all categories
        for category, rules in jazz_rules.items():
            if rules:
                for rule_name, rule_data in rules.items():
                    print(f"  • {rule_data['name']} (Severity: {rule_data['severity'].value})")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 4: Rule-breaking demonstrations
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n4. RULE-BREAKING DEMONSTRATIONS")
    print("-" * 70)
    
    demo = teacher.demonstrate_rule_break("parallel_fifths")
    print(f"\nRule: {demo['rule']['name'] if demo['rule'] else 'Parallel Fifths'}")
    print(f"\nFamous examples of breaking this rule:")
    
    for ex in demo['examples'][:3]:
        print(f"\n  {ex['artist']} - {ex['piece']}")
        print(f"    What they did: {ex['notation_detail']}")
        print(f"    Why it works: {ex['why_it_works']}")
        print(f"    Emotional effect: {ex['emotional_effect']}")
    
    print(f"\n  Key insight: {demo['key_insight']}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 5: Get rule-breaks for an emotion
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n5. RULE-BREAKS FOR GRIEF")
    print("-" * 70)
    
    grief_examples = teacher.get_examples_for_emotion("grief")
    print(f"\nRules to break for expressing grief:")
    
    for ex in grief_examples[:5]:
        print(f"\n  • Break: {ex['rule_name']}")
        print(f"    Example: {ex['artist']} - {ex['piece']}")
        print(f"    How: {ex['how'][:80]}...")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 6: Species counterpoint rules
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n6. SPECIES COUNTERPOINT RULES")
    print("-" * 70)
    
    cp_teacher = CounterpointTeacher()
    
    for species in [Species.FIRST, Species.FOURTH]:
        overview = cp_teacher.get_species_overview(species)
        print(f"\n{overview['name']}:")
        print(f"  Ratio: {overview['ratio']}")
        print(f"  Difficulty: {overview['difficulty']}")
        print(f"  Focus: {overview['focus']}")
        print(f"  Key concepts:")
        for concept in overview['key_concepts'][:3]:
            print(f"    - {concept}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 7: Harmony rules with modal interchange
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n7. HARMONY RULES - MODAL INTERCHANGE")
    print("-" * 70)
    
    rule = HarmonyRules.get_rule("modal_interchange")
    if rule:
        print(f"\nRule: {rule.name}")
        print(f"Description: {rule.description}")
        print(f"Severity: {rule.severity.value}")
        print(f"\nFamous examples:")
        for ex in rule.examples:
            print(f"  • {ex.get('artist')} - {ex.get('piece')}")
            print(f"    {ex.get('detail')}")
        print(f"\nEmotional uses: {', '.join(rule.emotional_uses)}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 8: Rhythm rules and groove pockets
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n8. RHYTHM RULES & GROOVE POCKETS")
    print("-" * 70)
    
    print("\nGenre pocket templates (from DAiW):")
    for genre in ["funk", "boom_bap", "dilla", "lofi_grief"]:
        pocket = RhythmRules.get_genre_pocket(genre)
        if pocket:
            print(f"\n  {genre.upper()}:")
            print(f"    Swing: {pocket['swing']*100:.0f}%")
            print(f"    Kick offset: {pocket['kick_offset_ms']:+d}ms")
            print(f"    Snare offset: {pocket['snare_offset_ms']:+d}ms")
            print(f"    Character: {pocket['character']}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Example 9: Full lesson creation
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n9. INTERACTIVE LESSON: MODAL INTERCHANGE")
    print("-" * 70)
    
    lesson = teacher.create_lesson("modal_interchange")
    print(f"\nLesson: {lesson.title}")
    print(f"\nExplanation: {lesson.explanation}")
    print(f"\nExercise: {lesson.exercise}")
    print(f"\nKey insight: {lesson.key_insight}")
    
    print()
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nPhilosophy reminders:")
    print("  • 'Interrogate Before Generate' - Understand emotion first")
    print("  • 'Every Rule-Break Needs Justification'")
    print("  • 'The wrong note played with conviction is the right note'")
    print("  • 'The grid is just a suggestion. The pocket is where life happens.'")


if __name__ == "__main__":
    main()
