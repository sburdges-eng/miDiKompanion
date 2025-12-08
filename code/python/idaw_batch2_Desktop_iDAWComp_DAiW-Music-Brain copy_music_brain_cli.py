"""
DAiW CLI - Command line interface for Music Brain toolkit

Usage:
    daiw extract <midi_file>                    Extract groove from MIDI
    daiw apply --genre <genre> <midi_file>      Apply groove template
    daiw humanize <midi_file> [options]         Apply drum humanization (Drunken Drummer)
    daiw analyze --chords <midi_file>           Analyze MIDI chord progression
    daiw analyze-audio <audio_file> [options]    Analyze audio file (BPM, key, chords, feel)
    daiw compare-audio <file1> <file2>          Compare two audio files
    daiw batch-analyze <files...>               Batch analyze multiple audio files
    daiw export-features <audio_file> -o <out>  Export audio features to JSON/CSV
    daiw generate [options]                     Generate harmony from intent or parameters
    daiw diagnose <progression>                 Diagnose harmonic issues
    daiw reharm <progression> [--style <style>] Generate reharmonizations
    daiw teach <topic>                          Interactive teaching mode

    daiw intent new [--title <title>]           Create new intent template
    daiw intent process <file>                  Generate elements from intent
    daiw intent suggest <emotion>               Suggest rules to break
    daiw intent list                            List all rule-breaking options
    daiw intent validate <file>                 Validate intent file

Humanization Styles:
    tight   - Minimal drift, confident (complexity=0.1, vulnerability=0.2)
    natural - Human feel, balanced (complexity=0.4, vulnerability=0.5)
    loose   - Relaxed, laid back (complexity=0.6, vulnerability=0.6)
    drunk   - Maximum chaos, fragile (complexity=0.9, vulnerability=0.8)
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Lazy imports to speed up CLI startup
def get_groove_module():
    from music_brain.groove import extract_groove, apply_groove
    return extract_groove, apply_groove


def get_humanize_module():
    from music_brain.groove import (
        humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
        list_presets, settings_from_preset, get_preset
    )
    return (humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
            list_presets, settings_from_preset, get_preset)

def get_structure_module():
    from music_brain.structure import analyze_chords, detect_sections
    return analyze_chords, detect_sections

def get_session_module():
    from music_brain.session import RuleBreakingTeacher
    return RuleBreakingTeacher


def get_intent_module():
    from music_brain.session.intent_schema import (
        CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
        SystemDirective, suggest_rule_break, validate_intent, list_all_rules
    )
    from music_brain.session.intent_processor import IntentProcessor, process_intent
    return (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
            SystemDirective, suggest_rule_break, validate_intent, list_all_rules,
            IntentProcessor, process_intent)


def get_harmony_module():
    from music_brain.harmony import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony
    return HarmonyGenerator, HarmonyResult, generate_midi_from_harmony


def cmd_extract(args):
    """Extract groove from MIDI file."""
    extract_groove, _ = get_groove_module()
    
    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return 1
    
    print(f"Extracting groove from: {midi_path}")
    groove = extract_groove(str(midi_path))
    
    output_path = midi_path.stem + "_groove.json"
    if args.output:
        output_path = args.output
    
    with open(output_path, 'w') as f:
        json.dump(groove.to_dict(), f, indent=2)
    
    print(f"Groove saved to: {output_path}")
    print(f"  Timing deviation: {groove.timing_stats['mean_deviation_ms']:.1f}ms avg")
    print(f"  Velocity range: {groove.velocity_stats['min']}-{groove.velocity_stats['max']}")
    print(f"  Swing factor: {groove.swing_factor:.2f}")
    return 0


def cmd_apply(args):
    """Apply groove template to MIDI file."""
    _, apply_groove = get_groove_module()

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return 1

    print(f"Applying {args.genre} groove to: {midi_path}")

    output_path = args.output or f"{midi_path.stem}_grooved.mid"
    apply_groove(str(midi_path), genre=args.genre, output=output_path, intensity=args.intensity)

    print(f"Output saved to: {output_path}")
    return 0


def cmd_humanize(args):
    """Apply drum humanization to MIDI file."""
    (humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
     list_presets, settings_from_preset, get_preset) = get_humanize_module()

    # Handle list-presets subcommand
    if hasattr(args, 'list_presets') and args.list_presets:
        presets = list_presets()
        print("\n=== Available Humanization Presets ===\n")
        for preset_name in sorted(presets):
            preset_data = get_preset(preset_name)
            desc = preset_data.get("description", "No description")
            groove = preset_data.get("groove_settings", {})
            c = groove.get("complexity", 0.5)
            v = groove.get("vulnerability", 0.5)
            print(f"  {preset_name}")
            print(f"    {desc}")
            print(f"    complexity={c:.2f}, vulnerability={v:.2f}")
            print()
        return 0

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return 1

    # Determine humanization approach
    if args.preset:
        # Load from preset
        try:
            settings = settings_from_preset(args.preset)
            preset_data = get_preset(args.preset)
            complexity = settings.complexity
            vulnerability = settings.vulnerability
            print(f"Humanizing drums with '{args.preset}' preset: {midi_path}")
            print(f"  ({preset_data.get('description', '')})")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.style:
        # Quick style preset
        print(f"Humanizing drums with '{args.style}' style: {midi_path}")
        complexity_map = {
            "tight": (0.1, 0.2),
            "natural": (0.4, 0.5),
            "loose": (0.6, 0.6),
            "drunk": (0.9, 0.8),
        }
        complexity, vulnerability = complexity_map.get(args.style, (0.4, 0.5))
        settings = GrooveSettings(complexity=complexity, vulnerability=vulnerability)
    else:
        # Manual complexity/vulnerability
        complexity = args.complexity
        vulnerability = args.vulnerability
        print(f"Humanizing drums (complexity={complexity:.2f}, vulnerability={vulnerability:.2f}): {midi_path}")
        settings = GrooveSettings(complexity=complexity, vulnerability=vulnerability)

    # Apply optional overrides
    if args.no_ghost_notes:
        settings.enable_ghost_notes = False

    output_path = args.output or f"{midi_path.stem}_humanized.mid"

    result_path = humanize_midi_file(
        input_path=str(midi_path),
        output_path=output_path,
        complexity=complexity,
        vulnerability=vulnerability,
        drum_channel=args.channel,
        settings=settings,
        seed=args.seed,
    )

    print(f"\n=== Drum Humanization Applied ===")
    print(f"  Complexity:    {complexity:.2f} (timing chaos)")
    print(f"  Vulnerability: {vulnerability:.2f} (dynamic fragility)")
    print(f"  Ghost notes:   {'enabled' if settings.enable_ghost_notes else 'disabled'}")
    print(f"  Output:        {result_path}")

    return 0


def cmd_analyze(args):
    """Analyze chord progression in MIDI file."""
    analyze_chords, detect_sections = get_structure_module()
    
    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return 1
    
    if args.chords:
        print(f"Analyzing chords in: {midi_path}")
        progression = analyze_chords(str(midi_path))
        
        print("\n=== Chord Analysis ===")
        print(f"Key: {progression.key}")
        print(f"Progression: {' - '.join(progression.chords)}")
        print(f"Roman numerals: {' - '.join(progression.roman_numerals)}")
        
        if progression.borrowed_chords:
            print(f"\nBorrowed chords detected:")
            for chord, source in progression.borrowed_chords.items():
                print(f"  {chord} ← borrowed from {source}")
    
    if args.sections:
        print(f"\nDetecting sections in: {midi_path}")
        sections = detect_sections(str(midi_path))
        
        print("\n=== Section Analysis ===")
        for section in sections:
            print(f"  {section.name}: bars {section.start_bar}-{section.end_bar} (energy: {section.energy:.2f})")
    
    return 0


def cmd_generate(args):
    """Generate harmony from intent or basic parameters."""
    HarmonyGenerator, HarmonyResult, generate_midi_from_harmony = get_harmony_module()

    generator = HarmonyGenerator()

    if args.intent_file:
        # Generate from intent JSON file
        intent_file = Path(args.intent_file)
        if not intent_file.exists():
            print(f"Error: Intent file not found: {intent_file}")
            return 1

        with open(intent_file, 'r') as f:
            intent_data = json.load(f)

        # Import intent schema to load from dict
        (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
         SystemDirective, suggest_rule_break, validate_intent, list_all_rules,
         IntentProcessor, process_intent) = get_intent_module()

        intent = CompleteSongIntent.from_dict(intent_data)

        print(f"Generating harmony from intent: {intent_file}")
        harmony = generator.generate_from_intent(intent)
    else:
        # Generate basic progression
        key = args.key or "C"
        mode = args.mode or "major"
        pattern = args.pattern or "I-V-vi-IV"

        print(f"Generating basic progression:")
        print(f"  Key: {key}")
        print(f"  Mode: {mode}")
        print(f"  Pattern: {pattern}")

        harmony = generator.generate_basic_progression(
            key=key,
            mode=mode,
            pattern=pattern
        )

    # Display results
    print("\n=== Generated Harmony ===")
    print(f"Key: {harmony.key} {harmony.mode}")
    print(f"Progression: {' - '.join(harmony.chords)}")
    if harmony.rule_break_applied:
        print(f"Rule break: {harmony.rule_break_applied}")
        print(f"Justification: {harmony.emotional_justification}")

    # Output to MIDI if requested
    if args.output:
        output_path = Path(args.output)
        tempo = args.tempo or 82

        generate_midi_from_harmony(harmony, str(output_path), tempo_bpm=tempo)
        print(f"\n✓ MIDI saved: {output_path}")
        print(f"  Tempo: {tempo} BPM")

    return 0


def cmd_diagnose(args):
    """Diagnose issues in a chord progression string."""
    from music_brain.structure.progression import diagnose_progression
    
    progression = args.progression
    print(f"Diagnosing: {progression}")
    
    diagnosis = diagnose_progression(progression)
    
    print("\n=== Harmonic Diagnosis ===")
    print(f"Key estimate: {diagnosis['key']}")
    print(f"Mode: {diagnosis['mode']}")
    
    if diagnosis['issues']:
        print("\nPotential issues:")
        for issue in diagnosis['issues']:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ No obvious issues detected")
    
    if diagnosis['suggestions']:
        print("\nSuggestions:")
        for suggestion in diagnosis['suggestions']:
            print(f"  → {suggestion}")
    
    return 0


def cmd_reharm(args):
    """Generate reharmonization suggestions."""
    from music_brain.structure.progression import generate_reharmonizations
    
    progression = args.progression
    style = args.style or "jazz"
    
    print(f"Reharmonizing: {progression}")
    print(f"Style: {style}")
    
    suggestions = generate_reharmonizations(progression, style=style, count=args.count)
    
    print("\n=== Reharmonization Suggestions ===")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {' - '.join(suggestion['chords'])}")
        print(f"   Technique: {suggestion['technique']}")
        print(f"   Mood shift: {suggestion['mood']}")
    
    return 0


def cmd_teach(args):
    """Interactive teaching mode."""
    RuleBreakingTeacher = get_session_module()
    
    topic = args.topic.lower().replace("-", "_").replace(" ", "_")
    
    valid_topics = [
        "rulebreaking", "rule_breaking", "borrowed", "borrowed_chords",
        "modal_mixture", "substitutions", "rhythm", "production"
    ]
    
    if topic not in valid_topics:
        print(f"Unknown topic: {args.topic}")
        print(f"Available topics: {', '.join(valid_topics)}")
        return 1
    
    teacher = RuleBreakingTeacher()
    
    if args.quick:
        # Quick single lesson
        teacher.quick_lesson(topic)
    else:
        # Interactive mode
        teacher.interactive_session(topic)
    
    return 0


def cmd_analyze_audio(args):
    """Analyze audio file for BPM, key, chords, and feel."""
    import json
    from pathlib import Path
    
    try:
        from music_brain.audio.analyzer import AudioAnalyzer
        from music_brain.audio.chord_detection import ChordDetector
    except ImportError as e:
        print(f"Error: Audio analysis requires librosa. Install with: pip install librosa")
        return 1
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    print(f"Analyzing audio file: {audio_path}")
    
    analyzer = AudioAnalyzer()
    
    # Perform analysis
    analysis = analyzer.analyze_file(
        str(audio_path),
        detect_key=args.key,
        detect_bpm=args.bpm,
        extract_features_flag=args.features,
        analyze_segments=args.segments,
        num_segments=args.segment_count,
        max_duration=args.max_duration,
    )
    
    # Display results
    print("\n=== Audio Analysis ===")
    print(f"Duration: {analysis.duration_seconds:.2f} seconds")
    print(f"Sample rate: {analysis.sample_rate} Hz")
    
    if analysis.bpm_result:
        print(f"\n🎵 Tempo:")
        print(f"  BPM: {analysis.bpm_result.bpm:.1f}")
        print(f"  Confidence: {analysis.bpm_result.confidence:.2%}")
        if analysis.bpm_result.tempo_alternatives:
            print(f"  Alternatives: {', '.join(f'{t:.1f}' for t in analysis.bpm_result.tempo_alternatives[:3])}")
    
    if analysis.key_result:
        print(f"\n🎹 Key:")
        print(f"  Key: {analysis.key_result.full_key}")
        print(f"  Confidence: {analysis.key_result.confidence:.2%}")
    
    if analysis.features:
        print(f"\n🎭 Feel Analysis:")
        print(f"  Tempo: {analysis.features.tempo_bpm:.1f} BPM")
        print(f"  Dynamic range: {analysis.features.dynamic_range_db:.1f} dB")
        print(f"  Swing estimate: {analysis.features.swing_estimate:.2f}")
    
    if analysis.feature_summary:
        print(f"\n📊 Feature Summary:")
        for key, value in sorted(analysis.feature_summary.items()):
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ')}: {value:.4f}")
            else:
                print(f"  {key.replace('_', ' ')}: {value}")
    
    if analysis.segments:
        print(f"\n🧩 Structural Segments:")
        for seg in analysis.segments:
            print(
                f"  {seg.label or 'segment'} :: "
                f"{seg.start_time:.2f}s – {seg.end_time:.2f}s "
                f"(energy={seg.energy:.3f})"
            )
    
    # Chord detection if requested
    if args.chords:
        print(f"\n🎼 Chord Detection:")
        detector = ChordDetector()
        progression = detector.detect_progression(str(audio_path), max_duration=args.max_duration)
        
        if progression.chords:
            print(f"  Detected chords: {' - '.join(progression.chord_sequence)}")
            print(f"  Confidence: {progression.confidence:.2%}")
            if progression.estimated_key:
                print(f"  Estimated key: {progression.estimated_key}")
        else:
            print("  No chords detected with sufficient confidence")
    
    # Save output if requested
    if args.output:
        output_data = analysis.to_dict()
        if args.chords:
            detector = ChordDetector()
            progression = detector.detect_progression(str(audio_path), max_duration=args.max_duration)
            output_data['chord_progression'] = progression.to_dict()
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Analysis saved to: {args.output}")
    
    return 0


def cmd_compare_audio(args):
    """Compare two audio files for BPM, key, and feel differences."""
    import json
    from pathlib import Path
    
    try:
        from music_brain.audio.analyzer import AudioAnalyzer
        from music_brain.audio.feel import compare_feel
    except ImportError as e:
        print(f"Error: Audio analysis requires librosa. Install with: pip install librosa")
        return 1
    
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    
    if not file1_path.exists():
        print(f"Error: Audio file not found: {file1_path}")
        return 1
    if not file2_path.exists():
        print(f"Error: Audio file not found: {file2_path}")
        return 1
    
    print(f"Comparing audio files:")
    print(f"  File 1: {file1_path.name}")
    print(f"  File 2: {file2_path.name}\n")
    
    analyzer = AudioAnalyzer()
    
    # Analyze both files
    print("Analyzing file 1...")
    analysis1 = analyzer.analyze_file(str(file1_path))
    
    print("Analyzing file 2...")
    analysis2 = analyzer.analyze_file(str(file2_path))
    
    # Comparison results
    comparison = {
        "file1": str(file1_path),
        "file2": str(file2_path),
        "comparison": {}
    }
    
    print("\n=== Audio Comparison ===\n")
    
    # Duration comparison
    dur_diff = abs(analysis1.duration_seconds - analysis2.duration_seconds)
    print(f"⏱️  Duration:")
    print(f"  File 1: {analysis1.duration_seconds:.2f}s")
    print(f"  File 2: {analysis2.duration_seconds:.2f}s")
    print(f"  Difference: {dur_diff:.2f}s")
    comparison["comparison"]["duration"] = {
        "file1": analysis1.duration_seconds,
        "file2": analysis2.duration_seconds,
        "difference": dur_diff
    }
    
    # BPM comparison
    if analysis1.bpm_result and analysis2.bpm_result:
        bpm_diff = abs(analysis1.bpm_result.bpm - analysis2.bpm_result.bpm)
        print(f"\n🎵 Tempo (BPM):")
        print(f"  File 1: {analysis1.bpm_result.bpm:.1f} BPM (confidence: {analysis1.bpm_result.confidence:.2%})")
        print(f"  File 2: {analysis2.bpm_result.bpm:.1f} BPM (confidence: {analysis2.bpm_result.confidence:.2%})")
        print(f"  Difference: {bpm_diff:.1f} BPM")
        comparison["comparison"]["bpm"] = {
            "file1": analysis1.bpm_result.bpm,
            "file2": analysis2.bpm_result.bpm,
            "difference": bpm_diff,
            "file1_confidence": analysis1.bpm_result.confidence,
            "file2_confidence": analysis2.bpm_result.confidence
        }
    
    # Key comparison
    if analysis1.key_result and analysis2.key_result:
        key_match = (analysis1.key_result.full_key == analysis2.key_result.full_key)
        print(f"\n🎹 Key:")
        print(f"  File 1: {analysis1.key_result.full_key} (confidence: {analysis1.key_result.confidence:.2%})")
        print(f"  File 2: {analysis2.key_result.full_key} (confidence: {analysis2.key_result.confidence:.2%})")
        print(f"  Match: {'✓ Yes' if key_match else '✗ No'}")
        comparison["comparison"]["key"] = {
            "file1": analysis1.key_result.full_key,
            "file2": analysis2.key_result.full_key,
            "match": key_match,
            "file1_confidence": analysis1.key_result.confidence,
            "file2_confidence": analysis2.key_result.confidence
        }
    
    # Feel comparison
    if analysis1.features and analysis2.features:
        print(f"\n🎭 Feel Analysis:")
        print(f"  File 1 Tempo: {analysis1.features.tempo_bpm:.1f} BPM")
        print(f"  File 2 Tempo: {analysis2.features.tempo_bpm:.1f} BPM")
        print(f"  File 1 Dynamic Range: {analysis1.features.dynamic_range_db:.1f} dB")
        print(f"  File 2 Dynamic Range: {analysis2.features.dynamic_range_db:.1f} dB")
        print(f"  File 1 Swing: {analysis1.features.swing_estimate:.2f}")
        print(f"  File 2 Swing: {analysis2.features.swing_estimate:.2f}")
        
        try:
            feel_comparison = compare_feel(str(file1_path), str(file2_path))
            if isinstance(feel_comparison, dict):
                overall = feel_comparison.get('overall_similarity', 0.0)
                print(f"\n  Overall Similarity: {overall:.2%}")
                print(f"  Tempo Similarity: {feel_comparison.get('tempo_similarity', 0.0):.2%}")
                print(f"  Swing Similarity: {feel_comparison.get('swing_similarity', 0.0):.2%}")
                print(f"  Energy Similarity: {feel_comparison.get('energy_similarity', 0.0):.2%}")
                comparison["comparison"]["feel"] = feel_comparison
        except Exception as e:
            print(f"  (Feel comparison unavailable: {e})")
    
    # Feature comparison (if detailed)
    if args.detailed and analysis1.feature_summary and analysis2.feature_summary:
        print(f"\n📊 Detailed Feature Comparison:")
        all_keys = set(analysis1.feature_summary.keys()) | set(analysis2.feature_summary.keys())
        for key in sorted(all_keys):
            val1 = analysis1.feature_summary.get(key, 0)
            val2 = analysis2.feature_summary.get(key, 0)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = abs(val1 - val2)
                print(f"  {key.replace('_', ' ')}: {val1:.4f} vs {val2:.4f} (diff: {diff:.4f})")
                comparison["comparison"].setdefault("features", {})[key] = {
                    "file1": val1,
                    "file2": val2,
                    "difference": diff
                }
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Comparison saved to: {args.output}")
    
    return 0


def cmd_batch_analyze(args):
    """Batch analyze multiple audio files."""
    import json
    import csv
    import sys
    from pathlib import Path
    from glob import glob
    
    try:
        from music_brain.audio.analyzer import AudioAnalyzer
    except ImportError as e:
        print(f"Error: Audio analysis requires librosa. Install with: pip install librosa")
        return 1
    
    analyzer = AudioAnalyzer()
    results = []
    
    # Collect files
    files_to_analyze = []
    for file_pattern in args.files:
        path = Path(file_pattern)
        if path.is_file():
            files_to_analyze.append(path)
        elif path.is_dir():
            if args.recursive:
                # Recursive search
                for ext in ['*.wav', '*.mp3', '*.aiff', '*.flac', '*.m4a', '*.ogg']:
                    files_to_analyze.extend(Path(path).rglob(ext))
            else:
                # Non-recursive
                for ext in ['*.wav', '*.mp3', '*.aiff', '*.flac', '*.m4a', '*.ogg']:
                    files_to_analyze.extend(Path(path).glob(ext))
        else:
            # Try glob pattern
            files_to_analyze.extend([Path(f) for f in glob(file_pattern)])
    
    # Remove duplicates
    files_to_analyze = list(set(files_to_analyze))
    
    if not files_to_analyze:
        print("Error: No audio files found to analyze.")
        return 1
    
    print(f"Found {len(files_to_analyze)} audio file(s) to analyze...\n")
    
    # Analyze each file
    for i, file_path in enumerate(files_to_analyze, 1):
        print(f"[{i}/{len(files_to_analyze)}] Analyzing: {file_path.name}")
        
        try:
            analysis = analyzer.analyze_file(
                str(file_path),
                max_duration=args.max_duration
            )
            
            # Extract key data
            result = {
                "file": str(file_path),
                "filename": file_path.name,
                "duration_seconds": analysis.duration_seconds,
                "sample_rate": analysis.sample_rate,
            }
            
            if analysis.bpm_result:
                result["bpm"] = analysis.bpm_result.bpm
                result["bpm_confidence"] = analysis.bpm_result.confidence
            else:
                result["bpm"] = None
                result["bpm_confidence"] = None
            
            if analysis.key_result:
                result["key"] = analysis.key_result.full_key
                result["key_confidence"] = analysis.key_result.confidence
            else:
                result["key"] = None
                result["key_confidence"] = None
            
            if analysis.features:
                result["feel_tempo"] = analysis.features.tempo_bpm
                result["dynamic_range_db"] = analysis.features.dynamic_range_db
                result["swing_estimate"] = analysis.features.swing_estimate
            else:
                result["feel_tempo"] = None
                result["dynamic_range_db"] = None
                result["swing_estimate"] = None
            
            if analysis.feature_summary:
                result.update(analysis.feature_summary)
            
            results.append(result)
            print(f"  ✓ BPM: {result.get('bpm', 'N/A')}, Key: {result.get('key', 'N/A')}")
            
        except Exception as e:
            print(f"  ✗ Error analyzing {file_path.name}: {e}")
            results.append({
                "file": str(file_path),
                "filename": file_path.name,
                "error": str(e)
            })
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        if args.format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {output_path} (JSON)")
        elif args.format == 'csv':
            if results:
                # Get all possible keys
                all_keys = set()
                for r in results:
                    all_keys.update(r.keys())
                
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    writer.writerows(results)
                print(f"\n✓ Results saved to: {output_path} (CSV)")
    else:
        # Print to stdout
        if args.format == 'json':
            print("\n" + json.dumps(results, indent=2))
        elif args.format == 'csv':
            if results:
                all_keys = set()
                for r in results:
                    all_keys.update(r.keys())
                writer = csv.DictWriter(sys.stdout, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(results)
    
    successful = len([r for r in results if 'error' not in r])
    print(f"\n✓ Analyzed {successful}/{len(files_to_analyze)} files successfully")
    return 0


def cmd_export_features(args):
    """Export audio features to JSON or CSV file."""
    import json
    import csv
    from pathlib import Path
    
    try:
        from music_brain.audio.analyzer import AudioAnalyzer
        from music_brain.audio.chord_detection import ChordDetector
    except ImportError as e:
        print(f"Error: Audio analysis requires librosa. Install with: pip install librosa")
        return 1
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    output_path = Path(args.output)
    
    print(f"Analyzing: {audio_path.name}")
    print(f"Exporting to: {output_path}\n")
    
    analyzer = AudioAnalyzer()
    
    # Perform analysis
    analysis = analyzer.analyze_file(
        str(audio_path),
        analyze_segments=args.include_segments
    )
    
    # Build export data
    export_data = {
        "file": str(audio_path),
        "filename": audio_path.name,
        "duration_seconds": analysis.duration_seconds,
        "sample_rate": analysis.sample_rate,
    }
    
    # BPM
    if analysis.bpm_result:
        export_data["bpm"] = {
            "value": analysis.bpm_result.bpm,
            "confidence": analysis.bpm_result.confidence,
            "alternatives": analysis.bpm_result.tempo_alternatives[:5]
        }
    
    # Key
    if analysis.key_result:
        export_data["key"] = {
            "key": analysis.key_result.key,
            "mode": analysis.key_result.mode.value if hasattr(analysis.key_result.mode, 'value') else str(analysis.key_result.mode),
            "full_key": analysis.key_result.full_key,
            "confidence": analysis.key_result.confidence
        }
    
    # Features
    if analysis.features:
        export_data["feel"] = analysis.features.to_dict()
    
    # Feature summary
    if analysis.feature_summary:
        export_data["features"] = analysis.feature_summary
    
    # Segments
    if args.include_segments and analysis.segments:
        export_data["segments"] = [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "energy": seg.energy,
                "label": seg.label
            }
            for seg in analysis.segments
        ]
    
    # Chords
    if args.include_chords:
        try:
            detector = ChordDetector()
            progression = detector.detect_progression(str(audio_path))
            export_data["chords"] = progression.to_dict()
        except Exception as e:
            export_data["chords"] = {"error": str(e)}
    
    # Write output
    if args.format == 'json':
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✓ Features exported to: {output_path} (JSON)")
    elif args.format == 'csv':
        # Flatten for CSV
        csv_data = []
        row = {
            "file": export_data["file"],
            "filename": export_data["filename"],
            "duration": export_data["duration_seconds"],
            "sample_rate": export_data["sample_rate"],
        }
        
        if "bpm" in export_data:
            row["bpm"] = export_data["bpm"]["value"]
            row["bpm_confidence"] = export_data["bpm"]["confidence"]
        
        if "key" in export_data:
            row["key"] = export_data["key"]["full_key"]
            row["key_confidence"] = export_data["key"]["confidence"]
        
        if "features" in export_data:
            for key, value in export_data["features"].items():
                if isinstance(value, (int, float)):
                    row[key] = value
        
        csv_data.append(row)
        
        with open(output_path, 'w', newline='') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        print(f"✓ Features exported to: {output_path} (CSV)")
    
    return 0


def cmd_intent(args):
    """Handle intent-based song generation."""
    (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
     SystemDirective, suggest_rule_break, validate_intent, list_all_rules,
     IntentProcessor, process_intent) = get_intent_module()
    
    if args.subcommand == 'new':
        # Create new intent from template
        print("Creating new song intent...")
        
        intent = CompleteSongIntent(
            title=args.title or "Untitled Song",
            song_root=SongRoot(
                core_event="[What happened?]",
                core_resistance="[What holds you back?]",
                core_longing="[What do you want to feel?]",
                core_stakes="Personal",
                core_transformation="[How should you feel at the end?]",
            ),
            song_intent=SongIntent(
                mood_primary="[Primary emotion]",
                mood_secondary_tension=0.5,
                imagery_texture="[Visual/tactile quality]",
                vulnerability_scale="Medium",
                narrative_arc="Climb-to-Climax",
            ),
            technical_constraints=TechnicalConstraints(
                technical_genre="[Genre]",
                technical_tempo_range=(80, 120),
                technical_key="F",
                technical_mode="major",
                technical_groove_feel="Organic/Breathing",
                technical_rule_to_break="",
                rule_breaking_justification="",
            ),
            system_directive=SystemDirective(
                output_target="Chord progression",
                output_feedback_loop="Harmony",
            ),
        )
        
        output = args.output or "song_intent.json"
        intent.save(output)
        print(f"Template saved to: {output}")
        print("\nEdit the file to fill in your intent, then run:")
        print(f"  daiw intent process {output}")
    
    elif args.subcommand == 'process':
        # Process intent to generate elements
        if not args.file:
            print("Error: Please specify an intent file")
            return 1
        
        intent_path = Path(args.file)
        if not intent_path.exists():
            print(f"Error: File not found: {intent_path}")
            return 1
        
        print(f"Processing intent: {intent_path}")
        intent = CompleteSongIntent.load(str(intent_path))
        
        # Validate first
        issues = validate_intent(intent)
        if issues and not args.force:
            print("\n⚠️  Intent validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nFix issues or use --force to proceed anyway")
            return 1
        
        # Process
        result = process_intent(intent)
        
        # Display results
        print("\n" + "=" * 60)
        print("🎵 GENERATED ELEMENTS")
        print("=" * 60)
        
        # Harmony
        harmony = result['harmony']
        print(f"\n📌 HARMONY ({harmony.rule_broken})")
        print(f"   Progression: {' - '.join(harmony.chords)}")
        print(f"   Roman: {' - '.join(harmony.roman_numerals)}")
        print(f"   Effect: {harmony.rule_effect}")
        
        # Groove
        groove = result['groove']
        print(f"\n📌 GROOVE ({groove.rule_broken})")
        print(f"   Pattern: {groove.pattern_name}")
        print(f"   Tempo: {groove.tempo_bpm} BPM")
        print(f"   Effect: {groove.rule_effect}")
        
        # Arrangement
        arr = result['arrangement']
        print(f"\n📌 ARRANGEMENT ({arr.rule_broken})")
        for section in arr.sections:
            print(f"   {section['name']}: {section['bars']} bars @ {section['energy']:.0%} energy")
        
        # Production
        prod = result['production']
        print(f"\n📌 PRODUCTION ({prod.rule_broken})")
        print(f"   Vocal: {prod.vocal_treatment}")
        for note in prod.eq_notes[:2]:
            print(f"   EQ: {note}")
        
        print("\n" + "=" * 60)
        
        # Save output if requested
        if args.output:
            import json
            output_data = {
                "intent_summary": result['intent_summary'],
                "harmony": {
                    "chords": harmony.chords,
                    "roman_numerals": harmony.roman_numerals,
                    "rule_broken": harmony.rule_broken,
                    "effect": harmony.rule_effect,
                },
                "groove": {
                    "pattern": groove.pattern_name,
                    "tempo": groove.tempo_bpm,
                    "swing": groove.swing_factor,
                },
                "arrangement": {
                    "sections": arr.sections,
                    "dynamic_arc": arr.dynamic_arc,
                },
                "production": {
                    "vocal_treatment": prod.vocal_treatment,
                    "eq_notes": prod.eq_notes,
                    "dynamics_notes": prod.dynamics_notes,
                },
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nOutput saved to: {args.output}")
    
    elif args.subcommand == 'suggest':
        # Suggest rules to break based on emotion
        emotion = args.emotion
        suggestions = suggest_rule_break(emotion)
        
        print(f"\n🎯 Suggested rules to break for '{emotion}':\n")
        
        if not suggestions:
            print(f"  No specific suggestions for '{emotion}'")
            print("  Try: grief, anger, nostalgia, defiance, dissociation")
        else:
            for i, sug in enumerate(suggestions, 1):
                print(f"{i}. {sug['rule']}")
                print(f"   What: {sug['description']}")
                print(f"   Effect: {sug['effect']}")
                print(f"   Use when: {sug['use_when']}")
                print()
    
    elif args.subcommand == 'list':
        # List all available rules
        rules = list_all_rules()
        
        print("\n📋 Available Rule-Breaking Options:\n")
        for category, rule_list in rules.items():
            print(f"  {category}:")
            for rule in rule_list:
                print(f"    - {rule}")
            print()
    
    elif args.subcommand == 'validate':
        # Validate an intent file
        if not args.file:
            print("Error: Please specify an intent file")
            return 1
        
        intent_path = Path(args.file)
        if not intent_path.exists():
            print(f"Error: File not found: {intent_path}")
            return 1
        
        intent = CompleteSongIntent.load(str(intent_path))
        issues = validate_intent(intent)
        
        if issues:
            print("\n⚠️  Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print("✅ Intent is valid!")
            return 0
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='daiw',
        description='DAiW - Digital Audio intelligent Workstation CLI'
    )
    parser.add_argument('--version', action='version', version='%(prog)s 0.2.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract groove from MIDI')
    extract_parser.add_argument('midi_file', help='MIDI file to extract from')
    extract_parser.add_argument('-o', '--output', help='Output JSON file')
    
    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply groove template')
    apply_parser.add_argument('midi_file', help='MIDI file to process')
    apply_parser.add_argument('-g', '--genre', default='funk',
                              choices=['funk', 'jazz', 'rock', 'hiphop', 'edm', 'latin'],
                              help='Genre groove template')
    apply_parser.add_argument('-o', '--output', help='Output MIDI file')
    apply_parser.add_argument('-i', '--intensity', type=float, default=0.5,
                              help='Groove intensity 0.0-1.0')

    # Humanize command (Drunken Drummer)
    humanize_parser = subparsers.add_parser('humanize', help='Apply drum humanization (Drunken Drummer)')
    humanize_parser.add_argument('midi_file', nargs='?', help='MIDI file to humanize')
    humanize_parser.add_argument('-o', '--output', help='Output MIDI file')
    humanize_parser.add_argument('-p', '--preset',
                                 help='Emotional preset (use --list-presets to see options)')
    humanize_parser.add_argument('-s', '--style',
                                 choices=['tight', 'natural', 'loose', 'drunk'],
                                 help='Quick style preset')
    humanize_parser.add_argument('-c', '--complexity', type=float, default=0.5,
                                 help='Timing chaos 0.0-1.0 (ignored if --style/--preset used)')
    humanize_parser.add_argument('-v', '--vulnerability', type=float, default=0.5,
                                 help='Dynamic fragility 0.0-1.0 (ignored if --style/--preset used)')
    humanize_parser.add_argument('--channel', type=int, default=9,
                                 help='MIDI drum channel (default: 9, i.e. channel 10)')
    humanize_parser.add_argument('--no-ghost-notes', action='store_true',
                                 help='Disable ghost note generation')
    humanize_parser.add_argument('--seed', type=int,
                                 help='Random seed for reproducibility')
    humanize_parser.add_argument('--list-presets', action='store_true',
                                 help='List available emotional presets')

    # Analyze command (MIDI)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze MIDI file')
    analyze_parser.add_argument('midi_file', help='MIDI file to analyze')
    analyze_parser.add_argument('-c', '--chords', action='store_true', help='Analyze chords')
    analyze_parser.add_argument('-s', '--sections', action='store_true', help='Detect sections')
    
    # Analyze-audio command
    analyze_audio_parser = subparsers.add_parser('analyze-audio', help='Analyze audio file (BPM, key, chords, feel)')
    analyze_audio_parser.add_argument('audio_file', help='Audio file to analyze (WAV, MP3, AIFF, etc.)')
    analyze_audio_parser.add_argument('--no-bpm', dest='bpm', action='store_false',
                                      help='Disable BPM detection (default: enabled)')
    analyze_audio_parser.add_argument('--no-key', dest='key', action='store_false',
                                      help='Disable key detection (default: enabled)')
    analyze_audio_parser.add_argument('--no-features', dest='features', action='store_false',
                                      help='Skip advanced feature extraction (default: enabled)')
    analyze_audio_parser.add_argument('--no-segments', dest='segments', action='store_false',
                                      help='Skip structural segmentation (default: enabled)')
    analyze_audio_parser.add_argument('--segment-count', type=int, default=4,
                                      help='Target number of segments for structural analysis')
    analyze_audio_parser.add_argument('--chords', action='store_true', help='Detect chord progression')
    analyze_audio_parser.add_argument('--max-duration', type=float, help='Maximum duration to analyze (seconds)')
    analyze_audio_parser.add_argument('-o', '--output', help='Save analysis to JSON file')
    analyze_audio_parser.set_defaults(bpm=True, key=True, features=True, segments=True)
    
    # Compare-audio command
    compare_audio_parser = subparsers.add_parser('compare-audio', help='Compare two audio files (BPM, key, feel)')
    compare_audio_parser.add_argument('file1', help='First audio file to compare')
    compare_audio_parser.add_argument('file2', help='Second audio file to compare')
    compare_audio_parser.add_argument('--detailed', action='store_true', help='Show detailed feature comparison')
    compare_audio_parser.add_argument('-o', '--output', help='Save comparison to JSON file')
    
    # Batch-analyze command
    batch_analyze_parser = subparsers.add_parser('batch-analyze', help='Batch analyze multiple audio files')
    batch_analyze_parser.add_argument('files', nargs='+', help='Audio files to analyze (or directory)')
    batch_analyze_parser.add_argument('--recursive', '-r', action='store_true', help='Recursively scan directories')
    batch_analyze_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    batch_analyze_parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    batch_analyze_parser.add_argument('--max-duration', type=float, help='Maximum duration per file (seconds)')
    
    # Export-features command
    export_features_parser = subparsers.add_parser('export-features', help='Export audio features to file')
    export_features_parser.add_argument('audio_file', help='Audio file to analyze')
    export_features_parser.add_argument('-o', '--output', required=True, help='Output file path')
    export_features_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    export_features_parser.add_argument('--include-segments', action='store_true', help='Include segment analysis')
    export_features_parser.add_argument('--include-chords', action='store_true', help='Include chord detection')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate harmony from intent or parameters')
    generate_parser.add_argument('-i', '--intent-file', help='Intent JSON file')
    generate_parser.add_argument('-k', '--key', help='Key (e.g., C, F, Bb)')
    generate_parser.add_argument('-m', '--mode', choices=['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian'],
                                 help='Mode/scale')
    generate_parser.add_argument('-p', '--pattern', help='Roman numeral pattern (e.g., "I-V-vi-IV")')
    generate_parser.add_argument('-o', '--output', help='Output MIDI file')
    generate_parser.add_argument('-t', '--tempo', type=int, help='Tempo in BPM (default: 82)')

    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Diagnose chord progression')
    diagnose_parser.add_argument('progression', help='Chord progression (e.g., "F-C-Am-Dm")')
    
    # Reharm command
    reharm_parser = subparsers.add_parser('reharm', help='Generate reharmonizations')
    reharm_parser.add_argument('progression', help='Chord progression to reharmonize')
    reharm_parser.add_argument('-s', '--style', default='jazz',
                               choices=['jazz', 'pop', 'rnb', 'classical', 'experimental'],
                               help='Reharmonization style')
    reharm_parser.add_argument('-n', '--count', type=int, default=3,
                               help='Number of suggestions')
    
    # Teach command
    teach_parser = subparsers.add_parser('teach', help='Interactive teaching mode')
    teach_parser.add_argument('topic', help='Topic to learn (rulebreaking, borrowed, etc.)')
    teach_parser.add_argument('-q', '--quick', action='store_true', help='Quick single lesson')
    
    # Intent command with subcommands
    intent_parser = subparsers.add_parser('intent', help='Intent-based song generation')
    intent_subparsers = intent_parser.add_subparsers(dest='subcommand', help='Intent commands')
    
    # intent new
    intent_new = intent_subparsers.add_parser('new', help='Create new intent template')
    intent_new.add_argument('-t', '--title', help='Song title')
    intent_new.add_argument('-o', '--output', help='Output file (default: song_intent.json)')
    
    # intent process
    intent_process = intent_subparsers.add_parser('process', help='Process intent to generate elements')
    intent_process.add_argument('file', help='Intent JSON file')
    intent_process.add_argument('-o', '--output', help='Save output to JSON')
    intent_process.add_argument('-f', '--force', action='store_true', help='Proceed despite validation issues')
    
    # intent suggest
    intent_suggest = intent_subparsers.add_parser('suggest', help='Suggest rules to break')
    intent_suggest.add_argument('emotion', help='Target emotion (grief, anger, nostalgia, etc.)')
    
    # intent list
    intent_subparsers.add_parser('list', help='List all rule-breaking options')
    
    # intent validate
    intent_validate = intent_subparsers.add_parser('validate', help='Validate intent file')
    intent_validate.add_argument('file', help='Intent JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        'extract': cmd_extract,
        'apply': cmd_apply,
        'humanize': cmd_humanize,
        'analyze': cmd_analyze,
        'analyze-audio': cmd_analyze_audio,
        'compare-audio': cmd_compare_audio,
        'batch-analyze': cmd_batch_analyze,
        'export-features': cmd_export_features,
        'generate': cmd_generate,
        'diagnose': cmd_diagnose,
        'reharm': cmd_reharm,
        'teach': cmd_teach,
        'intent': cmd_intent,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
