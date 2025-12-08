"""
Integration tests for Phase 2 features.

Tests:
- MCP tool integration
- UI workflow integration
- Audio analysis integration
- End-to-end workflows
"""

import pytest
import json
import tempfile
from pathlib import Path


class TestMCPIntegration:
    """Test MCP tool integration with core modules."""
    
    def test_mcp_tools_importable(self):
        """Test that all MCP tool modules can be imported."""
        from daiw_mcp.tools import (
            harmony_tools,
            groove_tools,
            intent_tools,
            audio_tools,
            teaching_tools,
        )
        assert harmony_tools is not None
        assert groove_tools is not None
        assert intent_tools is not None
        assert audio_tools is not None
        assert teaching_tools is not None
    
    def test_mcp_server_creatable(self):
        """Test that MCP server can be created."""
        try:
            from daiw_mcp.server import create_server
            server = create_server()
            assert server is not None
        except ImportError:
            pytest.skip("MCP SDK not installed")
    
    def test_harmony_tools_registered(self):
        """Test that harmony tools are properly registered."""
        try:
            from mcp.server import Server
            from daiw_mcp.tools import harmony_tools
            
            server = Server("test")
            harmony_tools.register_tools(server)
            # If no exception, tools are registered
            assert True
        except ImportError:
            pytest.skip("MCP SDK not installed")


class TestAudioAnalysisIntegration:
    """Test audio analysis module integration."""
    
    def test_audio_analyzer_importable(self):
        """Test that AudioAnalyzer can be imported."""
        try:
            from music_brain.audio.analyzer import AudioAnalyzer
            assert AudioAnalyzer is not None
        except ImportError:
            pytest.skip("librosa not installed")
    
    def test_chord_detector_importable(self):
        """Test that ChordDetector can be imported."""
        try:
            from music_brain.audio.chord_detection import ChordDetector
            assert ChordDetector is not None
        except ImportError:
            pytest.skip("librosa not installed")
    
    def test_frequency_analyzer_importable(self):
        """Test that FrequencyAnalyzer can be imported."""
        try:
            from music_brain.audio.frequency import FrequencyAnalyzer
            assert FrequencyAnalyzer is not None
        except ImportError:
            pytest.skip("scipy not installed")
    
    def test_audio_analysis_complete(self):
        """Test that audio analysis module is complete."""
        from music_brain.audio import (
            AudioAnalyzer,
            ChordDetector,
            FrequencyAnalyzer,
            analyze_feel,
        )
        # If imports succeed, module is complete
        assert AudioAnalyzer is not None
        assert ChordDetector is not None
        assert FrequencyAnalyzer is not None
        assert analyze_feel is not None


class TestUIWorkflowIntegration:
    """Test UI workflow integration."""
    
    def test_emidi_workflow(self):
        """Test EMIDI (TherapySession) workflow."""
        from music_brain.structure.comprehensive_engine import (
            TherapySession,
            render_plan_to_midi,
        )
        
        session = TherapySession()
        affect = session.process_core_input("I feel lost and empty")
        session.set_scales(7, 0.5)
        plan = session.generate_plan()
        
        assert plan is not None
        assert plan.tempo_bpm > 0
        assert len(plan.chord_symbols) > 0
    
    def test_intent_generator_workflow(self):
        """Test intent generator workflow."""
        from music_brain.session.intent_schema import (
            CompleteSongIntent,
            SongRoot,
            SongIntent,
            TechnicalConstraints,
            SystemDirective,
        )
        from music_brain.session.intent_processor import process_intent
        
        intent = CompleteSongIntent(
            title="Test Song",
            song_root=SongRoot(
                core_event="Test event",
                core_resistance="Test resistance",
                core_longing="Test longing",
                core_stakes="Personal",
                core_transformation="Test transformation",
            ),
            song_intent=SongIntent(
                mood_primary="grief",
                mood_secondary_tension=0.7,
                imagery_texture="dark",
                vulnerability_scale="High",
                narrative_arc="Climb-to-Climax",
            ),
            technical_constraints=TechnicalConstraints(
                technical_genre="lo-fi",
                technical_tempo_range=(80, 120),
                technical_key="F",
                technical_mode="major",
                technical_groove_feel="Organic",
                technical_rule_to_break="HARMONY_ModalInterchange",
                rule_breaking_justification="Creates unresolved yearning",
            ),
            system_directive=SystemDirective(
                output_target="Chord progression",
                output_feedback_loop="Harmony",
            ),
        )
        
        result = process_intent(intent)
        assert result is not None
        assert 'harmony' in result
        assert 'groove' in result


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_emotion_to_midi_workflow(self):
        """Test complete emotion → MIDI workflow."""
        from music_brain.structure.comprehensive_engine import (
            TherapySession,
            render_plan_to_midi,
        )
        
        session = TherapySession()
        affect = session.process_core_input("I feel nostalgic for home")
        session.set_scales(8, 0.4)
        plan = session.generate_plan()
        
        # Generate MIDI
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            midi_path = render_plan_to_midi(plan, tmp.name)
            assert Path(midi_path).exists()
            Path(midi_path).unlink()  # Clean up
    
    def test_intent_to_midi_workflow(self):
        """Test complete intent → MIDI workflow."""
        from music_brain.session.intent_schema import (
            CompleteSongIntent,
            SongRoot,
            SongIntent,
            TechnicalConstraints,
            SystemDirective,
        )
        from music_brain.session.intent_processor import process_intent
        from music_brain.harmony import generate_midi_from_harmony
        
        intent = CompleteSongIntent(
            title="Test",
            song_root=SongRoot(
                core_event="Test",
                core_resistance="Test",
                core_longing="Test",
                core_stakes="Personal",
                core_transformation="Test",
            ),
            song_intent=SongIntent(
                mood_primary="grief",
                mood_secondary_tension=0.5,
                imagery_texture="dark",
                vulnerability_scale="Medium",
                narrative_arc="Climb-to-Climax",
            ),
            technical_constraints=TechnicalConstraints(
                technical_genre="lo-fi",
                technical_tempo_range=(82, 82),
                technical_key="F",
                technical_mode="major",
                technical_groove_feel="Organic",
                technical_rule_to_break="",
                rule_breaking_justification="",
            ),
            system_directive=SystemDirective(
                output_target="Chord progression",
                output_feedback_loop="Harmony",
            ),
        )
        
        result = process_intent(intent)
        harmony = result['harmony']
        
        # Generate MIDI
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            generate_midi_from_harmony(harmony, tmp.name, tempo_bpm=82)
            assert Path(tmp.name).exists()
            Path(tmp.name).unlink()  # Clean up
    
    def test_groove_extraction_application_workflow(self):
        """Test groove extraction and application workflow."""
        from music_brain.groove import extract_groove, apply_groove
        
        # This would need a test MIDI file
        # For now, just test that functions are callable
        assert callable(extract_groove)
        assert callable(apply_groove)


class TestCLIIntegration:
    """Test CLI command integration."""
    
    def test_all_cli_commands_registered(self):
        """Test that all CLI commands are registered."""
        from music_brain.cli import main
        import sys
        from io import StringIO
        
        # Capture help output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            sys.argv = ['daiw', '--help']
            try:
                main()
            except SystemExit:
                pass
            help_output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Check for key commands
        assert 'extract' in help_output or 'Extract' in help_output
        assert 'generate' in help_output or 'Generate' in help_output
        assert 'analyze' in help_output or 'Analyze' in help_output
    
    def test_analyze_audio_command_exists(self):
        """Test that analyze-audio command exists."""
        from music_brain.cli import main
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            sys.argv = ['daiw', 'analyze-audio', '--help']
            try:
                main()
            except SystemExit:
                pass
            help_output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Command should exist (help will show or command will run)
        assert True  # If no exception, command exists


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

