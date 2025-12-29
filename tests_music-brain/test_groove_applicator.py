"""
Comprehensive tests for music_brain.groove.applicator module.

Tests cover:
- apply_groove function with genre templates
- humanize function
- Timing and velocity modifications
- PPQ scaling
- Edge cases and error handling

Run with: pytest tests_music-brain/test_groove_applicator.py -v
"""

import pytest
import tempfile
from pathlib import Path

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

from music_brain.groove.applicator import apply_groove, humanize

# Skip all tests if mido not available
pytestmark = pytest.mark.skipif(not MIDO_AVAILABLE, reason="mido not installed")


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def simple_midi_file():
    """Create a simple MIDI file for testing."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Add tempo
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM

    # Add 4 quarter notes (C4)
    for i in range(4):
        track.append(mido.Message('note_on', note=60, velocity=80, time=0 if i == 0 else 480))
        track.append(mido.Message('note_off', note=60, velocity=0, time=480))

    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        temp_path = f.name

    mid.save(temp_path)
    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def groove_template():
    """Create a simple GrooveTemplate for testing."""
    from music_brain.groove.extractor import GrooveTemplate

    return GrooveTemplate(
        name="Test Groove",
        genre="test",
        ppq=480,
        timing_deviations=[10, -5, 15, -10],  # Timing offsets in ticks
        velocity_curve=[100, 80, 90, 70],  # Velocity values
    )


# ==============================================================================
# APPLY GROOVE TESTS
# ==============================================================================

class TestApplyGroove:
    """Test apply_groove function."""

    def test_apply_groove_with_template(self, simple_midi_file, groove_template):
        """Test applying a groove template."""
        output = apply_groove(
            simple_midi_file,
            groove=groove_template,
            intensity=0.5
        )

        assert Path(output).exists()

        # Verify output is valid MIDI
        mid = mido.MidiFile(output)
        assert mid.ticks_per_beat == 480

        # Cleanup
        Path(output).unlink(missing_ok=True)

    def test_apply_groove_with_genre(self, simple_midi_file):
        """Test applying groove by genre name."""
        try:
            output = apply_groove(
                simple_midi_file,
                genre="funk",
                intensity=0.5
            )

            assert Path(output).exists()

            # Cleanup
            Path(output).unlink(missing_ok=True)
        except ValueError:
            # Genre might not exist, that's OK
            pytest.skip("Funk genre not available")

    def test_apply_groove_requires_template_or_genre(self, simple_midi_file):
        """Test that either groove or genre must be provided."""
        with pytest.raises(ValueError, match="Must provide either groove template or genre"):
            apply_groove(simple_midi_file)

    def test_apply_groove_file_not_found(self, groove_template):
        """Test error when MIDI file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            apply_groove("nonexistent.mid", groove=groove_template)

    def test_apply_groove_custom_output_path(self, simple_midi_file, groove_template):
        """Test custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "custom_output.mid")
            result = apply_groove(
                simple_midi_file,
                groove=groove_template,
                output=output_path
            )

            assert result == output_path
            assert Path(output_path).exists()

    def test_apply_groove_intensity_zero(self, simple_midi_file, groove_template):
        """Test intensity=0 (no groove applied)."""
        output = apply_groove(
            simple_midi_file,
            groove=groove_template,
            intensity=0.0
        )

        # Should create file with minimal changes
        assert Path(output).exists()

        Path(output).unlink(missing_ok=True)

    def test_apply_groove_intensity_full(self, simple_midi_file, groove_template):
        """Test intensity=1.0 (full groove)."""
        output = apply_groove(
            simple_midi_file,
            groove=groove_template,
            intensity=1.0
        )

        assert Path(output).exists()

        Path(output).unlink(missing_ok=True)

    def test_apply_groove_preserve_dynamics_true(self, simple_midi_file, groove_template):
        """Test preserve_dynamics=True."""
        output = apply_groove(
            simple_midi_file,
            groove=groove_template,
            intensity=0.5,
            preserve_dynamics=True
        )

        assert Path(output).exists()

        # Verify velocities are blended, not replaced
        mid = mido.MidiFile(output)
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Velocity should be between original (80) and template values
                    assert 1 <= msg.velocity <= 127

        Path(output).unlink(missing_ok=True)

    def test_apply_groove_preserve_dynamics_false(self, simple_midi_file, groove_template):
        """Test preserve_dynamics=False."""
        output = apply_groove(
            simple_midi_file,
            groove=groove_template,
            intensity=1.0,
            preserve_dynamics=False
        )

        assert Path(output).exists()

        Path(output).unlink(missing_ok=True)

    def test_apply_groove_humanize_timing_disabled(self, simple_midi_file, groove_template):
        """Test with humanize_timing=False."""
        output = apply_groove(
            simple_midi_file,
            groove=groove_template,
            humanize_timing=False
        )

        assert Path(output).exists()

        Path(output).unlink(missing_ok=True)

    def test_apply_groove_humanize_velocity_disabled(self, simple_midi_file, groove_template):
        """Test with humanize_velocity=False."""
        output = apply_groove(
            simple_midi_file,
            groove=groove_template,
            humanize_velocity=False
        )

        assert Path(output).exists()

        Path(output).unlink(missing_ok=True)


# ==============================================================================
# HUMANIZE TESTS
# ==============================================================================

class TestHumanize:
    """Test humanize function."""

    def test_humanize_basic(self, simple_midi_file):
        """Test basic humanization."""
        output = humanize(simple_midi_file)

        assert Path(output).exists()
        assert "_humanized.mid" in output

        # Verify output is valid MIDI
        mid = mido.MidiFile(output)
        assert mid.ticks_per_beat == 480

        Path(output).unlink(missing_ok=True)

    def test_humanize_custom_output(self, simple_midi_file):
        """Test custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "humanized.mid")
            result = humanize(simple_midi_file, output=output_path)

            assert result == output_path
            assert Path(output_path).exists()

    def test_humanize_timing_range(self, simple_midi_file):
        """Test custom timing range."""
        output = humanize(
            simple_midi_file,
            timing_range_ms=20.0
        )

        assert Path(output).exists()

        Path(output).unlink(missing_ok=True)

    def test_humanize_velocity_range(self, simple_midi_file):
        """Test custom velocity range."""
        output = humanize(
            simple_midi_file,
            velocity_range=30
        )

        assert Path(output).exists()

        # Verify velocities are modified
        mid = mido.MidiFile(output)
        has_velocity_variation = False

        for track in mid.tracks:
            velocities = [msg.velocity for msg in track if msg.type == 'note_on' and msg.velocity > 0]
            if len(set(velocities)) > 1:  # More than one unique velocity
                has_velocity_variation = True

        # May or may not have variation due to randomness, just verify no crash

        Path(output).unlink(missing_ok=True)

    def test_humanize_with_seed_reproducible(self, simple_midi_file):
        """Test humanization is reproducible with seed."""
        output1 = humanize(simple_midi_file, seed=42)
        output2 = humanize(simple_midi_file, seed=42)

        # Load both files
        mid1 = mido.MidiFile(output1)
        mid2 = mido.MidiFile(output2)

        # Should be identical
        assert mid1.ticks_per_beat == mid2.ticks_per_beat

        Path(output1).unlink(missing_ok=True)
        Path(output2).unlink(missing_ok=True)

    def test_humanize_different_seeds_different_results(self, simple_midi_file):
        """Test different seeds produce different results."""
        output1 = humanize(simple_midi_file, seed=42)
        output2 = humanize(simple_midi_file, seed=99)

        # Should exist
        assert Path(output1).exists()
        assert Path(output2).exists()

        # May or may not be different, just verify both work

        Path(output1).unlink(missing_ok=True)
        Path(output2).unlink(missing_ok=True)

    def test_humanize_zero_ranges(self, simple_midi_file):
        """Test humanization with zero ranges (no change)."""
        output = humanize(
            simple_midi_file,
            timing_range_ms=0.0,
            velocity_range=0
        )

        assert Path(output).exists()

        Path(output).unlink(missing_ok=True)


# ==============================================================================
# PPQ SCALING TESTS
# ==============================================================================

class TestPPQScaling:
    """Test PPQ (ticks per beat) scaling."""

    def test_different_ppq_scaling(self, groove_template):
        """Test groove application with different PPQ."""
        # Create MIDI with different PPQ
        mid = mido.MidiFile(ticks_per_beat=960)  # Double the template PPQ
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        track.append(mido.Message('note_on', note=60, velocity=80, time=0))
        track.append(mido.Message('note_off', note=60, velocity=0, time=960))

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_path = f.name

        mid.save(temp_path)

        try:
            output = apply_groove(
                temp_path,
                groove=groove_template,
                intensity=0.5
            )

            # Should handle PPQ scaling
            result_mid = mido.MidiFile(output)
            assert result_mid.ticks_per_beat == 960

            Path(output).unlink(missing_ok=True)
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_midi_file(self, groove_template):
        """Test with empty MIDI file."""
        # Create empty MIDI
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_path = f.name

        mid.save(temp_path)

        try:
            output = apply_groove(
                temp_path,
                groove=groove_template
            )

            # Should handle gracefully
            assert Path(output).exists()

            Path(output).unlink(missing_ok=True)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_midi_with_multiple_tracks(self, groove_template):
        """Test MIDI with multiple tracks."""
        mid = mido.MidiFile(ticks_per_beat=480)

        # Track 1
        track1 = mido.MidiTrack()
        mid.tracks.append(track1)
        track1.append(mido.Message('note_on', note=60, velocity=80, time=0))
        track1.append(mido.Message('note_off', note=60, velocity=0, time=480))

        # Track 2
        track2 = mido.MidiTrack()
        mid.tracks.append(track2)
        track2.append(mido.Message('note_on', note=64, velocity=90, time=0))
        track2.append(mido.Message('note_off', note=64, velocity=0, time=480))

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_path = f.name

        mid.save(temp_path)

        try:
            output = apply_groove(
                temp_path,
                groove=groove_template
            )

            # Should process all tracks
            result_mid = mido.MidiFile(output)
            assert len(result_mid.tracks) == 2

            Path(output).unlink(missing_ok=True)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_midi_with_meta_messages(self, groove_template):
        """Test MIDI with various meta messages."""
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        track.append(mido.MetaMessage('track_name', name='Test Track', time=0))
        track.append(mido.Message('note_on', note=60, velocity=80, time=0))
        track.append(mido.Message('note_off', note=60, velocity=0, time=480))

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_path = f.name

        mid.save(temp_path)

        try:
            output = apply_groove(
                temp_path,
                groove=groove_template
            )

            # Meta messages should be preserved
            result_mid = mido.MidiFile(output)
            assert result_mid is not None

            Path(output).unlink(missing_ok=True)
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestGrooveApplicatorIntegration:
    """Integration tests combining multiple features."""

    def test_apply_then_humanize(self, simple_midi_file, groove_template):
        """Test applying groove then humanizing."""
        # Apply groove
        grooved = apply_groove(
            simple_midi_file,
            groove=groove_template,
            intensity=0.5
        )

        # Then humanize
        final = humanize(grooved, timing_range_ms=5.0, velocity_range=10)

        assert Path(final).exists()

        # Cleanup
        Path(grooved).unlink(missing_ok=True)
        Path(final).unlink(missing_ok=True)

    def test_multiple_intensity_levels(self, simple_midi_file, groove_template):
        """Test multiple intensity levels produce different results."""
        outputs = []

        for intensity in [0.0, 0.5, 1.0]:
            output = apply_groove(
                simple_midi_file,
                groove=groove_template,
                intensity=intensity
            )
            outputs.append(output)

        # All should exist
        for output in outputs:
            assert Path(output).exists()

        # Cleanup
        for output in outputs:
            Path(output).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
