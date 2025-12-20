#include "export/StemExporter.h"
#include "common/Types.h"
#include "common/MusicConstants.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <map>

namespace midikompanion {
using namespace kelly::MusicConstants;

StemExporter::StemExporter() {
    clearError();
}

std::vector<StemExporter::StemResult> StemExporter::exportAllStems(
    const GeneratedMidi& midi,
    const juce::File& outputDirectory,
    const ExportOptions& options)
{
    clearError();

    // Get all available track names
    std::vector<std::string> trackNames;
    if (!midi.melody.empty()) trackNames.push_back("melody");
    if (!midi.bass.empty()) trackNames.push_back("bass");
    if (!midi.chords.empty()) trackNames.push_back("chords");
    if (!midi.counterMelody.empty()) trackNames.push_back("counterMelody");
    if (!midi.pad.empty()) trackNames.push_back("pad");
    if (!midi.strings.empty()) trackNames.push_back("strings");
    if (!midi.fills.empty()) trackNames.push_back("fills");
    if (!midi.rhythm.empty()) trackNames.push_back("rhythm");
    if (!midi.drumGroove.empty()) trackNames.push_back("drums");

    return exportSelectedStems(midi, trackNames, outputDirectory, options);
}

std::vector<StemExporter::StemResult> StemExporter::exportSelectedStems(
    const GeneratedMidi& midi,
    const std::vector<std::string>& trackNames,
    const juce::File& outputDirectory,
    const ExportOptions& options)
{
    clearError();

    std::vector<StemResult> results;

    // Create output directory if it doesn't exist
    if (!outputDirectory.createDirectory()) {
        setError("Could not create output directory: " + outputDirectory.getFullPathName());
        return results;
    }

    // Calculate duration
    double durationSeconds = options.durationSeconds > 0.0f
        ? static_cast<double>(options.durationSeconds)
        : calculateDuration(midi);

    if (durationSeconds <= 0.0) {
        durationSeconds = 10.0;  // Default 10 seconds
    }

    // Export each track
    for (size_t i = 0; i < trackNames.size(); ++i) {
        const auto& trackName = trackNames[i];

        // Progress callback
        if (progressCallback_) {
            progressCallback_(static_cast<int>(i), static_cast<int>(trackNames.size()), juce::String(trackName));
        }

        // Generate filename
        juce::File outputFile = generateStemFilename(trackName, outputDirectory, options.format, options.filenameSuffix);

        // Export track
        StemResult result = exportTrack(midi, trackName, outputFile, options);
        results.push_back(result);
    }

    return results;
}

StemExporter::StemResult StemExporter::exportTrack(
    const GeneratedMidi& midi,
    const std::string& trackName,
    const juce::File& outputFile,
    const ExportOptions& options)
{
    StemResult result;
    result.trackName = trackName;
    result.filepath = outputFile.getFullPathName();

    clearError();

    // Calculate duration
    double durationSeconds = options.durationSeconds > 0.0f
        ? static_cast<double>(options.durationSeconds)
        : calculateDuration(midi);

    if (durationSeconds <= 0.0) {
        durationSeconds = 10.0;  // Default 10 seconds
    }

    // Render track to audio
    juce::AudioBuffer<float> audioBuffer;

    if (trackName == "chords") {
        // Render chords
        audioBuffer = renderChords(midi.chords, durationSeconds, options.sampleRate, 0);
    } else {
        // Render MIDI notes
        std::vector<MidiNote> notes = getTrackNotes(midi, trackName);
        if (notes.empty()) {
            result.success = false;
            result.errorMessage = "Track not found or empty: " + juce::String(trackName);
            return result;
        }
        audioBuffer = renderMidiNotes(notes, durationSeconds, options.sampleRate, 0);
    }

    if (audioBuffer.getNumSamples() == 0) {
        result.success = false;
        result.errorMessage = "Failed to render audio for track: " + juce::String(trackName);
        return result;
    }

    // Normalize if requested
    if (options.normalizeStems) {
        normalizeBuffer(audioBuffer);
    }

    // Write to file
    if (!writeAudioFile(audioBuffer, outputFile, options)) {
        result.success = false;
        result.errorMessage = getLastError();
        return result;
    }

    result.success = true;
    result.numSamples = audioBuffer.getNumSamples();
    result.durationSeconds = static_cast<double>(result.numSamples) / options.sampleRate;

    return result;
}

StemExporter::StemResult StemExporter::exportVocalStem(
    const std::vector<MidiNote>& vocalNotes,
    const std::vector<juce::String>& lyrics,
    const juce::File& outputFile,
    const ExportOptions& options)
{
    StemResult result;
    result.trackName = "vocals";
    result.filepath = outputFile.getFullPathName();

    clearError();

    if (vocalNotes.empty()) {
        result.success = false;
        result.errorMessage = "No vocal notes to export";
        return result;
    }

    // Calculate duration from vocal notes
    double durationSeconds = options.durationSeconds > 0.0f
        ? static_cast<double>(options.durationSeconds)
        : 10.0;  // Default

    // Find last note end time
    // Convert ticks to seconds (assuming 120 BPM and 480 ticks per beat)
    const double ticksPerSecond = (120.0 / 60.0) * 480.0;  // 960 ticks per second at 120 BPM
    for (const auto& note : vocalNotes) {
        double startTime = static_cast<double>(note.startTick) / ticksPerSecond;
        double duration = static_cast<double>(note.durationTicks) / ticksPerSecond;
        double noteEnd = startTime + duration;
        if (noteEnd > durationSeconds) {
            durationSeconds = noteEnd + 0.5;  // Add 0.5s padding
        }
    }

    // Render vocals
    juce::AudioBuffer<float> audioBuffer = renderVocals(vocalNotes, options.sampleRate);

    if (audioBuffer.getNumSamples() == 0) {
        result.success = false;
        result.errorMessage = "Failed to render vocal audio";
        return result;
    }

    // Normalize if requested
    if (options.normalizeStems) {
        normalizeBuffer(audioBuffer);
    }

    // Write to file
    if (!writeAudioFile(audioBuffer, outputFile, options)) {
        result.success = false;
        result.errorMessage = getLastError();
        return result;
    }

    result.success = true;
    result.numSamples = audioBuffer.getNumSamples();
    result.durationSeconds = static_cast<double>(result.numSamples) / options.sampleRate;

    return result;
}

juce::AudioBuffer<float> StemExporter::renderMidiNotes(
    const std::vector<MidiNote>& notes,
    double durationSeconds,
    double sampleRate,
    int channel)
{
    const int numSamples = static_cast<int>(durationSeconds * sampleRate);
    juce::AudioBuffer<float> buffer(1, numSamples);  // Mono output
    buffer.clear();

    // Simple sine wave synthesis for each note
    // Convert ticks to seconds (assuming 120 BPM and 480 ticks per beat)
    const double ticksPerSecond = (120.0 / 60.0) * 480.0;  // 960 ticks per second at 120 BPM
    for (const auto& note : notes) {
        double startTime = static_cast<double>(note.startTick) / ticksPerSecond;
        double duration = static_cast<double>(note.durationTicks) / ticksPerSecond;
        int startSample = static_cast<int>(startTime * sampleRate);
        int endSample = static_cast<int>((startTime + duration) * sampleRate);

        if (startSample >= numSamples || endSample < 0) {
            continue;  // Note outside buffer range
        }

        startSample = std::max(0, startSample);
        endSample = std::min(numSamples, endSample);

        // Calculate frequency from MIDI note number
        float frequency = 440.0f * std::pow(2.0f, (static_cast<float>(note.pitch) - 69.0f) / 12.0f);

        // Generate sine wave with envelope
        float* channelData = buffer.getWritePointer(0);
        for (int sample = startSample; sample < endSample; ++sample) {
            float time = static_cast<float>(sample) / static_cast<float>(sampleRate);
            float noteTime = time - startTime;

            // Simple ADSR envelope
            float amplitude = note.velocity / 127.0f;
            if (noteTime < 0.01f) {
                // Attack: 10ms
                amplitude *= noteTime / 0.01f;
            } else if (noteTime > duration - 0.1f) {
                // Release: 100ms
                float releaseTime = duration - noteTime;
                amplitude *= std::max(0.0f, releaseTime / 0.1f);
            }

            // Generate sine wave
            float phase = 2.0f * 3.14159265359f * frequency * time;
            channelData[sample] += amplitude * std::sin(phase) * 0.3f;  // Scale to prevent clipping
        }
    }

    return buffer;
}

juce::AudioBuffer<float> StemExporter::renderChords(
    const std::vector<Chord>& chords,
    double durationSeconds,
    double sampleRate,
    int channel)
{
    const int numSamples = static_cast<int>(durationSeconds * sampleRate);
    juce::AudioBuffer<float> buffer(1, numSamples);  // Mono output
    buffer.clear();

    // Render each chord as multiple notes
    // Convert beats to ticks (assuming 480 ticks per beat)
    const int ticksPerBeat = 480;
    for (const auto& chord : chords) {
        // Convert chord to MIDI notes
        std::vector<MidiNote> chordNotes;
        for (int pitch : chord.pitches) {
            MidiNote note;
            note.pitch = pitch;
            note.startTick = static_cast<int>(chord.startBeat * ticksPerBeat);
            note.durationTicks = static_cast<int>(chord.duration * ticksPerBeat);
            note.velocity = 100;  // Default velocity for chords
            chordNotes.push_back(note);
        }

        // Render chord notes
        juce::AudioBuffer<float> chordBuffer = renderMidiNotes(chordNotes, durationSeconds, sampleRate, channel);

        // Mix into main buffer
        buffer.addFrom(0, 0, chordBuffer, 0, 0, numSamples);
    }

    return buffer;
}

juce::AudioBuffer<float> StemExporter::renderVocals(
    const std::vector<MidiNote>& vocalNotes,
    double sampleRate)
{
    // Calculate duration
    // Convert ticks to seconds (assuming 120 BPM and 480 ticks per beat)
    const double ticksPerSecond = (120.0 / 60.0) * 480.0;  // 960 ticks per second at 120 BPM
    double durationSeconds = 0.0;
    for (const auto& note : vocalNotes) {
        double startTime = static_cast<double>(note.startTick) / ticksPerSecond;
        double duration = static_cast<double>(note.durationTicks) / ticksPerSecond;
        double noteEnd = startTime + duration;
        if (noteEnd > durationSeconds) {
            durationSeconds = noteEnd;
        }
    }
    durationSeconds += 0.5;  // Add padding

    // Use same rendering as MIDI notes (simple synthesis)
    // In production, would use VoiceSynthesizer for realistic vocals
    return renderMidiNotes(vocalNotes, durationSeconds, sampleRate, 0);
}

void StemExporter::normalizeBuffer(juce::AudioBuffer<float>& buffer, float targetLevel) {
    if (buffer.getNumSamples() == 0) {
        return;
    }

    // Find peak
    float peak = 0.0f;
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
        const float* channelData = buffer.getReadPointer(channel);
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
            peak = std::max(peak, std::abs(channelData[sample]));
        }
    }

    if (peak > 0.0f && peak < targetLevel) {
        // Apply gain to reach target level
        float gain = targetLevel / peak;
        buffer.applyGain(gain);
    } else if (peak > targetLevel) {
        // Reduce gain to target level
        float gain = targetLevel / peak;
        buffer.applyGain(gain);
    }
}

std::vector<MidiNote> StemExporter::getTrackNotes(const GeneratedMidi& midi, const std::string& trackName) const {
    if (trackName == "melody") {
        return midi.melody;
    } else if (trackName == "bass") {
        return midi.bass;
    } else if (trackName == "counterMelody") {
        return midi.counterMelody;
    } else if (trackName == "pad") {
        return midi.pad;
    } else if (trackName == "strings") {
        return midi.strings;
    } else if (trackName == "fills") {
        return midi.fills;
    } else if (trackName == "rhythm") {
        return midi.rhythm;
    } else if (trackName == "drums" || trackName == "drumGroove") {
        return midi.drumGroove;
    }

    return {};
}

double StemExporter::calculateDuration(const GeneratedMidi& midi) const {
    double maxDuration = 0.0;

    // Check all note tracks
    // Convert ticks to seconds (assuming 120 BPM and 480 ticks per beat)
    const double ticksPerSecond = (120.0 / 60.0) * 480.0;  // 960 ticks per second at 120 BPM
    auto checkNotes = [&maxDuration, ticksPerSecond](const std::vector<MidiNote>& notes) {
        for (const auto& note : notes) {
            double startTime = static_cast<double>(note.startTick) / ticksPerSecond;
            double duration = static_cast<double>(note.durationTicks) / ticksPerSecond;
            double noteEnd = startTime + duration;
            if (noteEnd > maxDuration) {
                maxDuration = noteEnd;
            }
        }
    };

    checkNotes(midi.melody);
    checkNotes(midi.bass);
    checkNotes(midi.counterMelody);
    checkNotes(midi.pad);
    checkNotes(midi.strings);
    checkNotes(midi.fills);
    checkNotes(midi.rhythm);
    checkNotes(midi.drumGroove);

    // Check chords
    for (const auto& chord : midi.chords) {
        double chordEnd = chord.startBeat + chord.duration;
        if (chordEnd > maxDuration) {
            maxDuration = chordEnd;
        }
    }

    return maxDuration > 0.0 ? maxDuration + 0.5 : 10.0;  // Add 0.5s padding, default 10s
}

juce::File StemExporter::generateStemFilename(
    const std::string& trackName,
    const juce::File& outputDirectory,
    Format format,
    const std::string& suffix)
{
    juce::String filename = juce::String(trackName);

    if (!suffix.empty()) {
        filename += "_" + juce::String(suffix);
    }

    filename += getFileExtension(format);

    return outputDirectory.getChildFile(filename);
}

juce::String StemExporter::getFileExtension(Format format) const {
    switch (format) {
        case Format::WAV:
            return ".wav";
        case Format::AIFF:
            return ".aiff";
        case Format::FLAC:
            return ".flac";
        default:
            return ".wav";
    }
}

bool StemExporter::writeAudioFile(
    const juce::AudioBuffer<float>& buffer,
    const juce::File& file,
    const ExportOptions& options)
{
    clearError();

    // Create appropriate audio format writer
    std::unique_ptr<juce::AudioFormatWriter> writer;

    if (options.format == Format::WAV) {
        juce::WavAudioFormat wavFormat;
        writer.reset(wavFormat.createWriterFor(
            new juce::FileOutputStream(file),
            options.sampleRate,
            buffer.getNumChannels(),
            options.bitDepth,
            {},
            0
        ));
    } else if (options.format == Format::AIFF) {
        juce::AiffAudioFormat aiffFormat;
        writer.reset(aiffFormat.createWriterFor(
            new juce::FileOutputStream(file),
            options.sampleRate,
            buffer.getNumChannels(),
            options.bitDepth,
            {},
            0
        ));
    } else if (options.format == Format::FLAC) {
        juce::FlacAudioFormat flacFormat;
        writer.reset(flacFormat.createWriterFor(
            new juce::FileOutputStream(file),
            options.sampleRate,
            buffer.getNumChannels(),
            options.bitDepth,
            {},
            0
        ));
    }

    if (!writer) {
        setError("Could not create audio format writer for: " + file.getFullPathName());
        return false;
    }

    // Write audio data
    bool success = writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());

    if (!success) {
        setError("Failed to write audio data to: " + file.getFullPathName());
        return false;
    }

    return true;
}

} // namespace midikompanion
