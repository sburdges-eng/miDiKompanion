# MIDI Keyboards in C++ Programming - Complete Guide

**Date:** 2025-01-XX  
**Purpose:** Comprehensive guide on integrating MIDI keyboards into C++ applications

---

## Table of Contents

1. [Overview](#overview)
2. [MIDI Libraries for C++](#midi-libraries-for-c)
3. [MIDI Message Format](#midi-message-format)
4. [Basic Implementation](#basic-implementation)
5. [Advanced Features](#advanced-features)
6. [Platform-Specific Details](#platform-specific-details)
7. [Code Examples](#code-examples)
8. [Best Practices](#best-practices)

---

## Overview

MIDI keyboards communicate with C++ programs through the MIDI (Musical Instrument Digital Interface) protocol. The integration involves:

1. **Device Enumeration** - Discovering available MIDI keyboards
2. **Input Handling** - Receiving MIDI messages from the keyboard
3. **Message Parsing** - Interpreting MIDI data (notes, controls, etc.)
4. **Output (Optional)** - Sending MIDI messages back to the keyboard
5. **Real-Time Processing** - Handling MIDI events with low latency

---

## MIDI Libraries for C++

### 1. RtMidi

**Description:** Cross-platform C++ library for real-time MIDI I/O  
**Platforms:** Windows, macOS, Linux  
**License:** MIT  
**Website:** https://github.com/thestk/rtmidi

**Features:**
- Single header + source file (easy integration)
- Cross-platform API (Windows Multimedia, macOS CoreMIDI, Linux ALSA/JACK)
- Separate classes for input (`RtMidiIn`) and output (`RtMidiOut`)
- Callback-based and polling modes
- Device enumeration
- Timestamped messages (delta time in seconds)

**Pros:**
- Lightweight and simple
- Well-documented
- Active development
- No external dependencies

**Cons:**
- No MIDI 2.0 support
- No built-in timing/sequencing
- Basic API (no high-level abstractions)

---

### 2. libremidi

**Description:** Modern C++20 library for MIDI 1 and MIDI 2  
**Platforms:** Windows, macOS, Linux, WebMIDI  
**License:** MIT  
**Website:** https://github.com/celtera/libremidi

**Features:**
- MIDI 1.0 and MIDI 2.0 support
- C++20 modern design
- Device enumeration and hotplug support
- Consistent API across platforms
- MIDI file I/O
- Real-time and file-based MIDI

**Pros:**
- Modern C++ design
- MIDI 2.0 support
- Hotplug support
- Active development

**Cons:**
- Requires C++20
- Less mature than RtMidi
- Larger codebase

---

### 3. JUCE

**Description:** Comprehensive C++ framework for audio/MIDI applications  
**Platforms:** Windows, macOS, Linux, iOS, Android  
**License:** GPL/Commercial  
**Website:** https://juce.com

**Features:**
- Complete audio/MIDI framework
- Plugin development (VST, AU, AAX)
- GUI framework
- Cross-platform abstractions
- Real-time audio processing
- MIDI message handling
- Device management

**Pros:**
- Most comprehensive solution
- Industry standard for plugins
- Excellent documentation
- Active community

**Cons:**
- Large framework (overkill for simple projects)
- Commercial license required for closed-source
- Steeper learning curve

---

### 4. Drumstick

**Description:** C++ MIDI libraries using Qt  
**Platforms:** Windows, macOS, Linux  
**License:** GPL  
**Website:** https://drumstick.sourceforge.io

**Features:**
- Qt-based (good for GUI applications)
- ALSA wrapper for Linux
- Standard MIDI File (SMF) support
- Real-time MIDI I/O
- GUI widgets

**Pros:**
- Good Qt integration
- MIDI file support
- GUI components

**Cons:**
- Qt dependency
- Less active development
- Linux-focused (ALSA)

---

### 5. CFugue

**Description:** High-level music programming library  
**Platforms:** Cross-platform  
**License:** LGPL  
**Website:** https://cfugue.sourceforge.net

**Features:**
- High-level abstraction
- Music notation support
- Simplified MIDI programming

**Pros:**
- Easy to use
- High-level API

**Cons:**
- Less control over MIDI details
- Limited real-time capabilities

---

## MIDI Message Format

### Message Structure

MIDI messages consist of:
1. **Status Byte** - Message type and channel (0x80-0xFF)
2. **Data Bytes** - Message-specific data (0x00-0x7F)

### Common Message Types

#### Note On (0x90-0x9F)
```
Status: 0x9n (n = channel 0-15)
Data 1: Note number (0-127, C4 = 60)
Data 2: Velocity (0-127, 0 = note off)
```

#### Note Off (0x80-0x8F)
```
Status: 0x8n (n = channel 0-15)
Data 1: Note number (0-127)
Data 2: Release velocity (0-127, often 64)
```

#### Control Change (0xB0-0xBF)
```
Status: 0xBn (n = channel 0-15)
Data 1: Controller number (0-127)
Data 2: Controller value (0-127)
```

Common CC numbers:
- 1: Modulation wheel
- 7: Volume
- 10: Pan
- 64: Sustain pedal
- 71-74: Sound controllers

#### Pitch Bend (0xE0-0xEF)
```
Status: 0xEn (n = channel 0-15)
Data 1: LSB (0-127)
Data 2: MSB (0-127)
Combined: 14-bit value (0-16383, center = 8192)
```

#### Program Change (0xC0-0xCF)
```
Status: 0xCn (n = channel 0-15)
Data 1: Program number (0-127)
```

#### Aftertouch (0xD0-0xDF - Channel, 0xA0-0xAF - Polyphonic)
```
Status: 0xDn (channel) or 0xAn (polyphonic)
Data 1: Pressure value (0-127) or Note number (polyphonic)
Data 2: Pressure value (polyphonic only)
```

---

## Basic Implementation

### Using RtMidi

#### 1. Device Enumeration

```cpp
#include <RtMidi.h>
#include <iostream>
#include <vector>

void listMidiDevices() {
    RtMidiIn midiin;
    unsigned int nPorts = midiin.getPortCount();
    
    std::cout << "Available MIDI Input Devices:\n";
    for (unsigned int i = 0; i < nPorts; i++) {
        std::string portName = midiin.getPortName(i);
        std::cout << "  [" << i << "] " << portName << std::endl;
    }
}
```

#### 2. MIDI Input with Callback

```cpp
#include <RtMidi.h>
#include <iostream>
#include <vector>

// Callback function to handle MIDI messages
void midiCallback(double deltatime, 
                  std::vector<unsigned char> *message, 
                  void *userData) {
    unsigned int nBytes = message->size();
    
    if (nBytes > 0) {
        unsigned char status = message->at(0);
        unsigned char messageType = status & 0xF0;
        unsigned char channel = status & 0x0F;
        
        // Note On
        if (messageType == 0x90 && nBytes >= 3) {
            unsigned char note = message->at(1);
            unsigned char velocity = message->at(2);
            
            if (velocity > 0) {
                std::cout << "Note On: Channel " << (int)channel 
                          << ", Note " << (int)note 
                          << ", Velocity " << (int)velocity << std::endl;
            } else {
                // Note Off (velocity 0)
                std::cout << "Note Off: Channel " << (int)channel 
                          << ", Note " << (int)note << std::endl;
            }
        }
        // Note Off
        else if (messageType == 0x80 && nBytes >= 3) {
            unsigned char note = message->at(1);
            unsigned char velocity = message->at(2);
            
            std::cout << "Note Off: Channel " << (int)channel 
                      << ", Note " << (int)note 
                      << ", Velocity " << (int)velocity << std::endl;
        }
        // Control Change
        else if (messageType == 0xB0 && nBytes >= 3) {
            unsigned char controller = message->at(1);
            unsigned char value = message->at(2);
            
            std::cout << "CC: Channel " << (int)channel 
                      << ", Controller " << (int)controller 
                      << ", Value " << (int)value << std::endl;
        }
        // Pitch Bend
        else if (messageType == 0xE0 && nBytes >= 3) {
            unsigned char lsb = message->at(1);
            unsigned char msb = message->at(2);
            int bendValue = (msb << 7) | lsb;  // 14-bit value
            
            std::cout << "Pitch Bend: Channel " << (int)channel 
                      << ", Value " << bendValue << std::endl;
        }
    }
}

int main() {
    RtMidiIn *midiin = new RtMidiIn();
    
    // List available devices
    unsigned int nPorts = midiin->getPortCount();
    if (nPorts == 0) {
        std::cout << "No MIDI input devices found!" << std::endl;
        delete midiin;
        return 1;
    }
    
    std::cout << "Available MIDI Input Devices:\n";
    for (unsigned int i = 0; i < nPorts; i++) {
        std::cout << "  [" << i << "] " << midiin->getPortName(i) << std::endl;
    }
    
    // Open first available port (or let user choose)
    midiin->openPort(0);
    
    // Set callback function
    midiin->setCallback(&midiCallback);
    
    // Don't ignore any messages
    midiin->ignoreTypes(false, false, false);
    
    std::cout << "\nReading MIDI input... Press Enter to quit.\n";
    std::cin.get();
    
    delete midiin;
    return 0;
}
```

#### 3. MIDI Output

```cpp
#include <RtMidi.h>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    RtMidiOut *midiout = new RtMidiOut();
    
    // List available output devices
    unsigned int nPorts = midiout->getPortCount();
    if (nPorts == 0) {
        std::cout << "No MIDI output devices found!" << std::endl;
        delete midiout;
        return 1;
    }
    
    std::cout << "Available MIDI Output Devices:\n";
    for (unsigned int i = 0; i < nPorts; i++) {
        std::cout << "  [" << i << "] " << midiout->getPortName(i) << std::endl;
    }
    
    // Open first available port
    midiout->openPort(0);
    
    // Send Note On (Middle C, Channel 0, Velocity 100)
    std::vector<unsigned char> message;
    message.push_back(0x90);  // Note On, Channel 0
    message.push_back(60);    // Middle C
    message.push_back(100);   // Velocity
    
    midiout->sendMessage(&message);
    std::cout << "Sent Note On: Middle C\n";
    
    // Wait 500ms
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Send Note Off
    message[0] = 0x80;  // Note Off, Channel 0
    message[2] = 0;      // Release velocity
    midiout->sendMessage(&message);
    std::cout << "Sent Note Off: Middle C\n";
    
    delete midiout;
    return 0;
}
```

---

### Using libremidi

#### Basic Input Example

```cpp
#include <libremidi/libremidi.hpp>
#include <iostream>

int main() {
    libremidi::midi_in midiin;
    
    // List available input devices
    auto ports = midiin.get_ports();
    std::cout << "Available MIDI Input Devices:\n";
    for (size_t i = 0; i < ports.size(); i++) {
        std::cout << "  [" << i << "] " << ports[i] << std::endl;
    }
    
    if (ports.empty()) {
        std::cout << "No MIDI input devices found!" << std::endl;
        return 1;
    }
    
    // Open first port
    midiin.open_port(0);
    
    // Set callback
    midiin.set_callback([](const libremidi::message& message) {
        if (message.size() >= 3) {
            unsigned char status = message[0];
            unsigned char messageType = status & 0xF0;
            
            if (messageType == 0x90) {  // Note On
                unsigned char note = message[1];
                unsigned char velocity = message[2];
                std::cout << "Note On: " << (int)note 
                          << ", Velocity: " << (int)velocity << std::endl;
            } else if (messageType == 0x80) {  // Note Off
                unsigned char note = message[1];
                std::cout << "Note Off: " << (int)note << std::endl;
            }
        }
    });
    
    std::cout << "\nReading MIDI input... Press Enter to quit.\n";
    std::cin.get();
    
    return 0;
}
```

---

## Advanced Features

### 1. MIDI Message Parser Class

```cpp
#include <RtMidi.h>
#include <vector>
#include <functional>

class MidiMessageParser {
public:
    struct NoteEvent {
        unsigned char channel;
        unsigned char note;
        unsigned char velocity;
        bool isNoteOn;
    };
    
    struct ControlChange {
        unsigned char channel;
        unsigned char controller;
        unsigned char value;
    };
    
    struct PitchBend {
        unsigned char channel;
        int value;  // 14-bit: 0-16383, center = 8192
    };
    
    // Callback function types
    using NoteCallback = std::function<void(const NoteEvent&)>;
    using CCCallback = std::function<void(const ControlChange&)>;
    using PitchBendCallback = std::function<void(const PitchBend&)>;
    
    void setNoteCallback(NoteCallback cb) { noteCallback_ = cb; }
    void setCCCallback(CCCallback cb) { ccCallback_ = cb; }
    void setPitchBendCallback(PitchBendCallback cb) { pitchBendCallback_ = cb; }
    
    void parseMessage(const std::vector<unsigned char>& message) {
        if (message.empty()) return;
        
        unsigned char status = message[0];
        unsigned char messageType = status & 0xF0;
        unsigned char channel = status & 0x0F;
        
        switch (messageType) {
            case 0x90:  // Note On
            case 0x80:  // Note Off
                if (message.size() >= 3) {
                    NoteEvent event;
                    event.channel = channel;
                    event.note = message[1];
                    event.velocity = message[2];
                    event.isNoteOn = (messageType == 0x90) && (event.velocity > 0);
                    
                    if (noteCallback_) {
                        noteCallback_(event);
                    }
                }
                break;
                
            case 0xB0:  // Control Change
                if (message.size() >= 3) {
                    ControlChange cc;
                    cc.channel = channel;
                    cc.controller = message[1];
                    cc.value = message[2];
                    
                    if (ccCallback_) {
                        ccCallback_(cc);
                    }
                }
                break;
                
            case 0xE0:  // Pitch Bend
                if (message.size() >= 3) {
                    PitchBend pb;
                    pb.channel = channel;
                    unsigned char lsb = message[1];
                    unsigned char msb = message[2];
                    pb.value = (msb << 7) | lsb;
                    
                    if (pitchBendCallback_) {
                        pitchBendCallback_(pb);
                    }
                }
                break;
        }
    }
    
private:
    NoteCallback noteCallback_;
    CCCallback ccCallback_;
    PitchBendCallback pitchBendCallback_;
};
```

### 2. Thread-Safe MIDI Handler

```cpp
#include <RtMidi.h>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>

class ThreadSafeMidiHandler {
public:
    struct MidiEvent {
        double timestamp;
        std::vector<unsigned char> message;
    };
    
    ThreadSafeMidiHandler() : running_(false) {}
    
    void start(RtMidiIn* midiin) {
        running_ = true;
        midiin->setCallback(&ThreadSafeMidiHandler::midiCallback, this);
        midiin->ignoreTypes(false, false, false);
    }
    
    void stop() {
        running_ = false;
    }
    
    bool getNextEvent(MidiEvent& event) {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (eventQueue_.empty()) {
            return false;
        }
        event = eventQueue_.front();
        eventQueue_.pop();
        return true;
    }
    
private:
    static void midiCallback(double deltatime, 
                            std::vector<unsigned char> *message, 
                            void *userData) {
        ThreadSafeMidiHandler* handler = 
            static_cast<ThreadSafeMidiHandler*>(userData);
        
        if (handler->running_) {
            MidiEvent event;
            event.timestamp = deltatime;
            event.message = *message;
            
            std::lock_guard<std::mutex> lock(handler->queueMutex_);
            handler->eventQueue_.push(event);
        }
    }
    
    std::queue<MidiEvent> eventQueue_;
    std::mutex queueMutex_;
    std::atomic<bool> running_;
};
```

### 3. MIDI Keyboard State Tracker

```cpp
#include <map>
#include <vector>

class MidiKeyboardState {
public:
    struct NoteState {
        bool isPressed;
        unsigned char velocity;
        double timestamp;
    };
    
    void noteOn(unsigned char channel, unsigned char note, 
                unsigned char velocity, double timestamp) {
        NoteKey key = {channel, note};
        NoteState state;
        state.isPressed = true;
        state.velocity = velocity;
        state.timestamp = timestamp;
        pressedNotes_[key] = state;
    }
    
    void noteOff(unsigned char channel, unsigned char note, 
                 double timestamp) {
        NoteKey key = {channel, note};
        pressedNotes_.erase(key);
    }
    
    bool isNotePressed(unsigned char channel, unsigned char note) const {
        NoteKey key = {channel, note};
        return pressedNotes_.find(key) != pressedNotes_.end();
    }
    
    std::vector<unsigned char> getPressedNotes(unsigned char channel) const {
        std::vector<unsigned char> notes;
        for (const auto& pair : pressedNotes_) {
            if (pair.first.channel == channel) {
                notes.push_back(pair.first.note);
            }
        }
        return notes;
    }
    
    void clear() {
        pressedNotes_.clear();
    }
    
private:
    struct NoteKey {
        unsigned char channel;
        unsigned char note;
        
        bool operator<(const NoteKey& other) const {
            if (channel != other.channel) {
                return channel < other.channel;
            }
            return note < other.note;
        }
    };
    
    std::map<NoteKey, NoteState> pressedNotes_;
};
```

---

## Platform-Specific Details

### Windows

**API:** Windows Multimedia API (winmm.dll)

```cpp
// Windows-specific MIDI output example
#include <windows.h>
#include <mmsystem.h>

void sendMidiNoteWindows(unsigned char note, unsigned char velocity) {
    HMIDIOUT hMidiOut;
    midiOutOpen(&hMidiOut, 0, 0, 0, CALLBACK_NULL);
    
    // Note On: 0x90 (status) | 0x00 (channel) = 0x90
    DWORD msg = 0x007F0090 | (note << 8) | (velocity << 16);
    midiOutShortMsg(hMidiOut, msg);
    
    Sleep(500);  // Hold note
    
    // Note Off
    msg = 0x00000080 | (note << 8);
    midiOutShortMsg(hMidiOut, msg);
    
    midiOutClose(hMidiOut);
}
```

### macOS

**API:** CoreMIDI Framework

```cpp
// macOS uses CoreMIDI (abstracted by RtMidi/libremidi)
// No platform-specific code needed when using libraries
```

### Linux

**API:** ALSA Sequencer or JACK

```cpp
// Linux uses ALSA or JACK (abstracted by RtMidi/libremidi)
// No platform-specific code needed when using libraries
```

---

## Code Examples

### Complete MIDI Keyboard Application

```cpp
#include <RtMidi.h>
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>

class MidiKeyboardApp {
public:
    MidiKeyboardApp() : midiin_(nullptr), running_(false) {}
    
    ~MidiKeyboardApp() {
        stop();
    }
    
    bool initialize() {
        midiin_ = new RtMidiIn();
        
        unsigned int nPorts = midiin_->getPortCount();
        if (nPorts == 0) {
            std::cout << "No MIDI input devices found!" << std::endl;
            return false;
        }
        
        std::cout << "Available MIDI Input Devices:\n";
        for (unsigned int i = 0; i < nPorts; i++) {
            std::cout << "  [" << i << "] " 
                      << midiin_->getPortName(i) << std::endl;
        }
        
        // Open first device
        midiin_->openPort(0);
        midiin_->setCallback(&MidiKeyboardApp::midiCallback, this);
        midiin_->ignoreTypes(false, false, false);
        
        return true;
    }
    
    void start() {
        running_ = true;
        std::cout << "MIDI input active. Press Enter to quit.\n";
        std::cin.get();
    }
    
    void stop() {
        running_ = false;
        if (midiin_) {
            delete midiin_;
            midiin_ = nullptr;
        }
    }
    
    void printNoteName(unsigned char note) {
        const char* noteNames[] = {
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
        };
        int octave = (note / 12) - 1;
        int noteIndex = note % 12;
        std::cout << noteNames[noteIndex] << octave;
    }
    
private:
    static void midiCallback(double deltatime,
                            std::vector<unsigned char> *message,
                            void *userData) {
        MidiKeyboardApp* app = static_cast<MidiKeyboardApp*>(userData);
        app->handleMidiMessage(*message);
    }
    
    void handleMidiMessage(const std::vector<unsigned char>& message) {
        if (message.empty()) return;
        
        unsigned char status = message[0];
        unsigned char messageType = status & 0xF0;
        unsigned char channel = status & 0x0F;
        
        if (messageType == 0x90 && message.size() >= 3) {
            unsigned char note = message[1];
            unsigned char velocity = message[2];
            
            if (velocity > 0) {
                std::cout << "[Note On] Channel " << (int)channel << ", ";
                printNoteName(note);
                std::cout << " (" << (int)note << "), Velocity: " 
                          << (int)velocity << std::endl;
                
                pressedNotes_[note] = true;
            } else {
                // Note Off (velocity 0)
                std::cout << "[Note Off] Channel " << (int)channel << ", ";
                printNoteName(note);
                std::cout << " (" << (int)note << ")" << std::endl;
                
                pressedNotes_.erase(note);
            }
        }
        else if (messageType == 0x80 && message.size() >= 3) {
            unsigned char note = message[1];
            std::cout << "[Note Off] Channel " << (int)channel << ", ";
            printNoteName(note);
            std::cout << " (" << (int)note << ")" << std::endl;
            
            pressedNotes_.erase(note);
        }
        else if (messageType == 0xB0 && message.size() >= 3) {
            unsigned char controller = message[1];
            unsigned char value = message[2];
            
            std::cout << "[CC] Channel " << (int)channel 
                      << ", Controller " << (int)controller 
                      << ", Value " << (int)value << std::endl;
        }
        else if (messageType == 0xE0 && message.size() >= 3) {
            unsigned char lsb = message[1];
            unsigned char msb = message[2];
            int bendValue = (msb << 7) | lsb;
            float bendSemitones = ((bendValue - 8192) / 8192.0f) * 2.0f;
            
            std::cout << "[Pitch Bend] Channel " << (int)channel 
                      << ", Value: " << bendValue 
                      << " (" << bendSemitones << " semitones)" << std::endl;
        }
    }
    
    RtMidiIn* midiin_;
    std::atomic<bool> running_;
    std::map<unsigned char, bool> pressedNotes_;
};

int main() {
    MidiKeyboardApp app;
    
    if (!app.initialize()) {
        return 1;
    }
    
    app.start();
    app.stop();
    
    return 0;
}
```

---

## Best Practices

### 1. Error Handling

```cpp
void safeMidiOperation() {
    try {
        RtMidiIn midiin;
        // ... operations
    } catch (RtMidiError& error) {
        std::cerr << "MIDI Error: " << error.getMessage() << std::endl;
    }
}
```

### 2. Resource Management

```cpp
// Use RAII for automatic cleanup
class MidiInput {
    std::unique_ptr<RtMidiIn> midiin_;
    
public:
    MidiInput() : midiin_(std::make_unique<RtMidiIn>()) {}
    ~MidiInput() {
        // Automatic cleanup
    }
};
```

### 3. Thread Safety

- Use mutexes for shared MIDI data
- Use atomic variables for flags
- Consider lock-free queues for high-performance scenarios

### 4. Latency Considerations

- Process MIDI in real-time threads
- Minimize allocations in callbacks
- Use pre-allocated buffers
- Consider priority scheduling (real-time threads)

### 5. Message Filtering

```cpp
// Filter specific channels or message types
bool shouldProcessMessage(const std::vector<unsigned char>& message,
                          unsigned char targetChannel) {
    if (message.empty()) return false;
    
    unsigned char channel = message[0] & 0x0F;
    return channel == targetChannel;
}
```

---

## Integration with DAiW

For DAiW-Music-Brain integration:

1. **Receive MIDI from keyboard** - Use RtMidi/libremidi to capture input
2. **Parse MIDI messages** - Extract notes, velocity, CC values
3. **Convert to DAiW format** - Map MIDI to internal data structures
4. **Process with DAiW** - Feed into harmony/groove/intent processors
5. **Generate MIDI output** - Send results back to keyboard or DAW

### Example Integration Point

```cpp
// Pseudo-code for DAiW integration
class DaiwMidiBridge {
    MidiKeyboardState keyboardState_;
    HarmonyGenerator harmonyGen_;
    
    void onMidiNote(const MidiMessageParser::NoteEvent& event) {
        // Update keyboard state
        if (event.isNoteOn) {
            keyboardState_.noteOn(event.channel, event.note, 
                                  event.velocity, getCurrentTime());
        } else {
            keyboardState_.noteOff(event.channel, event.note, 
                                   getCurrentTime());
        }
        
        // Feed to DAiW harmony generator
        auto chord = harmonyGen_.analyzeCurrentChord(keyboardState_);
        // ... process with DAiW
    }
};
```

---

## Summary

### Library Recommendations

- **Simple Projects:** RtMidi
- **Modern C++20 Projects:** libremidi
- **Audio Plugins/DAWs:** JUCE
- **Qt Applications:** Drumstick
- **High-Level Music:** CFugue

### Key Concepts

1. **MIDI is event-based** - Messages arrive asynchronously
2. **Real-time processing** - Low latency is critical
3. **Thread safety** - MIDI callbacks run in separate threads
4. **Message parsing** - Status byte determines message type
5. **Platform abstraction** - Libraries hide OS-specific details

---

**Last Updated:** 2025-01-XX  
**Note:** Code examples are simplified. Production code should include comprehensive error handling, resource management, and thread safety measures.

