#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace penta::osc {

/**
 * OSC message value types
 */
using OSCValue = std::variant<
    int32_t,
    float,
    std::string,
    std::vector<uint8_t>  // blob
>;

/**
 * OSC message structure
 */
struct OSCMessage {
    static constexpr size_t kMaxAddressLength = 128;
    static constexpr size_t kMaxArguments = 16;
    
    std::array<char, kMaxAddressLength> address;
    std::array<OSCValue, kMaxArguments> arguments;
    size_t argumentCount;
    uint64_t timestamp;  // Samples or system time
    
    OSCMessage() 
        : address{}
        , arguments{}
        , argumentCount(0)
        , timestamp(0) {}
    
    void setAddress(const char* addr);
    void addInt(int32_t value);
    void addFloat(float value);
    void addString(const char* value);
    void clear();
};

/**
 * Lock-free message queue for RT-safe OSC communication
 * Single-producer, single-consumer queue
 */
class RTMessageQueue {
public:
    explicit RTMessageQueue(size_t capacity = 4096);
    ~RTMessageQueue();
    
    // Non-copyable, non-movable
    RTMessageQueue(const RTMessageQueue&) = delete;
    RTMessageQueue& operator=(const RTMessageQueue&) = delete;
    
    // RT-safe: Push message (returns false if full)
    bool push(const OSCMessage& message) noexcept;
    
    // RT-safe: Pop message (returns false if empty)
    bool pop(OSCMessage& outMessage) noexcept;
    
    // RT-safe: Check if queue is empty
    bool isEmpty() const noexcept;
    
    // RT-safe: Get approximate size
    size_t size() const noexcept;
    
    size_t capacity() const noexcept { return capacity_; }
    
private:
    std::vector<OSCMessage> buffer_;
    size_t capacity_;
    std::atomic<size_t> writeIndex_;
    std::atomic<size_t> readIndex_;
};

} // namespace penta::osc
