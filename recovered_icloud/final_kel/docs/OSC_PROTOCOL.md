# OSC Protocol Documentation

## Python Brain ↔ C++ Body Communication

This document describes the OSC (Open Sound Control) protocol used for communication between the Python Brain Server (`brain_server.py`) and the C++ Plugin Body.

## Architecture

- **Python Brain Server**: Listens on port 5005 (default), sends responses to client-specified port (default 5006)
- **C++ Plugin Client**: Sends requests to port 5005, listens for responses on port 5006 (default)

## Message Format

All messages use JSON strings as arguments for structured data exchange.

### Request Messages (C++ → Python)

#### `/daiw/generate`

Request music generation from Python brain.

**Arguments:**

- `arg[0]` (string): JSON object with:

  ```json
  {
    "text": "user input text",
    "motivation": 0.5,
    "chaos": 0.5,
    "vulnerability": 0.5,
    "response_port": 5006
  }
  ```

**Response:** `/daiw/generate/response`

- `arg[0]` (string): JSON object with:

  ```json
  {
    "status": "success",
    "tempo": 120,
    "key": "C",
    "time_sig": [4, 4],
    "notes": [],
    "chords": []
  }
  ```

#### `/daiw/analyze/chords`

Request chord progression analysis.

**Arguments:**

- `arg[0]` (string): Chord progression (e.g., "C F G Am")

**Response:** `/daiw/analyze/chords/response`

- `arg[0]` (string): JSON object with:

  ```json
  {
    "status": "success",
    "diagnosis": "analysis text"
  }
  ```

#### `/daiw/intent/process`

Request intent file processing.

**Arguments:**

- `arg[0]` (string): Path to intent file

**Response:** `/daiw/intent/process/response`

- `arg[0]` (string): JSON object with:

  ```json
  {
    "status": "success",
    "message": "Processed intent from <file>"
  }
  ```

#### `/daiw/intent/suggest`

Request intent suggestions based on emotion.

**Arguments:**

- `arg[0]` (string): Emotion name

**Response:** `/daiw/intent/suggest/response`

- `arg[0]` (string): JSON object with:

  ```json
  {
    "status": "success",
    "suggestions": [
      "HARMONY_ModalInterchange",
      "RHYTHM_Displacement",
      "PRODUCTION_LoFi"
    ]
  }
  ```

#### `/daiw/ping`

Health check / connection test.

**Arguments:** None

**Response:** `/daiw/ping/response`

- `arg[0]` (string): JSON object with:

  ```json
  {
    "status": "ok",
    "timestamp": 1234567890.123,
    "daiw_available": true
  }
  ```

## Error Responses

All endpoints may return error responses:

```json
{
  "status": "error",
  "message": "Error description"
}
```

## Usage Examples

### Python Server

```bash
python python/brain_server.py --host 127.0.0.1 --port 5005 --response-port 5006
```

### C++ Client

```cpp
#include "bridge/OSCClient.h"

kelly::OSCClient client;
client.connect("127.0.0.1", 5005, 5006);

// Request generation
client.requestGenerate(
    "I feel happy today",
    0.7f,  // motivation
    0.3f,  // chaos
    0.5f,  // vulnerability
    [](const std::string& response) {
        // Parse JSON response
        std::cout << "Response: " << response << std::endl;
    }
);

// Process messages periodically (e.g., in timer)
client.processMessages();
```

## Implementation Notes

1. **Thread Safety**: OSC messages are received on a separate thread. Callbacks should be thread-safe or use message queues.

2. **Timeouts**: Requests timeout after 5 seconds if no response is received.

3. **Connection**: The C++ client should connect on plugin initialization and disconnect on shutdown.

4. **Error Handling**: All requests should handle timeout and connection errors gracefully.

## Port Configuration

- Default server port: 5005
- Default response port: 5006
- Both can be configured via command-line arguments (Python) or constructor parameters (C++)
