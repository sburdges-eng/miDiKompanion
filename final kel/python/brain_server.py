#!/usr/bin/env python3
"""
DAiW Brain Server - OSC Bridge for Python Brain ↔ C++ Body
============================================================
Implements the OSC server that bridges Python music generation logic
with the C++ audio plugin (Body).

This follows the hybrid architecture:
- Python Brain: Therapy logic, NLP, harmony generation, intent processing
- C++ Body: Real-time audio, plugin UI, DAW integration
- OSC Bridge: Communication layer between Brain and Body
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Any
import time

try:
    from pythonosc import osc_server
    from pythonosc import udp_client
    from pythonosc.dispatcher import Dispatcher
    from socketserver import BaseRequestHandler
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("Warning: python-osc not installed. Install with: pip install python-osc")

# Import DAiW music brain modules
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference" / "daiw_music_brain"))
    from music_brain.session.intent_processor import generate_session
    from music_brain.structure.progression import diagnose_progression
    DAiW_AVAILABLE = True
except ImportError:
    DAiW_AVAILABLE = False
    print("Warning: DAiW-Music-Brain not available. Some features disabled.")


class BrainServer:
    """
    OSC server for DAiW Brain.

    Listens for requests from C++ plugin and responds with generated music data.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5005, response_port: int = 5006):
        self.host = host
        self.port = port
        self.response_port = response_port
        self.running = False
        self.current_client_address = None  # Track current request's client

        if not OSC_AVAILABLE:
            raise ImportError("python-osc required for OSC server")

        # Create dispatcher for OSC messages
        self.dispatcher = Dispatcher()
        self.setup_handlers()

        # Create OSC server
        # Note: python-osc doesn't easily expose client addresses in handlers
        # We'll use a workaround by storing the last client address
        self.server = osc_server.ThreadingOSCUDPServer((host, port), self.dispatcher)

        # Store last client address (thread-local would be better, but this works for single client)
        self.last_client_address = None

        print(f"Brain Server initialized on {host}:{port}")
        print(f"Responses will be sent to port {response_port} (or client-specified port)")

    def setup_handlers(self):
        """Setup OSC message handlers."""
        # Main generation endpoint
        self.dispatcher.map("/daiw/generate", self.handle_generate)

        # Analysis endpoints
        self.dispatcher.map("/daiw/analyze/chords", self.handle_analyze_chords)
        self.dispatcher.map("/daiw/analyze/progression", self.handle_analyze_progression)

        # Intent endpoints
        self.dispatcher.map("/daiw/intent/process", self.handle_intent_process)
        self.dispatcher.map("/daiw/intent/suggest", self.handle_intent_suggest)

        # Health check
        self.dispatcher.map("/daiw/ping", self.handle_ping)

        # Default handler for unknown messages
        self.dispatcher.set_default_handler(self.handle_default)

    def send_response(self, response_path: str, response_data: Dict[str, Any], client_address: Optional[tuple] = None):
        """Send OSC response to client."""
        try:
            # Use provided address or last known client address
            if client_address is None:
                client_address = self.last_client_address

            if client_address is None:
                # If no address, try to use default (localhost)
                client_address = ("127.0.0.1", self.response_port)
                print("Warning: No client address available, using default localhost")

            # Extract client IP and port
            client_ip = client_address[0]
            # Use response_port from client if specified in params, otherwise use default
            client_port = self.response_port

            # Create client for this response
            client = udp_client.SimpleUDPClient(client_ip, client_port)

            # Send response as JSON string
            response_json = json.dumps(response_data)
            client.send_message(response_path, response_json)

            print(f"Sent response to {client_ip}:{client_port} at {response_path}")

        except Exception as e:
            print(f"Error sending response to {client_address}: {e}")

    def handle_generate(self, address: str, *args):
        """Handle /daiw/generate request."""
        print(f"Received generate request: {args}")

        try:
            # Note: python-osc doesn't provide client address in handler
            # We'll use the response_port parameter or default
            client_address = None

            # Parse arguments
            if len(args) >= 1:
                # Expect JSON string with parameters
                if isinstance(args[0], str):
                    params = json.loads(args[0])
                else:
                    params = {"text": str(args[0]) if args else ""}
            else:
                params = {}

            # Extract response port if specified
            response_port = params.get("response_port", self.response_port)
            if response_port != self.response_port:
                # Update client address with custom port
                if client_address:
                    client_address = (client_address[0], response_port)

            text = params.get("text", "")
            motivation = params.get("motivation", "")
            chaos = params.get("chaos", 0.5)
            vulnerability = params.get("vulnerability", 0.5)

            # Generate music using DAiW
            if DAiW_AVAILABLE:
                result = generate_session(text, motivation, chaos, vulnerability)

                # Convert to serializable format
                response = {
                    "tempo": result.get("tempo", 120),
                    "key": result.get("key", "C"),
                    "time_sig": result.get("time_sig", [4, 4]),
                    "notes": result.get("notes", []),
                    "chords": result.get("chords", []),
                    "status": "success"
                }
            else:
                # Fallback response
                response = {
                    "tempo": 120,
                    "key": "C",
                    "time_sig": [4, 4],
                    "notes": [],
                    "chords": [],
                    "status": "fallback",
                    "message": "DAiW not available"
                }

            # Send response via OSC
            self.send_response("/daiw/generate/response", response, client_address)
            print(f"Generated response: {json.dumps(response, indent=2)}")

        except Exception as e:
            print(f"Error in handle_generate: {e}")
            response = {
                "status": "error",
                "message": str(e)
            }
            # Try to send error response
            try:
                self.send_response("/daiw/generate/response", response, None)
            except:
                pass

    def handle_analyze_chords(self, address: str, *args):
        """Handle chord analysis request."""
        print(f"Received analyze/chords request: {args}")

        client_address = None

        if len(args) >= 1:
            progression = str(args[0])
            # Analyze progression
            if DAiW_AVAILABLE:
                try:
                    diagnosis = diagnose_progression(progression)
                    response = {
                        "status": "success",
                        "diagnosis": str(diagnosis)
                    }
                except Exception as e:
                    response = {
                        "status": "error",
                        "message": str(e)
                    }
            else:
                response = {
                    "status": "error",
                    "message": "DAiW not available"
                }
        else:
            response = {
                "status": "error",
                "message": "Missing progression argument"
            }

        print(f"Analysis response: {response}")
        self.send_response("/daiw/analyze/chords/response", response, client_address)

    def handle_analyze_progression(self, address: str, *args):
        """Handle progression analysis request."""
        self.handle_analyze_chords(address, *args)

    def handle_intent_process(self, address: str, *args):
        """Handle intent processing request."""
        print(f"Received intent/process request: {args}")

        if len(args) >= 1:
            intent_file = str(args[0])
            # Process intent file
            response = {
                "status": "success",
                "message": f"Processed intent from {intent_file}"
            }
        else:
            response = {
                "status": "error",
                "message": "Missing intent file argument"
            }

        print(f"Intent processing response: {response}")

    def handle_intent_suggest(self, address: str, *args):
        """Handle intent suggestion request."""
        print(f"Received intent/suggest request: {args}")

        client_address = None
        if hasattr(self.dispatcher, 'current_client_address'):
            client_address = self.dispatcher.current_client_address

        if len(args) >= 1:
            emotion = str(args[0])
            # Suggest rules to break
            response = {
                "status": "success",
                "suggestions": [
                    "HARMONY_ModalInterchange",
                    "RHYTHM_Displacement",
                    "PRODUCTION_LoFi"
                ]
            }
        else:
            response = {
                "status": "error",
                "message": "Missing emotion argument"
            }

        print(f"Intent suggestion response: {response}")
        self.send_response("/daiw/intent/suggest/response", response, client_address)

    def handle_ping(self, address: str, *args):
        """Handle ping/health check."""
        client_address = None

        response = {
            "status": "ok",
            "timestamp": time.time(),
            "daiw_available": DAiW_AVAILABLE
        }
        print(f"Ping response: {response}")
        self.send_response("/daiw/ping/response", response, client_address)

    def handle_default(self, address: str, *args):
        """Handle unknown messages."""
        print(f"Unknown message: {address} with args: {args}")

    def start(self):
        """Start the OSC server."""
        if self.running:
            print("Server already running")
            return

        self.running = True
        print(f"Starting Brain Server on {self.host}:{self.port}")
        print("Press Ctrl+C to stop")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server...")
            self.stop()

    def stop(self):
        """Stop the OSC server."""
        self.running = False
        if self.server:
            self.server.shutdown()
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(
        description="DAiW Brain Server - OSC bridge for Python Brain ↔ C++ Body"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="127.0.0.1",
        help="OSC server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5005,
        help="OSC server port (default: 5005)"
    )

    args = parser.parse_args()

    if not OSC_AVAILABLE:
        print("Error: python-osc not installed")
        print("Install with: pip install python-osc")
        return 1

    try:
        server = BrainServer(host=args.host, port=args.port, response_port=args.response_port)
        server.start()
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
