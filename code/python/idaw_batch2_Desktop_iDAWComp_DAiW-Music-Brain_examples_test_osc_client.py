"""
Test OSC Client for DAiW Brain Server
======================================

This script tests the OSC communication with the brain server.
Run this while brain_server.py is running to verify the bridge works.

Usage:
    python examples/test_osc_client.py
"""

import time
import json
import sys

try:
    from pythonosc import udp_client, osc_server, dispatcher
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("[ERROR] pythonosc not installed. Install with: pip install python-osc")
    sys.exit(1)


def test_generate_request():
    """Test sending a generation request to the brain server."""
    print("[TEST] Testing generation request...")
    
    # Create client (sends to server's listen port)
    client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
    
    # Create receiver (listens on server's send port)
    dispatcher_obj = dispatcher.Dispatcher()
    received_messages = []
    
    def handle_result(address, *args):
        """Handle result from server."""
        if address == "/daiw/result":
            json_data = str(args[0]) if args else "{}"
            received_messages.append(("result", json_data))
            print(f"[RESULT] Received result: {json_data[:200]}...")
        elif address == "/daiw/error":
            json_data = str(args[0]) if args else "{}"
            received_messages.append(("error", json_data))
            print(f"[ERROR] Received error: {json_data}")
    
    def handle_pong(address, *args):
        """Handle pong from server."""
        received_messages.append(("pong", args))
        print(f"[PONG] Server is alive: {args}")
    
    dispatcher_obj.map("/daiw/result", handle_result)
    dispatcher_obj.map("/daiw/error", handle_result)
    dispatcher_obj.map("/daiw/pong", handle_pong)
    
    # Start receiver
    server = osc_server.ThreadingOSCUDPServer(
        ("127.0.0.1", 9001),
        dispatcher_obj
    )
    
    import threading
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    print("[INFO] Test client ready. Sending requests...")
    print()
    
    # Test 1: Ping
    print("[TEST 1] Sending ping...")
    client.send_message("/daiw/ping", [])
    time.sleep(0.5)
    
    # Test 2: Generate request
    print("[TEST 2] Sending generation request...")
    text = "I feel deep grief and longing for what was lost"
    motivation = 7.0
    chaos = 5.0
    vulnerability = 6.0
    
    client.send_message(
        "/daiw/generate",
        [text, motivation, chaos, vulnerability]
    )
    print(f"  Text: {text}")
    print(f"  Motivation: {motivation}")
    print(f"  Chaos: {chaos}")
    print(f"  Vulnerability: {vulnerability}")
    print()
    
    # Wait for response
    print("[WAIT] Waiting for response (5 seconds)...")
    time.sleep(5)
    
    # Check results
    print()
    print("[RESULTS]")
    if received_messages:
        for msg_type, data in received_messages:
            if msg_type == "result":
                try:
                    result = json.loads(data)
                    print(f"  ✓ Received result:")
                    print(f"    Status: {result.get('status')}")
                    if 'affect' in result:
                        print(f"    Affect: {result['affect'].get('primary')}")
                    if 'plan' in result:
                        plan = result['plan']
                        print(f"    Tempo: {plan.get('tempo_bpm')} BPM")
                        print(f"    Key: {plan.get('key')} {plan.get('mode')}")
                        print(f"    Chords: {' - '.join(plan.get('chords', []))}")
                    if 'midi_events' in result:
                        print(f"    MIDI Events: {len(result['midi_events'])} events")
                except json.JSONDecodeError:
                    print(f"    Raw data: {data[:200]}...")
            elif msg_type == "error":
                print(f"  ✗ Received error: {data}")
            elif msg_type == "pong":
                print(f"  ✓ Server is alive")
    else:
        print("  ✗ No messages received")
        print("  [HINT] Make sure brain_server.py is running:")
        print("         python brain_server.py")
    
    # Cleanup
    server.shutdown()
    print()
    print("[DONE] Test complete")


if __name__ == "__main__":
    if not OSC_AVAILABLE:
        sys.exit(1)
    
    test_generate_request()

