"""
Tests for DAiW OSC Brain Server
================================

Tests the OSC communication protocol and server functionality.
"""

import json
import time
import threading
import pytest

try:
    from pythonosc import udp_client, osc_server, dispatcher
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    pytestmark = pytest.mark.skip("python-osc not installed")

from brain_server import DAiWBrainServer


@pytest.fixture
def osc_server_instance():
    """Create a test OSC server instance."""
    if not OSC_AVAILABLE:
        pytest.skip("python-osc not available")
    
    server = DAiWBrainServer(listen_port=9002, send_port=9003)
    return server


@pytest.fixture
def test_client():
    """Create a test OSC client."""
    if not OSC_AVAILABLE:
        pytest.skip("python-osc not available")
    
    return udp_client.SimpleUDPClient("127.0.0.1", 9002)


class TestOSCServer:
    """Test OSC server functionality."""
    
    def test_server_initialization(self, osc_server_instance):
        """Test that server initializes correctly."""
        assert osc_server_instance.listen_port == 9002
        assert osc_server_instance.send_port == 9003
        assert osc_server_instance.client is not None
        assert osc_server_instance.dispatcher is not None
    
    def test_ping_pong(self, osc_server_instance, test_client):
        """Test ping/pong health check."""
        received = []
        
        def handle_pong(address, *args):
            received.append(("pong", args))
        
        # Create receiver for pong
        dispatcher_obj = dispatcher.Dispatcher()
        dispatcher_obj.map("/daiw/pong", handle_pong)
        receiver = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", 9003),
            dispatcher_obj
        )
        
        # Start receiver in background
        receiver_thread = threading.Thread(target=receiver.serve_forever)
        receiver_thread.daemon = True
        receiver_thread.start()
        
        # Start server in background
        server_thread = threading.Thread(target=osc_server_instance.start)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(0.5)
        
        # Send ping
        test_client.send_message("/daiw/ping", [])
        
        # Wait for response
        time.sleep(0.5)
        
        # Cleanup
        receiver.shutdown()
        osc_server_instance.stop()
        
        # Verify pong received
        assert len(received) > 0
        assert received[0][0] == "pong"
    
    def test_generate_request(self, osc_server_instance, test_client):
        """Test generation request handling."""
        received = []
        
        def handle_result(address, *args):
            if address == "/daiw/result":
                json_data = str(args[0]) if args else "{}"
                received.append(("result", json_data))
            elif address == "/daiw/error":
                json_data = str(args[0]) if args else "{}"
                received.append(("error", json_data))
        
        # Create receiver
        dispatcher_obj = dispatcher.Dispatcher()
        dispatcher_obj.map("/daiw/result", handle_result)
        dispatcher_obj.map("/daiw/error", handle_result)
        receiver = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", 9003),
            dispatcher_obj
        )
        
        # Start receiver
        receiver_thread = threading.Thread(target=receiver.serve_forever)
        receiver_thread.daemon = True
        receiver_thread.start()
        
        # Start server
        server_thread = threading.Thread(target=osc_server_instance.start)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(0.5)
        
        # Send generation request
        test_client.send_message(
            "/daiw/generate",
            ["I feel deep grief", 7.0, 5.0, 6.0]
        )
        
        # Wait for response
        time.sleep(2.0)
        
        # Cleanup
        receiver.shutdown()
        osc_server_instance.stop()
        
        # Verify result received
        assert len(received) > 0
        assert received[0][0] in ["result", "error"]
        
        if received[0][0] == "result":
            result_data = json.loads(received[0][1])
            assert result_data["status"] == "success"
            assert "affect" in result_data
            assert "plan" in result_data
            assert "midi_events" in result_data
    
    def test_invalid_request(self, osc_server_instance, test_client):
        """Test error handling for invalid requests."""
        received = []
        
        def handle_error(address, *args):
            json_data = str(args[0]) if args else "{}"
            received.append(("error", json_data))
        
        # Create receiver
        dispatcher_obj = dispatcher.Dispatcher()
        dispatcher_obj.map("/daiw/error", handle_error)
        receiver = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", 9003),
            dispatcher_obj
        )
        
        # Start receiver
        receiver_thread = threading.Thread(target=receiver.serve_forever)
        receiver_thread.daemon = True
        receiver_thread.start()
        
        # Start server
        server_thread = threading.Thread(target=osc_server_instance.start)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(0.5)
        
        # Send invalid request (empty text)
        test_client.send_message(
            "/daiw/generate",
            ["", 7.0, 5.0, 6.0]
        )
        
        # Wait for response
        time.sleep(1.0)
        
        # Cleanup
        receiver.shutdown()
        osc_server_instance.stop()
        
        # Verify error received
        assert len(received) > 0
        assert received[0][0] == "error"
    
    def test_server_stats(self, osc_server_instance):
        """Test server statistics tracking."""
        stats = osc_server_instance.get_stats()
        
        assert "requests_received" in stats
        assert "requests_processed" in stats
        assert "errors" in stats
        assert "uptime" in stats
        
        assert stats["requests_received"] == 0
        assert stats["requests_processed"] == 0
        assert stats["errors"] == 0

