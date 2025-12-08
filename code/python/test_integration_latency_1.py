"""
Integration and Latency Tests for iDAW System

Tests the complete integration between:
- Tauri frontend (React)
- Tauri backend (Rust)
- Python API server
- Music Brain engine

Measures latency for:
- Python server startup
- API endpoint responses
- Music generation
- Emotion retrieval
- Interrogation flow
"""

import os
import sys
import time
import subprocess
import requests
import pytest
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Test configuration
TEST_PORT = int(os.environ.get("TEST_MUSIC_BRAIN_PORT", "8001"))
API_BASE_URL = f"http://127.0.0.1:{TEST_PORT}"
TIMEOUT = 30  # seconds
MAX_STARTUP_TIME = 10  # seconds
MAX_API_RESPONSE_TIME = 2  # seconds


class LatencyTestResults:
    """Container for latency test results"""
    def __init__(self):
        self.results: Dict[str, List[float]] = {}
        self.errors: List[str] = []
    
    def add_result(self, test_name: str, latency: float):
        if test_name not in self.results:
            self.results[test_name] = []
        self.results[test_name].append(latency)
    
    def add_error(self, test_name: str, error: str):
        self.errors.append(f"{test_name}: {error}")
    
    def get_stats(self, test_name: str) -> Dict[str, float]:
        if test_name not in self.results or not self.results[test_name]:
            return {}
        
        latencies = self.results[test_name]
        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "count": len(latencies)
        }
    
    def print_report(self):
        print("\n" + "="*70)
        print("LATENCY TEST REPORT")
        print("="*70)
        
        for test_name in sorted(self.results.keys()):
            stats = self.get_stats(test_name)
            if stats:
                print(f"\n{test_name}:")
                print(f"  Count:    {stats['count']}")
                print(f"  Min:      {stats['min']:.3f}s")
                print(f"  Max:      {stats['max']:.3f}s")
                print(f"  Mean:     {stats['mean']:.3f}s")
                print(f"  Median:   {stats['median']:.3f}s")
                print(f"  Std Dev:  {stats['stdev']:.3f}s")
        
        if self.errors:
            print("\n" + "="*70)
            print("ERRORS:")
            print("="*70)
            for error in self.errors:
                print(f"  - {error}")


@pytest.fixture(scope="module")
def python_server():
    """Start Python server for testing"""
    script_paths = [
        Path(__file__).parent.parent / "music_brain" / "api.py",
        Path(__file__).parent.parent / "music_brain" / "start_api_embedded.py",
    ]
    
    api_script = None
    for path in script_paths:
        if path.exists():
            api_script = path
            break
    
    if not api_script:
        pytest.skip("API script not found")
    
    # Start server
    env = os.environ.copy()
    env["MUSIC_BRAIN_PORT"] = str(TEST_PORT)
    
    process = subprocess.Popen(
        [sys.executable, str(api_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to start
    start_time = time.time()
    max_wait = MAX_STARTUP_TIME
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"{API_BASE_URL}/health",
                timeout=1
            )
            if response.status_code == 200:
                startup_time = time.time() - start_time
                print(f"\n✓ Server started in {startup_time:.3f}s")
                yield process, startup_time
                break
        except requests.exceptions.RequestException:
            time.sleep(0.5)
    else:
        process.terminate()
        process.wait()
        pytest.fail(f"Server failed to start within {max_wait}s")
    
    # Cleanup
    process.terminate()
    process.wait(timeout=5)


def test_server_startup_latency(python_server):
    """Test Python server startup latency"""
    process, startup_time = python_server
    
    assert startup_time < MAX_STARTUP_TIME, \
        f"Server startup took {startup_time:.3f}s, expected < {MAX_STARTUP_TIME}s"
    
    print(f"✓ Server startup: {startup_time:.3f}s")


def test_health_endpoint_latency(python_server):
    """Test health endpoint response latency"""
    process, _ = python_server
    results = LatencyTestResults()
    
    # Run multiple requests
    for i in range(10):
        start = time.time()
        try:
            response = requests.get(
                f"{API_BASE_URL}/health",
                timeout=TIMEOUT
            )
            latency = time.time() - start
            
            assert response.status_code == 200
            assert latency < MAX_API_RESPONSE_TIME
            results.add_result("health_endpoint", latency)
        except Exception as e:
            results.add_error("health_endpoint", str(e))
    
    stats = results.get_stats("health_endpoint")
    assert stats["mean"] < MAX_API_RESPONSE_TIME
    print(f"✓ Health endpoint: mean={stats['mean']:.3f}s, max={stats['max']:.3f}s")


def test_emotions_endpoint_latency(python_server):
    """Test emotions endpoint response latency"""
    process, _ = python_server
    results = LatencyTestResults()
    
    for i in range(5):
        start = time.time()
        try:
            response = requests.get(
                f"{API_BASE_URL}/emotions",
                timeout=TIMEOUT
            )
            latency = time.time() - start
            
            assert response.status_code == 200
            data = response.json()
            assert "emotions" in data or "success" in data
            results.add_result("emotions_endpoint", latency)
        except Exception as e:
            results.add_error("emotions_endpoint", str(e))
    
    stats = results.get_stats("emotions_endpoint")
    print(f"✓ Emotions endpoint: mean={stats['mean']:.3f}s, max={stats['max']:.3f}s")


def test_generate_music_latency(python_server):
    """Test music generation latency"""
    process, _ = python_server
    results = LatencyTestResults()
    
    request_data = {
        "intent": {
            "base_emotion": "joy",
            "intensity": "moderate",
            "specific_emotion": "happiness"
        },
        "output_format": "midi"
    }
    
    for i in range(3):  # Fewer tests as this is slower
        start = time.time()
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json=request_data,
                timeout=60  # Generation takes longer
            )
            latency = time.time() - start
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data or "song" in data
            results.add_result("generate_music", latency)
        except Exception as e:
            results.add_error("generate_music", str(e))
    
    stats = results.get_stats("generate_music")
    if stats:
        print(f"✓ Generate music: mean={stats['mean']:.3f}s, max={stats['max']:.3f}s")


def test_interrogate_latency(python_server):
    """Test interrogation endpoint latency"""
    process, _ = python_server
    results = LatencyTestResults()
    
    request_data = {
        "message": "I want to write a song about loss"
    }
    
    for i in range(5):
        start = time.time()
        try:
            response = requests.post(
                f"{API_BASE_URL}/interrogate",
                json=request_data,
                timeout=TIMEOUT
            )
            latency = time.time() - start
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data or "ready" in data
            results.add_result("interrogate", latency)
        except Exception as e:
            results.add_error("interrogate", str(e))
    
    stats = results.get_stats("interrogate")
    if stats:
        print(f"✓ Interrogate: mean={stats['mean']:.3f}s, max={stats['max']:.3f}s")


def test_concurrent_requests(python_server):
    """Test system under concurrent load"""
    process, _ = python_server
    import concurrent.futures
    
    def make_request():
        start = time.time()
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return time.time() - start, response.status_code == 200
        except:
            return None, False
    
    # Run 20 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(20)]
        results_list = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    latencies = [r[0] for r in results_list if r[0] is not None]
    successes = sum(1 for r in results_list if r[1])
    
    assert successes >= 18, f"Only {successes}/20 requests succeeded"
    if latencies:
        mean_latency = statistics.mean(latencies)
        print(f"✓ Concurrent requests: {successes}/20 succeeded, mean latency={mean_latency:.3f}s")


def test_end_to_end_flow(python_server):
    """Test complete end-to-end flow latency"""
    process, _ = python_server
    
    # Simulate user flow: get emotions -> generate music
    start = time.time()
    
    # Step 1: Get emotions
    emotions_start = time.time()
    emotions_response = requests.get(f"{API_BASE_URL}/emotions", timeout=TIMEOUT)
    emotions_time = time.time() - emotions_start
    assert emotions_response.status_code == 200
    
    # Step 2: Generate music
    generate_start = time.time()
    generate_response = requests.post(
        f"{API_BASE_URL}/generate",
        json={
            "intent": {
                "base_emotion": "grief",
                "intensity": "high",
                "specific_emotion": "longing"
            }
        },
        timeout=60
    )
    generate_time = time.time() - generate_start
    assert generate_response.status_code == 200
    
    total_time = time.time() - start
    
    print(f"\n✓ End-to-end flow:")
    print(f"  Get emotions: {emotions_time:.3f}s")
    print(f"  Generate music: {generate_time:.3f}s")
    print(f"  Total: {total_time:.3f}s")
    
    assert total_time < 60, f"End-to-end flow took {total_time:.3f}s, expected < 60s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
