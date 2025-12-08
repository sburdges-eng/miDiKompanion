"""
Standalone API Latency Tests

Tests API latency when server is already running (manual start required).
"""

import time
import requests
import pytest
import statistics

API_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 5


@pytest.mark.skipif(
    not pytest.config.getoption("--run-api-tests", default=False),
    reason="Requires --run-api-tests flag and running server"
)
def test_health_endpoint_latency():
    """Test health endpoint latency"""
    latencies = []
    
    for i in range(10):
        start = time.time()
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
            latency = time.time() - start
            assert response.status_code == 200
            latencies.append(latency)
        except requests.exceptions.RequestException:
            pytest.skip("Server not running")
    
    mean_latency = statistics.mean(latencies)
    max_latency = max(latencies)
    
    print(f"\n✓ Health endpoint: mean={mean_latency:.3f}s, max={max_latency:.3f}s")
    assert mean_latency < 0.5, f"Mean latency {mean_latency:.3f}s too high"


@pytest.mark.skipif(
    not pytest.config.getoption("--run-api-tests", default=False),
    reason="Requires --run-api-tests flag and running server"
)
def test_emotions_endpoint_latency():
    """Test emotions endpoint latency"""
    start = time.time()
    try:
        response = requests.get(f"{API_BASE_URL}/emotions", timeout=TIMEOUT)
        latency = time.time() - start
        assert response.status_code == 200
        print(f"\n✓ Emotions endpoint: {latency:.3f}s")
        assert latency < 2.0, f"Latency {latency:.3f}s too high"
    except requests.exceptions.RequestException:
        pytest.skip("Server not running")
