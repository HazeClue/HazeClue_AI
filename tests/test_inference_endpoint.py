import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure the parent directory is in the sys.path so we can import api
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.inference_endpoint import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "model_mves" in response.json()
    assert "model_rard" in response.json()

def test_inference_success():
    # Create a dummy 14x512 matrix
    dummy_data = [[0.0 for _ in range(512)] for _ in range(14)]
    
    response = client.post(
        "/api/inference",
        json={"data": dummy_data, "mode": "SAFE"}
    )
    
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "probability" in json_response
    assert "smoothed_output" in json_response
    assert "mode" in json_response
    assert json_response["mode"] == "SAFE"
    assert "sqi_mean" in json_response
    assert "accepted" in json_response

def test_inference_invalid_shape():
    # Create an invalid shape (14x100 instead of 14x512)
    invalid_data = [[0.0 for _ in range(100)] for _ in range(14)]
    
    response = client.post(
        "/api/inference",
        json={"data": invalid_data}
    )
    
    assert response.status_code == 400
    assert "Expected shape (14, 512)" in response.json()["detail"]

def test_inference_invalid_channels():
    # Create an invalid shape (10x512 instead of 14x512)
    invalid_data = [[0.0 for _ in range(512)] for _ in range(10)]
    
    response = client.post(
        "/api/inference",
        json={"data": invalid_data}
    )
    
    assert response.status_code == 400
    assert "Expected shape (14, 512)" in response.json()["detail"]
