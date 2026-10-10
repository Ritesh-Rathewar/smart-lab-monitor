import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_normal(client):
    r = client.post("/api/predict", json={"cpu": 10, "ram": 30, "disk": 40, "net_mbps": 5})
    assert r.get_json()["status"] == "Normal"


def test_predict_critical(client):
    r = client.post("/api/predict", json={"cpu": 98, "ram": 95, "disk": 60, "net_mbps": 5})
    assert r.get_json()["status"] == "Critical"


def test_predict_missing_field(client):
    r = client.post("/api/predict", json={"cpu": 10})
    assert r.status_code == 400


def test_live_endpoint(client):
    r = client.get("/api/live")
    assert r.status_code == 200
    assert r.get_json()["status"] in ("Normal", "Warning", "Critical")
