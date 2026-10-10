import json
import os
import sys

os.environ["LOCAL_AGENT"] = "0"          # don't start the self-monitoring thread in tests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import app as app_module
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def send(client, name, cpu, ram, disk=40, net=5, **extra):
    body = {"hostname": name, "cpu": cpu, "ram": ram, "disk": disk, "net_mbps": net}
    body.update(extra)
    return client.post("/api/ingest", json=body)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_ingest_rejects_bad_hostname(client):
    assert send(client, "bad name<script>", 10, 10).status_code == 400


def test_ingest_rejects_out_of_range(client):
    assert send(client, "T-RANGE", 150, 10).status_code == 400


def test_ingest_rejects_missing_field(client):
    r = client.post("/api/ingest", json={"hostname": "T-MISSING", "cpu": 10})
    assert r.status_code == 400


def test_api_key_is_enforced(client, monkeypatch):
    monkeypatch.setattr(app_module, "AGENT_KEY", "secret")
    assert send(client, "T-KEY", 10, 10).status_code == 401
    r = client.post("/api/ingest", headers={"X-API-Key": "secret"},
                    json={"hostname": "T-KEY", "cpu": 10, "ram": 10, "disk": 10})
    assert r.status_code == 200


def test_machine_appears_with_forecast(client):
    for cpu in (20, 22, 21, 23, 22):
        send(client, "T-FLAT", cpu, 40)
    lab = client.get("/api/lab").get_json()
    m = next(x for x in lab["machines"] if x["name"] == "T-FLAT")
    assert m["online"] is True
    assert len(m["forecast"]) == 3
    assert m["status_now"] == "Normal"
    d = client.get("/api/machine/T-FLAT").get_json()
    assert len(d["history"]) == 5


def test_rising_load_is_predicted_before_it_happens(client):
    res = None
    for cpu, ram in [(30, 50), (40, 55), (50, 60), (60, 66), (68, 70), (74, 73)]:
        res = send(client, "T-RISING", cpu, ram).get_json()
    # nothing is Critical yet, but the forecast must raise the alarm
    assert res["status_now"] in ("Normal", "Warning")
    assert res["status_pred"] in ("Warning", "Critical")
    assert res["eta_warning_s"] is not None


def test_critical_machine_creates_alert_and_advice(client):
    res = send(client, "T-HOT", 97, 95, procs=[{"name": "matlab.exe", "cpu": 70, "mem": 50}]).get_json()
    send(client, "T-HOT", 98, 96)
    res = send(client, "T-HOT", 98, 96, procs=[{"name": "matlab.exe", "cpu": 70, "mem": 50}]).get_json()
    assert res["status_now"] == "Critical"
    assert "matlab.exe" in res["recommendation"]
    alerts = client.get("/api/lab").get_json()["alerts"]
    assert any(a["machine"] == "T-HOT" and a["level"] == "Critical" for a in alerts)


def test_unknown_machine_is_404(client):
    assert client.get("/api/machine/NOPE").status_code == 404


def test_stateless_predict(client):
    r = client.post("/api/predict", json={"cpu": 10, "ram": 30, "disk": 40})
    assert r.status_code == 200
    assert r.get_json()["status_pred"] == "Normal"
    bad = client.post("/api/predict", json={"cpu": 10})
    assert bad.status_code == 400


def test_prometheus_metrics(client):
    send(client, "T-PROM", 10, 10)
    body = client.get("/metrics").get_data(as_text=True)
    assert 'lab_cpu_percent{machine="T-PROM"}' in body


def test_offline_machine_leaves_prometheus(client):
    send(client, "T-GONE", 10, 10)
    assert 'machine="T-GONE"' in client.get("/metrics").get_data(as_text=True)
    app_module.machines["T-GONE"]["last_seen"] -= 3600      # pretend it went silent
    assert 'machine="T-GONE"' not in client.get("/metrics").get_data(as_text=True)


def test_model_quality_gate():
    """MLOps gate: the forecast must beat the naive 'no change' guess by 5%+."""
    with open("model_info.json") as f:
        info = json.load(f)
    assert info["improvement_pct"] >= 5, f"Model too weak: {info['improvement_pct']}%"
