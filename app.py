"""Smart Lab Resource Monitoring & Prediction - Flask app."""
import logging
import os

import joblib
import pandas as pd
import psutil
from flask import Flask, jsonify, render_template, request
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

FEATURES = ["cpu", "ram", "disk", "net_mbps"]
LABELS = {0: "Normal", 1: "Warning", 2: "Critical"}
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# Prometheus gauges (Grafana will read these later)
g_cpu = Gauge("lab_cpu_percent", "CPU usage %")
g_ram = Gauge("lab_ram_percent", "RAM usage %")
g_disk = Gauge("lab_disk_percent", "Disk usage %")
g_net = Gauge("lab_net_mbps", "Network usage Mbps")
g_status = Gauge("lab_status", "0=Normal 1=Warning 2=Critical")

_last_net = {"bytes": None, "time": None}


def collect_metrics():
    """Read real metrics of the machine this app runs on."""
    import time
    c = psutil.net_io_counters()
    now, total = time.time(), c.bytes_sent + c.bytes_recv
    mbps = 0.0
    if _last_net["bytes"] is not None:
        dt = max(now - _last_net["time"], 1e-6)
        mbps = (total - _last_net["bytes"]) * 8 / dt / 1_000_000
    _last_net.update(bytes=total, time=now)
    return {
        "cpu": psutil.cpu_percent(interval=0.5),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "net_mbps": round(mbps, 2),
    }


def predict_status(m):
    row = pd.DataFrame([[m[f] for f in FEATURES]], columns=FEATURES)
    code = int(model.predict(row)[0])
    return code, LABELS[code]


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/api/live")
def live():
    m = collect_metrics()
    code, label = predict_status(m)
    g_cpu.set(m["cpu"]); g_ram.set(m["ram"]); g_disk.set(m["disk"])
    g_net.set(m["net_mbps"]); g_status.set(code)
    if code == 2:
        logging.warning("ALERT: machine is CRITICAL %s", m)  # alert hook
    return jsonify(metrics=m, status=label, code=code)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify(error=f"missing fields: {missing}"), 400
    code, label = predict_status(data)
    return jsonify(status=label, code=code)


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
