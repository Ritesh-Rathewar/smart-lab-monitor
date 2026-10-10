"""Smart Lab Resource Monitoring & Prediction - server.

Lab PCs (agent.py) send readings to /api/ingest. For every machine the server
keeps the recent history, FORECASTS CPU/RAM with the ML model, works out when
it will become Warning/Critical, and suggests what to do.
"""
import hmac
import json
import logging
import math
import os
import re
import socket
import threading
import time
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from agent import Collector, clean_name
from features import CRIT, FEATURES, HORIZONS, WARN, make_features, status_of

LABELS = ["Normal", "Warning", "Critical"]
CODE = {name: i for i, name in enumerate(LABELS)}
SAMPLE_SECONDS = float(os.getenv("SAMPLE_SECONDS", "5"))
AGENT_KEY = os.getenv("AGENT_KEY", "")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
INFO_PATH = os.getenv("INFO_PATH", "model_info.json")
ONLINE_WINDOW = max(30.0, SAMPLE_SECONDS * 4)
HISTORY_LEN = 120
MIN_READINGS = 3
MAX_MACHINES = 200
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
LOCAL_NAME = os.getenv("LOCAL_NAME") or clean_name(socket.gethostname())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)
model = joblib.load(MODEL_PATH)
try:
    with open(INFO_PATH) as f:
        model_info = json.load(f)
except FileNotFoundError:
    model_info = {"version": "unknown"}

# ---------- Prometheus metrics (for Grafana later) ----------
_lbl = ["machine"]
g_cpu = Gauge("lab_cpu_percent", "CPU usage %", _lbl)
g_ram = Gauge("lab_ram_percent", "RAM usage %", _lbl)
g_disk = Gauge("lab_disk_percent", "Disk usage %", _lbl)
g_net = Gauge("lab_net_mbps", "Network usage Mbps", _lbl)
g_status = Gauge("lab_status", "Current status 0=Normal 1=Warning 2=Critical", _lbl)
g_pred_status = Gauge("lab_predicted_status", "Worst predicted status", _lbl)
g_pred_cpu = Gauge("lab_predicted_cpu_percent", "Forecast CPU % (middle horizon)", _lbl)
g_pred_ram = Gauge("lab_predicted_ram_percent", "Forecast RAM % (middle horizon)", _lbl)
g_online = Gauge("lab_machines_online", "Machines currently online")
g_model = Gauge("lab_model_info", "Model in use", ["version"])
g_model.labels(version=str(model_info.get("version"))).set(1)

machines = {}
alerts = deque(maxlen=50)
lock = threading.Lock()


# ---------- helpers ----------
def fmt_secs(s):
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    m, r = divmod(s, 60)
    return f"{m}m {r}s" if r else f"{m}m"


def forecast_from(cpu_list, ram_list, disk, net):
    """Core ML call: forecast CPU/RAM at each horizon."""
    f = make_features(cpu_list, ram_list, disk, net)
    X = pd.DataFrame([[f[k] for k in FEATURES]], columns=FEATURES)
    y = np.clip(model.predict(X)[0], 0, 100)
    n = len(HORIZONS)
    out = []
    for i, h in enumerate(HORIZONS):
        c, r = float(y[i]), float(y[n + i])
        out.append({"horizon": h, "seconds": h * SAMPLE_SECONDS,
                    "cpu": round(c, 1), "ram": round(r, 1),
                    "status": LABELS[status_of(c, r, disk, net)]})
    return out


def crossing_eta(now_val, points, threshold):
    """Seconds until a rising value crosses the threshold (None = not expected)."""
    if now_val > threshold:
        return 0.0
    prev_t, prev_v = 0.0, now_val
    for t, v in points:
        if v > threshold:
            return prev_t + (threshold - prev_v) / (v - prev_v) * (t - prev_t)
        prev_t, prev_v = t, v
    return None


def eta_for(last, fc, limits):
    """Earliest time any of CPU/RAM crosses its limit -> (seconds, resource)."""
    best = (None, None)
    for res in ("cpu", "ram"):
        pts = [(f["seconds"], f[res]) for f in fc]
        e = crossing_eta(last[res], pts, limits[res])
        if e is not None and (best[0] is None or e < best[0]):
            best = (e, res)
    return best


def recommend(last, level, res, procs):
    if level == 0:
        return "Healthy. No action needed."
    parts = []
    if last["disk"] > WARN["disk"]:
        parts.append(f"Disk is {last['disk']:.0f}% full: delete temp files or old logs.")
    if res in ("cpu", "ram"):
        key, label = ("cpu", "CPU") if res == "cpu" else ("mem", "RAM")
        top = sorted(procs, key=lambda p: p[key], reverse=True)[:1]
        if top and top[0][key] >= 5:
            parts.append(f"{label} is the bottleneck. Biggest consumer: "
                         f"{top[0]['name']} ({top[0][key]:.0f}%). Close it or move "
                         f"heavy jobs to a free machine.")
        else:
            parts.append(f"{label} is the bottleneck. Close unused programs or move "
                         f"heavy jobs to a free machine.")
    if last["net_mbps"] > WARN["net"]:
        parts.append("Network usage is very high: check for large downloads.")
    return " ".join(parts) or "Keep an eye on this machine."


def add_alert(name, level, message):
    alerts.appendleft({"time": datetime.now().strftime("%H:%M:%S"),
                       "machine": name, "level": level, "message": message})


def clean_procs(raw):
    out = []
    if not isinstance(raw, list):
        return out
    for p in raw[:10]:
        try:
            out.append({"name": str(p["name"])[:40],
                        "cpu": round(max(0.0, min(float(p["cpu"]), 100.0)), 1),
                        "mem": round(max(0.0, min(float(p["mem"]), 100.0)), 1)})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_reading(data):
    name = str(data.get("hostname", ""))
    if not NAME_RE.match(name):
        raise ValueError("hostname must be 1-40 letters, digits, - _ .")
    r = {"hostname": name}
    for k in ("cpu", "ram", "disk"):
        v = float(data[k])
        if not (math.isfinite(v) and 0 <= v <= 100):
            raise ValueError(f"{k} must be between 0 and 100")
        r[k] = v
    net = float(data.get("net_mbps", 0))
    if not (math.isfinite(net) and 0 <= net <= 100000):
        raise ValueError("net_mbps out of range")
    r["net_mbps"] = net
    r["procs"] = clean_procs(data.get("procs"))
    return r


def ingest_reading(r):
    """Store a reading, forecast, raise alerts. Returns the machine summary."""
    name = r["hostname"]
    with lock:
        if name not in machines and len(machines) >= MAX_MACHINES:
            return None
        m = machines.setdefault(name, {
            "history": deque(maxlen=HISTORY_LEN), "level": 0, "forecast": None,
            "eta_warn": (None, None), "eta_crit": (None, None),
            "recommendation": "Collecting data...", "procs": [], "last_seen": 0})
        m["history"].append({k: r[k] for k in ("cpu", "ram", "disk", "net_mbps")})
        m["last_seen"] = time.time()
        m["procs"] = r["procs"]
        hist = list(m["history"])
        last = hist[-1]
        code_now = status_of(last["cpu"], last["ram"], last["disk"], last["net_mbps"])

        if len(hist) >= MIN_READINGS:
            fc = forecast_from([h["cpu"] for h in hist], [h["ram"] for h in hist],
                               last["disk"], last["net_mbps"])
            m["forecast"] = fc
            m["eta_warn"] = eta_for(last, fc, WARN)
            m["eta_crit"] = eta_for(last, fc, CRIT)
            worst = max(CODE[f["status"]] for f in fc)
        else:
            fc, worst = None, code_now

        level = max(code_now, worst)
        res = m["eta_crit"][1] if level == 2 else m["eta_warn"][1]
        if res is None:
            res = "ram" if last["ram"] >= last["cpu"] else "cpu"
        m["recommendation"] = recommend(last, level, res, m["procs"]) if fc else "Collecting data..."

        if level > m["level"] and level >= 1:
            if code_now >= level:
                msg = (f"{name} is {LABELS[level]} right now "
                       f"(CPU {last['cpu']:.0f}%, RAM {last['ram']:.0f}%, Disk {last['disk']:.0f}%).")
            else:
                eta = m["eta_crit"][0] if level == 2 else m["eta_warn"][0]
                msg = (f"{name}: {res.upper()} predicted to reach {LABELS[level]}"
                       + (f" in ~{fmt_secs(eta)}." if eta is not None else "."))
            add_alert(name, LABELS[level], msg)
            if level == 2:
                logging.warning("ALERT %s", msg)
        elif level == 0 and m["level"] >= 1:
            add_alert(name, "Normal", f"{name} is back to normal.")
        m["level"] = level

        g_cpu.labels(machine=name).set(last["cpu"])
        g_ram.labels(machine=name).set(last["ram"])
        g_disk.labels(machine=name).set(last["disk"])
        g_net.labels(machine=name).set(last["net_mbps"])
        g_status.labels(machine=name).set(code_now)
        g_pred_status.labels(machine=name).set(worst)
        if fc:
            mid = fc[len(fc) // 2]
            g_pred_cpu.labels(machine=name).set(mid["cpu"])
            g_pred_ram.labels(machine=name).set(mid["ram"])
        return public(name, m, time.time())


def public(name, m, now):
    last = m["history"][-1]
    online = now - m["last_seen"] <= ONLINE_WINDOW
    fc = m["forecast"]
    status_now = LABELS[status_of(last["cpu"], last["ram"], last["disk"], last["net_mbps"])]
    status_pred = "Learning" if fc is None else LABELS[max(CODE[f["status"]] for f in fc)]
    return {
        "name": name, "is_local": name == LOCAL_NAME, "online": online,
        "seconds_since": round(now - m["last_seen"], 1),
        "readings": len(m["history"]),
        "now": {k: round(last[k], 1) for k in ("cpu", "ram", "disk", "net_mbps")},
        "status_now": status_now, "status_pred": status_pred, "forecast": fc,
        "eta_warning_s": m["eta_warn"][0], "eta_warning_res": m["eta_warn"][1],
        "eta_critical_s": m["eta_crit"][0], "eta_critical_res": m["eta_crit"][1],
        "recommendation": m["recommendation"], "procs": m["procs"],
    }


# ---------- this server also monitors itself ----------
def _local_loop():
    col = Collector()
    while True:
        try:
            r = col.read()
            r["hostname"] = LOCAL_NAME
            ingest_reading(parse_reading(r))
        except Exception:
            logging.exception("local agent error")
        time.sleep(SAMPLE_SECONDS)


if os.getenv("LOCAL_AGENT", "1") == "1":
    threading.Thread(target=_local_loop, daemon=True).start()


# ---------- routes ----------
@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/api/model-info")
def info():
    return jsonify(model_info)


@app.route("/api/ingest", methods=["POST"])
def ingest():
    if AGENT_KEY and not hmac.compare_digest(request.headers.get("X-API-Key", ""), AGENT_KEY):
        return jsonify(error="unauthorized"), 401
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="send a JSON object"), 400
    try:
        reading = parse_reading(data)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"bad reading: {e}"), 400
    res = ingest_reading(reading)
    if res is None:
        return jsonify(error="too many machines"), 429
    return jsonify(res)


@app.route("/api/lab")
def lab():
    now = time.time()
    with lock:
        items = [public(n, m, now) for n, m in sorted(machines.items())]
    online = [x for x in items if x["online"]]
    g_online.set(len(online))
    summary = {
        "total": len(items), "online": len(online), "offline": len(items) - len(online),
        "normal": sum(1 for x in online if x["status_pred"] in ("Normal", "Learning")
                      and x["status_now"] == "Normal"),
        "warning": sum(1 for x in online if "Warning" in (x["status_now"], x["status_pred"])
                       and "Critical" not in (x["status_now"], x["status_pred"])),
        "critical": sum(1 for x in online if "Critical" in (x["status_now"], x["status_pred"])),
    }
    return jsonify(machines=items, summary=summary, alerts=list(alerts),
                   model=model_info, sample_seconds=SAMPLE_SECONDS)


@app.route("/api/machine/<name>")
def machine(name):
    with lock:
        m = machines.get(name)
        if m is None:
            return jsonify(error="unknown machine"), 404
        out = public(name, m, time.time())
        out["history"] = list(m["history"])[-60:]
    return jsonify(out)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Stateless forecast: send values (and optional recent history) -> forecast."""
    d = request.get_json(silent=True) or {}
    try:
        cpu, ram = float(d["cpu"]), float(d["ram"])
        disk, net = float(d["disk"]), float(d.get("net_mbps", 0))
        ch = [float(x) for x in d.get("cpu_history", [])] + [cpu]
        rh = [float(x) for x in d.get("ram_history", [])] + [ram]
        if len(ch) != len(rh) or not all(0 <= v <= 100 for v in ch + rh + [disk]):
            raise ValueError("histories must have equal length, values 0-100")
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"bad input: {e}"), 400
    fc = forecast_from(ch, rh, disk, net)
    worst = max(CODE[f["status"]] for f in fc)
    return jsonify(forecast=fc, status_pred=LABELS[worst])


_PER_MACHINE = [g_cpu, g_ram, g_disk, g_net, g_status, g_pred_status, g_pred_cpu, g_pred_ram]


def prune_offline_metrics():
    """Machines that stopped reporting must disappear from Prometheus,
    otherwise their last value (maybe 'Critical') would stay forever."""
    now = time.time()
    online = 0
    with lock:
        for name, m in machines.items():
            if now - m["last_seen"] <= ONLINE_WINDOW:
                online += 1
                continue
            for g in _PER_MACHINE:
                try:
                    g.remove(name)
                except KeyError:
                    pass
    g_online.set(online)


@app.route("/metrics")
def metrics():
    prune_offline_metrics()
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
