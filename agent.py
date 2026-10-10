"""Lab PC agent: reads this computer's usage and sends it to the monitor server.

On every lab PC you want to watch:
    pip install psutil
    python agent.py --server http://SERVER-ADDRESS:5000
"""
import argparse
import json
import re
import socket
import time
import urllib.request

import psutil


class Collector:
    """Reads CPU, RAM, disk, network and the top processes of THIS computer."""

    def __init__(self):
        self.last_bytes = None
        self.last_time = None
        self.ncpu = psutil.cpu_count() or 1
        psutil.cpu_percent(None)
        for p in psutil.process_iter():          # prime per-process CPU counters
            try:
                p.cpu_percent(None)
            except (psutil.Error, OSError):
                pass

    def _top_processes(self):
        procs = []
        for p in psutil.process_iter(["name", "cpu_percent", "memory_percent"]):
            info = p.info
            name = info.get("name") or "?"
            if name in ("System Idle Process", "idle"):
                continue
            procs.append({"name": name[:40],
                          "cpu": round((info.get("cpu_percent") or 0) / self.ncpu, 1),
                          "mem": round(info.get("memory_percent") or 0, 1)})
        by_cpu = sorted(procs, key=lambda x: x["cpu"], reverse=True)[:5]
        by_mem = sorted(procs, key=lambda x: x["mem"], reverse=True)[:5]
        seen, out = set(), []
        for p in by_cpu + by_mem:
            if p["name"] not in seen:
                seen.add(p["name"])
                out.append(p)
        return out

    def read(self):
        c = psutil.net_io_counters()
        now, total = time.time(), c.bytes_sent + c.bytes_recv
        mbps = 0.0
        if self.last_bytes is not None:
            dt = max(now - self.last_time, 1e-6)
            mbps = (total - self.last_bytes) * 8 / dt / 1_000_000
        self.last_bytes, self.last_time = total, now
        return {
            "cpu": psutil.cpu_percent(interval=0.5),
            "ram": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
            "net_mbps": round(min(mbps, 100000.0), 2),
            "procs": self._top_processes(),
        }


def clean_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:40] or "lab-pc"


def post(server, key, payload):
    req = urllib.request.Request(
        server.rstrip("/") + "/api/ingest",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:5000")
    ap.add_argument("--key", default="", help="agent key, if the server needs one")
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument("--name", default=clean_name(socket.gethostname()))
    a = ap.parse_args()
    col = Collector()
    print(f"Agent '{a.name}' sending to {a.server} every {a.interval}s (Ctrl+C to stop)")
    while True:
        try:
            r = col.read()
            r["hostname"] = a.name
            res = post(a.server, a.key, r)
            print(f"cpu {r['cpu']:5.1f}%  ram {r['ram']:5.1f}%  -> {res.get('status_pred')}")
        except Exception as e:  # keep running if the server is briefly down
            print("could not send:", e)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
