"""Pretend to be a whole lab of PCs (for demos and screenshots).

Terminal 1:  python app.py
Terminal 2:  python simulate_lab.py
"""
import argparse
import json
import time
import urllib.request

import numpy as np

from simulator import MachineSim


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
    ap.add_argument("--key", default="")
    ap.add_argument("--machines", type=int, default=6)
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument("--stress", type=float, default=0.03,
                    help="chance per reading that a heavy load starts")
    a = ap.parse_args()
    rng = np.random.default_rng()
    sims = {f"LAB-PC-{i:02d}": MachineSim(rng, a.stress) for i in range(1, a.machines + 1)}
    print(f"Simulating {len(sims)} lab PCs -> {a.server} (Ctrl+C to stop)")
    while True:
        for name, sim in sims.items():
            cpu, ram, disk, net = sim.step()
            payload = {"hostname": name, "cpu": round(cpu, 1), "ram": round(ram, 1),
                       "disk": round(disk, 1), "net_mbps": round(net, 1),
                       "procs": sim.top_processes()}
            try:
                res = post(a.server, a.key, payload)
                print(f"{name}: cpu {cpu:5.1f} ram {ram:5.1f} -> {res.get('status_pred')}")
            except Exception as e:
                print("could not send:", e)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
