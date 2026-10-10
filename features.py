"""Shared by training AND the app, so both compute features the same way."""
import numpy as np

HORIZONS = [6, 12, 24]            # forecast this many readings ahead
FEATURES = ["cpu", "ram", "disk", "net_mbps",
            "cpu_d3", "cpu_d6", "ram_d3", "ram_d6", "cpu_std6", "ram_std6"]
TARGETS = [f"{r}_h{h}" for r in ("cpu", "ram") for h in HORIZONS]

WARN = {"cpu": 70, "ram": 75, "disk": 85, "net": 85}
CRIT = {"cpu": 90, "ram": 90, "disk": 95}


def status_of(cpu, ram, disk, net=0.0):
    """0 = Normal, 1 = Warning, 2 = Critical."""
    if cpu > CRIT["cpu"] or ram > CRIT["ram"] or disk > CRIT["disk"]:
        return 2
    if (cpu > WARN["cpu"] or ram > WARN["ram"]
            or disk > WARN["disk"] or net > WARN["net"]):
        return 1
    return 0


def _delta(values, k):
    return values[-1] - values[max(0, len(values) - 1 - k)]


def make_features(cpu_hist, ram_hist, disk, net):
    """History lists are oldest -> newest. Uses the last 7 readings."""
    c, r = list(cpu_hist)[-7:], list(ram_hist)[-7:]
    return {
        "cpu": c[-1], "ram": r[-1], "disk": disk, "net_mbps": net,
        "cpu_d3": _delta(c, 3), "cpu_d6": _delta(c, 6),
        "ram_d3": _delta(r, 3), "ram_d6": _delta(r, 6),
        "cpu_std6": float(np.std(c)), "ram_std6": float(np.std(r)),
    }
