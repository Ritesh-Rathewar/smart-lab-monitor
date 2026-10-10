"""Simulates a lab PC: normal use plus random 'load events'
(a heavy program starts, a memory leak, many students at once...)."""
import numpy as np

CULPRITS = ["chrome.exe", "python.exe", "matlab.exe", "jupyter-lab",
            "vscode", "docker.exe", "teams.exe", "android-studio"]
LIGHT_APPS = ["explorer.exe", "svchost.exe", "notepad.exe", "firefox.exe"]


class MachineSim:
    def __init__(self, rng, event_prob=0.04):
        self.rng = rng
        self.event_prob = event_prob
        self.cpu = rng.uniform(5, 50)
        self.ram = rng.uniform(20, 60)
        self.disk = rng.uniform(30, 85)
        self.net = rng.uniform(0, 30)
        self.left = 0
        self.d_cpu = 0.0
        self.d_ram = 0.0
        self.culprit = None

    def start_event(self, kind=None):
        rng = self.rng
        kind = kind or str(rng.choice(["cpu", "ram", "both"]))
        self.left = int(rng.integers(12, 45))
        self.d_cpu = float(rng.uniform(1.5, 5)) if kind in ("cpu", "both") else 0.0
        self.d_ram = float(rng.uniform(0.6, 2.2)) if kind in ("ram", "both") else 0.0
        self.culprit = str(rng.choice(CULPRITS))

    def step(self):
        rng = self.rng
        if self.left == 0 and rng.random() < self.event_prob:
            self.start_event()
        if self.left > 0:
            self.left -= 1
            if self.d_cpu:
                self.cpu += self.d_cpu + rng.normal(0, 2)
            else:
                self.cpu += (35 - self.cpu) * 0.15 + rng.normal(0, 2)
            if self.d_ram:
                self.ram += self.d_ram + rng.normal(0, 0.8)
            else:
                self.ram += (45 - self.ram) * 0.08 + rng.normal(0, 1)
            if self.left == 0:
                self.culprit = None
        else:
            self.cpu += (35 - self.cpu) * 0.15 + rng.normal(0, 3)
            self.ram += (45 - self.ram) * 0.08 + rng.normal(0, 1.2)
        self.disk += rng.normal(0.02, 0.05)
        self.net = abs(self.net + rng.normal(0, 8))
        self.cpu = float(np.clip(self.cpu, 0, 100))
        self.ram = float(np.clip(self.ram, 0, 100))
        self.disk = float(np.clip(self.disk, 0, 100))
        self.net = float(np.clip(self.net, 0, 100))
        return self.cpu, self.ram, self.disk, self.net

    def top_processes(self):
        """Fake process list (for the demo simulator)."""
        procs = [{"name": str(self.rng.choice(LIGHT_APPS)),
                  "cpu": round(float(self.rng.uniform(0, 4)), 1),
                  "mem": round(float(self.rng.uniform(1, 5)), 1)}]
        if self.culprit:
            procs.insert(0, {"name": self.culprit,
                             "cpu": round(self.cpu * 0.7, 1),
                             "mem": round(self.ram * 0.55, 1)})
        return procs
