"""Identità e capacità della macchina, più il campionatore di RAM e potenza."""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import threading
import time
from pathlib import Path


def cpu_info() -> dict:
    model, flags, cores, threads, phys = None, set(), set(), 0, "0"
    try:
        for line in open("/proc/cpuinfo"):
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k == "processor":
                threads += 1
            elif k == "model name" and not model:
                model = v
            elif k == "flags" and not flags:
                flags = set(v.split())
            elif k == "physical id":
                phys = v
            elif k == "core id":
                cores.add((phys, v))
    except OSError:
        pass
    return {
        "cpu_model": model or platform.processor(),
        "physical_cores": len(cores) or (os.cpu_count() or 1),
        "logical_threads": threads or (os.cpu_count() or 1),
        "avx": "avx" in flags,
        "avx2": "avx2" in flags,
        "fma": "fma" in flags,
        "avx512": any(f.startswith("avx512") for f in flags),
        "flags": sorted(f for f in flags if f.startswith(("avx", "fma", "f16c", "sse4", "ssse3"))),
    }


def meminfo_gb() -> dict:
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            out[k] = int(v.strip().split()[0]) / 1024 / 1024
    except OSError:
        pass
    return {"total_gb": round(out.get("MemTotal", 0), 2), "available_gb": round(out.get("MemAvailable", 0), 2)}


def detect_machine(machines_cfg: dict, forced: str | None = None) -> tuple[str, dict]:
    host = socket.gethostname().lower()
    info = {"hostname": host, **cpu_info(), "ram_gb": meminfo_gb()["total_gb"], "kernel": platform.release()}
    if forced:
        if forced not in machines_cfg:
            raise SystemExit(f"macchina '{forced}' non definita in matrix.yaml")
        return forced, {**info, **machines_cfg[forced]}
    for mid, m in machines_cfg.items():
        if m.get("hostname", "").lower() == host:
            return mid, {**info, **m}
    raise SystemExit(f"hostname '{host}' non corrisponde a nessuna macchina in matrix.yaml: usa --machine")


def _children(pid: int) -> list[int]:
    out = []
    try:
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
                except (OSError, IndexError, ValueError):
                    continue
                if ppid == pid:
                    out.append(int(d.name))
    except OSError:
        pass
    return out


def tree_rss_mb(pid: int) -> float:
    total, stack, seen = 0.0, [pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            for line in open(f"/proc/{p}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) / 1024
                    break
        except OSError:
            continue
        stack.extend(_children(p))
    return total


class Monitor:
    """Campiona RSS del processo (albero incluso) e, se configurato, i watt da un comando esterno.

    power_cmd: comando che stampa la potenza istantanea in watt (es. curl alla presa smart).
    """

    def __init__(self, pid_fn, power_cmd: str | None = None, interval: float = 1.0):
        self.pid_fn, self.power_cmd, self.interval = pid_fn, power_cmd, interval
        self.peak_rss = 0.0
        self.energy_j = 0.0
        self.watts: list[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self.t0 = self.t1 = None

    def _power(self) -> float | None:
        try:
            out = subprocess.run(self.power_cmd, shell=True, capture_output=True, text=True, timeout=5).stdout
            return float(out.strip().split()[0])
        except Exception:
            return None

    def _loop(self):
        last = time.monotonic()
        while not self._stop.is_set():
            pid = self.pid_fn()
            if pid:
                self.peak_rss = max(self.peak_rss, tree_rss_mb(pid))
            if self.power_cmd:
                w = self._power()
                t = time.monotonic()
                if w is not None:
                    self.watts.append(w)
                    self.energy_j += w * (t - last)
                last = t
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t0 = time.monotonic()
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=10)
        self.t1 = time.monotonic()

    def snapshot(self) -> dict:
        return {
            "peak_rss_mb": round(self.peak_rss, 1) or None,
            "energy_j": round(self.energy_j, 1) if self.watts else None,
            "avg_power_w": round(sum(self.watts) / len(self.watts), 2) if self.watts else None,
        }
