"""Utility comuni: hash, json canonico, esecuzione processi, classificazione errori."""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def canon(obj) -> str:
    """JSON canonico: stesso contenuto -> stessa stringa -> stesso hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def h(obj, n: int = 16) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()[:n]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            d.update(b)
    return d.hexdigest()


def sha256_files(paths) -> str:
    d = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        d.update(p.name.encode())
        d.update(sha256_file(p).encode())
    return d.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class LabError(Exception):
    """Errore con una classe, così si può filtrare e ripetere per tipo."""

    def __init__(self, cls: str, msg: str, log: str | None = None):
        super().__init__(msg)
        self.cls = cls
        self.log = log


_PATTERNS = [
    ("illegal_instruction", r"Illegal instruction|SIGILL|illegal hardware instruction"),
    ("oom", r"Cannot allocate memory|out of memory|failed to allocate|std::bad_alloc|OOMKilled|Killed"),
    ("model_load", r"failed to load model|error loading model|unknown model architecture|invalid magic"),
    ("unsupported", r"not supported|unsupported|unknown argument|invalid argument|unrecognized arguments"),
    ("download", r"HTTP Error 40[134]|Repository Not Found|404 Client Error"),
    ("connection", r"Connection refused|Connection reset|RemoteDisconnected|timed out"),
]


def classify(text: str, rc: int | None = None) -> str:
    if rc in (132, -4):
        return "illegal_instruction"
    if rc in (137, -9):
        return "oom"
    for cls, pat in _PATTERNS:
        if re.search(pat, text or "", re.I):
            return cls
    return "error"


def tail(path: Path, n: int = 4000) -> str:
    try:
        data = Path(path).read_bytes()
        return data[-n:].decode("utf-8", "replace")
    except OSError:
        return ""


def run_cmd(cmd, log: Path, timeout: float | None = None, env=None, cwd=None, check=True,
            stdout_only=False) -> str:
    """Esegue un comando salvando tutto l'output nel log. Ritorna stdout+stderr,
    oppure solo stdout con stdout_only (stderr finisce comunque nel log)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **(env or {})}
    with open(log, "ab") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n".encode())
        f.flush()
        start = f.tell()
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE if stdout_only else f,
                             stderr=f if stdout_only else subprocess.STDOUT, env=full_env,
                             cwd=cwd, start_new_session=True)
        try:
            so, _ = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            kill_group(p)
            raise LabError("timeout", f"timeout dopo {timeout}s: {cmd[0]}", str(log))
        except BaseException:
            kill_group(p)
            raise
        if stdout_only and so:
            f.write(so)
    out = Path(log).read_bytes()[start:].decode("utf-8", "replace")
    if stdout_only:
        so_txt = (so or b"").decode("utf-8", "replace")
        if check and rc != 0:
            raise LabError(classify(out, rc), f"exit {rc}: {' '.join(map(str, cmd[:3]))}...\n{out[-1500:]}", str(log))
        return so_txt
    if check and rc != 0:
        raise LabError(classify(out, rc), f"exit {rc}: {' '.join(map(str, cmd[:3]))}...\n{out[-1500:]}", str(log))
    return out


def start_bg(cmd, log: Path, env=None, cwd=None) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "ab")
    f.write(f"\n$ {' '.join(map(str, cmd))}\n".encode())
    f.flush()
    return subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, **(env or {})}, cwd=cwd, start_new_session=True)


def kill_group(p: subprocess.Popen, grace: float = 10):
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=grace)
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
