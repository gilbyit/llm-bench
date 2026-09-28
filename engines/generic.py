"""Motore descritto interamente in matrix.yaml: un comando (binario, script o container) che espone
un'API compatibile OpenAI. Copre KoboldCpp, llamafile, OpenVINO Model Server, i server Python per
Transformers e ONNX Runtime GenAI, e il motore finto usato per i collaudi.

Aggiungere un motore di questo tipo non richiede codice: basta una voce in `engines:`.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

from ..util import LabError, free_port, run_cmd, sha256_file, start_bg
from .base import Engine, Server

LAB_DIR = Path(__file__).resolve().parent.parent


class Generic(Engine):
    kind = "generic"

    def __init__(self, name, cfg, ctx):
        super().__init__(name, cfg, ctx)
        self.supported_params = tuple(cfg.get("supported_params", ["threads", "ctx"]))
        self.param_args = cfg.get("param_args", {})
        self.dir = ctx.data_dir / "engines" / name
        self.image = cfg.get("image")

    # --- binario ---------------------------------------------------------------------------
    def _url(self):
        d = self.cfg.get("download") or {}
        if not self.ctx.machine.get("avx2") and d.get("url_noavx2"):
            return d["url_noavx2"]
        return d.get("url")

    @property
    def bin(self) -> Path | None:
        d = self.cfg.get("download")
        if self.cfg.get("bin"):
            return Path(os.path.expanduser(self.cfg["bin"]))
        if d:
            return self.dir / d.get("file", Path(self._url()).name)
        return None

    def prepare(self, log, update=False):
        if self.image:
            have = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True).returncode == 0
            if update or not have:
                run_cmd(["docker", "pull", self.image], log, timeout=3600)
        b = self.bin
        if b and self.cfg.get("download") and (update or not b.exists()):
            self.dir.mkdir(parents=True, exist_ok=True)
            url = self._url()
            with open(log, "a") as f:
                f.write(f"download {url}\n")
            req = urllib.request.Request(url, headers={"User-Agent": "gilpa-lab/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(b, "wb") as out:
                while chunk := r.read(1 << 22):
                    out.write(chunk)
            b.chmod(b.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        for step in self.cfg.get("prepare_cmds", []):
            run_cmd([self._fmt(x, {}) for x in step], log, timeout=7200)

    def version(self) -> str:
        if self.cfg.get("version_cmd"):
            out = subprocess.run([self._fmt(x, {}) for x in self.cfg["version_cmd"]],
                                 capture_output=True, text=True, timeout=120)
            if out.returncode:
                raise LabError("engine_unavailable", f"{self.name}: version_cmd fallito: {out.stderr[-300:]}")
            return "cmd:" + " ".join(out.stdout.split())[:80]
        if self.image:
            out = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", self.image],
                                 capture_output=True, text=True)
            if out.returncode:
                raise LabError("engine_unavailable", f"immagine {self.image} assente: lancia prepare")
            return f"docker:{out.stdout.strip()[7:19]}"
        b = self.bin
        if b and b.exists():
            return f"bin:{sha256_file(b)[:12]}"
        raise LabError("engine_unavailable", f"{self.name}: binario assente, lancia prepare")

    # --- argomenti -------------------------------------------------------------------------
    def supports_value(self, param, value):
        m = self.param_args.get(param)
        if m is None or value is None or not isinstance(m, dict):
            return True
        return str(value) in {str(k) for k in m}

    def _fmt(self, s, vals):
        base = {"python": sys.executable, "lab": str(LAB_DIR), "data": str(self.ctx.data_dir.resolve()),
                "bin": str(self.bin) if self.bin else ""}
        return str(s).format(**{**base, **vals})

    def start(self, ref, params, model_cfg, log, draft=None) -> Server:
        port = free_port()
        vals = {"port": port,
                "model": str(ref.local.resolve()) if ref.local and ref.local.exists() else ref.source,
                "model_repo": ref.source, "model_dir": str(ref.local.resolve()) if ref.local else "",
                **{k: v for k, v in params.items() if v is not None}}
        args = [self._fmt(x, vals) for x in self.cfg["cmd"]]
        for p, mapping in self.param_args.items():
            v = params.get(p)
            if v is None:
                continue
            if isinstance(mapping, dict):
                args += [self._fmt(x, vals) for x in mapping.get(str(v), mapping.get(v, []))]
            else:
                args += [self._fmt(x, vals) for x in mapping]
        if not self.ctx.machine.get("avx2"):
            args += self.cfg.get("noavx2_args", [])
        args += model_cfg.get(f"{self.name}_args", [])
        cname = None
        if self.image:
            cname = f"lab-{self.name}-{port}"
            d = str(self.ctx.data_dir.resolve())
            env = sum((["-e", f"{k}={self._fmt(v, vals)}"] for k, v in (self.cfg.get("env") or {}).items()), [])
            cmd = ["docker", "run", "--rm", "--name", cname, "--network", "host", "-v", f"{d}:{d}", *env,
                   *self.cfg.get("docker_args", []), self.image, *args]
            proc = start_bg(cmd, log)
        else:
            env = {k: self._fmt(v, vals) for k, v in (self.cfg.get("env") or {}).items()}
            proc = start_bg(args, log, env=env)

        def pid():
            if not cname:
                return proc.pid
            r = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", cname], capture_output=True, text=True)
            return int(r.stdout.strip() or 0) or None

        def stop():
            if cname:
                subprocess.run(["docker", "stop", "-t", "10", cname], capture_output=True)

        extra = dict(self.cfg.get("request_extra") or {})
        if params.get("thinking") == "off":
            extra.update(self.cfg.get("thinking_off_body") or {})
        api = self.cfg.get("api_base", "/v1")
        return Server(f"http://127.0.0.1:{port}{api}", self.cfg.get("model_name", "lab"), proc, log, pid, stop,
                      extra, "</think>" if params.get("thinking") == "on" else None)
