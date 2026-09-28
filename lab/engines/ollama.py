"""Ollama in Docker: importa il GGUF con un Modelfile, così usa esattamente gli stessi pesi."""
from __future__ import annotations

import subprocess
import time

from ..client import Client
from ..util import LabError, free_port, h, run_cmd, start_bg
from .base import Engine, Server


class Ollama(Engine):
    kind = "ollama"
    supported_params = ("threads", "ctx", "batch", "fa", "kv", "thinking")

    def supports_value(self, param, value):
        if param == "kv":
            return value in ("f16", "q8_0", "q4_0", None)
        return True

    def prepare(self, log, update=False):
        img = self.cfg.get("image", "ollama/ollama:latest")
        have = subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode == 0
        if update or not have:
            run_cmd(["docker", "pull", img], log, timeout=3600)

    def version(self):
        img = self.cfg.get("image", "ollama/ollama:latest")
        out = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", img], capture_output=True, text=True)
        if out.returncode:
            raise LabError("engine_unavailable", f"immagine {img} assente: lancia prepare")
        return f"docker:{out.stdout.strip()[7:19]}"

    def start(self, ref, params, model_cfg, log, draft=None) -> Server:
        port = free_port()
        d = str(self.ctx.data_dir.resolve())
        store = self.ctx.data_dir.resolve() / "engines" / "ollama-store"
        store.mkdir(parents=True, exist_ok=True)
        cname = f"lab-ollama-{port}"
        env = {"OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
               "OLLAMA_FLASH_ATTENTION": "1" if params.get("fa") == "on" else "0",
               "OLLAMA_KV_CACHE_TYPE": params.get("kv") or "f16", "OLLAMA_MODELS": str(store)}
        cmd = ["docker", "run", "--rm", "--name", cname, "--network", "host", "-v", f"{d}:{d}"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        proc = start_bg(cmd + [self.cfg.get("image", "ollama/ollama:latest")], log)

        def stop():
            subprocess.run(["docker", "stop", "-t", "10", cname], capture_output=True)

        def pid():
            r = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", cname], capture_output=True, text=True)
            return int(r.stdout.strip() or 0) or None

        t0 = time.monotonic()
        while time.monotonic() - t0 < 120:
            if proc.poll() is not None:
                raise LabError("engine_start", "container ollama terminato", str(log))
            try:
                Client(f"http://127.0.0.1:{port}", "").get("/api/version")
                break
            except Exception:
                time.sleep(2)
        name = f"lab-{h([ref.sha256, params], 8)}"
        mf = ref.local.resolve().parent / f"Modelfile.{name}"
        lines = [f"FROM {ref.local.resolve()}"]
        for k, v in (("num_thread", params.get("threads")), ("num_ctx", params.get("ctx")),
                     ("num_batch", params.get("batch"))):
            if v:
                lines.append(f"PARAMETER {k} {v}")
        mf.write_text("\n".join(lines) + "\n")
        try:
            run_cmd(["docker", "exec", cname, "ollama", "create", name, "-f", str(mf)], log, timeout=1800)
        except LabError:
            stop()
            raise
        extra = {}
        if params.get("thinking") == "off":
            extra.update(self.cfg.get("thinking_off_body", {"think": False}))
        return Server(f"http://127.0.0.1:{port}/v1", name, proc, log, pid, stop, extra,
                      "</think>" if params.get("thinking") == "on" else None)
