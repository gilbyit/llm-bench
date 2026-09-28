"""Interfaccia comune dei motori di inferenza."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..client import Client
from ..util import LabError, classify, kill_group, tail

ALL_PARAMS = ("threads", "threads_batch", "fa", "kv", "batch", "ubatch", "ctx", "thinking",
              "prompt_cache", "draft")


@dataclass
class Server:
    base_url: str
    model_name: str
    proc: subprocess.Popen | None
    log: Path
    pid_fn: Callable[[], int | None]
    stop_fn: Callable[[], None]
    extra_body: dict = field(default_factory=dict)   # campi da aggiungere a ogni richiesta
    think_end: str | None = None

    def client(self, timeout=1800) -> Client:
        return Client(self.base_url, self.model_name, "sk-lab", timeout, self.extra_body)

    def stop(self):
        try:
            self.stop_fn()
        finally:
            if self.proc:
                kill_group(self.proc)

    def alive(self) -> bool:
        return self.proc is None or self.proc.poll() is None


class Engine:
    kind = "base"

    def __init__(self, name: str, cfg: dict, ctx):
        self.name, self.cfg, self.ctx = name, cfg, ctx
        self.formats = set(cfg.get("formats", ["gguf"]))
        self.requires_cpu = cfg.get("requires_cpu", [])
        self.max_params_b = cfg.get("max_params_b")
        self.only_machines = cfg.get("machines")

    # --- metadati usati dal pianificatore -------------------------------------------------
    supported_params: tuple = ALL_PARAMS
    has_tools = False  # llama-bench, llama-perplexity, llama-quantize

    def supports_value(self, param: str, value) -> bool:
        return True

    def normalize(self, params: dict) -> dict:
        """I parametri che il motore non espone diventano None, così le celle equivalenti coincidono."""
        return {k: (v if k in self.supported_params else None) for k, v in params.items()}

    def compatible(self, machine: dict, model_cfg: dict, quant_cfg: dict) -> str | None:
        if quant_cfg["format"] not in self.formats:
            return f"formato {quant_cfg['format']} non gestito da {self.name}"
        for flag in self.requires_cpu:
            if not machine.get(flag):
                return f"{self.name} richiede {flag.upper()}"
        if self.max_params_b and (model_cfg.get("params_b") or 0) > self.max_params_b:
            return f"{self.name} limitato a modelli fino a {self.max_params_b}B"
        if self.only_machines and machine["id"] not in self.only_machines:
            return f"{self.name} non previsto su {machine['id']}"
        allowed = model_cfg.get("engines")
        if allowed and self.name not in allowed:
            return f"{model_cfg['id']} gira solo su {', '.join(allowed)}"
        return None

    # --- ciclo di vita ---------------------------------------------------------------------
    def prepare(self, log: Path, update: bool = False):
        pass

    def version(self) -> str:
        raise NotImplementedError

    def start(self, ref, params: dict, model_cfg: dict, log: Path, draft=None) -> Server:
        raise NotImplementedError

    def tool(self, name: str, args: list, log: Path, timeout: float) -> str:
        raise LabError("unsupported", f"{self.name} non ha lo strumento {name}")

    # --- utilità ---------------------------------------------------------------------------
    def wait_ready(self, server: Server, timeout: float):
        c = Client(server.base_url, server.model_name, "sk-lab")
        ready_url = self.cfg.get("ready_path")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not server.alive():
                rc = server.proc.returncode if server.proc else None
                txt = tail(server.log)
                raise LabError(classify(txt, rc), f"{self.name} terminato durante l'avvio (exit {rc})\n{txt[-1500:]}",
                               str(server.log))
            if ready_url:
                try:
                    Client(server.base_url.split("/v")[0], "").get(ready_url, timeout=5)
                    return
                except Exception:
                    pass
            elif c.ready():
                return
            time.sleep(2)
        txt = tail(server.log)
        server.stop()
        raise LabError("timeout", f"{self.name} non pronto dopo {timeout}s\n{txt[-1500:]}", str(server.log))
