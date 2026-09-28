"""llama.cpp e derivati con la stessa riga di comando: immagine Docker ufficiale, build nativa,
ik_llama.cpp, bitnet.cpp. Sono gli unici motori che espongono llama-bench, llama-perplexity e
llama-quantize, quindi gli unici su cui girano i test di velocità pura e di KL divergence."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..util import LabError, h, run_cmd, sha256_file, start_bg, free_port
from .base import Engine, Server


class LlamaCpp(Engine):
    kind = "llamacpp"
    has_tools = True

    def __init__(self, name, cfg, ctx):
        super().__init__(name, cfg, ctx)
        self.docker = cfg.get("image_server") is not None
        self.fa_style = cfg.get("fa_style", "value")        # value: -fa on|off ; switch: -fa
        self.jinja = cfg.get("jinja", True)
        self.think_off = cfg.get("thinking_off_args", ["--reasoning-budget", "0"])
        self.extra = cfg.get("extra_args", [])
        root = ctx.data_dir / "engines" / name
        self.src = Path(cfg["src_dir"]) if cfg.get("src_dir") else root / "src"
        self.bin_dir = Path(cfg["bin_dir"]) if cfg.get("bin_dir") else self.src / "build" / "bin"

    def supports_value(self, param, value):
        if param == "kv" and value not in ("f16", "q8_0", "q4_0", "q5_0", "q5_1", "iq4_nl", "bf16", None):
            return False
        return True

    # --- preparazione ----------------------------------------------------------------------
    def prepare(self, log, update=False):
        if self.docker:
            for img in {self.cfg["image_server"], self.cfg.get("image_full")} - {None}:
                have = subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode == 0
                if update or not have:
                    run_cmd(["docker", "pull", img], log, timeout=3600)
            return
        if self.cfg.get("bin_dir"):
            if not (self.bin_dir / "llama-server").exists():
                raise LabError("engine_unavailable", f"{self.bin_dir}/llama-server non trovato")
            return
        repo, ref = self.cfg["repo"], self.cfg.get("ref", "master")
        if not (self.src / ".git").exists():
            self.src.parent.mkdir(parents=True, exist_ok=True)
            run_cmd(["git", "clone", "--recursive", repo, self.src], log, timeout=3600)
        elif update:
            run_cmd(["git", "-C", self.src, "fetch", "--all", "--tags"], log, timeout=600)
        if update or not (self.bin_dir / "llama-server").exists():
            run_cmd(["git", "-C", self.src, "checkout", ref], log, timeout=120)
            if update:
                subprocess.run(["git", "-C", str(self.src), "pull", "--ff-only"], capture_output=True)
            for step in self.cfg.get("pre_build", []):
                run_cmd(step, log, timeout=3600, cwd=self.src)
            flags = self.cfg.get("cmake_flags", ["-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF"])
            run_cmd(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", *flags], log, timeout=600, cwd=self.src)
            run_cmd(["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count() or 2),
                     "--target", "llama-server", "llama-bench", "llama-perplexity", "llama-quantize"],
                    log, timeout=7200, cwd=self.src)

    def version(self) -> str:
        if self.docker:
            out = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", self.cfg["image_server"]],
                                 capture_output=True, text=True)
            if out.returncode:
                raise LabError("engine_unavailable", f"immagine {self.cfg['image_server']} assente: lancia prepare")
            full = self.cfg.get("image_full")
            fid = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", full],
                                 capture_output=True, text=True).stdout.strip() if full else ""
            return f"docker:{out.stdout.strip()[7:19]}:{fid[7:19]}"
        if (self.src / ".git").exists():
            c = subprocess.run(["git", "-C", str(self.src), "rev-parse", "--short=12", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
            if c and (self.bin_dir / "llama-server").exists():
                return f"git:{c}:{h(self.cfg.get('cmake_flags', []), 6)}"
        b = self.bin_dir / "llama-server"
        if b.exists():
            return f"bin:{sha256_file(b)[:12]}"
        raise LabError("engine_unavailable", f"{self.name}: llama-server non compilato, lancia prepare")

    # --- argomenti -------------------------------------------------------------------------
    def _fa(self, v):
        if v is None:
            return []
        if self.fa_style == "switch":
            return ["-fa"] if v == "on" else []
        return ["-fa", v]

    def server_args(self, model_path, params, model_cfg, port, draft=None):
        p = params
        a = ["-m", model_path, "--host", "127.0.0.1", "--port", str(port), "-np", "1"]
        if p.get("threads"):
            a += ["-t", p["threads"], "-tb", p.get("threads_batch") or p["threads"]]
        if p.get("ctx"):
            a += ["-c", p["ctx"]]
        if p.get("batch"):
            a += ["-b", p["batch"]]
        if p.get("ubatch"):
            a += ["-ub", p["ubatch"]]
        a += self._fa(p.get("fa"))
        if p.get("kv") and p["kv"] != "f16":
            a += ["-ctk", p["kv"], "-ctv", p["kv"]]
        if self.jinja:
            a += ["--jinja"]
        if p.get("thinking") == "off":
            a += self.think_off
        if draft is not None:
            a += ["-md", draft.local, "--draft-max", str(p.get("draft_max", 16)), "--draft-min", "1"]
        a += model_cfg.get("llamacpp_args", []) + self.extra
        return [str(x) for x in a]

    def _docker_prefix(self, name, image, entry=None):
        d = str(self.ctx.data_dir.resolve())
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "host", "-v", f"{d}:{d}"]
        if self.cfg.get("docker_args"):
            cmd += self.cfg["docker_args"]
        return cmd + [image] + ([entry] if entry else [])

    # --- server ----------------------------------------------------------------------------
    def start(self, ref, params, model_cfg, log, draft=None) -> Server:
        port = free_port()
        args = self.server_args(ref.local.resolve(), params, model_cfg, port, draft)
        cname = None
        if self.docker:
            cname = f"lab-{self.name}-{port}"
            cmd = self._docker_prefix(cname, self.cfg["image_server"]) + args
        else:
            cmd = [self.bin_dir / "llama-server", *args]
        proc = start_bg(cmd, log)

        def pid():
            if not cname:
                return proc.pid
            r = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", cname], capture_output=True, text=True)
            return int(r.stdout.strip() or 0) or None

        def stop():
            if cname:
                subprocess.run(["docker", "stop", "-t", "10", cname], capture_output=True)

        extra = {}
        if params.get("prompt_cache") == "off":
            extra["cache_prompt"] = False
        return Server(f"http://127.0.0.1:{port}/v1", "lab", proc, log, pid, stop, extra,
                      "</think>" if params.get("thinking") == "on" else None)

    # --- strumenti -------------------------------------------------------------------------
    _TOOLS = {"bench": ("llama-bench", "--bench"), "perplexity": ("llama-perplexity", "--perplexity"),
              "quantize": ("llama-quantize", "--quantize")}

    def tool(self, name, args, log, timeout):
        binary, flag = self._TOOLS[name]
        args = [str(a) for a in args]
        if self.docker:
            img = self.cfg.get("image_full")
            if not img:
                raise LabError("unsupported", f"{self.name}: manca image_full per {binary}")
            cmd = self._docker_prefix(f"lab-{name}-{os.getpid()}", img, flag) + args
        else:
            b = self.bin_dir / binary
            if not b.exists():
                raise LabError("engine_unavailable", f"{b} non trovato")
            cmd = [b, *args]
        return run_cmd(cmd, log, timeout=timeout)

    def bench_args(self, params):
        p = params
        a = []
        if p.get("threads"):
            a += ["-t", p["threads"]]
        if p.get("fa") is not None:
            a += ["-fa", "1" if p["fa"] == "on" else "0"]
        if p.get("kv") and p["kv"] != "f16":
            a += ["-ctk", p["kv"], "-ctv", p["kv"]]
        if p.get("batch"):
            a += ["-b", p["batch"]]
        if p.get("ubatch"):
            a += ["-ub", p["ubatch"]]
        return [str(x) for x in a]

    def quantize(self, src: Path, dst: Path, qtype: str, imatrix: Path | None, log: Path):
        args = (["--imatrix", imatrix] if imatrix else []) + [src, dst, qtype]
        self.tool("quantize", args, log, timeout=7200)
