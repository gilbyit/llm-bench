"""Test di velocità pura: llama-bench e LocalScore."""
from __future__ import annotations

import json
import re
import urllib.request

from ..util import LabError, run_cmd, sha256_file
from .base import Result, Test


class LlamaBench(Test):
    kind = "llama_bench"
    VERSION = "1"
    needs_server = False
    needs_tools = True
    default_depends = ("machine", "model", "quant", "engine", "params.threads", "params.fa", "params.kv",
                       "params.batch", "params.ubatch")

    def run(self, rt):
        c = self.cfg
        args = ["-m", rt.ref.local.resolve(), "-p", c.get("pp", 512), "-n", c.get("tg", 128),
                "-r", c.get("reps", 3), "-o", "json", *rt.engine.bench_args(rt.params)]
        out = rt.engine.tool("bench", args, rt.log, timeout=self.lab.cfg["timeouts"].get("tool_s", 7200))
        start = out.find("[")
        try:
            rows = json.loads(out[start:out.rfind("]") + 1])
        except Exception:
            raise LabError("parse", f"output di llama-bench non leggibile: {out[:300]}")
        m, samples = {}, []
        for r in rows:
            tag = "pp" if r.get("n_gen", 0) == 0 else "tg"
            m[f"{tag}_tps"] = (r.get("avg_ts"), "tok/s")
            m[f"{tag}_tps_std"] = (r.get("stddev_ts"), "tok/s")
            samples.append({"case_id": f"{tag}{r.get('n_prompt') or r.get('n_gen')}", "rep": 0,
                            "prompt_tps" if tag == "pp" else "gen_tps": r.get("avg_ts"),
                            "build": r.get("build_commit"), "cpu": r.get("cpu_info"), "type_k": r.get("type_k")})
        return Result(m, samples, {"raw": rows})


class LocalScore(Test):
    """LocalScore (localscore.ai): punteggio unico con modelli ufficiali fissi. Una volta per macchina.

    Comando e URL sono in matrix.yaml perché il progetto cambia spesso: vanno verificati."""
    kind = "localscore"
    VERSION = "1"
    needs_server = False
    level = "machine"
    default_depends = ("machine",)

    @property
    def bin(self):
        return self.lab.data_dir / "engines" / "localscore" / "localscore"

    def data_fingerprint(self):
        parts = [sha256_file(self.bin)[:12] if self.bin.exists() else "no-bin"]
        for f in sorted((self.lab.data_dir / "engines" / "localscore").glob("*.gguf")):
            parts.append(f.name)
        return ":".join(parts)

    def prepare(self, log):
        d = self.bin.parent
        d.mkdir(parents=True, exist_ok=True)
        todo = [(self.cfg["bin_url"], self.bin)] + [(u, d / u.rsplit("/", 1)[1]) for u in self.cfg.get("model_urls", [])]
        for url, dest in todo:
            if dest.exists():
                continue
            with open(log, "a") as f:
                f.write(f"download {url}\n")
            req = urllib.request.Request(url, headers={"User-Agent": "gilpa-lab/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as out:
                while b := r.read(1 << 22):
                    out.write(b)
        self.bin.chmod(0o755)

    def run(self, rt):
        d = self.bin.parent
        results = {}
        for model in sorted(d.glob("*.gguf")):
            cmd = [x.format(bin=self.bin, model=model) for x in self.cfg["cmd"]]
            out = run_cmd(cmd, rt.log, timeout=self.lab.cfg["timeouts"].get("tool_s", 7200))
            tag = model.stem
            for key, rx in {"score": r"LocalScore[^\d]*(\d+(?:\.\d+)?)",
                            "pp_tps": r"[Pp]rompt[^\n]*?(\d+(?:\.\d+)?)\s*tok/s",
                            "tg_tps": r"[Gg]eneration[^\n]*?(\d+(?:\.\d+)?)\s*tok/s",
                            "ttft_ms": r"(?:TTFT|[Tt]ime to [Ff]irst [Tt]oken)[^\d]*(\d+(?:\.\d+)?)"}.items():
                mm = re.search(rx, out)
                if mm:
                    results[f"{tag}.{key}"] = (float(mm.group(1)), None)
        if not results:
            raise LabError("parse", "nessun valore riconosciuto nell'output di LocalScore: controlla cmd nel log")
        return Result(results)
