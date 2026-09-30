"""KL divergence della quantizzazione rispetto al riferimento (BF16, o Q8_0 se BF16 non entra in RAM).

I logit del riferimento si calcolano una volta per (modello, file di riferimento, corpus) e si
riusano per tutte le quantizzazioni: sono un artefatto con la sua impronta.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..util import LabError, h, sha256_file
from .base import Result, Test, load_hf

PATTERNS = {
    "kld_mean": r"Mean\s+KLD:\s+([-\d.eE+]+)",
    "kld_median": r"Median\s+KLD:\s+([-\d.eE+]+)",
    "kld_p99": r"99\.0%\s+KLD:\s+([-\d.eE+]+)",
    "kld_max": r"Maximum\s+KLD:\s+([-\d.eE+]+)",
    "same_top_p": r"Same top p:\s+([-\d.eE+]+)",
    "ppl_q": r"Mean PPL\(Q\)\s*:\s*([-\d.eE+]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([-\d.eE+]+)",
    "delta_p_mean": r"Mean\s+Δp:\s+([-\d.eE+]+)",
}


class KLD(Test):
    kind = "kld"
    VERSION = "2"
    needs_server = False
    needs_tools = True
    default_depends = ("model", "quant", "engine")

    def corpus_path(self, name) -> Path:
        return self.lab.data_dir / "corpora" / f"{name}.txt"

    def data_fingerprint(self):
        parts = []
        for c in self.cfg.get("corpora", []):
            p = self.corpus_path(c)
            parts.append(f"{c}:{sha256_file(p)[:12]}" if p.exists() else f"{c}:missing")
        return ",".join(parts)

    def prepare(self, log):
        corpora = self.lab.cfg.get("corpora", {})
        for name in self.cfg.get("corpora", []):
            p = self.corpus_path(name)
            if p.exists():
                continue
            spec = corpora[name]
            ds = load_hf(self.lab, spec["hf"], spec.get("config"), spec.get("split", "test"),
                         streaming=spec.get("streaming", False))
            out, total, cap = [], 0, spec.get("max_chars", 10 ** 9)
            for row in ds:
                t = row[spec.get("field", "text")]
                out.append(t)
                total += len(t)
                if total >= cap:
                    break
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(out), encoding="utf-8")

    def _base_quant(self, rt):
        max_gb = rt.lab.machine.get("max_model_gb", 12)
        for q in self.cfg.get("base", ["BF16", "Q8_0"]):
            try:
                ref = rt.lab.artifacts.resolve(rt.model_cfg["id"], q)
            except LabError:
                continue
            if ref.size_gb and ref.size_gb <= max_gb:
                return q, ref
        raise LabError("unsupported", "nessun riferimento (BF16/Q8_0) entra in RAM")

    def applicable(self, machine, model_cfg, quant_id, engine, ignore_machines=False):
        r = super().applicable(machine, model_cfg, quant_id, engine, ignore_machines)
        if r:
            return r
        if quant_id in ("BF16", "F16"):
            return "la quantizzazione di riferimento non si confronta con sé stessa"
        return None

    def run(self, rt):
        bq, bref = self._base_quant(rt)
        if bq == rt.cell["quant"]:
            return Result({}, extra={"base": bq, "note": "è il riferimento: KLD non applicabile"})
        threads = rt.params.get("threads") or rt.lab.machine["physical_cores"]
        tmo = rt.lab.cfg["timeouts"].get("tool_s", 7200)
        metrics = {}
        for corpus in self.cfg.get("corpora", []):
            cpath = self.corpus_path(corpus)
            if not cpath.exists():
                raise LabError("missing_data", f"corpus {corpus} assente: lancia prepare")
            key = f"kldbase:{h([rt.model_cfg['id'], bref.sha256, sha256_file(cpath)[:16], self.cfg.get('chunks'), self.cfg.get('ctx'), rt.engine.version()])}"
            row = rt.lab.store.get_artifact(key)
            base_file = rt.lab.data_dir / "kld" / f"{key.split(':')[1]}.kld"
            if not (row and base_file.exists()):
                rt.lab.artifacts.ensure(bref, rt.lab.log, rt.lab.quantize_fn())
                base_file.parent.mkdir(parents=True, exist_ok=True)
                rt.engine.tool("perplexity", ["-m", bref.local.resolve(), "-f", cpath.resolve(),
                                              "--kl-divergence-base", base_file.resolve(),
                                              "-c", self.cfg.get("ctx", 512), "--chunks", self.cfg.get("chunks", 20),
                                              "-t", threads], rt.log, tmo)
                rt.lab.store.save_artifact(key, "kld-base", bref.source, base_file, base_file.stat().st_size,
                                           None, {"base_quant": bq, "corpus": corpus})
            out = rt.engine.tool("perplexity", ["-m", rt.ref.local.resolve(), "--kl-divergence-base",
                                                base_file.resolve(), "--kl-divergence",
                                                "-c", self.cfg.get("ctx", 512), "-t", threads],
                                 rt.log, tmo)
            found = 0
            for name, rx in PATTERNS.items():
                mm = re.search(rx, out)
                if mm:
                    found += 1
                    unit = "%" if name in ("same_top_p", "delta_p_mean") else None
                    metrics[f"{corpus}.{name}"] = (float(mm.group(1)), unit)
            if not found:
                raise LabError("parse", f"output di llama-perplexity non riconosciuto ({corpus})")
        metrics["base_is_bf16"] = (1.0 if bq == "BF16" else 0.0, None)
        return Result(metrics, extra={"base": bq, "base_sha": bref.sha256})
