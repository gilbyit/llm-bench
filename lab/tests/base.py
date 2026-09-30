"""Interfaccia comune dei test.

Ogni test dichiara `depends_on`: le parti della configurazione da cui il suo risultato dipende.
È la chiave della modularità: un test di qualità a temperatura 0 non dipende dal numero di
thread, quindi variare i thread non lo fa ripetere; llama-bench non dipende dai casi del test
intenti, quindi modificare cases_v2.json non lo fa ripetere.

Valori possibili in depends_on: machine, model, quant, engine, params (tutti), params.<nome>.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ..util import LabError

SPEED_AND_QUALITY = ("machine", "model", "quant", "engine", "params")
QUALITY = ("machine", "model", "quant", "engine", "params.kv", "params.thinking", "params.ctx")


@dataclass
class RunCtx:
    cell: dict
    engine: object
    ref: object
    model_cfg: dict
    params: dict
    log: Path
    work: Path
    server: object = None
    lab: object = None          # accesso a config, artefatti, store, macchina

    def client(self):
        return self.server.client(timeout=self.lab.cfg["timeouts"].get("request_s", 1800))


@dataclass
class Result:
    metrics: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class Test:
    kind = "base"
    VERSION = "1"                # incrementalo quando cambia la logica: rifà solo questo test
    needs_server = True
    needs_tools = False
    level = "model"              # "machine": gira una volta per macchina, senza modello
    default_depends = SPEED_AND_QUALITY

    def __init__(self, name: str, cfg: dict, lab):
        self.name, self.cfg, self.lab = name, cfg, lab
        self.depends_on = tuple(cfg.get("depends_on", self.default_depends))
        self._dfp = None

    # --- impronta dei dati -----------------------------------------------------------------
    def data_fingerprint(self) -> str:
        """Hash di tutto ciò che il test usa oltre al modello: dataset, script, versioni di librerie."""
        return ""

    def cached_data_fp(self) -> str:
        if self._dfp is None:
            self._dfp = self.data_fingerprint()
        return self._dfp

    def options(self) -> dict:
        """Opzioni che cambiano il risultato (limit, seed, prompt...). Entrano nella chiave logica."""
        skip = {"kind", "depends_on", "machines", "engines", "models", "est", "enabled", "description"}
        return {k: v for k, v in self.cfg.items() if k not in skip}

    def prepare(self, log: Path):
        pass

    def applicable(self, machine: dict, model_cfg: dict | None, quant_id: str | None, engine,
                   ignore_machines: bool = False) -> str | None:
        # ignore_machines: lo sweep chiede di eseguire il test anche fuori da `machines` (vedi `run_on`)
        if not ignore_machines and self.cfg.get("machines") and machine["id"] not in self.cfg["machines"]:
            return f"{self.name} previsto solo su {', '.join(self.cfg['machines'])}"
        if self.level == "machine":
            return None
        if self.cfg.get("engines") and engine.name not in self.cfg["engines"]:
            return f"{self.name} previsto solo con {', '.join(self.cfg['engines'])}"
        if self.cfg.get("models") and model_cfg["id"] not in self.cfg["models"]:
            return f"{self.name} previsto solo per {', '.join(self.cfg['models'])}"
        if self.needs_tools and not engine.has_tools:
            return f"{self.name} richiede gli strumenti di llama.cpp, assenti in {engine.name}"
        return None

    def run(self, rt: RunCtx) -> Result:
        raise NotImplementedError


# --- aiuti per i test generativi ----------------------------------------------------------
def sample(items: list, limit: int | None, seed: int) -> list:
    if not limit or limit >= len(items):
        return list(items)
    idx = sorted(random.Random(seed).sample(range(len(items)), limit))
    return [items[i] for i in idx]


def median(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(statistics.median(vals), 3) if vals else None


def speed_summary(samples: list) -> dict:
    return {
        "wall_med_s": (median([s.get("wall_s") for s in samples]), "s"),
        "ttft_med_s": (median([s.get("ttft_s") for s in samples]), "s"),
        "prompt_tps_med": (median([s.get("prompt_tps") for s in samples]), "tok/s"),
        "gen_tps_med": (median([s.get("gen_tps") for s in samples]), "tok/s"),
        "prompt_tokens_med": (median([s.get("prompt_tokens") for s in samples]), "tok"),
        "gen_tokens_med": (median([s.get("gen_tokens") for s in samples]), "tok"),
        "errors": (sum(1 for s in samples if s.get("error")), "n"),
    }


def load_hf(lab, path, name=None, split="test", streaming=False):
    try:
        import datasets  # noqa
    except ImportError:
        raise LabError("missing_dep", "manca il pacchetto 'datasets': esegui ./lab/setup.sh")
    import os
    os.environ.setdefault("HF_HOME", str(lab.data_dir / "hf_home"))
    try:
        return datasets.load_dataset(path, name, split=split, streaming=streaming)
    except Exception as e:
        raise LabError("download", f"dataset {path}/{name}: {e}")


def lib_version(mod: str) -> str:
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:
        return "assente"


def check_errors(samples, max_ratio=0.5):
    """Se più di metà dei casi fallisce per errore (non per risposta sbagliata), il run è un errore."""
    n = len(samples)
    bad = sum(1 for s in samples if s.get("error"))
    if n and bad / n > max_ratio:
        first = next(s["error"] for s in samples if s.get("error"))
        from ..util import classify
        raise LabError(classify(first), f"{bad}/{n} casi in errore, es.: {first[:500]}")
