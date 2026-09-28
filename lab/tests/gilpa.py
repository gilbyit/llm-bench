"""Il test custom: gilpa-llm-bench (bench.py + cases_v2.json), usato come scatola nera.

bench.py non viene modificato. Si lancia con l'endpoint del motore in prova e si legge il CSV che
produce. L'impronta dei dati è l'hash di bench.py, dei casi e dello schema SQL: se cambi un caso,
si ripete solo questo test, su tutte le combinazioni.
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

from ..util import LabError, run_cmd, sha256_file
from .base import SPEED_AND_QUALITY, Result, Test, median


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class GilpaIntent(Test):
    kind = "gilpa_intent"
    VERSION = "1"
    default_depends = SPEED_AND_QUALITY

    @property
    def bench_dir(self) -> Path:
        return self.lab.bench_dir

    def data_fingerprint(self):
        files = [self.bench_dir / f for f in self.cfg.get("files", ["bench.py", "cases_v2.json", "seed.sql"])]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            return "missing:" + ",".join(missing)
        return ",".join(f"{f.name}:{sha256_file(f)[:12]}" for f in files)

    def prepare(self, log):
        if not (self.bench_dir / "bench.py").exists():
            raise LabError("missing_data", f"bench.py non trovato in {self.bench_dir} (paths.bench_dir)")

    def _one(self, rt, label, idx):
        c = self.cfg
        cmd = [sys.executable, self.bench_dir / "bench.py", "--base-url", rt.server.base_url,
               "--model", rt.server.model_name, "--api-key-env", "LAB_KEY", "--label", label,
               "--tasks", ",".join(c.get("tasks", ["intent_v3"]))]
        if c.get("extra_tokens") is not None:
            cmd += ["--extra-tokens", str(c["extra_tokens"])]
        if rt.params.get("prompt_cache") == "off":
            cmd += ["--cold"]
        cmd += [str(a) for a in c.get("bench_args", [])]
        out = run_cmd(cmd, rt.log, timeout=rt.lab.cfg["timeouts"].get("test_s", 86400),
                      env={"LAB_KEY": "sk-lab"}, cwd=self.bench_dir)
        mm = re.search(r"Dettagli:\s*(\S+\.csv)", out)
        if not mm:
            raise LabError("parse", "bench.py non ha stampato 'Dettagli: <file.csv>'")
        path = Path(mm.group(1))
        if not path.is_absolute():
            path = self.bench_dir / path
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        return rows, path

    def run(self, rt):
        n_inv = int(self.cfg.get("invocations", 1))
        label = f"lab-{rt.cell['logical_key'][:10]}"
        all_rows, files = [], []
        for i in range(n_inv):
            rows, path = self._one(rt, f"{label}-{i}" if n_inv > 1 else label, i)
            for r in rows:
                r["_inv"] = i
            all_rows += rows
            files.append(str(path))
        if not all_rows:
            raise LabError("parse", "CSV di bench.py vuoto")
        errs = [r for r in all_rows if r.get("error")]
        if len(errs) > len(all_rows) / 2:
            from ..util import classify
            raise LabError(classify(errs[0]["error"]), f"{len(errs)}/{len(all_rows)} casi in errore: {errs[0]['error'][:300]}")

        metrics, samples = {}, []
        for r in all_rows:
            ok = r.get("correct") == "1"
            ent = r.get("entities_ok")
            allok = ok and (ent in ("1", "", None))
            samples.append({"case_id": f"{r.get('task')}:{r.get('case')}", "rep": r["_inv"],
                            "correct": 1.0 if allok else 0.0, "wall_s": _num(r.get("wall_s")),
                            "prompt_tokens": _num(r.get("prompt_tokens")), "gen_tokens": _num(r.get("gen_tokens")),
                            "prompt_tps": _num(r.get("prompt_tps")), "gen_tps": _num(r.get("gen_tps")),
                            "error": r.get("error") or None, "intent_ok": ok, "entities_ok": ent,
                            "clean": r.get("clean"), "json_ok": r.get("json_ok"), "output": (r.get("output") or "")[:500]})
        for task in sorted({r.get("task") for r in all_rows}):
            R = [r for r in all_rows if r.get("task") == task]
            S = [s for s in samples if s["case_id"].startswith(task + ":")]
            n = len(R)
            per_inv = [sum(s["correct"] for s in S if s["rep"] == i) for i in range(n_inv)]
            cases = n // n_inv if n_inv else n
            metrics.update({
                f"{task}.n": (cases, "n"),
                f"{task}.tutto_giusto": (statistics.mean(per_inv), "n"),
                f"{task}.tutto_giusto_pct": (100 * statistics.mean(per_inv) / cases if cases else None, "%"),
                f"{task}.intent_pct": (100 * sum(r.get("correct") == "1" for r in R) / n, "%"),
                f"{task}.json_pct": (100 * sum(r.get("json_ok") == "1" for r in R) / n, "%"),
                f"{task}.wall_med_s": (median([_num(r.get("wall_s")) for r in R]), "s"),
                f"{task}.wall_max_s": (max([_num(r.get("wall_s")) or 0 for r in R]), "s"),
                f"{task}.wall_first_s": (_num(R[0].get("wall_s")), "s"),
                f"{task}.prompt_tps_med": (median([_num(r.get("prompt_tps")) for r in R]), "tok/s"),
                f"{task}.gen_tps_med": (median([_num(r.get("gen_tps")) for r in R]), "tok/s"),
                f"{task}.prompt_tokens_med": (median([_num(r.get("prompt_tokens")) for r in R]), "tok"),
                f"{task}.gen_tokens_med": (median([_num(r.get("gen_tokens")) for r in R]), "tok"),
                f"{task}.errors": (sum(1 for r in R if r.get("error")), "n"),
            })
            cl = [r.get("clean") for r in R if r.get("clean") not in ("", None)]
            if cl:
                metrics[f"{task}.clean_pct"] = (100 * sum(x == "1" for x in cl) / len(cl), "%")
            ent = [r.get("entities_ok") for r in R if r.get("entities_ok") not in ("", None)]
            if ent:
                metrics[f"{task}.entities_pct"] = (100 * sum(x == "1" for x in ent) / len(ent), "%")
            if n_inv > 1:
                by_case = {}
                for s in S:
                    by_case.setdefault(s["case_id"], []).append(s["correct"])
                metrics[f"{task}.min_inv"] = (min(per_inv), "n")
                metrics[f"{task}.max_inv"] = (max(per_inv), "n")
                metrics[f"{task}.stable_cases_pct"] = (
                    100 * sum(1 for v in by_case.values() if len(set(v)) == 1) / len(by_case), "%")
        return Result(metrics, samples, {"csv": files})
