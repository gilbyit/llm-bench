"""Espansione della matrice (sweep) in celle, regole di compatibilità, chiavi e stime di durata."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from .util import h

PARAM_ORDER = ("threads", "threads_batch", "fa", "kv", "batch", "ubatch", "ctx", "thinking", "prompt_cache", "draft")


def project(cell: dict, deps) -> dict:
    out = {}
    for d in deps:
        if d == "params":
            out["params"] = cell["params"]
        elif d.startswith("params."):
            out.setdefault("p", {})[d[7:]] = cell["params"].get(d[7:])
        else:
            out[d] = cell.get(d)
    return out


def logical_key(test, cell) -> str:
    return h({"test": test.name, "opts": test.options(), "cell": project(cell, test.depends_on)}, 24)


def _as_list(v, allv):
    if v in (None, "all"):
        return list(allv)
    return [v] if isinstance(v, str) else list(v)


class Planner:
    def __init__(self, lab):
        self.lab = lab
        self.cfg = lab.cfg

    # --- parametri -------------------------------------------------------------------------
    def _param_sets(self, sweep) -> list[dict]:
        base = {**self.cfg["defaults"]["params"], **(sweep.get("params_fixed") or {})}
        sets = []
        if sweep.get("params_list"):
            sets = [{**base, **p} for p in sweep["params_list"]]
        else:
            grid = sweep.get("params") or {}
            keys = list(grid)
            for combo in itertools.product(*[grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]):
                sets.append({**base, **dict(zip(keys, combo))})
        return sets or [base]

    def _concrete(self, params: dict, engine, model_cfg) -> tuple[dict | None, str | None]:
        m = self.lab.machine
        p = dict(params)
        for k in ("threads", "threads_batch"):
            if p.get(k) == "physical":
                p[k] = m["physical_cores"]
            elif p.get(k) == "logical":
                p[k] = m["logical_threads"]
        if p.get("threads_batch") in (None, p.get("threads")):
            p["threads_batch"] = None
        if "thinking" not in (model_cfg.get("capabilities") or []):
            p["thinking"] = None
        if p.get("thinking") == "on" and m.get("thinking") is False:
            return None, f"thinking disattivato su {m['id']}"
        if p.get("draft") in ("none", None):
            p["draft"] = None
        p = engine.normalize(p)
        for k, v in p.items():
            if v is not None and not engine.supports_value(k, v):
                return None, f"{engine.name} non supporta {k}={v}"
        if p.get("kv") not in (None, "f16") and p.get("fa") == "off":
            return None, "KV cache quantizzata richiede flash attention"
        if p.get("draft") and not engine.has_tools:
            return None, "decodifica speculativa prevista solo con llama.cpp"
        return {k: p.get(k) for k in PARAM_ORDER}, None

    def _excluded(self, model, quant, engine, test) -> str | None:
        """Regole `exclude` della macchina: combinazioni che si è deciso di non eseguire."""
        for rule in self.lab.machine.get("exclude") or []:
            if all(val in rule[key] for key, val in (("models", model), ("quants", quant),
                                                     ("engines", engine), ("tests", test)) if rule.get(key)):
                return "escluso: " + rule.get("reason", f"regola exclude di {self.lab.machine['id']}")
        return None

    # --- espansione ------------------------------------------------------------------------
    def expand(self, sweeps=None, include_disabled=False) -> list[dict]:
        cfg, lab = self.cfg, self.lab
        models = [mid for mid, m in cfg["models"].items() if m.get("enabled", True)]
        quants_all = list(cfg["quants"])
        engines_all = [e for e, c in cfg["engines"].items() if c.get("enabled", True)]
        cells, seen = [], {}
        for sname, sw in cfg["sweeps"].items():
            if sweeps and sname not in sweeps:
                continue
            if not sweeps and not sw.get("enabled", True) and not include_disabled:
                continue
            tests = [t for t in lab.expand_tests(sw.get("tests", "base")) if t in lab.tests]
            # run_on: macchine su cui questo sweep esegue i suoi test anche se il test è limitato ad altre
            ign = lab.machine["id"] in (sw.get("run_on") or [])
            machine_tests = [t for t in tests if lab.tests[t].level == "machine"]
            model_tests = [t for t in tests if lab.tests[t].level != "machine"]
            for t in machine_tests:
                test = lab.tests[t]
                cell = {"sweep": sname, "machine": lab.machine["id"], "model": None, "quant": None, "engine": None,
                        "params": {}, "test": t, "test_opts": test.options()}
                reason = test.applicable(lab.machine, None, None, None, ign)
                self._add(cells, seen, cell, test, reason)
            for mid in _as_list(sw.get("models"), models):
                if mid not in cfg["models"]:
                    raise SystemExit(f"sweep {sname}: modello '{mid}' non definito")
                m = cfg["models"][mid]
                if not m.get("enabled", True):
                    continue
                allowed = m.get("allowed_quants")
                qlist = _as_list(sw.get("quants"), quants_all)
                if allowed:
                    q2 = [q for q in qlist if q in allowed]
                    qlist = q2 or (allowed[:1] if sw.get("fallback_quant", True) else [])
                elist = _as_list(sw.get("engines"), engines_all)
                if m.get("engines") and not set(elist) & set(m["engines"]):
                    elist = list(m["engines"])  # es. BitNet: gira solo sul suo motore
                for qid in qlist:
                    if qid not in cfg["quants"]:
                        raise SystemExit(f"sweep {sname}: quantizzazione '{qid}' non definita")
                    q = cfg["quants"][qid]
                    over = (m.get("quants") or {}).get(qid) or {}
                    srcs = m.get("sources") or {}
                    if not (q.get("make") or over.get("make") or over.get("local_path")
                            or q.get("source", q["format"]) in srcs):
                        continue  # il modello non esiste in quel formato: nessuna cella
                    for eid in elist:
                        if eid not in lab.engines:
                            continue
                        eng = lab.engines[eid]
                        if q["format"] not in eng.formats:
                            continue  # formato non pertinente: non è un'incompatibilità da registrare
                        ereason = eng.compatible(lab.machine, m, q)
                        if m.get("machines") and lab.machine["id"] not in m["machines"]:
                            ereason = ereason or f"{mid} previsto solo su {', '.join(m['machines'])}"
                        for ps in self._param_sets(sw):
                            params, preason = self._concrete(ps, eng, m) if not ereason else (
                                {k: ps.get(k) for k in PARAM_ORDER}, None)
                            if params is None:
                                params = {k: ps.get(k) for k in PARAM_ORDER}
                            for t in model_tests:
                                test = lab.tests[t]
                                cell = {"sweep": sname, "machine": lab.machine["id"], "model": mid, "quant": qid,
                                        "engine": eid, "params": params, "test": t, "test_opts": test.options()}
                                reason = (ereason or preason or test.applicable(lab.machine, m, qid, eng, ign)
                                          or self._excluded(mid, qid, eid, t))
                                self._add(cells, seen, cell, test, reason)
        order_m = {m: i for i, m in enumerate(cfg["models"])}
        order_q = {q: i for i, q in enumerate(cfg["quants"])}
        order_e = {e: i for i, e in enumerate(cfg["engines"])}
        order_t = {t: i for i, t in enumerate(cfg["tests"])}
        cells.sort(key=lambda c: (c["model"] is not None, order_m.get(c["model"], -1), order_q.get(c["quant"], -1),
                                  order_e.get(c["engine"], -1), h(c["params"]), order_t.get(c["test"], 0)))
        return cells

    def _add(self, cells, seen, cell, test, reason):
        cell["logical_key"] = logical_key(test, cell)
        cell["skip_reason"] = reason
        prev = seen.get(cell["logical_key"])
        if prev is not None:
            if cell["sweep"] not in prev["sweep"].split(","):
                prev["sweep"] += "," + cell["sweep"]
            if prev["skip_reason"] and not reason:
                prev["skip_reason"] = None  # basta uno sweep che la rende eseguibile
            return
        seen[cell["logical_key"]] = cell
        cells.append(cell)

    # --- stime -----------------------------------------------------------------------------
    def estimate_s(self, cell) -> float:
        """Stima grezza in secondi, da parametri macchina in matrix.yaml. Serve per ordini di grandezza."""
        if cell.get("skip_reason"):
            return 0.0
        test = self.lab.tests[cell["test"]]
        est = test.cfg.get("est") or {}
        if "minutes" in est:
            return est["minutes"] * 60
        if not cell["model"]:
            return 600.0
        m = self.cfg["models"][cell["model"]]
        q = self.cfg["quants"][cell["quant"]]
        pb = m.get("active_params_b") or m.get("params_b") or 4
        size_gb = (m.get("params_b") or 4) * q.get("bits", 4.8) / 8
        act_gb = pb * q.get("bits", 4.8) / 8
        sp = self.lab.machine.get("speed") or {}
        eng = self.cfg["engines"][cell["engine"]].get("speed_factor", 1.0)
        tg = sp.get("tg_gb_per_s", 10) / max(act_gb, 0.2) * eng
        pp = sp.get("pp_tps_1b", 25) / max(pb, 0.3) * eng
        if cell["params"].get("threads") and cell["params"]["threads"] < self.lab.machine["physical_cores"]:
            pp *= cell["params"]["threads"] / self.lab.machine["physical_cores"]
        n = est.get("n", test.cfg.get("limit") or 50)
        ptok, gtok = est.get("prompt_tok", 200), est.get("gen_tok", 50)
        if cell["params"].get("thinking") == "on":
            gtok *= est.get("thinking_mult", 6)
        if test.kind == "llama_bench":
            r = test.cfg.get("reps", 3) + 1
            return r * (test.cfg.get("pp", 512) / pp + test.cfg.get("tg", 128) / tg) + 30
        if test.kind == "kld":
            toks = test.cfg.get("chunks", 20) * test.cfg.get("ctx", 512) * len(test.cfg.get("corpora", [1]))
            return toks / pp * 1.2 + size_gb * 5
        return n * (ptok / pp + gtok / tg) + 30
