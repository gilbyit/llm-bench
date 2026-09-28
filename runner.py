"""Orchestratore: prepara, pianifica ed esegue la matrice, un modello alla volta."""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import yaml

from . import engines as engines_mod
from . import tests as tests_mod
from .artifacts import Artifacts
from .planner import Planner
from .store import Store
from .sysinfo import Monitor, detect_machine, meminfo_gb
from .tests.base import RunCtx
from .util import LabError, h, now

LAB_DIR = Path(__file__).resolve().parent


class Ctx:
    """Contesto passato ai motori (evita dipendenze circolari con Lab)."""

    def __init__(self, data_dir, machine):
        self.data_dir, self.machine = data_dir, machine


class Lab:
    def __init__(self, config: Path, machine: str | None = None, quiet=False):
        self.config_path = Path(config).resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text())
        base = self.config_path.parent
        paths = self.cfg.get("paths", {})
        self.data_dir = (base / paths.get("data", "../labdata")).resolve()
        self.bench_dir = (base / paths.get("bench_dir", "..")).resolve()
        self.db_path = (base / paths.get("db", "../results/lab.sqlite")).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        self.cfg.setdefault("timeouts", {})
        self.cfg.setdefault("defaults", {}).setdefault("params", {})
        for mid, m in self.cfg["models"].items():
            m["id"] = mid
        mid, info = detect_machine(self.cfg["machines"], machine)
        self.machine = {"id": mid, **info}
        self.power_cmd = (self.cfg.get("power") or {}).get("cmd")
        self.store = Store(self.db_path)
        self.store.save_machine(mid, self.machine)
        self.artifacts = Artifacts(self.cfg, self.store, self.data_dir)
        ctx = Ctx(self.data_dir, self.machine)
        self.engines = OrderedDict((n, engines_mod.build(n, c, ctx)) for n, c in self.cfg["engines"].items()
                                   if c.get("enabled", True))
        self.tests = OrderedDict((n, tests_mod.build(n, c or {}, self)) for n, c in self.cfg["tests"].items()
                                 if (c or {}).get("enabled", True))
        self.planner = Planner(self)
        self.main_log = self.data_dir / "logs" / f"lab-{time.strftime('%Y%m%d')}.log"
        self.main_log.parent.mkdir(parents=True, exist_ok=True)
        self._ev_cache = {}

    # --- utilità ---------------------------------------------------------------------------
    def log(self, msg: str):
        line = f"[{now()}] {msg}"
        if not self.quiet:
            print(line, flush=True)
        with open(self.main_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def expand_tests(self, spec) -> list[str]:
        suites = self.cfg.get("suites", {})
        items = [spec] if isinstance(spec, str) else list(spec)
        out = []
        for it in items:
            for t in (self.expand_tests(suites[it]) if it in suites else [it]):
                if t not in out:
                    out.append(t)
        return out

    def quantize_fn(self):
        eid = self.cfg.get("quantize_engine", "llamacpp-native")
        eng = self.engines.get(eid)
        if not eng or not eng.has_tools:
            return None
        log = self.data_dir / "logs" / "quantize.log"
        return lambda src, dst, qt, im: eng.quantize(src, dst, qt, im, log)

    def engine_version(self, eid) -> str:
        if eid not in self._ev_cache:
            try:
                v = self.engines[eid].version()
                self.store.save_engine_version(self.machine["id"], eid, v)
                self._ev_cache[eid] = v
            except LabError as e:
                self._ev_cache[eid] = e
        v = self._ev_cache[eid]
        if isinstance(v, LabError):
            raise v
        return v

    # --- preparazione ----------------------------------------------------------------------
    def prepare(self, engines=None, tests=None, models=False, update=False, sweeps=None):
        """Prepara motori, dati dei test e (a richiesta) scarica in anticipo tutti i modelli."""
        report = []
        logd = self.data_dir / "logs" / "prepare"
        for eid, eng in self.engines.items():
            if engines and eid not in engines:
                continue
            if eng.compatible(self.machine, {"id": "-", "params_b": 0}, {"format": next(iter(eng.formats))}):
                reason = eng.compatible(self.machine, {"id": "-", "params_b": 0}, {"format": next(iter(eng.formats))})
                report.append(("motore", eid, "saltato", reason))
                continue
            self.log(f"prepare motore {eid}")
            try:
                eng.prepare(logd / f"engine-{eid}.log", update=update)
                self._ev_cache.pop(eid, None)
                report.append(("motore", eid, "ok", self.engine_version(eid)))
            except Exception as e:
                report.append(("motore", eid, "ERRORE", f"{getattr(e, 'cls', type(e).__name__)}: {str(e)[:200]}"))
        for tid, t in self.tests.items():
            if tests and tid not in tests:
                continue
            self.log(f"prepare test {tid}")
            try:
                t.prepare(logd / f"test-{tid}.log")
                t._dfp = None
                report.append(("test", tid, "ok", t.cached_data_fp()[:60]))
            except Exception as e:
                report.append(("test", tid, "ERRORE", f"{getattr(e, 'cls', type(e).__name__)}: {str(e)[:200]}"))
        seen = set()
        for c in self.planner.expand(sweeps):
            if c["model"] is None or c.get("skip_reason") or (c["model"], c["quant"]) in seen:
                continue
            seen.add((c["model"], c["quant"]))
            try:
                ref = self.artifacts.resolve(c["model"], c["quant"])
                if models:
                    self.log(f"download {c['model']} {c['quant']} ({ref.size_gb} GB)")
                    self.artifacts.ensure(ref, self.log, self.quantize_fn())
                report.append(("modello", f"{c['model']} {c['quant']}", "ok", f"{ref.size_gb} GB {ref.source}"))
            except Exception as e:
                report.append(("modello", f"{c['model']} {c['quant']}", "ERRORE", str(e)[:200]))
        return report

    # --- impronta --------------------------------------------------------------------------
    def fingerprint(self, cell, ref=None, draft_ref=None) -> tuple[str | None, dict]:
        test = self.tests[cell["test"]]
        deps = test.depends_on
        res = {"test_version": test.VERSION, "test_data_fp": test.cached_data_fp()}
        if cell["engine"] and "engine" in deps:
            res["engine_version"] = self.engine_version(cell["engine"])
        if ref is not None:
            res["file_size_gb"] = ref.size_gb
            if {"model", "quant"} & set(deps):
                if not ref.sha256:
                    return None, res
                res["model_sha"] = ref.sha256
        if draft_ref is not None and ("params" in deps or "params.draft" in deps):
            res["draft_sha"] = draft_ref.sha256
        fp = h({"lk": cell["logical_key"], **{k: v for k, v in res.items() if k != "file_size_gb"}}, 24)
        return fp, res

    def is_pending(self, cell, fp, mode) -> bool:
        latest = self.store.latest(cell["logical_key"])
        if mode.get("force"):
            return True
        if mode.get("only_errors"):
            return bool(latest and latest["status"] in ("error", "invalidated"))
        if fp and self.store.ok_with_fingerprint(fp):
            return False
        if latest and latest["status"] == "error" and latest["fingerprint"] == fp and not mode.get("retry_errors"):
            return False
        return True

    # --- esecuzione ------------------------------------------------------------------------
    def run(self, cells, mode):
        n_int = self.store.mark_interrupted()
        if n_int:
            self.log(f"{n_int} esecuzioni rimaste a metà da un giro precedente segnate come 'interrupted'")
        stats = {"ok": 0, "error": 0, "skipped": 0, "done_before": 0}
        # 1) celle non eseguibili: registrate una volta, con il motivo
        runnable = []
        for c in cells:
            if c.get("skip_reason"):
                latest = self.store.latest(c["logical_key"])
                if not (latest and latest["status"] == "skipped" and latest["error_msg"] == c["skip_reason"]):
                    self.store.record(c, "skipped", error_class="incompatible", error_msg=c["skip_reason"])
                stats["skipped"] += 1
            else:
                runnable.append(c)
        # 2) test di macchina
        for c in [c for c in runnable if c["model"] is None]:
            fp, res = self.fingerprint(c)
            if self.is_pending(c, fp, mode):
                self._exec(c, fp, res, None, None, None, stats)
            else:
                stats["done_before"] += 1
        # 3) modello per modello
        by_model = OrderedDict()
        for c in runnable:
            if c["model"] is not None:
                by_model.setdefault(c["model"], []).append(c)
        for mid, mcells in by_model.items():
            self.log(f"=== modello {mid}: {len(mcells)} celle ===")
            downloaded = []
            by_quant = OrderedDict()
            for c in mcells:
                by_quant.setdefault(c["quant"], []).append(c)
            for qid, qcells in by_quant.items():
                self._run_quant(mid, qid, qcells, mode, stats, downloaded)
            if not self.cfg.get("keep_models", True):
                for ref in downloaded:
                    self.artifacts.delete_local(ref)
        return stats

    def _fail_all(self, cells, e: LabError, stats, ref=None):
        for c in cells:
            try:
                fp, res = self.fingerprint(c, ref) if ref else (None, {})
            except LabError:
                fp, res = None, {}
            self.store.record(c, "error", fp, res, error_class=e.cls, error_msg=str(e), log_path=e.log)
            stats["error"] += 1

    def _run_quant(self, mid, qid, qcells, mode, stats, downloaded):
        m = self.cfg["models"][mid]
        try:
            ref = self.artifacts.resolve(mid, qid)
        except LabError as e:
            todo = [c for c in qcells if self.is_pending(c, None, mode)
                    and not (self.store.latest(c["logical_key"]) or {"status": ""})["status"] == "ok"]
            self.log(f"  {qid}: risoluzione fallita ({e.cls}): {e}")
            self._fail_all(todo, e, stats)
            return
        # quali celle sono da fare (senza scaricare nulla)
        plan = []
        for c in qcells:
            try:
                draft_ref = self._draft_ref(c)
                fp, res = self.fingerprint(c, ref, draft_ref)
            except LabError as e:
                self.store.record(c, "error", None, {}, error_class=e.cls, error_msg=str(e), log_path=e.log)
                stats["error"] += 1
                continue
            if self.is_pending(c, fp, mode):
                plan.append((c, fp, res, draft_ref))
            else:
                stats["done_before"] += 1
        if not plan:
            return
        max_gb = m.get("max_model_gb") or self.machine.get("max_model_gb", 12)
        if ref.size_gb and ref.size_gb > max_gb:
            msg = f"file da {ref.size_gb} GB oltre il limite di {max_gb} GB per {self.machine['id']}"
            for c, fp, res, _ in plan:
                self.store.record(c, "skipped", fp, res, error_class="ram", error_msg=msg)
                stats["skipped"] += 1
            return
        if mode.get("dry"):
            for c, *_ in plan:
                self.log(f"  [dry] {qid} {c['engine']} {c['test']} {json.dumps(c['params'])}")
            return
        self.log(f"  {qid}: {len(plan)} test da eseguire, file {ref.size_gb} GB")
        try:
            had = ref.local.exists() if ref.local else True
            self.artifacts.ensure(ref, self.log, self.quantize_fn())
            if not had:
                downloaded.append(ref)
            for _, _, _, d in plan:
                if d is not None:
                    self.artifacts.ensure(d, self.log, self.quantize_fn())
        except LabError as e:
            self._fail_all([p[0] for p in plan], e, stats, ref)
            return
        # recompute fingerprint for locally made quants (sha known only now)
        plan = [(c, *self.fingerprint(c, ref, d), d) if fp is None else (c, fp, res, d) for c, fp, res, d in plan]
        by_engine = OrderedDict()
        for p in plan:
            by_engine.setdefault(p[0]["engine"], []).append(p)
        for eid, eplan in by_engine.items():
            eng = self.engines[eid]
            groups = OrderedDict()
            for p in eplan:
                groups.setdefault(json.dumps(p[0]["params"], sort_keys=True), []).append(p)
            for pkey, gplan in groups.items():
                params = json.loads(pkey)
                offline = [p for p in gplan if not self.tests[p[0]["test"]].needs_server]
                online = [p for p in gplan if self.tests[p[0]["test"]].needs_server]
                for c, fp, res, d in offline:
                    self._exec(c, fp, res, eng, ref, None, stats, params=params)
                if online:
                    self._run_server_group(eng, ref, m, params, online, stats)

    def _draft_ref(self, cell):
        d = cell["params"].get("draft")
        if not d:
            return None
        dm, dq = d.split(":", 1)
        return self.artifacts.resolve(dm, dq)

    def _log_dir(self, c) -> Path:
        parts = [self.machine["id"], c["model"] or "_macchina", c["quant"] or "-", c["engine"] or "-"]
        return self.data_dir / "logs" / "runs" / Path(*parts)

    def _start_server(self, eng, ref, m, params, draft, log):
        t = self.cfg["timeouts"].get("server_start_s", 900)
        avail = meminfo_gb()["available_gb"]
        if ref.size_gb and avail and ref.size_gb > avail * 0.95:
            raise LabError("oom", f"RAM disponibile {avail} GB < file {ref.size_gb} GB")
        srv = eng.start(ref, params, m, log, draft)
        try:
            eng.wait_ready(srv, t)
        except BaseException:
            srv.stop()
            raise
        return srv

    def _run_server_group(self, eng, ref, m, params, plan, stats):
        draft = plan[0][3]
        slog = self._log_dir(plan[0][0]) / f"server-{h(params, 8)}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        self.log(f"  avvio {eng.name} {' '.join(f'{k}={v}' for k, v in params.items() if v is not None)}")
        try:
            srv = self._start_server(eng, ref, m, params, draft, slog)
        except LabError as e:
            self.log(f"  avvio fallito ({e.cls}): {str(e)[:200]}")
            for c, fp, res, _ in plan:
                self.store.record(c, "error", fp, res, error_class=e.cls, error_msg=str(e), log_path=str(slog))
                stats["error"] += 1
            return
        try:
            for c, fp, res, _ in plan:
                if not srv.alive():
                    self.log("  il server è morto: riavvio")
                    try:
                        srv.stop()
                        srv = self._start_server(eng, ref, m, params, draft, slog)
                    except LabError as e:
                        self.store.record(c, "error", fp, res, error_class=e.cls, error_msg=f"riavvio fallito: {e}",
                                          log_path=str(slog))
                        stats["error"] += 1
                        continue
                self._exec(c, fp, res, eng, ref, srv, stats, params=params, server_log=slog)
        finally:
            srv.stop()

    def _exec(self, c, fp, res, eng, ref, srv, stats, params=None, server_log=None):
        test = self.tests[c["test"]]
        rid = self.store.start_run(c, fp, res)
        log = self._log_dir(c) / f"{c['test']}-{c['logical_key'][:10]}-{rid}.log"
        work = self.data_dir / "work" / str(rid)
        work.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        m = self.cfg["models"].get(c["model"]) if c["model"] else None
        rt = RunCtx(c, eng, ref, m, params or c["params"], log, work, srv, self)
        label = f"{c['model'] or 'macchina'} {c['quant'] or ''} {c['engine'] or ''} {c['test']}"
        self.log(f"    ▶ {label} (run {rid})")
        t0 = time.monotonic()
        pid_fn = srv.pid_fn if srv else (lambda: None)
        mon = Monitor(pid_fn, self.power_cmd, (self.cfg.get("power") or {}).get("interval_s", 1.0))
        extra = {"server_log": str(server_log)} if server_log else {}
        try:
            with mon:
                result = test.run(rt)
            dur = time.monotonic() - t0
            self.store.finish_run(rid, "ok", log_path=str(log), duration_s=round(dur, 1), monitor=mon.snapshot(),
                                  metrics=result.metrics, samples=result.samples, extra={**extra, **result.extra})
            stats["ok"] += 1
            key = next((k for k in result.metrics if k.endswith(("tutto_giusto_pct", "acc_pct", "tg_tps", "kld_mean",
                                                                    "compliance_pct", "f1"))), None)
            show = f"{key}={result.metrics[key][0]:.3g}" if key and result.metrics[key][0] is not None else ""
            self.log(f"    ✓ {dur:.0f}s {show}")
        except KeyboardInterrupt:
            self.store.finish_run(rid, "error", error_class="interrupted", error_msg="interrotto dall'utente",
                                  log_path=str(log), duration_s=round(time.monotonic() - t0, 1))
            raise
        except LabError as e:
            self.store.finish_run(rid, "error", error_class=e.cls, error_msg=str(e), log_path=e.log or str(log),
                                  duration_s=round(time.monotonic() - t0, 1), monitor=mon.snapshot(), extra=extra)
            stats["error"] += 1
            self.log(f"    ✗ {e.cls}: {str(e)[:200]}")
        except Exception as e:
            tb = traceback.format_exc()
            with open(log, "a") as f:
                f.write(tb)
            self.store.finish_run(rid, "error", error_class="bug", error_msg=f"{type(e).__name__}: {e}\n{tb[-1500:]}",
                                  log_path=str(log), duration_s=round(time.monotonic() - t0, 1), extra=extra)
            stats["error"] += 1
            self.log(f"    ✗ bug: {type(e).__name__}: {e}")
        finally:
            if not self.cfg.get("keep_work", False):
                shutil.rmtree(work, ignore_errors=True)
