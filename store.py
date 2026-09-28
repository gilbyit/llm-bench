"""Archivio dei risultati in SQLite.

Ogni esecuzione di un test su una combinazione (cella) è una riga di `runs`. Le righe non si
sovrascrivono mai: un nuovo tentativo aggiunge una riga, così lo storico resta leggibile.
La vista `v_latest` dà l'ultimo tentativo per ciascuna cella logica.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .util import now

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
  id TEXT PRIMARY KEY, info_json TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  logical_key TEXT NOT NULL,      -- cella: test + parti della configurazione da cui dipende
  fingerprint TEXT,               -- logical_key + versioni risolte (sha modello, versione motore, dati test)
  sweep TEXT, machine TEXT, model TEXT, quant TEXT, engine TEXT,
  params_json TEXT, test TEXT, test_opts_json TEXT,
  test_version TEXT, test_data_fp TEXT, engine_version TEXT, model_sha TEXT,
  status TEXT NOT NULL,           -- running | ok | error | skipped | invalidated
  error_class TEXT, error_msg TEXT, log_path TEXT,
  started_at TEXT, finished_at TEXT, duration_s REAL,
  peak_rss_mb REAL, energy_j REAL, avg_power_w REAL,
  file_size_gb REAL,
  attempt INTEGER DEFAULT 1,
  extra_json TEXT
);
CREATE INDEX IF NOT EXISTS runs_lk ON runs(logical_key);
CREATE INDEX IF NOT EXISTS runs_fp ON runs(fingerprint);
CREATE TABLE IF NOT EXISTS metrics (
  run_id INTEGER NOT NULL REFERENCES runs(id), name TEXT NOT NULL, value REAL, unit TEXT,
  PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS samples (
  run_id INTEGER NOT NULL REFERENCES runs(id), case_id TEXT, rep INTEGER,
  correct REAL, wall_s REAL, ttft_s REAL, prompt_tokens INTEGER, gen_tokens INTEGER,
  prompt_tps REAL, gen_tps REAL, error TEXT, extra_json TEXT
);
CREATE INDEX IF NOT EXISTS samples_run ON samples(run_id);
CREATE TABLE IF NOT EXISTS artifacts (
  key TEXT PRIMARY KEY, kind TEXT, source TEXT, path TEXT, size_bytes INTEGER, sha256 TEXT,
  meta_json TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS engine_versions (
  machine TEXT, engine TEXT, version TEXT, info_json TEXT, checked_at TEXT,
  PRIMARY KEY (machine, engine)
);

DROP VIEW IF EXISTS v_latest;
CREATE VIEW v_latest AS
  SELECT r.* FROM runs r
  JOIN (SELECT logical_key, MAX(id) AS id FROM runs WHERE status != 'running' GROUP BY logical_key) l
    ON r.id = l.id;

DROP VIEW IF EXISTS v_results;
CREATE VIEW v_results AS
  SELECT r.id AS run_id, r.machine, r.model, r.quant, r.engine, r.params_json, r.test, r.sweep,
         r.status, r.engine_version, r.model_sha, r.file_size_gb, r.peak_rss_mb, r.energy_j,
         r.duration_s, r.finished_at, m.name AS metric, m.value, m.unit
  FROM v_latest r LEFT JOIN metrics m ON m.run_id = r.id;

DROP VIEW IF EXISTS v_errors;
CREATE VIEW v_errors AS
  SELECT id, machine, model, quant, engine, params_json, test, error_class,
         substr(error_msg, 1, 300) AS error_msg, log_path, finished_at
  FROM v_latest WHERE status = 'error';
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # --- macchine, artefatti, motori -------------------------------------------------------
    def save_machine(self, mid: str, info: dict):
        self.db.execute("INSERT OR REPLACE INTO machines VALUES (?,?,?)", (mid, json.dumps(info), now()))

    def get_artifact(self, key: str):
        return self.db.execute("SELECT * FROM artifacts WHERE key=?", (key,)).fetchone()

    def save_artifact(self, key, kind, source, path, size, sha, meta=None):
        self.db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?)",
                        (key, kind, source, str(path), size, sha, json.dumps(meta or {}), now()))

    def save_engine_version(self, machine, engine, version, info=None):
        self.db.execute("INSERT OR REPLACE INTO engine_versions VALUES (?,?,?,?,?)",
                        (machine, engine, version, json.dumps(info or {}), now()))

    def get_engine_version(self, machine, engine):
        r = self.db.execute("SELECT version FROM engine_versions WHERE machine=? AND engine=?",
                            (machine, engine)).fetchone()
        return r["version"] if r else None

    # --- esecuzioni ------------------------------------------------------------------------
    def latest(self, logical_key: str):
        return self.db.execute(
            "SELECT * FROM runs WHERE logical_key=? AND status!='running' ORDER BY id DESC LIMIT 1",
            (logical_key,)).fetchone()

    def ok_with_fingerprint(self, fp: str) -> bool:
        return self.db.execute("SELECT 1 FROM runs WHERE fingerprint=? AND status='ok' LIMIT 1",
                               (fp,)).fetchone() is not None

    def attempts(self, logical_key: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM runs WHERE logical_key=? AND status IN ('ok','error')",
                               (logical_key,)).fetchone()[0]

    def start_run(self, cell: dict, fp: str | None, resolved: dict) -> int:
        cur = self.db.execute(
            """INSERT INTO runs (logical_key, fingerprint, sweep, machine, model, quant, engine, params_json,
               test, test_opts_json, test_version, test_data_fp, engine_version, model_sha, status,
               started_at, file_size_gb, attempt)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'running', ?, ?, ?)""",
            (cell["logical_key"], fp, cell["sweep"], cell["machine"], cell["model"], cell["quant"],
             cell["engine"], json.dumps(cell["params"], sort_keys=True), cell["test"],
             json.dumps(cell.get("test_opts") or {}, sort_keys=True), resolved.get("test_version"),
             resolved.get("test_data_fp"), resolved.get("engine_version"), resolved.get("model_sha"),
             now(), resolved.get("file_size_gb"), self.attempts(cell["logical_key"]) + 1))
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str, *, error_class=None, error_msg=None, log_path=None,
                   duration_s=None, monitor=None, metrics=None, samples=None, extra=None):
        mon = monitor or {}
        self.db.execute("BEGIN")
        try:
            self.db.execute(
                """UPDATE runs SET status=?, error_class=?, error_msg=?, log_path=?, finished_at=?,
                   duration_s=?, peak_rss_mb=?, energy_j=?, avg_power_w=?, extra_json=? WHERE id=?""",
                (status, error_class, (error_msg or "")[:4000] or None, log_path, now(), duration_s,
                 mon.get("peak_rss_mb"), mon.get("energy_j"), mon.get("avg_power_w"),
                 json.dumps(extra) if extra else None, run_id))
            for name, v in (metrics or {}).items():
                value, unit = v if isinstance(v, tuple) else (v, None)
                if value is not None:
                    self.db.execute("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?)",
                                    (run_id, name, float(value), unit))
            for s in samples or []:
                known = {k: s.get(k) for k in ("case_id", "rep", "correct", "wall_s", "ttft_s", "prompt_tokens",
                                                "gen_tokens", "prompt_tps", "gen_tps", "error")}
                rest = {k: v for k, v in s.items() if k not in known}
                self.db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                (run_id, *known.values(), json.dumps(rest, default=str) if rest else None))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def record(self, cell, status, fp=None, resolved=None, **kw) -> int:
        rid = self.start_run(cell, fp, resolved or {})
        self.finish_run(rid, status, **kw)
        return rid

    def mark_interrupted(self) -> int:
        cur = self.db.execute("UPDATE runs SET status='error', error_class='interrupted', "
                              "error_msg='esecuzione interrotta', finished_at=? WHERE status='running'", (now(),))
        return cur.rowcount

    def invalidate(self, where: str, args: list) -> int:
        """Segna come 'invalidated' l'ultimo risultato delle celle che corrispondono al filtro."""
        keys = [r[0] for r in self.db.execute(f"SELECT logical_key FROM v_latest WHERE {where}", args)]
        for k in keys:
            self.db.execute("UPDATE runs SET status='invalidated' WHERE logical_key=? "
                            "AND status IN ('ok','error','skipped')", (k,))
        return len(keys)
