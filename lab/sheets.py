"""Copia dei risultati su un Google Sheet, a mano a mano che i test finiscono.

Il foglio ha uno script Apps Script (lab/sheets_apps_script.gs) pubblicato come app web: riceve
righe in JSON e le scrive. Qui non serve nessuna libreria Google né un account di servizio: solo
l'URL dell'app web e un token condiviso.

Tre schede:
  Risultati  una riga per cella (ultimo tentativo), aggiornata in place
  Stato      una riga per macchina e sweep: fatte, da fare, errori, stima; più la riga "in corso"
  Log        le righe del log del laboratorio, in coda (lo script tiene le ultime 3000)

L'invio avviene in un thread a parte e non blocca mai i test: se la rete o Google non rispondono,
le righe si riprovano per un po' e poi si lasciano perdere. `lab.sh sync` rimanda tutto dal
database, quindi niente va perso davvero.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
import urllib.request
from collections import Counter, defaultdict

from .util import now

# Metrica mostrata nella colonna "risultato", in ordine di preferenza (suffisso del nome)
PRIMARY = ("tutto_giusto_pct", "acc_pct", "compliance_pct", "prompt_acc_pct", "f1", "tg_tps", "kld_mean")


def _num(v):
    return round(v, 4) if isinstance(v, float) else v


class SheetSync:
    def __init__(self, cfg: dict, machine_id: str, log_fn=None):
        cfg = cfg or {}
        self.url = os.environ.get(cfg.get("url_env", "LAB_SHEETS_URL")) or cfg.get("url")
        self.token = os.environ.get(cfg.get("token_env", "LAB_SHEETS_TOKEN")) or cfg.get("token")
        self.enabled = bool(self.url and self.token) and cfg.get("enabled", True)
        self.machine = machine_id
        self.log_lines = cfg.get("log", True)
        self.timeout = cfg.get("timeout_s", 30)
        self._warn = log_fn or print
        self._q: queue.Queue = queue.Queue()
        self._logbuf: list = []
        self._lock = threading.Lock()
        self._fails = 0
        self._th = None
        if self.enabled:
            self._th = threading.Thread(target=self._worker, name="sheets", daemon=True)
            self._th.start()
            atexit.register(self.close)

    # --- API usata dal laboratorio ---------------------------------------------------------
    def upsert(self, sheet: str, key: str, rows: list[dict]):
        if self.enabled and rows:
            self._q.put({"op": "upsert", "sheet": sheet, "key": key, "rows": rows})

    def log(self, line: str):
        if self.enabled and self.log_lines:
            with self._lock:
                self._logbuf.append({"ora": now(), "macchina": self.machine, "riga": line[:500]})

    def close(self, wait_s: float = 20):
        """Svuota la coda prima di uscire (al massimo wait_s secondi)."""
        if not self.enabled or not self._th:
            return
        self._q.put(None)
        self._th.join(wait_s)

    # --- invio -----------------------------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        body = json.dumps({"token": self.token, **payload}, default=str).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        # Apps Script risponde con un 302: urllib lo segue con una GET, che è quello che serve
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            txt = r.read().decode("utf-8", "replace")
        try:
            res = json.loads(txt)
        except json.JSONDecodeError:
            raise RuntimeError(f"risposta non JSON (URL dell'app web giusto?): {txt[:120]}")
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "errore sconosciuto"))
        return res

    def _flush_log(self):
        with self._lock:
            buf, self._logbuf = self._logbuf, []
        if buf:
            self._send({"op": "append", "sheet": "Log", "rows": buf})

    def _send(self, payload, tries=4):
        for i in range(tries):
            try:
                self._post(payload)
                self._fails = 0
                return True
            except Exception as e:  # rete, quota Google, script non pubblicato...
                err = e
                time.sleep(min(60, 5 * 2 ** i))
        self._fails += 1
        if self._fails in (1, 10, 100):
            self._warn(f"[sheets] invio fallito ({err}); i dati restano nel database, `lab.sh sync` li rimanda")
        return False

    def _worker(self):
        last_log = 0.0
        while True:
            try:
                item = self._q.get(timeout=15)
            except queue.Empty:
                item = "tick"
            if item is None:
                self._flush_log()
                return
            if item != "tick":
                # unisce gli upsert consecutivi sulla stessa scheda (es. tante celle saltate)
                batch = [item]
                while True:
                    try:
                        nxt = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        self._q.put(None)
                        break
                    batch.append(nxt)
                merged = defaultdict(list)
                for b in batch:
                    merged[(b["sheet"], b["key"])] += b["rows"]
                for (sheet, key), rows in merged.items():
                    for i in range(0, len(rows), 500):
                        self._send({"op": "upsert", "sheet": sheet, "key": key, "rows": rows[i:i + 500]})
            if time.monotonic() - last_log > 30:
                self._flush_log()
                last_log = time.monotonic()


# --- conversione dal database alle righe del foglio ---------------------------------------
def result_row(db, run_id: int) -> dict | None:
    r = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if r is None:
        return None
    p = json.loads(r["params_json"] or "{}")
    mets = {m["name"]: m["value"] for m in db.execute("SELECT name, value FROM metrics WHERE run_id=?", (run_id,))}
    ci = db.execute("SELECT COUNT(correct) n, SUM(correct) k FROM samples WHERE run_id=? AND correct IS NOT NULL",
                    (run_id,)).fetchone()

    def pick(*suffixes):
        for suf in suffixes:
            for k, v in mets.items():
                if k.endswith(suf):
                    return k, v
        return None, None

    pname, pval = pick(*PRIMARY)
    wall = pick("wall_med_s")[1]
    pp = pick("pp_tps", "prompt_tps_med")[1]
    tg = pick("tg_tps", "gen_tps_med")[1]
    err = f"{r['error_class']}: {(r['error_msg'] or '').splitlines()[0][:200]}" if r["error_class"] else ""
    row = {
        "cella": r["logical_key"],
        "aggiornato": r["finished_at"] or now(),
        "macchina": r["machine"],
        "modello": r["model"] or "(macchina)",
        "quant": r["quant"] or "",
        "motore": r["engine"] or "",
        "test": r["test"],
        "stato": r["status"],
        "metrica": pname or "",
        "valore": _num(pval),
        "giusti": f"{int(ci['k'])}/{ci['n']}" if ci["n"] else "",
        "ci95": _wilson(ci["k"] or 0, ci["n"]) if ci["n"] else "",
        "wall_med_s": _num(wall),
        "pp_tps": _num(pp),
        "tg_tps": _num(tg),
        "kld_it": _num(mets.get("wiki_it.kld_mean")),
        "kld_en": _num(mets.get("wiki_en.kld_mean")),
        "durata_min": round(r["duration_s"] / 60, 1) if r["duration_s"] else "",
        "file_gb": r["file_size_gb"] or "",
        "threads": p.get("threads"),
        "fa": p.get("fa"),
        "kv": p.get("kv"),
        "thinking": p.get("thinking"),
        "cache": p.get("prompt_cache"),
        "altri_param": ", ".join(f"{k}={v}" for k, v in p.items() if v is not None and k not in
                                 ("threads", "fa", "kv", "thinking", "prompt_cache", "ctx", "batch", "ubatch")),
        "errore": err,
        "sweep": r["sweep"],
        "motore_versione": r["engine_version"] or "",
        "tentativo": r["attempt"],
        "run_id": r["id"],
    }
    return {k: ("" if v is None else v) for k, v in row.items()}


def _wilson(k, n, z=1.96):
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * (c - m):.0f}-{100 * (c + m):.0f}%"


def status_rows(lab, cells=None) -> list[dict]:
    """Una riga per (macchina, sweep): come `plan`, senza la verifica delle impronte."""
    cells = cells if cells is not None else lab.planner.expand()
    cnt, est = defaultdict(Counter), defaultdict(float)
    for c in cells:
        sw = c["sweep"].split(",")[0]
        if c.get("skip_reason"):
            cnt[sw]["incompatibili"] += 1
            continue
        latest = lab.store.latest(c["logical_key"])
        st = latest["status"] if latest else "nuova"
        if st == "ok":
            cnt[sw]["fatte"] += 1
        elif st == "error":
            cnt[sw]["errori"] += 1
        else:
            cnt[sw]["da_fare"] += 1
            est[sw] += lab.planner.estimate_s(c)
    rows = []
    for sw, c in cnt.items():
        todo = c["fatte"] + c["errori"] + c["da_fare"]
        rows.append({"chiave": f"{lab.machine['id']}|{sw}", "macchina": lab.machine["id"], "sweep": sw,
                     "fatte": c["fatte"], "da_fare": c["da_fare"], "errori": c["errori"],
                     "incompatibili": c["incompatibili"],
                     "completamento": f"{100 * c['fatte'] / todo:.0f}%" if todo else "",
                     "stima_ore_rimaste": round(est[sw] / 3600, 1), "aggiornato": now()})
    return rows


def running_row(lab, text: str) -> dict:
    return {"chiave": f"{lab.machine['id']}|IN CORSO", "macchina": lab.machine["id"], "sweep": "▶ in corso",
            "attività": text, "aggiornato": now()}
