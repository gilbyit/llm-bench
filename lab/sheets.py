"""Copia dei risultati su un Google Sheet, a mano a mano che i test finiscono.

Scrive con un account di servizio Google (libreria gspread): il foglio è condiviso con l'email
dell'account come Editor, e la chiave JSON sta sulla macchina fuori da git. Nessun altro accesso.

Tre schede, create da sole se mancano:
  Stato      una riga per macchina e sweep: fatte, da fare, errori, stima; più la riga "in corso"
  Risultati  una riga per cella (ultimo tentativo), aggiornata in place
  Log        le righe del log del laboratorio, in coda (tiene le ultime 3000)

Le colonne nuove si aggiungono in fondo da sole. L'invio avviene in un thread a parte e non blocca
mai i test: se la rete o Google non rispondono le righe si riprovano per un po' e poi si lasciano
perdere. `lab.sh sync` rimanda tutto dal database, quindi niente va perso davvero.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from .util import brief, now

# Metrica mostrata nella colonna "metrica/valore", in ordine di preferenza (suffisso del nome)
PRIMARY = ("tutto_giusto_pct", "acc_pct", "compliance_pct", "prompt_acc_pct", "f1", "tg_tps", "kld_mean")
ORDER = ("Stato", "Risultati", "Log")
LOG_MAX = 3000


def _num(v):
    return round(v, 4) if isinstance(v, float) else v


class _GSheet:
    """Upsert e append su un foglio Google via gspread. Tutto RAW: '38/38' resta testo, non una data."""

    def __init__(self, sheet_id: str, cred_file: str):
        self.sheet_id, self.cred_file = sheet_id, cred_file
        self._sh = None
        self._ws = {}
        self._log_rows = None

    def _book(self):
        if self._sh is None:
            try:
                import gspread
            except ImportError:
                raise RuntimeError("manca gspread: lab/.venv/bin/pip install gspread")
            self._sh = gspread.service_account(filename=self.cred_file).open_by_key(self.sheet_id)
        return self._sh

    def ws(self, name):
        if name not in self._ws:
            import gspread
            sh = self._book()
            try:
                w = sh.worksheet(name)
            except gspread.WorksheetNotFound:
                idx = ORDER.index(name) if name in ORDER else None
                w = sh.add_worksheet(title=name, rows=200, cols=10, index=idx)
                w.freeze(rows=1)
            self._ws[name] = w
        return self._ws[name]

    def _header(self, w, head, rows, key):
        fresh = []
        if key and key not in head:
            fresh.append(key)
        for r in rows:
            for k in r:
                if k not in head and k not in fresh:
                    fresh.append(k)
        if fresh:
            head = head + fresh
            if len(head) > w.col_count:
                w.add_cols(len(head) - w.col_count)
            w.update(range_name="A1", values=[head], value_input_option="RAW")
            w.format("1:1", {"textFormat": {"bold": True}})
        return head

    def upsert(self, name, key, rows):
        from gspread.utils import rowcol_to_a1
        w = self.ws(name)
        vals = w.get_all_values(value_render_option="UNFORMATTED_VALUE")   # i numeri restano numeri
        head = self._header(w, vals[0] if vals else [], rows, key)
        data = [r + [""] * (len(head) - len(r)) for r in vals[1:]]
        kc = head.index(key)
        index = {r[kc]: i for i, r in enumerate(data) if r[kc] != ""}
        touched = []
        for obj in rows:
            k = str(obj[key])
            i = index.get(k)
            if i is None:
                i = len(data)
                data.append([""] * len(head))
                index[k] = i
            for c, h in enumerate(head):
                if h in obj:
                    data[i][c] = obj[h]
            touched.append(i)
        need = len(data) + 1
        if need > w.row_count:
            w.add_rows(need - w.row_count + 100)
        last = rowcol_to_a1(1, len(head)).rstrip("0123456789")
        upd = [{"range": f"A{i + 2}:{last}{i + 2}", "values": [data[i]]} for i in sorted(set(touched))]
        w.batch_update(upd, value_input_option="RAW")

    def delete(self, name, key, keys, where=None):
        """Cancella le righe la cui colonna `key` è in `keys` (e che rispettano `where`: {colonna: valore})."""
        w = self.ws(name)
        vals = w.get_all_values()
        if not vals or key not in vals[0]:
            return 0
        head, keys = vals[0], {str(k) for k in keys}
        kc = head.index(key)
        cond = [(head.index(c), str(v)) for c, v in (where or {}).items() if c in head]
        hit = [i + 2 for i, r in enumerate(vals[1:])
               if len(r) > kc and r[kc] in keys and all(len(r) > c and r[c] == v for c, v in cond)]
        # dal fondo, a blocchi contigui: gli indici sopra non si spostano e si usano poche chiamate
        end = None
        for i in sorted(hit, reverse=True) + [None]:
            if end is not None and (i is None or i != start - 1):
                w.delete_rows(start, end)
                end = None
            if i is not None:
                if end is None:
                    end = i
                start = i
        return len(hit)

    def append(self, name, rows):
        w = self.ws(name)
        vals_head = w.row_values(1)
        head = self._header(w, vals_head, rows, None)
        w.append_rows([[r.get(h, "") for h in head] for r in rows], value_input_option="RAW",
                      table_range="A1")
        if self._log_rows is None:
            self._log_rows = len(w.col_values(1)) - 1
        else:
            self._log_rows += len(rows)
        if self._log_rows > LOG_MAX + 200:
            w.delete_rows(2, self._log_rows - LOG_MAX + 1)
            self._log_rows = LOG_MAX


class SheetSync:
    def __init__(self, cfg: dict, machine_id: str, log_fn=None, base_dir: Path | None = None):
        cfg = cfg or {}
        sheet_id = os.environ.get(cfg.get("sheet_id_env", "LAB_SHEET_ID")) or cfg.get("sheet_id")
        cred = os.environ.get(cfg.get("credentials_env", "LAB_SHEET_CREDENTIALS")) or cfg.get("credentials")
        if cred and base_dir and not Path(cred).is_absolute():
            cred = str((base_dir / cred).resolve())
        self.enabled = bool(sheet_id and cred) and cfg.get("enabled", True)
        if self.enabled and not Path(cred).exists():
            (log_fn or print)(f"[sheets] chiave {cred} non trovata: copia sul foglio disattivata")
            self.enabled = False
        self.gs = _GSheet(sheet_id, cred) if self.enabled else None
        self.machine = machine_id
        self.log_lines = cfg.get("log", True)
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
    def _post(self, payload: dict):
        if payload["op"] == "upsert":
            self.gs.upsert(payload["sheet"], payload["key"], payload["rows"])
        elif payload["op"] == "delete":
            self.gs.delete(payload["sheet"], payload["key"], payload["keys"], payload.get("where"))
        else:
            self.gs.append(payload["sheet"], payload["rows"])

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
            except Exception as e:  # rete, quota Google (60 scritture/min), foglio non condiviso...
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
def _note(db, r) -> str:
    """Dettaglio che la metrica principale nasconde, ricavato dai campioni già salvati (vale anche
    per i risultati vecchi: basta `lab.sh sync`). Serve soprattutto a segnalare i punteggi che non
    misurano il modello ma un difetto della prova."""
    if r["status"] != "ok":
        return ""
    S = []
    for s in db.execute("SELECT correct, error, extra_json FROM samples WHERE run_id=?", (r["id"],)):
        try:
            ex = json.loads(s["extra_json"]) if s["extra_json"] else {}
        except (TypeError, ValueError):
            ex = {}
        if not s["error"]:
            S.append((s["correct"] or 0, ex))
    if not S:
        return ""
    test = r["test"]
    if test.startswith("bfcl"):
        simple = [(c, ex) for c, ex in S if not str(ex.get("cat", "")).endswith("irrelevance")]
        irr = [(c, ex) for c, ex in S if str(ex.get("cat", "")).endswith("irrelevance")]
        note = (f"chiamate semplici {int(sum(c for c, _ in simple))}/{len(simple)}, "
                f"astensione {int(sum(c for c, _ in irr))}/{len(irr)}")
        if simple and not any(ex.get("n_calls") for _, ex in simple):
            note = ("NON VALIDO: nessuna chiamata in tutto il test, il template non passa le funzioni "
                    "al modello. " + note)
        return note
    if test.startswith("belebele"):
        def pred(ex):
            if "pred" in ex:
                return ex["pred"]
            m = re.search(r"\b([ABCD])\b", str(ex.get("answer", "")).upper())
            return m.group(1) if m else None
        preds = [pred(ex) for _, ex in S]
        missing = sum(1 for p in preds if p is None)
        top, n_top = Counter(p for p in preds if p).most_common(1)[0] if any(preds) else (None, 0)
        notes = []
        if missing * 4 >= len(S):
            notes.append(f"NON VALIDO: {missing}/{len(S)} risposte senza lettera (ragionamento o testo tagliato)")
        elif missing:
            notes.append(f"{missing}/{len(S)} risposte senza lettera")
        if top and n_top * 2 > len(S):
            notes.append(f"risponde {top} in {n_top} casi su {len(S)}")
        return "; ".join(notes)
    if test.startswith("json"):
        return f"JSON valido {sum(1 for _, ex in S if ex.get('valid'))}/{len(S)}"
    return ""


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
    err = f"{r['error_class']}: {brief(r['error_msg'] or '')}" if r["error_class"] else ""
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
        "ram_gb": round(r["peak_rss_mb"] / 1024, 2) if r["peak_rss_mb"] else "",
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
    try:
        row["note"] = _note(db, r)
    except Exception as e:   # una nota non deve mai far perdere la riga
        row["note"] = f"(nota non calcolata: {type(e).__name__})"
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
