"""Riga di comando: python3 -m lab <comando> [opzioni]

  prepare     scarica/compila i motori, scarica i dati dei test, verifica (o scarica) i modelli
  plan        mostra quante celle ci sono, quante già fatte, quante da fare, e una stima dei tempi
  run         esegue le celle da fare, modello per modello
  retry       ripete solo le celle finite in errore (filtrabili)
  status      riepilogo per stato, test e modello
  errors      elenco degli errori con classe e log
  invalidate  segna come da rifare le celle che corrispondono ai filtri
  export      esporta CSV (run, metriche lunghe e larghe, campioni, errori)
  sync        rimanda al Google Sheet tutti i risultati e lo stato (vedi `sheets:` in matrix.yaml)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .runner import LAB_DIR, Lab


def add_filters(p):
    p.add_argument("--model", action="append", help="filtra per modello (ripetibile)")
    p.add_argument("--quant", action="append")
    p.add_argument("--engine", action="append")
    p.add_argument("--test", action="append")
    p.add_argument("--sweep", action="append", help="usa solo questi sweep (anche se disabilitati)")
    p.add_argument("--param", action="append", default=[], help="filtra per parametro, es. threads=2")


def match(cell, a) -> bool:
    for key in ("model", "quant", "engine", "test"):
        vals = getattr(a, key, None)
        if vals and cell.get(key) not in vals:
            return False
    for kv in getattr(a, "param", []) or []:
        k, _, v = kv.partition("=")
        if str(cell["params"].get(k)) != v:
            return False
    return True


def where_clause(a):
    w, args = ["1=1"], []
    for key in ("model", "quant", "engine", "test"):
        vals = getattr(a, key, None)
        if vals:
            w.append(f"{key} IN ({','.join('?' * len(vals))})")
            args += vals
    for kv in getattr(a, "param", []) or []:
        k, _, v = kv.partition("=")
        w.append("json_extract(params_json, ?) = ?")
        args += [f"$.{k}", int(v) if v.isdigit() else v]
    if getattr(a, "cls", None):
        w.append("error_class = ?")
        args.append(a.cls)
    return " AND ".join(w), args


def fmt_h(s):
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


def cmd_plan(lab, a):
    cells = [c for c in lab.planner.expand(a.sweep, include_disabled=bool(a.sweep)) if match(c, a)]
    rows = defaultdict(Counter)
    est = defaultdict(float)
    reasons = Counter()
    for c in cells:
        sw = c["sweep"].split(",")[0]
        if c.get("skip_reason"):
            rows[sw]["incompatibili"] += 1
            reasons[c["skip_reason"]] += 1
            continue
        latest = lab.store.latest(c["logical_key"])
        st = latest["status"] if latest else "nuova"
        if st == "ok" and not a.check:
            rows[sw]["fatte"] += 1
            continue
        if a.check and latest:
            try:
                ref = lab.artifacts.resolve(c["model"], c["quant"]) if c["model"] else None
                fp, res = lab.fingerprint(c, ref)
            except Exception:
                fp, res = None, {}
            if st == "ok" and fp and lab.store.ok_with_fingerprint(fp):
                rows[sw]["fatte"] += 1
                continue
            if st == "ok":
                why = [k for k in ("test_version", "test_data_fp", "engine_version", "model_sha")
                       if res.get(k) and latest[k] != res.get(k)]
                st = "cambiata:" + "+".join(why or ["?"])
        if st == "error":
            rows[sw]["errori (retry per ripeterli)"] += 1
            continue
        rows[sw][st if st != "nuova" else "da fare"] += 1
        est[sw] += lab.planner.estimate_s(c)
    print(f"Macchina: {lab.machine['id']} ({lab.machine['cpu_model']}, AVX2={'sì' if lab.machine['avx2'] else 'no'})\n")
    total = 0
    for sw, cnt in rows.items():
        total += est[sw]
        parts = ", ".join(f"{k} {v}" for k, v in sorted(cnt.items()))
        print(f"{sw:22} {parts:70} stima {fmt_h(est[sw])}")
    print(f"\nTotale celle: {len(cells)}   stima da fare: {fmt_h(total)} ({total / 86400:.1f} giorni)")
    if a.reasons and reasons:
        print("\nMotivi di incompatibilità:")
        for r, n in reasons.most_common():
            print(f"  {n:5}  {r}")
    if a.list:
        for c in cells:
            if not c.get("skip_reason"):
                print(f"  {c['model']} {c['quant']} {c['engine']} {c['test']} {json.dumps(c['params'])}")


def cmd_run(lab, a, only_errors=False):
    cells = [c for c in lab.planner.expand(a.sweep, include_disabled=bool(a.sweep)) if match(c, a)]
    mode = {"force": a.force, "retry_errors": getattr(a, "retry_errors", False) or only_errors,
            "only_errors": only_errors, "dry": a.dry}
    if only_errors and getattr(a, "cls", None):
        keep = {r["logical_key"] for r in lab.store.db.execute(
            "SELECT logical_key FROM v_latest WHERE status='error' AND error_class=?", (a.cls,))}
        cells = [c for c in cells if c["logical_key"] in keep]
    lab.log(f"celle selezionate: {len(cells)}")
    try:
        stats = lab.run(cells, mode)
    except KeyboardInterrupt:
        lab.log("interrotto: le celle completate restano salvate, rilancia per proseguire")
        lab.sheet_status("fermo: interrotto")
        lab.sheets.close()
        sys.exit(130)
    lab.log(f"fine: {stats}")


def cmd_status(lab, a):
    w, args = where_clause(a)
    q = f"SELECT test, status, COUNT(*) n FROM v_latest WHERE {w} GROUP BY test, status ORDER BY test"
    by = defaultdict(dict)
    for r in lab.store.db.execute(q, args):
        by[r["test"]][r["status"]] = r["n"]
    print(f"{'test':26} {'ok':>6} {'errore':>7} {'saltate':>8} {'invalid.':>9}")
    for t, d in by.items():
        print(f"{t:26} {d.get('ok', 0):6} {d.get('error', 0):7} {d.get('skipped', 0):8} {d.get('invalidated', 0):9}")
    print()
    q = f"SELECT model, status, COUNT(*) n FROM v_latest WHERE {w} AND model IS NOT NULL GROUP BY model, status"
    bym = defaultdict(dict)
    for r in lab.store.db.execute(q, args):
        bym[r["model"]][r["status"]] = r["n"]
    print(f"{'modello':26} {'ok':>6} {'errore':>7} {'saltate':>8}")
    for mdl, d in bym.items():
        print(f"{mdl:26} {d.get('ok', 0):6} {d.get('error', 0):7} {d.get('skipped', 0):8}")


def cmd_errors(lab, a):
    w, args = where_clause(a)
    rows = list(lab.store.db.execute(f"SELECT * FROM v_errors WHERE {w} ORDER BY id", args))
    cls = Counter(r["error_class"] for r in rows)
    print("Per classe: " + ", ".join(f"{k} {v}" for k, v in cls.most_common()) + "\n")
    for r in rows[-a.last:]:
        print(f"#{r['id']} [{r['error_class']}] {r['model']} {r['quant']} {r['engine']} {r['test']} {r['params_json']}")
        print(f"    {(r['error_msg'] or '').splitlines()[0][:160] if r['error_msg'] else ''}")
        print(f"    log: {r['log_path']}")


def cmd_prune(lab, a):
    """Toglie dal database e dal foglio le celle "orfane": quelle che il piano attuale non contiene più
    (chiave cambiata dopo una modifica a matrix.yaml o al codice). Sono i doppioni e gli errori vecchi."""
    keys = {c["logical_key"] for c in lab.planner.expand()}
    rows = [r for r in lab.store.db.execute(
        "SELECT logical_key, model, quant, engine, test, status, finished_at FROM v_latest WHERE machine=?",
        (lab.machine["id"],)) if r["logical_key"] not in keys]
    keep_ok = [r for r in rows if r["status"] == "ok" and not a.all]
    rows = [r for r in rows if r["status"] != "ok" or a.all]
    cnt = Counter(r["status"] for r in rows)
    for r in rows:
        if r["status"] != "skipped" or a.verbose:
            print(f"  {r['status']:8} {r['model']} {r['quant']} {r['engine']} {r['test']}  ({(r['finished_at'] or '')[:16]})")
    print(f"\norfane: {', '.join(f'{k} {v}' for k, v in cnt.items()) or 'nessuna'}"
          + (f"; {len(keep_ok)} con risultato ok lasciate (--all per toglierle)" if keep_ok else ""))
    if a.dry or not rows:
        return
    ks = [r["logical_key"] for r in rows]
    db = lab.store.db
    for i in range(0, len(ks), 400):
        part = ks[i:i + 400]
        q = ",".join("?" * len(part))
        ids = f"SELECT id FROM runs WHERE machine=? AND logical_key IN ({q})"
        args = [lab.machine["id"], *part]
        db.execute(f"DELETE FROM metrics WHERE run_id IN ({ids})", args)
        db.execute(f"DELETE FROM samples WHERE run_id IN ({ids})", args)
        db.execute(f"DELETE FROM runs WHERE machine=? AND logical_key IN ({q})", args)
    db.commit()
    print(f"database: tolte {len(ks)} celle")
    if lab.sheets.enabled:
        ok = lab.sheets._send({"op": "delete", "sheet": "Risultati", "key": "cella", "keys": ks,
                               "where": {"macchina": lab.machine["id"]}}, tries=2)
        lab.sheet_status()
        lab.sheets.close(120)
        print("foglio: righe tolte" if ok else "foglio: cancellazione non riuscita, toglile a mano o riprova")


def cmd_invalidate(lab, a):
    w, args = where_clause(a)
    if w == "1=1" and not a.all:
        sys.exit("nessun filtro: per invalidare tutto usa --all")
    n = lab.store.invalidate(w, args)
    print(f"{n} celle segnate da rifare")


def cmd_export(lab, a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    db = lab.store.db
    pkeys = ["threads", "threads_batch", "fa", "kv", "batch", "ubatch", "ctx", "thinking", "prompt_cache", "draft"]
    where = "" if a.history else "WHERE id IN (SELECT id FROM v_latest)"
    runs = [dict(r) for r in db.execute(f"SELECT * FROM runs {where} ORDER BY id")]
    machines = {r["id"]: json.loads(r["info_json"]) for r in db.execute("SELECT * FROM machines")}
    for r in runs:
        p = json.loads(r.pop("params_json") or "{}")
        for k in pkeys:
            r[f"p_{k}"] = p.get(k)
        r["cpu"] = machines.get(r["machine"], {}).get("cpu_model")
        r["avx2"] = machines.get(r["machine"], {}).get("avx2")
        m = lab.cfg["models"].get(r["model"] or "", {})
        r["family"], r["params_b"] = m.get("family"), m.get("params_b")
        r["bits"] = lab.cfg["quants"].get(r["quant"] or "", {}).get("bits")
    metrics = defaultdict(dict)
    for row in db.execute("SELECT run_id, name, value FROM metrics"):
        metrics[row["run_id"]][row["name"]] = row["value"]
    tok = {row["run_id"]: row["g"] for row in db.execute("SELECT run_id, SUM(gen_tokens) g FROM samples GROUP BY run_id")}
    for r in runs:
        if r.get("energy_j") and tok.get(r["id"]):
            metrics[r["id"]]["gen_tokens_per_joule"] = tok[r["id"]] / r["energy_j"]

    import math
    def wilson(k, n, z=1.96):
        if not n:
            return None, None
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return round(100 * (c - m), 1), round(100 * (c + m), 1)

    for row in db.execute("SELECT run_id, COUNT(correct) n, SUM(correct) k FROM samples "
                          "WHERE correct IS NOT NULL GROUP BY run_id"):
        lo, hi = wilson(row["k"] or 0, row["n"])
        metrics[row["run_id"]]["ci95_low_pct"] = lo
        metrics[row["run_id"]]["ci95_high_pct"] = hi
        metrics[row["run_id"]]["n_samples"] = row["n"]

    def write(name, rows, fields=None):
        if not fields:
            seen = []
            for r in rows:
                for k in r:
                    if k not in seen:
                        seen.append(k)
            fields = [k for k in seen if not k.startswith("m.")] + sorted(k for k in seen if k.startswith("m."))
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"  {out / name}: {len(rows)} righe")

    base_cols = [k for k in runs[0]] if runs else []
    write("runs.csv", runs, base_cols)
    long_rows = [{**{k: r[k] for k in base_cols if k not in ("error_msg", "extra_json", "log_path")},
                  "metric": n, "value": v} for r in runs for n, v in metrics.get(r["id"], {}).items()]
    write("metrics_long.csv", long_rows)
    wide = []
    for r in runs:
        if r["status"] != "ok":
            continue
        row = {k: r[k] for k in base_cols if k not in ("error_msg", "extra_json", "log_path", "test_opts_json")}
        row.update({f"m.{n}": v for n, v in metrics.get(r["id"], {}).items()})
        wide.append(row)
    write("metrics_wide.csv", wide)
    ids = {r["id"] for r in runs}
    samples = [dict(s) for s in db.execute("SELECT * FROM samples") if s["run_id"] in ids]
    write("samples.csv", samples)
    write("errors.csv", [r for r in runs if r["status"] == "error"], base_cols)
    print("\nPer analisi più libere il database SQLite è", lab.db_path, "(viste v_latest, v_results, v_errors)")


def cmd_sync(lab, a):
    from .sheets import result_row
    if not lab.sheets.enabled:
        sys.exit("foglio non configurato: vedi `sheets:` in matrix.yaml e lab/README.md")
    ids = [r["id"] for r in lab.store.db.execute("SELECT id FROM v_latest ORDER BY id")]
    rows = [r for r in (result_row(lab.store.db, i) for i in ids) if r]
    print(f"invio {len(rows)} risultati e lo stato di {lab.machine['id']}...")
    for i in range(0, len(rows), 500):
        lab.sheets._send({"op": "upsert", "sheet": "Risultati", "key": "cella", "rows": rows[i:i + 500]}, tries=2)
    lab.sheet_status("fermo" if a.idle else None)
    lab.sheets.close(120)
    print("fatto")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(LAB_DIR / "matrix.yaml"))
    ap.add_argument("--machine", help="forza l'identità della macchina (nasgul, z87...)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--engine", action="append")
    p.add_argument("--test", action="append")
    p.add_argument("--sweep", action="append")
    p.add_argument("--models", action="store_true", help="scarica subito anche tutti i modelli della matrice")
    p.add_argument("--update", action="store_true", help="aggiorna i motori (nuova build/immagine)")
    p.add_argument("--skip-engines", action="store_true")
    p.add_argument("--skip-tests", action="store_true")

    p = sub.add_parser("plan")
    add_filters(p)
    p.add_argument("--check", action="store_true", help="verifica le impronte (motivo per cui una cella è da rifare)")
    p.add_argument("--reasons", action="store_true", help="elenca i motivi di incompatibilità")
    p.add_argument("--list", action="store_true")

    for name in ("run", "retry"):
        p = sub.add_parser(name)
        add_filters(p)
        p.add_argument("--force", action="store_true", help="riesegue anche le celle già fatte")
        p.add_argument("--dry", action="store_true", help="mostra cosa farebbe senza eseguire")
        if name == "run":
            p.add_argument("--retry-errors", action="store_true", help="ripete anche gli errori già noti")
        else:
            p.add_argument("--cls", help="solo errori di questa classe (oom, timeout, illegal_instruction...)")

    for name in ("status", "errors", "invalidate"):
        p = sub.add_parser(name)
        add_filters(p)
        p.add_argument("--cls")
        if name == "errors":
            p.add_argument("--last", type=int, default=50)
        if name == "invalidate":
            p.add_argument("--all", action="store_true")

    p = sub.add_parser("export")
    p.add_argument("--out", default=str(LAB_DIR.parent / "results" / "export"))
    p.add_argument("--history", action="store_true", help="include tutti i tentativi, non solo l'ultimo")

    p = sub.add_parser("prune", help="toglie da database e foglio le celle che il piano non contiene più")
    p.add_argument("--dry", action="store_true", help="mostra soltanto cosa toglierebbe")
    p.add_argument("--all", action="store_true", help="toglie anche le orfane con risultato ok")
    p.add_argument("--verbose", action="store_true", help="elenca anche le orfane 'skipped'")

    p = sub.add_parser("sync")
    p.add_argument("--idle", action="store_true", help="segna la macchina come ferma nella riga 'in corso'")

    a = ap.parse_args(argv)
    import signal

    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    lab = Lab(Path(a.config), a.machine)
    if a.cmd == "prepare":
        rep = lab.prepare(engines=[] if a.skip_engines else a.engine, tests=[] if a.skip_tests else a.test,
                          models=a.models, update=a.update, sweeps=a.sweep)
        if a.skip_engines:
            rep = [r for r in rep if r[0] != "motore"]
        print()
        for kind, name, st, info in rep:
            print(f"{kind:8} {name:34} {st:8} {info}")
        bad = sum(1 for r in rep if r[2] == "ERRORE")
        print(f"\n{bad} problemi" if bad else "\ntutto pronto")
    elif a.cmd == "plan":
        cmd_plan(lab, a)
    elif a.cmd == "run":
        cmd_run(lab, a)
    elif a.cmd == "retry":
        cmd_run(lab, a, only_errors=True)
    elif a.cmd == "status":
        cmd_status(lab, a)
    elif a.cmd == "errors":
        cmd_errors(lab, a)
    elif a.cmd == "invalidate":
        cmd_invalidate(lab, a)
    elif a.cmd == "export":
        cmd_export(lab, a)
    elif a.cmd == "sync":
        cmd_sync(lab, a)
    elif a.cmd == "prune":
        cmd_prune(lab, a)


if __name__ == "__main__":
    main()
