#!/usr/bin/env python3
"""GILPA LLM bench v2: misura velocita' e qualita' di un endpoint OpenAI-compatibile.

Solo libreria standard. Esempi:
  python3 bench.py --base-url http://localhost:8480/v1 --model qwen3-1.7b --local
  python3 bench.py --base-url https://api.groq.com/openai/v1 --model <modello> --api-key-env GILPA_LLM_API_KEY

Task disponibili (--tasks, separati da virgola):
  intent_v2, sql_v2            nuovi, default
  intent_short, intent_full, sql   quelli del primo giro, invariati (per confronto)
"""
import argparse, csv, json, os, pathlib, re, sqlite3, statistics, sys, time
import urllib.error, urllib.request

HERE = pathlib.Path(__file__).parent
SEED = (HERE / "seed.sql").read_text(encoding="utf-8")
DDL, DATA = SEED.split("-- DATA")
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
CASES_V2 = json.loads((HERE / "cases_v2.json").read_text(encoding="utf-8"))
TODAY = "2026-09-17"

INTENTS = ["shop_list", "add_component", "list_projects", "estimate_order", "last_activity",
           "log_activity", "start_pomodoro", "weekly_plan", "unknown"]
SHOPS = ["Action", "Amazon", "AliExpress", "Leroy Merlin", "eBay"]
PROJECTS = ["NASGUL", "Album Lia", "Domotica Zigbee", "Classifica auto", "Sito Pentawa"]
CATEGORIES = ["cinema", "gaming", "workout", "pomodoro", "social", "admin"]

# ---------------------------------------------------------------- v1 (invariato)
NULLABLE_STR = {"type": ["string", "null"]}
NULLABLE_NUM = {"type": ["number", "null"]}
INTENT_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": INTENTS}, "shop": NULLABLE_STR,
                   "project": NULLABLE_STR, "item": NULLABLE_STR, "quantity": NULLABLE_NUM,
                   "price": NULLABLE_NUM, "category": NULLABLE_STR},
    "required": ["intent", "shop", "project", "item", "quantity", "price", "category"],
    "additionalProperties": False,
}
SQL_SCHEMA = {"type": "object", "properties": {"sql": {"type": "string"}},
              "required": ["sql"], "additionalProperties": False}

INTENT_SHORT = f"""Sei il motore di intent di GILPA, assistente personale di Gil. Data di oggi: {TODAY}.
Classifica il messaggio in UN intent ed estrai le entita' (null se assenti). Rispondi solo con JSON.
Intent:
- shop_list: Gil va o ordina in un negozio e vuole sapere cosa comprare li'
- add_component: aggiungere un componente/materiale da comprare
- list_projects: elencare i progetti
- estimate_order: stimare la spesa di un ordine
- last_activity: da quanto non fa un'attivita'
- log_activity: registrare un'attivita' appena svolta
- start_pomodoro: avviare un timer pomodoro
- weekly_plan: quando sono pianificati i blocchi di tempo
- unknown: tutto il resto
Categorie attivita': cinema, gaming, workout, pomodoro, social, admin.
Negozi noti: Action, Amazon, AliExpress, Leroy Merlin, eBay."""

INTENT_FULL = INTENT_SHORT + f"""

Schema del database (per contesto):
{DDL.strip()}

Progetti esistenti: NASGUL, Album Lia, Domotica Zigbee, Classifica auto, Sito Pentawa.
Alias negozi: action/da action -> Action; amazon/amazon it -> Amazon; ali/aliexpress -> AliExpress;
leroy/leroy merlin -> Leroy Merlin; ebay -> eBay."""

SQL_SYSTEM = f"""Sei il generatore SQL di GILPA. Data di oggi: {TODAY}.
Scrivi UNA sola query SQLite di sola lettura (SELECT) che risponde alla domanda. Rispondi solo con JSON {{"sql": "..."}}.
Valori ammessi: components.status in (to_buy, ordered, delivered, cancelled); projects.status in (active, paused, done);
priority in (high, medium, low); activity_log.category in (cinema, gaming, workout, pomodoro, social, admin);
time_blocks.day_of_week 0=lunedi' ... 6=domenica. Nomi negozio canonici: Action, Amazon, AliExpress, Leroy Merlin, eBay.
Il costo di un componente e' estimated_price*quantity.

{DDL.strip()}"""

# ---------------------------------------------------------------- v2
# 1) Grammatica stretta: shop / project / category possono valere SOLO i nomi canonici o null.
#    In GILPA queste liste vanno generate dal DB (tabelle shops, projects) a ogni avvio.
# 2) Solo "intent" e' obbligatorio: i campi assenti si omettono, cosi' l'output e' corto.
def _enum_or_null(values):
    return {"anyOf": [{"type": "string", "enum": values}, {"type": "null"}]}

INTENT_SCHEMA_V2 = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": INTENTS},
                   "shop": _enum_or_null(SHOPS),
                   "project": _enum_or_null(PROJECTS),
                   "category": _enum_or_null(CATEGORIES),
                   "item": NULLABLE_STR, "quantity": NULLABLE_NUM,
                   "price": NULLABLE_NUM, "duration_min": NULLABLE_NUM},
    "required": ["intent"],
    "additionalProperties": False,
}

# 3) Esempi per ogni intent, comprese le forme implicite. Nessun esempio coincide con un caso di test.
#    La data sta in fondo: in produzione e' l'unica parte che cambia, e cosi' non invalida la cache.
INTENT_V2 = f"""Sei il motore di intent di GILPA, assistente personale di Gil.
Classifica il messaggio in UN intent ed estrai le entita'. Rispondi con JSON su UNA riga, senza spazi
ne' a capo. Ometti i campi che non compaiono nel messaggio. Non inventare valori.

Intent:
- shop_list: Gil va, passa, e' o sta per ordinare in un negozio. Vale anche se non chiede niente: nominare il negozio dove sta andando basta.
- add_component: c'e' un oggetto da comprare o mettere in lista (con o senza prezzo).
- estimate_order: chiede un TOTALE di spesa. Un prezzo nel messaggio da solo non basta.
- list_projects: vuole vedere i progetti.
- last_activity: chiede da quanto tempo non fa qualcosa, o quando l'ha fatto l'ultima volta.
- log_activity: racconta una cosa che HA FATTO (film visto, partita, allenamento, serata, commissioni).
- start_pomodoro: vuole avviare un timer di lavoro.
- weekly_plan: chiede quando e' pianificato qualcosa nella settimana.
- unknown: tutto il resto.

Campi: shop, project, category, item (testo libero), quantity, price (euro), duration_min (minuti).
Negozi: Action, Amazon, AliExpress (detto anche "ali"), Leroy Merlin (detto anche "leroy"), eBay.
Progetti: NASGUL, Album Lia, Domotica Zigbee, Classifica auto, Sito Pentawa.
Categorie attivita': cinema (film), gaming (videogiochi), workout (palestra, corsa), pomodoro, social (amici, uscite), admin (banca, bollette, burocrazia).

Esempi:
faccio un salto da Action
{{"intent":"shop_list","shop":"Action"}}
apro eBay, c'e' qualcosa da prendere?
{{"intent":"shop_list","shop":"eBay"}}
aggiungi un cavo HDMI per NASGUL, Amazon, 8 euro
{{"intent":"add_component","shop":"Amazon","project":"NASGUL","item":"cavo HDMI","price":8}}
servono 4 viti M3 per la classifica auto
{{"intent":"add_component","project":"Classifica auto","item":"viti M3","quantity":4}}
quanto mi costa svuotare la lista di eBay?
{{"intent":"estimate_order","shop":"eBay"}}
elenca i progetti attivi
{{"intent":"list_projects"}}
da quanto non vedo gli amici?
{{"intent":"last_activity","category":"social"}}
stasera ho visto Interstellar
{{"intent":"log_activity","category":"cinema","item":"Interstellar"}}
ho fatto 45 minuti di corsa
{{"intent":"log_activity","category":"workout","duration_min":45}}
parti con un pomodoro su Classifica auto
{{"intent":"start_pomodoro","project":"Classifica auto"}}
cosa ho in programma giovedi'?
{{"intent":"weekly_plan"}}
che ore sono a Tokyo?
{{"intent":"unknown"}}

Data di oggi: {TODAY}."""

_V3_FIXES = [
    ("- shop_list: Gil va, passa, e' o sta per ordinare in un negozio. Vale anche se non chiede niente: nominare il negozio dove sta andando basta.",
     "- shop_list: vuole sapere COSA c'e' da comprare in un negozio (cosa manca, cosa serve, cosa c'e' in lista), "
     "oppure dice che ci va, ci passa o sta per ordinare li'. Nominare il negozio dove sta andando basta."),
    ("- estimate_order: chiede un TOTALE di spesa. Un prezzo nel messaggio da solo non basta.",
     "- estimate_order: chiede QUANTO spende, cioe' una cifra totale. Se chiede COSA comprare e' shop_list, anche se parla di ordine o di lista."),
    ("quanto mi costa svuotare la lista di eBay?\n{\"intent\":\"estimate_order\",\"shop\":\"eBay\"}",
     "quanto mi costa svuotare la lista del Leroy Merlin?\n{\"intent\":\"estimate_order\",\"shop\":\"Leroy Merlin\"}"),
]
INTENT_V3 = INTENT_V2
for old, new in _V3_FIXES:
    assert old in INTENT_V3, f"testo non trovato: {old[:40]}"
    INTENT_V3 = INTENT_V3.replace(old, new)

# SQL v2: stesse query di prima, con le regole che avrebbero evitato gli errori del primo giro.
SQL_V2 = SQL_SYSTEM + """
Regole:
- Valori di testo SEMPRE tra apici singoli: status = 'to_buy'. Mai virgolette doppie, mai senza apici.
- Usa una sola tabella quando basta. Fai JOIN solo se ti servono colonne di un'altra tabella:
  molte righe hanno project_id NULL e una JOIN le farebbe sparire.
- Il negozio e' il testo in components.shop. La tabella shops non serve per filtrare.
- "Da comprare", "mi manca", "devo prendere" = status = 'to_buy' e nient'altro.
- time_blocks = cosa e' PIANIFICATO (giorni, orari). activity_log = cosa e' GIA' STATO FATTO (date)."""

TASKS = {  # nome: (tipo, system prompt, schema, max_tokens, casi)
    "intent_short": ("intent", INTENT_SHORT, INTENT_SCHEMA, 160, CASES["intent"]),
    "intent_full":  ("intent", INTENT_FULL, INTENT_SCHEMA, 160, CASES["intent"]),
    "sql":          ("sql", SQL_SYSTEM, SQL_SCHEMA, 300, CASES["sql"]),
    "intent_v2":    ("intent", INTENT_V2, INTENT_SCHEMA_V2, 120, CASES_V2["intent"]),
    "sql_v2":       ("sql", SQL_V2, SQL_SCHEMA, 300, CASES["sql"]),
    "intent_v3":    ("intent", INTENT_V3, INTENT_SCHEMA_V2, 120, CASES_V2["intent"]),
}
CLEAN_FIELDS = ["shop", "project", "category", "quantity", "price", "duration_min"]


def post(url, body, api_key, timeout):
    headers = {"Content-Type": "application/json", "User-Agent": "gilpa-bench/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = json.dumps(body).encode()
    for attempt in range(8):
        req = urllib.request.Request(url, data=payload, headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            return data, time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 7:
                raise
            wait = float(e.headers.get("retry-after") or 15) + 1
            print(f"    429, aspetto {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)

def ask(args, system, user, schema, max_tokens):
    body = {"model": args.model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    body["max_tokens"] = max_tokens + args.extra_tokens
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    if args.format == "schema":
        # "strict" all'OpenAI richiede tutti i campi obbligatori: lo schema v2 non lo e'
        strict = set(schema.get("required", [])) == set(schema["properties"])
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "out", "strict": strict, "schema": schema}}
    elif args.format == "object":
        body["response_format"] = {"type": "json_object"}
    if args.local:  # parametri specifici di llama-server
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["cache_prompt"] = not args.cold
    data, wall = post(args.base_url.rstrip("/") + "/chat/completions", body, args.api_key, args.timeout)
    content = data["choices"][0]["message"].get("content") or ""
    t = data.get("timings") or {}
    u = data.get("usage") or {}
    m = {"wall_s": round(wall, 2),
         "prompt_tokens": t.get("prompt_n", u.get("prompt_tokens")),
         "prompt_tps": round(t["prompt_per_second"], 1) if "prompt_per_second" in t else None,
         "gen_tokens": t.get("predicted_n", u.get("completion_tokens")),
         "gen_tps": round(t["predicted_per_second"], 1) if "predicted_per_second" in t else None}
    return content, m


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


def check_entities(got, expect):
    for k, v in expect.items():
        if k == "intent":
            continue
        g = got.get(k)
        if isinstance(v, (int, float)):
            if not isinstance(g, (int, float)) or isinstance(g, bool) or abs(g - v) > 0.01:
                return False
        elif g is None or v not in str(g).lower():
            return False
    return True


def check_clean(got, case):
    """True se il modello NON ha riempito campi che nel messaggio non ci sono.
    'item' e' testo libero e non viene controllato; 'allow' elenca i campi tollerati per quel caso."""
    ok = set(case["expect"]) | set(case.get("allow", []))
    for k in CLEAN_FIELDS:
        if k not in ok and got.get(k) not in (None, "", "null"):
            return False
    return True


def norm(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 2)
    return None if v is None else str(v).strip().lower()


def rows_match(got, exp):
    g = [set(map(norm, r)) for r in got]
    e = [set(map(norm, r)) for r in exp]
    if len(g) != len(e):
        return False
    for er in e:
        for i, gr in enumerate(g):
            if er <= gr:
                g.pop(i)
                break
        else:
            return False
    return True


def run_sql(sql):
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        raise ValueError("non e' una SELECT")
    con = sqlite3.connect(":memory:")
    con.executescript(SEED)
    con.execute("PRAGMA query_only=ON")
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="nome modello da inviare all'endpoint")
    ap.add_argument("--label", help="etichetta per i risultati (default = model)")
    ap.add_argument("--api-key-env", help="variabile d'ambiente con la API key")
    ap.add_argument("--local", action="store_true", help="endpoint llama-server (abilita timings e cache_prompt)")
    ap.add_argument("--cold", action="store_true", help="disabilita la prompt cache (misura il prefill completo)")
    ap.add_argument("--format", choices=["schema", "object", "none"], default="schema")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tasks", default="intent_v2,sql_v2", help="scelte: " + ", ".join(TASKS))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--extra-tokens", type=int, default=0)
    args = ap.parse_args()
    args.api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    label = args.label or args.model
    tasks = args.tasks.split(",")
    for t in tasks:
        if t not in TASKS:
            sys.exit(f"Task sconosciuto: {t}. Scelte: {', '.join(TASKS)}")

    rows, cold_s = [], {}
    for rep in range(args.repeat):
        for task in tasks:
            kind, system, schema, max_tok, cases = TASKS[task]
            if rep == 0:
                # Una richiesta a vuoto per task: carica il system prompt in cache e ci dice quanto
                # costa la PRIMA richiesta dopo un riavvio o un cambio di prompt.
                try:
                    _, m = ask(args, system, "ciao", schema, 20)
                    cold_s[task] = m["wall_s"]
                    print(f"[{label}] {task}: prima richiesta a freddo {m['wall_s']}s "
                          f"({m.get('prompt_tokens')} token di prompt)", flush=True)
                except Exception as e:
                    sys.exit(f"Endpoint non raggiungibile: {e}")
            for case in cases:
                row = {"label": label, "task": task, "case": case["id"], "rep": rep, "json_ok": 0,
                       "correct": 0, "entities_ok": None, "clean": None, "error": "", "output": ""}
                try:
                    content, m = ask(args, system, case["msg"], schema, max_tok)
                    row.update(m)
                    row["output"] = content[:300].replace("\n", " ")
                    obj = parse_json(content)
                    row["json_ok"] = 1
                    if kind == "intent":
                        row["correct"] = int(obj.get("intent") == case["expect"]["intent"])
                        row["entities_ok"] = int(check_entities(obj, case["expect"]))
                        row["clean"] = int(check_clean(obj, case))
                    else:
                        row["correct"] = int(rows_match(run_sql(obj["sql"]), run_sql(case["ref"])))
                except urllib.error.HTTPError as e:
                    row["error"] = f"HTTP {e.code}: {e.read()[:200]!r}"
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:200]
                rows.append(row)
                print(f"  {task:13} {case['id']} ok={row['correct']} ent={row['entities_ok']} "
                      f"clean={row['clean']} wall={row.get('wall_s')}s gen={row.get('gen_tokens')} "
                      f"tg={row.get('gen_tps')} {row['error']}", flush=True)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    path = out / f"{label}-{'cold' if args.cold else 'warm'}-{stamp}.csv"
    fields = ["label", "task", "case", "rep", "json_ok", "correct", "entities_ok", "clean", "wall_s",
              "prompt_tokens", "prompt_tps", "gen_tokens", "gen_tps", "error", "output"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def med(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(statistics.median(vals), 1) if vals else "-"

    def pct(vals):
        vals = [v for v in vals if v is not None]
        return f"{100 * sum(vals) / len(vals):.0f}" if vals else "-"

    print(f"\n== {label} ({'cold' if args.cold else 'warm'}) ==")
    print(f"{'task':13} {'n':>3} {'json%':>6} {'acc%':>6} {'ent%':>6} {'clean%':>7} {'g_tok':>6} "
          f"{'tg t/s':>7} {'wall med':>9} {'wall max':>9} {'freddo s':>9}")
    for task in tasks:
        r = [x for x in rows if x["task"] == task]
        if not r:
            continue
        walls = [x.get("wall_s") for x in r if x.get("wall_s") is not None]
        print(f"{task:13} {len(r):>3} {pct([x['json_ok'] for x in r]):>6} {pct([x['correct'] for x in r]):>6} "
              f"{pct([x['entities_ok'] for x in r]):>6} {pct([x['clean'] for x in r]):>7} "
              f"{med([x.get('gen_tokens') for x in r]):>6} {med([x.get('gen_tps') for x in r]):>7} "
              f"{med(walls):>9} {max(walls) if walls else '-':>9} {cold_s.get(task, '-'):>9}")
    wrong = [f"{x['task']}:{x['case']}" for x in rows if not x["correct"]]
    if wrong:
        print("\nSbagliati: " + " ".join(wrong))
    print(f"Dettagli: {path}")


if __name__ == "__main__":
    main()
