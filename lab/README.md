# lab: la matrice di test per "LLM su hardware datato"

`lab/` è l'orchestratore che esegue il piano `llm-hardware-datato-piano-test.md`: prepara motori e
dati, prende i modelli uno alla volta e per ciascuno prova quantizzazioni, motori e parametri,
lanciando i test pubblici e il nostro test intenti (`bench.py`, che non viene modificato).

Tutto finisce in un database SQLite. Ogni risultato porta con sé un'**impronta** di ciò da cui
dipende, così dopo una modifica si ripete solo quello che la modifica tocca.

## Installazione

Prerequisiti di sistema (una volta, su NASGUL e sulla Z87):

```bash
sudo apt install python3-venv git cmake build-essential   # build nativa di llama.cpp e ik_llama.cpp
# docker è già presente su NASGUL (serve per llama.cpp ufficiale, Ollama, OpenVINO)
```

Poi, dalla cartella del repo:

```bash
./lab/setup.sh          # crea lab/.venv con lm-eval, datasets, torch CPU, ecc.
./lab/lab.sh prepare    # compila/scarica i motori e scarica i dataset dei test
./lab/lab.sh plan       # quante celle, quante fatte, stima dei tempi
./lab/lab.sh run        # esegue
```

La macchina viene riconosciuta dall'hostname (`machines:` in `matrix.yaml`). Se l'hostname è
diverso: `./lab/lab.sh --machine z87 run`.

## I comandi

| Comando | Cosa fa |
|---|---|
| `prepare` | Compila llama.cpp e ik_llama.cpp, scarica immagini Docker e binari, scarica i dataset, verifica che ogni modello abbia il file per ogni quantizzazione. `--models` scarica anche tutti i GGUF subito. `--update` ricompila/aggiorna i motori (e quindi fa ripetere i loro test). |
| `plan` | Riassunto per sweep: fatte, da fare, errori, incompatibili, stima ore. `--check` dice *perché* una cella è da rifare (versione del test, dati, motore, file del modello). `--reasons` elenca le incompatibilità. `--list` mostra le celle. |
| `run` | Esegue ciò che manca. Un modello alla volta: tutte le sue quantizzazioni, poi motori, poi parametri; il server parte una volta per combinazione e serve tutti i test di quella combinazione. Si può interrompere con Ctrl-C e rilanciare: riparte da dove era. |
| `retry` | Ripete solo le celle finite in errore. `--cls oom`, `--cls timeout`, ecc. per una classe sola. |
| `status` | Conteggi per test e per modello. |
| `errors` | Elenco degli errori con classe, messaggio e percorso del log. |
| `invalidate` | Segna da rifare le celle che corrispondono ai filtri (es. dopo aver scoperto un problema). |
| `export` | CSV per analisi e grafici in `results/export/`. |

Filtri comuni a quasi tutti: `--model`, `--quant`, `--engine`, `--test`, `--sweep`, `--param threads=2`.
Esempi:

```bash
./lab/lab.sh run --model qwen3.5-4b                     # solo la baseline
./lab/lab.sh run --sweep quant-scan --test kld          # solo la KLD della scansione quant
./lab/lab.sh retry --cls timeout --engine ollama        # rifà i timeout di Ollama
./lab/lab.sh run --dry                                  # mostra cosa farebbe
```

## Modularità: cosa si ripete dopo una modifica

Una **cella** è: macchina + modello + quantizzazione + motore + parametri + test + opzioni del test.
Ogni test dichiara da quali di queste parti dipende (`depends_on`), e la chiave della cella usa solo
quelle. Esempio: i test di qualità a temperatura 0 non dipendono dal numero di thread o dal batch,
quindi lo sweep sui thread non li ripete. llama-bench non dipende dai casi del test intenti.

All'esecuzione si aggiunge l'impronta di ciò che è stato davvero usato: SHA256 del GGUF, versione
del motore (commit git o immagine Docker), hash dei dati del test, versione del codice del test.

| Modifica | Cosa si ripete |
|---|---|
| Cambi `cases_v2.json`, `bench.py` o `seed.sql` | solo i test gilpa, su tutte le combinazioni |
| Cambi `limit`, `seed` o un'altra opzione di un test | solo quel test |
| `prepare --update --engine llamacpp-native` (nuovo commit) | tutti i test fatti con quel motore, nient'altro |
| L'autore ricarica un GGUF su Hugging Face (SHA diverso) | i test di quel modello e quella quantizzazione |
| Aggiungi un modello, una quant, un valore di parametro | solo le celle nuove |
| Correggi la logica di un test in `lab/tests/` | alzi `VERSION` in quella classe: si ripete solo quel test |
| Cambi i default dei parametri | le celle i cui parametri cambiano davvero, per i test che ne dipendono |

Lo storico non si cancella mai: un nuovo tentativo aggiunge una riga, e le viste prendono l'ultima.

## Errori

Ogni errore è salvato con una classe, il messaggio e il log completo, e resta ripetibile per la
singola cella. Classi principali:

| Classe | Significato |
|---|---|
| `illegal_instruction` | Il binario usa istruzioni che la CPU non ha (tipico: build generica su CPU senza AVX2). È un risultato. |
| `oom` | Memoria insufficiente |
| `timeout` | Avvio del server o test oltre il limite in `timeouts:` |
| `download` | Repo o file inesistente su Hugging Face, o download fallito |
| `model_load` | Il motore non riconosce l'architettura o il file |
| `unsupported` | Opzione non supportata dal motore |
| `engine_unavailable` | Motore non preparato: lancia `prepare` |
| `connection` | Il server è caduto durante il test (viene riavviato per i test successivi) |
| `interrupted` | Esecuzione interrotta (Ctrl-C, spegnimento) |
| `bug` | Eccezione nel codice del laboratorio: il log ha il traceback |

Un errore già registrato con la stessa impronta non viene ripetuto da `run` (evita di rifare ogni
volta un avvio che fallisce in 10 minuti): serve `retry` o `run --retry-errors`. Se invece cambia
qualcosa da cui la cella dipende, `run` la riprova da solo.

Le combinazioni impossibili (OpenVINO senza AVX2, gpt-oss-20b su NASGUL, KV quantizzata senza flash
attention, modello più grande di `max_model_gb`) sono registrate come `skipped` con il motivo, così
nei grafici un buco ha una spiegazione.

## Dove sono i dati

- `results/lab.sqlite`: tabelle `runs` (una riga per esecuzione), `metrics` (formato lungo:
  run, nome, valore, unità), `samples` (ogni singolo caso, con tempi e token), `artifacts` (file e
  SHA256), `engine_versions`, `machines`. Viste: `v_latest` (ultimo tentativo per cella),
  `v_results` (ultimo tentativo con le metriche), `v_errors`.
- `labdata/logs/runs/<macchina>/<modello>/<quant>/<motore>/`: log di server e test; per lm-eval
  anche i JSON completi dei risultati e dei campioni.
- `./lab/lab.sh export` scrive in `results/export/`:
  - `runs.csv`: una riga per cella, con i parametri in colonne `p_*`, CPU, famiglia, bit;
  - `metrics_long.csv`: una riga per metrica (comodo per pivot e grafici);
  - `metrics_wide.csv`: una riga per cella con le metriche in colonne `m.*`;
  - `samples.csv`: i singoli casi;
  - `errors.csv`.

Esempi di letture dirette:

```sql
-- accuratezza del test intenti per modello e quantizzazione, con velocità
SELECT model, quant, engine, machine,
       MAX(CASE WHEN metric='intent_v3.tutto_giusto' THEN value END) AS giusti,
       MAX(CASE WHEN metric='intent_v3.wall_med_s' THEN value END)   AS wall_s
FROM v_results WHERE test='gilpa_intent' AND status='ok'
GROUP BY run_id ORDER BY giusti DESC, wall_s;

-- effetto di AVX2: stessa cella su due macchine
SELECT model, quant, machine, value AS pp_tps FROM v_results
WHERE test='llama_bench' AND metric='pp_tps' ORDER BY model, quant, machine;

-- KLD contro dimensione del file
SELECT model, quant, file_size_gb, value FROM v_results
WHERE test='kld' AND metric='wiki_it.kld_mean' ORDER BY model, file_size_gb;
```

## Foglio Google (stato in tempo reale)

Il laboratorio può copiare ogni risultato su un Google Sheet appena la cella finisce, così lo stato
si vede dal telefono senza entrare su NASGUL. Non servono librerie Google né account di servizio:
il foglio contiene uno script (`lab/sheets_apps_script.gs`) pubblicato come app web, e il
laboratorio gli manda le righe con un token.

Schede:

| Scheda | Contenuto |
|---|---|
| `Stato` | una riga per macchina e sweep (fatte, da fare, errori, % completamento, ore stimate) e una riga "▶ in corso" per macchina con la cella attuale |
| `Risultati` | una riga per cella, aggiornata in place all'ultimo tentativo: stato, metrica principale, giusti con intervallo di confidenza, tempi, pp/tg, KLD, parametri, errore |
| `Log` | il log del laboratorio (ultime 3000 righe) |

Installazione, una volta:

1. Nel foglio: *Estensioni > Apps Script*, incolla `lab/sheets_apps_script.gs`, salva.
2. *Impostazioni progetto > Proprietà dello script*: `LAB_TOKEN` = una stringa lunga a caso
   (`openssl rand -hex 24`).
3. Seleziona la funzione `setup` ed eseguila una volta (chiede i permessi).
4. *Esegui il deployment > Nuovo deployment > App web*, "Esegui come: Me", "Chi può accedere:
   Chiunque". Copia l'URL che finisce con `/exec`.
5. Su ogni macchina, in `lab/.env` (escluso da git):

   ```bash
   LAB_SHEETS_URL=https://script.google.com/macros/s/.../exec
   LAB_SHEETS_TOKEN=la-stessa-stringa-del-punto-2
   ```

6. `./lab/lab.sh sync` manda tutto quello che c'è già nel database e verifica il collegamento.

Da lì in poi `run` aggiorna il foglio da solo. L'invio avviene in un thread separato: se Google non
risponde i test proseguono, e `sync` rimanda tutto in seguito. "Chiunque" vuol dire che l'URL non
richiede un login, ma senza il token lo script rifiuta la richiesta: non pubblicare l'URL insieme al
token. Se modifichi lo script: *Gestisci deployment > Modifica > Versione: nuova* (l'URL non cambia).
NASGUL e Z87 possono scrivere sullo stesso foglio: le righe hanno la macchina nella chiave.

## matrix.yaml

- `models`: nell'ordine di esecuzione. `sources.gguf.repo` è il repo Hugging Face; il file si
  trova dal nome della quantizzazione. Per un file già scaricato:
  `quants: {Q4_K_M: {local_path: /percorso/file.gguf}}`. `llamacpp_args` per flag specifici
  (es. `--swa-full` per Gemma). `allowed_quants`, `engines`, `machines` restringono.
- `quants`: con `make` una quantizzazione viene prodotta in locale con `llama-quantize` dal BF16,
  con o senza imatrix: è il confronto pulito "stessa base, cambia solo la calibrazione".
- `engines`: `kind: llamacpp` per llama.cpp e derivati, `kind: generic` per qualunque server
  OpenAI descritto con comando e mappa dei parametri (un motore nuovo non richiede codice).
- `tests`: opzioni di ciascun test. `est` serve solo alle stime di `plan`.
- `suites`: gruppi di test riutilizzabili.
- `sweeps`: cosa incrociare. Le celle uguali in sweep diversi si eseguono una volta.
  `params` è una griglia (prodotto cartesiano), `params_list` un elenco esplicito.

## I test (versione ridotta)

Su NASGUL un token di prompt costa circa 0,17 s, quindi la prima cosa tagliata sono i prompt
lunghi, poi il numero di casi (mai sotto ~40). I test di qualità pubblici girano solo sulla Z87
(`quality_machines` in `matrix.yaml`): a temperatura 0 l'accuratezza non dipende dalla CPU.
NASGUL fa llama-bench e il test intenti, che misura qualità e latenza reale insieme.

| Nome | Tipo | Riduzione rispetto al piano |
|---|---|---|
| `llama_bench` | velocità | nessuna: pp512/tg128, 3 ripetizioni. Solo motori llama.cpp |
| `kld` | quantizzazione | 10 chunk da 512 token per corpus (wikitext-2 e Wikipedia IT) invece di 20 |
| `ifeval` | istruzioni | 80 casi, risposta massima 512 token |
| `multi_if` | istruzioni IT | 40 casi, 2 turni invece di 3 (il terzo quasi raddoppia il contesto) |
| `json_free` / `json_grammar` | formato | JSONSchemaBench zero-shot, solo schemi entro 1500 caratteri, 50 casi; stessi casi senza e con la grammatica del motore. Sostituisce la versione lm-eval 2-shot (prompt ~1500 token) |
| `evalita_ner_re` | italiano nativo | un solo dataset NER (ADG) + RE, due prompt ciascuno, 25 casi per sottotask |
| `evalita_sa` | sentiment | 100 casi, forma generativa, F1 SENTIPOLC come l'originale |
| `belebele_it` / `belebele_en` | comprensione | 80 casi ciascuno |
| `bfcl` | strumenti | 40 `simple_python` + 40 `irrelevance`, controllo argomenti semplificato |
| `gilpa_intent` | custom | nessuna: `bench.py` come scatola nera, su entrambe le macchine |
| `gsm8k` | ragionamento | 50 casi, solo nel confronto thinking on/off |

Esclusi (in `matrix.yaml` con `enabled: false`, riattivabili): riassunto Evalita (articoli da
~1100 token), JSONSchemaBench via lm-eval, MMLU-ProX, LocalScore.

Suite: `base` per ogni modello in Q4_K_M; `controllo` (llama-bench, KLD, Belebele IT, JSON
libero, test intenti) per il Q8_0 di controllo, la scansione delle quantizzazioni e il 9B
compresso; `motori` (llama-bench, test intenti, JSON con grammatica) per il confronto tra motori,
perché a parità di pesi i motori cambiano velocità, template e grammatica, non la conoscenza.

Stima con `plan` (ordine di grandezza): circa 20 ore su NASGUL e 7 giorni sulla Z87. Il piano
integrale era di 83 giorni su NASGUL. Per un modello in suite base sulla Z87 servono circa 6 ore,
e i risultati arrivano un modello alla volta.

Belebele, Evalita SA, JSONSchema e BFCL usano prompt e valutatori nostri, perché llama-server non
espone i logprob per la scelta multipla. I numeri sono confrontabili tra le nostre configurazioni;
con le classifiche pubbliche vale il confronto relativo, non il valore assoluto. Nell'articolo va
dichiarata la riduzione (casi, turni, filtro sugli schemi).

## Consumo

`power.cmd` in `matrix.yaml` è un comando che stampa i watt istantanei (una presa smart con API
HTTP). Se impostato, ogni esecuzione registra energia e potenza media, e l'export calcola i token
generati per joule. Se è `null` i campi restano vuoti.

## Motori: stato

| Motore | Stato |
|---|---|
| llama.cpp nativo, Docker ufficiale | collaudati su un modello di prova (avvio, bench, perplexity, quantize) |
| ik_llama.cpp | stessa riga di comando di llama.cpp con `-fa` senza valore: da verificare al primo giro |
| Ollama | importa il GGUF con un Modelfile. Thinking off via `think: false`: da verificare |
| KoboldCpp, llamafile | binari scaricati da GitHub; URL e opzioni da verificare |
| OpenVINO Model Server | scarica il modello da HF da sé; solo Z87 |
| ONNX Runtime GenAI, Transformers | server Python minimo in `shims/openai_shim.py`, senza grammatiche né strumenti |
| bitnet.cpp | la build ufficiale passa da `setup_env.py`: probabile che serva adattare `cmake_flags` o puntare `bin_dir` a una build fatta a mano |

## Aggiungere cose

- **Un modello**: una voce in `models`, nella posizione in cui vuoi che venga eseguito.
- **Un motore con server OpenAI**: una voce `kind: generic` con `cmd` e `param_args`.
- **Un test**: una classe in `lab/tests/` con `run()` che restituisce metriche e campioni, e
  `depends_on`; poi la registri in `lab/tests/__init__.py`. Se è un task lm-eval generativo basta
  una voce `kind: lmeval` in `matrix.yaml`.
