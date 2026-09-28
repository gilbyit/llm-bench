# gilpa-llm-bench

Banco di prova per decidere se (e dove) far girare l'LLM di GILPA. Misura velocità e qualità
su **qualsiasi endpoint OpenAI-compatibile**: llama-server su NASGUL, Groq, un'eventuale GPU o un
worker cloud. Stessi casi, stessi numeri, confronto diretto.

## Cosa misura

Tre task, costruiti sui casi d'uso reali di `GILPA_schema.md`:

| task | cosa fa | metrica di qualità |
|---|---|---|
| `intent_short` | classifica intent + entità, system prompt corto (~300 token) | intent corretto, entità corrette |
| `intent_full` | stesso task con schema DB e alias nel prompt (~650 token) | idem, e mostra il costo del prefill |
| `sql` | genera una SELECT, eseguita su un DB SQLite di esempio | risultato identico alla query di riferimento |

Velocità: token del prompt, prefill (`pp t/s`), generazione (`tg t/s`), tempo totale per richiesta.
Output vincolato a JSON schema (grammatica di llama.cpp), quindi `json%` misura soprattutto i
provider che non lo applicano.

## Uso su NASGUL

Requisiti: Docker, python3, curl. Nessuna dipendenza Python.

```bash
chmod +x run_models.sh
./run_models.sh                      # warm: prompt cache attiva, come in produzione
./run_models.sh models.txt --cold    # cold: prefill completo a ogni richiesta
THREADS=4 ./run_models.sh            # confronto thread fisici vs logici
```

I GGUF finiscono in `./models` (riusati tra un giro e l'altro), i CSV in `./results`.

Baseline cloud con lo stesso script:

```bash
python3 bench.py --base-url https://api.groq.com/openai/v1 --model <modello> \
  --api-key-env GILPA_LLM_API_KEY --label groq
```

(se il provider rifiuta `json_schema`, aggiungi `--format object`)

## Cosa controllare durante il test

- Nel log deve comparire `AVX = 1` e `AVX2 = 0`: conferma che il backend usa la variante giusta.
- Temperature CPU (`watch sensors`): NASGUL è fanless, un giro lungo al 100% è un test termico.
- Gli altri servizi (OMV, GILPA, torrent) rallentano i numeri: annota cosa girava.

## Come leggere i risultati

- **warm vs cold**: GILPA ha un system prompt fisso, quindi in produzione vale il warm
  (llama-server riusa la KV cache del prefisso). Il cold è il caso peggiore: primo messaggio,
  cambio di prompt, riavvio.
- Soglia di usabilità per una chat: `wall med` sotto ~5 s, `wall max` sotto ~15 s.
- Soglia di affidabilità: `acc%` intent almeno 90, `sql` almeno 80. Sotto, il modello va
  usato solo come classificatore con query costruite da template, non per SQL libero.
