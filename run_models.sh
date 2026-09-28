#!/usr/bin/env bash
# Avvia llama-server (Docker, CPU) per ogni modello in models.txt e lancia bench.py.
# Uso: ./run_models.sh [models.txt] [argomenti extra per bench.py, es. --tasks intent_v2 --repeat 2]
#
# Formato di models.txt:   etichetta   repo:quantizzazione   [opzioni extra per llama-server]
# Esempio (Gemma ha bisogno di --swa-full perche' la cache del prompt funzioni):
#   gemma3-4b   ggml-org/gemma-3-4b-it-GGUF:Q4_K_M   --swa-full
set -euo pipefail
cd "$(dirname "$0")"

MODELS_FILE=${1:-models.txt}; shift || true
PORT=${PORT:-8480}
THREADS=${THREADS:-2}          # i5-3470T: 2 core fisici; prova anche 4
CTX=${CTX:-4096}
IMAGE=${IMAGE:-ghcr.io/ggml-org/llama.cpp:server}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}   # include il download del GGUF
NAME=gilpa-bench

mkdir -p models results
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

while read -r label spec extra; do
  [[ -z "${label:-}" || "$label" == \#* ]] && continue
  echo "=== $label  ($spec) ${extra:-}"
  cleanup
  # -np 1: uno slot solo, tutto il contesto per lui. $extra volutamente senza virgolette.
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" -p "$PORT:8080" \
    -v "$PWD/models:/models" -e LLAMA_CACHE=/models \
    "$IMAGE" -hf "$spec" -c "$CTX" -t "$THREADS" -np 1 --jinja --host 0.0.0.0 --port 8080 ${extra:-} >/dev/null

  waited=0
  until curl -sf "http://localhost:$PORT/health" >/dev/null; do
    if [[ -z "$(docker ps -q -f name=$NAME)" ]]; then
      echo "!! container terminato, ultime righe di log:"; docker logs --tail 30 "$NAME" || true
      continue 2
    fi
    (( waited >= LOAD_TIMEOUT )) && { echo "!! timeout caricamento"; continue 2; }
    sleep 10; waited=$((waited + 10))
  done

  # conferma quali istruzioni CPU usa il backend (atteso: AVX = 1, AVX2 = 0)
  docker logs "$NAME" 2>&1 | grep -m1 -i "system_info" || true

  python3 bench.py --base-url "http://localhost:$PORT/v1" --model "$label" --label "$label" --local "$@" \
    | tee "results/$label-v2-console.txt"
done < "$MODELS_FILE"
