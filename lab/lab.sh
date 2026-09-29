#!/usr/bin/env bash
# Lancia il laboratorio con il Python del suo ambiente. Uso: ./lab/lab.sh plan | run | status ...
here="$(cd "$(dirname "$0")" && pwd)"
py="$here/.venv/bin/python"
[ -x "$py" ] || { echo "ambiente assente: esegui prima $here/setup.sh"; exit 1; }
# segreti locali (URL e token del foglio Google): lab/.env, escluso da git
[ -f "$here/.env" ] && { set -a; . "$here/.env"; set +a; }
cd "$here/.." && exec "$py" -m lab "$@"
