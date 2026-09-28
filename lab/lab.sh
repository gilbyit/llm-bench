#!/usr/bin/env bash
# Lancia il laboratorio con il Python del suo ambiente. Uso: ./lab/lab.sh plan | run | status ...
here="$(cd "$(dirname "$0")" && pwd)"
py="$here/.venv/bin/python"
[ -x "$py" ] || { echo "ambiente assente: esegui prima $here/setup.sh"; exit 1; }
cd "$here/.." && exec "$py" -m lab "$@"
