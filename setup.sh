#!/usr/bin/env bash
# Crea l'ambiente Python del laboratorio in lab/.venv (non tocca il Python di sistema).
# Prerequisiti di sistema (Debian/OMV):  apt install python3-venv git cmake build-essential docker.io
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q -U pip wheel
# PyTorch solo CPU (serve alle metriche di Evalita e al motore transformers): evita i ~2 GB di CUDA
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q --no-deps bfcl-eval
# dati NLTK usati dai verificatori IFEval (IFEval e Multi-IF)
.venv/bin/python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"
echo
echo "Ambiente pronto. Da qui in poi:  ./lab/lab.sh <comando>   (es. ./lab/lab.sh plan)"
