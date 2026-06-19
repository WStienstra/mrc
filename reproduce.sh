#!/usr/bin/env bash
# MRC reproduce script.
# Runs the realbench benchmark from a fresh clone and prints the headline
# table. The original results are committed in bench/realbench-results.txt
# so you can diff if the numbers ever change.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> Installing Python deps (numpy, matplotlib)"
python3 -m pip install --quiet --user -r requirements.txt

echo "==> Smoke-importing the codec"
python3 -c "import sys; sys.path.insert(0, 'src'); import mrc, mrc3; print('OK')"

echo "==> Running realbench (this takes a few minutes)"
cd bench
python3 realbench.py

echo
echo "==> DONE. Compare bench/realbench-results.txt to the committed copy."
echo "==> Headline table (first slice rows):"
head -25 realbench-results.txt
