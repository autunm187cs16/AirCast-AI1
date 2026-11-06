#!/usr/bin/env bash
# AirCast AI launcher (macOS / Linux)
set -e
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi
exec ./.venv/bin/python -m streamlit run app.py
