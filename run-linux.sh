#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x "venv313/bin/python" ]; then
    python3 -m venv venv313
fi

venv313/bin/python -m pip install -r requirements.txt
exec venv313/bin/python main.py "$@"