#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# venv neu erstellen falls es nicht funktioniert (z.B. NixOS-Pfade ungültig)
if [ ! -f "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import sys; sys.exit(0)" &>/dev/null; then
    echo "venv ungültig oder fehlt – wird neu erstellt..."
    python3 -m venv --clear "$VENV"
    "$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

"$VENV/bin/python" -m streamlit run "$SCRIPT_DIR/app.py"
