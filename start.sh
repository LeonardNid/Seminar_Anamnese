#!/usr/bin/env bash
# Startet die Anwendung innerhalb der benötigten Nix-Umgebung,
# da PyArrow Systembibliotheken (wie libstdc++.so.6) benötigt.
nix-shell --command "./.venv/bin/python -m streamlit run app.py"
