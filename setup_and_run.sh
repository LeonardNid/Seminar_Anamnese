#!/usr/bin/env bash
# =============================================================================
# EC2 One-Command Setup — konfigurierbar per Umgebungsvariablen
# =============================================================================
# Produktion (EC2):
#   HF_TOKEN=hf_xxx bash setup_and_run.sh https://github.com/USER/Seminar.git
#
# Test mit leichten Modellen:
#   HF_TOKEN=hf_xxx \
#   WHISPER_MODEL=small \
#   OLLAMA_MODEL=llama3.2 \
#   bash setup_and_run.sh https://github.com/USER/Seminar.git
#
# Ohne Diarization (kein HF_TOKEN nötig):
#   WHISPER_MODEL=small OLLAMA_MODEL=llama3.2 \
#   bash setup_and_run.sh https://github.com/USER/Seminar.git
# =============================================================================

set -euo pipefail

GITHUB_REPO="${1:-}"
PROJECT_DIR="$HOME/Seminar"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
OLLAMA_MODEL="${OLLAMA_MODEL:-hf.co/QuantFactory/Llama-3.1-SauerkrautLM-70b-Instruct-GGUF:Q4_K_M}"

# ── Farben für Output ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

# ── Repo-URL prüfen ───────────────────────────────────────────────────────────
if [ -z "$GITHUB_REPO" ]; then
    fail "Keine GitHub-URL angegeben. Aufruf: bash setup_and_run.sh https://github.com/USER/Seminar.git"
fi

# ── HF Token (optional) ───────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    warn "Kein HF_TOKEN gesetzt — Speaker-Diarization wird übersprungen."
    HF_TOKEN=""
fi

# ── System-Pakete ─────────────────────────────────────────────────────────────
step "System-Pakete installieren …"
sudo apt-get update -y -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    git git-lfs \
    ffmpeg curl

# ── Ollama installieren ───────────────────────────────────────────────────────
step "Ollama installieren …"
if command -v ollama &>/dev/null; then
    warn "Ollama ist bereits installiert: $(ollama --version)"
else
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# ── Ollama starten ────────────────────────────────────────────────────────────
step "Ollama-Server starten …"
if systemctl is-active --quiet ollama 2>/dev/null; then
    warn "Ollama-Service läuft bereits."
else
    ollama serve > /tmp/ollama.log 2>&1 &
    echo "Warte 8 Sekunden auf Ollama …"
    sleep 8
fi

# ── Modell pullen ─────────────────────────────────────────────────────────────
step "Ollama-Modell pullen: ${OLLAMA_MODEL} …"
ollama pull "$OLLAMA_MODEL"

# ── Repo klonen ───────────────────────────────────────────────────────────────
step "GitHub-Repo klonen …"
git lfs install --skip-repo
if [ -d "$PROJECT_DIR/.git" ]; then
    warn "Repo existiert bereits — führe git pull aus."
    git -C "$PROJECT_DIR" pull
    git -C "$PROJECT_DIR" lfs pull
else
    git clone "$GITHUB_REPO" "$PROJECT_DIR"
    git -C "$PROJECT_DIR" lfs pull
fi

cd "$PROJECT_DIR"
step "Arbeitsverzeichnis: $(pwd)"

# ── Python-Umgebung ───────────────────────────────────────────────────────────
step "Python-venv einrichten …"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
step "Python-Abhängigkeiten installiert."

# ── .env schreiben ────────────────────────────────────────────────────────────
step ".env-Datei erstellen …"
cat > .env << EOF
HF_TOKEN=${HF_TOKEN}
WHISPER_MODEL=${WHISPER_MODEL}
OLLAMA_MODEL=${OLLAMA_MODEL}
SAUERKRAUT_BASE_URL=http://localhost:11434/v1
SAUERKRAUT_API_KEY=ollama
OPENAI_API_KEY=not-needed
SPEECHMATICS_API_KEY=not-needed
SPEECHMATICS_URL=https://asr.api.speechmatics.com/v2
EOF
echo ".env geschrieben."

# ── Batch starten ─────────────────────────────────────────────────────────────
step "Starte Batch-Verarbeitung (Whisper ${WHISPER_MODEL} + ${OLLAMA_MODEL}) …"
echo ""
echo "  Fortschritt verfolgen:  tail -f batch_ec2_log.txt"
echo "  Abbrechen:              Ctrl+C  (Checkpoint wird gespeichert, Neustart möglich)"
echo ""

python batch_ec2.py

echo ""
step "Fertig! Ergebnisse in history.json"
python3 -c "
import json
h = json.load(open('history.json'))
print(f'  {len(h)} Einträge in history.json')
"
