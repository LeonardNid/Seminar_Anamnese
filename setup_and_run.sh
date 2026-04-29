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
#
# Test-Durchlauf (nur 2 kleine Dateien):
#   TEST_MODE=1 WHISPER_MODEL=small OLLAMA_MODEL=llama3.2:1b \
#   bash setup_and_run.sh https://github.com/USER/Seminar.git
# =============================================================================

set -euo pipefail

GITHUB_REPO="$(echo "${1:-}" | xargs)"
PROJECT_DIR="$HOME/Seminar"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
OLLAMA_MODEL="${OLLAMA_MODEL:-hf.co/QuantFactory/Llama-3.1-SauerkrautLM-70b-Instruct-GGUF:Q4_K_M}"
TEST_MODE="${TEST_MODE:-0}"

# ── Farben für Output ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail() {
  echo -e "${RED}[FAIL]${NC}  $1"
  exit 1
}

# ── Repo-URL prüfen ───────────────────────────────────────────────────────────
if [ -z "$GITHUB_REPO" ]; then
  fail "Keine GitHub-URL angegeben. Aufruf: bash setup_and_run.sh https://github.com/USER/Seminar.git"
fi

# ── HF Token (optional) ───────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
  warn "Kein HF_TOKEN gesetzt — Speaker-Diarization wird übersprungen."
  HF_TOKEN=""
fi

# ── Pre-Flight-Checks ────────────────────────────────────────────────────────
if [ "$TEST_MODE" = "1" ]; then
  warn "TEST_MODE=1 — Pre-Flight-Checks (GPU, Disk) werden übersprungen."
else

step "Pre-Flight-Checks …"

# GPU: nvidia-smi muss da sein; wenn nicht → falsches AMI gewählt
if ! command -v nvidia-smi &>/dev/null; then
  fail "nvidia-smi nicht gefunden — falsches AMI. Starte eine Instanz mit
  'Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4 (Ubuntu 22.04)' (im EC2-Console-Suchfeld)"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  | head -1 \
  | while IFS=',' read -r gpu_name gpu_mem; do
      step "GPU erkannt: ${gpu_name} |${gpu_mem}"
    done

# Disk-Space: mindestens 80 GB frei (Ollama ~40GB, Whisper ~3GB, venv ~5GB, Reserve)
FREE_GB=$(df -BG / | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [ "${FREE_GB:-0}" -lt 80 ]; then
  fail "Nur ${FREE_GB} GB auf / frei — mindestens 80 GB benötigt. Starte die Instanz mit ≥100 GB EBS gp3 als Root-Volume."
fi
step "Disk-Space OK: ${FREE_GB} GB frei"

fi  # end TEST_MODE skip

# ── System-Pakete ─────────────────────────────────────────────────────────────
step "System-Pakete installieren …"
sudo apt-get update -y -qq
sudo apt-get install -y -qq \
  python3 python3-pip python3-venv \
  git ffmpeg curl

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
  ollama serve >/tmp/ollama.log 2>&1 &
  echo "Warte auf Ollama-Server …"
  for i in $(seq 1 60); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
      echo "Ollama bereit nach ${i}s"
      break
    fi
    if [ "$i" = "60" ]; then
      fail "Ollama-Server nicht erreichbar nach 60s — siehe /tmp/ollama.log"
    fi
    sleep 1
  done
fi

# ── Modell pullen ─────────────────────────────────────────────────────────────
step "Ollama-Modell pullen: ${OLLAMA_MODEL} …"
for _attempt in 1 2 3; do
  if ollama pull "$OLLAMA_MODEL"; then
    break
  fi
  warn "Ollama-Pull Versuch ${_attempt}/3 fehlgeschlagen — retry in 10s"
  sleep 10
  if [ "$_attempt" = "3" ]; then
    fail "Ollama-Pull nach 3 Versuchen gescheitert"
  fi
done
ollama list | grep -qF "${OLLAMA_MODEL%%:*}" \
  || fail "Ollama-Modell nach Pull nicht in 'ollama list' — Pull unvollständig"

# ── Repo klonen ───────────────────────────────────────────────────────────────
step "GitHub-Repo klonen (nur benötigte Dateien) …"
if [ -d "$PROJECT_DIR/.git" ]; then
  warn "Repo existiert bereits — führe git pull aus."
  git -C "$PROJECT_DIR" pull
else
  git clone --no-checkout --depth 1 "$GITHUB_REPO" "$PROJECT_DIR"
  git -C "$PROJECT_DIR" sparse-checkout init --cone
  git -C "$PROJECT_DIR" sparse-checkout set audio batch_ec2.py requirements.txt
  git -C "$PROJECT_DIR" checkout main
fi

cd "$PROJECT_DIR"
mkdir -p logs
step "Arbeitsverzeichnis: $(pwd)"

# ── Python-Umgebung ───────────────────────────────────────────────────────────
step "Python-venv einrichten …"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
step "Python-Abhängigkeiten installiert."

if [ "$TEST_MODE" != "1" ]; then
  step "Smoke-Test: torch sieht GPU …"
  python -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False — falsches torch-Wheel installiert (CPU statt CUDA)')
print(f'  torch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')
" || fail "Smoke-Test gescheitert — Batch-Start abgebrochen"
fi

# ── .env schreiben ────────────────────────────────────────────────────────────
step ".env-Datei erstellen …"
cat >.env <<EOF
HF_TOKEN=${HF_TOKEN}
WHISPER_MODEL=${WHISPER_MODEL}
OLLAMA_MODEL=${OLLAMA_MODEL}
TEST_MODE=${TEST_MODE}
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
echo "  Fortschritt verfolgen:  tail -f ~/Seminar/logs/batch_ec2_log.txt"
echo "  LLM live beobachten:    tail -f ~/Seminar/logs/llm_live.txt"
echo "  Audio überspringen:     touch ~/Seminar/logs/skip_current"
echo "  Batch abbrechen:        Ctrl+C  (Checkpoint wird gespeichert, Neustart möglich)"
echo ""

python batch_ec2.py

echo ""
step "Fertig! Ergebnisse in history.json"
python3 -c "
import json
h = json.load(open('history.json'))
print(f'  {len(h)} Einträge in history.json')
"
