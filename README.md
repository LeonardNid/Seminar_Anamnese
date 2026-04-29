# Seminar Anamnese — AWS EC2 Quick Start

## Instanz aufsetzen

| Setting | Wert |
|---|---|
| **AMI** | `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4 (Ubuntu 22.04)` (EC2-Console-Suche) |
| **Instance-Typ** | `g5.2xlarge` — A10G-GPU (24 GB VRAM), 8 vCPUs, 32 GB RAM |
| **Root-Disk** | ≥ **100 GB** EBS gp3 (Default 30 GB reicht nicht — Ollama 70b-Modell ~40 GB allein) |
| **Security Group** | Port 22 (SSH) genügt für den Batch-Run |

## One-Command-Start

```bash
HF_TOKEN=hf_xxx bash <(curl -fsSL https://raw.githubusercontent.com/LeonardNid/Seminar_Anamnese/main/setup_and_run.sh) https://github.com/LeonardNid/Seminar_Anamnese.git
```

`HF_TOKEN` wird für pyannote Speaker-Diarization benötigt. Ohne Token läuft der Batch ohne Diarization (Whisper-Rohtext wird direkt genutzt).

## Erwartete Ausgabe der Pre-Flight-Checks (erste ~2 Min)

```
[SETUP] Pre-Flight-Checks …
[SETUP] GPU erkannt: NVIDIA A10G | 23028 MiB
[SETUP] Disk-Space OK: 87 GB frei
[SETUP] System-Pakete installieren …
[SETUP] Ollama installieren …
[SETUP] Ollama-Server starten …
Ollama bereit nach 3s
[SETUP] Ollama-Modell pullen: hf.co/QuantFactory/Llama-3.1-SauerkrautLM-70b-Instruct-GGUF:Q4_K_M …
```

Der Ollama-Pull des 70b-Modells dauert **30–60 Minuten** (≈40 GB) — das ist normal. Danach folgt der automatische Batch-Start.

## Batch überwachen (zweites SSH-Fenster)

```bash
tail -f ~/Seminar/logs/batch_ec2_log.txt   # Fortschritt & Timing
tail -f ~/Seminar/logs/llm_live.txt        # LLM-Tokens live
touch ~/Seminar/logs/skip_current          # aktuelle Datei überspringen
```

## Test mit kleinen Modellen (schnelle Iteration)

```bash
TEST_MODE=1 WHISPER_MODEL=small OLLAMA_MODEL=llama3.2:1b \
  bash <(curl -fsSL https://raw.githubusercontent.com/LeonardNid/Seminar_Anamnese/main/setup_and_run.sh) \
  https://github.com/LeonardNid/Seminar_Anamnese.git
```

## Ergebnis sichern

Nach dem Batch-Run `history.json` herunterladen:

```bash
scp -i KEY.pem ubuntu@EC2_IP:~/Seminar/history.json ./history_ec2_run4.json
```
