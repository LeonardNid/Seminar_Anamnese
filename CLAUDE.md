# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A proof-of-concept pipeline for AI-assisted medical documentation. Audio recordings of doctor-patient conversations are processed through three stages:

1. **STT** — Speech-to-text via Whisper (local) or Speechmatics (cloud), optionally combined with pyannote speaker diarization to produce `SPEAKER_00:` / `SPEAKER_01:` labels
2. **Format** — LLM replaces generic speaker labels with `Arzt:` / `[Patientenname]:` (text unchanged)
3. **SOAP** — LLM converts the formatted transcript into structured medical documentation (Subjective / Objective / Assessment / Plan)

All results are written to `history.json` (one entry per run, newest first).

## Running the project

**Streamlit UI (local development):**
```bash
# On NixOS — PyArrow needs system libs from nix-shell
bash start.sh

# On standard Linux/Ubuntu
source .venv/bin/activate
streamlit run app.py
```

**Headless batch on EC2 (one command from scratch):**
```bash
HF_TOKEN=hf_xxx bash <(curl -fsSL https://raw.githubusercontent.com/LeonardNid/Seminar_Anamnese/main/setup_and_run.sh) https://github.com/LeonardNid/Seminar_Anamnese.git
```

**Run only 2 small test files (faster iteration):**
```bash
TEST_MODE=1 WHISPER_MODEL=small OLLAMA_MODEL=llama3.2 \
  bash <(curl -fsSL https://raw.githubusercontent.com/LeonardNid/Seminar_Anamnese/main/setup_and_run.sh) \
  https://github.com/LeonardNid/Seminar_Anamnese.git
```

**Monitor a running batch:**
```bash
tail -f ~/Seminar/logs/batch_ec2_log.txt   # progress & timing
tail -f ~/Seminar/logs/llm_live.txt        # live LLM/STT output token by token
touch ~/Seminar/logs/skip_current          # skip current audio (batch continues)
```

**Convert video to audio:**
```bash
nix-shell -p ffmpeg --run 'ffmpeg -i "input.mp4" -vn -c:a copy "output.m4a"'
```

## Architecture

### Two execution modes

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — record/upload audio, choose models, view history |
| `batch_ec2.py` | Headless batch runner for EC2 — no Streamlit dependency |
| `batch_local.py` | Earlier local batch runner (Whisper turbo + Sauerkraut 8b) |
| `batch_cloud.py` | Earlier cloud batch runner (Speechmatics + GPT-4o) |
| `batch_llama32.py` | Reuses Whisper STT from history, reruns LLM with llama3.2 |
| `setup_and_run.sh` | One-command EC2 bootstrap: apt → Ollama → git sparse-checkout → venv → batch |

### `batch_ec2.py` internals

- **GPU auto-detection** at startup; falls back to CPU/int8 if no CUDA
- **STT** streams segments live to `logs/llm_live.txt` as Whisper produces them (generator, not batch)
- **LLM calls** use `stream=True`; tokens are written live to stdout and `logs/llm_live.txt`
- **Skip signal**: `check_skip()` polls `logs/skip_current` every 1 s during LLM streaming and before each phase; on skip the partial text generated so far is saved (not discarded)
- **Checkpoint** (`logs/batch_ec2_checkpoint.json`) tracks completed files so interrupted runs can resume
- All model names and paths are overridable via env vars: `WHISPER_MODEL`, `OLLAMA_MODEL`, `HF_TOKEN`, `TEST_MODE`

### `history.json` schema (one entry)

```json
{
  "id": "unix timestamp string",
  "timestamp": "DD.MM.YYYY HH:MM:SS",
  "name": "optional display name",
  "raw": "raw STT output",
  "formatted": "speaker-labelled transcript",
  "soap": "SOAP notes",
  "stt_model": "...",
  "llm_model": "...",
  "language": "de",
  "audio_file": "audio/filename.wav",
  "audio_size_bytes": 12345678,
  "aborted": true,           // only if manually skipped
  "abort_reason": "manual_skip",
  "stats": {
    "stt_duration_s": 42.1,
    "format_duration_s": 18.3,
    "soap_duration_s": 22.7,
    "total_duration_s": 83.1,
    "raw_char_count": 4200,
    "formatted_char_count": 4250,
    "soap_char_count": 980
  }
}
```

### Audio files

11 recordings live in `audio/` (committed to GitHub, compressed to 16 kHz mono to stay under the 100 MB GitHub limit). Accepted formats: `.wav`, `.mp3`, `.m4a`, `.ogg`.

### Environment variables (`.env`, never committed)

| Variable | Used by |
|----------|---------|
| `HF_TOKEN` | pyannote speaker diarization (optional; skipped if absent) |
| `WHISPER_MODEL` | faster-whisper model name (default: `large-v3`) |
| `OLLAMA_MODEL` | Ollama model name (default: SauerkrautLM 70b GGUF) |
| `SAUERKRAUT_BASE_URL` | Ollama OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | GPT-4o in app.py |
| `SPEECHMATICS_API_KEY` | Speechmatics cloud STT in app.py |
| `TEST_MODE` | `1` = only 2 small files in batch_ec2.py |

## Current status (April 2026)

`history.json` contains results for all 11 audio files across **3 completed batch runs**:

| Run | STT | LLM | Script |
|-----|-----|-----|--------|
| 1 | Whisper large-v3-turbo (local) | SauerkrautLM 8b | `batch_local.py` |
| 2 | Whisper large-v3-turbo (reused) | llama3.2 | `batch_llama32.py` |
| 3 | Speechmatics (cloud) | GPT-4o | `batch_cloud.py` |

**Next step:** A 12th audio file will be added and a 4th batch run will be executed on AWS EC2 using `batch_ec2.py` with Whisper **large-v3** (not turbo) + **SauerkrautLM 70b** via Ollama. This is the highest-quality local setup and requires an instance with ≥48 GB RAM (or GPU with ≥24 GB VRAM).

## Git remotes

| Remote | URL |
|--------|-----|
| `github` | `https://github.com/LeonardNid/Seminar_Anamnese.git` (primary, EC2 pulls from here) |
| `origin` | Forgejo server via SSH (private) |

The EC2 bootstrap does a **sparse checkout** — only `audio/`, `batch_ec2.py`, and `requirements.txt` are downloaded (no Streamlit code or docs).
