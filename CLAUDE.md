# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A proof-of-concept pipeline for AI-assisted medical documentation. Audio recordings of doctor-patient conversations are processed through three stages:

1. **STT** — Speech-to-text via Whisper (local), Speechmatics, or AssemblyAI (cloud), optionally with pyannote speaker diarization to produce `SPEAKER_00:` / `SPEAKER_01:` labels
2. **Format** — LLM replaces generic speaker labels with `Arzt:` / `[Patientenname]:` (text otherwise unchanged)
3. **SOAP** — LLM converts the formatted transcript into structured SOAP medical documentation

All batch results are stored in `results/` as per-combination JSON files. The project is in evaluation/analysis phase — all 5 model combinations have been run and are being analysed.

## Running the project

**Streamlit UI (local development):**
```bash
# On NixOS — PyArrow needs system libs from nix-shell
bash start.sh

# On standard Linux/Ubuntu
source .venv/bin/activate
streamlit run app.py
```

**Run analysis scripts** (always from their directory so relative imports work):
```bash
cd skript/wer  && python3 wer_whisper_sauerkraut.py   # regenerate a WER doc
cd skript/llm  && python3 llm_check_whisper_llama32.py # regenerate an LLM-check doc
cd skript/soap && python3 soap_strukturcheck.py        # SOAP structural check
cd skript/soap && python3 soap_eval_runner.py          # run Claude SOAP evaluator (55 entries)
cd skript/soap && python3 soap_eval_auswertung.py      # regenerate SOAP eval summary doc
cd skript/soap && python3 soap_eval_test.py            # test one SOAP eval entry
```

**Convert video to audio:**
```bash
nix-shell -p ffmpeg --run 'ffmpeg -i "input.mp4" -vn -c:a copy "output.m4a"'
```

## Architecture

### Execution modes

Batch scripts live in `skript/batches/` (moved from root). Each script ran one combination and wrote results to `results/`:

| Script | STT | LLM | Output |
|--------|-----|-----|--------|
| `batch_local.py` | Whisper turbo (local) | SauerkrautLM 8b | `results/sauerkraut.json` |
| `batch_llama32.py` | Whisper turbo (reused STT) | llama3.2 | `results/history.json` |
| `batch_cloud.py` | Speechmatics (cloud) | GPT-4o | `results/speechmatics.json` |
| `batch_assemblyai.py` | AssemblyAI (cloud) | GPT-4o | `results/assemblyai.json` |
| `batch_gemma4.py` | Whisper turbo (local) | gemma4 | `results/gemma4.json` |
| `batch_pwc.py` | All 5 combinations | — | Single PWC file only; **HF_TOKEN mandatory** |
| `batch_ec2.py` | Whisper large-v3 (EC2) | SauerkrautLM 70b | future run |

### Data files

| File | Contents |
|------|----------|
| `results/history_no_speaker.json` | **Primary analysis source** — all 55 runs (5 combos × 11 scenarios) with speaker labels stripped from `raw` and `formatted` fields |
| `results/soap_eval_results.json` | Claude's SOAP quality evaluation for all 55 runs |
| `results/*.json` | Per-combination raw results (with speaker labels intact) |
| `docs/Seminar Ground Truth Texte.md` | Reference transcripts used for WER calculation |
| `docs/speaker/speaker_ground_truth.md` | Expected speaker labels per scenario |

### Analysis scripts (`skript/`)

Three analysis layers, each with a shared base module and thin per-combination wrappers:

**`skript/wer/`** — WER (Word Error Rate): compares RAW STT output vs Ground Truth. Uses `wer_base.py`; each wrapper sets `stt_filter`, `llm_filter`, title, output file.

**`skript/llm/`** — LLM fidelity check: compares RAW STT vs FORMATTED (finds where LLM changed text beyond speaker labels). Uses `llm_check_base.py`. Both scripts use `results/history_no_speaker.json` and write to `docs/llm/`.

**`skript/soap/`** — SOAP evaluation:
- `soap_strukturcheck.py` — detects presence/emptiness of S/O/A/P sections via regex
- `soap_eval_runner.py` — calls `claude -p` for each of 55 entries; saves to `results/soap_eval_results.json`; resumes after interruption (skips entries without `fehler` key)
- `soap_eval_auswertung.py` — reads eval results and writes summary to `docs/soap_eval_auswertung.md`
- `soap_eval_prompt.md` — the actual prompt sent to Claude (edit this to change evaluation criteria)

### SOAP evaluation scoring

Per section (S/O/A/P): **0–2 points**, max **8 total**.
- **S and O**: 2=complete, 1=minor gaps or harmless hallucination, 0=hallucination contradicts transcript
- **A and P** (stricter): 2=correct+no hallucinations, 1=minor gap only, **0=any hallucination** (wrong diagnosis / unplanned intervention)
- Urteil: ≥7 → `akzeptabel`, 4–6 → `ueberarbeitung_noetig`, ≤3 → `nicht_verwendbar`
- Auto-skip (no Claude call): empty SOAP or LLM error rate >90%

### `history_no_speaker.json` schema (one entry)

```json
{
  "id": "unix timestamp string",
  "timestamp": "DD.MM.YYYY HH:MM:SS",
  "raw": "STT output (speaker prefixes stripped)",
  "formatted": "LLM-formatted transcript (speaker prefixes stripped)",
  "soap": "SOAP notes",
  "stt_model": "...",
  "llm_model": "...",
  "audio_file": "audio/filename.wav"
}
```

### Scenario mapping (audio filename → analysis name)

| Audio file | Scenario name |
|---|---|
| `OriginalDC.m4a` | OriginalDC |
| `OriginalDCWhiteNoise.m4a` | OriginalDC+Noise |
| `OriginalLapInMitte.wav` | LapInMitte |
| `OriginalLapBeiArzt.wav` | LapBeiArzt |
| `SelbstkorrekturLapInMitte.wav` | Selbstkorrekturen |
| `UnterbrechungLapInMitte.wav` | Unterbrechungen |
| `GedankenprüngeLapInMitte.wav` | Gedankensprünge |
| `MeinungswechselLapinMitte.wav` | Meinungswechsel |
| `ChaosLapInMitte.wav` | Chaos |
| `Das Anamnesegespräch...wav` | Anamnesegespräch |
| `Anamnesegesrpäch PWC.mp3` | PWC |

### Environment variables (`.env`, never committed)

| Variable | Used by |
|----------|---------|
| `HF_TOKEN` | pyannote speaker diarization (optional; skipped if absent) |
| `WHISPER_MODEL` | faster-whisper model name (default: `large-v3`) |
| `OLLAMA_MODEL` | Ollama model name (default: SauerkrautLM 70b GGUF) |
| `SAUERKRAUT_BASE_URL` | Ollama OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | GPT-4o in app.py |
| `SPEECHMATICS_API_KEY` | Speechmatics cloud STT |
| `TEST_MODE` | `1` = only 2 small files in batch_ec2.py |

## Git remotes

| Remote | URL |
|--------|-----|
| `github` | `https://github.com/LeonardNid/Seminar_Anamnese.git` (primary, EC2 pulls from here) |
| `origin` | Forgejo server via SSH (private) |
