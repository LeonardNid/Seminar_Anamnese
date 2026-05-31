# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A proof-of-concept pipeline for AI-assisted medical documentation. Audio recordings of doctor-patient conversations are processed through three stages:

1. **STT** — Speech-to-text via Whisper (local), Speechmatics, or AssemblyAI (cloud), optionally with pyannote speaker diarization to produce `SPEAKER_00:` / `SPEAKER_01:` labels
2. **Format** — LLM identifies which speaker label is the doctor/patient (JSON only); a deterministic script replaces labels and adds line breaks per speaker turn (`speaker_format.py`)
3. **SOAP** — LLM converts the formatted transcript into structured SOAP medical documentation

## Versioning: v1 (Präsentation) ↔ v2 (ongoing)

All v1 results (Präsentation stand) are frozen in `results/v1_praesentation/` and `docs/v1_praesentation/`. The v1 analysis scripts still point there. New results go into `results/v2/` and `docs/v2/`.

## Running the project

**Streamlit UI (local development):**
```bash
# On NixOS — PyArrow needs system libs from nix-shell
bash start.sh

# On standard Linux/Ubuntu
source .venv/bin/activate
streamlit run app.py
```

**Run v1 analysis scripts** (always from their directory so relative imports work):
```bash
cd skript/wer  && python3 wer_whisper_sauerkraut.py   # regenerate WER doc → docs/v1_praesentation/wer/
cd skript/llm  && python3 llm_check_whisper_llama32.py # LLM-check doc → docs/v1_praesentation/llm/
cd skript/soap && python3 soap_strukturcheck.py        # SOAP structural check → docs/v1_praesentation/
cd skript/soap && python3 soap_eval_runner.py          # Claude SOAP evaluator (55 entries) → results/v1_praesentation/
cd skript/soap && python3 soap_eval_auswertung.py      # SOAP eval summary → docs/v1_praesentation/
cd skript/soap && python3 soap_eval_test.py            # test one SOAP eval entry
```

**Run v2 scripts:**
```bash
cd skript/format   && python3 redo_formatting.py          # re-derive formatted with new logic → results/v2/
cd skript/soap_v2  && python3 soap_prompt_test.py         # 12 SOAP generations (GT input) → results/v2/
cd skript/soap_v2  && python3 soap_prompt_test_eval.py    # evaluate with Claude → results/v2/
cd skript/soap_v2  && python3 soap_prompt_test_auswertung.py  # comparison report → docs/v2/
```

**Convert video to audio:**
```bash
nix-shell -p ffmpeg --run 'ffmpeg -i "input.mp4" -vn -c:a copy "output.m4a"'
```

## Architecture

### Execution modes

Batch scripts live in `skript/batches/` (moved from root). Each script ran one combination and wrote results to `results/v1_praesentation/`:

| Script | STT | LLM | Output |
|--------|-----|-----|--------|
| `batch_local.py` | Whisper turbo (local) | SauerkrautLM 8b | `results/v1_praesentation/sauerkraut.json` |
| `batch_llama32.py` | Whisper turbo (reused STT) | llama3.2 | `results/v1_praesentation/history.json` |
| `batch_cloud.py` | Speechmatics (cloud) | GPT-4o | `results/v1_praesentation/speechmatics.json` |
| `batch_assemblyai.py` | AssemblyAI (cloud) | GPT-4o | `results/v1_praesentation/assemblyai.json` |
| `batch_gemma4.py` | Whisper turbo (local) | gemma4 | `results/v1_praesentation/gemma4.json` |
| `batch_pwc.py` | All 5 combinations | — | Single PWC file only; **HF_TOKEN mandatory** |
| `batch_ec2.py` | Whisper large-v3 (EC2) | SauerkrautLM 70b | future run |

### Formatting module: `speaker_format.py` (v2)

Handles all three STT label formats robustly:
- **Whisper+pyannote** (inline): `SPEAKER_00: text`, `SPEAKER_??: text`
- **AssemblyAI** (inline): `Speaker A: text`, `Speaker B: text`
- **Speechmatics** (label on own line): `SPEAKER: S1\n<text lines>\nSPEAKER: S2\n…`

API: `parse_turns(raw)` → `distinct_labels(turns)` → `identify_roles(raw, labels, client, model)` → `apply_mapping(turns, mapping)`

The LLM only returns a JSON role-mapping `{label: "Arzt"/"Name"/"Patient(in)"}`. Text replacement and per-turn line breaks are done deterministically.

### Data files

| File | Contents |
|------|----------|
| `results/v1_praesentation/history_no_speaker.json` | **v1 primary analysis source** — all 55 runs (5 combos × 11 scenarios) with speaker labels stripped |
| `results/v1_praesentation/soap_eval_results.json` | Claude's SOAP quality evaluation for all 55 runs |
| `results/v1_praesentation/*.json` | Per-combination raw results (with speaker labels intact) |
| `results/v2/history_v2_formatted.json` | v2 re-derived formatted transcripts (deterministic label replacement) |
| `results/v2/soap_prompt_test.json` | 12 SOAP generations for prompt engineering test |
| `results/v2/soap_prompt_test_eval.json` | Claude evaluations for the 12 SOAP tests |
| `docs/Seminar Ground Truth Texte.md` | Reference transcripts used for WER calculation |
| `docs/speaker/speaker_ground_truth.md` | Expected speaker labels per scenario |

### Analysis scripts (`skript/`)

**`skript/wer/`** — WER (Word Error Rate): reads `results/v1_praesentation/history_no_speaker.json` vs Ground Truth. Writes to `docs/v1_praesentation/wer/`.

**`skript/llm/`** — LLM fidelity check: reads `results/v1_praesentation/history_no_speaker.json`. Writes to `docs/v1_praesentation/llm/`.

**`skript/soap/`** — v1 SOAP evaluation (reads/writes `v1_praesentation/`):
- `soap_strukturcheck.py` — detects presence/emptiness of S/O/A/P sections via regex
- `soap_eval_runner.py` — calls `claude -p` for each of 55 entries; saves to `results/v1_praesentation/soap_eval_results.json`
- `soap_eval_auswertung.py` — summary → `docs/v1_praesentation/soap_eval_auswertung.md`
- `soap_eval_prompt.md` — the actual prompt sent to Claude (shared between v1 and v2 eval)

**`skript/format/`** — v2 formatting validation:
- `redo_formatting.py` — applies new deterministic formatting to all 55 v1 entries; validates word-for-word text preservation across all 3 STT label formats

**`skript/soap_v2/`** — v2 SOAP prompt engineering:
- `soap_prompts.py` — SOAP_BASELINE and SOAP_KANDIDAT prompt constants
- `soap_prompt_test.py` — generates 12 SOAPs (3 scenarios × 2 models × 2 prompts) using Ground Truth as input
- `soap_prompt_test_eval.py` — evaluates 12 SOAPs with Claude
- `soap_prompt_test_auswertung.py` — comparison report Baseline vs. Kandidat

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
