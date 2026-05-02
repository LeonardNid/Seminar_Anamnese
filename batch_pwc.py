#!/usr/bin/env python3
"""
Alle 5 Modell-Kombinationen für audio/Anamnesegesrpäch PWC.mp3:
  1. Speechmatics + GPT-4o
  2. AssemblyAI + GPT-4o
  3. Whisper large-v3-turbo + llama3.2
  4. Whisper large-v3-turbo + SauerkrautLM-8b-Instruct
  5. Whisper large-v3-turbo + gemma4:e4b

Whisper-STT wird einmalig ausgeführt und für alle 3 lokalen Runs wiederverwendet.
Checkpoint: logs/batch_pwc_checkpoint.json
Log:        logs/batch_pwc_log.txt
Live-LLM:   logs/llm_live.txt
Skip:       touch logs/skip_current
"""

import os
import glob
import json
import time
import tempfile
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
os.makedirs("logs", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
# Filename uses NFD-encoded ä; glob resolves the actual on-disk name regardless of NFC/NFD
_matches   = glob.glob("audio/Anamnesegesrp*PWC.mp3")
AUDIO_FILE = _matches[0] if _matches else "audio/Anamnesegesrpäch PWC.mp3"
HISTORY_FILE         = "history.json"
CHECKPOINT_FILE      = "logs/batch_pwc_checkpoint.json"
LOG_FILE             = "logs/batch_pwc_log.txt"
LIVE_FILE            = "logs/llm_live.txt"
SKIP_FILE            = "logs/skip_current"
LANGUAGE             = "de"

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
SPEECHMATICS_URL     = os.getenv("SPEECHMATICS_URL", "https://asr.api.speechmatics.com/v2")
ASSEMBLYAI_API_KEY   = os.getenv("ASSEMBLYAI_API_KEY")
ASSEMBLYAI_BASE_URL  = "https://api.assemblyai.com/v2"
HF_TOKEN             = os.getenv("HF_TOKEN")
OLLAMA_API_KEY       = os.getenv("SAUERKRAUT_API_KEY", "ollama")
OLLAMA_BASE_URL      = os.getenv("SAUERKRAUT_BASE_URL", "http://localhost:11434/v1")

SAUERKRAUT_MODEL = "hf.co/QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF:Q4_K_M"
LLAMA32_MODEL    = "llama3.2"
GEMMA4_MODEL     = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

COMBINATIONS = [
    {
        "key":       "speechmatics+gpt4o",
        "stt":       "speechmatics",
        "llm":       "gpt4o",
        "stt_label": "Speechmatics (Cloud)",
        "llm_label": "OpenAI GPT-4o",
    },
    {
        "key":       "assemblyai+gpt4o",
        "stt":       "assemblyai",
        "llm":       "gpt4o",
        "stt_label": "AssemblyAI (Cloud)",
        "llm_label": "OpenAI GPT-4o",
    },
    {
        "key":       "whisper+llama3.2",
        "stt":       "whisper",
        "llm":       "llama3.2",
        "stt_label": "Whisper Large-v3-turbo (Lokal)",
        "llm_label": "llama3.2",
    },
    {
        "key":       "whisper+sauerkraut8b",
        "stt":       "whisper",
        "llm":       "sauerkraut8b",
        "stt_label": "Whisper Large-v3-turbo (Lokal)",
        "llm_label": "Llama-3.1-SauerkrautLM-8b-Instruct",
    },
    {
        "key":       "whisper+gemma4b",
        "stt":       "whisper",
        "llm":       "gemma4b",
        "stt_label": "Whisper Large-v3-turbo (Lokal)",
        "llm_label": GEMMA4_MODEL,
    },
]

# ── Logging ───────────────────────────────────────────────────────────────────
log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")

# ── Skip-Signal ───────────────────────────────────────────────────────────────
def check_skip():
    if os.path.exists(SKIP_FILE):
        os.remove(SKIP_FILE)
        return True
    return False

# ── Checkpoint ────────────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"done": []}

def mark_done(key):
    cp = load_checkpoint()
    if key not in cp["done"]:
        cp["done"].append(key)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)

# ── History ───────────────────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_history(raw, formatted, soap, meta):
    history = load_history()
    entry = {
        "id":        str(int(time.time())),
        "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "name":      "",
        "raw":       raw,
        "formatted": formatted,
        "soap":      soap,
    }
    entry.update(meta)
    history.insert(0, entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
ollama_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)

# ── STT: Speechmatics ─────────────────────────────────────────────────────────
def transcribe_speechmatics(audio_bytes):
    url     = f"{SPEECHMATICS_URL}/jobs/"
    headers = {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}
    config  = {
        "type": "transcription",
        "transcription_config": {
            "language":        LANGUAGE,
            "operating_point": "enhanced",
            "diarization":     "speaker",
        },
    }
    resp = requests.post(
        url,
        headers=headers,
        data={"config": json.dumps(config)},
        files={"data_file": (os.path.basename(AUDIO_FILE), audio_bytes, "audio/mpeg")},
    )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    log(f"  Speechmatics Job: {job_id}")

    status_url     = f"{SPEECHMATICS_URL}/jobs/{job_id}"
    transcript_url = f"{SPEECHMATICS_URL}/jobs/{job_id}/transcript?format=txt"
    polls = 0
    while True:
        time.sleep(5)
        polls += 1
        sr = requests.get(status_url, headers=headers)
        sr.raise_for_status()
        status = sr.json()["job"]["status"]
        if polls % 6 == 0:
            log(f"  Speechmatics polling… status={status} ({polls*5}s)")
        if status == "done":
            break
        elif status in ("rejected", "deleted", "expired"):
            return f"Fehler: Job status={status}"

    tr = requests.get(transcript_url, headers=headers)
    tr.raise_for_status()
    tr.encoding = "utf-8"
    return tr.text

# ── STT: AssemblyAI ───────────────────────────────────────────────────────────
def transcribe_assemblyai():
    headers = {"Authorization": ASSEMBLYAI_API_KEY}

    with open(AUDIO_FILE, "rb") as f:
        up = requests.post(f"{ASSEMBLYAI_BASE_URL}/upload", headers=headers, data=f)
    up.raise_for_status()
    audio_url = up.json()["upload_url"]
    log(f"  AssemblyAI Upload fertig")

    body = {
        "audio_url":     audio_url,
        "speech_models": ["universal-3-pro"],
        "language_code": LANGUAGE,
        "speaker_labels": True,
    }
    sub = requests.post(
        f"{ASSEMBLYAI_BASE_URL}/transcript",
        headers={**headers, "Content-Type": "application/json"},
        json=body,
    )
    sub.raise_for_status()
    job_id = sub.json()["id"]
    log(f"  AssemblyAI Job: {job_id}")

    poll_url = f"{ASSEMBLYAI_BASE_URL}/transcript/{job_id}"
    polls = 0
    while True:
        time.sleep(5)
        polls += 1
        pr = requests.get(poll_url, headers=headers)
        pr.raise_for_status()
        data   = pr.json()
        status = data["status"]
        if polls % 6 == 0:
            log(f"  AssemblyAI polling… status={status} ({polls*5}s)")
        if status == "completed":
            break
        elif status == "error":
            return f"Fehler: {data.get('error', 'unbekannt')}"

    utterances = data.get("utterances") or []
    if not utterances:
        return data.get("text") or "Fehler: Kein Text transkribiert"
    return "\n".join(f"Speaker {u['speaker']}: {u['text']}" for u in utterances)

# ── STT: Whisper + optional pyannote ─────────────────────────────────────────
_whisper_model    = None
_diarize_pipeline = None
_whisper_raw      = None  # cached raw after first run

def _init_whisper():
    global _whisper_model, _diarize_pipeline
    if _whisper_model is not None:
        return

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    log(f"Lade Whisper large-v3-turbo (device={device}, compute_type={compute_type}) …")
    from faster_whisper import WhisperModel
    t0 = time.time()
    _whisper_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    log(f"Whisper geladen in {round(time.time()-t0,1)}s")

    if HF_TOKEN:
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
            log("Lade pyannote speaker-diarization-3.1 …")
            t0 = time.time()
            _diarize_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=HF_TOKEN
            )
            log(f"pyannote geladen in {round(time.time()-t0,1)}s")
        except Exception as e:
            log(f"WARNUNG: pyannote nicht verfügbar ({e}) — ohne Diarisierung")
    else:
        log("HF_TOKEN nicht gesetzt — Whisper ohne Sprecher-Diarisierung")

def transcribe_whisper(audio_bytes):
    global _whisper_raw
    if _whisper_raw is not None:
        log(f"  Whisper-Transkript aus Cache ({len(_whisper_raw)} Zeichen)")
        return _whisper_raw

    _init_whisper()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, _ = _whisper_model.transcribe(
            tmp_path,
            language=LANGUAGE,
            beam_size=5,
            word_timestamps=bool(_diarize_pipeline),
        )
        whisper_segments = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
        ]

        if not _diarize_pipeline:
            raw = " ".join(s["text"] for s in whisper_segments)
        else:
            diarization = _diarize_pipeline(tmp_path).speaker_diarization

            def get_speaker(start, end):
                overlap = {}
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    o = min(turn.end, end) - max(turn.start, start)
                    if o > 0:
                        overlap[speaker] = overlap.get(speaker, 0) + o
                return max(overlap, key=overlap.get) if overlap else "SPEAKER_??"

            lines = []
            current_speaker = None
            current_text    = []
            for seg in whisper_segments:
                speaker = get_speaker(seg["start"], seg["end"])
                if speaker != current_speaker:
                    if current_text:
                        lines.append(f"{current_speaker}: {' '.join(current_text)}")
                    current_speaker = speaker
                    current_text    = [seg["text"]]
                else:
                    current_text.append(seg["text"])
            if current_text:
                lines.append(f"{current_speaker}: {' '.join(current_text)}")
            raw = "\n".join(lines)
    finally:
        os.remove(tmp_path)

    _whisper_raw = raw
    return raw

# ── STT dispatch ──────────────────────────────────────────────────────────────
def run_stt(combo, audio_bytes):
    if combo["stt"] == "speechmatics":
        return transcribe_speechmatics(audio_bytes)
    elif combo["stt"] == "assemblyai":
        return transcribe_assemblyai()
    elif combo["stt"] == "whisper":
        return transcribe_whisper(audio_bytes)

# ── LLM Prompts ───────────────────────────────────────────────────────────────
FORMAT_PROMPT_DE = """
Du bist ein hilfreicher Assistent. Hier ist ein Rohtranskript eines Arzt-Patienten-Gesprächs mit generischen Sprecher-Labels (z.B. Speaker 1, Speaker 2, S1, S2 oder SPEAKER_00).
Deine Aufgabe:
1. Identifiziere anhand des Kontextes, wer der Arzt und wer der Patient ist.
2. Finde den Namen des Patienten heraus, falls er sich vorstellt.
3. Schreibe das Transkript um, indem du die generischen Sprecher-Labels ersetzt durch "Arzt:" und "[Name des Patienten]:" (oder "Patient(in):", falls kein Name genannt wird).
4. Verändere den eigentlichen gesprochenen Text NICHT. Ergänze nichts, lösche nichts, fasse nichts zusammen.

KRITISCHE REGELN:
- Erstelle unter keinen Umständen eine Zusammenfassung, SOAP-Notes oder Diagnosen.
- Deine EINZIGE Aufgabe ist das Suchen und Ersetzen der Sprecher-Labels im Text.
- Gib AUSSCHLIESSLICH das formatierte Transkript zurück und beginne sofort mit dem ersten Sprecher, ohne einleitende Worte.
"""

SOAP_PROMPT_DE = """
Du bist ein hochqualifizierter medizinischer Assistent. Deine Aufgabe ist es,
ein Transkript eines Arzt-Patienten-Gesprächs in strukturierte medizinische
Dokumentation im SOAP-Format (Subjective, Objective, Assessment, Plan) umzuwandeln.

Format-Vorgaben:
- S (Subjektiv): Symptome und Beschwerden aus Sicht des Patienten.
- O (Objektiv): Beobachtungen und messbare Parameter durch den Arzt.
- A (Assessment): Einschätzung, mögliche Diagnosen.
- P (Plan): Geplante Untersuchungen, Therapie, Medikation.

Bitte antworte ausschließlich mit den formatierten SOAP Notes auf Deutsch und vermeide
jegliche einleitenden oder abschließenden Floskeln. Nutze eine professionelle, präzise und klinische Ausdrucksweise.
"""

# ── LLM: GPT-4o (kein Streaming, kein Skip) ──────────────────────────────────
def _llm_openai(system_prompt, user_content):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "", False

# ── LLM: Ollama (streaming + Skip-Signal) ─────────────────────────────────────
def _llm_ollama(model_name, system_prompt, user_content, phase_name):
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LIVE_FILE, "w", encoding="utf-8") as lf:
        lf.write(f"[{ts}] === {phase_name} ({model_name}) ===\n\n")

    print(f"\n{'─'*60}\n[LIVE] {phase_name} ({model_name})\n{'─'*60}", flush=True)

    stream = ollama_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
        stream=True,
    )

    collected       = []
    last_skip_check = time.time()
    live_fh         = open(LIVE_FILE, "a", encoding="utf-8")
    try:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                print(delta, end="", flush=True)
                live_fh.write(delta)
                live_fh.flush()
                collected.append(delta)
            if time.time() - last_skip_check >= 1.0:
                last_skip_check = time.time()
                if check_skip():
                    partial = "".join(collected)
                    live_fh.write(f"\n\n[SKIP nach {len(partial)} Zeichen]\n")
                    live_fh.close()
                    print(f"\n{'─'*60}", flush=True)
                    log(f"⏭  SKIP — {phase_name} abgebrochen ({len(partial)} Zeichen bisher)")
                    return partial or None, True
    except Exception:
        live_fh.close()
        raise
    live_fh.close()
    result = "".join(collected)
    print(f"\n{'─'*60}", flush=True)
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LIVE_FILE, "a", encoding="utf-8") as lf:
        lf.write(f"\n\n[{ts}] === {phase_name} fertig ({len(result)} Zeichen) ===\n")
    return result, False

# ── LLM dispatch ──────────────────────────────────────────────────────────────
def llm_call(combo, system_prompt, user_content, phase_name):
    if combo["llm"] == "gpt4o":
        return _llm_openai(system_prompt, user_content)
    elif combo["llm"] == "llama3.2":
        return _llm_ollama(LLAMA32_MODEL, system_prompt, user_content, phase_name)
    elif combo["llm"] == "sauerkraut8b":
        return _llm_ollama(SAUERKRAUT_MODEL, system_prompt, user_content, phase_name)
    elif combo["llm"] == "gemma4b":
        return _llm_ollama(GEMMA4_MODEL, system_prompt, user_content, phase_name)

# ── Startup checks ────────────────────────────────────────────────────────────
log("=== PWC Batch Start ===")
log(f"Audiodatei: {AUDIO_FILE}")

if not os.path.exists(AUDIO_FILE):
    log(f"FEHLER: Datei nicht gefunden: {AUDIO_FILE}")
    raise SystemExit(1)

cloud_combos  = [c for c in COMBINATIONS if c["llm"] == "gpt4o"]
ollama_combos = [c for c in COMBINATIONS if c["llm"] != "gpt4o"]

if cloud_combos and not OPENAI_API_KEY:
    log("WARNUNG: OPENAI_API_KEY nicht gesetzt — GPT-4o-Kombinationen werden übersprungen.")
if any(c["stt"] == "speechmatics" for c in cloud_combos) and not SPEECHMATICS_API_KEY:
    log("WARNUNG: SPEECHMATICS_API_KEY nicht gesetzt — Speechmatics wird übersprungen.")
if any(c["stt"] == "assemblyai" for c in cloud_combos) and not ASSEMBLYAI_API_KEY:
    log("WARNUNG: ASSEMBLYAI_API_KEY nicht gesetzt — AssemblyAI wird übersprungen.")

audio_size = os.path.getsize(AUDIO_FILE)
size_mb    = round(audio_size / (1024 * 1024), 1)
log(f"Dateigröße: {size_mb} MB")

with open(AUDIO_FILE, "rb") as f:
    audio_bytes = f.read()

checkpoint   = load_checkpoint()
already_done = set(checkpoint["done"])
todo = [c for c in COMBINATIONS if c["key"] not in already_done]
log(f"Kombinationen: {len(COMBINATIONS)} | Bereits erledigt: {len(already_done)} | Verbleibend: {len(todo)}")
if already_done:
    log(f"Überspringe: {list(already_done)}")

# ── Main loop ─────────────────────────────────────────────────────────────────
for idx, combo in enumerate(todo, start=1):
    log(f"\n{'='*60}")
    log(f"Kombination {idx}/{len(todo)}: {combo['stt_label']} + {combo['llm_label']}")
    log(f"{'='*60}")

    if combo["llm"] == "gpt4o" and not OPENAI_API_KEY:
        log("FEHLER: OPENAI_API_KEY nicht gesetzt — überspringe.")
        continue
    if combo["stt"] == "speechmatics" and not SPEECHMATICS_API_KEY:
        log("FEHLER: SPEECHMATICS_API_KEY nicht gesetzt — überspringe.")
        continue
    if combo["stt"] == "assemblyai" and not ASSEMBLYAI_API_KEY:
        log("FEHLER: ASSEMBLYAI_API_KEY nicht gesetzt — überspringe.")
        continue

    pipeline_start = time.time()
    abort_reason   = None

    # ── STT ──────────────────────────────────────────────────────────────────
    log(f"Phase 1/3: STT ({combo['stt_label']}) …")
    stt_start = time.time()
    try:
        raw = run_stt(combo, audio_bytes)
    except Exception as e:
        log(f"FEHLER STT: {e}")
        continue
    stt_dur = round(time.time() - stt_start, 2)

    if raw and raw.startswith("Fehler"):
        log(f"STT Fehler: {raw}")
        continue

    log(f"STT fertig: {stt_dur}s | {len(raw)} Zeichen")

    # ── Skip check vor Format ─────────────────────────────────────────────────
    if check_skip():
        log("⏭  SKIP-Signal vor Format — überspringe diese Kombination.")
        continue

    # ── Format ───────────────────────────────────────────────────────────────
    log(f"Phase 2/3: Transkript formatieren ({combo['llm_label']}) …")
    format_start = time.time()
    formatted    = raw
    try:
        result, skipped = llm_call(combo, FORMAT_PROMPT_DE, f"Rohtranskript:\n\n{raw}", "Format-LLM")
        if skipped:
            formatted    = result if result else raw
            abort_reason = "manual_skip"
        else:
            formatted = result
    except Exception as e:
        log(f"FEHLER Format: {e} — weiter mit Rohtranskript")
    format_dur = round(time.time() - format_start, 2)
    if not abort_reason:
        log(f"Format fertig: {format_dur}s | {len(formatted)} Zeichen")

    # ── Skip check vor SOAP ───────────────────────────────────────────────────
    if not abort_reason and check_skip():
        log("⏭  SKIP-Signal vor SOAP.")
        abort_reason = "manual_skip"

    # ── SOAP ─────────────────────────────────────────────────────────────────
    soap     = ""
    soap_dur = 0
    if not abort_reason:
        log(f"Phase 3/3: SOAP-Notes ({combo['llm_label']}) …")
        soap_start = time.time()
        try:
            result, skipped = llm_call(combo, SOAP_PROMPT_DE, f"Hier ist das Transkript:\n\n{formatted}", "SOAP-LLM")
            if skipped:
                soap         = result if result else ""
                abort_reason = "manual_skip"
            else:
                soap = result
        except Exception as e:
            log(f"FEHLER SOAP: {e}")
            soap = f"Fehler: {e}"
        soap_dur = round(time.time() - soap_start, 2)
        if not abort_reason:
            log(f"SOAP fertig: {soap_dur}s | {len(soap)} Zeichen")

    total_dur = round(time.time() - pipeline_start, 2)

    # ── Speichern ─────────────────────────────────────────────────────────────
    meta = {
        "stt_model":        combo["stt_label"],
        "llm_model":        combo["llm_label"],
        "language":         LANGUAGE,
        "audio_file":       AUDIO_FILE,
        "audio_size_bytes": audio_size,
        "stats": {
            "stt_duration_s":       stt_dur,
            "format_duration_s":    format_dur,
            "soap_duration_s":      soap_dur,
            "total_duration_s":     total_dur,
            "raw_char_count":       len(raw),
            "formatted_char_count": len(formatted),
            "soap_char_count":      len(soap),
        },
    }

    if abort_reason:
        log(f"⏭  Abgebrochen ({abort_reason}) nach {total_dur}s — bisheriger Stand wird gespeichert.")
        meta["aborted"]      = True
        meta["abort_reason"] = abort_reason
        save_to_history(raw, formatted, soap, meta)
        mark_done(combo["key"])
        log("Abgebrochener Eintrag in history.json gespeichert ✓")
    else:
        log(f"Gesamt: {total_dur}s  (STT {stt_dur}s + Format {format_dur}s + SOAP {soap_dur}s)")
        save_to_history(raw, formatted, soap, meta)
        mark_done(combo["key"])
        log("Gespeichert in history.json ✓  |  Checkpoint aktualisiert ✓")

log(f"\n{'='*60}")
log("=== BATCH ABGESCHLOSSEN ===")
cp = load_checkpoint()
log(f"Erfolgreich: {len(cp['done'])}/{len(COMBINATIONS)}")
log_fh.close()
