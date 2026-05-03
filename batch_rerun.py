#!/usr/bin/env python3
"""
Rerun-Batch: führt alle abgebrochenen Einträge aus history.json erneut vollständig durch.
Die ursprünglichen abgebrochenen Einträge bleiben unverändert in history.json erhalten.
Neue vollständige Einträge werden zusätzlich hinzugefügt.

Checkpoint: logs/batch_rerun_checkpoint.json
Log:        logs/batch_rerun_log.txt
Live:       logs/llm_live.txt
Skip:       touch logs/skip_current
"""

import os
import json
import time
import glob
import tempfile
import subprocess
import warnings
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
os.makedirs("logs", exist_ok=True)
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")

# ── GPU detection ─────────────────────────────────────────────────────────────
try:
    import torch
    _gpu_available = torch.cuda.is_available()
except ImportError:
    _gpu_available = False

_device       = "cuda" if _gpu_available else "cpu"
_compute_type = "float16" if _gpu_available else "int8"

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN            = os.getenv("HF_TOKEN", "")
OLLAMA_API_KEY      = os.getenv("SAUERKRAUT_API_KEY", "ollama")
OLLAMA_BASE_URL     = os.getenv("SAUERKRAUT_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
HISTORY_FILE        = "history.json"
ABORTED_FILE        = "aborted_entries.json"
CHECKPOINT_FILE     = "logs/batch_rerun_checkpoint.json"
LOG_FILE            = "logs/batch_rerun_log.txt"
LIVE_FILE           = "logs/llm_live.txt"
SKIP_FILE           = "logs/skip_current"
LANGUAGE            = "de"

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
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# ── Prompts ───────────────────────────────────────────────────────────────────
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

# ── STT: Whisper + pyannote (Pflicht) ─────────────────────────────────────────
_whisper_model    = None
_diarize_pipeline = None

def _init_whisper():
    global _whisper_model, _diarize_pipeline
    if _whisper_model is not None:
        return
    log(f"Lade Whisper large-v3-turbo (device={_device}, compute_type={_compute_type}) …")
    t0 = time.time()
    _whisper_model = WhisperModel("large-v3-turbo", device=_device, compute_type=_compute_type)
    log(f"Whisper geladen in {round(time.time()-t0,1)}s")

    from pyannote.audio import Pipeline as PyannotePipeline
    log("Lade pyannote speaker-diarization-3.1 …")
    t0 = time.time()
    _diarize_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=HF_TOKEN
    )
    log(f"pyannote geladen in {round(time.time()-t0,1)}s")

def transcribe_whisper(audio_path):
    _init_whisper()

    ts = datetime.now().strftime("%H:%M:%S")
    with open(LIVE_FILE, "w", encoding="utf-8") as lf:
        lf.write(f"[{ts}] === STT (Whisper large-v3-turbo) ===\n\n")
    print(f"\n{'─'*60}\n[LIVE] STT — Whisper large-v3-turbo\n{'─'*60}", flush=True)

    segments_gen, _ = _whisper_model.transcribe(
        audio_path, language=LANGUAGE, beam_size=5, word_timestamps=True
    )
    whisper_segments = []
    live_fh = open(LIVE_FILE, "a", encoding="utf-8")
    for s in segments_gen:
        text = s.text.strip()
        line = f"[{s.start:6.1f}s → {s.end:6.1f}s]  {text}"
        print(line, flush=True)
        live_fh.write(line + "\n")
        live_fh.flush()
        whisper_segments.append({"start": s.start, "end": s.end, "text": text})
    live_fh.close()
    print(f"{'─'*60}", flush=True)

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True, capture_output=True,
        )
        log("Diarization läuft (pyannote, CPU — kann mehrere Minuten dauern) …")
        diarization = _diarize_pipeline(wav_path).speaker_diarization
        log("Diarization fertig.")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

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
    return "\n".join(lines)

# ── LLM: Ollama (streaming) ───────────────────────────────────────────────────
def llm_call_ollama(model_name, system_prompt, user_content, phase_name):
    client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LIVE_FILE, "w", encoding="utf-8") as lf:
        lf.write(f"[{ts}] === {phase_name} ({model_name}) ===\n\n")
    print(f"\n{'─'*60}\n[LIVE] {phase_name} ({model_name})\n{'─'*60}", flush=True)

    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
        stream=True,
    )
    collected = []
    last_skip_check = time.time()
    live_fh = open(LIVE_FILE, "a", encoding="utf-8")
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
                    log(f"⏭  SKIP — {phase_name} abgebrochen ({len(partial)} Zeichen bisher).")
                    return partial or None, True
    except Exception:
        live_fh.close()
        raise
    live_fh.close()
    result = "".join(collected)
    print(f"\n{'─'*60}", flush=True)
    return result, False

# ── LLM: GPT-4o ──────────────────────────────────────────────────────────────
def llm_call_openai(system_prompt, user_content):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "", False

def llm_call(stt_label, llm_label, system_prompt, user_content, phase_name):
    llm_lower = llm_label.lower()
    if "gpt" in llm_lower:
        return llm_call_openai(system_prompt, user_content)
    else:
        return llm_call_ollama(llm_label, system_prompt, user_content, phase_name)

# ── Startup ───────────────────────────────────────────────────────────────────
log("=== Rerun Batch Start ===")

if not os.path.exists(ABORTED_FILE):
    log(f"FEHLER: {ABORTED_FILE} nicht gefunden. Bitte zuerst aborted_entries.json erstellen.")
    raise SystemExit(1)

aborted = json.load(open(ABORTED_FILE, encoding="utf-8"))
log(f"Abgebrochene Einträge geladen: {len(aborted)}")

needs_whisper = any("Whisper" in e.get("stt_model", "") for e in aborted)
if needs_whisper and not HF_TOKEN:
    log("FEHLER: HF_TOKEN nicht gesetzt — Speaker-Diarization ist Pflicht. Abbruch.")
    raise SystemExit(1)

checkpoint   = load_checkpoint()
already_done = set(checkpoint["done"])
todo = [e for e in aborted if e["id"] not in already_done]
log(f"Gesamt: {len(aborted)} | Bereits erledigt: {len(already_done)} | Verbleibend: {len(todo)}")

# ── Main loop ─────────────────────────────────────────────────────────────────
for idx, src in enumerate(todo, start=len(already_done) + 1):
    audio_file = src.get("audio_file", "")
    stt_label  = src.get("stt_model", "")
    llm_label  = src.get("llm_model", "")

    log(f"\n{'='*60}")
    log(f"Job {idx}/{len(aborted)}: {audio_file}")
    log(f"STT: {stt_label}  |  LLM: {llm_label}")
    log(f"{'='*60}")

    if not os.path.exists(audio_file):
        log(f"FEHLER: Audiodatei nicht gefunden — überspringe.")
        continue

    audio_size = os.path.getsize(audio_file)
    size_mb    = round(audio_size / (1024 * 1024), 1)
    log(f"Audio: {size_mb} MB")

    pipeline_start = time.time()
    abort_reason   = None

    # ── STT ──────────────────────────────────────────────────────────────────
    raw = ""
    stt_dur = 0
    log(f"Phase 1/3: STT ({stt_label}) …")
    stt_start = time.time()
    try:
        if "Whisper" in stt_label:
            raw = transcribe_whisper(audio_file)
        else:
            log(f"FEHLER: STT-Methode '{stt_label}' nicht unterstützt.")
            continue
    except Exception as e:
        log(f"FEHLER STT: {e} — überspringe.")
        continue
    stt_dur = round(time.time() - stt_start, 2)
    log(f"STT fertig: {stt_dur}s | {len(raw)} Zeichen")

    if check_skip():
        log("⏭  SKIP-Signal vor Format.")
        abort_reason = "manual_skip"

    # ── Format ───────────────────────────────────────────────────────────────
    formatted  = raw
    format_dur = 0
    if not abort_reason:
        log(f"Phase 2/3: Transkript formatieren ({llm_label}) …")
        format_start = time.time()
        try:
            result, skipped = llm_call(stt_label, llm_label, FORMAT_PROMPT_DE, f"Rohtranskript:\n\n{raw}", "Format-LLM")
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

    if not abort_reason and check_skip():
        log("⏭  SKIP-Signal vor SOAP.")
        abort_reason = "manual_skip"

    # ── SOAP ─────────────────────────────────────────────────────────────────
    soap     = ""
    soap_dur = 0
    if not abort_reason:
        log(f"Phase 3/3: SOAP-Notes ({llm_label}) …")
        soap_start = time.time()
        try:
            result, skipped = llm_call(stt_label, llm_label, SOAP_PROMPT_DE, f"Hier ist das Transkript:\n\n{formatted}", "SOAP-LLM")
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
        "stt_model":        stt_label,
        "llm_model":        llm_label,
        "language":         src.get("language", LANGUAGE),
        "audio_file":       audio_file,
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
        log(f"⏭  Erneut abgebrochen ({abort_reason}) — bisheriger Stand wird gespeichert.")
        meta["aborted"]      = True
        meta["abort_reason"] = abort_reason
    else:
        log(f"Gesamt: {total_dur}s  (STT {stt_dur}s + Format {format_dur}s + SOAP {soap_dur}s)")

    save_to_history(raw, formatted, soap, meta)
    mark_done(src["id"])
    log("Gespeichert in history.json ✓  |  Checkpoint aktualisiert ✓")

log(f"\n{'='*60}")
log("=== RERUN BATCH ABGESCHLOSSEN ===")
cp = load_checkpoint()
log(f"Erfolgreich: {len(cp['done'])}/{len(aborted)}")
log_fh.close()
