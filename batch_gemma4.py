#!/usr/bin/env python3
"""
Batch runner: reuse AssemblyAI STT output from history.json, run only
Format+SOAP with gemma4:26b (or any model set via OLLAMA_MODEL).

Monitoring:
  tail -f logs/batch_gemma4_log.txt   # Fortschritt & Statistiken
  tail -f logs/llm_live.txt           # LLM-Ausgabe live (Token für Token)

Aktuelle Audio manuell überspringen (Batch läuft weiter):
  touch logs/skip_current
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SAUERKRAUT_API_KEY  = os.getenv("SAUERKRAUT_API_KEY", "ollama")
SAUERKRAUT_BASE_URL = os.getenv("SAUERKRAUT_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME          = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
HISTORY_FILE        = "history.json"
CHECKPOINT_FILE     = "logs/batch_gemma4_checkpoint.json"
LOG_FILE            = "logs/batch_gemma4_log.txt"
LIVE_FILE           = "logs/llm_live.txt"
SKIP_FILE           = "logs/skip_current"
SOURCE_STT_LABEL    = "AssemblyAI (Cloud)"
STT_LABEL           = SOURCE_STT_LABEL
LLM_LABEL           = MODEL_NAME
LANGUAGE            = "de"

AUDIO_FILE_ORDER = [
    "audio/Das Anamnesegespräch Teil 1 medizinische Fachsprachprüfung Fall Unfall - ärztesprech (128k).wav",
    "audio/ChaosLapInMitte.wav",
    "audio/GedankenprüngeLapInMitte.wav",
    "audio/MeinungswechselLapinMitte.wav",
    "audio/OriginalDCEng.m4a",
    "audio/OriginalDC.m4a",
    "audio/OriginalDCWhiteNoise.m4a",
    "audio/OriginalLapBeiArzt.wav",
    "audio/OriginalLapInMitte.wav",
    "audio/SelbstkorrekturLapInMitte.wav",
    "audio/UnterbrechungLapInMitte.wav",
    "audio/Ambient_Listening_Test (1).m4a",
]

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
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

def mark_done(file_name):
    cp = load_checkpoint()
    if file_name not in cp["done"]:
        cp["done"].append(file_name)
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

# ── Load source transcripts ───────────────────────────────────────────────────
def load_assemblyai_transcripts():
    history = load_history()
    seen = {}
    for entry in history:
        af = entry.get("audio_file", "")
        if entry.get("stt_model") == SOURCE_STT_LABEL and af and af not in seen:
            seen[af] = entry
    return seen

# ── LLM helpers ───────────────────────────────────────────────────────────────
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

llm_client = OpenAI(api_key=SAUERKRAUT_API_KEY, base_url=SAUERKRAUT_BASE_URL)

def llm_call(system_prompt, user_content, phase_name="LLM"):
    """
    Streaming LLM-Aufruf. Tokens erscheinen live auf stdout und in logs/llm_live.txt.
    Manueller Abbruch: touch logs/skip_current
    Gibt (text, skipped) zurück.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LIVE_FILE, "w", encoding="utf-8") as lf:
        lf.write(f"[{ts}] === {phase_name} ===\n\n")

    print(f"\n{'─'*60}\n[LIVE] {phase_name}\n{'─'*60}", flush=True)

    stream = llm_client.chat.completions.create(
        model=MODEL_NAME,
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
                    log(f"⏭  SKIP — {phase_name} abgebrochen ({len(partial)} Zeichen bisher gespeichert).")
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

# ── Main ──────────────────────────────────────────────────────────────────────
log("=== gemma4 Batch Start (STT wiederverwendet von AssemblyAI) ===")
log(f"LLM: {MODEL_NAME}  |  STT: wiederverwendet von {SOURCE_STT_LABEL}")

source_map = load_assemblyai_transcripts()
log(f"AssemblyAI-Transkripte geladen: {len(source_map)} Dateien")
for f in AUDIO_FILE_ORDER:
    status = "OK" if f in source_map else "FEHLT"
    log(f"  [{status}] {f}")

checkpoint   = load_checkpoint()
already_done = set(checkpoint["done"])
todo = [f for f in AUDIO_FILE_ORDER if f not in already_done]
log(f"\nGesamt: {len(AUDIO_FILE_ORDER)} | Bereits erledigt: {len(already_done)} | Verbleibend: {len(todo)}")

for idx, file_name in enumerate(todo, start=len(already_done) + 1):
    log(f"\n{'='*60}")
    log(f"Job {idx}/{len(AUDIO_FILE_ORDER)}: {file_name}")
    log(f"{'='*60}")

    if file_name not in source_map:
        log("FEHLER: Kein AssemblyAI-Transkript gefunden — überspringe.")
        continue

    source_entry = source_map[file_name]
    raw = source_entry["raw"]
    audio_size = source_entry.get("audio_size_bytes", 0)
    size_mb = round(audio_size / (1024 * 1024), 1)
    log(f"Rohtranskript: {len(raw)} Zeichen | Audio: {size_mb} MB (aus history)")

    pipeline_start = time.time()
    abort_reason   = None

    # ── Skip-Check vor Format ─────────────────────────────────────────────────
    if check_skip():
        log("⏭  SKIP-Signal vor Format — überspringe diese Datei.")
        abort_reason = "manual_skip"

    # ── Format ───────────────────────────────────────────────────────────────
    formatted = raw
    format_dur = 0
    if not abort_reason:
        log(f"Phase 1/2: Transkript formatieren ({MODEL_NAME}) …")
        format_start = time.time()
        try:
            result, skipped = llm_call(FORMAT_PROMPT_DE, f"Rohtranskript:\n\n{raw}", "Format-LLM")
            if skipped:
                formatted = result if result else raw
                abort_reason = "manual_skip"
            else:
                formatted = result
        except Exception as e:
            log(f"FEHLER Format: {e} — weiter mit Rohtranskript")
        format_dur = round(time.time() - format_start, 2)
        if not abort_reason:
            log(f"Format fertig: {format_dur}s | {len(formatted)} Zeichen")

    # ── Skip-Check vor SOAP ───────────────────────────────────────────────────
    if not abort_reason and check_skip():
        log("⏭  SKIP-Signal vor SOAP — überspringe diese Datei.")
        abort_reason = "manual_skip"

    # ── SOAP ─────────────────────────────────────────────────────────────────
    soap = ""
    soap_dur = 0
    if not abort_reason:
        log(f"Phase 2/2: SOAP-Notes generieren ({MODEL_NAME}) …")
        soap_start = time.time()
        try:
            result, skipped = llm_call(SOAP_PROMPT_DE, f"Hier ist das Transkript:\n\n{formatted}", "SOAP-LLM")
            if skipped:
                soap = result if result else ""
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
        "stt_model":        STT_LABEL,
        "llm_model":        LLM_LABEL,
        "language":         LANGUAGE,
        "audio_file":       file_name,
        "audio_size_bytes": audio_size,
        "stats": {
            "stt_duration_s":       0,
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
        mark_done(file_name)
        log("Abgebrochener Eintrag in history.json gespeichert ✓")
    else:
        log(f"Gesamt (LLM only): {total_dur}s  (Format {format_dur}s + SOAP {soap_dur}s)")
        save_to_history(raw, formatted, soap, meta)
        mark_done(file_name)
        log("Gespeichert in history.json ✓  |  Checkpoint aktualisiert ✓")

log(f"\n{'='*60}")
log("=== BATCH ABGESCHLOSSEN ===")
cp = load_checkpoint()
log(f"Erfolgreich gespeichert: {len(cp['done'])}/{len(AUDIO_FILE_ORDER)}")
log_fh.close()
