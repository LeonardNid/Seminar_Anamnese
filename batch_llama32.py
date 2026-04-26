#!/usr/bin/env python3
"""
Batch runner: reuse existing Whisper STT output, run only Format+SOAP with llama3.2.
Reads raw transcripts from history.json (Whisper+Sauerkraut entries),
skips STT entirely, runs Format+SOAP with llama3.2.
Checkpoint: batch_llama32_checkpoint.json
Log: batch_llama32_log.txt
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SAUERKRAUT_API_KEY  = os.getenv("SAUERKRAUT_API_KEY", "dummy-key")
SAUERKRAUT_BASE_URL = os.getenv("SAUERKRAUT_BASE_URL", "http://localhost:11434/v1")
HISTORY_FILE        = "history.json"
CHECKPOINT_FILE     = "logs/batch_llama32_checkpoint.json"
LOG_FILE            = "logs/batch_llama32_log.txt"
MODEL_NAME          = "llama3.2"
STT_LABEL           = "Whisper Large-v3-turbo (Lokal)"
LLM_LABEL           = "llama3.2"
LANGUAGE            = "de"

AUDIO_FILE_ORDER = [
    "Das Anamnesegespräch Teil 1 medizinische Fachsprachprüfung Fall Unfall - ärztesprech (128k).wav",
    "ChaosLapInMitte.wav",
    "GedankenprüngeLapInMitte.wav",
    "MeinungswechselLapinMitte.wav",
    "OriginalDCEng.m4a",
    "OriginalDC.m4a",
    "OriginalDCWhiteNoise.m4a",
    "OriginalLapBeiArzt.wav",
    "OriginalLapInMitte.wav",
    "SelbstkorrekturLapInMitte.wav",
    "UnterbrechungLapInMitte.wav",
]

# ── Logging ───────────────────────────────────────────────────────────────────
log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")

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
def load_whisper_transcripts():
    history = load_history()
    seen = {}
    for entry in history:
        af = entry.get("audio_file", "")
        if (entry.get("stt_model") == "Whisper Large-v3-turbo (Lokal)"
                and entry.get("llm_model") == "Llama-3.1-SauerkrautLM-8b-Instruct"
                and af and af not in seen):
            seen[af] = entry
    return seen

# ── LLM ──────────────────────────────────────────────────────────────────────
FORMAT_PROMPT_DE = """
Du bist ein hilfreicher Assistent. Hier ist ein Rohtranskript eines Arzt-Patienten-Gesprächs mit generischen Sprecher-Labels (z.B. Speaker 1, Speaker 2, S1, S2 oder SPEAKER_00).
Deine Aufgabe:
1. Identifiziere anhand des Kontextes, wer der Arzt und wer der Patient ist.
2. Finde den Namen des Patienten heraus, falls er sich vorstellt.
3. Schreibe das Transkript um, indem du die generischen Sprecher-Labels ersetzt durch "Arzt:" und "[Name des Patienten]:" (oder "Patient(in):", falls kein Name genannt wird).
4. Verändere den eigentlichen gesprochenen Text NICHT. Ergänze nichts, loesche nichts, fasse nichts zusammen.

KRITISCHE REGELN:
- Erstelle unter keinen Umstaenden eine Zusammenfassung, SOAP-Notes oder Diagnosen.
- Deine EINZIGE Aufgabe ist das Suchen und Ersetzen der Sprecher-Labels im Text.
- Gib AUSSCHLIESSLICH das formatierte Transkript zurueck und beginne sofort mit dem ersten Sprecher, ohne einleitende Worte.
"""

SOAP_PROMPT_DE = """
Du bist ein hochqualifizierter medizinischer Assistent. Deine Aufgabe ist es,
ein Transkript eines Arzt-Patienten-Gesprächs in strukturierte medizinische
Dokumentation im SOAP-Format (Subjective, Objective, Assessment, Plan) umzuwandeln.

Format-Vorgaben:
- S (Subjektiv): Symptome und Beschwerden aus Sicht des Patienten.
- O (Objektiv): Beobachtungen und messbare Parameter durch den Arzt.
- A (Assessment): Einschaetzung, moegliche Diagnosen.
- P (Plan): Geplante Untersuchungen, Therapie, Medikation.

Bitte antworte ausschliesslich mit den formatierten SOAP Notes auf Deutsch und vermeide
jegliche einleitenden oder abschliessenden Floskeln. Nutze eine professionelle, praezise und klinische Ausdrucksweise.
"""

llm_client = OpenAI(api_key=SAUERKRAUT_API_KEY, base_url=SAUERKRAUT_BASE_URL)

def llm_call(system_prompt, user_content):
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
        stream=False,
    )
    return response.choices[0].message.content or ""

# ── Main ──────────────────────────────────────────────────────────────────────
log("=== Llama3.2 Batch Start (STT reused from Whisper) ===")
log(f"LLM: {MODEL_NAME}  |  STT: wiederverwendet von Whisper Large-v3-turbo")

source_map = load_whisper_transcripts()
log(f"Whisper-Transkripte geladen: {len(source_map)} Dateien")
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
        log("FEHLER: Kein Whisper-Transkript gefunden — ueberspringe.")
        continue

    source_entry = source_map[file_name]
    raw = source_entry["raw"]
    audio_size = source_entry.get("audio_size_bytes", 0)
    size_mb = round(audio_size / (1024 * 1024), 1)
    log(f"Rohtranskript: {len(raw)} Zeichen | Audio: {size_mb} MB (aus history)")

    pipeline_start = time.time()

    # ── Format ───────────────────────────────────────────────────────────────
    log(f"Phase 1/2: Transkript formatieren ({MODEL_NAME}) ...")
    format_start = time.time()
    try:
        formatted = llm_call(FORMAT_PROMPT_DE, f"Rohtranskript:\n\n{raw}")
    except Exception as e:
        log(f"FEHLER Format: {e} — weiter mit Rohtranskript")
        formatted = raw
    format_dur = round(time.time() - format_start, 2)
    log(f"Format fertig: {format_dur}s | {len(formatted)} Zeichen")

    # ── SOAP ─────────────────────────────────────────────────────────────────
    log(f"Phase 2/2: SOAP-Notes generieren ({MODEL_NAME}) ...")
    soap_start = time.time()
    try:
        soap = llm_call(SOAP_PROMPT_DE, f"Hier ist das Transkript:\n\n{formatted}")
    except Exception as e:
        log(f"FEHLER SOAP: {e}")
        soap = f"Fehler bei der SOAP-Generierung: {e}"
    soap_dur = round(time.time() - soap_start, 2)
    total_dur = round(time.time() - pipeline_start, 2)
    log(f"SOAP fertig: {soap_dur}s | {len(soap)} Zeichen")
    log(f"Gesamt (LLM only): {total_dur}s  (Format {format_dur}s + SOAP {soap_dur}s)")

    # ── Speichern ─────────────────────────────────────────────────────────────
    if soap.startswith("Fehler"):
        log("SOAP enthaelt Fehler — nicht gespeichert.")
    else:
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
        save_to_history(raw, formatted, soap, meta)
        mark_done(file_name)
        log("Gespeichert in history.json OK  |  Checkpoint aktualisiert OK")

log(f"\n{'='*60}")
log("=== BATCH ABGESCHLOSSEN ===")
cp = load_checkpoint()
log(f"Erfolgreich gespeichert: {len(cp['done'])}/{len(AUDIO_FILE_ORDER)}")
log_fh.close()
