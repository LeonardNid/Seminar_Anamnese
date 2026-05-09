#!/usr/bin/env python3
"""
Standalone cloud batch runner.
Processes every audio file with AssemblyAI (cloud STT, speaker diarization) + OpenAI GPT-4o.
No Streamlit dependency — runs as a plain Python script.
Results are appended to history.json after each file.
Checkpoint file: logs/batch_assemblyai_checkpoint.json
Log file: logs/batch_assemblyai_log.txt
"""

import os
import json
import time
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY  = os.getenv("ASSEMBLYAI_API_KEY")
HISTORY_FILE        = "history.json"
CHECKPOINT_FILE     = "logs/batch_assemblyai_checkpoint.json"
LOG_FILE            = "logs/batch_assemblyai_log.txt"
STT_LABEL           = "AssemblyAI (Cloud)"
LLM_LABEL           = "OpenAI GPT-4o"
LANGUAGE            = "de"

AUDIO_FILES = [
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
    "audio/Anamnesegesrpäch PWC.mp3",
]

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
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

# ── Clients ───────────────────────────────────────────────────────────────────
if not OPENAI_API_KEY:
    log("FEHLER: OPENAI_API_KEY nicht gesetzt.")
    raise SystemExit(1)
if not ASSEMBLYAI_API_KEY:
    log("FEHLER: ASSEMBLYAI_API_KEY nicht gesetzt.")
    raise SystemExit(1)

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── STT: AssemblyAI ───────────────────────────────────────────────────────────
def transcribe_assemblyai(file_path):
    headers = {"Authorization": ASSEMBLYAI_API_KEY}
    try:
        # 1) Upload
        with open(file_path, "rb") as f:
            up = requests.post(
                f"{ASSEMBLYAI_BASE_URL}/upload",
                headers=headers,
                data=f,
            )
        up.raise_for_status()
        audio_url = up.json()["upload_url"]
        log(f"  Upload fertig: {audio_url[:60]}…")

        # 2) Submit
        body = {
            "audio_url":    audio_url,
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
        log(f"  Job eingereicht: {job_id}")

        # 3) Poll
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
                log(f"  Polling… status={status} ({polls*5}s)")
            if status == "completed":
                break
            elif status == "error":
                return f"Fehler: {data.get('error', 'unbekannt')}"

        # 4) Format utterances
        utterances = data.get("utterances") or []
        if not utterances:
            return data.get("text") or "Fehler: Kein Text transkribiert"
        lines = [f"Speaker {u['speaker']}: {u['text']}" for u in utterances]
        return "\n".join(lines)

    except Exception as e:
        return f"Fehler bei der Transkription: {e}"

# ── LLM: OpenAI GPT-4o ────────────────────────────────────────────────────────
FORMAT_PROMPT_DE = """
Du bist ein hilfreicher Assistent. Hier ist ein Rohtranskript eines Arzt-Patienten-Gesprächs mit generischen Sprecher-Labels (z.B. Speaker A, Speaker B).
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

def llm_call(system_prompt, user_content):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
        stream=False,
    )
    return response.choices[0].message.content or ""

# ── Main loop ─────────────────────────────────────────────────────────────────
log("=== AssemblyAI Batch Start ===")
log(f"STT: {STT_LABEL}  |  LLM: {LLM_LABEL}")

checkpoint   = load_checkpoint()
already_done = set(checkpoint["done"])
todo = [f for f in AUDIO_FILES if f not in already_done]

log(f"Gesamt: {len(AUDIO_FILES)} | Bereits erledigt: {len(already_done)} | Verbleibend: {len(todo)}")
if already_done:
    log(f"Überspringe: {list(already_done)}")

for idx, file_name in enumerate(todo, start=len(already_done) + 1):
    log(f"\n{'='*60}")
    log(f"Job {idx}/{len(AUDIO_FILES)}: {file_name}")
    log(f"{'='*60}")

    if not os.path.exists(file_name):
        log("FEHLER: Datei nicht gefunden — überspringe.")
        continue

    size_mb = round(os.path.getsize(file_name) / (1024 * 1024), 1)
    log(f"Dateigröße: {size_mb} MB")
    pipeline_start = time.time()

    # ── STT ──────────────────────────────────────────────────────────────────
    log("Phase 1/3: STT (AssemblyAI Cloud) läuft …")
    stt_start = time.time()
    raw = transcribe_assemblyai(file_name)
    stt_dur = round(time.time() - stt_start, 2)

    if raw.startswith("Fehler"):
        log(f"STT Fehler: {raw}")
        continue

    log(f"STT fertig: {stt_dur}s | {len(raw)} Zeichen")

    # ── Format ───────────────────────────────────────────────────────────────
    log("Phase 2/3: Transkript formatieren (GPT-4o) …")
    format_start = time.time()
    try:
        formatted = llm_call(FORMAT_PROMPT_DE, f"Rohtranskript:\n\n{raw}")
    except Exception as e:
        log(f"FEHLER Format: {e} — weiter mit Rohtranskript")
        formatted = raw
    format_dur = round(time.time() - format_start, 2)
    log(f"Format fertig: {format_dur}s | {len(formatted)} Zeichen")

    # ── SOAP ─────────────────────────────────────────────────────────────────
    log("Phase 3/3: SOAP-Notes generieren (GPT-4o) …")
    soap_start = time.time()
    try:
        soap = llm_call(SOAP_PROMPT_DE, f"Hier ist das Transkript:\n\n{formatted}")
    except Exception as e:
        log(f"FEHLER SOAP: {e}")
        soap = f"Fehler bei der SOAP-Generierung: {e}"
    soap_dur = round(time.time() - soap_start, 2)
    total_dur = round(time.time() - pipeline_start, 2)
    log(f"SOAP fertig: {soap_dur}s | {len(soap)} Zeichen")
    log(f"Gesamt: {total_dur}s  (STT {stt_dur}s + Format {format_dur}s + SOAP {soap_dur}s)")

    # ── Speichern ─────────────────────────────────────────────────────────────
    if soap.startswith("Fehler"):
        log("SOAP enthält Fehler — nicht in History gespeichert.")
    else:
        audio_size = os.path.getsize(file_name)
        meta = {
            "stt_model":        STT_LABEL,
            "llm_model":        LLM_LABEL,
            "language":         LANGUAGE,
            "audio_file":       file_name,
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
        save_to_history(raw, formatted, soap, meta)
        mark_done(file_name)
        log("Gespeichert in history.json ✓  |  Checkpoint aktualisiert ✓")

log(f"\n{'='*60}")
log("=== BATCH ABGESCHLOSSEN ===")
cp = load_checkpoint()
log(f"Erfolgreich gespeichert: {len(cp['done'])}/{len(AUDIO_FILES)}")
log(f"Dateien: {cp['done']}")
log_fh.close()
