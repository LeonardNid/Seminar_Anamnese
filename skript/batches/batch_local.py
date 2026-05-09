#!/usr/bin/env python3
"""
Standalone local batch runner.
Processes every audio file with Whisper large-v3-turbo + Sauerkraut LLM.
No Streamlit dependency — runs as a plain Python script.
Results are appended to history.json after each file.
A checkpoint file (batch_checkpoint.json) tracks which files are done
so the script can be safely restarted without re-processing completed files.
Progress and timing are written to batch_log.txt in real-time.
"""

import os
import json
import time
import tempfile
import sys
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SAUERKRAUT_API_KEY  = os.getenv("SAUERKRAUT_API_KEY", "dummy-key")
SAUERKRAUT_BASE_URL = os.getenv("SAUERKRAUT_BASE_URL", "http://localhost:11434/v1")
HF_TOKEN            = os.getenv("HF_TOKEN")
HISTORY_FILE        = "history.json"
CHECKPOINT_FILE     = "logs/batch_checkpoint.json"
LOG_FILE            = "logs/batch_log.txt"
MODEL_NAME          = "hf.co/QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF:Q4_K_M"
STT_LABEL           = "Whisper Large-v3-turbo (Lokal)"
LLM_LABEL           = "Llama-3.1-SauerkrautLM-8b-Instruct"
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

# ── Models (loaded once) ──────────────────────────────────────────────────────
log("=== Batch Start ===")
log(f"Lade Whisper large-v3-turbo …")
t0 = time.time()
whisper_model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
log(f"Whisper geladen in {round(time.time()-t0,1)}s")

log("Lade pyannote speaker-diarization-3.1 …")
t0 = time.time()
diarize_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)
log(f"pyannote geladen in {round(time.time()-t0,1)}s")

llm_client = OpenAI(api_key=SAUERKRAUT_API_KEY, base_url=SAUERKRAUT_BASE_URL)

# ── STT ───────────────────────────────────────────────────────────────────────
def transcribe(audio_path):
    segments, _ = whisper_model.transcribe(
        audio_path, language=LANGUAGE, beam_size=5, word_timestamps=True
    )
    whisper_segments = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]

    diarization = diarize_pipeline(audio_path).speaker_diarization

    def get_speaker(start, end):
        overlap = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            o = min(turn.end, end) - max(turn.start, start)
            if o > 0:
                overlap[speaker] = overlap.get(speaker, 0) + o
        return max(overlap, key=overlap.get) if overlap else "SPEAKER_??"

    lines = []
    current_speaker = None
    current_text = []
    for seg in whisper_segments:
        speaker = get_speaker(seg["start"], seg["end"])
        if speaker != current_speaker:
            if current_text:
                lines.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [seg["text"]]
        else:
            current_text.append(seg["text"])
    if current_text:
        lines.append(f"{current_speaker}: {' '.join(current_text)}")
    return "\n".join(lines)

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

# ── Main loop ─────────────────────────────────────────────────────────────────
checkpoint = load_checkpoint()
already_done = set(checkpoint["done"])
todo = [f for f in AUDIO_FILES if f not in already_done]

log(f"Gesamt: {len(AUDIO_FILES)} Dateien | Bereits erledigt: {len(already_done)} | Verbleibend: {len(todo)}")
if already_done:
    log(f"Überspringe: {list(already_done)}")

for idx, file_name in enumerate(todo, start=len(already_done) + 1):
    log(f"\n{'='*60}")
    log(f"Job {idx}/{len(AUDIO_FILES)}: {file_name}")
    log(f"{'='*60}")

    if not os.path.exists(file_name):
        log(f"FEHLER: Datei nicht gefunden — überspringe.")
        continue

    size_mb = round(os.path.getsize(file_name) / (1024 * 1024), 1)
    log(f"Dateigröße: {size_mb} MB")

    pipeline_start = time.time()

    # ── STT ──────────────────────────────────────────────────────────────────
    log("Phase 1/3: STT (Whisper + pyannote) läuft …")
    stt_start = time.time()
    try:
        # Für m4a: temporäre Kopie anlegen damit pyannote den Pfad lesen kann
        ext = os.path.splitext(file_name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            with open(file_name, "rb") as src:
                tmp.write(src.read())
            tmp_path = tmp.name

        raw = transcribe(tmp_path)
        os.remove(tmp_path)
    except Exception as e:
        log(f"FEHLER STT: {e}")
        continue
    stt_dur = round(time.time() - stt_start, 2)
    log(f"STT fertig: {stt_dur}s | {len(raw)} Zeichen")

    # ── Format ───────────────────────────────────────────────────────────────
    log("Phase 2/3: Transkript formatieren (Sauerkraut) …")
    format_start = time.time()
    try:
        formatted = llm_call(FORMAT_PROMPT_DE, f"Rohtranskript:\n\n{raw}")
    except Exception as e:
        log(f"FEHLER Format: {e} — weiter mit Rohtranskript")
        formatted = raw
    format_dur = round(time.time() - format_start, 2)
    log(f"Format fertig: {format_dur}s | {len(formatted)} Zeichen")

    # ── SOAP ─────────────────────────────────────────────────────────────────
    log("Phase 3/3: SOAP-Notes generieren (Sauerkraut) …")
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
        meta = {
            "stt_model":        STT_LABEL,
            "llm_model":        LLM_LABEL,
            "language":         LANGUAGE,
            "audio_file":       file_name,
            "audio_size_bytes": int(size_mb * 1024 * 1024),
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
        log(f"Gespeichert in history.json ✓  |  Checkpoint aktualisiert ✓")

log(f"\n{'='*60}")
log("=== BATCH ABGESCHLOSSEN ===")
cp = load_checkpoint()
log(f"Erfolgreich gespeichert: {len(cp['done'])}/{len(AUDIO_FILES)}")
log(f"Dateien: {cp['done']}")
log_fh.close()
