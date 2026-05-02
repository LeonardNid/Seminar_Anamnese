#!/usr/bin/env python3
"""
Einzel-Run: audio/Anamnesegesrpäch PWC.mp3
STT: Speechmatics (Cloud)  |  LLM: OpenAI GPT-4o
"""

import os, json, time, requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
SPEECHMATICS_URL     = os.getenv("SPEECHMATICS_URL", "https://asr.api.speechmatics.com/v2")
HISTORY_FILE         = "history.json"
AUDIO_FILE           = "audio/Anamnesegesrpäch PWC.mp3"
STT_LABEL            = "Speechmatics (Cloud)"
LLM_LABEL            = "OpenAI GPT-4o"
LANGUAGE             = "de"

# ── Pflicht-Token-Check ───────────────────────────────────────────────────────
for var in ("OPENAI_API_KEY", "SPEECHMATICS_API_KEY", "ASSEMBLYAI_API_KEY", "HF_TOKEN"):
    if not os.getenv(var):
        raise SystemExit(f"FEHLER: {var} nicht gesetzt.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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
        "name":      "Anamnesegespräch PWC",
        "raw":       raw,
        "formatted": formatted,
        "soap":      soap,
    }
    entry.update(meta)
    history.insert(0, entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


def transcribe_speechmatics(audio_bytes, file_name):
    url = f"{SPEECHMATICS_URL}/jobs/"
    headers = {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}
    config = {
        "type": "transcription",
        "transcription_config": {
            "language": LANGUAGE,
            "operating_point": "enhanced",
            "diarization": "speaker",
        }
    }
    ext = os.path.splitext(file_name)[1].lower()
    mime_map = {".m4a": "audio/x-m4a", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg"}
    mime = mime_map.get(ext, "audio/mpeg")

    resp = requests.post(
        url,
        headers=headers,
        data={"config": json.dumps(config)},
        files={"data_file": (file_name, audio_bytes, mime)},
    )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    log(f"  Speechmatics Job: {job_id}")

    status_url    = f"{SPEECHMATICS_URL}/jobs/{job_id}"
    transcript_url = f"{SPEECHMATICS_URL}/jobs/{job_id}/transcript?format=txt"

    polls = 0
    while True:
        time.sleep(5)
        polls += 1
        sr = requests.get(status_url, headers=headers)
        sr.raise_for_status()
        status = sr.json()["job"]["status"]
        if polls % 6 == 0:
            log(f"  Polling… status={status} ({polls*5}s)")
        if status == "done":
            break
        elif status in ("rejected", "deleted", "expired"):
            raise RuntimeError(f"Speechmatics Job status={status}")

    tr = requests.get(transcript_url, headers=headers)
    tr.raise_for_status()
    tr.encoding = "utf-8"
    return tr.text


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
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


# ── Main ──────────────────────────────────────────────────────────────────────
log(f"=== Run: {AUDIO_FILE} ===")

if not os.path.exists(AUDIO_FILE):
    raise SystemExit(f"FEHLER: Datei nicht gefunden: {AUDIO_FILE}")

size_mb = round(os.path.getsize(AUDIO_FILE) / (1024 * 1024), 1)
log(f"Dateigröße: {size_mb} MB")
pipeline_start = time.time()

# Phase 1: STT
log("Phase 1/3: STT (Speechmatics Cloud) …")
stt_start = time.time()
with open(AUDIO_FILE, "rb") as f:
    audio_bytes = f.read()
raw = transcribe_speechmatics(audio_bytes, AUDIO_FILE)
stt_dur = round(time.time() - stt_start, 2)
log(f"STT fertig: {stt_dur}s | {len(raw)} Zeichen")

# Phase 2: Format
log("Phase 2/3: Transkript formatieren (GPT-4o) …")
format_start = time.time()
formatted = llm_call(FORMAT_PROMPT_DE, f"Rohtranskript:\n\n{raw}")
format_dur = round(time.time() - format_start, 2)
log(f"Format fertig: {format_dur}s | {len(formatted)} Zeichen")

# Phase 3: SOAP
log("Phase 3/3: SOAP-Notes (GPT-4o) …")
soap_start = time.time()
soap = llm_call(SOAP_PROMPT_DE, f"Hier ist das Transkript:\n\n{formatted}")
soap_dur = round(time.time() - soap_start, 2)
total_dur = round(time.time() - pipeline_start, 2)
log(f"SOAP fertig: {soap_dur}s | {len(soap)} Zeichen")
log(f"Gesamt: {total_dur}s  (STT {stt_dur}s + Format {format_dur}s + SOAP {soap_dur}s)")

meta = {
    "stt_model":        STT_LABEL,
    "llm_model":        LLM_LABEL,
    "language":         LANGUAGE,
    "audio_file":       AUDIO_FILE,
    "audio_size_bytes": len(audio_bytes),
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
log("Gespeichert in history.json ✓")
