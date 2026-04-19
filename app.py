import streamlit as st
import os
import time
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
import tempfile
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAUERKRAUT_API_KEY = os.getenv("SAUERKRAUT_API_KEY", "dummy-key")
SAUERKRAUT_BASE_URL = os.getenv("SAUERKRAUT_BASE_URL")
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
SPEECHMATICS_URL = os.getenv("SPEECHMATICS_URL", "https://asr.api.speechmatics.com/v2")
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="AI-Anamnesis PoC", page_icon="🩺", layout="wide")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
sauerkraut_client = OpenAI(api_key=SAUERKRAUT_API_KEY, base_url=SAUERKRAUT_BASE_URL) if SAUERKRAUT_BASE_URL else None

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_to_history(raw, formatted, soap, meta=None):
    if raw and formatted and soap and not soap.startswith("Fehler"):
        history = load_history()
        entry = {
            "id": str(int(time.time())),
            "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "name": "",
            "raw": raw,
            "formatted": formatted,
            "soap": soap
        }
        if meta:
            entry.update(meta)
        history.insert(0, entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

def rename_history_entry(entry_id, new_name):
    history = load_history()
    for entry in history:
        if entry["id"] == entry_id:
            entry["name"] = new_name
            break
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def delete_history_entry(entry_id):
    history = load_history()
    history = [entry for entry in history if entry["id"] != entry_id]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

BATCH_FIRST_FILE = "Das Anamnesegespräch Teil 1 medizinische Fachsprachprüfung Fall Unfall - ärztesprech (128k).wav"

def get_audio_files():
    """Return audio files in the project directory, with BATCH_FIRST_FILE sorted to the top."""
    audio_exts = {".wav", ".mp3", ".m4a", ".ogg"}
    files = sorted(
        f for f in os.listdir(".")
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in audio_exts
    )
    if BATCH_FIRST_FILE in files:
        files.remove(BATCH_FIRST_FILE)
        files.insert(0, BATCH_FIRST_FILE)
    return files

def transcribe_audio_speechmatics(audio_bytes, language="de"):
    if not SPEECHMATICS_API_KEY:
        return "Fehler: SPEECHMATICS_API_KEY nicht gefunden."

    url = f"{SPEECHMATICS_URL}/jobs/"
    headers = {
        "Authorization": f"Bearer {SPEECHMATICS_API_KEY}"
    }

    config = {
        "type": "transcription",
        "transcription_config": {
            "language": language,
            "operating_point": "enhanced",
            "diarization": "speaker"
        }
    }

    data = {
        "config": json.dumps(config)
    }

    files = {
        "data_file": ("audio.wav", audio_bytes, "audio/wav")
    }

    try:
        # Submit Job
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        job_id = response.json()["id"]

        # Poll for completion
        status_url = f"{SPEECHMATICS_URL}/jobs/{job_id}"
        transcript_url = f"{SPEECHMATICS_URL}/jobs/{job_id}/transcript?format=txt"

        with st.spinner("Transkription läuft (Speechmatics)..."):
            while True:
                status_res = requests.get(status_url, headers=headers)
                status_res.raise_for_status()
                status = status_res.json()["job"]["status"]

                if status == "done":
                    break
                elif status in ["rejected", "deleted", "expired"]:
                    return f"Fehler: Job status is {status}"

                time.sleep(2)

            # Get Transcript
            transcript_res = requests.get(transcript_url, headers=headers)
            transcript_res.raise_for_status()

            # Ensure correct decoding of special characters (Umlaute, ß)
            transcript_res.encoding = 'utf-8'
            return transcript_res.text

    except Exception as e:
        return f"Fehler bei der Transkription: {str(e)}"

@st.cache_resource
def load_whisper_model():
    return WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

@st.cache_resource
def load_diarization_pipeline():
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

def transcribe_audio_whisper(audio_bytes, lang_code="de"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        whisper_lang = lang_code if lang_code in ["de", "en"] else None

        with st.spinner("Transkription läuft (Whisper lokal)..."):
            model = load_whisper_model()
            segments, _ = model.transcribe(
                tmp_path, language=whisper_lang, beam_size=5, word_timestamps=True
            )
            whisper_segments = [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments
            ]

        with st.spinner("Sprecher werden erkannt (pyannote lokal)..."):
            diarize = load_diarization_pipeline()
            diarization = diarize(tmp_path).speaker_diarization

        os.remove(tmp_path)

        # Jedem Whisper-Segment den Sprecher zuweisen, der in diesem Zeitfenster am längsten aktiv war
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

    except Exception as e:
        return f"Fehler bei der Whisper-Transkription: {str(e)}"

def format_transcript(transcript, lang_code="de", llm_model="OpenAI GPT-4o"):
    active_client = client if llm_model == "OpenAI GPT-4o" else sauerkraut_client
    model_name = "gpt-4o" if llm_model == "OpenAI GPT-4o" else "hf.co/QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF:Q4_K_M"

    if not active_client:
        yield f"Fehler: API Client für {llm_model} ist nicht konfiguriert."
        return

    if lang_code == "en":
        system_prompt = """
            You are a helpful assistant. Here is a raw transcript of a doctor-patient conversation with generic speaker labels (e.g., Speaker 1, Speaker 2, SPEAKER_00).
            Your task:
            1. Identify from the context who is the doctor and who is the patient.
            2. Find out the patient's name if they introduce themselves.
            3. Rewrite the transcript by replacing the generic speaker labels with "Doctor:" and "[Patient's Name]:" (or "Patient:", if no name is mentioned).
            4. DO NOT change the actual spoken text. Do not add, remove, or summarize anything.

            CRITICAL RULES:
            - Under no circumstances should you generate a summary, SOAP notes, or extra notes.
            - Your ONLY task is to replace the speaker labels.
            - Return ONLY the formatted transcript, starting immediately with the first speaker.
            """
    elif lang_code == "de":
        system_prompt = """
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
    else:
        system_prompt = """
            You are a helpful assistant. Here is a raw transcript of a doctor-patient conversation with generic speaker labels (e.g., Speaker 1, Speaker 2, SPEAKER_00).
            Your task:
            1. Identify from the context who is the doctor and who is the patient.
            2. Find out the patient's name if they introduce themselves.
            3. Rewrite the transcript by replacing the generic speaker labels with "Doctor:" / "Arzt:" and "[Patient's Name]:" (or "Patient:", if no name is mentioned), matching the language of the transcript.
            4. DO NOT change the actual spoken text. Do not add, remove, or summarize anything.

            CRITICAL RULES:
            - Under no circumstances should you generate a summary, SOAP notes, or extra notes.
            - Your ONLY task is to replace the speaker labels.
            - Return ONLY the formatted transcript, starting immediately with the first speaker, without introductory words.
            """

    try:
        response = active_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rohtranskript:\n\n{transcript}"}
            ],
            temperature=0.2,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Fehler bei der Formatierung: {str(e)}"

def generate_soap_notes(transcript, lang_code="de", llm_model="OpenAI GPT-4o"):
    active_client = client if llm_model == "OpenAI GPT-4o" else sauerkraut_client
    model_name = "gpt-4o" if llm_model == "OpenAI GPT-4o" else "hf.co/QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF:Q4_K_M"

    if not active_client:
        yield f"Fehler: API Client für {llm_model} ist nicht konfiguriert."
        return

    if lang_code == "en":
        system_prompt = """
        You are a highly qualified medical assistant. Your task is to convert a transcript
        of a doctor-patient conversation into structured medical documentation in SOAP format
        (Subjective, Objective, Assessment, Plan).

        Format Guidelines:
        - S (Subjective): Symptoms and complaints from the patient's perspective.
        - O (Objective): Observations and measurable parameters by the doctor.
        - A (Assessment): Medical assessment, possible diagnoses.
        - P (Plan): Planned examinations, therapy, medication.

        Please respond exclusively with the formatted SOAP Notes in English and avoid
        any introductory or concluding phrases. Use a professional, precise, and clinical tone.
        """
    elif lang_code == "de":
        system_prompt = """
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
    else:
        system_prompt = """
        You are a highly qualified medical assistant. Your task is to convert a transcript
        of a doctor-patient conversation into structured medical documentation in SOAP format
        (Subjective, Objective, Assessment, Plan).

        Format Guidelines:
        - S (Subjective/Subjektiv): Symptoms and complaints from the patient's perspective.
        - O (Objective/Objektiv): Observations and measurable parameters by the doctor.
        - A (Assessment): Medical assessment, possible diagnoses.
        - P (Plan): Planned examinations, therapy, medication.

        Please respond exclusively with the formatted SOAP Notes in the language of the transcript (German or English) and avoid
        any introductory or concluding phrases. Use a professional, precise, and clinical tone.
        """

    try:
        response = active_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hier ist das Transkript:\n\n{transcript}"}
            ],
            temperature=0.2,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Fehler bei der Zusammenfassung: {str(e)}"

def run_pipeline(audio_bytes, stt_model, llm_model, language, file_name=""):
    """Run the full pipeline non-interactively and return (raw, formatted, soap, meta) with timing stats."""
    pipeline_start = time.time()

    # STT
    stt_start = time.time()
    if stt_model == "Whisper Large-v3-turbo (Lokal)":
        raw = transcribe_audio_whisper(audio_bytes, language)
    else:
        raw = transcribe_audio_speechmatics(audio_bytes, language)
    stt_duration = round(time.time() - stt_start, 2)

    if raw.startswith("Fehler"):
        return None, raw

    # Format transcript
    format_start = time.time()
    formatted = "".join(format_transcript(raw, language, llm_model))
    format_duration = round(time.time() - format_start, 2)

    if formatted.startswith("Fehler"):
        formatted = raw

    # SOAP notes
    soap_start = time.time()
    soap = "".join(generate_soap_notes(formatted, language, llm_model))
    soap_duration = round(time.time() - soap_start, 2)

    total_duration = round(time.time() - pipeline_start, 2)

    meta = {
        "stt_model": stt_model,
        "llm_model": llm_model,
        "language": language,
        "audio_file": file_name,
        "audio_size_bytes": len(audio_bytes),
        "stats": {
            "stt_duration_s": stt_duration,
            "format_duration_s": format_duration,
            "soap_duration_s": soap_duration,
            "total_duration_s": total_duration,
            "raw_char_count": len(raw),
            "formatted_char_count": len(formatted),
            "soap_char_count": len(soap),
        }
    }

    return (raw, formatted, soap, meta), None

# ── UI Layout ────────────────────────────────────────────────────────────────

st.title("🩺 AI-Anamnesis PoC")
st.markdown("""
Dieses Tool dient der Aufzeichnung von Arzt-Patienten-Gesprächen.
Die Audio-Daten werden asynchron über die **Speechmatics API** transkribiert und anschließend mittels **OpenAI GPT-4o** in eine strukturierte **SOAP-Notiz** überführt.
""")

st.divider()

# ── History Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("📚 Verlauf")
if "viewing_history" not in st.session_state:
    st.session_state.viewing_history = False
if "current_history_id" not in st.session_state:
    st.session_state.current_history_id = None

history_data = load_history()

if not history_data:
    st.sidebar.info("Noch keine Einträge vorhanden.")
else:
    if st.sidebar.button("➕ Neue Aufnahme", type="primary"):
        st.session_state.raw_transcript = None
        st.session_state.formatted_transcript = None
        st.session_state.soap_notes = None
        st.session_state.viewing_history = False
        st.rerun()

    st.sidebar.divider()

    for entry in history_data:
        display_name = entry.get("name") if entry.get("name") else entry['timestamp']

        col_btn, col_del = st.sidebar.columns([4, 1])
        with col_btn:
            if st.button(f"🗓️ {display_name}", key=f"btn_{entry['id']}", use_container_width=True):
                st.session_state.raw_transcript = entry.get("raw")
                st.session_state.formatted_transcript = entry.get("formatted")
                st.session_state.soap_notes = entry.get("soap")
                st.session_state.viewing_history = True
                st.session_state.current_history_id = entry['id']
        with col_del:
            if st.button("🗑️", key=f"del_{entry['id']}", help="Eintrag löschen"):
                delete_history_entry(entry['id'])
                if st.session_state.get("current_history_id") == entry['id']:
                    st.session_state.raw_transcript = None
                    st.session_state.formatted_transcript = None
                    st.session_state.soap_notes = None
                    st.session_state.viewing_history = False
                    st.session_state.current_history_id = None
                st.rerun()

# ── Main Tabs ────────────────────────────────────────────────────────────────
main_tab1, main_tab2 = st.tabs(["🎙️ Aufnahme & Ergebnisse", "🔬 Batch-Test"])

# ── Tab 1: Aufnahme & Ergebnisse ─────────────────────────────────────────────
with main_tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Gespräch aufzeichnen oder hochladen")

        st_model_option = st.radio(
            "Spracherkennungs-Modell (STT):",
            options=["Speechmatics (Cloud)", "Whisper Large-v3-turbo (Lokal)"],
            horizontal=True
        )

        llm_model_option = st.radio(
            "Sprachmodell (LLM):",
            options=["OpenAI GPT-4o", "Llama-3.1-SauerkrautLM-8b-Instruct"],
            horizontal=True
        )

        language_option = st.radio(
            "Sprache des Gesprächs:",
            options=["Deutsch", "Englisch", "Automatisch (Auto)"],
            horizontal=True
        )

        lang_code = "de"
        if language_option == "Englisch":
            lang_code = "en"
        elif language_option == "Automatisch (Auto)":
            lang_code = "auto"

        tab1, tab2 = st.tabs(["🎙️ Aufnehmen", "📁 Hochladen"])

        audio_bytes = None
        audio_file_name = ""

        with tab1:
            st.write("Bitte auf das Mikrofon-Symbol klicken, um die Aufnahme zu starten und zu stoppen.")
            # 'pause_threshold' erhöht, damit die Aufnahme nicht sofort bei Sprechpausen abbricht (Standard ist 0.8s)
            recorded_audio = audio_recorder(
                text="Aufnehmen / Stoppen",
                recording_color="#e81123",
                neutral_color="#009688",
                pause_threshold=60.0
            )
            if recorded_audio:
                audio_bytes = recorded_audio
                audio_file_name = "Aufnahme"

        with tab2:
            uploaded_file = st.file_uploader("Wähle eine Audiodatei aus", type=["wav", "mp3", "m4a", "ogg"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                audio_file_name = uploaded_file.name

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            process_button = st.button("Audio verarbeiten (Transkription & Zusammenfassung)", type="primary", use_container_width=True)

    with col2:
        st.subheader("Ergebnisse")

        # Initialize session state variables
        for key, default in [
            ("raw_transcript", None),
            ("formatted_transcript", None),
            ("soap_notes", None),
            ("processing_stage", None),
            ("abort_format", False),
            ("llm_model_at_process", None),
            ("lang_code_at_process", None),
            ("stt_model_at_process", None),
            ("stt_duration", None),
            ("format_duration", None),
            ("soap_duration", None),
            ("pipeline_start", None),
            ("audio_size_bytes", None),
            ("audio_file_name_at_process", None),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        # Start processing when button is clicked
        if audio_bytes and process_button:
            st.session_state.raw_transcript = None
            st.session_state.formatted_transcript = None
            st.session_state.soap_notes = None
            st.session_state.abort_format = False
            st.session_state.processing_stage = 'transcribe'
            st.session_state.llm_model_at_process = llm_model_option
            st.session_state.lang_code_at_process = lang_code
            st.session_state.stt_model_at_process = st_model_option
            st.session_state.stt_duration = None
            st.session_state.format_duration = None
            st.session_state.soap_duration = None
            st.session_state.pipeline_start = time.time()
            st.session_state.audio_size_bytes = len(audio_bytes)
            st.session_state.audio_file_name_at_process = audio_file_name
            st.rerun()

        # ── Stage: transcription ──────────────────────────────────────────────
        if st.session_state.processing_stage == 'transcribe':
            st.markdown("### Transkription läuft...")
            stt_start = time.time()
            if st.session_state.stt_model_at_process == "Whisper Large-v3-turbo (Lokal)":
                transcript = transcribe_audio_whisper(audio_bytes, st.session_state.lang_code_at_process)
            else:
                transcript = transcribe_audio_speechmatics(audio_bytes, st.session_state.lang_code_at_process)
            st.session_state.stt_duration = round(time.time() - stt_start, 2)

            if transcript.startswith("Fehler"):
                st.error(transcript)
                st.session_state.processing_stage = None
            else:
                st.session_state.raw_transcript = transcript
                st.session_state.processing_stage = 'format'
                st.rerun()

        # ── Stage: format transcript ──────────────────────────────────────────
        elif st.session_state.processing_stage == 'format':
            st.markdown("### Transkription")
            with st.expander("Rohes Transkript anzeigen (STT Output)", expanded=False):
                st.write(st.session_state.raw_transcript)

            col_info, col_abort = st.columns([3, 1])
            with col_info:
                st.info(f"Formatiere Transkript und identifiziere Sprecher ({st.session_state.llm_model_at_process})...")
            with col_abort:
                if st.button("Abbrechen", key="abort_format_btn", use_container_width=True):
                    st.session_state.abort_format = True
                    st.session_state.format_duration = 0.0
                    st.session_state.processing_stage = 'soap'
                    st.rerun()

            format_start = time.time()
            with st.expander("Genaue Transkription anzeigen", expanded=True):
                stream = format_transcript(
                    st.session_state.raw_transcript,
                    st.session_state.lang_code_at_process,
                    st.session_state.llm_model_at_process,
                )
                st.session_state.formatted_transcript = st.write_stream(stream)
            st.session_state.format_duration = round(time.time() - format_start, 2)

            st.session_state.processing_stage = 'soap'
            st.rerun()

        # ── Stage: SOAP generation ────────────────────────────────────────────
        elif st.session_state.processing_stage == 'soap':
            st.markdown("### Transkription")
            with st.expander("Rohes Transkript anzeigen (STT Output)", expanded=False):
                st.write(st.session_state.raw_transcript)
            with st.expander("Genaue Transkription anzeigen", expanded=True):
                if st.session_state.formatted_transcript:
                    st.write(st.session_state.formatted_transcript)
                else:
                    st.write(st.session_state.raw_transcript)

            if st.session_state.abort_format:
                st.warning("Formatierung abgebrochen – SOAP wird aus Rohtranskript erstellt.")

            soap_input = st.session_state.formatted_transcript if st.session_state.formatted_transcript else st.session_state.raw_transcript

            st.markdown("### Medizinische Dokumentation (SOAP)")
            st.info(f"Generiere SOAP Notes ({st.session_state.llm_model_at_process})...")
            soap_start = time.time()
            stream_soap = generate_soap_notes(soap_input, st.session_state.lang_code_at_process, st.session_state.llm_model_at_process)
            st.session_state.soap_notes = st.write_stream(stream_soap)
            st.session_state.soap_duration = round(time.time() - soap_start, 2)

            total_duration = round(time.time() - st.session_state.pipeline_start, 2) if st.session_state.pipeline_start else None

            meta = {
                "stt_model": st.session_state.stt_model_at_process,
                "llm_model": st.session_state.llm_model_at_process,
                "language": st.session_state.lang_code_at_process,
                "audio_file": st.session_state.audio_file_name_at_process or "",
                "audio_size_bytes": st.session_state.audio_size_bytes or 0,
                "stats": {
                    "stt_duration_s": st.session_state.stt_duration,
                    "format_duration_s": st.session_state.format_duration,
                    "soap_duration_s": st.session_state.soap_duration,
                    "total_duration_s": total_duration,
                    "raw_char_count": len(st.session_state.raw_transcript or ""),
                    "formatted_char_count": len(st.session_state.formatted_transcript or ""),
                    "soap_char_count": len(st.session_state.soap_notes or ""),
                }
            }

            st.session_state.processing_stage = None
            save_to_history(st.session_state.raw_transcript, st.session_state.formatted_transcript, st.session_state.soap_notes, meta)
            st.rerun()

        elif st.session_state.formatted_transcript or st.session_state.raw_transcript:
            if st.session_state.get("viewing_history") and st.session_state.get("current_history_id"):
                st.info("Sie betrachten einen Eintrag aus dem Verlauf.")
                current_entry = next((e for e in history_data if e["id"] == st.session_state.current_history_id), None)
                if current_entry:
                    col_name1, col_name2 = st.columns([3, 1])
                    with col_name1:
                        new_name = st.text_input("Eintrag umbenennen:", value=current_entry.get("name") or "", label_visibility="collapsed", placeholder="Neuer Name für den Eintrag")
                    with col_name2:
                        if st.button("Speichern", use_container_width=True, key="save_name_btn"):
                            rename_history_entry(current_entry["id"], new_name)
                            st.rerun()

                    # ── Stats panel ───────────────────────────────────────────
                    if current_entry.get("stats"):
                        s = current_entry["stats"]
                        with st.expander("📊 Statistiken", expanded=True):
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("STT-Dauer", f"{s.get('stt_duration_s', '–')} s")
                            m2.metric("Format-Dauer", f"{s.get('format_duration_s', '–')} s")
                            m3.metric("SOAP-Dauer", f"{s.get('soap_duration_s', '–')} s")
                            m4.metric("Gesamt", f"{s.get('total_duration_s', '–')} s")

                            st.divider()
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(f"**STT-Modell:** {current_entry.get('stt_model', '–')}")
                                st.write(f"**LLM-Modell:** {current_entry.get('llm_model', '–')}")
                                st.write(f"**Sprache:** {current_entry.get('language', '–')}")
                            with c2:
                                size_kb = round(current_entry.get('audio_size_bytes', 0) / 1024, 1)
                                st.write(f"**Audio:** {current_entry.get('audio_file', '–')} ({size_kb} KB)")
                                st.write(f"**Rohtranskript:** {s.get('raw_char_count', '–')} Zeichen")
                                st.write(f"**Formatiert:** {s.get('formatted_char_count', '–')} Zeichen")
                                st.write(f"**SOAP:** {s.get('soap_char_count', '–')} Zeichen")

                st.divider()

            st.markdown("### Transkription")

            if st.session_state.raw_transcript:
                with st.expander("Rohes Transkript anzeigen (STT Output)", expanded=False):
                    st.write(st.session_state.raw_transcript)

            display_transcript = st.session_state.formatted_transcript or st.session_state.raw_transcript or ""
            with st.expander("Genaue Transkription anzeigen", expanded=True):
                st.write(display_transcript)

            st.download_button(
                label="Transkript Herunterladen",
                data=display_transcript,
                file_name="transkript.txt",
                mime="text/plain"
            )

            st.divider()

            if st.session_state.soap_notes:
                st.markdown("### Medizinische Dokumentation (SOAP)")
                if st.session_state.soap_notes.startswith("Fehler"):
                    st.error(st.session_state.soap_notes)
                else:
                    st.success("Erfolgreich generiert!")
                    st.markdown(st.session_state.soap_notes)
        elif not audio_bytes:
            st.info("Bitte zeichnen Sie zunächst ein Audio auf, um die Ergebnisse hier zu sehen.")

# ── Tab 2: Batch-Test ─────────────────────────────────────────────────────────
with main_tab2:
    st.subheader("🔬 Batch-Test: Alle Audiodateien automatisch verarbeiten")
    st.markdown(
        "Führe die komplette Pipeline (STT → Format → SOAP) für mehrere Audiodateien und "
        "Modell-Kombinationen aus. Ergebnisse werden automatisch im Verlauf gespeichert."
    )

    audio_files = get_audio_files()

    if not audio_files:
        st.warning("Keine Audiodateien im aktuellen Verzeichnis gefunden (.wav, .mp3, .m4a, .ogg).")
    else:
        bcol1, bcol2 = st.columns([2, 1])

        with bcol1:
            st.markdown("**Audiodateien auswählen:**")
            selected_files = st.multiselect(
                "Dateien",
                options=audio_files,
                default=audio_files,
                format_func=lambda f: f"{f}  ({round(os.path.getsize(f) / (1024 * 1024), 1)} MB)",
                label_visibility="collapsed",
            )

        with bcol2:
            st.markdown("**Modell-Konfiguration:**")
            batch_stt_options = st.multiselect(
                "STT-Modelle:",
                options=["Speechmatics (Cloud)", "Whisper Large-v3-turbo (Lokal)"],
                default=["Speechmatics (Cloud)", "Whisper Large-v3-turbo (Lokal)"],
            )
            batch_llm_options = st.multiselect(
                "LLM-Modelle:",
                options=["OpenAI GPT-4o", "Llama-3.1-SauerkrautLM-8b-Instruct"],
                default=["OpenAI GPT-4o"],
            )
            batch_language = st.selectbox(
                "Sprache:",
                options=["de", "en", "auto"],
                index=0,
            )

        st.divider()

        total_jobs = len(selected_files) * len(batch_stt_options) * len(batch_llm_options)
        jobs_ready = bool(selected_files and batch_stt_options and batch_llm_options)

        if jobs_ready:
            st.info(
                f"**{total_jobs} Job(s)** werden ausgeführt: "
                f"{len(selected_files)} Datei(en) × "
                f"{len(batch_stt_options)} STT-Modell(e) × "
                f"{len(batch_llm_options)} LLM-Modell(e)"
            )

        start_batch = st.button(
            f"▶ Batch starten ({total_jobs} Jobs)",
            type="primary",
            disabled=not jobs_ready,
            key="start_batch_btn",
        )

        if start_batch and jobs_ready:
            jobs = [
                (f, stt, llm)
                for f in selected_files
                for stt in batch_stt_options
                for llm in batch_llm_options
            ]

            progress_bar = st.progress(0, text="Batch läuft...")
            job_header = st.empty()      # "Job X/N – Datei | STT | LLM"
            stage_status = st.empty()    # aktuelle Stage
            log_area = st.empty()        # wachsender Log aller abgeschlossenen Steps
            batch_results = []
            log_lines = []

            def log(line):
                log_lines.append(line)
                log_area.markdown("\n".join(log_lines))

            for i, (file_path, stt_model, llm_model) in enumerate(jobs):
                pipeline_start = time.time()
                job_header.info(
                    f"**Job {i+1}/{len(jobs)}** — `{file_path}`  \n"
                    f"STT: **{stt_model}**  |  LLM: **{llm_model}**  |  Sprache: **{batch_language}**"
                )
                log(f"\n---\n**Job {i+1}/{len(jobs)}:** `{file_path}` | {stt_model} | {llm_model}")

                try:
                    with open(file_path, "rb") as af:
                        audio_data = af.read()
                    size_mb = round(len(audio_data) / (1024 * 1024), 1)
                    log(f"&nbsp;&nbsp;📂 Datei geladen: {size_mb} MB")

                    # ── STT ──────────────────────────────────────────────────
                    stage_status.info(f"🎙️ **STT läuft...** ({stt_model})")
                    stt_start = time.time()
                    if stt_model == "Whisper Large-v3-turbo (Lokal)":
                        raw = transcribe_audio_whisper(audio_data, batch_language)
                    else:
                        raw = transcribe_audio_speechmatics(audio_data, batch_language)
                    stt_dur = round(time.time() - stt_start, 2)

                    if raw.startswith("Fehler"):
                        stage_status.error(f"STT Fehler: {raw}")
                        log(f"&nbsp;&nbsp;❌ STT Fehler nach {stt_dur}s: {raw[:100]}")
                        batch_results.append({
                            "Datei": file_path, "STT": stt_model, "LLM": llm_model,
                            "Status": f"❌ STT: {raw[:60]}",
                            "Gesamt (s)": None, "STT (s)": stt_dur,
                            "Format (s)": None, "SOAP (s)": None,
                            "Rohtext (Z)": None, "Format (Z)": None, "SOAP (Z)": None,
                        })
                        progress_bar.progress((i + 1) / len(jobs))
                        continue

                    log(f"&nbsp;&nbsp;✅ STT fertig: **{stt_dur}s** | {len(raw)} Zeichen")

                    # ── Format ───────────────────────────────────────────────
                    stage_status.info(f"✏️ **Transkript wird formatiert...** ({llm_model})")
                    format_start = time.time()
                    formatted = "".join(format_transcript(raw, batch_language, llm_model))
                    format_dur = round(time.time() - format_start, 2)

                    if formatted.startswith("Fehler"):
                        log(f"&nbsp;&nbsp;⚠️ Format Fehler ({format_dur}s) – weiter mit Rohtranskript")
                        formatted = raw
                    else:
                        log(f"&nbsp;&nbsp;✅ Formatierung fertig: **{format_dur}s** | {len(formatted)} Zeichen")

                    # ── SOAP ─────────────────────────────────────────────────
                    stage_status.info(f"📋 **SOAP-Notes werden generiert...** ({llm_model})")
                    soap_start = time.time()
                    soap = "".join(generate_soap_notes(formatted, batch_language, llm_model))
                    soap_dur = round(time.time() - soap_start, 2)
                    total_dur = round(time.time() - pipeline_start, 2)

                    if soap.startswith("Fehler"):
                        log(f"&nbsp;&nbsp;❌ SOAP Fehler nach {soap_dur}s: {soap[:100]}")
                        batch_results.append({
                            "Datei": file_path, "STT": stt_model, "LLM": llm_model,
                            "Status": f"❌ SOAP: {soap[:60]}",
                            "Gesamt (s)": total_dur, "STT (s)": stt_dur,
                            "Format (s)": format_dur, "SOAP (s)": soap_dur,
                            "Rohtext (Z)": len(raw), "Format (Z)": len(formatted), "SOAP (Z)": None,
                        })
                        progress_bar.progress((i + 1) / len(jobs))
                        continue

                    log(f"&nbsp;&nbsp;✅ SOAP fertig: **{soap_dur}s** | {len(soap)} Zeichen")
                    log(f"&nbsp;&nbsp;🏁 Job abgeschlossen: **Gesamt {total_dur}s** (STT {stt_dur}s + Format {format_dur}s + SOAP {soap_dur}s)")

                    # ── Speichern ─────────────────────────────────────────────
                    meta = {
                        "stt_model": stt_model,
                        "llm_model": llm_model,
                        "language": batch_language,
                        "audio_file": file_path,
                        "audio_size_bytes": len(audio_data),
                        "stats": {
                            "stt_duration_s": stt_dur,
                            "format_duration_s": format_dur,
                            "soap_duration_s": soap_dur,
                            "total_duration_s": total_dur,
                            "raw_char_count": len(raw),
                            "formatted_char_count": len(formatted),
                            "soap_char_count": len(soap),
                        }
                    }
                    save_to_history(raw, formatted, soap, meta)
                    batch_results.append({
                        "Datei": file_path, "STT": stt_model, "LLM": llm_model,
                        "Status": "✅ OK",
                        "Gesamt (s)": total_dur, "STT (s)": stt_dur,
                        "Format (s)": format_dur, "SOAP (s)": soap_dur,
                        "Rohtext (Z)": len(raw), "Format (Z)": len(formatted), "SOAP (Z)": len(soap),
                    })

                except Exception as e:
                    elapsed = round(time.time() - pipeline_start, 2)
                    log(f"&nbsp;&nbsp;❌ Unerwarteter Fehler nach {elapsed}s: {str(e)[:120]}")
                    batch_results.append({
                        "Datei": file_path, "STT": stt_model, "LLM": llm_model,
                        "Status": f"❌ {str(e)[:80]}",
                        "Gesamt (s)": elapsed, "STT (s)": None,
                        "Format (s)": None, "SOAP (s)": None,
                        "Rohtext (Z)": None, "Format (Z)": None, "SOAP (Z)": None,
                    })

                progress_bar.progress((i + 1) / len(jobs), text=f"Job {i+1}/{len(jobs)} abgeschlossen")

            ok_count = sum(1 for r in batch_results if r["Status"].startswith("✅"))
            job_header.empty()
            stage_status.success(f"✅ Batch abgeschlossen — {ok_count} von {len(jobs)} Jobs erfolgreich.")
            log(f"\n---\n**Batch fertig: {ok_count}/{len(jobs)} OK**")

            st.markdown("#### Ergebnisse")
            st.dataframe(batch_results, use_container_width=True)
            st.rerun()
