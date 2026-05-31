"""
Wendet die neue deterministische Formatierungslogik auf alle 55 Einträge
aus results/v1_praesentation/history.json an.

Ausgaben:
  results/v2/history_v2_formatted.json  — gleiche Struktur, neues 'formatted'
  docs/v2/format_v2_validierung.md      — Bericht: Wort-Identität je Eintrag

Validierung: Wortfolge von 'formatted' (ohne Rollen-Präfixe) ==
             Wortfolge von 'raw' (ohne Labels) → beweist Text-Erhalt.

Voraussetzung: lokales Ollama läuft (für llama3.2/SauerkrautLM/gemma4)
               OPENAI_API_KEY in .env (für GPT-4o / AssemblyAI / Speechmatics)
"""
import sys
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Repo-Root aus Skript-Pfad berechnen (skript/format/ → ../../)
REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from speaker_format import parse_turns, distinct_labels, identify_roles, apply_mapping

IN_FILE  = REPO / "results" / "v1_praesentation" / "history.json"
OUT_FILE = REPO / "results" / "v2" / "history_v2_formatted.json"
RPT_FILE = REPO / "docs" / "v2" / "format_v2_validierung.md"

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OLLAMA_API_KEY   = os.getenv("SAUERKRAUT_API_KEY", "ollama")
OLLAMA_BASE_URL  = os.getenv("SAUERKRAUT_BASE_URL", "http://localhost:11434/v1")

_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_ollama_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)


def _resolve_client(llm_model: str):
    """Wählt Client und Modellname analog zu _resolve_llm in app.py."""
    m = llm_model.lower()
    if "gpt" in m or "openai" in m:
        return _openai_client, "gpt-4o"
    if "llama3.2" in m or "llama 3.2" in m:
        return _ollama_client, "llama3.2"
    if "gemma4" in m:
        return _ollama_client, "gemma4:e4b"
    # Sauerkraut und alles andere → Ollama
    return _ollama_client, "hf.co/QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF:Q4_K_M"


_PUNCT = re.compile(r'[.,!?;:\-–—()\[\]"\'„"«»/]')
_LABEL_PREFIX = re.compile(
    r'^\s*(?:Arzt|Patient(?:in)?|Frau\s+\w+|Herr\s+\w+|\w+):\s*', re.IGNORECASE
)


def _words(text: str) -> list[str]:
    # Label-Präfixe am Zeilenanfang entfernen, dann Satzzeichen normalisieren
    lines = text.splitlines()
    cleaned = [_LABEL_PREFIX.sub('', l) for l in lines]
    norm = _PUNCT.sub(' ', ' '.join(cleaned)).lower()
    return [w for w in norm.split() if w]


def _words_raw(text: str) -> list[str]:
    """Entfernt bekannte STT-Label-Formate aus raw-Text."""
    # Speechmatics-Labels (eigene Zeile)
    text = re.sub(r'^\s*SPEAKER:\s*S\w+\s*$', '', text, flags=re.MULTILINE)
    # Inline-Labels
    text = re.sub(r'^\s*(?:SPEAKER_[\w?]+|Speaker\s+\w+):\s*', '', text, flags=re.MULTILINE)
    norm = _PUNCT.sub(' ', text).lower()
    return [w for w in norm.split() if w]


def process_entry(e: dict) -> tuple[str, bool, str]:
    """Gibt (formatted, wort_identisch, hinweis) zurück."""
    raw = e.get("raw", "")
    llm_model = e.get("llm_model", "")

    turns = parse_turns(raw)
    if not turns:
        return raw, True, "keine Labels erkannt — Rohtext unverändert"

    client, model = _resolve_client(llm_model)
    if client is None:
        return raw, False, f"kein Client für '{llm_model}' verfügbar (API-Key fehlt?)"

    labels = distinct_labels(turns)
    mapping = identify_roles(raw, labels, client, model)
    formatted = apply_mapping(turns, mapping)

    raw_words = _words_raw(raw)
    fmt_words = _words(formatted)
    identical = raw_words == fmt_words

    if not identical:
        hinweis = f"WORT-UNTERSCHIED: raw={len(raw_words)} fmt={len(fmt_words)} Wörter"
    else:
        hinweis = f"OK ({len(raw_words)} Wörter, Mapping: {mapping})"

    return formatted, identical, hinweis


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RPT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Lese {IN_FILE} …")
    with open(IN_FILE, encoding="utf-8") as f:
        data = json.load(f)
    print(f"{len(data)} Einträge geladen.")

    # Bereits vorhandene Ergebnisse laden (Resume)
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resume: {len(done_ids)} bereits verarbeitet.")
    else:
        results = []
        done_ids = set()

    report_rows: list[str] = []
    ok_count = 0
    fail_count = 0

    for idx, e in enumerate(data, 1):
        eid = e.get("id", str(idx))
        audio = e.get("audio_file", "?").split("/")[-1]
        model = e.get("llm_model", "?")
        stt   = e.get("stt_model", "?")

        if eid in done_ids:
            print(f"[{idx:2d}] übersprungen (bereits vorhanden): {audio} | {model}")
            continue

        print(f"[{idx:2d}/{len(data)}] {audio:40s} | {model:15s} … ", end="", flush=True)

        try:
            formatted, identical, hinweis = process_entry(e)
        except Exception as ex:
            formatted = e.get("raw", "")
            identical = False
            hinweis = f"FEHLER: {ex}"

        new_entry = {**e, "formatted": formatted}
        results.append(new_entry)
        done_ids.add(eid)

        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        status = "OK" if identical else "FAIL"
        print(f"{status} — {hinweis[:80]}")
        report_rows.append(f"| {idx} | {audio} | {stt[:30]} | {model[:12]} | {status} | {hinweis[:100]} |")

        if identical:
            ok_count += 1
        else:
            fail_count += 1

    # Markdown-Report schreiben
    out = [
        "# Formatierungs-Validierung v2",
        "",
        f"Gesamt: {len(data)} Einträge | OK: {ok_count} | FAIL: {fail_count}",
        "",
        "| # | Audio | STT | LLM | Status | Hinweis |",
        "|---|-------|-----|-----|--------|---------|",
    ] + report_rows

    RPT_FILE.write_text('\n'.join(out) + '\n', encoding="utf-8")
    print(f"\nFertig. Bericht: {RPT_FILE}")
    print(f"Ergebnis: {ok_count} OK, {fail_count} FAIL")


if __name__ == "__main__":
    main()
