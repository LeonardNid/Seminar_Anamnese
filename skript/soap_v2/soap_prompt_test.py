"""
SOAP-Prompt-Test: 3 Szenarien × 2 Modelle × 2 Prompts = 12 SOAP-Generierungen.

Eingabe: Ground-Truth-Texte aus docs/Seminar Ground Truth Texte.md
Ausgabe: results/v2/soap_prompt_test.json (inkrementell, resume-fähig)

Voraussetzungen:
  - OPENAI_API_KEY in .env (für GPT-4o)
  - Ollama läuft lokal mit llama3.2 (für llama3.2)
"""
import sys
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO / "skript" / "soap_v2"))
from soap_prompts import PROMPTS

GT_FILE  = REPO / "docs" / "Seminar Ground Truth Texte.md"
OUT_FILE = REPO / "results" / "v2" / "soap_prompt_test.json"

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OLLAMA_API_KEY  = os.getenv("SAUERKRAUT_API_KEY", "ollama")
OLLAMA_BASE_URL = os.getenv("SAUERKRAUT_BASE_URL", "http://localhost:11434/v1")

_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_ollama_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)

SZENARIEN   = ["OriginalDC", "Chaos", "Anamnesegespräch"]
MODELLE     = [("gpt-4o", _openai_client), ("llama3.2", _ollama_client)]
PROMPT_KEYS = list(PROMPTS.keys())  # ["baseline", "kandidat"]

# ── Ground-Truth parsen ────────────────────────────────────────────────────

GT_NAME_MAP = {
    "original dc white noise":    "OriginalDC+Noise",
    "original dc":                "OriginalDC",
    "original lap in mitte":      "LapInMitte",
    "original lap bei arzt":      "LapBeiArzt",
    "selbstkorrekturen":          "Selbstkorrekturen",
    "unterbrechungen":            "Unterbrechungen",
    "gedankensprünge":            "Gedankensprünge",
    "meinungswechsel":            "Meinungswechsel",
    "chaos":                      "Chaos",
    "das anamnesegespräch":       "Anamnesegespräch",
    "anamnesegesrpäch pwc":       "PWC",
    "anamnesegespräch pwc":       "PWC",
}

def parse_gt_text() -> dict[str, str]:
    """Liest GT-Datei und gibt vollständigen Dialogtext pro Szenario zurück.

    Sprecher-Präfixe (**Arzt:**, **Patient:**) werden zu "Arzt:", "Patient:" etc.
    normalisiert (fett-Markdown entfernt), Text sonst unverändert.
    """
    text = GT_FILE.read_text(encoding="utf-8")
    sections = re.split(r'^#{1,4}\s+Ground Truth:\s*', text, flags=re.MULTILINE)
    gt = {}
    for sec in sections[1:]:
        lines = sec.split('\n')
        raw_name = lines[0].strip()
        key = None
        for k, v in GT_NAME_MAP.items():
            if k in raw_name.lower():
                key = v
                break
        if key is None:
            continue
        body_lines = lines[1:]
        # **Arzt:** → Arzt: usw.
        cleaned = [re.sub(r'\*\*([^*]+)\*\*', r'\1', l) for l in body_lines]
        body = '\n'.join(cleaned).strip()
        if body:
            gt[key] = body
    return gt


# ── LLM aufrufen ──────────────────────────────────────────────────────────

def call_soap(client, model_name: str, system_prompt: str, transcript: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Hier ist das Transkript:\n\n{transcript}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


# ── Hauptprogramm ──────────────────────────────────────────────────────────

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    gt = parse_gt_text()
    missing = [s for s in SZENARIEN if s not in gt]
    if missing:
        print(f"WARNUNG: Ground Truth fehlt für: {missing}")

    # Resume
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
        done_keys = {(r["scenario"], r["model"], r["prompt_variante"]) for r in results}
        print(f"Resume: {len(done_keys)} bereits vorhanden.")
    else:
        results = []
        done_keys = set()

    total = len(SZENARIEN) * len(MODELLE) * len(PROMPT_KEYS)
    idx = 0

    for szenario in SZENARIEN:
        transcript = gt.get(szenario, "")
        if not transcript:
            print(f"SKIP {szenario}: kein Ground-Truth-Text")
            continue

        for model_name, client in MODELLE:
            if client is None:
                print(f"SKIP {model_name}: kein API-Client verfügbar")
                continue

            for pk in PROMPT_KEYS:
                idx += 1
                key = (szenario, model_name, pk)
                if key in done_keys:
                    print(f"[{idx:2d}/{total}] {szenario:20s} | {model_name:10s} | {pk:10s} — übersprungen")
                    continue

                print(f"[{idx:2d}/{total}] {szenario:20s} | {model_name:10s} | {pk:10s} … ", end="", flush=True)
                try:
                    soap = call_soap(client, model_name, PROMPTS[pk], transcript)
                    status = "OK"
                except Exception as ex:
                    soap = f"FEHLER: {ex}"
                    status = "FEHLER"

                results.append({
                    "scenario":       szenario,
                    "model":          model_name,
                    "prompt_variante": pk,
                    "soap":           soap,
                    "transcript_len": len(transcript),
                })
                done_keys.add(key)

                with open(OUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                print(f"{status} ({len(soap)} Zeichen)")

    print(f"\nFertig. {len(results)} Einträge in {OUT_FILE}")


if __name__ == "__main__":
    main()
