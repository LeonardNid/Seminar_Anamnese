"""
Evaluiert die 12 SOAP-Generierungen aus soap_prompt_test.json mit Claude.

Eingabe:
  results/v2/soap_prompt_test.json   — 12 SOAP-Einträge
  docs/Seminar Ground Truth Texte.md — als Referenztranskript ({formatted})
  skript/soap/soap_eval_prompt.md    — Evaluator-Prompt-Template

Ausgabe:
  results/v2/soap_prompt_test_eval.json

Voraussetzung: 'claude' CLI verfügbar und eingeloggt.
"""
import sys
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).parent.parent.parent

IN_FILE    = REPO / "results" / "v2" / "soap_prompt_test.json"
OUT_FILE   = REPO / "results" / "v2" / "soap_prompt_test_eval.json"
PROMPT_TPL = (REPO / "skript" / "soap" / "soap_eval_prompt.md").read_text(encoding="utf-8")
GT_FILE    = REPO / "docs" / "Seminar Ground Truth Texte.md"

# ── Ground-Truth parsen (Dialogtext mit Sprecher-Präfixen) ────────────────

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
        cleaned = [re.sub(r'\*\*([^*]+)\*\*', r'\1', l) for l in body_lines]
        body = '\n'.join(cleaned).strip()
        if body:
            gt[key] = body
    return gt


# ── Claude aufrufen ────────────────────────────────────────────────────────

def urteil(score: int) -> str:
    if score >= 7: return "akzeptabel"
    if score >= 4: return "ueberarbeitung_noetig"
    return "nicht_verwendbar"


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r'```(?:json)?\s*', '', text).strip('`').strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def call_claude(prompt_text: str, retries: int = 2) -> str | None:
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ['claude', '-p', prompt_text],
                capture_output=True, text=True, timeout=120
            )
            output = result.stdout.strip()
            if not output and result.stderr:
                print(f"    STDERR: {result.stderr[:200]}")
            return output
        except subprocess.TimeoutExpired:
            print(f"    Timeout (Versuch {attempt+1}/{retries})")
            time.sleep(5)
        except Exception as ex:
            print(f"    Fehler: {ex}")
            time.sleep(5)
    return None


# ── Hauptprogramm ──────────────────────────────────────────────────────────

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    gt = parse_gt_text()

    with open(IN_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Resume
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
        done_keys = {(r["scenario"], r["model"], r["prompt_variante"]) for r in results
                     if "fehler" not in r}
        print(f"Resume: {len(done_keys)} OK-Einträge vorhanden.")
    else:
        results = []
        done_keys = set()

    total = len(data)
    for idx, e in enumerate(data, 1):
        szenario = e["scenario"]
        model    = e["model"]
        pk       = e["prompt_variante"]
        soap     = e.get("soap", "")
        key      = (szenario, model, pk)

        if key in done_keys:
            print(f"[{idx:2d}/{total}] {szenario:20s} | {model:10s} | {pk:10s} — übersprungen")
            continue

        print(f"[{idx:2d}/{total}] {szenario:20s} | {model:10s} | {pk:10s} … ", end="", flush=True)

        transcript = gt.get(szenario, "")
        if not transcript or not soap.strip():
            print("→ AUTO-SKIP (kein Transkript oder leeres SOAP)")
            results.append({
                "scenario": szenario, "model": model, "prompt_variante": pk,
                "auto_skip": True, "auto_skip_grund": "kein GT oder leeres SOAP",
                "S": {"score":0,"halluzinationen":[],"auslassungen":[]},
                "O": {"score":0,"halluzinationen":[],"auslassungen":[]},
                "A": {"score":0,"halluzinationen":[],"auslassungen":[]},
                "P": {"score":0,"halluzinationen":[],"auslassungen":[]},
                "gesamt_score": 0, "urteil": "nicht_verwendbar",
            })
        else:
            prompt = PROMPT_TPL.replace('{formatted}', transcript).replace('{soap}', soap)
            raw_response = call_claude(prompt)
            parsed = extract_json(raw_response)

            if parsed is None:
                preview = (raw_response or "")[:200]
                print(f"→ FEHLER (kein JSON). Antwort: {preview}")
                results.append({
                    "scenario": szenario, "model": model, "prompt_variante": pk,
                    "fehler": "kein_gueltiges_json", "raw_response_preview": preview,
                })
            else:
                gs = parsed.get("gesamt_score", 0)
                scores = f"S={parsed.get('S',{}).get('score','?')} O={parsed.get('O',{}).get('score','?')} A={parsed.get('A',{}).get('score','?')} P={parsed.get('P',{}).get('score','?')}"
                print(f"→ {scores} gesamt={gs} ({urteil(gs)})")
                results.append({
                    "scenario": szenario, "model": model, "prompt_variante": pk,
                    "auto_skip": False,
                    "S": parsed.get("S", {}),
                    "O": parsed.get("O", {}),
                    "A": parsed.get("A", {}),
                    "P": parsed.get("P", {}),
                    "gesamt_score": gs,
                    "urteil": urteil(gs),
                })

        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nFertig. {len(results)} Einträge in {OUT_FILE}")


if __name__ == "__main__":
    main()
