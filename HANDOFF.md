# Session Handoff — 2026-05-31

Diese Datei beschreibt den exakten Stand nach der Session vom 31.05.2026.
Nach dem Lesen und Einordnen kann sie gelöscht werden.

---

## Was in dieser Session gemacht wurde

### Ausgangslage
Die Seminar-Präsentation wurde gehalten. Alle 55 Durchläufe (5 Modellkombinationen × 11 Szenarien)
lagen fertig analysiert vor. Zwei Schwächen wurden identifiziert und angegangen.

---

## Phase 0 — Restrukturierung (FERTIG)

Ziel: v1-Präsentationsstand einfrieren, sauber von v2-Weiterentwicklung trennen.

### Was verschoben wurde (`git mv`)
- `results/*.json` → `results/v1_praesentation/` (alle 8 Dateien: history.json, history_no_speaker.json, soap_eval_results.json, speechmatics.json, assemblyai.json, sauerkraut.json, llama3.2.json, gemma4.json)
- `docs/wer/*.md` → `docs/v1_praesentation/wer/` (6 Dateien)
- `docs/llm/*.md` → `docs/v1_praesentation/llm/` (6 Dateien)
- `docs/soap/soap_eval_auswertung.md` → `docs/v1_praesentation/soap_eval_auswertung.md`
- `docs/soap/soap_strukturcheck.md` → `docs/v1_praesentation/soap_strukturcheck.md`
- `docs/speaker/speaker_check.md` → `docs/v1_praesentation/speaker_check.md`

**Unverändert geblieben** (geteilte Eingaben):
- `docs/Seminar Ground Truth Texte.md`
- `docs/soap/soap_eval_prompt.md` (Evaluator-Prompt, von v1 und v2 shared)
- `docs/speaker/speaker_ground_truth.md`

### Angepasste Pfadkonstanten in v1-Skripten
Alle diese Skripte lesen/schreiben jetzt `v1_praesentation/`:
- `skript/wer/wer_base.py` — DATA_FILE + Ausgabe-Pfad
- `skript/llm/llm_check_base.py` — DATA_FILE + Ausgabe-Pfad
- `skript/llm/llm_check_sauerkraut.py` — hartkodierte Pfade
- `skript/soap/soap_strukturcheck.py` — DATA_FILE + OUT_FILE
- `skript/soap/soap_eval_runner.py` — DATA_FILE + OUT_FILE
- `skript/soap/soap_eval_auswertung.py` — IN_FILE + OUT_FILE
- `skript/soap/soap_eval_test.py` — DATA_FILE

---

## Phase 1 — Deterministische Sprecher-Formatierung (FERTIG)

### Problem
Die LLM hat beim Formatieren nicht nur die Speaker-Labels ersetzt, sondern auch Text
gekürzt/zusammengefasst (in der LLM-Fidelity-Analyse messbar). Das führte zu verfälschten
SOAP-Eingaben.

### Neue Logik
Die LLM liefert nur noch eine JSON-Rollen-Zuordnung:
`{"SPEAKER_00": "Arzt", "SPEAKER_01": "Frau Weber"}` etc.
Das Ersetzen + Zeilenumbruch nach jedem Sprecherwechsel passiert deterministisch per Skript.

### Neues Modul: `speaker_format.py` (Repo-Root, neben app.py)
Behandelt alle 3 STT-Label-Formate:
- **Whisper+pyannote** (inline): `SPEAKER_00: text`, `SPEAKER_??: text`
- **AssemblyAI** (inline): `Speaker A: text`, `Speaker B: text`
- **Speechmatics** (Label auf eigener Zeile): `SPEAKER: S1\n<Text>\nSPEAKER: S2\n...`

API:
```python
from speaker_format import parse_turns, distinct_labels, identify_roles, apply_mapping

turns   = parse_turns(raw)            # [(label, text), ...]
labels  = distinct_labels(turns)       # eindeutige Labels in Reihenfolge
mapping = identify_roles(raw, labels, client, model)  # LLM → {label: Rolle}
result  = apply_mapping(turns, mapping)  # deterministisch, text unverändert
```

Getestet: alle 3 Formate korrekt geparst (Python-Test lief durch).

### Geändert: `app.py`
- Import hinzugefügt: `from speaker_format import parse_turns, distinct_labels, identify_roles, apply_mapping`
- `format_transcript()` komplett umgebaut: kein LLM-Volltranskript-Umschreiben mehr, nur noch LLM-Rollenabfrage + deterministisches `apply_mapping`. Generator-Signatur beibehalten (ein `yield`).

### Neues Validierungsskript: `skript/format/redo_formatting.py`
- Liest `results/v1_praesentation/history.json` (alle 55, Labels intakt)
- Wendet neue Logik an, schreibt `results/v2/history_v2_formatted.json`
- Validiert Wort-für-Wort-Texterhalt (ohne Labels) für alle 55 Einträge
- Bericht → `docs/v2/format_v2_validierung.md`
- Resume-fähig (überspringt bereits verarbeitete Einträge)

**NOCH NICHT AUSGEFÜHRT** — braucht Ollama lokal + OPENAI_API_KEY.

---

## Phase 2 — SOAP-Prompt-Engineering (FERTIG — Code, nicht Läufe)

### Problem
Der SOAP-Prompt war sehr einfach und führte zu Halluzinationen (v.a. in A und P),
was in der SOAP-Evaluation direkt zu Score 0 führte.

### Idee
Nur 3 Szenarien × 2 Modelle × 2 Prompts = 12 Läufe testen statt alle 55 neu.
Eingabe ist der Ground-Truth-Text (nicht STT-Output) → testet den Prompt isoliert.
Modelle: GPT-4o (Cloud) und llama3.2 (lokal) → wenn llama3.2 besser wird, werden alle besser.

### Neue Dateien in `skript/soap_v2/`

**`soap_prompts.py`** — enthält:
- `SOAP_BASELINE` (wörtlich der alte Prompt aus app.py)
- `SOAP_KANDIDAT` (neuer Prompt mit Anti-Halluzination-Regeln, „Keine Angabe im Gespräch", feste Struktur)

**`soap_prompt_test.py`** — 12 SOAP-Generierungen
- Liest GT-Texte aus `docs/Seminar Ground Truth Texte.md`
- Szenarien: OriginalDC (einfach), Chaos (schwer), Anamnesegespräch (lang)
- Modelle: gpt-4o, llama3.2
- Schreibt `results/v2/soap_prompt_test.json` (resume-fähig)

**`soap_prompt_test_eval.py`** — Claude-Evaluation der 12 SOAPs
- Nutzt `claude -p` + bestehenden `docs/soap/soap_eval_prompt.md`
- Schreibt `results/v2/soap_prompt_test_eval.json`

**`soap_prompt_test_auswertung.py`** — Bericht
- Vergleichstabelle Baseline vs. Kandidat (S/O/A/P, gesamt_score, Δ)
- Mittelwerte je Variante
- Schreibt `docs/v2/soap_prompt_test_auswertung.md`

**NOCH NICHT AUSGEFÜHRT** — braucht Ollama + OPENAI_API_KEY + `claude` CLI.

---

## Was morgen zu tun ist

### 1. Skripte ausführen (in dieser Reihenfolge)
```bash
# Formatierung neu ableiten + validieren (braucht Ollama lokal + OPENAI_API_KEY)
cd skript/format && python3 redo_formatting.py

# SOAP-Prompt-Test (braucht Ollama llama3.2 + OPENAI_API_KEY)
cd skript/soap_v2 && python3 soap_prompt_test.py

# Claude-Evaluation (braucht 'claude' CLI, eingeloggt)
cd skript/soap_v2 && python3 soap_prompt_test_eval.py

# Auswertungsbericht
cd skript/soap_v2 && python3 soap_prompt_test_auswertung.py
```

### 2. Ergebnisse prüfen
- `docs/v2/format_v2_validierung.md` — alle Einträge sollten "OK" zeigen
- `docs/v2/soap_prompt_test_auswertung.md` — Δ sollte positiv sein (Kandidat besser)

### 3. Evtl. SOAP-Kandidat-Prompt iterieren
Wenn die Ergebnisse noch nicht gut genug sind: `skript/soap_v2/soap_prompts.py` editieren,
`SOAP_KANDIDAT` anpassen, und die Skripte erneut laufen lassen (resume überspringt bereits OK-Einträge).
Um von vorn zu starten: `results/v2/soap_prompt_test.json` löschen.

---

## Aktueller Stand des Git-Repos

Noch **nicht committed**. Folgende Änderungen sind staged/unstaged:
- `R` (renamed via git mv): alle verschobenen docs/ und results/-Dateien
- `M` (modified): app.py, CLAUDE.md, alle 7 Analyse-Skripte
- `?` (untracked): speaker_format.py, skript/format/, skript/soap_v2/, HANDOFF.md

Für den Commit:
```bash
git add speaker_format.py skript/format/ skript/soap_v2/ HANDOFF.md CLAUDE.md
git add skript/wer/wer_base.py skript/llm/ skript/soap/soap_eval_runner.py skript/soap/soap_eval_auswertung.py skript/soap/soap_strukturcheck.py skript/soap/soap_eval_test.py
git add app.py
git commit -m "Restrukturierung v1↔v2: Speaker-Formatierung deterministisch, SOAP-Prompt-Engineering vorbereitet"
git push github main
git push origin main
```

---

## Wichtige Datei-Übersicht (neu)

```
speaker_format.py              ← neues Kernmodul (alle 3 STT-Formate)
app.py                         ← format_transcript() nutzt jetzt speaker_format.py
skript/format/redo_formatting.py   ← Validierung über alle 55 v1-Einträge
skript/soap_v2/soap_prompts.py         ← Baseline + Kandidat-Prompt
skript/soap_v2/soap_prompt_test.py     ← 12 SOAP-Generierungen (GT-Eingabe)
skript/soap_v2/soap_prompt_test_eval.py    ← Claude-Evaluation
skript/soap_v2/soap_prompt_test_auswertung.py  ← Bericht
results/v1_praesentation/      ← eingefroren, nicht anfassen
results/v2/                    ← neue Ergebnisse kommen hier rein
docs/v1_praesentation/         ← eingefroren, nicht anfassen
docs/v2/                       ← neue generierte Docs kommen hier rein
```
