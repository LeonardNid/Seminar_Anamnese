# LLM-Fehleranalyse: AssemblyAI + GPT-4o

> RAW STT → Formatted — Satzzeichen und Groß-/Kleinschreibung ignoriert. <br>
> Speaker-Label-Änderungen sind bereits aus der JSON entfernt.<br>
> **S** = Substitution | **D** = Löschung (im RAW, fehlt im FMT) | **I** = Einfügung (im FMT, nicht im RAW)<br>

---

## Modell-Informationen

| Komponente | Exakte Bezeichnung | Kontextfenster |
|---|---|---|
| STT | AssemblyAI Cloud — `universal-3-pro` (Speaker Diarization aktiviert) | — |
| LLM | `gpt-4o` (OpenAI) | 128.000 Tokens |

---

## Übersicht

| Szenario | RAW-Wörter | FMT-Wörter | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 229 | 229 | 0 | 0 | 0 | 0 | 0.0% |
| OriginalDC+Noise | 221 | 221 | 0 | 0 | 0 | 0 | 0.0% |
| LapInMitte | 227 | 227 | 0 | 0 | 0 | 0 | 0.0% |
| LapBeiArzt | 226 | 226 | 0 | 0 | 0 | 0 | 0.0% |
| Selbstkorrekturen | 187 | 187 | 0 | 0 | 0 | 0 | 0.0% |
| Unterbrechungen | 150 | 150 | 0 | 0 | 0 | 0 | 0.0% |
| Gedankensprünge | 190 | 190 | 0 | 0 | 0 | 0 | 0.0% |
| Meinungswechsel | 178 | 178 | 0 | 0 | 0 | 0 | 0.0% |
| Chaos | 260 | 260 | 0 | 0 | 0 | 0 | 0.0% |
| Anamnesegespräch | 2299 | 2299 | 0 | 0 | 0 | 0 | 0.0% |
| PWC | 1484 | 1480 | 0 | 4 | 0 | 4 | 0.3% |

---

## OriginalDC

**Fehlerrate: 0.0%** — RAW: 229 Wörter | FMT: 229 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## OriginalDC+Noise

**Fehlerrate: 0.0%** — RAW: 221 Wörter | FMT: 221 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## LapInMitte

**Fehlerrate: 0.0%** — RAW: 227 Wörter | FMT: 227 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## LapBeiArzt

**Fehlerrate: 0.0%** — RAW: 226 Wörter | FMT: 226 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Selbstkorrekturen

**Fehlerrate: 0.0%** — RAW: 187 Wörter | FMT: 187 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Unterbrechungen

**Fehlerrate: 0.0%** — RAW: 150 Wörter | FMT: 150 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Gedankensprünge

**Fehlerrate: 0.0%** — RAW: 190 Wörter | FMT: 190 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Meinungswechsel

**Fehlerrate: 0.0%** — RAW: 178 Wörter | FMT: 178 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Chaos

**Fehlerrate: 0.0%** — RAW: 260 Wörter | FMT: 260 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Anamnesegespräch

**Fehlerrate: 0.0%** — RAW: 2299 Wörter | FMT: 2299 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## PWC

**Fehlerrate: 0.3%** — RAW: 1484 Wörter | FMT: 1480 Wörter | S=0 D=4 I=0 | Fehler=4

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Löschung | `what` | `*(nicht da)*` | …nächsten behandlung danke [___] is it boss… |
| 2 | Löschung | `is` | `*(nicht da)*` | …behandlung danke what [___] it boss… |
| 3 | Löschung | `it` | `*(nicht da)*` | …danke what is [___] boss… |
| 4 | Löschung | `boss` | `*(nicht da)*` | …what is it [___]… |
