# LLM-Fehleranalyse: Speechmatics + GPT-4o

> RAW STT → Formatted — Satzzeichen und Groß-/Kleinschreibung ignoriert.
> Speaker-Label-Änderungen sind bereits aus der JSON entfernt.
> **S** = Substitution | **D** = Löschung (im RAW, fehlt im FMT) | **I** = Einfügung (im FMT, nicht im RAW)

---

## Übersicht

| Szenario | RAW-Wörter | FMT-Wörter | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 239 | 239 | 0 | 0 | 0 | 0 | 0.0% |
| OriginalDC+Noise | 187 | 187 | 0 | 0 | 0 | 0 | 0.0% |
| LapInMitte | 230 | 230 | 0 | 0 | 0 | 0 | 0.0% |
| LapBeiArzt | 228 | 228 | 0 | 0 | 0 | 0 | 0.0% |
| Selbstkorrekturen | 185 | 185 | 0 | 0 | 0 | 0 | 0.0% |
| Unterbrechungen | 138 | 138 | 0 | 0 | 0 | 0 | 0.0% |
| Gedankensprünge | 194 | 194 | 0 | 0 | 0 | 0 | 0.0% |
| Meinungswechsel | 176 | 176 | 0 | 0 | 0 | 0 | 0.0% |
| Chaos | 256 | 256 | 0 | 0 | 0 | 0 | 0.0% |
| Anamnesegespräch | 2282 | 2281 | 0 | 1 | 0 | 1 | 0.0% |
| PWC | 1452 | 1452 | 0 | 0 | 0 | 0 | 0.0% |

---

## OriginalDC

**Fehlerrate: 0.0%** — RAW: 239 Wörter | FMT: 239 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## OriginalDC+Noise

**Fehlerrate: 0.0%** — RAW: 187 Wörter | FMT: 187 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## LapInMitte

**Fehlerrate: 0.0%** — RAW: 230 Wörter | FMT: 230 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## LapBeiArzt

**Fehlerrate: 0.0%** — RAW: 228 Wörter | FMT: 228 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Selbstkorrekturen

**Fehlerrate: 0.0%** — RAW: 185 Wörter | FMT: 185 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Unterbrechungen

**Fehlerrate: 0.0%** — RAW: 138 Wörter | FMT: 138 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Gedankensprünge

**Fehlerrate: 0.0%** — RAW: 194 Wörter | FMT: 194 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Meinungswechsel

**Fehlerrate: 0.0%** — RAW: 176 Wörter | FMT: 176 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Chaos

**Fehlerrate: 0.0%** — RAW: 256 Wörter | FMT: 256 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Anamnesegespräch

**Fehlerrate: 0.0%** — RAW: 2282 Wörter | FMT: 2281 Wörter | S=0 D=1 I=0 | Fehler=1

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Löschung | `okay` | `*(nicht da)*` | …für sie da [___] alles klar ich… |

---

## PWC

**Fehlerrate: 0.0%** — RAW: 1452 Wörter | FMT: 1452 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*
