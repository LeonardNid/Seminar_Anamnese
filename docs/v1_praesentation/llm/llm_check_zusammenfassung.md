# LLM-Check-Zusammenfassung: Alle Modelle

> Gemessen wird die **Texttreue des LLM** beim Format-Schritt: RAW STT → Formatted.
> Speaker-Label-Änderungen sind bereits aus der JSON entfernt — nur inhaltliche Textänderungen zählen.
> Eine hohe Fehlerrate bedeutet, dass das LLM den Gesprächstext über das Label-Ersetzen hinaus verändert hat.

---

## Formel

```
Fehlerrate = Fehler / RAW-Wörter × 100 %

Fehler (Edit-Distanz) = S + D + I
```

| Symbol | Bedeutung |
|---|---|
| **S** | Substitution — Wort umformuliert |
| **D** | Deletion / Löschung — Wort/Passage aus RAW fehlt in FMT |
| **I** | Insertion / Einfügung — extra Inhalt in FMT, nicht im RAW |
| **RAW-Wörter** | Anzahl der Wörter im STT-Rohtranskript (Nenner) |

**Mikro-Fehlerrate** (gewichtet): Σ Fehler aller Szenarien ÷ Σ RAW-Wörter aller Szenarien.
Lange Dateien (Anamnesegespräch, PWC) dominieren diesen Wert.

**Makro-Fehlerrate** (ungewichtet): arithmetisches Mittel der Einzel-Fehlerraten.
Jedes Szenario zählt gleich.

> Hinweis: Der große Ausreißer bei Anamnesegespräch (97 %) entsteht, weil das lange Transkript
> das Kontextfenster lokaler Modelle übersteigt — das LLM gibt dann nur einen Bruchteil des Textes
> zurück (massive Löschungen). GPT-4o (128k Kontextfenster) ist davon nicht betroffen.

---

## Modell-Informationen

| Kürzel | LLM | Kontextfenster |
|---|---|---|
| **Whisper+Sauerkraut** | `hf.co/QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF:Q4_K_M` | 131.072 Tokens |
| **Whisper+llama3.2** | `llama3.2` — Llama 3.2 3B (Ollama) | 131.072 Tokens |
| **Whisper+gemma4** | `gemma4:e4b` (Ollama) | 131.072 Tokens |
| **Speechmatics+GPT4o** | `gpt-4o` (OpenAI) | 128.000 Tokens |
| **AssemblyAI+GPT4o** | `gpt-4o` (OpenAI) | 128.000 Tokens |

---

## Gesamtübersicht pro Modell

| Modell | Σ RAW-Wörter | Σ S | Σ D | Σ I | Σ Fehler | Mikro-Fehlerrate | Makro-Fehlerrate |
|---|---|---|---|---|---|---|---|
| Whisper+Sauerkraut | 5.663 | 112 | 2.722 | 229 | 3.063 | **54,1 %** | **19,7 %** |
| Whisper+llama3.2 | 5.650 | 104 | 2.437 | 76 | 2.617 | **46,3 %** | **15,1 %** |
| Whisper+gemma4 | 5.649 | 566 | 2.360 | 0 | 2.926 | **51,8 %** | **13,3 %** |
| Speechmatics+GPT4o | 5.567 | 0 | 1 | 0 | 1 | **0,0 %** | **0,0 %** |
| AssemblyAI+GPT4o | 5.651 | 0 | 4 | 0 | 4 | **0,1 %** | **0,0 %** |

**Berechnungsbeispiel Mikro-Fehlerrate (Whisper+Sauerkraut):**
```
Mikro-Fehlerrate = 3.063 / 5.663 × 100 = 54,1 %
```

**Berechnungsbeispiel Makro-Fehlerrate (Whisper+Sauerkraut):**
```
Makro-Fehlerrate = (4,2 + 0,0 + 0,0 + 0,9 + 36,2 + 11,2 + 0,0 + 1,6 + 19,4 + 97,1 + 46,5) / 11
                 = 217,1 / 11
                 = 19,7 %
```

---

## Fehlerrate pro Szenario (alle Modelle)

| Szenario | RAW¹ | Whisper+Sauerkraut | Whisper+llama3.2 | Whisper+gemma4 | Speechmatics+GPT4o | AssemblyAI+GPT4o |
|---|---|---|---|---|---|---|
| OriginalDC | ~232 | 4,2 % | 15,8 % | 0,4 % | **0,0 %** | **0,0 %** |
| OriginalDC+Noise | ~210 | **0,0 %** | 1,0 % | **0,0 %** | **0,0 %** | **0,0 %** |
| LapInMitte | ~227 | **0,0 %** | **0,0 %** | **0,0 %** | **0,0 %** | **0,0 %** |
| LapBeiArzt | ~228 | 0,9 % | 13,5 % | **0,0 %** | **0,0 %** | **0,0 %** |
| Selbstkorrekturen | ~190 | 36,2 % | 1,6 % | **0,0 %** | **0,0 %** | **0,0 %** |
| Unterbrechungen | ~143 | 11,2 % | 14,0 % | **0,0 %** | **0,0 %** | **0,0 %** |
| Gedankensprünge | ~190 | **0,0 %** | **0,0 %** | **0,0 %** | **0,0 %** | **0,0 %** |
| Meinungswechsel | ~181 | 1,6 % | 1,1 % | **0,0 %** | **0,0 %** | **0,0 %** |
| Chaos | ~255 | 19,4 % | 0,8 % | **0,0 %** | **0,0 %** | **0,0 %** |
| Anamnesegespräch | ~2.280 | 97,1 % ⚠ | 97,1 % ⚠ | 93,8 % ⚠ | **0,0 %** | **0,0 %** |
| PWC | ~1.499 | 46,5 % | 20,8 % | 52,6 % | **0,0 %** | **0,3 %** |

> ¹ RAW-Wörter variieren leicht je Modell (nicht-deterministischer Whisper-Run); hier gerundeter Mittelwert.
> ⚠ Kontextfenster überschritten — LLM hat nur einen Bruchteil des Textes ausgegeben (Bulk-Löschungen).
> Fettdruck = beste(r) Wert(e) je Szenario.

---

## Edit-Distanz-Aufschlüsselung pro Szenario

### Whisper+Sauerkraut

| Szenario | RAW | FMT | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 237 | 247 | 0 | 0 | 10 | 10 | 4,2 % |
| OriginalDC+Noise | 210 | 210 | 0 | 0 | 0 | 0 | 0,0 % |
| LapInMitte | 226 | 226 | 0 | 0 | 0 | 0 | 0,0 % |
| LapBeiArzt | 229 | 230 | 1 | 0 | 1 | 2 | 0,9 % |
| Selbstkorrekturen | 210 | 278 | 2 | 3 | 71 | 76 | 36,2 % |
| Unterbrechungen | 143 | 145 | 0 | 7 | 9 | 16 | 11,2 % |
| Gedankensprünge | 190 | 190 | 0 | 0 | 0 | 0 | 0,0 % |
| Meinungswechsel | 185 | 186 | 2 | 0 | 1 | 3 | 1,6 % |
| Chaos | 252 | 301 | 0 | 0 | 49 | 49 | 19,4 % |
| Anamnesegespräch | 2.269 | 152 | 87 | 2.117 | 0 | 2.204 | 97,1 % |
| PWC | 1.512 | 1.005 | 20 | 595 | 88 | 703 | 46,5 % |
| **Gesamt** | **5.663** | | **112** | **2.722** | **229** | **3.063** | **Mikro 54,1 % / Makro 19,7 %** |

### Whisper+llama3.2

| Szenario | RAW | FMT | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 247 | 269 | 5 | 6 | 28 | 39 | 15,8 % |
| OriginalDC+Noise | 210 | 210 | 2 | 0 | 0 | 2 | 1,0 % |
| LapInMitte | 226 | 226 | 0 | 0 | 0 | 0 | 0,0 % |
| LapBeiArzt | 229 | 252 | 6 | 1 | 24 | 31 | 13,5 % |
| Selbstkorrekturen | 190 | 193 | 0 | 0 | 3 | 3 | 1,6 % |
| Unterbrechungen | 143 | 159 | 4 | 0 | 16 | 20 | 14,0 % |
| Gedankensprünge | 190 | 190 | 0 | 0 | 0 | 0 | 0,0 % |
| Meinungswechsel | 183 | 185 | 0 | 0 | 2 | 2 | 1,1 % |
| Chaos | 252 | 253 | 1 | 0 | 1 | 2 | 0,8 % |
| Anamnesegespräch | 2.269 | 149 | 82 | 2.121 | 1 | 2.204 | 97,1 % |
| PWC | 1.511 | 1.203 | 4 | 309 | 1 | 314 | 20,8 % |
| **Gesamt** | **5.650** | | **104** | **2.437** | **76** | **2.617** | **Mikro 46,3 % / Makro 15,1 %** |

### Whisper+gemma4

| Szenario | RAW | FMT | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 235 | 234 | 0 | 1 | 0 | 1 | 0,4 % |
| OriginalDC+Noise | 210 | 210 | 0 | 0 | 0 | 0 | 0,0 % |
| LapInMitte | 226 | 226 | 0 | 0 | 0 | 0 | 0,0 % |
| LapBeiArzt | 229 | 229 | 0 | 0 | 0 | 0 | 0,0 % |
| Selbstkorrekturen | 200 | 200 | 0 | 0 | 0 | 0 | 0,0 % |
| Unterbrechungen | 143 | 143 | 0 | 0 | 0 | 0 | 0,0 % |
| Gedankensprünge | 190 | 190 | 0 | 0 | 0 | 0 | 0,0 % |
| Meinungswechsel | 183 | 183 | 0 | 0 | 0 | 0 | 0,0 % |
| Chaos | 252 | 252 | 0 | 0 | 0 | 0 | 0,0 % |
| Anamnesegespräch | 2.269 | 478 | 338 | 1.791 | 0 | 2.129 | 93,8 % |
| PWC | 1.512 | 944 | 228 | 568 | 0 | 796 | 52,6 % |
| **Gesamt** | **5.649** | | **566** | **2.360** | **0** | **2.926** | **Mikro 51,8 % / Makro 13,3 %** |

### Speechmatics+GPT4o

| Szenario | RAW | FMT | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 239 | 239 | 0 | 0 | 0 | 0 | 0,0 % |
| OriginalDC+Noise | 187 | 187 | 0 | 0 | 0 | 0 | 0,0 % |
| LapInMitte | 230 | 230 | 0 | 0 | 0 | 0 | 0,0 % |
| LapBeiArzt | 228 | 228 | 0 | 0 | 0 | 0 | 0,0 % |
| Selbstkorrekturen | 185 | 185 | 0 | 0 | 0 | 0 | 0,0 % |
| Unterbrechungen | 138 | 138 | 0 | 0 | 0 | 0 | 0,0 % |
| Gedankensprünge | 194 | 194 | 0 | 0 | 0 | 0 | 0,0 % |
| Meinungswechsel | 176 | 176 | 0 | 0 | 0 | 0 | 0,0 % |
| Chaos | 256 | 256 | 0 | 0 | 0 | 0 | 0,0 % |
| Anamnesegespräch | 2.282 | 2.281 | 0 | 1 | 0 | 1 | 0,0 % |
| PWC | 1.452 | 1.452 | 0 | 0 | 0 | 0 | 0,0 % |
| **Gesamt** | **5.567** | | **0** | **1** | **0** | **1** | **Mikro 0,0 % / Makro 0,0 %** |

### AssemblyAI+GPT4o

| Szenario | RAW | FMT | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 229 | 229 | 0 | 0 | 0 | 0 | 0,0 % |
| OriginalDC+Noise | 221 | 221 | 0 | 0 | 0 | 0 | 0,0 % |
| LapInMitte | 227 | 227 | 0 | 0 | 0 | 0 | 0,0 % |
| LapBeiArzt | 226 | 226 | 0 | 0 | 0 | 0 | 0,0 % |
| Selbstkorrekturen | 187 | 187 | 0 | 0 | 0 | 0 | 0,0 % |
| Unterbrechungen | 150 | 150 | 0 | 0 | 0 | 0 | 0,0 % |
| Gedankensprünge | 190 | 190 | 0 | 0 | 0 | 0 | 0,0 % |
| Meinungswechsel | 178 | 178 | 0 | 0 | 0 | 0 | 0,0 % |
| Chaos | 260 | 260 | 0 | 0 | 0 | 0 | 0,0 % |
| Anamnesegespräch | 2.299 | 2.299 | 0 | 0 | 0 | 0 | 0,0 % |
| PWC | 1.484 | 1.480 | 0 | 4 | 0 | 4 | 0,3 % |
| **Gesamt** | **5.651** | | **0** | **4** | **0** | **4** | **Mikro 0,1 % / Makro 0,0 %** |

---

## Makro-Fehlerrate Berechnung (alle Modelle)

```
Whisper+Sauerkraut:
  (4,2 + 0,0 + 0,0 + 0,9 + 36,2 + 11,2 + 0,0 + 1,6 + 19,4 + 97,1 + 46,5) / 11
= 217,1 / 11 = 19,7 %

Whisper+llama3.2:
  (15,8 + 1,0 + 0,0 + 13,5 + 1,6 + 14,0 + 0,0 + 1,1 + 0,8 + 97,1 + 20,8) / 11
= 165,7 / 11 = 15,1 %

Whisper+gemma4:
  (0,4 + 0,0 + 0,0 + 0,0 + 0,0 + 0,0 + 0,0 + 0,0 + 0,0 + 93,8 + 52,6) / 11
= 146,8 / 11 = 13,3 %

Speechmatics+GPT4o:
  (0,0 × 11) / 11 = 0,0 %
  (einziger Fehler: Anamnesegespräch 1/2282 = 0,04 % → gerundet 0,0 %)

AssemblyAI+GPT4o:
  (0,0 × 10 + 0,3) / 11 = 0,3 / 11 = 0,03 % ≈ 0,0 %
```
