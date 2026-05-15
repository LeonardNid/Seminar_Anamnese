# WER-Zusammenfassung: Alle Modelle

> Vergleich der STT-Qualität aller 5 Modell-Kombinationen.
> Gemessen wird ausschließlich die **Rohausgabe des STT** gegen den Ground-Truth-Text.
> Das LLM beeinflusst den WER-Wert nicht — er ist eine reine STT-Metrik.

---

## Formel

```
WER = Edit-Distanz / Ref-Wörter × 100 %

Edit-Distanz = S + D + I
```

| Symbol | Bedeutung |
|---|---|
| **S** | Substitution — Wort falsch erkannt (z.B. `das` → `es`) |
| **D** | Deletion / Löschung — Wort fehlt im STT-Output |
| **I** | Insertion / Einfügung — extra Wort im STT-Output, nicht im Ground Truth |
| **Ref-Wörter** | Anzahl der Wörter im Ground-Truth-Text (Nenner) |

**Mikro-WER** (gewichtet): Σ Edit-Distanzen aller Szenarien ÷ Σ Ref-Wörter aller Szenarien.
Lange Dateien (Anamnesegespräch, PWC) dominieren diesen Wert.

**Makro-WER** (ungewichtet): arithmetisches Mittel der Einzel-WER-Werte.
Jedes Szenario zählt gleich.

---

## Modell-Informationen

| Kürzel | STT | LLM |
|---|---|---|
| **Whisper+Sauerkraut** | faster-whisper large-v3-turbo (lokal) | `hf.co/QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF:Q4_K_M` |
| **Whisper+llama3.2** | faster-whisper large-v3-turbo (lokal) | `llama3.2` (Ollama) |
| **Whisper+gemma4** | faster-whisper large-v3-turbo (lokal) | `gemma4:e4b` (Ollama) |
| **Speechmatics+GPT4o** | Speechmatics Cloud, enhanced | `gpt-4o` (OpenAI) |
| **AssemblyAI+GPT4o** | AssemblyAI Cloud, universal-3-pro | `gpt-4o` (OpenAI) |

---

## Gesamtübersicht pro Modell

Basis: 11 Szenarien, Σ Ref-Wörter = **5.744**

| Modell | Σ S | Σ D | Σ I | Σ Edit-Dist | Mikro-WER | Makro-WER |
|---|---|---|---|---|---|---|
| Whisper+Sauerkraut | 337 | 239 | 166 | 742 | **12,9 %** | **13,9 %** |
| Whisper+llama3.2 | 336 | 239 | 154 | 729 | **12,7 %** | **13,2 %** |
| Whisper+gemma4 | 337 | 246 | 161 | 744 | **13,0 %** | **13,3 %** |
| Speechmatics+GPT4o | 289 | 277 | 100 | 666 | **11,6 %** | **11,4 %** |
| AssemblyAI+GPT4o | 178 | 151 | 58 | 387 | **6,7 %** | **8,9 %** |

**Berechnungsbeispiel Mikro-WER (AssemblyAI+GPT4o):**
```
Mikro-WER = 387 / 5.744 × 100 = 6,74 % ≈ 6,7 %
```

**Berechnungsbeispiel Makro-WER (AssemblyAI+GPT4o):**
```
Makro-WER = (4,3 + 33,2 + 4,3 + 2,7 + 8,7 + 13,1 + 2,6 + 1,1 + 18,0 + 4,4 + 5,9) / 11
          = 98,3 / 11
          = 8,94 % ≈ 8,9 %
```

---

## WER pro Szenario (alle Modelle)

| Szenario | Ref-Wörter | Whisper+Sauerkraut | Whisper+llama3.2 | Whisper+gemma4 | Speechmatics+GPT4o | AssemblyAI+GPT4o |
|---|---|---|---|---|---|---|
| OriginalDC.m4a | 232 | 8,6 % | 12,9 % | 7,8 % | 5,6 % | **4,3 %** |
| OriginalDCWhiteNoise.m4a | 229 | 41,0 % | 41,0 % | 41,0 % | 30,6 % | **33,2 %** |
| OriginalLapInMitte.wav | 231 | 6,5 % | 6,5 % | 6,5 % | **5,6 %** | **4,3 %** |
| OriginalLapBeiArzt.wav | 226 | 8,4 % | 9,3 % | 8,8 % | 5,3 % | **2,7 %** |
| SelbstkorrekturLapInMitte.wav | 183 | 21,3 % | 9,8 % | 15,8 % | 10,4 % | **8,7 %** |
| UnterbrechungLapInMitte.wav | 153 | 13,1 % | 13,1 % | 13,1 % | 17,0 % | **13,1 %** |
| GedankenprüngeLapInMitte.wav | 192 | 2,6 % | 2,6 % | 2,6 % | 4,2 % | **2,6 %** |
| MeinungswechselLapinMitte.wav | 179 | 7,8 % | 6,7 % | 6,7 % | 3,4 % | **1,1 %** |
| ChaosLapInMitte.wav | 272 | 15,8 % | 15,8 % | 15,8 % | 15,4 % | **18,0 %** |
| Das Anamnesegespräch.wav | 2317 | 6,5 % | 6,4 % | 7,1 % | **4,5 %** | 4,4 % |
| Anamnesegesrpäch PWC.mp3 | 1530 | 21,1 % | 21,0 % | 21,1 % | 23,1 % | **5,9 %** |

> Fettdruck = bester Wert je Szenario.
> Hinweis: Whisper+Sauerkraut / llama3.2 / gemma4 nutzen dieselbe STT-Engine, minimale Abweichungen entstehen durch nicht-deterministische Whisper-Runs.

---

## Edit-Distanz-Aufschlüsselung pro Szenario

### Whisper+Sauerkraut

| Szenario | Ref | S | D | I | Edit-Dist | WER |
|---|---|---|---|---|---|---|
| OriginalDC.m4a | 232 | 8 | 3 | 9 | 20 | 8,6 % |
| OriginalDCWhiteNoise.m4a | 229 | 59 | 27 | 8 | 94 | 41,0 % |
| OriginalLapInMitte.wav | 231 | 8 | 6 | 1 | 15 | 6,5 % |
| OriginalLapBeiArzt.wav | 226 | 12 | 2 | 5 | 19 | 8,4 % |
| SelbstkorrekturLapInMitte.wav | 183 | 11 | 0 | 28 | 39 | 21,3 % |
| UnterbrechungLapInMitte.wav | 153 | 8 | 11 | 1 | 20 | 13,1 % |
| GedankenprüngeLapInMitte.wav | 192 | 3 | 2 | 0 | 5 | 2,6 % |
| MeinungswechselLapinMitte.wav | 179 | 3 | 2 | 9 | 14 | 7,8 % |
| ChaosLapInMitte.wav | 272 | 15 | 24 | 4 | 43 | 15,8 % |
| Das Anamnesegespräch.wav | 2317 | 77 | 60 | 13 | 150 | 6,5 % |
| Anamnesegesrpäch PWC.mp3 | 1530 | 133 | 102 | 88 | 323 | 21,1 % |
| **Gesamt** | **5.744** | **337** | **239** | **166** | **742** | **Mikro 12,9 % / Makro 13,9 %** |

### Whisper+llama3.2

| Szenario | Ref | S | D | I | Edit-Dist | WER |
|---|---|---|---|---|---|---|
| OriginalDC.m4a | 232 | 8 | 3 | 19 | 30 | 12,9 % |
| OriginalDCWhiteNoise.m4a | 229 | 59 | 27 | 8 | 94 | 41,0 % |
| OriginalLapInMitte.wav | 231 | 8 | 6 | 1 | 15 | 6,5 % |
| OriginalLapBeiArzt.wav | 226 | 12 | 2 | 7 | 21 | 9,3 % |
| SelbstkorrekturLapInMitte.wav | 183 | 11 | 0 | 7 | 18 | 9,8 % |
| UnterbrechungLapInMitte.wav | 153 | 8 | 11 | 1 | 20 | 13,1 % |
| GedankenprüngeLapInMitte.wav | 192 | 3 | 2 | 0 | 5 | 2,6 % |
| MeinungswechselLapinMitte.wav | 179 | 3 | 2 | 7 | 12 | 6,7 % |
| ChaosLapInMitte.wav | 272 | 15 | 24 | 4 | 43 | 15,8 % |
| Das Anamnesegespräch.wav | 2317 | 76 | 60 | 13 | 149 | 6,4 % |
| Anamnesegesrpäch PWC.mp3 | 1530 | 133 | 102 | 87 | 322 | 21,0 % |
| **Gesamt** | **5.744** | **336** | **239** | **154** | **729** | **Mikro 12,7 % / Makro 13,2 %** |

### Whisper+gemma4

| Szenario | Ref | S | D | I | Edit-Dist | WER |
|---|---|---|---|---|---|---|
| OriginalDC.m4a | 232 | 8 | 3 | 7 | 18 | 7,8 % |
| OriginalDCWhiteNoise.m4a | 229 | 59 | 27 | 8 | 94 | 41,0 % |
| OriginalLapInMitte.wav | 231 | 8 | 6 | 1 | 15 | 6,5 % |
| OriginalLapBeiArzt.wav | 226 | 12 | 2 | 6 | 20 | 8,8 % |
| SelbstkorrekturLapInMitte.wav | 183 | 11 | 0 | 18 | 29 | 15,8 % |
| UnterbrechungLapInMitte.wav | 153 | 8 | 11 | 1 | 20 | 13,1 % |
| GedankenprüngeLapInMitte.wav | 192 | 3 | 2 | 0 | 5 | 2,6 % |
| MeinungswechselLapinMitte.wav | 179 | 3 | 2 | 7 | 12 | 6,7 % |
| ChaosLapInMitte.wav | 272 | 15 | 24 | 4 | 43 | 15,8 % |
| Das Anamnesegespräch.wav | 2317 | 77 | 67 | 21 | 165 | 7,1 % |
| Anamnesegesrpäch PWC.mp3 | 1530 | 133 | 102 | 88 | 323 | 21,1 % |
| **Gesamt** | **5.744** | **337** | **246** | **161** | **744** | **Mikro 13,0 % / Makro 13,3 %** |

### Speechmatics+GPT4o

| Szenario | Ref | S | D | I | Edit-Dist | WER |
|---|---|---|---|---|---|---|
| OriginalDC.m4a | 232 | 6 | 0 | 7 | 13 | 5,6 % |
| OriginalDCWhiteNoise.m4a | 229 | 26 | 43 | 1 | 70 | 30,6 % |
| OriginalLapInMitte.wav | 231 | 6 | 4 | 3 | 13 | 5,6 % |
| OriginalLapBeiArzt.wav | 226 | 6 | 2 | 4 | 12 | 5,3 % |
| SelbstkorrekturLapInMitte.wav | 183 | 9 | 4 | 6 | 19 | 10,4 % |
| UnterbrechungLapInMitte.wav | 153 | 11 | 15 | 0 | 26 | 17,0 % |
| GedankenprüngeLapInMitte.wav | 192 | 4 | 1 | 3 | 8 | 4,2 % |
| MeinungswechselLapinMitte.wav | 179 | 3 | 3 | 0 | 6 | 3,4 % |
| ChaosLapInMitte.wav | 272 | 18 | 20 | 4 | 42 | 15,4 % |
| Das Anamnesegespräch.wav | 2317 | 51 | 44 | 9 | 104 | 4,5 % |
| Anamnesegesrpäch PWC.mp3 | 1530 | 149 | 141 | 63 | 353 | 23,1 % |
| **Gesamt** | **5.744** | **289** | **277** | **100** | **666** | **Mikro 11,6 % / Makro 11,4 %** |

### AssemblyAI+GPT4o

| Szenario | Ref | S | D | I | Edit-Dist | WER |
|---|---|---|---|---|---|---|
| OriginalDC.m4a | 232 | 5 | 4 | 1 | 10 | 4,3 % |
| OriginalDCWhiteNoise.m4a | 229 | 42 | 21 | 13 | 76 | 33,2 % |
| OriginalLapInMitte.wav | 231 | 4 | 5 | 1 | 10 | 4,3 % |
| OriginalLapBeiArzt.wav | 226 | 4 | 1 | 1 | 6 | 2,7 % |
| SelbstkorrekturLapInMitte.wav | 183 | 10 | 1 | 5 | 16 | 8,7 % |
| UnterbrechungLapInMitte.wav | 153 | 5 | 9 | 6 | 20 | 13,1 % |
| GedankenprüngeLapInMitte.wav | 192 | 3 | 2 | 0 | 5 | 2,6 % |
| MeinungswechselLapinMitte.wav | 179 | 1 | 1 | 0 | 2 | 1,1 % |
| ChaosLapInMitte.wav | 272 | 17 | 22 | 10 | 49 | 18,0 % |
| Das Anamnesegespräch.wav | 2317 | 65 | 28 | 10 | 103 | 4,4 % |
| Anamnesegesrpäch PWC.mp3 | 1530 | 22 | 57 | 11 | 90 | 5,9 % |
| **Gesamt** | **5.744** | **178** | **151** | **58** | **387** | **Mikro 6,7 % / Makro 8,9 %** |

---

## Makro-WER Berechnung (alle Modelle)

```
Whisper+Sauerkraut:
  (8,6 + 41,0 + 6,5 + 8,4 + 21,3 + 13,1 + 2,6 + 7,8 + 15,8 + 6,5 + 21,1) / 11
= 152,7 / 11 = 13,9 %

Whisper+llama3.2:
  (12,9 + 41,0 + 6,5 + 9,3 + 9,8 + 13,1 + 2,6 + 6,7 + 15,8 + 6,4 + 21,0) / 11
= 145,1 / 11 = 13,2 %

Whisper+gemma4:
  (7,8 + 41,0 + 6,5 + 8,8 + 15,8 + 13,1 + 2,6 + 6,7 + 15,8 + 7,1 + 21,1) / 11
= 146,3 / 11 = 13,3 %

Speechmatics+GPT4o:
  (5,6 + 30,6 + 5,6 + 5,3 + 10,4 + 17,0 + 4,2 + 3,4 + 15,4 + 4,5 + 23,1) / 11
= 125,1 / 11 = 11,4 %

AssemblyAI+GPT4o:
  (4,3 + 33,2 + 4,3 + 2,7 + 8,7 + 13,1 + 2,6 + 1,1 + 18,0 + 4,4 + 5,9) / 11
= 98,3 / 11 = 8,9 %
```
