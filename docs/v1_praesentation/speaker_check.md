# Speaker-Identifikation: LLM-Analyse (Raw → Formatted)

> Prüft ob das LLM Speaker-Labels korrekt von SPEAKER_XX zu Arzt/Patientenname übersetzt hat.
> **A✓/✗** = Arzt-Label korrekt | **P✓/✗** = Patientenname korrekt
> **⚠ STT: nur 1 Speaker** = STT hat keine Diarisierung geliefert, LLM konnte Speaker nicht trennen
> **⚠ LLM Swap** = STT hatte 2 Speaker, aber LLM hat sie vertauscht

## Übersicht

| Szenario | Whisper+llama3.2 | Speechmatics+GPT4o | AssemblyAI+GPT4o | Whisper+Sauerkraut | Whisper+gemma4 |
|---|---|---|---|---|---|
| **OriginalDC** | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✓ |
| **OriginalDC+Noise** | A:✓ P:✓ | A:✓ P:✓ STT | A:✓ P:✓ | A:✓ P:✗ LLM | A:✓ P:✓ |
| **LapInMitte** | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✗ LLM | A:✓ P:✓ |
| **LapBeiArzt** | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✓ |
| **Selbstkorrekturen** | A:✓ P:✗ STT | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✓ STT | A:✓ P:✗ STT |
| **Unterbrechungen** | A:✓ P:✗ STT | A:✓ P:✓ | A:✓ P:✓ STT | A:✓ P:✗ STT | A:✓ P:✓ STT |
| **Gedankensprünge** | A:✓ P:✗ STT| A:✓ P:✓ STT | A:✓ P:✓ STT | A:✓ P:✗ STT | A:✓ P:✗ STT |
| **Meinungswechsel** | A:✓ P:✓ | A:✓ P:✓ STT | A:✓ P:✓ | A:✓ P:✗ LLM | A:✓ P:✓  |
| **Chaos** | A:✓ P:✗ LLM | A:✓ P:✓ | A:✓ P:✓ | A:✓ P:✗ LLM | A:✓ P:✓ |
| **Anamnesegespräch** | A:✗ P:✗ LLM | A:✗ P:✓ LLM  | A:✗ P:✓ LLM  | A:✗ P:✗ LLM | A:✗ P:✗ LLM |
| **PWC** | A:✗ P:✗ STT, LLM | A:✗ P:✗ STT, LLM | A:✗ P:✗ LLM | A:✗ P:✗ STT, LLM | A:✗ P:✗ STT, LLM |

## Statistiken

| Kombination | A✓ | A✗ | P✓ | P✗ | STT | LLM | 
|---|---|---|---|---|---|---|
| Whisper+llama3.2 | 9 | 2 | 5 | 6 | 4 | 3 |
| Speechmatics+GPT4o | 9 | 2 | 10 | 1 | 4 | 2 |
| AssemblyAI+GPT4o | 9 | 2 | 10 | 1 | 2 | 2 |
| Whisper+Sauerkraut | 9 | 2 | 3 | 8 | 4 | 6 |
| Whisper+gemma4 | 9 | 2 | 7 | 4 | 4 | 2 |

> A✗ immer 2 (Anamnesegespräch + PWC) weil dort Arzt einen Eigennamen haben sollte, nicht "Arzt".

---

## Details pro Modell

### Whisper+llama3.2

| Szenario | Arzt-Label | Patienten-Label | Anmerkungen |
|----------|-----------|----------------|-------------|
| OriginalDC | `Arzt` | `Frau Weber` | Formatted fehler am Anfang |
| OriginalDC+Noise | `Arzt` | `Frau Weber` | LLM name Swap, Arzt -> Herr Doktor |
| LapInMitte | `Arzt` | `Frau Weber` | — |
| LapBeiArzt | `Arzt` | `Frau Weber` | Formatted fehler am Anfang |
| Selbstkorrekturen | `Arzt` | *(fehlt)* | STT: 1 Speaker am anfang erkannt, danach nie wieder |
| Unterbrechungen | `Arzt` | *(fehlt)* | STT: 1 Speaker am anfang erkannt, danach nie wieder |
| Gedankensprünge | `Arzt` | `Herr Hielmanns` | STT: 2 speaker am anfang, jeweils einmal erkannt, danach nie wieder; falsch erkannt (GT: Herr Yilmaz) |
| Meinungswechsel | `Arzt` | `Frau Hoffmann` | — |
| Chaos | `Arzt` | `Patient` | Patient generisch (sollte Herr Schuster sein) |
| Anamnesegespräch | *(fehlt)* | *(fehlt)* | STT: Speaker 0/1 erkannt, llm hats ignoriert |
| PWC | `Arzt` | `Patient(in)` | STT: 2 Speaker erkannt, jedoch nicht korrekt/konsistent. LLM: Patient/Arzt statt Namen |

### Speechmatics+GPT4o

| Szenario | Arzt-Label | Patienten-Label | Anmerkungen |
|----------|-----------|----------------|-------------|
| OriginalDC | `Arzt` | `Frau Weber` | — |
| OriginalDC+Noise | `Arzt` | `Frau Weber` | STT: 1 Speaker am anfang erkannt, danach nie wieder. LLM: Dennoch 2 speaker durchgehend korrekt gelabled |
| LapInMitte | `Arzt` | `Frau Weber` | — |
| LapBeiArzt | `Arzt` | `Frau Weber` | — |
| Selbstkorrekturen | `Arzt` | `Herr Berger` | — |
| Unterbrechungen | `Arzt` | `Frau Klein` | — |
| Gedankensprünge | `Arzt` | `Yilmaz` | STT: 2 speaker erkannt, nicht durchgehend korrekt. LLM: Dennoch 2 speaker durchgehend korrekt gelabled |
| Meinungswechsel | `Arzt` | `Frau Hoffmann` | STT: 1 Speaker am anfang erkannt, danach nie wieder. LLM: Dennoch 2 speaker durchgehend korrekt gelabled |
| Chaos | `Arzt` | `Herr Schuster` | — |
| Anamnesegespräch | `Arzt` | `Julia Becker-Westphalen` | Arzt statt Nina Colette |
| PWC | `Arzt` | `Patientin` | Arzt/Patientin statt Namen; STT nicht durchgehend korrekt erkannt |

### AssemblyAI+GPT4o

| Szenario | Arzt-Label | Patienten-Label | Anmerkungen |
|----------|-----------|----------------|-------------|
| OriginalDC | `Arzt` | `Frau Weber` | — |
| OriginalDC+Noise | `Arzt` | `Frau Weber` | — |
| LapInMitte | `Arzt` | `Frau Weber` | — |
| LapBeiArzt | `Arzt` | `Frau Weber` | — |
| Selbstkorrekturen | `Arzt` | `Herr Berger` | — |
| Unterbrechungen | `Arzt` | `Frau Klein` | STT: 1 Speaker am anfang erkannt, danach nie wieder. LLM: Dennoch 2 speaker durchgehend korrekt gelabled |
| Gedankensprünge | `Arzt` | `Herr Yilmaz` | STT: speaker am ende falsch erkannt |
| Meinungswechsel | `Arzt` | `Frau Hoffmann` | — |
| Chaos | `Arzt` | `Herr Schuster` | — |
| Anamnesegespräch | `Arzt` | `Julia Becken-Westphalen` | Arzt statt Nina Colette |
| PWC | `Arzt` | `Patientin` | Arzt/Patientin statt Namen |

### Whisper+Sauerkraut

| Szenario | Arzt-Label | Patienten-Label | Anmerkungen |
|----------|-----------|----------------|-------------|
| OriginalDC | `Arzt` | `Frau Weber` | — |
| OriginalDC+Noise | `Arzt` | `Patient(in)` | LLM: Patientin generisch (sollte Frau Weber sein) |
| LapInMitte | `Arzt` | `Patientin` | LLM: Patientin generisch (sollte Frau Weber sein) |
| LapBeiArzt | `Arzt` | `Patientin (Frau Weber)` | — |
| Selbstkorrekturen | `Arzt` | `Herr Berger` | STT: 1 Speaker am anfang erkannt, danach nie wieder. LLM: Dennoch 2 speaker teilweise korrekt gelabled |
| Unterbrechungen | `Arzt` | `[Name des Patienten]` |STT: 1 Speaker am anfang erkannt, danach nie wieder. LLM: Dennoch 2 speaker teilweise korrekt gelabled; LLM: Patientin generisch (sollte Frau Klein sein)|
| Gedankensprünge | `Arzt` | `Herr Hielmanns` | STT: 1 Speaker am anfang erkannt, danach nie wieder; falsch erkannt (GT: Herr Yilmaz) |
| Meinungswechsel | `Arzt` | `Patient` | LLM: Patientin generisch (sollte Frau Hoffmann sein)  |
| Chaos | `Arzt` | `[Name des Patienten]` | Formatted fehler am Anfang; LLM: Patientin generisch (sollte Herr Schuster sein)|
| Anamnesegespräch | *(fehlt)* | *(fehlt)* | STT: Speaker 0/1 erkannt, LLM hat einfach zusammenfassung gemacht |
| PWC | `Arzt` | *(fehlt)* | STT: 1 Speaker am anfang erkannt, danach nie wieder; Formatted fehler am Anfang |

### Whisper+gemma4

| Szenario | Arzt-Label | Patienten-Label | Anmerkungen |
|----------|-----------|----------------|-------------|
| OriginalDC | `Arzt` | `Frau Weber` | — |
| OriginalDC+Noise | `Arzt` | `Frau Weber` | — |
| LapInMitte | `Arzt` | `Frau Weber` | — |
| LapBeiArzt | `Arzt` | `Frau Weber` | — |
| Selbstkorrekturen | `Arzt` | *(fehlt)* | STT: 1 Speaker am anfang erkannt, danach nie wieder |
| Unterbrechungen | `Arzt` | `Frau Klein` |  STT: 1 Speaker am anfang erkannt, danach nie wieder. LLM: Dennoch 2 speaker teilweise korrekt gelabled |
| Gedankensprünge | `Arzt` | `Hielmanns` | STT: 3 Speaker erkannt, LLM korrekt 2 davon als Arzt identifiziert |
| Meinungswechsel | `Arzt` | `[Frau Hoffmann]` | STT: 2 speaker erkannt, aber nicht durchgehend korrekt |
| Chaos | `Arzt` | `[Herr Schuster]` | — |
| Anamnesegespräch | `Arzt` | `Patient` | Arzt statt Nina Colette; LLM: Patientin generisch (sollte Julia Becken-Westphalen sein) |
| PWC | `Arzt` | *(fehlt)* | STT: 1 Speaker am anfang erkannt, danach nie wieder |
