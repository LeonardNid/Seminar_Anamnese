# Evaluation Report: AI Medical Documentation Pipeline

**Dateien:** `history.json` · `docs/Seminar Texte.md`  
**Einträge:** 30 (10 Audiodateien × 3 Modellkombinationen)

**Modelle:**

| Kürzel | STT | LLM |
|--------|-----|-----|
| Whisper+Sauerkraut | Whisper Large-v3-turbo (Lokal) | Llama-3.1-SauerkrautLM-8b-Instruct |
| Whisper+llama3.2 | Whisper Large-v3-turbo (Lokal) | llama3.2 |
| Speechmatics+GPT-4o | Speechmatics (Cloud) | OpenAI GPT-4o |

---

## 1. Übersicht: WER-Scores (STT vs. Original)

> WER = Word Error Rate. Niedriger ist besser. Nur für Dateien mit Ground-Truth (9 von 10). "Das Anamnesegespräch" hat keinen Referenztext.

| Audiodatei | Szenario | Whisper+Sauerkraut | Whisper+llama3.2 | Speechmatics+GPT-4o |
|-----------|----------|-------------------|-----------------|---------------------|
| Original (Laptop Mitte) | Original | 13.4% | 13.4% | 14.3% |
| Original (Laptop Arzt) | Original | 12.9% | 12.9% | 9.8% |
| Original (DC-Mikro) | Original | 12.5% | 12.5% | 12.5% |
| Original (DC + Rauschen) | Original | 45.5% | 45.5% | 33.5% |
| Selbstkorrekturen | Selbstkorrekturen | — | — | — |
| Unterbrechungen | Unterbrechungen | 19.7% | 19.7% | 25.2% |
| Gedankensprünge | Gedankensprünge | — | — | — |
| Meinungswechsel | Meinungswechsel | — | — | — |
| Chaos | Chaos | 22.3% | 22.3% | 26.4% |
| Anamnesegespräch (ext.) | — | — | — | — |

---

## 2. STT-Qualität: Original vs. Raw

### Original (Laptop Mitte)

**Metriken:**

| Modell | WER | Ins | Del | Sub | Char-Sim. |
|--------|-----|-----|-----|-----|-----------|
| Whisper+Sauerkraut | 13.4% | 5 | 1 | 24 | 93% |
| Whisper+llama3.2 | 13.4% | 5 | 1 | 24 | 93% |
| Speechmatics+GPT-4o | 14.3% | 9 | 3 | 20 | 92% |

**Beispielfehler (STT vs. Ground Truth):**

*Whisper+Sauerkraut:*
- 'morgen,' → 'morgen'
- 'weber' → 'weber,'
- 'führt sie' → 'fühlt ihr'
- 'doktor' → 'doktor,'
- 'notiz für akte' → 'notiz,'
- 'okay notiz' → 'okay, notiz,'
- 'da' → 'das'
- 'das' → 'es'

*Whisper+llama3.2:*
- 'morgen,' → 'morgen'
- 'weber' → 'weber,'
- 'führt sie' → 'fühlt ihr'
- 'doktor' → 'doktor,'
- 'notiz für akte' → 'notiz,'
- 'okay notiz' → 'okay, notiz,'
- 'da' → 'das'
- 'das' → 'es'

*Speechmatics+GPT-4o:*
- 'hallo' → 'hallo,'
- [fehlt] 'für akte'
- 'tagen,' → 'tagen'
- 'ja,' → 'ja'
- 'temperaturen,' → 'temperaturen'
- 'da schleim mit hoch' → 'das schleimetuch'
- 'das' → 'es'
- [extra] 'so, das ist'

### Original (Laptop Arzt)

**Metriken:**

| Modell | WER | Ins | Del | Sub | Char-Sim. |
|--------|-----|-----|-----|-----|-----------|
| Whisper+Sauerkraut | 12.9% | 1 | 3 | 25 | 92% |
| Whisper+llama3.2 | 12.9% | 1 | 3 | 25 | 92% |
| Speechmatics+GPT-4o | 9.8% | 1 | 3 | 18 | 91% |

**Beispielfehler (STT vs. Ground Truth):**

*Whisper+Sauerkraut:*
- 'hallo' → 'hallo,'
- 'huste' → 'fühlste'
- [fehlt] 'für akte'
- 'tagen,' → 'tagen'
- 'waren es' → 'war und ist'
- 'temperaturen,' → 'temperaturen'
- 'gelblichgrün,' → 'geltlich grün,'
- 'direkt,' → 'direkt'

*Whisper+llama3.2:*
- 'hallo' → 'hallo,'
- 'huste' → 'fühlste'
- [fehlt] 'für akte'
- 'tagen,' → 'tagen'
- 'waren es' → 'war und ist'
- 'temperaturen,' → 'temperaturen'
- 'gelblichgrün,' → 'geltlich grün,'
- 'direkt,' → 'direkt'

*Speechmatics+GPT-4o:*
- 'hallo' → 'hallo,'
- 'huste' → 'wusste'
- [fehlt] 'für akte'
- 'tagen,' → 'tagen'
- 'ja,' → 'ja'
- 'temperaturen,' → 'temperaturen und'
- 'eher gelblichgrün,' → 'gelblich grün,'
- 'direkt,' → 'direkt'

### Original (DC-Mikro)

**Metriken:**

| Modell | WER | Ins | Del | Sub | Char-Sim. |
|--------|-----|-----|-----|-----|-----------|
| Whisper+Sauerkraut | 12.5% | 4 | 0 | 24 | 81% |
| Whisper+llama3.2 | 12.5% | 4 | 0 | 24 | 81% |
| Speechmatics+GPT-4o | 12.5% | 10 | 2 | 16 | 91% |

**Beispielfehler (STT vs. Ground Truth):**

*Whisper+Sauerkraut:*
- 'hallo' → 'hallo,'
- 'für akte leitsymptom' → 'leitsymptom,'
- [extra] 'ungefähr'
- 'subfebrile' → 'subfibrile'
- 'febrile temperaturen,' → 'fibrile temperaturen'
- 'das' → 'es'
- 'schlechter' → 'schlechte'
- 'direkt,' → 'direkt'

*Whisper+llama3.2:*
- 'hallo' → 'hallo,'
- 'für akte leitsymptom' → 'leitsymptom,'
- [extra] 'ungefähr'
- 'subfebrile' → 'subfibrile'
- 'febrile temperaturen,' → 'fibrile temperaturen'
- 'das' → 'es'
- 'schlechter' → 'schlechte'
- 'direkt,' → 'direkt'

*Speechmatics+GPT-4o:*
- 'hallo' → 'hallo,'
- [fehlt] 'für akte'
- 'tagen,' → 'tagen'
- 'ja,' → 'ja'
- [extra] 'ungefähr'
- 'temperaturen,' → 'temperaturen'
- 'das' → 'es'
- 'gelblichgrün,' → 'gelblich grün,'

### Original (DC + Rauschen)

**Metriken:**

| Modell | WER | Ins | Del | Sub | Char-Sim. |
|--------|-----|-----|-----|-----|-----------|
| Whisper+Sauerkraut | 45.5% | 4 | 2 | 96 | 13% |
| Whisper+llama3.2 | 45.5% | 4 | 2 | 96 | 13% |
| Speechmatics+GPT-4o | 33.5% | 1 | 11 | 63 | 29% |

**Beispielfehler (STT vs. Ground Truth):**

*Whisper+Sauerkraut:*
- 'hallo' → 'hallo,'
- 'huste seit knapp' → 'bin heute in'
- 'ununterbrochen und mir' → 'unterbrochen es'
- 'für akte leitsymptom husten' → 'leitsymptom, husten,'
- 'rezidivierendes fieber' → 'präzidivierende tiber'
- [fehlt] 'fieber'
- 'waren' → 'war'
- '38,8 grad auf dem thermometer' → 'nachgekreist ich war gerade in der karte'

*Whisper+llama3.2:*
- 'hallo' → 'hallo,'
- 'huste seit knapp' → 'bin heute in'
- 'ununterbrochen und mir' → 'unterbrochen es'
- 'für akte leitsymptom husten' → 'leitsymptom, husten,'
- 'rezidivierendes fieber' → 'präzidivierende tiber'
- [fehlt] 'fieber'
- 'waren' → 'war'
- '38,8 grad auf dem thermometer' → 'nachgekreist ich war gerade in der karte'

*Speechmatics+GPT-4o:*
- 'hallo' → 'hallo,'
- 'huste seit knapp einer woche ununterbrochen und mir ist' → 'habe keine lust'
- [fehlt] 'für akte'
- 'tagen,' → 'tagen'
- 'fieber denn' → 'schon'
- 'ja,' → 'ja'
- '38,8' → '38,5'
- [fehlt] 'auf dem thermometer'

### Selbstkorrekturen

*Kein Ground-Truth-Text vorhanden — kein WER-Vergleich möglich.*

**Whisper+Sauerkraut:** 1303 Zeichen im Raw-Output  
**Whisper+llama3.2:** 1303 Zeichen im Raw-Output  
**Speechmatics+GPT-4o:** 1265 Zeichen im Raw-Output  

### Unterbrechungen

**Metriken:**

| Modell | WER | Ins | Del | Sub | Char-Sim. |
|--------|-----|-----|-----|-----|-----------|
| Whisper+Sauerkraut | 19.7% | 4 | 9 | 16 | 67% |
| Whisper+llama3.2 | 19.7% | 4 | 9 | 16 | 67% |
| Speechmatics+GPT-4o | 25.2% | 5 | 12 | 20 | 44% |

**Beispielfehler (STT vs. Ground Truth):**

*Whisper+Sauerkraut:*
- [extra] 'dass'
- [fehlt] 'haben'
- [extra] 'haben'
- 'drei' → '3'
- [fehlt] 'wo genau krampft es denn'
- 'dem' → 'den'
- 'strahl' → 'strahlt'
- 'weh,' → 'weh'

*Whisper+llama3.2:*
- [extra] 'dass'
- [fehlt] 'haben'
- [extra] 'haben'
- 'drei' → '3'
- [fehlt] 'wo genau krampft es denn'
- 'dem' → 'den'
- 'strahl' → 'strahlt'
- 'weh,' → 'weh'

*Speechmatics+GPT-4o:*
- [extra] 'dass'
- [fehlt] 'haben'
- [extra] 'haben'
- 'drei uhr' → '300'
- [fehlt] 'wo genau krampft es denn'
- 'dem' → 'den'
- 'strahl' → 'strahlt'
- [fehlt] 'es tut einfach'

### Gedankensprünge

*Kein Ground-Truth-Text vorhanden — kein WER-Vergleich möglich.*

**Whisper+Sauerkraut:** 1220 Zeichen im Raw-Output  
**Whisper+llama3.2:** 1220 Zeichen im Raw-Output  
**Speechmatics+GPT-4o:** 1251 Zeichen im Raw-Output  

### Meinungswechsel

*Kein Ground-Truth-Text vorhanden — kein WER-Vergleich möglich.*

**Whisper+Sauerkraut:** 1257 Zeichen im Raw-Output  
**Whisper+llama3.2:** 1257 Zeichen im Raw-Output  
**Speechmatics+GPT-4o:** 1133 Zeichen im Raw-Output  

### Chaos

**Metriken:**

| Modell | WER | Ins | Del | Sub | Char-Sim. |
|--------|-----|-----|-----|-----|-----------|
| Whisper+Sauerkraut | 22.3% | 4 | 12 | 43 | 63% |
| Whisper+llama3.2 | 22.3% | 4 | 12 | 43 | 63% |
| Speechmatics+GPT-4o | 26.4% | 8 | 13 | 49 | 36% |

**Beispielfehler (STT vs. Ground Truth):**

*Whisper+Sauerkraut:*
- [fehlt] 'ähm seit'
- 'nee, moment,' → 'ehm, moment'
- 'ach quatsch, ich meine das rechte' → 'ach, quatsch mein rechtes'
- 'notiz schwankschwindel' → 'notiz, schwankschwindeln'
- [extra] 'auch'
- 'genau so' → 'die'
- [fehlt] 'das nämlich bei meiner cousine auch angefangen die hatte'
- 'kopf,' → 'kopf'

*Whisper+llama3.2:*
- [fehlt] 'ähm seit'
- 'nee, moment,' → 'ehm, moment'
- 'ach quatsch, ich meine das rechte' → 'ach, quatsch mein rechtes'
- 'notiz schwankschwindel' → 'notiz, schwankschwindeln'
- [extra] 'auch'
- 'genau so' → 'die'
- [fehlt] 'das nämlich bei meiner cousine auch angefangen die hatte'
- 'kopf,' → 'kopf'

*Speechmatics+GPT-4o:*
- 'ja,' → 'ja'
- [fehlt] 'ähm'
- 'nee, moment,' → 'im moment'
- 'im' → 'ein'
- 'ach' → 'ach,'
- 'ich meine das rechte ohr,' → 'mein rechtes ohr'
- [extra] 'auch'
- 'genau so' → 'die'

### Anamnesegespräch (ext.)

*Kein Ground-Truth-Text vorhanden — kein WER-Vergleich möglich.*

**Whisper+Sauerkraut:** 14258 Zeichen im Raw-Output  
**Whisper+llama3.2:** 14258 Zeichen im Raw-Output  
**Speechmatics+GPT-4o:** 15643 Zeichen im Raw-Output  

---

## 3. Formatierung: Raw vs. Formatted

> Prüft ob Speaker-Labels korrekt ersetzt wurden und wie viel Text verändert wurde.

| Audiodatei | Modell | Speaker ersetzt | Arzt-Label | Text-Sim. | Wortdiff |
|-----------|--------|----------------|-----------|-----------|----------|
| Original (Laptop Mitte) | Whisper+Sauerkraut | ✓ | ✓ | 100% | +0 |
| Original (Laptop Mitte) | Whisper+llama3.2 | ✓ | ✓ | 100% | +0 |
| Original (Laptop Mitte) | Speechmatics+GPT-4o | ✓ | ✓ | 99% | +0 |
| Original (Laptop Arzt) | Whisper+Sauerkraut | ✓ | ✓ | 96% | +16 |
| Original (Laptop Arzt) | Whisper+llama3.2 | ✓ | ✓ | 84% | +4 |
| Original (Laptop Arzt) | Speechmatics+GPT-4o | ✓ | ✓ | 99% | +0 |
| Original (DC-Mikro) | Whisper+Sauerkraut | ✓ | ✓ | 97% | +9 |
| Original (DC-Mikro) | Whisper+llama3.2 | ✗ | ✓ | 91% | +3 |
| Original (DC-Mikro) | Speechmatics+GPT-4o | ✓ | ✓ | 99% | +0 |
| Original (DC + Rauschen) | Whisper+Sauerkraut | ✓ | ✓ | 98% | +5 |
| Original (DC + Rauschen) | Whisper+llama3.2 | ✓ | ✓ | 90% | +2 |
| Original (DC + Rauschen) | Speechmatics+GPT-4o | ✓ | ✓ | 99% | +0 |
| Selbstkorrekturen | Whisper+Sauerkraut | ✓ | ✓ | 83% | +67 |
| Selbstkorrekturen | Whisper+llama3.2 | ✓ | ✓ | 97% | -13 |
| Selbstkorrekturen | Speechmatics+GPT-4o | ✓ | ✓ | 63% | +0 |
| Unterbrechungen | Whisper+Sauerkraut | ✓ | ✓ | 57% | +16 |
| Unterbrechungen | Whisper+llama3.2 | ✓ | ✓ | 100% | +0 |
| Unterbrechungen | Speechmatics+GPT-4o | ✓ | ✓ | 97% | +0 |
| Gedankensprünge | Whisper+Sauerkraut | ✓ | ✓ | 100% | +0 |
| Gedankensprünge | Whisper+llama3.2 | ✓ | ✓ | 98% | -10 |
| Gedankensprünge | Speechmatics+GPT-4o | ✓ | ✓ | 98% | +4 |
| Meinungswechsel | Whisper+Sauerkraut | ✓ | ✓ | 98% | +0 |
| Meinungswechsel | Whisper+llama3.2 | ✓ | ✓ | 99% | +0 |
| Meinungswechsel | Speechmatics+GPT-4o | ✓ | ✓ | 99% | +0 |
| Chaos | Whisper+Sauerkraut | ✓ | ✓ | 89% | +60 |
| Chaos | Whisper+llama3.2 | ✓ | ✓ | 100% | +0 |
| Chaos | Speechmatics+GPT-4o | ✓ | ✓ | 99% | +0 |
| Anamnesegespräch (ext.) | Whisper+Sauerkraut | ✓ | ✗ | 1% | -2100 |
| Anamnesegespräch (ext.) | Whisper+llama3.2 | ✓ | ✗ | 2% | -2075 |
| Anamnesegespräch (ext.) | Speechmatics+GPT-4o | ✓ | ✓ | 91% | +147 |

**Auffälligkeiten:**

- **Original (Laptop Arzt) / Whisper+llama3.2:** Geringe Text-Ähnlichkeit (84%)
- **Original (DC-Mikro) / Whisper+llama3.2:** Speaker-Labels nicht vollständig ersetzt
- **Selbstkorrekturen / Whisper+Sauerkraut:** Geringe Text-Ähnlichkeit (83%); Großer Wortunterschied (+67 Wörter)
- **Selbstkorrekturen / Speechmatics+GPT-4o:** Geringe Text-Ähnlichkeit (63%)
- **Unterbrechungen / Whisper+Sauerkraut:** Geringe Text-Ähnlichkeit (57%)
- **Chaos / Whisper+Sauerkraut:** Großer Wortunterschied (+60 Wörter)
- **Anamnesegespräch (ext.) / Whisper+Sauerkraut:** Kein 'Arzt:'-Label im Formatted; Geringe Text-Ähnlichkeit (1%); Großer Wortunterschied (-2100 Wörter)
- **Anamnesegespräch (ext.) / Whisper+llama3.2:** Kein 'Arzt:'-Label im Formatted; Geringe Text-Ähnlichkeit (2%); Großer Wortunterschied (-2075 Wörter)
- **Anamnesegespräch (ext.) / Speechmatics+GPT-4o:** Großer Wortunterschied (+147 Wörter)

---

## 4. SOAP-Überprüfung

| Audiodatei | Modell | S | O | A | P | Struktur | Schlüsselbegriffe | Halluz.? |
|-----------|--------|---|---|---|---|----------|------------------|---------|
| Original (Laptop Mitte) | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 5/7 | Ja |
| Original (Laptop Mitte) | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 4/7 | Nein |
| Original (Laptop Mitte) | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 6/7 | Nein |
| Original (Laptop Arzt) | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 4/7 | Nein |
| Original (Laptop Arzt) | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/7 | Nein |
| Original (Laptop Arzt) | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 6/7 | Nein |
| Original (DC-Mikro) | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 5/7 | Nein |
| Original (DC-Mikro) | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/7 | Ja |
| Original (DC-Mikro) | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 6/7 | Nein |
| Original (DC + Rauschen) | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 3/7 | Nein |
| Original (DC + Rauschen) | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 3/7 | Nein |
| Original (DC + Rauschen) | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 5/7 | Nein |
| Selbstkorrekturen | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 6/7 | Nein |
| Selbstkorrekturen | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 6/7 | Nein |
| Selbstkorrekturen | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 | Nein |
| Unterbrechungen | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 2/5 | Nein |
| Unterbrechungen | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 | Nein |
| Unterbrechungen | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 | Ja |
| Gedankensprünge | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | Nein |
| Gedankensprünge | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | Nein |
| Gedankensprünge | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | Nein |
| Meinungswechsel | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | Nein |
| Meinungswechsel | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | Nein |
| Meinungswechsel | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | Nein |
| Chaos | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 | Nein |
| Chaos | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 | Nein |
| Chaos | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 | Nein |
| Anamnesegespräch (ext.) | Whisper+Sauerkraut | ✓ | ✓ | ✓ | ✓ | ✓ | — | Nein |
| Anamnesegespräch (ext.) | Whisper+llama3.2 | ✓ | ✓ | ✓ | ✓ | ✓ | — | Ja |
| Anamnesegespräch (ext.) | Speechmatics+GPT-4o | ✓ | ✓ | ✓ | ✓ | ✓ | — | Ja |

**SOAP-Details:**

- **Original (Laptop Mitte) / Whisper+Sauerkraut:** Schlüsselbegriffe nicht erwähnt: kamillentee, drogerie; Zahlen im SOAP nicht im Transkript: 5
- **Original (Laptop Mitte) / Whisper+llama3.2:** Schlüsselbegriffe nicht erwähnt: fieber, kamillentee, drogerie
- **Original (Laptop Mitte) / Speechmatics+GPT-4o:** Schlüsselbegriffe nicht erwähnt: drogerie
- **Original (Laptop Arzt) / Whisper+Sauerkraut:** Schlüsselbegriffe nicht erwähnt: fieber, kamillentee, drogerie
- **Original (Laptop Arzt) / Whisper+llama3.2:** Schlüsselbegriffe nicht erwähnt: kamillentee, drogerie
- **Original (Laptop Arzt) / Speechmatics+GPT-4o:** Schlüsselbegriffe nicht erwähnt: drogerie
- **Original (DC-Mikro) / Whisper+Sauerkraut:** Schlüsselbegriffe nicht erwähnt: kamillentee, drogerie
- **Original (DC-Mikro) / Whisper+llama3.2:** Schlüsselbegriffe nicht erwähnt: kamillentee, drogerie; Zahlen im SOAP nicht im Transkript: 7
- **Original (DC-Mikro) / Speechmatics+GPT-4o:** Schlüsselbegriffe nicht erwähnt: drogerie
- **Original (DC + Rauschen) / Whisper+Sauerkraut:** Schlüsselbegriffe nicht erwähnt: fieber, dyspnoe, kamillentee, drogerie
- **Original (DC + Rauschen) / Whisper+llama3.2:** Schlüsselbegriffe nicht erwähnt: fieber, dyspnoe, kamillentee, drogerie
- **Original (DC + Rauschen) / Speechmatics+GPT-4o:** Schlüsselbegriffe nicht erwähnt: kamillentee, drogerie
- **Selbstkorrekturen / Whisper+Sauerkraut:** Schlüsselbegriffe nicht erwähnt: lichtempfindlich
- **Selbstkorrekturen / Whisper+llama3.2:** Schlüsselbegriffe nicht erwähnt: lichtempfindlich
- **Unterbrechungen / Whisper+Sauerkraut:** Schlüsselbegriffe nicht erwähnt: erbrochen, hähnchen, bauchnabel
- **Unterbrechungen / Speechmatics+GPT-4o:** Zahlen im SOAP nicht im Transkript: 24
- **Gedankensprünge / Whisper+Sauerkraut:** Schlüsselbegriffe nicht erwähnt: knackt
- **Gedankensprünge / Whisper+llama3.2:** Schlüsselbegriffe nicht erwähnt: knackt
- **Gedankensprünge / Speechmatics+GPT-4o:** Schlüsselbegriffe nicht erwähnt: knackt
- **Meinungswechsel / Whisper+Sauerkraut:** Schlüsselbegriffe nicht erwähnt: angina
- **Meinungswechsel / Whisper+llama3.2:** Schlüsselbegriffe nicht erwähnt: angina
- **Meinungswechsel / Speechmatics+GPT-4o:** Schlüsselbegriffe nicht erwähnt: sodbrennen
- **Anamnesegespräch (ext.) / Whisper+llama3.2:** Zahlen im SOAP nicht im Transkript: 25
- **Anamnesegespräch (ext.) / Speechmatics+GPT-4o:** Zahlen im SOAP nicht im Transkript: 10, 170, 7, 8

---

## 5. Modellvergleich

### 5.1 STT-Qualität: Whisper vs. Speechmatics

> Ø WER über alle Dateien mit Ground-Truth

| Modell | Ø WER | Beste Datei | Schlechteste Datei |
|--------|-------|-------------|-------------------|
| Whisper+Sauerkraut | 21.1% | Original (DC-Mikro) (12.5%) | Original (DC + Rauschen) (45.5%) |
| Whisper+llama3.2 | 21.1% | Original (DC-Mikro) (12.5%) | Original (DC + Rauschen) (45.5%) |
| Speechmatics+GPT-4o | 20.3% | Original (Laptop Arzt) (9.8%) | Original (DC + Rauschen) (33.5%) |

### 5.2 Formatierungsqualität

| Modell | Ø Text-Sim. | Speaker-Fehler | Wortdiff > 20 |
|--------|-------------|---------------|--------------|
| Whisper+Sauerkraut | 82% | 0/10 | 3/10 |
| Whisper+llama3.2 | 86% | 1/10 | 1/10 |
| Speechmatics+GPT-4o | 94% | 0/10 | 1/10 |

### 5.3 SOAP-Qualität

| Modell | Struktur vollst. | Ø Schlüsselbegriff-Abdeckung | Halluz.-Fälle |
|--------|-----------------|------------------------------|--------------|
| Whisper+Sauerkraut | 10/10 | 70% | 1/10 |
| Whisper+llama3.2 | 10/10 | 76% | 2/10 |
| Speechmatics+GPT-4o | 10/10 | 88% | 2/10 |

### 5.4 Gesamtfazit

| Dimension | Bestes Modell | Schwächstes Modell |
|-----------|--------------|-------------------|
| STT (WER) | Speechmatics+GPT-4o (20.3%) | Whisper+Sauerkraut (21.1%) |
| Formatierung (Text-Sim.) | Speechmatics+GPT-4o (94%) | Whisper+Sauerkraut (82%) |
| SOAP (Schlüsselbegriffe) | Speechmatics+GPT-4o (88%) | Whisper+Sauerkraut (70%) |

**Stärken und Schwächen:**

- **Speechmatics+GPT-4o**: Stärkste Formatierung und SOAP-Qualität durch GPT-4o; Speechmatics-STT produziert manchmal fehlende Diarizerungsinfo (nur ein Speaker-Block).
- **Whisper+Sauerkraut**: Gute STT-Qualität durch Whisper-Diarisierung (SPEAKER_00/01); SauerkrautLM kompakter in SOAP, aber strukturell oft vollständig.
- **Whisper+llama3.2**: Gleiche STT-Qualität wie Sauerkraut; llama3.2 neigt zu mehr ausschweifenden SOAP-Texten und gelegentlichen inhaltlichen Abweichungen.
