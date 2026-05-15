# Präsentationsplan — AI-Anamnesis of Patients (Master-Seminar mit PwC)

## Context

Diese Präsentation schließt das Master-Seminar bei PwC ab (Anton Reinhard & Leonard Niedens, Thema „AI-Anamnesis of Patients", betreut von Nicolas Sell / Torben Sieg / Daniel Fauland). Eure Aufgabe war es, einen End-to-End-Workflow zu evaluieren, der Arzt-Patienten-Gespräche aufnimmt, transkribiert und in strukturierte SOAP-Notizen umwandelt. Ihr habt **5 Modell-Kombinationen × 11 Szenarien = 55 Runs** durchgeführt und entlang dreier Achsen vermessen (WER, LLM-Halluzinationsrate, SOAP-Qualität durch Claude-as-Judge).

**Rahmen:** 20 min Vortragszeit · max 5 min Demo · Folien auf **Englisch** · gemischtes Publikum (tech-affine PwC Cloud & AI + Prof) · **Burning Platform**-Storyline (Schmerz → Lösung) · Demo **nach** Pipeline-Erklärung und **vor** den Ergebnissen.

**Foliengestaltungs-Prinzipien (durchgängig anzuwenden — aus den PwC-Präsentationstechniken):**
- **Minimalismus**: Wenig Text, eine Kernaussage pro Folie. Wenn nicht in einem Satz sagbar → vereinfachen.
- **Dunkler Hintergrund** bevorzugt (oder PwC-Look: white mit orangen Akzenten — passend zum Lehr-Handout)
- **Max. 2 Schriftarten**, klare Hierarchie (groß → klein), keine Bullet-Wüsten
- **2-3 Farben max.**: empfohlen Anthrazit/Schwarz + ein Akzent (Orange #D04A02 wie PwC, oder Medizin-Blau)
- **Bilder/Icons unterstützen die Message**, sind nicht dekorativ
- **Animationen** nur als Stütze der Argumentation (Elemente einblenden, nicht als Spielerei)
- **Fazit oft als Folientitel** (nicht „Results: WER" sondern „Cloud STT cuts errors in half")
- Versucht **ein durchgängiges visuelles Motiv**: z.B. Stethoskop-Wellenform-Icon, das sich durch Übergänge zieht

**Sprech-Aufteilung 50/50:** Folien 1-8 = Sprecher A (Einleitung & Methodik) · Demo gemeinsam · Folien 9-16 = Sprecher B (Ergebnisse & Ausblick). **Grenze: Vor Folie 8 (Demo)** wechselt es zum zweiten Sprecher, der die Demo führt und in den Ergebnis-Teil übergeht.

---

## Foliensumme & Zeitbudget

**16 inhaltliche Folien + Titel + Q&A = 18 Folien · 20 min**

| Block | Folien | Zeit |
|---|---|---|
| Hook & Problem | 1-3 | ~2 min |
| Pipeline & Methodik | 4-7 | ~5 min |
| **Live-Demo** | 8 | **~4 min** |
| Ergebnisse | 9-12 | ~5 min |
| Limitationen & Datenschutz | 13-14 | ~2 min |
| Ausblick & Take-aways | 15-16 | ~1.5 min |
| Q&A | 17 | (außerhalb der 20 min) |

---

## Block 1: Hook & Burning Platform (Folien 1–3, ~2 min)

### Folie 1 — Title Slide

**Ziel:** Stimmung setzen, Thema kommunizieren ohne Vorlesen.

**Inhalt (minimal):**
- Title (large): *AI-Anamnesis of Patients*
- Subtitle: *Evaluating End-to-End STT + LLM Pipelines for Medical Documentation*
- Names: *Anton Reinhard · Leonard Niedens*
- Logos: FH Hannover + PwC (oben rechts klein)
- Datum (klein unten)

**Grafik (Hintergrund):**
Ganzflächiges, leicht abgedunkeltes Foto: **Arzt am Schreibtisch, vor Computer, im Hintergrund verschwommen ein wartender Patient**. Stimmungsbild — die Spannung „Dokumentation vs. menschliche Zuwendung". Über dem Foto eine 30%-Verdunklungs-Overlay-Schicht für Lesbarkeit. Titel oben links in weißer Serif-Schrift.

**Sprech-Text (was Sprecher A sagt):**
*„Stellt euch vor, ihr seid Patient. Ihr sitzt beim Arzt und erzählt von eurem Problem — und der Arzt schaut die ganze Zeit auf den Bildschirm und tippt."*

---

### Folie 2 — The Burning Platform: The 2-Hour Problem

**Ziel:** Schmerzpunkt etablieren. Publikum soll fühlen, dass es ein echtes Problem gibt.

**Inhalt (max. 3 Elemente):**
- Großzahl-Stilelement (analog zur PwC-Handout-Folie „3.000.000.000.000 $"):
  - **„2 hours"** in sehr großer Schrift (z.B. 200pt, oranger Akzent)
  - Untertitel: *per day spent by physicians on documentation*
- Quellenangabe klein unten: *„Annals of Internal Medicine, 2016 — for every hour of patient face-time, physicians spend ~2 hours on EHR documentation."*
- Eine zweite Statistik darunter, kleiner: *„49% of physicians report symptoms of burnout — administrative load is the #1 cited driver"* (Quelle: Medscape Physician Burnout Report)

**Grafik:** 
Nur Typografie — bewusst keine Grafik. **Die Zahl IST die Grafik.** Auf der rechten Folienhälfte optional ein Sanduhren-Icon (dünne Linien-Stil), das durchläuft.

**Sprech-Text:**
*„Zwei Stunden. Jeden Tag. Pro Arzt. Das ist mehr Zeit am Bildschirm als am Patienten. Und genau dort setzen wir an."*

---

### Folie 3 — The Question

**Ziel:** Die Forschungsfrage präzise formulieren, sodass alles Folgende einen klaren Zweck hat.

**Inhalt:**
- Zentrierte Frage (Serif, large):
  *„Can an AI listen to a doctor-patient conversation and produce usable medical documentation — completely automatically?"*
- Darunter, klein und in 3 Spalten als Kriterien:
  - **Accurate** — facts must match what was said
  - **Structured** — SOAP format (Subjective/Objective/Assessment/Plan)
  - **Trustworthy** — no fabricated diagnoses or treatments

**Grafik:**
Drei kleine Linien-Icons unter den Kriterien (z.B. Checkmark, Document-Outline, Shield). Bewusst kein zentrales Bild — die Frage soll wirken.

**Sprech-Text:**
*„Das ist die Frage, die uns durch dieses Seminar getragen hat. Drei Anforderungen: akkurat, strukturiert, vertrauenswürdig. Wir zeigen euch jetzt, was heute machbar ist — und wo die Grenzen liegen."*

---

## Block 2: Pipeline & Methodik (Folien 4–7, ~5 min)

### Folie 4 — Our Approach: A 3-Stage Pipeline

**Ziel:** Das mentale Modell aufbauen — Audio → Text → Structure. Eine einzige Grafik, die für die nächsten 15 min im Kopf bleibt.

**Inhalt:** Nur Titel (z.B. *„Three stages — one workflow"*) + zentrale Grafik. Kaum Text.

**Grafik (DETAILLIERT):**
Horizontaler Pipeline-Flow von links nach rechts, drei große Boxen mit Pfeilen dazwischen:

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   🎤  AUDIO     │ ──▶  │   📝  RAW TEXT   │ ──▶  │  📋  SOAP NOTE   │
│                 │      │                  │      │                  │
│  Doctor-Patient │      │  SPEAKER_00: ... │      │  S: ...          │
│  Conversation   │      │  SPEAKER_01: ... │      │  O: ...          │
└─────────────────┘      └──────────────────┘      │  A: ...          │
                                                    │  P: ...          │
        ▲                        ▲                  └──────────────────┘
        │                        │                          ▲
    ① STT Model              ② Format LLM              ③ SOAP LLM
    (Whisper / Speech-       (Replaces speakers          (Generates
     matics / AssemblyAI)     with Arzt / [Name])         structured note)
```

Jede Stage unten mit kleinem Modell-Logo-Reihe (Whisper-, Ollama-, OpenAI-Logo). Die drei Spalten in unterschiedlich abgestuftem Akzent-Orange (hell → mittel → dunkel), um den Fluss zu verdeutlichen.

**Animationen:**
1. Erst Box 1 erscheint (Audio-Welle animiert kurz)
2. Pfeil + Box 2 erscheinen → kurz darauf der Text-Inhalt darin
3. Pfeil + Box 3 → SOAP-Punkte erscheinen einzeln (S, O, A, P)

**Sprech-Text:**
*„Drei Stufen. Erstens: das Gespräch wird zu Text. Zweitens: ein LLM ersetzt 'SPEAKER_00' durch 'Arzt' und 'SPEAKER_01' durch den Patientennamen — sonst ändert es nichts. Drittens: aus diesem formatierten Transkript erzeugt ein LLM die SOAP-Notiz."*

---

### Folie 5 — Why These Models? Hardware Sets the Limit

**Ziel:** Begründen, warum genau diese 5 Kombinationen — und nicht 50. Hardware-Realismus zeigen.

**Inhalt:**
Titel: *„Hardware draws the line between what we can and cannot run locally"*

Zwei Spalten:

**Linke Spalte — „On-Prem Stack" (kann lokal laufen):**
- Whisper turbo (STT, ~800M params)
- SauerkrautLM 8B / Llama 3.2 / Gemma (LLM, 3-8B params)
- Hardware-Note: *„Tested on consumer GPU (24GB VRAM)"*

**Rechte Spalte — „What we'd love to run, but can't" (rot/grau durchgestrichen):**
- Whisper large-v3 (1.55B params)
- SauerkrautLM **70B** (~40GB+ VRAM)
- *„Would need EC2 g5.12xlarge → ~5€/h → not feasible for evaluation phase"*

Unten als horizontale Linie: **Cloud alternatives (Stage 1):** Speechmatics + AssemblyAI · **Cloud LLM (Stage 2/3):** GPT-4o

**Grafik:**
Zwei vertikal getrennte „Stacks" (gestapelte Boxen mit Modellname + Größe in Params). Auf der rechten Seite sind die Boxen halbtransparent und mit einem dünnen roten Kreuz markiert. Darunter eine zweite Reihe „Cloud Options" als horizontaler Streifen — visuell abgesetzt durch eine Wolken-Icon-Reihe.

**Sprech-Text:**
*„Die Modellauswahl ist nicht zufällig. Lokal sind wir hardwareseitig limitiert — wir hätten gerne Whisper large-v3 und ein 70B-LLM verwendet, aber das ist auf Consumer-Hardware nicht praktikabel. Deshalb haben wir zwei Welten verglichen: was läuft realistisch lokal — und was geht über die Cloud."*

---

### Folie 6 — Five Combinations · Eleven Scenarios

**Ziel:** Versuchsdesign auf einen Blick. Zeigen, dass das nicht ein Toy-Test war, sondern systematisch.

**Inhalt:**
Titel: *„5 × 11 = 55 controlled runs"*

**Grafik (Hauptelement) — die Test-Matrix:**

Eine 5×11-Heatmap-artige Matrix:
- **Y-Achse (5 Zeilen)** — Modell-Kombinationen mit jeweils Icon (Cloud/Local) + Modell-Namen-Kombi:
  1. ☁️ Speechmatics + GPT-4o
  2. ☁️ AssemblyAI + GPT-4o
  3. 🖥️ Whisper + SauerkrautLM 8B
  4. 🖥️ Whisper + Llama 3.2
  5. 🖥️ Whisper + Gemma
- **X-Achse (11 Spalten)** — Test-Szenarien mit Mini-Icon:
  - OriginalDC (clean baseline)
  - + White Noise (🔊)
  - Laptop-Mic Mid (🎙️)
  - Laptop-Mic Doctor-side (🎙️→)
  - Self-corrections (↩️)
  - Interruptions (✋)
  - Topic-jumps (🌀)
  - Mind-changes (🔁)
  - Chaos (💥 — all of the above)
  - Long anamnesis (~19 min) (⏳)
  - PWC physiotherapy (⏳)

Alle 55 Zellen werden zunächst leer/grau dargestellt — sie werden in den Ergebnis-Folien farblich „gefüllt". (Optional: schon dezent gefüllt als Vorgriff.)

Unter der Matrix: kleine Note *„Ground truth transcripts manually annotated — including hesitations, self-corrections, and verbal slips, to test how models handle real-world messiness."*

**Sprech-Text:**
*„Wir haben fünf Modell-Kombinationen gegen elf gezielt konstruierte Szenarien getestet. Vom sauberen Studio-Audio bis zum vollen Chaos — Patient korrigiert sich, Arzt unterbricht, Gedanken springen. Insgesamt 55 kontrollierte Runs. Die Ground Truths haben wir manuell erstellt, mit allen Versprechern und Wiederholungen — denn echtes Sprechen ist nicht sauber."*

---

### Folie 7 — How We Measured: Three Lenses

**Ziel:** Bewertungsmethodik transparent machen. Drei Achsen, klar getrennt.

**Inhalt:**
Titel: *„Three independent measurements per run"*

**Grafik:** Drei nebeneinander stehende Karten (gleichgroß), jede mit einem Icon oben, einem Begriff (groß), einer 1-Zeilen-Erklärung und einem Mini-Beispiel:

| 📏 **WER** | 🔍 **LLM Fidelity** | ⚕️ **SOAP Quality** |
|---|---|---|
| Word Error Rate | Did the LLM hallucinate? | Is the note medically usable? |
| RAW STT vs. Ground Truth | RAW STT vs. Formatted | Claude-as-Judge, 0–8 points |
| *„Insertions, deletions, substitutions"* | *„After we stripped speaker labels — what else changed?"* | *„Subjective + Objective + Assessment + Plan, each 0–2"* |

Unter den drei Karten ein dezenter Hinweis-Streifen:
*„SOAP scoring stricter on A & P: any hallucinated diagnosis or treatment = 0 points (patient safety)."*

**Animation:** Karten erscheinen nacheinander, während Sprecher die jeweilige Achse erklärt.

**Sprech-Text:**
*„Wir messen auf drei Ebenen, unabhängig voneinander. Erstens: WER — wie genau ist das STT? Zweitens: hat das LLM beim Formatieren etwas dazuerfunden? Drittens — und das ist der Härtetest: ist die fertige SOAP-Notiz medizinisch brauchbar? Dafür haben wir Claude Sonnet 4.6 als Bewerter eingesetzt — strenge Regel: jede halluzinierte Diagnose oder Therapie ist automatisch null Punkte. Wenn der Plan nicht im Gespräch stand, darf er auch nicht in der Akte stehen."*

---

## Block 3: Live-Demo (Folie 8, ~4 min)

### Folie 8 — Demo

**Ziel:** Beweis dass es funktioniert. Aha-Moment. Übergabe zum zweiten Sprecher.

**Inhalt:** Sehr minimalistisch, dient nur als Bühne für den Live-Schritt.

- Titel: *„Let's run it live"*
- Unten klein die Demo-Setup-Info: *„Audio: 'Meinungswechsel' (72s) · Pipeline: Speechmatics + GPT-4o · Streamlit UI"*
- Ein „Play"-Symbol zentriert (das per Klick die Demo startet, falls Slide-Software das unterstützt)

**Was passiert (Sprecher B übernimmt!):**

1. **(20s)** Sprecher B macht klare Übergabe: *„Bevor wir die Zahlen zeigen — sehen wir uns das in Aktion an."* Beschreibt kurz das Demo-Szenario: *„In dieser Aufnahme tippt der Arzt erst auf Angina Pectoris — und revidiert sich am Ende zu Reflux. Frage: behält die KI den Kontext und übernimmt die finale Diagnose?"*
2. **(70s)** Streamlit-UI öffnen, Audio abspielen lassen (alternativ: Audio im Slide einbetten und parallel das Streamlit-Fenster zeigen)
3. **(60s)** Modell laufen lassen — während es läuft erklärt Sprecher B, was im Hintergrund passiert: *„Speechmatics transkribiert gerade, GPT-4o formatiert, dann wird die SOAP-Note gestreamt..."*
4. **(60s)** SOAP-Output zeigen, Aufmerksamkeit lenken auf die **Assessment-Sektion**: ist „Reflux" und nicht „Angina" eingetragen? **Das ist der Aha-Moment.**
5. **(20s)** Kurzer „aber"-Cliffhanger: *„Sieht beeindruckend aus. Aber: war das jetzt ein lucky shot — oder ist das systematisch? Schauen wir auf die Zahlen."*

**WICHTIG — Backup-Plan (auf Slide 8b als versteckte Folie hinterlegen):**
- Pre-recorded Screen-Recording derselben Demo (max 90s, beschleunigt) falls Live-Demo scheitert
- Screenshot-Sequenz (5 Bilder: Audio-Upload → Transkript → Format → SOAP → Highlight Reflux) als Notfall

**Risiken:**
- Internet/Speechmatics-Latenz: 72s Audio + Cloud-Roundtrip kann Streamlit-mäßig 30-90s dauern → einplanen
- Vorher Streamlit gestartet halten und Audio gecached
- **Kein neuer Whisper-Lokal-Run** in der Demo (zu unvorhersehbar) — bewusst auf Cloud-Pipeline gehen

---

## Block 4: Ergebnisse (Folien 9–12, ~5 min)

### Folie 9 — Result #1: Cloud STT Halves Word Error

**Ziel:** Erste klare Ergebnis-Aussage. Fazit = Titel.

**Inhalt:**
Titel ist das Ergebnis: *„Cloud STT cuts WER nearly in half"*

**Grafik (DETAILLIERT):**
**Horizontales Balkendiagramm** mit den 5 Modellen auf der Y-Achse, WER% auf der X-Achse.

Bars sortiert von kleinstem zu größtem WER:
- AssemblyAI + GPT-4o: **8.9%** (Akzent-Orange, gefüllt) — *„best"*
- Speechmatics + GPT-4o: **11.4%** (Akzent-Orange, gefüllt)
- Whisper + Llama 3.2: **13.3%** (grau)
- Whisper + Gemma: **13.3%** (grau)
- Whisper + SauerkrautLM: **~14%** (grau)

Visuelle Trennung: oben zwei Cloud-Modelle in Farbe, drei lokale Modelle darunter ausgegraut → der Klassenunterschied wird sofort sichtbar.

Rechts neben dem Chart eine kleine Callout-Box:
- *„Outlier scenario: White Noise"*
- *„AssemblyAI: 33% WER · Whisper-local: 41% WER"*
- *„Microphone hardware matters more than the STT model"*

**Sprech-Text:**
*„Die Cloud-Modelle sind hier klar vorne — AssemblyAI mit 8,9 Prozent WER. Aber spannender als die Bestenliste: bei verrauschten Aufnahmen brechen ALLE ein. Das White-Noise-Szenario kostet jeden Anbieter 20 bis 30 Prozentpunkte. Die Lehre: Mikrofon-Qualität ist nicht ein 'nice-to-have' — sie ist die Untergrenze dessen, was die KI überhaupt erreichen kann."*

---

### Folie 10 — Result #2: Where LLMs Make Things Up

**Ziel:** Zeigen, dass das STT-Problem nur die halbe Wahrheit ist — LLMs erfinden auch Inhalt, vor allem lokal und vor allem in langen Gesprächen.

**Inhalt:**
Titel: *„Local LLMs hallucinate — and silently shorten long conversations"*

**Grafik (DETAILLIERT — Two-Panel-Layout):**

**Panel links: Bar Chart „Average LLM error rate"**
- Speechmatics + GPT-4o: **0.0%**
- AssemblyAI + GPT-4o: **0.03%**
- Whisper + Gemma: **13.4%**
- Whisper + Llama 3.2: **15.1%**
- Whisper + SauerkrautLM: **19.7%**

Die Cloud-LLMs visuell als „grüne Insel" (sehr dünne grüne Bars), die lokalen als deutlich dickere rot-orange Bars.

**Panel rechts: „Die Anamnese-Katastrophe" — Vergleichsbalken Wortzahl**
Zwei vertikale Balken nebeneinander:
- *„Ground Truth Anamnese conversation"*: **~2.269 words** (großer Balken, neutralfarben)
- *„After SauerkrautLM formatting"*: **152 words** (winziger Balken, rot)
- Großzahl-Label: *„−93% — content silently dropped"*

**Sprech-Text:**
*„Cloud LLMs halten sich an die Anweisung 'ändere nur die Speaker-Labels'. GPT-4o erfindet praktisch nichts. Die lokalen Modelle dagegen — schaut hier rechts. Bei einem 19-Minuten-Anamnese-Gespräch hat SauerkrautLM 8B von 2.269 Wörtern nur noch 152 übriggelassen. Das ist keine Formatierung mehr, das ist stille Datenlöschung. Und in der Medizin ist 'still gelöscht' gefährlicher als 'sichtbar falsch'."*

---

### Folie 11 — Result #3: The SOAP Quality Heatmap

**Ziel:** Das Kern-Ergebnis. Visuelle Verdichtung der 55 Runs auf eine Folie. Hier soll das Publikum am längsten verweilen.

**Inhalt:**
Titel: *„How usable is the final note? — Claude-judged on 8 dimensions"*

**Grafik (HAUPTGRAFIK der Präsentation, DETAILLIERT):**

**5×11 Heatmap** (gleicher Aufbau wie Folie 6, aber jetzt mit Farbfüllung):
- **Y-Achse:** 5 Modell-Kombinationen (sortiert nach Gesamt-Score, beste oben)
- **X-Achse:** 11 Szenarien
- **Zellfarbe:** Score 0–8
  - 7-8: Grün (akzeptabel)
  - 4-6: Gelb/Orange (überarbeitungs-bedürftig)
  - 0-3: Rot (unbrauchbar)
- **Zellinhalt:** Score-Zahl in der Mitte (z.B. „5" oder „2")

Rechts neben jeder Zeile eine **Summary-Spalte**:
| Model | Ø Score | ✓ / ⚠ / ✗ |
|---|---|---|
| Speechmatics + GPT-4o | **4.5** | 0 / 10 / 1 |
| AssemblyAI + GPT-4o | **4.3** | 0 / 9 / 2 |
| Whisper + Gemma | 3.4 | 0 / 5 / 6 |
| Whisper + Sauerkraut | 3.2 | 0 / 4 / 7 |
| Whisper + Llama 3.2 | **2.1** | 0 / 2 / 9 |

Unter der Heatmap ein **roter Hervorhebungs-Streifen** mit Schock-Aussage:
> ⚠️ *„Zero out of 55 runs reached 'acceptable' quality (≥7/8). The Plan section (P) averaged 0.0–0.3 / 2 across ALL models — every system hallucinated treatments."*

**Sprech-Text:**
*„Das ist die Master-Tabelle. 55 Bewertungen durch Claude. Drei Dinge, die euch hängenbleiben sollen: Erstens — die Cloud-Kombis sind oben, das ist konsistent zu dem, was wir vorher gesehen haben. Zweitens — Gemma war übrigens an den 'roten' Stellen oft loop-anfällig, das hat seine Ausfälle erklärt. Und drittens, der wichtigste Punkt: NICHT EIN EINZIGER Run hat die Schwelle 'akzeptabel' erreicht. Und der Plan-Teil — also was als nächstes mit dem Patienten passieren soll — war bei jedem Modell die größte Halluzinationsquelle. Generische 'Wir machen ein Röntgen, dann Antibiotika' — obwohl davon im Gespräch nie die Rede war."*

---

### Folie 12 — The Verdict: Best Combination — but not yet Ready

**Ziel:** Die Lessons aus Folie 11 in eine klare Aussage destillieren. Keine Heilsbotschaft.

**Inhalt:** Zwei-Spalten-Layout.

**Linke Spalte — „Winner":**
- Großschrift: **„Speechmatics + GPT-4o"**
- Drei Häkchen darunter:
  - ✅ Lowest hallucination rate (0%)
  - ✅ Best SOAP score (4.5 / 8)
  - ✅ Best speaker attribution (10/11 patients correctly identified)

**Rechte Spalte — „But":**
- Großschrift: **„Still not clinically usable"**
- Drei Warnungen darunter:
  - ⚠️ Zero runs reached „acceptable"
  - ⚠️ Plan section: 0.1 / 2 average
  - ⚠️ Performance collapses with bad audio

Unten zentriert (Schlusssatz, der hängen bleibt):
> *„The technology assists, but doesn't yet replace, a clinician's pen."*

**Sprech-Text:**
*„Wenn wir uns auf eine Kombination festlegen müssten: Speechmatics plus GPT-4o. Aber: 'best of five' ist nicht 'gut'. Wir würden heute keinem Arzt empfehlen, eine dieser Notizen ungesichtet zu übernehmen. Die KI assistiert — sie ersetzt noch nicht."*

---

## Block 5: Limitationen & Datenschutz (Folien 13–14, ~2 min)

### Folie 13 — What We Didn't Build (and Why)

**Ziel:** Bewusst gewählte Grenzen unserer Untersuchung erklären — proaktiv, statt es in Q&A aufzuarbeiten.

**Inhalt:**
Titel: *„Three deliberate boundaries"*

Drei nebeneinander angeordnete Karten:

**Karte 1: 🎙️ „Microphone hardware"**
- *„We tested 3 mic positions (DC, laptop-center, laptop-doctor-side) + white noise"*
- Erkenntnis: *„Mic quality dominates STT quality. A clip-on lavalier would likely solve 80% of WER outliers."*

**Karte 2: ⏱️ „No real-time mode"**
- *„We chose batch over streaming — to evaluate model quality fairly, decoupled from latency engineering"*
- Erkenntnis: *„Real-time is a productization step, not a model-evaluation step. Next iteration."*

**Karte 3: 🔄 „No prompt-engineering iteration"**
- *„We used the same prompt for all models — to compare models, not prompts"*
- Erkenntnis: *„Especially the Plan section is heavily prompt-sensitive. Big lever for the next phase."*

**Sprech-Text:**
*„Drei Sachen, die wir bewusst nicht gemacht haben — und warum. Hardware: ein Lavalier-Mikro am Arzt-Kragen würde wahrscheinlich die Hälfte unserer WER-Probleme beheben. Real-time: bewusst weggelassen, weil das Engineering-Aufwand ist, der die Modell-Frage nicht beantwortet. Und Prompt-Engineering: gleicher Prompt für alle, fair vergleichbar — aber wir wissen, gerade beim Plan-Teil liegt da viel Hebel."*

---

### Folie 14 — Privacy: The Hidden Trade-off

**Ziel:** Zeigen, dass „bestes Modell" nicht nur eine technische Frage ist.

**Inhalt:**
Titel: *„The best model and the safe model aren't the same"*

**Grafik (DETAILLIERT — zweidimensionales Quadranten-Diagramm):**

X-Achse: *„SOAP Quality (Score 0-8)"* — von links (schwach) nach rechts (stark)
Y-Achse: *„Data sovereignty (where is the audio processed?)"* — von unten (US Cloud) nach oben (Fully Local)

Fünf Punkte als Scatter-Plot:
- **Speechmatics + GPT-4o**: oben-rechts-Bereich (4.5 SOAP, partially EU — Speechmatics EU, GPT-4o US) — gelber Punkt
- **AssemblyAI + GPT-4o**: rechts-unten (4.3 SOAP, full US) — roter Punkt
- **Whisper + Sauerkraut**: oben-links (3.2 SOAP, full local) — grüner Punkt
- **Whisper + Llama 3.2**: oben-links (2.1 SOAP, full local) — grüner Punkt
- **Whisper + Gemma**: oben-links (3.4 SOAP, full local) — grüner Punkt

Visualisierter „idealer Quadrant" (oben rechts: hohe Qualität + voll lokal) bleibt **leer** — mit einem dezenten Stern und der Beschriftung *„Where we want to be"*.

Unter dem Quadranten ein Pfeil von „Whisper + Sauerkraut 8B (local)" zu einem gestrichelt umrandeten Punkt im idealen Quadranten, beschriftet:
*„Whisper large-v3 + SauerkrautLM 70B on dedicated EC2 → planned next run"*

**Sprech-Text:**
*„Datenschutz ist kein Bonus — in der Medizin ist es Pflicht. Unser bestes Setup schickt Audio in zwei Cloud-Welten. Unsere lokalen Setups bleiben im Krankenhaus, sind aber qualitativ schwächer. Der ideale Quadrant — hohe Qualität UND voll lokal — ist heute mit Consumer-Hardware nicht erreichbar. Genau das ist der nächste Schritt: ein 70B-Modell self-hosted."*

---

## Block 6: Ausblick & Take-aways (Folien 15–16, ~1.5 min)

### Folie 15 — What's Next

**Ziel:** Positiver Ausblick, was als nächstes käme. Konkret, nicht handwedelnd.

**Inhalt:**
Titel: *„Three concrete next steps"*

Drei Boxen vertikal gestapelt (oder horizontal als Roadmap-Streifen):

**1. 🚀 „Scale up the local stack"**
- *„Whisper large-v3 + SauerkrautLM 70B on dedicated EC2/on-prem GPU"*
- *„Expected: close the cloud-vs-local quality gap"*

**2. 🎯 „Prompt engineering on the Plan section"**
- *„Few-shot examples · constrained generation · 'cite-or-skip' for treatments"*
- *„Expected: lift the average P-score from 0.1 to 1.0+"*

**3. ⏱️ „Real-time streaming UX"**
- *„Stream STT during conversation, generate SOAP on-the-fly"*
- *„Expected: physician sees draft note before patient leaves the room"*

**Grafik:** Roadmap-Streifen von links nach rechts mit den drei Schritten als Stationen. Jede Station hat ein passendes Icon.

**Sprech-Text:**
*„Drei klare Hebel für die nächste Runde. Bessere lokale Modelle auf eigener Infrastruktur. Gezieltes Prompt-Engineering, vor allem für den Plan-Teil. Und Real-time, damit die Notiz schon fertig ist, wenn der Patient den Raum verlässt."*

---

### Folie 16 — Three Things to Remember

**Ziel:** Das Publikum hat — laut PwC-Folien — am Ende eine Sache im Kopf. Wir geben ihnen drei, klar nummeriert.

**Inhalt:**
Titel: *„If you remember only three things..."*

Drei sehr große, nummerierte Statements (eines pro Drittel der Folie):

**1.** *„The pipeline works — but the **Plan section** is where every model still hallucinates."*

**2.** *„**Hardware** (microphone) and **model size** dominate — the cloud gap will close as local models grow."*

**3.** *„AI today **assists** medical documentation. It does not yet **replace** it."*

Unten ein einzelnes Iconpaar (Stethoskop + Microchip) und Namens-/Mail-Zeile von Anton & Leonard für Kontakt.

**Sprech-Text:**
*„Drei Sätze, die wir euch mitgeben wollen. Erstens — die Pipeline funktioniert, aber der Therapieplan ist die offene Wunde. Zweitens — Mikrofon und Modellgröße sind die Stellschrauben, die Cloud-Lücke schließt sich, sobald wir lokal größer werden. Drittens — KI assistiert, sie ersetzt heute noch nicht. Danke!"*

---

### Folie 17 — Q&A

**Inhalt:** Minimal, eine PwC-Stil-Folie.
- Großes *„Questions?"* in zentraler Serif-Schrift
- Optional kleines Bild im Hintergrund (Hörsaal / Stethoskop)
- Klein unten: Kontakt-E-Mails der beiden Sprecher + ggf. GitHub-Link zum Repo

---

## Globale Hinweise für die KI-PPT-Erstellung

### Konsistente Designelemente (alle Folien)
- **Footer:** „AI-Anamnesis · Master Seminar · PwC × FH Hannover · 2026" + Folien-Nr. rechts
- **Header (außer Title-Slide):** Mini-Pipeline-Indikator (drei dezent verbundene Punkte oben rechts, der aktuell relevante Punkt hervorgehoben) → zeigt jederzeit, in welchem Pipeline-Schritt man ist
- **Akzentfarbe:** ein Orange (z.B. #D04A02 / PwC-Orange) ODER ein Medizin-tieferes Rot. Konsistent.
- **Schrift:** Eine Serif für Überschriften (z.B. PwC's „ITC Charter" oder Garamond), eine Sans für Body (Inter/Helvetica).
- **Foliennummerierung** dezent (PwC-Style: rechts unten klein)

### Animationen (sparsam einsetzen!)
- Nur dort, wo es die Argumentation stützt (Pipeline-Stages, Result-Reveal, Heatmap-Aufbau, drei-Karten-Reveal). Niemals zur Dekoration.
- Bei Folien 9-11 (Ergebnisse): Bars/Werte einblenden, *nachdem* Sprecher die Frage gestellt hat — nicht vorher (Spannung halten).

### Risiko-Mitigation
- **Demo-Backup** muss VOR der Präsentation als versteckte Folie und als separates Video bereitliegen
- **Bildschirmauflösung** vorher prüfen (Beamer-Test)
- **Streamlit-Cache** vorwärmen (Audio bereits einmal vor Demo durchlaufen lassen, Latenz hängt von Speechmatics ab)
- **Kabel** (HDMI + ggf. Adapter) selbst mitbringen

### Was wir bewusst weglassen
- Keine ausführlichen Code-Folien (max. 1 Snippet wenn überhaupt, eher gar keiner)
- Keine Folie „Über uns" am Anfang — wir sind nicht das Thema
- Kein Inhaltsverzeichnis als Folie 2 — Burning Platform direkt
- Keine separate „Methodik"-Großüberschriftsfolie — der Übergang ergibt sich aus dem Flow

---

## Kritische Quell-Dateien (zum Cross-Check der Zahlen)

| Datei | Was zu prüfen |
|---|---|
| `/home/leonardn/gitprojs/Seminar/docs/wer/wer_*.md` | WER-Werte auf Folie 9 |
| `/home/leonardn/gitprojs/Seminar/docs/llm/llm_check_*.md` | LLM-Fehlerraten auf Folie 10 |
| `/home/leonardn/gitprojs/Seminar/docs/soap_eval_auswertung.md` | SOAP-Scores auf Folie 11/12 |
| `/home/leonardn/gitprojs/Seminar/docs/soap_strukturcheck.md` | Anzahl Runs mit komplettem SOAP |
| `/home/leonardn/gitprojs/Seminar/docs/speaker/speaker_check.md` | Speaker-Attribution auf Folie 12 |
| `/home/leonardn/gitprojs/Seminar/results/history_no_speaker.json` | Falls Live-Beispiele aus echten Runs eingebaut werden |

**Empfehlung:** Bevor ihr die PPT mit einer KI generiert, einmal alle Zahlen aus den `.md`-Dateien gegen den Plan abgleichen — die Zahlen oben stammen aus einem Explore-Run, im Detail bitte verifizieren (vor allem die letzte Nachkomma-Stelle der WER-Werte).

---

## Verifikation des Plans

**Testkriterien, ob die Präsentation den Anforderungen genügt:**

1. **20-min-Test:** Plan einmal langsam vorlesen + Demo realistisch 4 min annehmen → muss in 20 ± 2 min landen. (Faustregel: 60-90 Sek pro inhaltlicher Folie + 4 min Demo = 16-22 min ✓)
2. **„Eine-Folie-eine-Message"-Test:** Jede Folie sollte in einem Satz zusammenfassbar sein. Die Folien-Titel selbst sind so formuliert, dass sie diese Hauptaussage tragen (PwC-Tipp: „Fazit als Titel").
3. **Burning-Platform-Test:** Folien 1-3 müssen ohne Lösung auskommen, nur Problem. ✓
4. **Demo-Test:** Folie 8 ist visuell minimal, damit die Demo den Raum bekommt. ✓
5. **Pflicht-Inhalte aus eurer Map:** ✅ STT-Auswahl + Hardware-Grenze (Folie 5) · ✅ Gemma-Loop (Folie 11 Sprech-Text) · ✅ SOAP-Eval mit Claude (Folie 7 + 11) · ✅ Datenschutz EU-Hosting (Folie 14) · ✅ Hardware-Abhängigkeit Mikro (Folie 9 + 13) · ✅ Real-time-Diskussion (Folie 13) · ✅ Ausblick Prompt-Engineering + bessere Modelle (Folie 15)
