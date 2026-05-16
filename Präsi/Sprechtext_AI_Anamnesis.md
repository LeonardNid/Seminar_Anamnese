# Sprechtext — AI-Anamnesis of Patients
## 20 Minuten · Sprecher A (Folien 1–9) · Demo · Sprecher B (Folien 11–20)

> **Lesehinweis:** Zeiten in Klammern sind kumuliert. Texte sind Richtwerte — nicht auswendig lernen, sondern den Kern verstehen und frei sprechen. ~130 Wörter/Minute als Pace. 

---

## SPRECHER A — Einleitung & Methodik

### Folie 1 — Title Slide *(0:00 – 0:30)*

> Stellt euch vor, ihr sitzt beim Arzt. Ihr erzählt von eurem Problem — und der Arzt schaut die ganze Zeit auf den Bildschirm und tippt. Das Gespräch dauert vielleicht zehn Minuten, aber die Dokumentation danach dauert zwanzig. Genau hier setzt unser Projekt an.

Notes:
18 sekunden <br>
Idee: 
Arzt hält kein Augenkontakt.
Ist so beschäftigt, dass er euch keine fragen stellen kann.

---

### Folie 2 — 2 Hours *(0:30 – 1:15)*

> Zwei Stunden. Jeden Tag. Pro Arzt. Das ist mehr Zeit am Bildschirm als am Patienten. Laut einer Studie aus den Annals of Internal Medicine verbringen Ärzte für jede Stunde direkte Patientenzeit etwa zwei Stunden mit Dokumentation in der elektronischen Patientenakte.
>
> Und das hat Folgen: 49 Prozent der Ärzte berichten von Burnout-Symptomen — und die Nummer eins unter den Ursachen ist nicht etwa der Patientenkontakt, sondern die administrative Last. Das ist die Burning Platform, an der wir ansetzen.


Notes: 45 sek <br>
Ideen: "Burning Platform" satz muss überarbeitet werden aber dieser term ist schmutz. <br>

---

### Folie 3 — The Question *(1:15 – 2:00)*

> Unsere Forschungsfrage für dieses Projekt war: Kann eine KI ein Arzt-Patienten-Gespräch mithören und daraus automatisch eine brauchbare medizinische Dokumentation erzeugen? Und „brauchbar" heißt für uns drei Dinge: **Akkurat** — die Fakten müssen stimmen. **Strukturiert** — im SOAP-Format, dem Medizinstandard. Und **vertrauenswürdig** — keine erfundenen Diagnosen oder Therapien.

Notes: 35 sek <br>
Ideen: Bei der aufzählung von Akkurat und so muss besser sein. Aktuell lesen wir einfach die folie vor.

---

### Folie 4 — Four Stages *(2:00 – 3:15)*

> Unsere Pipeline hat vier Stufen. *(Stufen nacheinander einblenden)*
>
> **Erstens: Audio Capture.** Das Gespräch wird aufgenommen — ganz klassisch als Audiodatei.
>
> **Zweitens: STT.** Also Speech-to-Text, das ist die Technologie, die gesprochene Sprache in geschriebenen Text umwandelt. Hier passiert auch die Diarisierung, also die Zuordnung, welcher Sprecher was gesagt hat. Das STT-Modell gibt uns so etwas wie „SPEAKER_A sagt dies, SPEAKER_B sagt das".
>
> **Drittens: Format LLM.** Ein LLM — ein Large Language Model — ersetzt diese anonymen Speaker-Labels durch echte Bezeichnungen: Bspw. wird SPEAKER_A durch "Arzt" ersetzt und SPEAKER_B durch "Frau Klein". Ansonsten ändert es nichts am Text.
>
> **Viertens: SOAP LLM.** Dasselbe LLM generiert aus dem formatierten Transkript die strukturierte SOAP-Notiz. Auf SOAP gehen wir später (noch genauer) ein.

Notes: 1min 15 sek <br>
Idee: 

---

### Folie 5 — The Contenders *(3:15 – 4:30)*

> Welche Modelle haben wir getestet? *(auf Tabellen zeigen)*
>
> Bei Speech-to-Text haben wir drei verschiedene Modelle gewählt: **AssemblyAI** und **Speechmatics** als Cloud-Dienste, und **Whisper turbo** von OpenAI als lokales Modell. Whisper ist Open Source und läuft auf eigener Hardware. Es gibt auch Whisper large-v3, das größere Modell — aber das war für uns nicht nur wegen der Hardware unrealistisch, sondern auch wegen der Geschwindigkeit. Whisper turbo ist ein optimierter Kompromiss aus Qualität und Speed.
>
> Bei den LLMs haben wir uns für **GPT-4o** als Cloud-Variante entschieden, und folgende drei lokale Modelle — **SauerkrautLM**, **Llama** und **Gemma**, alle haben zwischen 3 und 8 Milliarden Parametern. SauerkrautLM ist ein auf Deutsch feingetuntes Modell, was für medizinische Gespräche auf Deutsch besonders relevant ist.
>
> Daraus ergeben sich die folgenden fünf Kombinationen. Warum nicht Cloud-STT mit lokalem LLM oder umgekehrt? Wenn die Audiodaten eh schon in der Cloud sind, macht es wenig Sinn, das LLM dann lokal laufen zu lassen. Und wenn man lokal bleiben will wegen Datenschutz, dann muss die ganze Pipeline lokal sein.

Notes: 1 min 30 sek <br>
Idee: 

---

### Folie 6 — Hardware Draws the Line *(4:30 – 5:30)*

> Links seht ihr, was wir tatsächlich nutzen konnten: einen Laptop mit AMD-Prozessor, integrierter Grafik und 16 GB RAM. Das reicht für Whisper turbo und die 8B-Modelle — eng, aber es funktioniert.
>
> Rechts seht ihr, was wir gerne getestet hätten: Whisper large-v3 und SauerkrautLM 70B. Die Ergebnisse wären vermutlich deutlich besser gewesen. Wir haben tatsächlich versucht, über PwC eine AWS-Instanz mit GPU zu bekommen, um die 70B-Variante zu evaluieren — aber das hat im Zeitrahmen des Seminars leider nicht geklappt. Das wäre ein klarer nächster Schritt.

---

### Folie 7 — From Clean Recordings to Total Chaos *(5:30 – 6:30)*

> Wir haben elf Szenarien entworfen, bewusst in vier Schwierigkeitsstufen.
>
> **Standard**: saubere Aufnahme mit dem Diktiergerät, einmal Laptop in der Mitte, einmal beim Arzt. **Challenging**: Hintergrundgeräusche, Patient korrigiert sich selbst, häufige Unterbrechungen. **Complex**: Gedankensprünge, Meinungswechsel — der Arzt tippt erst auf Angina, revidiert sich dann zu Reflux. Und „Chaos" kombiniert alles. **Real-World**: Ein 19-minütiges echtes Anamnesegespräch und eine Physiotherapie-Aufnahme von PwC.
>
> Alle Ground Truths haben wir manuell erstellt — mit allen Versprechern, Füllwörtern und Wiederholungen. Denn echtes Sprechen ist nicht sauber, und genau das wollten wir testen.

---

### Folie 8 — Three Independent Measurements *(6:30 – 8:00)*

> Wir messen auf drei Achsen, unabhängig voneinander.
>
> **WER — Word Error Rate**: Wie genau ist die Transkription? Wir vergleichen den STT-Output direkt mit unserer manuellen Ground Truth und zählen Einfügungen, Löschungen und Ersetzungen.
>
> **LLM Fidelity**: Hat das LLM beim Formatieren etwas verändert, das es nicht hätte ändern sollen? Wir vergleichen den rohen STT-Text mit dem formatierten Text — nach Abzug der Speaker-Label-Änderungen. Alles andere, was sich verändert hat, ist ein Fehler.
>
> **SOAP Quality**: Ist die fertige Notiz medizinisch brauchbar? Hier haben wir **Claude als automatisierten Bewerter** eingesetzt — ein sogenanntes „LLM-as-Judge"-Setup. Warum nicht per Skript? Weil die Bewertung von medizinischem Text zu komplex für regelbasierte Auswertung ist — man muss den Sinn verstehen, nicht nur Wörter zählen. Warum nicht menschlich? Bei 55 Runs mit je vier SOAP-Sektionen wären das 220 manuelle Bewertungen — das ist im Seminar-Zeitrahmen nicht realistisch, und die Inter-Rater-Reliabilität wäre fragwürdig. Claude bewertet konsistent, reproduzierbar, und wir konnten den Bewertungsprompt genau kontrollieren.
>
> Wichtige Regel: Jede halluzinierte Diagnose oder Therapie gibt automatisch null Punkte. Wenn der Plan nicht im Gespräch stand, darf er nicht in der Akte stehen.

---

### Folie 9 — SOAP Explainer *(8:00 – 8:30)*

> Kurz zur Struktur: SOAP hat vier Sektionen. **S — Subjective**: Was sagt der Patient? Beschwerden, Symptome, eigene Schilderung. **O — Objective**: Was beobachtet der Arzt? Befunde, Vitalwerte. **A — Assessment**: Die Diagnose. **P — Plan**: Was passiert als nächstes — Therapie, Überweisung, Medikation. Jede Sektion haben wir auf einer Skala von 0 bis 2 bewertet, insgesamt also maximal 8 Punkte.

*(Übergabe an Sprecher B für Demo)*

---

## DEMO — Folie 10 *(8:30 – 12:30)*

### Sprecher B übernimmt

> Bevor wir die Zahlen anschauen — sehen wir uns das Ganze in Aktion an.
>
> *(Streamlit/UI öffnen)*
>
> In dieser Aufnahme passiert etwas Spannendes: Der Arzt tippt zunächst auf Angina Pectoris — also ein Herzproblem. Im Verlauf des Gesprächs revidiert er sich und stellt die Diagnose Reflux. Die Frage ist: Behält die KI den Kontext und übernimmt die finale Diagnose?
>
> *(Audio abspielen lassen, Pipeline durchlaufen lassen — ~90 Sekunden)*
>
> Während das läuft: Im Hintergrund transkribiert jetzt das STT-Modell, dann formatiert GPT-4o die Speaker-Labels, und dann wird die SOAP-Note generiert.
>
> *(Output zeigen)*
>
> Schauen wir uns das Ergebnis an. **Subjective** — hier sehen wir die Beschwerden des Patienten, gut zusammengefasst. **Objective** — Befunde vom Arzt. **Assessment** — und hier steht tatsächlich Reflux, nicht Angina. Das Modell hat den Meinungswechsel korrekt nachvollzogen.
>
> Aber jetzt der **Plan**: *(vorlesen)*. Und hier wird es problematisch. Das Modell schreibt hier Therapievorschläge rein, die im Gespräch nie erwähnt wurden. „Röntgen veranlassen", „Antibiotika in Betracht ziehen" — das ist frei erfunden. Genau das ist die Halluzination, die wir im Plan-Teil bei *jedem* Modell sehen. Der Plan ist die Achillesferse der Pipeline.
>
> *(Kurze Pause)*
>
> Sieht beeindruckend aus, oder? Aber war das jetzt ein Glücksfall — oder ist das systematisch? Schauen wir auf die Zahlen.

---

## SPRECHER B — Ergebnisse & Ausblick

### Folie 11 — Cloud Beats Local (WER) *(12:30 – 13:30)*

> Erste Achse: Transkriptionsqualität. Der Whisper-Durchschnitt liegt bei 13,5 Prozent Fehlerrate, die Cloud-Modelle deutlich darunter — Assembly bei 8,9 Prozent.
>
> Aber der spannendere Befund steht rechts: Im White-Noise-Szenario brechen *alle* Modelle ein. Assembly geht auf 33 Prozent, Whisper-lokal auf 41 Prozent. Die Lektion: Die Mikrofonqualität setzt die Untergrenze dessen, was die KI überhaupt erreichen kann. Das beste Modell hilft nichts, wenn das Audio schlecht ist.

---

### Folie 12 — Cloud Listens, Local Rewrites (LLM Fidelity) *(13:30 – 14:15)*

> Zweite Achse: Verändert das LLM beim Formatieren etwas, das es nicht soll? GPT-4o in der Cloud: praktisch null Prozent Fehler. Es macht genau das, was wir ihm sagen — nur die Speaker-Labels ersetzen, sonst nichts anfassen.
>
> Die lokalen Modelle? Sauerkraut kommt auf fast 20 Prozent Fehlerrate. Die schreiben um, kürzen, fügen hinzu. Sie folgen der Anweisung nicht.

---

### Folie 13 — The Anamnese Catastrophe *(14:15 – 15:15)*

> Und hier wird es richtig drastisch. Das 19-Minuten-Anamnesegespräch — unser längstes Szenario. Ground Truth: 2.269 Wörter. Nach dem Formatieren durch SauerkrautLM: 152 Wörter übrig. **93 Prozent** des Inhalts — still und leise gelöscht.
>
> Und das ist kein Einzelfall: Alle lokalen Modelle haben massive Probleme mit langen Gesprächen. Die Kontextfenster der kleinen Modelle sind zu begrenzt — bei 3 bis 8 Milliarden Parametern reicht der Kontext schlicht nicht aus, um ein 19-minütiges Gespräch vollständig zu verarbeiten. Das Modell fängt an zu kürzen, zu halluzinieren, oder bricht ab. In der Medizin ist „still gelöscht" gefährlicher als „sichtbar falsch".

---

### Folie 14 — SOAP Quality *(15:15 – 16:00)*

> Dritte Achse — der Härtetest. Die SOAP-Bewertung durch Claude. Zur Einordnung: Rot bedeutet unter 3 von 8 — nicht brauchbar. Gelb ist 3 bis 6 — braucht Überarbeitung. Grün wäre ab 6 — akzeptabel.
>
> GPT-4o-Kombinationen liegen im gelben Bereich, die lokalen Modelle größtenteils im roten. Und rechts die Verteilung: Kein einziges der 55 Ergebnisse hat die Schwelle „akzeptabel" erreicht. Null von 55. Und der Hauptgrund: Jedes Modell halluziniert im Plan-Teil. Therapievorschläge, die nie im Gespräch standen.

---

### Folie 15 — Verdict *(16:00 – 16:45)*

> Wenn wir uns auf einen Gewinner festlegen: Assembly plus GPT-4o. Niedrigste Fehlerrate bei der Transkription, null Halluzination beim Formatieren, bester SOAP-Score.
>
> Aber — und das ist der entscheidende Punkt: „Bester von fünf" heißt nicht „gut". Null Runs haben „akzeptabel" erreicht. Der Plan-Teil liegt im Schnitt bei 0,1 von 2 Punkten. Und bei schlechtem Audio bricht alles zusammen. Wir würden heute keinem Arzt empfehlen, eine dieser Notizen ungeprüft zu übernehmen.

---

### Folie 16 — Limitations *(16:45 – 17:15)*

> Drei Sachen, die wir bewusst nicht gemacht haben. **Kein Real-time**: Wir haben bewusst im Batch-Modus gearbeitet, um Modellqualität isoliert zu bewerten, ohne Latenz-Engineering. **Kein Prompt-Tuning**: Gleicher Prompt für alle Modelle — fair vergleichbar. Aber gerade beim Plan-Teil liegt da enormer Hebel. **Nur Deutsch**: Alle Szenarien waren auf Deutsch. Ob die Ergebnisse auf andere Sprachen übertragbar sind, ist offen.

---

### Folie 17 — Privacy *(17:15 – 17:45)*

> Datenschutz ist in der Medizin nicht optional. Unser bestes Setup schickt Audiodaten in die US-Cloud. Die lokalen Modelle bleiben auf eigener Hardware — aber liefern schlechtere Qualität. Der ideale Quadrant — oben rechts, hohe Qualität und voll lokal — ist heute leer. Genau da wollen wir hin.

---

### Folie 18 — Next Steps *(17:45 – 18:45)*

> Drei konkrete nächste Schritte.
>
> **Erstens: Größere lokale Modelle.** SauerkrautLM 70B auf dedizierter GPU. Das sollte die Qualitätslücke zur Cloud deutlich schließen.
>
> **Zweitens: Prompt Engineering für den Plan-Teil.** Few-Shot-Beispiele, Constrained Generation, „cite-or-skip" — also: Wenn eine Therapie nicht im Gespräch erwähnt wurde, soll das Modell sie auch nicht vorschlagen. Außerdem: Chunking für längere Gespräche, damit auch 19-Minuten-Anamnesen vollständig verarbeitet werden.
>
> **Drittens: Real-time Streaming.** Das STT läuft während des Gesprächs, die SOAP-Note wird live generiert. Ziel: Der Arzt hat den Entwurf fertig, bevor der Patient den Raum verlässt.

---

### Folie 19 — Three Things *(18:45 – 19:30)*

> Drei Sätze, die wir euch mitgeben. *(nacheinander einblenden)*
>
> **Eins:** Die Pipeline funktioniert — aber der Plan-Teil halluziniert bei jedem Modell.
>
> **Zwei:** Hardware und Modellgröße sind die Stellschrauben — die Cloud-Lücke schließt sich, sobald lokale Modelle wachsen.
>
> **Drei:** KI assistiert heute bei der medizinischen Dokumentation. Ersetzen tut sie es noch nicht.
>
> Danke!

---

### Folie 20 — Q&A *(19:30+)*

*(Keine vorbereitete Rede — Fragen aus dem Publikum)*

---

## Anhang: Vorbereitung auf Q&A-Fragen

### „Warum kein Mixed Setup — Cloud-STT mit lokalem LLM?"

> Wenn die Audiodaten eh schon in die Cloud gehen für das STT, dann ist der Datenschutzvorteil eines lokalen LLMs hinfällig. Und umgekehrt: Wenn man wegen Datenschutz lokal bleiben will, muss die ganze Kette lokal sein. Mixed ergibt für keinen Use Case einen Vorteil.

### „Warum nur ein lokales STT-Modell (Whisper)?"

> Es gibt im Open-Source-Bereich für deutsches STT praktisch keine ernst zu nehmende Alternative zu Whisper. Modelle wie wav2vec2 oder Vosk existieren, sind aber bei Deutsch deutlich schwächer — insbesondere bei Diarisierung. Whisper turbo war die beste Balance aus Qualität, Geschwindigkeit und Community-Support.

### „Warum genau SauerkrautLM / Llama / Gemma?"

> SauerkrautLM: Bewusst gewählt, weil es das einzige auf Deutsch feingetunte Open-Source-LLM in der 8B-Klasse ist — relevant für deutschsprachige medizinische Texte. Llama 3.2: Meta's aktuelles Flaggschiff in der 3B-Klasse, starke Benchmark-Werte. Gemma: Googles kompaktes Modell, das wir als Vergleich zum Meta-Ökosystem testen wollten. Alternativen wie Mistral 7B oder Phi-3 wären auch denkbar gewesen, aber wir mussten den Scope begrenzen.

### „Warum Claude als Judge statt menschlicher Bewertung?"

> 55 Runs × 4 SOAP-Sektionen = 220 Einzelbewertungen. Manuell durch zwei Personen wäre das eine Woche Arbeit — und dann hätten wir immer noch Inter-Rater-Reliabilitätsprobleme. Claude ist konsistent, reproduzierbar, und der Bewertungsprompt ist dokumentiert und versioniert. Das ist ein etablierter Ansatz in der ML-Evaluationsforschung — „LLM-as-Judge" wird auch in Papers von Google und Anthropic verwendet.

### „Wie zuverlässig ist Claude als Judge?"

> Wir haben den Bewertungsprompt iterativ optimiert und stichprobenartig gegen unsere eigene Einschätzung validiert. Perfekt ist es nicht — aber für ein relatives Ranking der Modelle untereinander ist es ausreichend valide. Absolute Scores sollte man mit Vorsicht interpretieren.
