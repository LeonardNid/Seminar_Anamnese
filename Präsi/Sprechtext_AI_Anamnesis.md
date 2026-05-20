# Sprechtext — AI-Anamnesis of Patients
## 20 Minuten · Sprecher A (Folien 1–9) · Demo · Sprecher B (Folien 11–20)

> **Lesehinweis:** Zeiten in Klammern sind kumuliert. Texte sind Richtwerte — nicht auswendig lernen, sondern den Kern verstehen und frei sprechen. ~130 Wörter/Minute als Pace. 


# Aufteilung: Anton (1-9), Leo (10-20)

Beide 1.5 min puffer

---

## SPRECHER A — Einleitung & Methodik

### Folie 1 — Title Slide *(0:00 – 0:30)*

> Stellt euch vor, ihr sitzt beim Arzt. Ihr erzählt von eurem Problem — aber der Arzt schaut euch gar nicht an. Er tippt. Ihr erzählt weiter — er tippt. Ihr seid fertig — er tippt noch. Das Gespräch dauert zehn Minuten, die Dokumentation danach zwanzig. Und weil er so beschäftigt ist, hat er euch nicht mal gefragt, seit wann die Beschwerden schon da sind. Genau hier setzt unser Projekt an.

Notes:
30 sekunden

---

### Folie 2 — 2 Hours *(0:30 – 1:15)*

> Zwei Stunden. Jeden Tag. Pro Arzt. Das ist mehr Zeit am Bildschirm als am Patienten. Laut einer Studie aus den Annals of Internal Medicine verbringen Ärzte für jede Stunde direkte Patientenzeit etwa zwei Stunden mit Dokumentation in der elektronischen Patientenakte.
>
> Und das hat Folgen: 49 Prozent der Ärzte berichten von Burnout-Symptomen — und die Nummer eins unter den Ursachen ist nicht etwa der Patientenkontakt, sondern die administrative Last. 


Notes: 45 sek 

---

### Folie 3 — The Question *(1:15 – 2:00)*

> Unsere Forschungsfrage war: Kann eine KI ein Arzt-Patienten-Gespräch mithören und daraus automatisch eine brauchbare medizinische Dokumentation erzeugen? Brauchbar klingt dabei einfach — aber dahinter stecken drei Anforderungen, die sich gegenseitig in die Quere kommen können. Die Fakten müssen stimmen. Das Format muss passen. Und das System darf sich nichts ausdenken, was nie gesagt wurde. Alle drei gleichzeitig zu erfüllen — das ist die eigentliche Herausforderung.

Notes: 45 sek

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

Notes: 1min 15 sek 

---

### Folie 5 — The Contenders *(3:15 – 4:30)*

> Welche Modelle haben wir getestet? *(auf Tabellen zeigen)*
>
> Bei Speech-to-Text haben wir drei verschiedene Modelle gewählt: **AssemblyAI** und **Speechmatics** als Cloud-Dienste, und **Whisper turbo** von OpenAI als lokales Modell. Whisper ist Open Source und läuft auf eigener Hardware. Es gibt auch Whisper large-v3, das größere Modell — aber das war für uns nicht nur wegen der Hardware unrealistisch, sondern auch wegen der Geschwindigkeit. Whisper turbo ist ein optimierter Kompromiss aus Qualität und Speed.
>
> Bei den LLMs haben wir uns für **GPT-4o** als Cloud-Variante entschieden, und folgende drei lokale Modelle — **SauerkrautLM**, **Llama** und **Gemma**, alle haben zwischen 3 und 8 Milliarden Parametern. SauerkrautLM ist ein auf Deutsch feingetuntes Modell, was für medizinische Gespräche auf Deutsch besonders relevant ist.
>
> Daraus ergeben sich die folgenden fünf Kombinationen. Warum nicht Cloud-STT mit lokalem LLM oder umgekehrt? Wenn die Audiodaten eh schon in der Cloud sind, macht es wenig Sinn, das LLM dann lokal laufen zu lassen. Und wenn man lokal bleiben will wegen Datenschutz, dann muss die ganze Pipeline lokal sein.

Notes: 1 min 30 sek 

---

### Folie 6 — Hardware Draws the Line *(4:30 – 5:30)*

> Links seht ihr, was wir tatsächlich nutzen konnten: einen Laptop mit AMD-Prozessor und 16 GB RAM. Das reicht für Whisper turbo und unsere lokalen Modelle — knapp, aber es funktioniert.
>
> Rechts seht ihr, was wir gerne getestet hätten: Whisper large-v3 und zb SauerkrautLM 70B. Bessere Hardware würde höchstwahrscheinlich zu deutlich besseren vollkommen lokalen Ergebnissen führen. Wir haben tatsächlich versucht in Kooperation mit PwC eine AWS-Instanz mit entsprechend starker GPU zu bekommen, um bessere lokale Varianten zu evaluieren — zunächst haben uns technische Probleme aufgehalten, und als die gelöst waren, war der Zeitrahmen zu knapp.

Notes: 1 min

---

### Folie 7 — From Clean Recordings to Total Chaos *(5:30 – 6:30)*


> Wir haben elf Szenarien entworfen und diese bewusst in vier Schwierigkeitsstufen aufgeteilt. Die Standardszenarien sind saubere Aufnahmen — einmal mit guter Mikrofon Hardware das Original und zweimal mit dem Laptop als Mikrofon — einmal mittig zwischen Arzt und Patient und einmal direkt vor dem Arzt. 
> Ab der zweiten Stufe wird es schwieriger: Einmal mit starken Hintergrundgeräuschen, dann mit Patienten die sich selbst korrigieren und noch (einmal) mit häufigen Unterbrechungen zwischen Arzt und Patient. 
> In den komplexen Szenarien springt der Patient zwischen Themen hin und her, der Arzt revidiert seine Einschätzung mitten im Gespräch, und im Chaos-Szenario kommt alles zusammen. 
> Die letzte Stufe sind echte Aufnahmen — ein 19-minütiges Anamnesegespräch und eine Physiotherapie-Sitzung bereitgestellt von PwC. 
> Alle Ground Truths haben wir manuell erstellt, mit Versprechern, Füllwörtern und Wiederholungen. Denn echtes Sprechen ist nicht sauber — und genau das wollten wir testen.

Notes: 1 min 25 sek
- bei laptop scenarien mit laserpoiner arbeiten
- Vielleicht "jeweils mit dem laptop mikro"

---

### Folie 8 — Three Independent Measurements *(6:30 – 8:00)*

> Wir messen auf drei Achsen, unabhängig voneinander. 
>
> Die erste ist die Word Error Rate — wie genau ist die Transkription? Dafür vergleichen wir den STT-Output mit unserer manuellen Ground Truth und zählen Einfügungen, Löschungen und Ersetzungen. 
>
> Die zweite Achse ist LLM Fidelity: Hat das Modell beim Formatieren etwas verändert, das es nicht hätte anfassen sollen? Wir vergleichen den rohen STT-Text mit dem formatierten Ergebnis — alles, was sich über die Speaker-Label-Änderungen hinaus verändert hat, zählt als Fehler. 
> 
> Die dritte Achse ist SOAP Quality, also ob die fertige Notiz medizinisch brauchbar ist. Hier haben wir Claude als automatisierten Bewerter eingesetzt. Per Skript wäre das nicht möglich gewesen — wir prüfen nicht Wort für Wort, sondern ob der Inhalt des Transkripts sinngemäß in der SOAP-Note landet. Das ist eine semantische Aufgabe, keine regelbasierte. Und manuell? 55 Runs, vier Sektionen jeweils — das wären 220 Einzelbewertungen, im Seminar-Zeitrahmen schlicht nicht realistisch. Claude bewertet konsistent, mit einem dokumentierten Prompt. Und ein weiterer Grund: Claude war kein Teil unserer Pipeline — damit gibt es keinen Self-Evaluation-Bias. Hätten wir GPT-4o als Bewerter genutzt, hätte es möglicherweise seine eigenen Outputs bevorzugt. 
>
> Eine Regel gilt dabei absolut: Jede halluzinierte Diagnose oder Therapie gibt null Punkte. Wenn der Plan nicht im Gespräch stand, darf er nicht in der SOAP-Zusammenfassung stehen.

Notes: 1 min 35 sek


---

### Folie 9 — SOAP Explainer *(8:00 – 8:30)*

> SOAP steht für vier Sektionen. 
> 
> Subjective erfasst, was der Patient selbst schildert — Beschwerden, Symptome, eigene Worte. 
> 
> Objective ist das, was der Arzt beobachtet, also Befunde und Vitalwerte. 
> 
> Assessment ist die Diagnose, und 
> 
> Plan beschreibt die nächsten Schritte — Therapie, Überweisung, Medikation. 
> 
> Jede dieser vier Sektionen haben wir auf einer Skala von null bis zwei bewertet, insgesamt also maximal acht Punkte.

Notes: 40 sek


565

---

## DEMO — Folie 10 *(8:30 – 12:30)*

Original Text von Mind-Change:

**Arzt:** Frau Hoffmann, was machen wir denn mit Ihnen? Sie fassen sich die ganze Zeit an die Brust.

**Patientin:** Es brennt so furchtbar hier hinter dem Brustbein. So ein richtig drückendes Brennen.

**Arzt:** Das nehmen wir sehr ernst. _Notiz: Leitsymptom retrosternaler Schmerz, Verdacht auf Angina Pectoris, eventuell kardiales Ereignis._ Strahlt das in den linken Arm oder in den Unterkiefer aus?

**Patientin:** Nein, gar nicht. Es ist nur direkt hinter dem Knochen.

**Arzt:** Haben Sie Atemnot oder kalten Schweiß?

**Patientin:** Überhaupt nicht. Es brennt einfach. Besonders schlimm ist es, wenn ich nach dem Essen flach auf dem Sofa liege. Und ich habe die ganze Zeit so einen eklig sauren Geschmack im Mund, ich muss ständig aufstoßen.

**Arzt:** Ah, warten Sie. Nach dem Essen und beim Liegen? Und ein saurer Geschmack?

**Patientin:** Ja, genau. Gestern Abend nach der Tomatensoße war es extrem.

**Arzt:** Okay, dann Kommando zurück. _Notiz: Korrektur. Kein Anhalt für kardiale Ischämie._ Das ändert die Sache komplett. Wir streichen den Verdacht auf Angina Pectoris oder einen Herzinfarkt. Das klingt ganz klassisch nach Sodbrennen, also einem Reflux. Die Magensäure steigt in Ihre Speiseröhre auf, das verursacht diesen Schmerz.

---

## 1. Zusammenfassung des Gesprächs

Frau Hoffmann kommt mit Brustschmerzen. Der Arzt denkt zunächst an Angina Pectoris — also ein Herzproblem — und notiert das. Dann fragt er weiter: Strahlt der Schmerz in den Arm aus? Atemnot? Alles verneint. Die Patientin erwähnt dann, dass es nach dem Essen und beim Liegen schlimmer wird, und dass sie einen sauren Geschmack im Mund hat. Daraufhin revidiert der Arzt seine Diagnose komplett — kein Herz, sondern klassischer Reflux. Das Gespräch ist damit ein perfekter Testfall für Meinungswechsel: Die finale Diagnose widerspricht der ersten Einschätzung.

Reflux — offiziell gastroösophageale Refluxkrankheit — ist wenn Magensäure zurück in die Speiseröhre steigt. Das verursacht das typische Sodbrennen, also dieses brennende Gefühl hinter dem Brustbein. Weil der Schmerz dort sitzt, wird es anfangs oft mit Herzproblemen verwechselt — genau wie in unserem Szenario.

---

## 2 & 3. Überarbeiteter Sprechtext

> Bevor wir die Zahlen anschauen — sehen wir uns einmal die Pipeline live an und das anhand des Szenarios: Mind-change, bei dem
> 
> Frau Hoffmann mit Brustschmerzen in die Praxis kommt und der Arzt zunächst ein Herzproblem vermutet. Erst im weiteren Gespräch — als die Patientin erwähnt, dass es nach dem Essen und beim Liegen schlimmer wird — erkennt er, dass es doch nur ein einfaches Sodbrennen ist. 
> 
> _(Streamlit öffnen)_
> 
> Wir lassen das Szenario mit AssemblyAI und GPT-4o laufen und die Aufnahme ist zwar mit 70 Sekunden sehr kurz — aber für unseren Proof of Concept reicht das vollkommen.
> 
> _(Start)_
> 
> _(STT zeigen)_
> 
> Wie wir sehen können, hat das STT den Text für die Audio erstellt und auch die Sprecher korrekt erkannt.
> 
> _(Formatted zeigen)_
> 
> Die LLM hat dann die Speaker-Labels korrekt durch „Arzt" und „Frau Hoffmann" ersetzt — sonst nichts verändert.
> 
> _(SOAP zeigen)_
> 
> Und jetzt zum eigentlich spannenden Teil, die SOAP-Zusammenfassung. 
>
> Im Subjective sehen wir die Beschwerden der Patientin: (Beschwerden kurz vorlesen) brennender Schmerz hinter dem Brustbein, schlimmer nach dem Essen und beim Liegen, saurer Geschmack im Mund. 
>
> Im Objective steht — korrekt — dass keine Befunde dokumentiert wurden, das Gespräch hat keine Untersuchung beinhaltet. 
>
> Im Assessment steht Reflux (Sodbrennen), nicht Angina. Der initiale Verdacht auf ein kardiales Ereignis ist explizit ausgeschlossen. Das Modell hat den Meinungswechsel korrekt nachvollzogen.
> 
> Und jetzt der Plan. (einige Stichpunkte vorlesen). 
> 
> Das klingt alles medizinisch plausibel — und genau das ist das Problem. Nichts davon wurde im Gespräch erwähnt. Das Modell hat sich das alles ausgedacht. Und weil es so plausibel klingt, fällt es möglicherweise im Alltag gar nicht auf. 
> Und ich denke, keiner von uns will, dass halluzinierte Therapievorschläge unbemerkt in einer Patientenakte landen. 
> Ob das ein Einzelfall war oder öfter passiert schauen wir uns jetzt an.
> 

Notes: Überarbeiten 

---

### Folie 11 — Cloud Beats Local (WER) *(12:30 – 13:30)*

> Schauen wir uns zuerst die erste Stufe der Pipeline an, also das STT und die Transkriptionsqualität. Whisper hat im Schnitt 13,5 Prozent aller Wörter falsch transkribiert, Speechmatics bei 11,4 und Assembly bei 8,9. Der Unterschied zwischen lokal und Cloud ist ehrlich gesagt kleiner als man erwarten würde — wir reden von ein paar Prozentpunkten, nicht von einer anderen Größenordnung.
>
> Ein weiterer spannender Befund steht rechts: Im White-Noise-Szenario brechen *alle* Modelle ein. Die Cloud Modelle wachsen auf über 30 Prozent, Whisper sogar auf 41 Prozent. Die Lektion: Die Mikrofonqualität setzt die Untergrenze dessen, was die KI überhaupt erreichen kann. Das beste Modell hilft nichts, wenn die Audio schlecht ist.

Notes: 55 sek

WER auf die x achse schreiben?

---

### Folie 12 — Cloud Listens, Local Rewrites (LLM Fidelity) *(13:30 – 14:15)*

> Jetzt zur zweiten Stufe der Pipeline. 
>
> Die LLM sollte nur die Lable der Speaker ersetzen und mehr nicht. Alle weiteren Änderungen zählen also als fehler und genau das haben wir gemessen.
>
> Sauerkraut kommt auf fast 20 Prozent Fehlerrate. Die lokalen Modelle schreiben um, kürzen, fügen hinzu und im extremfall, fassen sie das gesamte Gespräch zusammen. Sie folgen der Anweisung also nur eingeschränkt.
> 
> GPT-4o macht dagegen praktisch keine Fehler. Und was besonders bemerkenswert ist: In einigen Szenarien hat das STT nur einen einzigen Sprecher erkannt, also gar keine Diarisierung geliefert. GPT-4o hat trotzdem beide Sprecher durchgehend korrekt gelabelt — allein aus dem Kontext des Textes heraus.


Notes: 1 min

---

### Folie 13 — The Anamnese Catastrophe *(14:15 – 15:15)*

> Und hier wird es richtig drastisch, wir schauen uns jetzt eine dieser Zusammenfassungen an, nämlich Das 19-Minuten-Anamnesegespräch — unser längstes Szenario. Der Raw STT Output enthält 2.269 Wörter. Nach dem Formatieren durch SauerkrautLM sind nur noch 152 Wörter übrig. 

> Hier wurde also **93 Prozent** des Inhalts still und leise gelöscht.
>
> Und das ist kein Einzelfall. Alle lokalen Modelle haben massive Probleme mit längeren Gesprächen. 
>
>Interesanterweise haben alle Modelle das gleiche Kontextwindow, und dennoch versagen nur die lokalen Modelle.
>
> Das Hauptproblem an dieser Stelle ist, dass in der Medizin still gelöschte Information gefährlicher sind als sichtbar falsche.

Notes: 50 sek

---

### Folie 14 — SOAP Quality *(15:15 – 16:00)*

> Am Ende der Pipeline steht die entscheidende Frage — und zwar, ob die  SOAP-Zusammenfassung medizinisch brauchbar ist? Zur Einordnung: Rot liegt bei 0 bis inklusieve 3 Punkten und damit nicht brauchbar, Gelb zwischen 3 und 6 bedeutet überarbeitungsbedürftig, und mehr als 6 wäre akzeptabel.
> 
> Die GPT-4o-Kombinationen landen im gelben Bereich. Gemma und Sauerkraut liegen ebenfalls im gelben Bereich, aber am unteren Ende. Llama fällt als einziges in den roten Bereich. 
>
> Zusätlich kann man rechts in der Tabelle sehen, dass kein einziges der 55 Ergebnisse die Schwelle akzeptabel erreicht — null von 55. Der Plan-Teil ist bei jedem Modell halluziniert. 
>Überall stehen Therapievorschläge, die nie im Gespräch vorkamen.

Notes: 1 min

---

### Folie 15 — Verdict *(16:00 – 16:45)*

> Wenn wir uns auf Gewinner festlegen müssten, wären es einmal AssemblyAI und einmal GPT-4o. AssemblyAI, da sie die niedrigste Fehlerrate bei der Transkription hat 
>und GPT-4o hat null Halluzination beim Formatieren und den beste SOAP-Score.
> 
> Aber — und das ist der entscheidende Punkt: „Bester von fünf" heißt nicht unbedingt „gut". Bei SOAP haben null Runs „akzeptabel" erreicht, wie eben erwähnt. Der Plan-Teil liegt im Schnitt bei 0,1 von 2 Punkten. Und bei einer schlechten Audio bricht nach wie vor alles zusammen. Wir würden also heute keinem Arzt empfehlen, eine dieser Notizen ungeprüft zu übernehmen.

Notes: 45 sek

---

### Folie 16 — Limitations *(16:45 – 17:15)*

> Drei Dinge haben wir bewusst ausgelassen. 
> Wir haben bewusst auf Real-time verzichtet, obwohl das ursprünglich Teil der Aufgabenstellung war. Bei Streaming sieht das Modell immer nur einen Bruchteil des Gesprächs, wodurch die Qualität leidet. Dazu kamen Hardware-Grenzen. Wir haben uns deshalb für die Batch-Verarbeitung entschieden, um die Modellqualität sauber und vergleichbar messen zu können.
> Außerdem haben wir den Prompt nicht für einzelne Modelle optimiert, sondern für alle denselben verwendet — das macht den Vergleich fair, lässt aber gerade beim Plan-Teil von SOAP enormes Potenzial liegen. 
> Und alle Szenarien waren auf Deutsch. Ob die Ergebnisse auf andere Sprachen übertragbar sind, bleibt offen.

Notes: 45 sek

---

### Folie 17 — Privacy *(17:15 – 17:45)*

> Datenschutz ist in der Medizin nicht optional. Schauen wir uns an, wo die Modelle stehen.
>
> Die lokalen Modelle bleiben vollständig auf eigener Hardware — Datenschutz ist gewährleistet, aber die Qualität bleibt deutlich hinter den Cloud-Modellen zurück.
>
> Speechmatics ist ein Kompromiss — das STT läuft in der EU, das LLM jedoch in den USA.
>
> Assembly und GPT-4o liefern die beste Qualität, schicken aber die Audiodaten vollständig in die US-Cloud. Der ideale Quadrant — hohe Qualität und voll lokal — ist heute leer. Genau da wollen wir hin.

Notes: 40 sek

Hier keine Animationen

---

### Folie 18 — Next Steps *(17:45 – 18:45)*

> Wo geht es von hier aus weiter?
> Der erste Schritt sind größere lokale Modelle — SauerkrautLM 70B beispielsweise auf dedizierter GPU, um die Qualitätslücke zur Cloud zu schließen. 

> Der zweite Schritt ist Prompt Engineering. Zum Beispiel eine klare Regel: Wenn eine Therapie im Gespräch nicht erwähnt wurde, darf das Modell sie nicht vorschlagen. 
>Außerdem könnte Chunking helfen, damit lokale Modelle auch lange Gespräche vollständig verarbeiten können.

> Und drittens Real-time Streaming — das STT läuft dann während des Gesprächs. Ziel ist dann, dass der Arzt den Entwurf fertig hat, bevor der Patient den Raum verlässt.

Notes: 1 min

---

### Folie 19 — Three Things *(18:45 – 19:30)*

> Drei Sätze, die wir euch mitgeben wollen. *(nacheinander einblenden)*
>
> **Erstens:** Die Pipeline funktioniert — aber der Plan-Teil ist halluziniert bei jedem einzelnen Durchlauf.
>
> **Zweitens:** Hardware und Modellgröße sind die Stellschrauben — die Cloud-Lücke schließt sich, sobald lokale Modelle wachsen.
>
> **Drittens:** KI assistiert heute bei der medizinischen Dokumentation. Ersetzen tut sie es noch nicht.

Notes: 30 sek

1010

---

### Folie 20 — Q&A *(19:30+)*

> Vielen Dank für eure Aufmerksamkeit, gibt es noch fragen?

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

> 55 Runs × 4 SOAP-Sektionen = 220 Einzelbewertungen. Manuell durch zwei Personen wäre das eine Woche Arbeit. Claude ist konsistent, und der Bewertungsprompt ist dokumentiert und versioniert. Das ist ein etablierter Ansatz in der ML-Evaluationsforschung — LLM-as-Judge wurde 2023 von Forschern der UC Berkeley formalisiert und wird seitdem in der ganzen Community eingesetzt.

### „Wie zuverlässig ist Claude als Judge?"

> Wir haben den Bewertungsprompt iterativ optimiert und stichprobenartig gegen unsere eigene Einschätzung validiert. Perfekt ist es nicht — aber für ein relatives Ranking der Modelle untereinander ist es ausreichend valide. Absolute Scores sollte man mit Vorsicht interpretieren.

### "Warum kein Multimodel?"

Der Kernunterschied: Multimodale Modelle wie GPT-4o mit Audio-Input machen STT und Textverarbeitung in einem Schritt — das klingt praktisch, ist aber problematisch.

Ein multimodales LLM versteht das Gespräch und gibt aus, was es für gemeint hält — nicht zwingend das, was tatsächlich gesagt wurde. Ein dediziertes STT-Modell transkribiert akustisch, Wort für Wort. Für medizinische Dokumentation ist das entscheidend.

Dazu kommt: Dedizierte Modelle haben eingebaute Diarisierung, sind günstiger, schneller, und explizit auf mehrsprachige Daten trainiert. Und mit getrennten Modellen können wir jeden Schritt der Pipeline isoliert messen — steckt alles in einem Modell, wissen wir nicht mehr, wo ein Fehler entsteht.

### Time stats

 Meinungswechsel:
  
  ┌────────────────────────┬──────┬────────┬──────┬────────┐
  │      Kombination       │ STT  │ Format │ SOAP │ Gesamt │
  ├────────────────────────┼──────┼────────┼──────┼────────┤
  │ Speechmatics + GPT-4o  │ 0:10 │ 0:03   │ 0:02 │ 0:15   │
  ├────────────────────────┼──────┼────────┼──────┼────────┤
  │ AssemblyAI + GPT-4o    │ 0:18 │ 0:03   │ 0:04 │ 0:26   │
  ├────────────────────────┼──────┼────────┼──────┼────────┤
  │ Whisper + SauerkrautLM │ 1:49 │ 1:25   │ 1:12 │ 4:26   │
  ├────────────────────────┼──────┼────────┼──────┼────────┤
  │ Whisper + llama3.2     │ 2:52 │ 1:09   │ 0:56 │ 4:57   │
  ├────────────────────────┼──────┼────────┼──────┼────────┤
  │ Whisper + gemma4       │ 1:56 │ 0:50   │ 2:22 │ 5:08   │
  └────────────────────────┴──────┴────────┴──────┴────────┘

  ---
  Anamnesegespräch (~19 min):

  ┌────────────────────────┬───────┬────────┬──────┬────────┐
  │      Kombination       │  STT  │ Format │ SOAP │ Gesamt │
  ├────────────────────────┼───────┼────────┼──────┼────────┤
  │ AssemblyAI + GPT-4o    │ 0:36  │ 0:42   │ 0:06 │ 1:24   │
  ├────────────────────────┼───────┼────────┼──────┼────────┤
  │ Speechmatics + GPT-4o  │ 2:04  │ 0:24   │ 0:05 │ 2:34   │
  ├────────────────────────┼───────┼────────┼──────┼────────┤
  │ Whisper + gemma4       │ 26:23 │ 4:45   │ 2:55 │ 34:03  │
  ├────────────────────────┼───────┼────────┼──────┼────────┤
  │ Whisper + SauerkrautLM │ 23:51 │ 14:04  │ 1:31 │ 39:26  │
  ├────────────────────────┼───────┼────────┼──────┼────────┤
  │ Whisper + llama3.2     │ 35:26 │ 2:45   │ 1:20 │ 39:31  │
  └────────────────────────┴───────┴────────┴──────┴────────┘
