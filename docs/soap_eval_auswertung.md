# SOAP Evaluation: Auswertung

> Bewertet wird die Informationsübertragung vom Transkript in die SOAP-Notiz.
> Score pro Sektion: 0–2 | Gesamt: 0–8
> ✓ = akzeptabel (6–8) | ⚠ = Überarbeitung nötig (3–6) | ✗ = nicht verwendbar (0–3)
> `auto` = automatisch nicht_verwendbar (leeres SOAP oder >90% LLM-Fehlerrate)

---

## Übersicht

| Szenario | Whisper+llama3.2 | Speechmatics+GPT4o | AssemblyAI+GPT4o | Whisper+Sauerkraut | Whisper+gemma4 |
|---|---|---|---|---|---|
| OriginalDC | ⚠ 4/8 | ⚠ 4/8 | ✗ 3/8 | ⚠ 4/8 | ⚠ 4/8 |
| OriginalDC+Noise | ✗ 1/8 | ⚠ 4/8 | ⚠ 4/8 | ✗ 2/8 | ✗ 2/8 |
| LapInMitte | ✗ 2/8 | ⚠ 4/8 | ⚠ 4/8 | ✗ 3/8 | ⚠ 4/8 |
| LapBeiArzt | ✗ 1/8 | ⚠ 4/8 | ⚠ 4/8 | ✗ 3/8 | ✗ 3/8 |
| Selbstkorrekturen | ✗ 1/8 | ⚠ 4/8 | ✗ 3/8 | ✗ 3/8 | ✗ 3/8 |
| Unterbrechungen | ✗ 1/8 | ✗ 3/8 | ⚠ 4/8 | ✗ 3/8 | ⚠ 5/8 |
| Gedankensprünge | ✗ 3/8 | ⚠ 5/8 | ⚠ 4/8 | ⚠ 4/8 | ✗ 1/8 |
| Meinungswechsel | ⚠ 5/8 | ⚠ 6/8 | ⚠ 6/8 | ⚠ 5/8 | ⚠ 6/8 |
| Chaos | ✗ 1/8 | ⚠ 5/8 | ⚠ 5/8 | ⚠ 4/8 | ⚠ 4/8 |
| Anamnesegespräch | ✗ auto | ⚠ 4/8 | ⚠ 4/8 | ✗ auto | ✗ 2/8 |
| PWC | ✗ 2/8 | ⚠ 6/8 | ⚠ 6/8 | ✗ 1/8 | ✗ auto |

## Zusammenfassung pro Modell

| Modell | ✓ akzeptabel | ⚠ Überarb. | ✗ nicht verwendbar | Ø Score | Ø S | Ø O | Ø A | Ø P |
|---|---|---|---|---|---|---|---|---|
| Whisper+llama3.2 | 0 | 2 | 9 | 2.1 | 0.7 | 0.9 | 0.5 | 0.0 |
| Speechmatics+GPT4o | 0 | 10 | 1 | 4.5 | 1.5 | 2.0 | 0.9 | 0.1 |
| AssemblyAI+GPT4o | 0 | 9 | 2 | 4.3 | 1.4 | 1.9 | 0.9 | 0.1 |
| Whisper+Sauerkraut | 0 | 4 | 7 | 3.2 | 0.9 | 1.6 | 0.4 | 0.3 |
| Whisper+gemma4 | 0 | 5 | 6 | 3.4 | 1.3 | 1.6 | 0.5 | 0.0 |

---

## Details pro Modell

### Whisper+llama3.2

| Szenario | S | O | A | P | Gesamt | Urteil |
|---|---|---|---|---|---|---|
| OriginalDC | 1 | 1 | 2 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| OriginalDC+Noise | 0 | 1 | 0 | 0 | **1/8** | ✗ nicht_verwendbar |
| LapInMitte | 1 | 1 | 0 | 0 | **2/8** | ✗ nicht_verwendbar |
| LapBeiArzt | 1 | 0 | 0 | 0 | **1/8** | ✗ nicht_verwendbar |
| Selbstkorrekturen | 0 | 1 | 0 | 0 | **1/8** | ✗ nicht_verwendbar |
| Unterbrechungen | 1 | 0 | 0 | 0 | **1/8** | ✗ nicht_verwendbar |
| Gedankensprünge | 1 | 2 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| Meinungswechsel | 1 | 2 | 2 | 0 | **5/8** | ⚠ ueberarbeitung_noetig |
| Chaos | 0 | 1 | 0 | 0 | **1/8** | ✗ nicht_verwendbar |
| Anamnesegespräch | — | — | — | — | **auto** | ✗ nicht_verwendbar (`LLM-Fehlerrate 93% > 90%`) |
| PWC | 1 | 0 | 1 | 0 | **2/8** | ✗ nicht_verwendbar |

**Halluzinationen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | P | Antibiotika als Therapie wurden im Transkript nie erwähnt |
| OriginalDC | P | Amoxicillin 500 mg viermal täglich ist vollständig halluziniert |
| OriginalDC | P | Nachuntersuchung in zwei Wochen wurde im Transkript nicht besprochen |
| OriginalDC+Noise | S | Schmerzen beim Atmen werden als vorhanden beschrieben, obwohl im Transkript explizit 'Keine thorakalen Schmerzen' notiert wurde |
| OriginalDC+Noise | S | Messung des 'Schleimumfangs' – im Transkript wurde die Körpertemperatur gemessen, kein Schleimumfang |
| OriginalDC+Noise | A | 'Andere Lungenkrankheiten' als mögliche Diagnose hinzugefügt – im Transkript nicht erwähnt |
| OriginalDC+Noise | P | 'Überprüfung des Schleimumfangs' – im Transkript nicht als Maßnahme erwähnt |
| OriginalDC+Noise | P | 'Überwachung von Symptomen und Anzeichen einer Erkrankung' – generische Maßnahme, nicht im Transkript erwähnt |
| LapInMitte | A | Diagnoseeinschätzung 'bakterielle Atemwegsinfektion' kommt im Transkript nicht vor |
| LapInMitte | A | Differentialdiagnosen 'Bronchitis' und 'Pneumonie' werden im Transkript nicht genannt |
| LapInMitte | P | Bluttest als geplante Maßnahme wird im Transkript nicht erwähnt |
| LapInMitte | P | Röntgenaufnahme wird im Transkript nicht erwähnt |
| LapInMitte | P | Antibiotika-Empfehlung kommt im Transkript nicht vor |
| LapInMitte | P | Empfehlung 'Hustenbonbons vermeiden' ist halluziniert — der Arzt äußert sich dazu nicht |
| LapBeiArzt | O | "Reizhusten" – die Patientin hat einen trockenen Reizhusten explizit verneint; Husten ist stark produktiv. |
| LapBeiArzt | A | "Allergische Reaktion auf ein bestimmtes Hausmittel" – kein Hinweis auf eine Allergie im Transkript vorhanden. |
| LapBeiArzt | P | "Antibiotika zur Behandlung einer möglichen bakteriellen Infektion" – im Transkript nicht erwähnt. |
| LapBeiArzt | P | "Bronchodilatatormittel zur Linderung der Dyspnoe" – im Transkript nicht erwähnt. |
| Selbstkorrekturen | S | Schmerz strahlt in die linke Schläfe aus — Patient korrigierte sich im Transkript explizit zu 'rechte Schläfe' |
| Selbstkorrekturen | O | 'Leichte Besserung nach Ibuprofene 600' als Tatsache dargestellt — im Transkript (Arztnotiz) als 'fraglich' bezeichnet |
| Selbstkorrekturen | A | Migräne als mögliche Diagnose — wird im Transkript nicht genannt |
| Selbstkorrekturen | A | Tensionsschmerzen als mögliche Diagnose — wird im Transkript nicht genannt |
| Selbstkorrekturen | A | 'Neurologische Ursache' als Einschätzung — wird im Transkript nicht erwähnt |
| Selbstkorrekturen | P | Blutdruck- und Blutzuckerüberprüfung — nicht im Transkript erwähnt |
| Selbstkorrekturen | P | MRT oder CT-Scan — nicht im Transkript erwähnt |
| Selbstkorrekturen | P | Antihistaminika zur Symptomlinderung — nicht im Transkript erwähnt |
| Selbstkorrekturen | P | Körperliche Untersuchung als geplante Maßnahme — nicht im Transkript erwähnt |
| Selbstkorrekturen | P | Fortsetzung der Ibuprofene 600 als Therapieplan — nicht im Transkript erwähnt |
| Unterbrechungen | O | Frau Klein sei bei der Untersuchung 'leicht unruhig und ängstlich' – kein Untersuchungsbefund im Transkript |
| Unterbrechungen | O | 'Es gibt keine anderen besonderen Symptome wie Übelkeit, Erbrechen oder Durchfall' – widerspricht dem Transkript explizit, da Erbrechen (zweimal) klar erwähnt wird |
| Unterbrechungen | A | Verdacht auf akute Pankreatitis – kommt im Transkript nicht vor und wird vom Arzt zu keinem Zeitpunkt erwähnt oder angedeutet |
| Unterbrechungen | P | Bluttest für die Magensäure – nicht im Transkript |
| Unterbrechungen | P | Urintest – nicht im Transkript |
| Unterbrechungen | P | Empfehlung, bei Verschlechterung sofort den Notarzt zu rufen – nicht im Transkript |
| Gedankensprünge | S | Die SOAP stellt den Vergleich mit 2014 als ähnliche Schulterbeschwerden dar; tatsächlich war 2014 ein Kreuzbandriss am Knie – kein Schulterproblem. |
| Gedankensprünge | A | Die Kreuzbandriss-Vorgeschichte wird als stützendes Argument für die Schulterdiagnose angeführt; das Transkript stellt keinen solchen diagnostischen Zusammenhang her. |
| Gedankensprünge | P | Ultraschalluntersuchung der Schulter |
| Gedankensprünge | P | Röntgenaufnahmen zum Frakturausschluss |
| Gedankensprünge | P | Konservative Behandlung mit Schmerzlinderung und Physiotherapie |
| Meinungswechsel | P | Magenspülung oder Endoskopie wird im Transkript nicht erwähnt |
| Meinungswechsel | P | Medikamente (Antacida, Protonenhemmer) werden im Transkript nicht erwähnt |
| Meinungswechsel | P | Empfehlung zur Ernährungsumstellung und kein Trinken vor dem Schlafen wird im Transkript nicht erwähnt |
| Chaos | S | Schwindel wird als 'seit Samstagabend' angegeben, obwohl Patient sich im Transkript explizit auf Sonntagmorgen korrigiert |
| Chaos | S | Geschichte der Bekannten mit Akustikusneurinom wird zu einem 'Karzinom-Imperz der in die Röhre eingeschlagen wurde' entstellt — inhaltlich unverständlich und faktisch falsch |
| Chaos | A | SOAP schreibt, Neuritis vestibularis werde 'durch eine Entzündung des Gleichgewichtsorgans erklärt' — im Transkript wird dieser Verdacht explizit verworfen, nicht bestätigt |
| Chaos | P | Nachuntersuchung des rechten Ohrs wird empfohlen, obwohl im Transkript keine Wiedervorstellung vereinbart wird — Arzt impliziert, das Problem sei nach der Spülung erledigt |
| PWC | O | Frau Grasbeutner 'lebt in einem Haus' – im Transkript explizit als Wohnung im zweiten Stock angegeben. |
| PWC | O | Beim Hometrainer 'keine spezifischen Übungen durchgeführt' – sie hat damit Beugen und Strecken geübt. |
| PWC | P | Empfehlung für Heimphysiotherapie ('Physiotherapeut bei Hause') – wurde im Transkript nicht erwähnt. |
| PWC | P | 'Übungen zur Stärkung des Kniegelenks' als konkreter Planpunkt – im Transkript wird kein Therapieplan besprochen, es handelt sich nur um die Erstanamnese. |

**Auslassungen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | S | Belastungsdyspnoe beim Treppensteigen in den zweiten Stock (muss auf halbem Weg anhalten) wird ungenau als 'beim Aufstehen oder anstrengender körperlicher Aktivität' beschrieben |
| OriginalDC | S | Keine Ruhedyspnoe nicht explizit erwähnt |
| OriginalDC | O | Belastungsdyspnoe-Beschreibung ungenau: 'beim Aufstehen' statt 'beim Treppensteigen in den zweiten Stock' |
| OriginalDC | O | Auskultation der Lunge wurde begonnen, Befund fehlt (Transkript bricht ab) |
| OriginalDC+Noise | S | Keine Erwähnung der Medikamentenanamnese (keine spezifische Medikation, nur supportive Hausmittel) |
| OriginalDC+Noise | O | Medikamentenstatus fehlt (keine spezifische Formmedikation, nur Hausmittel) |
| LapInMitte | S | Medikamenteneinnahme / Hausmittel (Hustenbonbons, Kamillentee) fehlen vollständig |
| LapInMitte | S | Keine Ruhedyspnoe wird nicht explizit erwähnt |
| LapInMitte | O | Keine Angabe zu fehlender spezifischer Vormedikation / Hausmitteln |
| LapInMitte | O | Beginn der Auskultation (letzte Arzthandlung im Transkript) wird nicht erwähnt |
| LapBeiArzt | S | Keine Ruhedyspnoe und keine thorakalen Schmerzen wurden explizit verneint – fehlt in der SOAP. |
| LapBeiArzt | S | Vormedikation fehlt: Patientin nimmt nur Hustenbonbons und Hausmittel, keine spezifische Medikation. |
| Selbstkorrekturen | S | Patient verneinte Übelkeit explizit ('Nein, zum Glück nicht') |
| Selbstkorrekturen | S | Wirkung von Ibuprofen 600 war laut Transkript 'fraglich', nicht bestätigt |
| Selbstkorrekturen | O | Kein Übelkeit (explizit verneint, relevanter negativer Befund) |
| Selbstkorrekturen | O | Paracetamol 500 ohne Wirkung fehlt im objektiven Befund |
| Unterbrechungen | S | Patientin musste direkt auf die Toilette rennen (möglicher Hinweis auf Durchfall) – nicht erwähnt |
| Unterbrechungen | S | Nahrungsanamnese fehlt vollständig: Salat mittags, Grillen abends, möglicherweise nicht durchgegartes Hähnchen (rosa) – wichtiger ätiologischer Hinweis |
| Gedankensprünge | S | Arm blockiert beim Heben, kommt nicht über Kopfhöhe. |
| Meinungswechsel | S | Patientin berichtet, dass sie ständig aufstoßen muss – fehlt in der SOAP |
| Chaos | S | Ohrenschmerzen rechts ('mein rechtes Ohr, das tut auch weh') werden nicht erwähnt |
| Chaos | S | Ibuprofen-Einnahme heute Morgen (800 mg, zwei 400er) fehlt als Medikamentenanamnese |
| Chaos | O | Beobachtung des Arztes beim Eintreten ('wackelig auf den Beinen') fehlt |
| PWC | S | Zwei Krankenhausbesuche werden nicht differenziert: erster Besuch ohne Befund (Röntgen), zweiter Besuch am nächsten Morgen mit Schwellung und Diagnose. |
| PWC | S | Knie war am Morgen nach dem Unfall angeschwollen. |
| PWC | S | Erste Maßnahmen direkt nach dem Unfall (Kühlen, Hochlagern) fehlen. |
| PWC | S | Volleyball seit 4 Jahren, zweimal pro Woche im Verein – relevant für Therapieziel. |
| PWC | S | Therapieziel der Patientin (zuerst normal gehen, dann wieder Volleyball spielen) fehlt. |
| PWC | S | Es war der erste Physiotherapiebesuch der Patientin überhaupt. |
| PWC | O | Physiotherapie im Krankenhaus (einmal täglich, Krückengang, Fußabrollen, Treppensteigen, Beugen üben) fehlt. |
| PWC | O | MRT-Befund zur Diagnosesicherung wird nicht erwähnt. |
| PWC | O | Muskelabbau durch einwöchige Immobilisierung fehlt. |
| PWC | O | Schmerzmittel zuhause mitgegeben, aber nicht gebraucht – SOAP sagt fälschlicherweise sie habe sie 'benötigt'. |
| PWC | A | Status ist postoperativ (ca. 6 Wochen nach Op.) – die Diagnose ist gesichert, nicht nur 'wahrscheinlich'. |
| PWC | A | Muskelatrophie durch Immobilisierung als relevanter Befund fehlt. |
| PWC | A | Aktueller Funktionsstatus (Krückengang bereits gut integriert, aber Treppensteigen noch eingeschränkt) zu knapp dargestellt. |
| PWC | P | Therapieziel der Patientin (Wiederaufnahme des Volleyballsports) ist nicht im Plan aufgegriffen. |

---

### Speechmatics+GPT4o

| Szenario | S | O | A | P | Gesamt | Urteil |
|---|---|---|---|---|---|---|
| OriginalDC | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| OriginalDC+Noise | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| LapInMitte | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| LapBeiArzt | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| Selbstkorrekturen | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| Unterbrechungen | 1 | 2 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| Gedankensprünge | 1 | 2 | 2 | 0 | **5/8** | ⚠ ueberarbeitung_noetig |
| Meinungswechsel | 2 | 2 | 2 | 0 | **6/8** | ⚠ ueberarbeitung_noetig |
| Chaos | 1 | 2 | 2 | 0 | **5/8** | ⚠ ueberarbeitung_noetig |
| Anamnesegespräch | 0 | 2 | 2 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| PWC | 1 | 2 | 2 | 1 | **6/8** | ⚠ ueberarbeitung_noetig |

**Halluzinationen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | A | Verdacht auf bakterielle Infektion sowie konkrete Differenzialdiagnosen Bronchitis/Pneumonie werden im Transkript nicht genannt — der Arzt erwähnt nur 'Verdacht auf rezidivierendes Fieber'. |
| OriginalDC | P | Röntgenaufnahme des Thorax wird im Transkript nicht erwähnt. |
| OriginalDC | P | Blutuntersuchung (CRP, Leukozyten) wird im Transkript nicht erwähnt. |
| OriginalDC | P | Antibiotische Therapie wird im Transkript nicht erwähnt. |
| OriginalDC | P | Schleimlösende Mittel und Antipyretika werden im Transkript nicht erwähnt. |
| OriginalDC | P | Empfehlung zu Flüssigkeitszufuhr und Ruhe wird im Transkript nicht erwähnt. |
| OriginalDC+Noise | A | Verdacht auf akute Bronchitis oder Pneumonie wird im Transkript nie genannt |
| OriginalDC+Noise | A | Differentialdiagnose virale Infektion der oberen Atemwege mit sekundärer bakterieller Infektion erscheint nicht im Transkript |
| OriginalDC+Noise | P | Röntgenaufnahme des Thorax wird im Transkript nicht angeordnet oder erwähnt |
| OriginalDC+Noise | P | Blutuntersuchung (CRP, Leukozytenzahl) wird im Transkript nicht erwähnt |
| OriginalDC+Noise | P | Erwägung einer antibiotischen Therapie kommt im Transkript nicht vor |
| OriginalDC+Noise | P | Nachkontrolle in einer Woche wird im Transkript nicht vereinbart |
| LapInMitte | A | Verdacht auf akute Bronchitis als Diagnose genannt – kommt im Transkript nicht vor. |
| LapInMitte | A | Pneumonie als Differentialdiagnose genannt – kommt im Transkript nicht vor. |
| LapInMitte | P | Blutuntersuchung (CRP, Leukozytenzahl) – im Transkript nicht erwähnt. |
| LapInMitte | P | Sputumkultur – im Transkript nicht erwähnt. |
| LapInMitte | P | Röntgen-Thorax – im Transkript nicht erwähnt. |
| LapInMitte | P | Antibiotische Therapie bei Bestätigung einer bakteriellen Infektion – im Transkript nicht erwähnt. |
| LapInMitte | P | Schleimlösende Mittel und Antipyretika – im Transkript nicht erwähnt. |
| LapInMitte | P | Nachkontrolle in einer Woche – im Transkript nicht erwähnt. |
| LapBeiArzt | A | Verdacht auf bakterielle Bronchitis oder Pneumonie wird im Transkript nicht geäußert. |
| LapBeiArzt | A | Differenzialdiagnose virale Atemwegsinfektion kommt im Transkript nicht vor. |
| LapBeiArzt | P | Röntgenaufnahme des Thorax wird im Transkript nicht erwähnt. |
| LapBeiArzt | P | Blutuntersuchung (CRP, Leukozyten) wird im Transkript nicht erwähnt. |
| LapBeiArzt | P | Antibiotische Therapie wird im Transkript nicht erwähnt. |
| LapBeiArzt | P | Empfehlung zu Flüssigkeitszufuhr und Ruhe wird im Transkript nicht ausgesprochen. |
| Selbstkorrekturen | A | Diagnose 'Verdacht auf Migräne oder Spannungskopfschmerz' wird im Transkript an keiner Stelle genannt oder angedeutet. |
| Selbstkorrekturen | A | Aussage 'keine Anzeichen für sekundäre Kopfschmerzen aufgrund anderer neurologischer Symptome' — eine neurologische Beurteilung findet im Transkript nicht statt. |
| Selbstkorrekturen | P | Neurologische Untersuchung zur Differenzierung von Migräne und Spannungskopfschmerz — im Transkript nicht erwähnt. |
| Selbstkorrekturen | P | Empfehlung zur Vermeidung von Triggerfaktoren (Stress, Nahrungsmittel) — im Transkript nicht erwähnt. |
| Selbstkorrekturen | P | Verschreibung eines Triptans zur Akutbehandlung — im Transkript nicht erwähnt. |
| Selbstkorrekturen | P | Überweisung an einen Neurologen — im Transkript nicht erwähnt. |
| Selbstkorrekturen | P | Aufklärung über Nebenwirkungen der Medikation — im Transkript nicht erwähnt. |
| Unterbrechungen | S | SOAP nennt 'ohne Ausstrahlung in die Leiste' — Patientin hat Leistenausstrahlung nicht explizit verneint, sondern nur Rückenausstrahlung ausgeschlossen |
| Unterbrechungen | A | Differentialdiagnose 'Gastroenteritis' wird genannt, obwohl dieser Begriff im Transkript nicht vorkommt |
| Unterbrechungen | A | Differentialdiagnose 'Magen-Darm-Infektion' wird genannt, obwohl dieser Begriff im Transkript nicht vorkommt |
| Unterbrechungen | A | 'Gastrointestinale Infektion' als primäre Einschätzung geht über das Transkript hinaus — der Arzt äußerte nur den Verdacht auf unzureichend gegartes Hähnchen als Ursache |
| Unterbrechungen | P | Flüssigkeitszufuhr zur Dehydrierungsprävention wurde im Gespräch nicht besprochen |
| Unterbrechungen | P | Vorstellung in der Notaufnahme bei Verschlechterung wurde nicht erwähnt |
| Unterbrechungen | P | Wiedervorstellung mit weiterer Diagnostik (Blutuntersuchung, Stuhlprobe) nach 24 Stunden wurde nicht besprochen |
| Unterbrechungen | P | Ernährungsberatung mit Vermeidung schwer verdaulicher Lebensmittel wurde nicht erwähnt |
| Unterbrechungen | P | Weitere Beobachtung der Symptome und Temperatur wurde im Transkript nicht explizit als Plan genannt |
| Gedankensprünge | S | SOAP nennt 'rechte Schulter', im Transkript wird die betroffene Seite nicht erwähnt. |
| Gedankensprünge | P | MRT der Schulter wird als Maßnahme genannt – im Transkript nicht erwähnt. |
| Gedankensprünge | P | Schonung der Schulter und Schmerzmittelverordnung werden genannt – im Transkript nicht erwähnt. |
| Gedankensprünge | P | Überweisung an einen Orthopäden wird genannt – im Transkript nicht erwähnt. |
| Meinungswechsel | P | Medikamentöse Therapie mit Protonenpumpenhemmern wird empfohlen — im Transkript nicht erwähnt. |
| Meinungswechsel | P | Überweisung zur gastroenterologischen Abklärung — im Transkript nicht erwähnt. |
| Chaos | S | Die Bezugsperson mit Akustikusneurinom wird als 'familiäre Erfahrung' bezeichnet, obwohl im Transkript nur 'Die' (unbekannte Person) erwähnt wird – keine Familienbeziehung angegeben. |
| Chaos | P | Beobachtung und Rücksprache bei anhaltenden oder neuen Symptomen – im Transkript wird kein Folgetermin oder Wiedervorstellung erwähnt. |
| Anamnesegespräch | P | Schmerzmanagement mit Analgetika nach Bedarf wurde vom Arzt im Gespräch nicht als Maßnahme genannt |

**Auslassungen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | A | Der im Transkript explizit notierte 'Verdacht auf rezidivierendes Fieber' fehlt in der SOAP. |
| OriginalDC+Noise | O | Verdacht auf rezidivierendes Fieber (Arzt-Notation im Transkript) fehlt in O |
| OriginalDC+Noise | A | Arzt notiert 'Verdacht auf rezidivierendes Fieber' – dieser Verdacht fehlt in der Assessment |
| LapInMitte | A | Arzt notiert im Transkript 'Verdacht auf rezidivierendes Fieber' – fehlt im Assessment. |
| Chaos | S | Schwindel begann laut Patientenkorrektur und Arztnotiz seit Sonntagmorgen, nicht seit Samstagabend (SOAP übernimmt die uncorrected erste Aussage). |
| Anamnesegespräch | S | Chirurgische Vorgeschichte fehlt: Hallux-Valgus-OP am rechten Fuß vor zwei Jahren |
| Anamnesegespräch | S | Raucherstatus nicht dokumentiert: früher 6–7 Jahre geraucht, seit 8 Jahren aufgehört |
| Anamnesegespräch | S | Auslandsreise nach Singapur vor zwei Monaten (2 Wochen, geschäftlich) nicht erwähnt |
| Anamnesegespräch | S | Frühere gelegentliche Ritalin-Einnahme (2–3 Mal bei beruflichem Stress) fehlt |
| PWC | S | Erster Krankenhausbesuch (mit Röntgen) wird nicht erwähnt – Arzt fand damals nichts, was zur Verzögerung der Diagnose führte. |
| PWC | S | Am nächsten Morgen konnte die Patientin nicht aufstehen und das Knie war stark angeschwollen – relevanter Symptomverlauf fehlt. |
| PWC | S | Es ist der erste Physiotherapiebesuch der Patientin überhaupt (sie hatte noch keine Erfahrung). |
| PWC | P | Im Transkript wurde kein konkreter Behandlungsplan besprochen – es handelt sich ausschließlich um eine Anamnesephase; die genannten Maßnahmen sind sinnvolle Schlussfolgerungen, aber nicht explizit vereinbart. |

---

### AssemblyAI+GPT4o

| Szenario | S | O | A | P | Gesamt | Urteil |
|---|---|---|---|---|---|---|
| OriginalDC | 1 | 2 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| OriginalDC+Noise | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| LapInMitte | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| LapBeiArzt | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| Selbstkorrekturen | 1 | 2 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| Unterbrechungen | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| Gedankensprünge | 1 | 1 | 2 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| Meinungswechsel | 2 | 2 | 2 | 0 | **6/8** | ⚠ ueberarbeitung_noetig |
| Chaos | 1 | 2 | 2 | 0 | **5/8** | ⚠ ueberarbeitung_noetig |
| Anamnesegespräch | 0 | 2 | 2 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| PWC | 1 | 2 | 2 | 1 | **6/8** | ⚠ ueberarbeitung_noetig |

**Halluzinationen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | A | Verdacht auf akute Bronchitis wurde im Transkript nicht geäußert |
| OriginalDC | A | Verdacht auf Pneumonie wurde im Transkript nicht geäußert |
| OriginalDC | P | Röntgenaufnahme des Thorax wurde nicht erwähnt |
| OriginalDC | P | Blutuntersuchung (CRP, Leukozyten) wurde nicht erwähnt |
| OriginalDC | P | Antibiotische Therapie wurde nicht erwähnt |
| OriginalDC | P | Schleimlösende Mittel und Antipyretika wurden nicht erwähnt |
| OriginalDC | P | Nachkontrolle in einer Woche wurde nicht erwähnt |
| OriginalDC+Noise | A | Differenzialdiagnose Bronchitis wird genannt, obwohl im Transkript nicht erwähnt. |
| OriginalDC+Noise | A | Differenzialdiagnose Pneumonie wird genannt, obwohl im Transkript nicht erwähnt. |
| OriginalDC+Noise | P | Röntgenaufnahme des Thorax wird als Maßnahme genannt, findet sich nicht im Transkript. |
| OriginalDC+Noise | P | Einleitung einer antibiotischen Therapie wird erwogen, findet sich nicht im Transkript. |
| OriginalDC+Noise | P | Nachkontrolle in einer Woche wird empfohlen, findet sich nicht im Transkript. |
| LapInMitte | A | Verdacht auf akute Bronchitis oder Pneumonie wird im Transkript nicht als Verdachtsdiagnose genannt. |
| LapInMitte | A | Differenzialdiagnose 'Virale Infektion der oberen Atemwege mit sekundärer bakterieller Superinfektion' kommt im Transkript nicht vor. |
| LapInMitte | P | Röntgenaufnahme des Thorax wird im Transkript nicht erwähnt. |
| LapInMitte | P | Blutuntersuchung mit CRP und Blutbild wird im Transkript nicht erwähnt. |
| LapInMitte | P | Antibiotische Therapie bei bakterieller Infektion wird im Transkript nicht erwähnt. |
| LapInMitte | P | Schleimlösende Mittel und Antipyretika werden im Transkript nicht erwähnt. |
| LapInMitte | P | Beratung über Flüssigkeitszufuhr und Ruhe wird im Transkript nicht erwähnt. |
| LapInMitte | P | Follow-up-Termin in einer Woche wird im Transkript nicht erwähnt. |
| LapBeiArzt | A | Verdacht auf bakterielle Infektion der unteren Atemwege wird im Transkript nicht genannt. |
| LapBeiArzt | A | Differenzialdiagnose virale Infektion mit sekundärer bakterieller Superinfektion kommt im Transkript nicht vor. |
| LapBeiArzt | A | Verdacht auf Bronchitis oder Pneumonie wird im Transkript nicht geäußert. |
| LapBeiArzt | P | Röntgen-Thorax wird im Transkript nicht erwähnt. |
| LapBeiArzt | P | Erwägung einer antibiotischen Therapie kommt im Transkript nicht vor. |
| LapBeiArzt | P | Empfehlung zur Fortsetzung von Kamillentee und Hustenbonbons wird vom Arzt nicht ausgesprochen (nur vom Patienten erwähnt). |
| LapBeiArzt | P | Reevaluation nach 3–5 Tagen wird im Transkript nicht genannt. |
| Selbstkorrekturen | S | Ibuprofen 600 mg wird als 'leichte Besserung' dokumentiert, obwohl der Arzt die Wirkung im Transkript nur als 'fraglich' notiert und der Patient keine Besserung bestätigt hat |
| Selbstkorrekturen | A | 'Migräne mit Aura' – Aura-Symptome (z.B. Sehstörungen) wurden im Transkript an keiner Stelle erwähnt |
| Selbstkorrekturen | P | Verschreibung eines Triptans – im Gespräch nicht besprochen |
| Selbstkorrekturen | P | Kopfschmerztagebuch – im Gespräch nicht erwähnt |
| Selbstkorrekturen | P | Nachkontrolle in einer Woche – im Gespräch nicht vereinbart |
| Selbstkorrekturen | P | Empfehlung zur Triggervermeidung (grelles Licht) – im Gespräch nicht besprochen |
| Selbstkorrekturen | P | Neurologische Untersuchung – im Gespräch nicht angesprochen |
| Unterbrechungen | A | Salmonellen als spezifischer Erreger werden im Transkript nicht erwähnt |
| Unterbrechungen | P | Körperliche Untersuchung des Abdomens – im Transkript nicht erwähnt |
| Unterbrechungen | P | Labordiagnostik (Blutbild, CRP, Elektrolyte, Stuhlprobe) – im Transkript nicht erwähnt |
| Unterbrechungen | P | Antiemetika und Flüssigkeitszufuhr – im Transkript nicht erwähnt |
| Unterbrechungen | P | Stationäre Aufnahme bei Verschlechterung – im Transkript nicht erwähnt |
| Gedankensprünge | S | Schulter und Arm werden als 'rechts' bezeichnet — welche Seite betroffen ist, wird im Transkript nicht genannt |
| Gedankensprünge | O | Schulter und Arm werden als 'rechts' bezeichnet — welche Seite betroffen ist, wird im Transkript nicht genannt |
| Gedankensprünge | P | Ultraschalluntersuchung der Schulter — im Transkript nicht erwähnt |
| Gedankensprünge | P | MRT der Schulter — im Transkript nicht erwähnt |
| Gedankensprünge | P | NSAR (Ibuprofen 400 mg, 3x täglich) — im Transkript nicht erwähnt |
| Gedankensprünge | P | Schonung und Vermeidung belastender Aktivitäten — im Transkript nicht erwähnt |
| Gedankensprünge | P | Physiotherapie — im Transkript nicht erwähnt |
| Gedankensprünge | P | Nachkontrolle in 1 Woche — im Transkript nicht erwähnt |
| Meinungswechsel | P | Empfehlung zur Anpassung der Ernährung und Vermeidung säurehaltiger Lebensmittel wurde im Transkript nicht ausgesprochen. |
| Meinungswechsel | P | Beratung zur Vermeidung von Liegen nach dem Essen wurde im Transkript nicht erwähnt. |
| Meinungswechsel | P | Erwägung einer medikamentösen Therapie mit Protonenpumpenhemmern (PPI) kommt im Transkript nicht vor. |
| Meinungswechsel | P | Empfehlung zur Beobachtung der Symptome und Rücksprache bei Verschlechterung wurde im Transkript nicht genannt. |
| Chaos | S | Angst vor Tumor auf 'familiäre Erfahrungen' zurückgeführt – im Transkript war es eine Bekannte ('die Frau von der Kopfbewegung'), kein Familienmitglied. |
| Chaos | P | 'Beobachtung der Symptome nach der Ohrenspülung' wird als Maßnahme genannt, ist im Transkript jedoch nicht erwähnt – der Arzt sagt lediglich, der Spuk sei dann vorbei. |
| Anamnesegespräch | S | SOAP dokumentiert 'kein Drogenkonsum', obwohl Patientin explizit gelegentliche Ritalin-Einnahme (2–3 Mal) eingeräumt hat |
| Anamnesegespräch | P | Schmerzmanagement mit Analgetika wurde im Transkript vom Arzt nicht angesprochen oder angeordnet |

**Auslassungen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | S | Keine Ruheatemnot ("keine Rohdyspnoe") wurde vom Arzt explizit notiert, fehlt in der SOAP |
| OriginalDC+Noise | A | Arzt-Notiz erwähnt explizit 'Verdacht auf ein rezidivierendes Fieber', fehlt in der SOAP. |
| Unterbrechungen | S | Patientin musste wegen der Krämpfe sofort zur Toilette – möglicher Hinweis auf Diarrhö/Stuhldrang fehlt |
| Gedankensprünge | S | Zeitangabe 'vorgestern' als Beginn der Beschwerden fehlt |
| Anamnesegespräch | S | Familienanamnese fehlt vollständig (Großvater: Leberzirrhose verstorben; Großmutter: Brustkrebs; Schwester: Asthma und Neurodermitis) |
| Anamnesegespräch | S | Reiseanamnese fehlt (Singapur, geschäftlich, vor 2 Monaten, 2 Wochen Dauer) |
| Anamnesegespräch | S | Raucheranamnese fehlt (6–7 Jahre geraucht, vor 8 Jahren aufgehört) |
| Anamnesegespräch | S | Alkoholkonsum stark untertrieben: 'gelegentlich' statt 'fast jeden Abend ein Glas Wein plus 2–3 Gläser am Wochenende' |
| PWC | S | Erster Krankenhausbesuch mit Röntgen war negativ (Arzt fand nichts) – erst beim zweiten Besuch am nächsten Tag wurde der vordere Kreuzbandriss korrekt diagnostiziert. |
| PWC | S | Stationäre physiotherapeutische Behandlung im Krankenhaus (Krückengang, Abrollen des Fußes, Treppensteigen üben) wird nicht erwähnt. |
| PWC | S | Volleyball-Vorgeschichte (seit 4 Jahren, zweimal wöchentlich im Verein) fehlt als Kontext für das Therapieziel. |
| PWC | O | Ausgeprägter Muskelabbau durch die Liegezeit im Krankenhaus wird nicht erwähnt. |
| PWC | O | Patientin verlässt die Wohnung kaum – minimale Alltagsmobilität außerhalb der Wohnung nicht dokumentiert. |
| PWC | P | Im Transkript wurde kein konkreter Therapieplan besprochen – die SOAP formuliert einen generischen Plan, der zwar plausibel ist, aber keine im Gespräch vereinbarten Maßnahmen wiedergibt. |

---

### Whisper+Sauerkraut

| Szenario | S | O | A | P | Gesamt | Urteil |
|---|---|---|---|---|---|---|
| OriginalDC | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| OriginalDC+Noise | 1 | 1 | 0 | 0 | **2/8** | ✗ nicht_verwendbar |
| LapInMitte | 1 | 2 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| LapBeiArzt | 2 | 1 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| Selbstkorrekturen | 1 | 2 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| Unterbrechungen | 1 | 2 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| Gedankensprünge | 0 | 2 | 2 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| Meinungswechsel | 1 | 2 | 2 | 0 | **5/8** | ⚠ ueberarbeitung_noetig |
| Chaos | 0 | 2 | 0 | 2 | **4/8** | ⚠ ueberarbeitung_noetig |
| Anamnesegespräch | — | — | — | — | **auto** | ✗ nicht_verwendbar (`LLM-Fehlerrate 93% > 90%`) |
| PWC | 0 | 0 | 0 | 1 | **1/8** | ✗ nicht_verwendbar |

**Halluzinationen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | A | Verdacht auf Pneumonie wird explizit genannt, obwohl der Arzt im Transkript keine Pneumonie-Diagnose äußert. |
| OriginalDC | A | Einschätzung 'bakterielle oder viral-bakterielle Infektion' kommt im Transkript nicht vor. |
| OriginalDC | A | Aussage, dass genauere Diagnose durch weitere Untersuchungen gestellt werden kann, wird im Transkript nicht besprochen. |
| OriginalDC | P | Röntgenaufnahme der Lunge als geplante Maßnahme wird im Transkript nicht erwähnt. |
| OriginalDC | P | Bluttests werden im Transkript nicht besprochen. |
| OriginalDC | P | Sputumkultur wird im Transkript nicht erwähnt. |
| OriginalDC | P | Beginn einer Antibiotika- oder antiviralen Therapie wird im Transkript nicht angesprochen. |
| OriginalDC | P | Empfehlung eines Hustenstillstandsmedikaments kommt im Transkript nicht vor. |
| OriginalDC | P | Bewegungsempfehlungen und Schlafhygiene zur Vermeidung von Belastungsdyspnoe werden im Transkript nicht besprochen. |
| OriginalDC+Noise | A | Chronische Bronchitis als Differenzialdiagnose — im Transkript nicht erwähnt |
| OriginalDC+Noise | A | Infektion als Ursache der Temperaturen — wird im Transkript nicht als Diagnose genannt |
| OriginalDC+Noise | P | Weitere Laborwerte — im Transkript nicht besprochen |
| OriginalDC+Noise | P | Hustenstillende Medikamente als Therapiebeginn — nicht erwähnt |
| OriginalDC+Noise | P | Empfehlung eines Folge-Arzttermins — nicht im Transkript |
| OriginalDC+Noise | P | Ratschlag zum Vermeiden von Hausmitteln — im Transkript nicht geäußert |
| LapInMitte | A | Die Auskultation habe Befunde geliefert, die auf eine Erkrankung des unteren Atemwegs hinweisen — im Transkript wurde die Auskultation nur begonnen, kein Befund ist dokumentiert |
| LapInMitte | P | Röntgenaufnahme der Lunge angeordnet — im Transkript nicht erwähnt |
| LapInMitte | P | Antibiotikatherapie in Betracht gezogen — im Transkript nicht erwähnt |
| LapInMitte | P | Nachsorgeuntersuchung in 3–5 Tagen — im Transkript nicht erwähnt |
| LapInMitte | P | Ärztliche Empfehlung, Hausmittel weiterzunehmen und sich hinzulegen — im Transkript nicht als ärztliche Anweisung ausgesprochen |
| LapBeiArzt | A | Diagnose 'möglicherweise bakterielle Pneumonie' wurde vom Arzt im Transkript nicht gestellt |
| LapBeiArzt | P | Röntgenaufnahme der Lunge wurde im Transkript nicht angeordnet |
| LapBeiArzt | P | Bakteriologische Sputumuntersuchung wurde nicht erwähnt |
| LapBeiArzt | P | Antibiotikagabe wurde nicht erwähnt |
| LapBeiArzt | P | Fortgesetzte Überwachung von Temperatur und Dyspnoe wurde nicht besprochen |
| Selbstkorrekturen | A | Halluzinierte Ursache: 'Entzündung oder Stress' – im Transkript nicht erwähnt |
| Selbstkorrekturen | A | Halluzinierte Diagnose: 'Migräneanfall' – Arzt stellt im Transkript keine Diagnose |
| Selbstkorrekturen | A | Fehlerhafte Begründung: 'Schmerzen breiten sich von einer Seite zur anderen aus' – Patient korrigierte sich selbst, der Schmerz ist einseitig rechts |
| Selbstkorrekturen | P | Halluzinierte Maßnahme: Blutuntersuchung auf Entzündungszeichen – im Transkript nicht erwähnt |
| Selbstkorrekturen | P | Halluzinierte Maßnahme: Patient soll sich am nächsten Tag melden – kein solcher Plan im Transkript |
| Selbstkorrekturen | P | Halluzinierte Empfehlung: andere/weitere Schmerzmittel einnehmen – nicht im Transkript besprochen |
| Unterbrechungen | S | Schmerz strahlt in den Leistenbereich aus — Patientin sagte nur 'überall', bestätigte keine Leistenausstrahlung explizit |
| Unterbrechungen | A | Reizdarmschleimhautentzündung als mögliche Diagnose — kommt im Transkript nicht vor |
| Unterbrechungen | A | Magengeschwüre als mögliche Diagnose — kommt im Transkript nicht vor |
| Unterbrechungen | P | Bluttests — im Transkript nicht erwähnt |
| Unterbrechungen | P | Ultraschalluntersuchung — im Transkript nicht erwähnt |
| Unterbrechungen | P | Antibiotika-Behandlung — im Transkript nicht erwähnt |
| Unterbrechungen | P | Hydratationsempfehlung — im Transkript nicht erwähnt |
| Unterbrechungen | P | Ernährungsumstellung — im Transkript nicht erwähnt |
| Unterbrechungen | P | Stationäre Behandlung — im Transkript nicht erwähnt |
| Gedankensprünge | S | Die SOAP beschreibt 2014 als 'ähnliche Erkrankung nach einem Skiunfall', im Transkript war das 2014er Ereignis jedoch ein Kreuzbandriss am Knie – keine Schultererkrankung |
| Gedankensprünge | P | MRI-Untersuchung im Transkript nicht erwähnt |
| Gedankensprünge | P | Physiotherapie im Transkript nicht erwähnt |
| Gedankensprünge | P | Mögliche operative Behandlung im Transkript nicht erwähnt |
| Gedankensprünge | P | Empfehlung, das Tennisspielen einzustellen, im Transkript nicht erwähnt |
| Gedankensprünge | P | Regelmäßige Überwachung und Nachsorge im Transkript nicht erwähnt |
| Meinungswechsel | P | Empfehlung zur Endoskopie – im Transkript nicht erwähnt |
| Meinungswechsel | P | Medikamentöse Behandlung mit Protonenpumpenhemmern – im Transkript nicht erwähnt |
| Meinungswechsel | P | Lebensstiländerungen (Alkohol, Koffein, Gewicht) – im Transkript nicht erwähnt |
| Meinungswechsel | P | Ausführliche Anamnese als geplante Maßnahme – im Transkript nicht erwähnt |
| Chaos | S | Schmerzen im 'linken Ohr' — Transkript: Patient korrigiert sich explizit auf rechtes Ohr ('mein rechtes Ohr, das tut auch weh') |
| Chaos | S | 'aufgrund einer früheren Operation an der Akustikusnerven' — nicht der Patient, sondern eine andere Person hatte das Akustikusneurinom und die OP |
| Chaos | A | 'Neuritis vestibularis' als Diagnosebestandteil — Arzt hat diesen Verdacht explizit revidiert ('Verdacht auf Neuritis vestibularis revidiert'); Diagnose ist Zerumen-Pfropf |
| PWC | S | SOAP behauptet die Diagnose wurde 'nach einer Woche im Krankenhaus' gestellt – im Transkript wurde sie am nächsten Morgen beim zweiten Krankenhausbesuch gestellt; die Woche im KH war post-operativ. |
| PWC | S | SOAP sagt sie konnte direkt nach der Verletzung nicht aufstehen – im Transkript war das Aufsteh-Problem erst am nächsten Morgen; direkt nach der Verletzung konnte sie nur das Knie nicht ausstrecken. |
| PWC | A | Wiederholt fälschlicherweise, die Diagnose sei 'nach einer Woche im Krankenhaus' gestellt worden – widerspricht dem Transkript direkt. |

**Auslassungen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC+Noise | S | Medikamenten-Anamnese fehlt: keine spezifische Medikation, lediglich Hausmittel |
| OriginalDC+Noise | S | Keine Ruhedyspnö nicht erwähnt |
| OriginalDC+Noise | O | Keine Ruhedyspnö fehlt als negativer Befund |
| LapInMitte | S | Vormedikation fehlt: Patientin nimmt Hustenbonbons und Kamillentee — nicht in S erwähnt |
| LapBeiArzt | O | Die begonnene Auskultation wird nicht dokumentiert; die O-Sektion wiederholt nur subjektive Patientenangaben anstatt objektiver Arztbefunde |
| Selbstkorrekturen | S | Extreme Lichtempfindlichkeit des Patienten fehlt komplett |
| Selbstkorrekturen | S | Schmerz zieht vom Nacken hoch – SOAP beschreibt fälschlich eine Ausbreitung von der rechten Schläfe aus |
| Selbstkorrekturen | S | Explizite Verneinung von Übelkeit (kein Erbrechen) fehlt |
| Unterbrechungen | S | Patientin musste um 3 Uhr nachts zur Toilette rennen (mögliche Durchfallsymptomatik) |
| Unterbrechungen | S | Nahrungsanamnese fehlt: Salat mittags, gegrilltes Hähnchen abends, möglicherweise nicht durchgegart |
| Unterbrechungen | S | Erbrechen zweimal — SOAP erwähnt nur Erbrechen, nicht die Häufigkeit |
| Unterbrechungen | A | Hinweis auf möglicherweise nicht durchgegarten Hähnchen als naheliegender Auslöser (Lebensmittelvergiftung) fehlt komplett |
| Gedankensprünge | S | Patient kann den Arm nicht über Kopfhöhe heben ('blockiert richtig') |
| Meinungswechsel | S | Spezifischer Auslöser 'Tomatensauce gestern Abend' wird nicht erwähnt |
| Chaos | S | Zeitangabe falsch: Sonntag morgen (nicht Sonntagabend) |
| PWC | S | Erster Krankenhausbesuch am Verletzungstag: Röntgen gemacht, Arzt fand nichts Auffälliges. |
| PWC | S | Kühlung und Hochlagerung unmittelbar nach der Verletzung. |
| PWC | S | Knie am nächsten Morgen stark angeschwollen. |
| PWC | S | Es ist ihr erster Physiotherapiebesuch überhaupt. |
| PWC | O | Volleyballspielerin seit 4 Jahren, Training 2x pro Woche. |
| PWC | O | Wohnsituation: 2. Stock, ca. 20–30 Treppen (therapeutisch relevant). |
| PWC | O | Muskelabbau durch Immobilisation während des Krankenhausaufenthalts. |
| PWC | O | Physiotherapeutische Behandlung bereits im Krankenhaus (einmal täglich). |
| PWC | O | Schmerztabletten wurden eingenommen, werden zum Gesprächszeitpunkt aber nicht mehr genommen. |
| PWC | P | Treppensteigen (2. Stock, 20–30 Stufen) als konkretes kurzfristiges Rehabilitationsziel wird nicht genannt. |

---

### Whisper+gemma4

| Szenario | S | O | A | P | Gesamt | Urteil |
|---|---|---|---|---|---|---|
| OriginalDC | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| OriginalDC+Noise | 1 | 1 | 0 | 0 | **2/8** | ✗ nicht_verwendbar |
| LapInMitte | 2 | 2 | 0 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| LapBeiArzt | 1 | 2 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| Selbstkorrekturen | 2 | 1 | 0 | 0 | **3/8** | ✗ nicht_verwendbar |
| Unterbrechungen | 1 | 2 | 2 | 0 | **5/8** | ⚠ ueberarbeitung_noetig |
| Gedankensprünge | 0 | 1 | 0 | 0 | **1/8** | ✗ nicht_verwendbar |
| Meinungswechsel | 2 | 2 | 2 | 0 | **6/8** | ⚠ ueberarbeitung_noetig |
| Chaos | 1 | 2 | 1 | 0 | **4/8** | ⚠ ueberarbeitung_noetig |
| Anamnesegespräch | 1 | 1 | 0 | 0 | **2/8** | ✗ nicht_verwendbar |
| PWC | — | — | — | — | **auto** | ✗ nicht_verwendbar (`SOAP ist leer`) |

**Halluzinationen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | A | Diagnose 'Bronchitis/Pneumonie' wird im Transkript nicht genannt oder angedeutet |
| OriginalDC | P | Röntgen-Thorax zur Pneumonieabklärung wurde im Transkript nicht erwähnt |
| OriginalDC | P | Symptomatische Behandlung von Husten und Dyspnoe wurde nicht besprochen |
| OriginalDC | P | Medikationsanpassung wurde nicht thematisiert |
| OriginalDC | P | Sputumkultur zur Abklärung bakterieller Infektion wurde nicht erwähnt |
| OriginalDC+Noise | S | Sputum wird als 'verfärbt' beschrieben – das Transkript nennt 'kudelnd', was keine Farbangabe ist |
| OriginalDC+Noise | O | 'Patientin ist bei Bewusstsein und kooperativ' – im Transkript nicht erwähnt |
| OriginalDC+Noise | A | 'Akute Infektion der Atemwege' als Diagnose – im Transkript nicht genannt; dort wird nur Verdacht auf präzidivierende Tuberkulose dokumentiert |
| OriginalDC+Noise | P | Weiterführende Diagnostik (Lungenfunktionsprüfung, Röntgen-Thorax, Sputumkultur) – im Transkript nicht besprochen |
| OriginalDC+Noise | P | Symptomatische Behandlung von Dyspnö und Husten – im Transkript nicht besprochen |
| OriginalDC+Noise | P | Engmaschige Überwachung und Follow-up – im Transkript nicht besprochen |
| LapInMitte | A | Verdacht auf akute Bronchitis oder Pneumonie — im Transkript nicht genannt. |
| LapInMitte | A | Einordnung als bakterielle oder virale Atemwegsinfektion — im Transkript nicht erwähnt. |
| LapInMitte | P | Röntgen-Thorax zur Diagnostik — im Transkript nicht erwähnt. |
| LapInMitte | P | Sputumkultur zur Erregeridentifikation — im Transkript nicht erwähnt. |
| LapInMitte | P | Antibiotische Therapie als mögliche Maßnahme — im Transkript nicht erwähnt. |
| LapInMitte | P | Empfehlung ausreichender Flüssigkeitszufuhr als Arztanweisung — im Transkript kein Hinweis darauf als ärztliche Empfehlung. |
| LapBeiArzt | S | Tee als getrunkenes Mittel explizit genannt, im Transkript steht nur sinngemäß 'viel trinken/Flüssigkeit' (STT-Fehler: 'vielfamilien'). |
| LapBeiArzt | A | Verdacht auf Bronchitis/Pneumonie diagnostiziert, obwohl im Transkript ausschließlich 'Verdacht auf rezidivierendes Fieber' vom Arzt notiert wurde. |
| LapBeiArzt | P | Röntgen-Thorax und Sputumkultur empfohlen, obwohl im Transkript keinerlei Diagnostikplan besprochen wurde. |
| LapBeiArzt | P | Antibiotische Therapie erwähnt, findet im Transkript keine Grundlage. |
| LapBeiArzt | P | Empfehlung zur Rückkonsultation bei Verschlechterung ist nicht im Transkript enthalten. |
| Selbstkorrekturen | O | 'Klinische Untersuchung zeigt ausgeprägte Photophobie' — die Photophobie ist ein selbst berichtetes Symptom des Patienten, keine Untersuchungsfeststellung; eine klinische Untersuchung ist im Transkript nicht dokumentiert. |
| Selbstkorrekturen | A | Diagnosevorschlag 'Migräne' ist im Transkript nicht genannt. |
| Selbstkorrekturen | A | Diagnosevorschlag 'sekundärer Spannungskopfschmerz' ist im Transkript nicht genannt. |
| Selbstkorrekturen | P | Empfehlung abortiver Medikamente (z.B. Triptane) findet sich nicht im Transkript. |
| Selbstkorrekturen | P | Empfehlung zur Trigger-Vermeidung findet sich nicht im Transkript. |
| Selbstkorrekturen | P | Empfehlung zur Wiedervorstellung bei Verschlechterung findet sich nicht im Transkript. |
| Selbstkorrekturen | P | Aufklärung über Photophobie und Schmerzanamnese findet sich nicht im Transkript. |
| Unterbrechungen | S | Durchfall wird explizit genannt, obwohl im Transkript nur 'auf die Toilette rennen musste' steht – Diarrhö ist nicht ausdrücklich bestätigt |
| Unterbrechungen | P | ORS (orale Rehydrationslösung) wird empfohlen – im Transkript nicht erwähnt |
| Unterbrechungen | P | Schonkost (Zwieback, Reis) wird empfohlen – im Transkript nicht erwähnt |
| Unterbrechungen | P | Stuhlprobe als diagnostische Maßnahme genannt – im Transkript nicht erwähnt |
| Unterbrechungen | P | Follow-up-Kriterien (anhaltendes Fieber, Dehydratation) formuliert – im Transkript nicht erwähnt |
| Gedankensprünge | S | Patient habe 2014 'ähnliche Schulterprobleme, die chirurgisch behandelt wurden' — im Transkript verneint der Patient Schulterprobleme 2014 explizit ('Nein, nein') und beschreibt stattdessen einen Kreuzbandriss am Knie. |
| Gedankensprünge | O | 'Schmerz bei passiven und aktiven Bewegungen' — im Transkript wird nur eine aktive Bewegungseinschränkung (Armbeben) besprochen; passive Bewegungstests wurden nicht durchgeführt oder erwähnt. |
| Gedankensprünge | A | Differentialdiagnosen Impingement-Syndrom und Tendinitis werden genannt, obwohl diese im Transkript keinerlei Erwähnung finden. |
| Gedankensprünge | P | Bildgebende Diagnostik (Ultraschall/MRT) wurde im Transkript nicht besprochen. |
| Gedankensprünge | P | Schmerzmanagement und lokale Kühlung wurden nicht erwähnt. |
| Gedankensprünge | P | Physiotherapeutische Nachsorge wurde nicht erwähnt. |
| Gedankensprünge | P | Aufklärung über Belastungsgrenzen wurde nicht erwähnt. |
| Meinungswechsel | P | Beginn einer Protonenpumpenhemmer-Therapie (PPI) — im Transkript nicht erwähnt |
| Meinungswechsel | P | Ernährungsberatung zur Identifizierung von Reflux-Auslösern — im Transkript nicht erwähnt |
| Meinungswechsel | P | Empfehlung zur Vermeidung von flachem Liegen nach dem Essen — im Transkript nicht erwähnt |
| Meinungswechsel | P | Keine weiteren kardialen Untersuchungen indiziert — im Transkript nicht explizit als Maßnahme formuliert |
| Chaos | S | SOAP schreibt dem Patienten selbst eine frühere Akustikusneurinom-Operation zu; im Transkript bezieht sich 'Die' auf eine dritte Person (Bekannte/Verwandte), nicht auf den Patienten |
| Chaos | A | 'Obstruktion des Mittelohrs' kommt im Transkript nicht vor; der Cerumenpfropf sitzt im Außengehörgang, nicht im Mittelohr |
| Chaos | P | 'Bei anhaltenden oder sich verschlimmernden Symptomen ist eine Nachkontrolle erforderlich' wird im Transkript an keiner Stelle erwähnt – halluzinierte Maßnahme |
| Anamnesegespräch | A | Diagnose 'Kniearthrose' wird im Transkript nicht gestellt – der Arzt nennt keine Diagnose |
| Anamnesegespräch | A | Differenzialdiagnosen (entzündliche Arthritis, muskuloskelettale Überlastung) sind im Transkript nicht erwähnt |
| Anamnesegespräch | P | Bildgebende Diagnostik (Röntgenaufnahme) wird im Transkript nicht angeordnet |
| Anamnesegespräch | P | Physiotherapie wird im Transkript nicht empfohlen |
| Anamnesegespräch | P | Medikamentöse Therapieanpassung oder Schmerzmedikation wird nicht besprochen |
| Anamnesegespräch | P | Gewichtsmanagement und Belastungsanpassung werden im Transkript nicht thematisiert |
| Anamnesegespräch | P | Nachkontrolltermin wird im Transkript nicht vereinbart |

**Auslassungen:**

| Szenario | Sektion | Beschreibung |
|---|---|---|
| OriginalDC | A | Der Arzt notiert im Transkript 'Verdacht auf rezidivierendes Fieber' als Arbeitsdiagnose — fehlt in der SOAP |
| LapInMitte | A | Arzt-Notiz 'Verdacht auf rezidivierendes Fieber' aus dem Transkript fehlt. |
| LapBeiArzt | A | 'Verdacht auf rezidivierendes Fieber' aus dem Transkript fehlt vollständig in der Assessment-Sektion. |
| Unterbrechungen | S | Patientin sagte 'es tut einfach überall weh', was die lokale Eingrenzung 'primär über den Bauchnabel' etwas relativiert |
| Gedankensprünge | S | Das laute Knacken bei Schulter beim Aufschlag wird nicht erwähnt. |
| Chaos | S | Patient erwähnt zwei kleine Kinder zu Hause als emotionalen Kontext für seine Karzinophobie – in der SOAP nicht dokumentiert |
| Anamnesegespräch | S | Ernährung nicht erwähnt (Patient versucht ausgewogen zu essen, isst manchmal etwas zu viel) |
| Anamnesegespräch | S | Familienstand fehlt (verheiratet, zwei Kinder) |
| Anamnesegespräch | S | Schlafdauer (7–8 Stunden) nicht angegeben, nur 'gut' dokumentiert |
| Anamnesegespräch | O | Schmerzlokalisation nicht dokumentiert – Patient zeigte dem Arzt die konkrete Schmerzstelle am Knie |

---

