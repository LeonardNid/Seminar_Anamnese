# WER-Analyse: AssemblyAI (Cloud) + GPT-4o

> Vollständige Fehleranalyse: STT-Rohausgabe vs. Ground Truth
> Satzzeichen und Groß-/Kleinschreibung werden ignoriert.
> **S** = Substitution, **D** = Löschung (fehlt im STT), **I** = Einfügung (extra im STT)

## Übersicht

| Audiodatei | Ref-Wörter | Hyp-Wörter | S | D | I | Edit-Dist | WER |
|-----------|-----------|-----------|---|---|---|-----------|-----|
| OriginalDC.m4a | 232 | 229 | 5 | 4 | 1 | 10 | **4.3%** |
| OriginalDCWhiteNoise.m4a | 229 | 221 | 42 | 21 | 13 | 76 | **33.2%** |
| OriginalLapInMitte.wav | 231 | 227 | 4 | 5 | 1 | 10 | **4.3%** |
| OriginalLapBeiArzt.wav | 226 | 226 | 4 | 1 | 1 | 6 | **2.7%** |
| SelbstkorrekturLapInMitte.wav | 183 | 187 | 10 | 1 | 5 | 16 | **8.7%** |
| UnterbrechungLapInMitte.wav | 153 | 150 | 5 | 9 | 6 | 20 | **13.1%** |
| GedankenprüngeLapInMitte.wav | 192 | 190 | 3 | 2 | 0 | 5 | **2.6%** |
| MeinungswechselLapinMitte.wav | 179 | 178 | 1 | 1 | 0 | 2 | **1.1%** |
| ChaosLapInMitte.wav | 272 | 260 | 17 | 22 | 10 | 49 | **18.0%** |
| Das Anamnesegespräch.wav | 2317 | 2299 | 65 | 28 | 10 | 103 | **4.4%** |
| Anamnesegesrpäch PWC.mp3 | 1530 | 1484 | 22 | 57 | 11 | 90 | **5.9%** |


---

## OriginalDC.m4a

**WER: 4.3%** — Referenz: 232 Wörter | Hypothese: 229 Wörter | S=5 D=4 I=1 | Edit-Distanz=10

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `sieben` | `7` | …leitsymptom husten seit [___] tagen verdacht auf… |
| 2 | Substitution | `grad` | `c` | …maximum 38 8 [___] celsius und wenn… |
| 3 | Löschung | `celsius` | *(fehlt)* | …38 8 grad [___] und wenn sie… |
| 4 | Substitution | `das` | `es` | …hoch oder ist [___] eher ein trockener… |
| 5 | Einfügung | *(nicht da)* | `auch` | …doll und muss [___] der hälfte kurz… |
| 6 | Substitution | `ruhedyspnoe` | `rohdyspnoe` | …belastungsdyspnoe aber keine [___] keine thorakalen schmerzen… |
| 7 | Löschung | `die` | *(fehlt)* | …ich werde jetzt [___] lunge abhören ihre… |
| 8 | Löschung | `lunge` | *(fehlt)* | …werde jetzt die [___] abhören ihre lunge… |
| 9 | Löschung | `abhören` | *(fehlt)* | …jetzt die lunge [___] ihre lunge abhören… |
| 10 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## OriginalDCWhiteNoise.m4a

**WER: 33.2%** — Referenz: 229 Wörter | Hypothese: 221 Wörter | S=42 D=21 I=13 | Edit-Distanz=76

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `huste` | `habe` | …herr doktor ich [___] seit knapp einer… |
| 2 | Substitution | `seit` | `das` | …doktor ich huste [___] knapp einer woche… |
| 3 | Substitution | `knapp` | `merke` | …ich huste seit [___] einer woche ununterbrochen… |
| 4 | Substitution | `einer` | `ich` | …huste seit knapp [___] woche ununterbrochen und… |
| 5 | Substitution | `woche` | `an` | …seit knapp einer [___] ununterbrochen und mir… |
| 6 | Substitution | `ununterbrochen` | `meiner` | …knapp einer woche [___] und mir ist… |
| 7 | Einfügung | *(nicht da)* | `haut` | …einer woche ununterbrochen [___] mir ist ständig… |
| 8 | Einfügung | *(nicht da)* | `meiner` | …woche ununterbrochen und [___] ist ständig heiß… |
| 9 | Einfügung | *(nicht da)* | `bauchdecke` | …woche ununterbrochen und [___] ist ständig heiß… |
| 10 | Substitution | `sieben` | `7` | …leitsymptom husten seit [___] tagen verdacht auf… |
| 11 | Einfügung | *(nicht da)* | `ein` | …tagen verdacht auf [___] fieber haben sie… |
| 12 | Substitution | `8` | `3` | …waren es 38 [___] grad auf dem… |
| 13 | Löschung | `auf` | *(fehlt)* | …38 8 grad [___] dem thermometer okay… |
| 14 | Löschung | `dem` | *(fehlt)* | …8 grad auf [___] thermometer okay notiz… |
| 15 | Löschung | `thermometer` | *(fehlt)* | …grad auf dem [___] okay notiz subfebrile… |
| 16 | Substitution | `8` | `5` | …temperaturen maximum 38 [___] grad celsius und… |
| 17 | Einfügung | *(nicht da)* | `ach` | …38 8 grad [___] und wenn sie… |
| 18 | Einfügung | *(nicht da)* | `38` | …38 8 grad [___] und wenn sie… |
| 19 | Einfügung | *(nicht da)* | `grad` | …38 8 grad [___] und wenn sie… |
| 20 | Substitution | `das` | `es` | …hoch oder ist [___] eher ein trockener… |
| 21 | Substitution | `da` | `also` | …ein trockener reizhusten [___] kommt richtig viel… |
| 22 | Substitution | `kommt` | `wenn` | …trockener reizhusten da [___] richtig viel hoch… |
| 23 | Einfügung | *(nicht da)* | `ich` | …reizhusten da kommt [___] viel hoch das… |
| 24 | Substitution | `viel` | `finde` | …da kommt richtig [___] hoch das ist… |
| 25 | Substitution | `hoch` | `ist` | …kommt richtig viel [___] das ist so… |
| 26 | Substitution | `das` | `es` | …richtig viel hoch [___] ist so dickflüssig… |
| 27 | Löschung | `ist` | *(fehlt)* | …viel hoch das [___] so dickflüssig und… |
| 28 | Substitution | `grün` | `also` | …und eher gelblich [___] würde sagen würde… |
| 29 | Löschung | `sagen` | *(fehlt)* | …gelblich grün würde [___] würde ich sagen… |
| 30 | Löschung | `würde` | *(fehlt)* | …grün würde sagen [___] ich sagen das… |
| 31 | Substitution | `echt` | `nicht` | …sagen das ist [___] eklig das ist… |
| 32 | Substitution | `eklig` | `englisch` | …das ist echt [___] das ist wichtig… |
| 33 | Substitution | `purulentem` | `kohlentem` | …produktiver husten mit [___] sputum haben sie… |
| 34 | Substitution | `schlechter` | `schlechte` | …gefühl dass sie [___] luft bekommen also… |
| 35 | Substitution | `weh` | `eigentlich` | …luft bekommen also [___] tut es in… |
| 36 | Substitution | `tut` | `ist` | …bekommen also weh [___] es in der… |
| 37 | Substitution | `es` | `mein` | …also weh tut [___] in der brust… |
| 38 | Substitution | `in` | `husten` | …weh tut es [___] der brust nicht… |
| 39 | Löschung | `der` | *(fehlt)* | …tut es in [___] brust nicht direkt… |
| 40 | Löschung | `brust` | *(fehlt)* | …es in der [___] nicht direkt aber… |
| 41 | Löschung | `wenn` | *(fehlt)* | …nicht direkt aber [___] ich die treppen… |
| 42 | Substitution | `die` | `treibe` | …aber wenn ich [___] treppen in den… |
| 43 | Substitution | `treppen` | `da` | …wenn ich die [___] in den zweiten… |
| 44 | Substitution | `in` | `ganz` | …ich die treppen [___] den zweiten stock… |
| 45 | Substitution | `den` | `gerne` | …die treppen in [___] zweiten stock hochlaufe… |
| 46 | Substitution | `zweiten` | `staub` | …treppen in den [___] stock hochlaufe schnaufe… |
| 47 | Substitution | `stock` | `hoch` | …in den zweiten [___] hochlaufe schnaufe ich… |
| 48 | Substitution | `hochlaufe` | `glaube` | …den zweiten stock [___] schnaufe ich schon… |
| 49 | Löschung | `schnaufe` | *(fehlt)* | …zweiten stock hochlaufe [___] ich schon ganz… |
| 50 | Löschung | `schon` | *(fehlt)* | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 51 | Löschung | `ganz` | *(fehlt)* | …schnaufe ich schon [___] schön doll und… |
| 52 | Löschung | `schön` | *(fehlt)* | …ich schon ganz [___] doll und muss… |
| 53 | Löschung | `doll` | *(fehlt)* | …schon ganz schön [___] und muss auf… |
| 54 | Löschung | `und` | *(fehlt)* | …ganz schön doll [___] muss auf der… |
| 55 | Löschung | `muss` | *(fehlt)* | …schön doll und [___] auf der hälfte… |
| 56 | Löschung | `auf` | *(fehlt)* | …doll und muss [___] der hälfte kurz… |
| 57 | Löschung | `der` | *(fehlt)* | …und muss auf [___] hälfte kurz anhalten… |
| 58 | Löschung | `hälfte` | *(fehlt)* | …muss auf der [___] kurz anhalten gut… |
| 59 | Löschung | `kurz` | *(fehlt)* | …auf der hälfte [___] anhalten gut dass… |
| 60 | Löschung | `anhalten` | *(fehlt)* | …der hälfte kurz [___] gut dass sie… |
| 61 | Substitution | `belastungsdyspnoe` | `belastungsdispnöe` | …erwähnen notiz deutliche [___] keine ruhedyspnoe keine… |
| 62 | Substitution | `ruhedyspnoe` | `ruhedispnöe` | …deutliche belastungsdyspnoe keine [___] keine thorakalen schmerzen… |
| 63 | Substitution | `lutsche` | `bin` | …medikamente dagegen ich [___] nur diese normalen… |
| 64 | Substitution | `nur` | `schon` | …dagegen ich lutsche [___] diese normalen hustenbonbons… |
| 65 | Substitution | `diese` | `mal` | …ich lutsche nur [___] normalen hustenbonbons aus… |
| 66 | Substitution | `normalen` | `gesund` | …lutsche nur diese [___] hustenbonbons aus der… |
| 67 | Substitution | `hustenbonbons` | `aber` | …nur diese normalen [___] aus der drogerie… |
| 68 | Einfügung | *(nicht da)* | `ich` | …diese normalen hustenbonbons [___] der drogerie und… |
| 69 | Einfügung | *(nicht da)* | `muss` | …diese normalen hustenbonbons [___] der drogerie und… |
| 70 | Einfügung | *(nicht da)* | `morgens` | …diese normalen hustenbonbons [___] der drogerie und… |
| 71 | Substitution | `drogerie` | `tür` | …hustenbonbons aus der [___] und trinke viel… |
| 72 | Einfügung | *(nicht da)* | `kriechen` | …aus der drogerie [___] trinke viel kamillentee… |
| 73 | Substitution | `kamillentee` | `wasser` | …und trinke viel [___] alles klar notiz… |
| 74 | Substitution | `vormedikation` | `formmedikation` | …notiz keine spezifische [___] lediglich supportive hausmittel… |
| 75 | Einfügung | *(nicht da)* | `so` | …atmen sie dafür [___] durch den mund… |
| 76 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## OriginalLapInMitte.wav

**WER: 4.3%** — Referenz: 231 Wörter | Hypothese: 227 Wörter | S=4 D=5 I=1 | Edit-Distanz=10

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `sieben` | `7` | …leitsymptom husten seit [___] tagen verdacht auf… |
| 2 | Substitution | `das` | `es` | …hoch oder ist [___] eher ein trockener… |
| 3 | Löschung | `das` | *(fehlt)* | …das ist so [___] ist so dickflüssig… |
| 4 | Löschung | `ist` | *(fehlt)* | …ist so das [___] so dickflüssig dickflüssig… |
| 5 | Löschung | `so` | *(fehlt)* | …so das ist [___] dickflüssig dickflüssig und… |
| 6 | Löschung | `dickflüssig` | *(fehlt)* | …ist so dickflüssig [___] und eher gelblich… |
| 7 | Löschung | `schon` | *(fehlt)* | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 8 | Einfügung | *(nicht da)* | `auch` | …doll und muss [___] der hälfte kurz… |
| 9 | Substitution | `belastungsdyspnoe` | `belastungsdispnoe` | …erwähnen notiz deutliche [___] keine ruhedyspnoe keine… |
| 10 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## OriginalLapBeiArzt.wav

**WER: 2.7%** — Referenz: 226 Wörter | Hypothese: 226 Wörter | S=4 D=1 I=1 | Edit-Distanz=6

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `sieben` | `7` | …leitsymptom husten seit [___] tagen verdacht auf… |
| 2 | Substitution | `zweiten` | `2` | …treppen in den [___] stock hochlaufe schnaufe… |
| 3 | Löschung | `schon` | *(fehlt)* | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 4 | Substitution | `auf` | `auch` | …doll und muss [___] der hälfte kurz… |
| 5 | Einfügung | *(nicht da)* | `in` | …und muss auf [___] hälfte kurz anhalten… |
| 6 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## SelbstkorrekturLapInMitte.wav

**WER: 8.7%** — Referenz: 183 Wörter | Hypothese: 187 Wörter | S=10 D=1 I=5 | Edit-Distanz=16

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `ne` | `nee` | …habe seit dienstag [___] warten sie mal… |
| 2 | Löschung | `äh` | *(fehlt)* | …linke schläfe aus [___] quatsch ich zeig… |
| 3 | Substitution | `zeig` | `zeige` | …äh quatsch ich [___] s gerade falsch… |
| 4 | Substitution | `s` | `es` | …quatsch ich zeig [___] gerade falsch rum… |
| 5 | Einfügung | *(nicht da)* | `und` | …aber nicht geholfen [___] hab ich am… |
| 6 | Substitution | `hab` | `habe` | …nicht geholfen dann [___] ich am nachmittag… |
| 7 | Einfügung | *(nicht da)* | `habe` | …vor dem schlaf [___] eine ibuprofen 400… |
| 8 | Einfügung | *(nicht da)* | `ich` | …vor dem schlaf [___] eine ibuprofen 400… |
| 9 | Substitution | `ibuprofen` | `ibuprofene` | …schlaf noch eine [___] 400 genommen oder… |
| 10 | Substitution | `waren` | `war` | …400 genommen oder [___] das 600 die… |
| 11 | Substitution | `das` | `es` | …genommen oder waren [___] 600 die rosafarbenen… |
| 12 | Einfügung | *(nicht da)* | `doch` | …oder waren das [___] die rosafarbenen aus… |
| 13 | Einfügung | *(nicht da)* | `die` | …oder waren das [___] die rosafarbenen aus… |
| 14 | Substitution | `rosafarbenen` | `rosafarbene` | …das 600 die [___] aus der großen… |
| 15 | Substitution | `pochender` | `wochenlanger` | …davon gut notiz [___] kopfschmerz rechtsseitig seit… |
| 16 | Substitution | `rechtsseitig` | `rechtseitig` | …notiz pochender kopfschmerz [___] seit mittwoch keine… |


---

## UnterbrechungLapInMitte.wav

**WER: 13.1%** — Referenz: 153 Wörter | Hypothese: 150 Wörter | S=5 D=9 I=6 | Edit-Distanz=20

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `drei` | `3` | …ich bin um [___] uhr aufgewacht und… |
| 2 | Löschung | `wo` | *(fehlt)* | …dass ich direkt [___] genau krampft es… |
| 3 | Löschung | `genau` | *(fehlt)* | …ich direkt wo [___] krampft es denn… |
| 4 | Löschung | `krampft` | *(fehlt)* | …direkt wo genau [___] es denn auf… |
| 5 | Löschung | `es` | *(fehlt)* | …wo genau krampft [___] denn auf die… |
| 6 | Löschung | `denn` | *(fehlt)* | …genau krampft es [___] auf die toilette… |
| 7 | Einfügung | *(nicht da)* | `wo` | …toilette rennen musste [___] ist direkt über… |
| 8 | Einfügung | *(nicht da)* | `genau` | …toilette rennen musste [___] ist direkt über… |
| 9 | Einfügung | *(nicht da)* | `krampft` | …toilette rennen musste [___] ist direkt über… |
| 10 | Einfügung | *(nicht da)* | `s` | …toilette rennen musste [___] ist direkt über… |
| 11 | Einfügung | *(nicht da)* | `denn` | …toilette rennen musste [___] ist direkt über… |
| 12 | Löschung | `da` | *(fehlt)* | …ja heute morgen [___] war es bei… |
| 13 | Einfügung | *(nicht da)* | `5` | …es bei 38 [___] 38 was komma… |
| 14 | Substitution | `zwei` | `2` | …38 was komma [___] komma fünf komma… |
| 15 | Substitution | `fünf` | `5` | …komma zwei komma [___] komma drei und… |
| 16 | Substitution | `drei` | `3` | …komma fünf komma [___] und ich habe… |
| 17 | Löschung | `mit` | *(fehlt)* | …haben wir gegrillt [___] den nachbarn vielleicht… |
| 18 | Löschung | `den` | *(fehlt)* | …wir gegrillt mit [___] nachbarn vielleicht etwas… |
| 19 | Löschung | `nachbarn` | *(fehlt)* | …gegrillt mit den [___] vielleicht etwas vom… |
| 20 | Substitution | `das` | `was` | …etwas vom grill [___] nicht ganz durch… |


---

## GedankenprüngeLapInMitte.wav

**WER: 2.6%** — Referenz: 192 Wörter | Hypothese: 190 Wörter | S=3 D=2 I=0 | Edit-Distanz=5

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `vier` | `4` | …da lag ich [___] wochen flach dr… |
| 2 | Substitution | `ne` | `eine` | …entzündet das war [___] totale katastrophe der… |
| 3 | Löschung | `laut` | *(fehlt)* | …hat es laut [___] geknackt und seitdem… |
| 4 | Substitution | `rotatorenmanschetten` | `rotatorenmanschettenruptur` | …notiz verdacht auf [___] ruptur nach sporttrauma… |
| 5 | Löschung | `ruptur` | *(fehlt)* | …verdacht auf rotatorenmanschetten [___] nach sporttrauma… |


---

## MeinungswechselLapinMitte.wav

**WER: 1.1%** — Referenz: 179 Wörter | Hypothese: 178 Wörter | S=1 D=1 I=0 | Edit-Distanz=2

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `kardiales` | `kardinales` | …angina pectoris eventuell [___] ereignis strahlt das… |
| 2 | Löschung | `atemnot` | *(fehlt)* | …haben sie atemnot [___] oder kalten schweiß… |


---

## ChaosLapInMitte.wav

**WER: 18.0%** — Referenz: 272 Wörter | Hypothese: 260 Wörter | S=17 D=22 I=10 | Edit-Distanz=49

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `wackelig` | `wacklig` | …sie wirken etwas [___] auf den beinen… |
| 2 | Einfügung | *(nicht da)* | `ähm` | …so schwindelig seit [___] samstagabend nee moment… |
| 3 | Substitution | `nee` | `ne` | …seit seit samstagabend [___] moment samstag war… |
| 4 | Substitution | `ist` | `genauso` | …rechts seit sonntag [___] der schwindel auch… |
| 5 | Substitution | `der` | `hat` | …seit sonntag ist [___] schwindel auch abhängig… |
| 6 | Substitution | `schwindel` | `das` | …sonntag ist der [___] auch abhängig von… |
| 7 | Substitution | `auch` | `die` | …ist der schwindel [___] abhängig von der… |
| 8 | Substitution | `abhängig` | `frau` | …der schwindel auch [___] von der kopfbewegung… |
| 9 | Löschung | `genau` | *(fehlt)* | …von der kopfbewegung [___] so hat das… |
| 10 | Löschung | `so` | *(fehlt)* | …der kopfbewegung genau [___] hat das nämlich… |
| 11 | Löschung | `hat` | *(fehlt)* | …kopfbewegung genau so [___] das nämlich auch… |
| 12 | Löschung | `das` | *(fehlt)* | …genau so hat [___] nämlich auch bei… |
| 13 | Löschung | `nämlich` | *(fehlt)* | …so hat das [___] auch bei meiner… |
| 14 | Löschung | `auch` | *(fehlt)* | …hat das nämlich [___] bei meiner cousine… |
| 15 | Löschung | `bei` | *(fehlt)* | …das nämlich auch [___] meiner cousine angefangen… |
| 16 | Löschung | `meiner` | *(fehlt)* | …nämlich auch bei [___] cousine angefangen die… |
| 17 | Löschung | `cousine` | *(fehlt)* | …auch bei meiner [___] angefangen die hat… |
| 18 | Löschung | `angefangen` | *(fehlt)* | …bei meiner cousine [___] die hat dann… |
| 19 | Substitution | `akustikusneurinom` | `acousticus` | …wie hieß das [___] glaub ich die… |
| 20 | Substitution | `glaub` | `neurinom` | …hieß das akustikusneurinom [___] ich die musste… |
| 21 | Einfügung | *(nicht da)* | `glaube` | …das akustikusneurinom glaub [___] die musste direkt… |
| 22 | Substitution | `hab` | `habe` | …und operiert werden [___] ich jetzt auch… |
| 23 | Substitution | `karzinophobie` | `carzinophobie` | …notiz patient äußert [___] dass es ein… |
| 24 | Einfügung | *(nicht da)* | `ich` | …patient äußert karzinophobie [___] es ein tumor… |
| 25 | Einfügung | *(nicht da)* | `habe` | …patient äußert karzinophobie [___] es ein tumor… |
| 26 | Einfügung | *(nicht da)* | `da` | …patient äußert karzinophobie [___] es ein tumor… |
| 27 | Einfügung | *(nicht da)* | `echt` | …patient äußert karzinophobie [___] es ein tumor… |
| 28 | Einfügung | *(nicht da)* | `panik` | …patient äußert karzinophobie [___] es ein tumor… |
| 29 | Substitution | `hab` | `habe` | …äußerst unwahrscheinlich ich [___] da echt panik… |
| 30 | Löschung | `da` | *(fehlt)* | …unwahrscheinlich ich hab [___] echt panik wissen… |
| 31 | Löschung | `echt` | *(fehlt)* | …ich hab da [___] panik wissen sie… |
| 32 | Löschung | `panik` | *(fehlt)* | …hab da echt [___] wissen sie ich… |
| 33 | Löschung | `wissen` | *(fehlt)* | …da echt panik [___] sie ich hab… |
| 34 | Löschung | `sie` | *(fehlt)* | …echt panik wissen [___] ich hab ja… |
| 35 | Löschung | `ich` | *(fehlt)* | …panik wissen sie [___] hab ja zwei… |
| 36 | Löschung | `hab` | *(fehlt)* | …wissen sie ich [___] ja zwei kleine… |
| 37 | Substitution | `mal` | `erstmal` | …lassen sie mich [___] in ihr rechtes… |
| 38 | Löschung | `drückt` | *(fehlt)* | …aus ohrenschmalz der [___] richtig fest auf… |
| 39 | Einfügung | *(nicht da)* | `drückt` | …auf das trommelfell [___] verursacht den schmerz… |
| 40 | Substitution | `oft` | `auch` | …pfeifen und oft [___] oft auch den… |
| 41 | Löschung | `und` | *(fehlt)* | …jetzt kurz aus [___] dann sollte der… |
| 42 | Löschung | `habe` | *(fehlt)* | …sei dank ich [___] ich hatte schon… |
| 43 | Löschung | `ich` | *(fehlt)* | …dank ich habe [___] hatte schon wieder… |
| 44 | Substitution | `sie` | `die` | …ja es waren [___] zwei kleinen notiz… |
| 45 | Substitution | `obturans` | `obutrans` | …intervention nötig cerumen [___] rechtsseitig vormedikation 800mg… |
| 46 | Substitution | `800mg` | `800` | …obturans rechtsseitig vormedikation [___] ibuprofen genau das… |
| 47 | Einfügung | *(nicht da)* | `mg` | …rechtsseitig vormedikation 800mg [___] genau das können… |
| 48 | Einfügung | *(nicht da)* | `dann` | …das können sie [___] morgen dann wieder… |
| 49 | Löschung | `dann` | *(fehlt)* | …sie ab morgen [___] wieder weglassen… |


---

## Das Anamnesegespräch.wav

**WER: 4.4%** — Referenz: 2317 Wörter | Hypothese: 2299 Wörter | S=65 D=28 I=10 | Edit-Distanz=103

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `colette` | `collette` | …name ist nina [___] und ich bin… |
| 2 | Substitution | `westphalen` | `westfalen` | …e n bindestrich [___] w e s… |
| 3 | Substitution | `danke` | `dankeschön` | …n alles klar [___] schön frau becken… |
| 4 | Löschung | `schön` | *(fehlt)* | …alles klar danke [___] frau becken westphalen… |
| 5 | Substitution | `westphalen` | `westfalen` | …schön frau becken [___] wie alt sind… |
| 6 | Substitution | `ah` | `oh` | …27 märz 1987 [___] schön herzlichen glückwunsch… |
| 7 | Einfügung | *(nicht da)* | `m` | …denn 1 70 [___] 70 alles klar… |
| 8 | Einfügung | *(nicht da)* | `m` | …70 1 70 [___] klar und wie… |
| 9 | Substitution | `doktor` | `dr` | …ist der herr [___] becker der herr… |
| 10 | Substitution | `doktor` | `dr` | …becker der herr [___] becker wie der… |
| 11 | Löschung | `beschmerzen` | *(fehlt)* | …jetzt für beschmerzen [___] beschwerden schuldigung haben… |
| 12 | Substitution | `schuldigung` | `entschuldigung` | …beschmerzen beschmerzen beschwerden [___] haben sie schmerzen… |
| 13 | Einfügung | *(nicht da)* | `wirklich` | …bewegen weil ich [___] schmerzen habe am… |
| 14 | Löschung | `ja` | *(fehlt)* | …habe am daumen [___] und er ist… |
| 15 | Substitution | `beides` | `weil` | …richtig stark verletzt [___] richtig geschwollen ist… |
| 16 | Einfügung | *(nicht da)* | `es` | …stark verletzt beides [___] geschwollen ist und… |
| 17 | Substitution | `wehtut` | `weh` | …und auch sehr [___] okay knie ist… |
| 18 | Einfügung | *(nicht da)* | `tut` | …auch sehr wehtut [___] knie ist auch… |
| 19 | Substitution | `einen` | `ein` | …haben sie denn [___] fahrradhelm getragen leider… |
| 20 | Substitution | `daraus` | `draus` | …ich habe jetzt [___] gelernt und werde… |
| 21 | Löschung | `am` | *(fehlt)* | …sie haben hinten [___] auf der linken… |
| 22 | Löschung | `ja` | *(fehlt)* | …das ist richtig [___] haben sie irgendeine… |
| 23 | Substitution | `das` | `es` | …nicht so stark [___] geht tatsächlich am… |
| 24 | Substitution | `eins` | `1` | …einer schmerzskala wobei [___] sehr leichten schmerzen… |
| 25 | Substitution | `zehn` | `10` | …schmerzen entspricht und [___] sehr starken schmerzen… |
| 26 | Substitution | `sieben` | `7` | …so auf die [___] zu vor allem… |
| 27 | Substitution | `acht` | `8` | …ich sagen bei [___] wenn ich sitze… |
| 28 | Löschung | `es` | *(fehlt)* | …zu bewegen ist [___] wirklich unerträglich okay… |
| 29 | Substitution | `an` | `in` | …handgelenk aus oder [___] andere finger auch… |
| 30 | Substitution | `dran` | `daran` | …kann mich gut [___] erinnern ja ich… |
| 31 | Substitution | `laktose` | `laktoseintoleranz` | …ich hatte eine [___] intoleranz vor einigen… |
| 32 | Löschung | `intoleranz` | *(fehlt)* | …hatte eine laktose [___] vor einigen jahren… |
| 33 | Substitution | `drei` | `3` | …bei mir vor [___] wochen eine histaminunverträglichkeit… |
| 34 | Einfügung | *(nicht da)* | `mhm` | …eine histaminunverträglichkeit festgestellt [___] äußert sich die… |
| 35 | Substitution | `dekollete` | `dekolletébereich` | …ausschlag hier im [___] bereich okay sonst… |
| 36 | Löschung | `bereich` | *(fehlt)* | …hier im dekollete [___] okay sonst gibt… |
| 37 | Substitution | `zwei` | `2` | …ich wurde vor [___] jahren am fuß… |
| 38 | Substitution | `sieben` | `7` | …sie die pille [___] oder acht jahren… |
| 39 | Substitution | `acht` | `8` | …pille sieben oder [___] jahren okay die… |
| 40 | Substitution | `600er` | `600` | …ich glaube 600 [___] alles klar sind… |
| 41 | Substitution | `alles` | `ja` | …glaube 600 600er [___] klar sind sie… |
| 42 | Löschung | `klar` | *(fehlt)* | …600 600er alles [___] sind sie geimpft… |
| 43 | Löschung | `ich` | *(fehlt)* | …eher nicht ja [___] muss ins krankenhaus… |
| 44 | Löschung | `muss` | *(fehlt)* | …nicht ja ich [___] ins krankenhaus hätte… |
| 45 | Löschung | `ins` | *(fehlt)* | …ja ich muss [___] krankenhaus hätte ich… |
| 46 | Löschung | `krankenhaus` | *(fehlt)* | …ich muss ins [___] hätte ich gewusst… |
| 47 | Substitution | `ihn` | `den` | …muss hätte ich [___] mitgenommen ja ich… |
| 48 | Substitution | `muss` | `brauche` | …mitgenommen ja ich [___] auch gestehen dass… |
| 49 | Substitution | `auch` | `den` | …ja ich muss [___] gestehen dass ich… |
| 50 | Substitution | `gestehen` | `nicht` | …ich muss auch [___] dass ich ihn… |
| 51 | Substitution | `dass` | `weil` | …muss auch gestehen [___] ich ihn nicht… |
| 52 | Substitution | `ihn` | `den` | …gestehen dass ich [___] nicht bei mir… |
| 53 | Substitution | `hab` | `habe` | …zeit nein ich [___] gar keine sonstigen… |
| 54 | Löschung | `ja` | *(fehlt)* | …auch wirklich alle [___] wie sieht es… |
| 55 | Löschung | `aber` | *(fehlt)* | …relevant ist ja [___] ja wann haben… |
| 56 | Löschung | `ja` | *(fehlt)* | …ist ja aber [___] wann haben sie… |
| 57 | Löschung | `damit` | *(fehlt)* | …wann haben sie [___] aufgehört das müssten… |
| 58 | Substitution | `acht` | `8` | …müssten jetzt schon [___] jahre sein seitdem… |
| 59 | Substitution | `sechs` | `6` | …sie geraucht damals [___] sieben jahre sechs… |
| 60 | Substitution | `sieben` | `7` | …geraucht damals sechs [___] jahre sechs sieben… |
| 61 | Substitution | `sechs` | `6` | …sechs sieben jahre [___] sieben jahre okay… |
| 62 | Substitution | `sieben` | `7` | …sieben jahre sechs [___] jahre okay gut… |
| 63 | Substitution | `zwei` | `2` | …dann gerne auch [___] oder drei gläser… |
| 64 | Substitution | `drei` | `3` | …auch zwei oder [___] gläser okay dieses… |
| 65 | Löschung | `und` | *(fehlt)* | …sehr lange arbeiten [___] ja habe zwei… |
| 66 | Substitution | `habe` | `hab` | …arbeiten und ja [___] zwei drei mal… |
| 67 | Substitution | `zwei` | `2` | …und ja habe [___] drei mal ritalin… |
| 68 | Substitution | `drei` | `3` | …ja habe zwei [___] mal ritalin genommen… |
| 69 | Substitution | `wars` | `war` | …genommen okay das [___] aber aber jetzt… |
| 70 | Substitution | `aber` | `s` | …okay das wars [___] aber jetzt schon… |
| 71 | Löschung | `jetzt` | *(fehlt)* | …wars aber aber [___] schon ja sehr… |
| 72 | Löschung | `schon` | *(fehlt)* | …aber aber jetzt [___] ja sehr gut… |
| 73 | Löschung | `gut` | *(fehlt)* | …gut okay prima [___] kurz zu ihrer… |
| 74 | Löschung | `okay` | *(fehlt)* | …großvater hatte leberzirrhose [___] und ist leider… |
| 75 | Löschung | `oh` | *(fehlt)* | …auch daran gestorben [___] das tut mir… |
| 76 | Substitution | `becken` | `böcken` | …sie geschwister frau [___] westphalen ich habe… |
| 77 | Substitution | `es` | `s` | …ja sonst geht [___] ihr gut sehr… |
| 78 | Substitution | `fünf` | `5` | …verheiratet ja seit [___] monaten wie schön… |
| 79 | Substitution | `zwei` | `2` | …ich war vor [___] monaten geschäftlich in… |
| 80 | Substitution | `zwei` | `2` | …waren sie da [___] wochen insgesamt zwei… |
| 81 | Substitution | `zwei` | `2` | …zwei wochen insgesamt [___] wochen insgesamt okay… |
| 82 | Substitution | `ich` | `sich` | …um abzugleichen dass [___] auch wirklich alles… |
| 83 | Einfügung | *(nicht da)* | `alles` | …auch wirklich alles [___] notiert habe vorher… |
| 84 | Substitution | `drüber` | `darüber` | …keinerlei positive auskunft [___] geben das was… |
| 85 | Substitution | `erstmal` | `erst` | …müssen wir wirklich [___] mrt bilder von… |
| 86 | Einfügung | *(nicht da)* | `mal` | …wir wirklich erstmal [___] bilder von machen… |
| 87 | Substitution | `untersuchungen` | `untersuchung` | …auch gleich die [___] direkt durchführen wenn… |
| 88 | Löschung | `dem` | *(fehlt)* | …wenn wir mit [___] mit der aufnahme… |
| 89 | Löschung | `mit` | *(fehlt)* | …wir mit dem [___] der aufnahme fertig… |
| 90 | Substitution | `nochmal` | `noch` | …ja alles klar [___] kurz zum abgleich… |
| 91 | Einfügung | *(nicht da)* | `mal` | …alles klar nochmal [___] zum abgleich sie… |
| 92 | Substitution | `sieben` | `7` | …dort mit einer [___] beschrieben und haben… |
| 93 | Substitution | `acht` | `8` | …schmerzintensität mit einer [___] beschrieben bei bewegung… |
| 94 | Substitution | `zehn` | `10` | …bewegung unerträglich also [___] oder mehr als… |
| 95 | Substitution | `zehn` | `10` | …oder mehr als [___] auch dieser schmerz… |
| 96 | Substitution | `sie` | `ihnen` | …dem unfall dass [___] nur kurz danach… |
| 97 | Substitution | `zwei` | `2` | …rechten fuß vor [___] jahren da wurde… |
| 98 | Löschung | `genau` | *(fehlt)* | …gott sei dank [___] bis auf die… |
| 99 | Substitution | `noch` | `nochmal` | …eingebracht vielen dank [___] mal dafür habe… |
| 100 | Löschung | `mal` | *(fehlt)* | …vielen dank noch [___] dafür habe ich… |
| 101 | Löschung | `ja` | *(fehlt)* | …ist alles richtig [___] perfekt sehr gut… |
| 102 | Substitution | `erstmal` | `erst` | …es das jetzt [___] von meiner seite… |
| 103 | Einfügung | *(nicht da)* | `mal` | …das jetzt erstmal [___] meiner seite wir… |


---

## Anamnesegesrpäch PWC.mp3

**WER: 5.9%** — Referenz: 1530 Wörter | Hypothese: 1484 Wörter | S=22 D=57 I=11 | Edit-Distanz=90

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Einfügung | *(nicht da)* | `grüß` | …theresa ihre physiotherapeutin [___] dürfen sie gerne… |
| 2 | Einfügung | *(nicht da)* | `gott` | …theresa ihre physiotherapeutin [___] dürfen sie gerne… |
| 3 | Substitution | `ein` | `ihr` | …dass wir über [___] problem reden warum… |
| 4 | Löschung | `27` | *(fehlt)* | …sie sagen 27 [___] jahre und sie… |
| 5 | Löschung | `büroangestellte` | *(fehlt)* | …arbeiten als büroangestellte [___] und da arbeiten… |
| 6 | Substitution | `ja` | `eh` | …dann können wir [___] schon starten warum… |
| 7 | Löschung | `also` | *(fehlt)* | …training bin ich [___] ich bin gesprungen… |
| 8 | Löschung | `ich` | *(fehlt)* | …bin ich also [___] bin gesprungen beim… |
| 9 | Löschung | `bin` | *(fehlt)* | …ich also ich [___] gesprungen beim netzfahren… |
| 10 | Substitution | `gelandet` | `gelacht` | …haben dann schief [___] und sind sie… |
| 11 | Einfügung | *(nicht da)* | `eine` | …eine halbe minute [___] circa dass ich… |
| 12 | Substitution | `hoch` | `hochgeklappert` | …und ein bisschen [___] gelagert und dann… |
| 13 | Löschung | `gelagert` | *(fehlt)* | …ein bisschen hoch [___] und dann hat… |
| 14 | Löschung | `ja` | *(fehlt)* | …bin ich eigentlich [___] duschen gegangen und… |
| 15 | Löschung | `ja` | *(fehlt)* | …ein hobby ist [___] genau und dass… |
| 16 | Löschung | `genau` | *(fehlt)* | …hobby ist ja [___] und dass sie… |
| 17 | Einfügung | *(nicht da)* | `genau` | …auf jeden fall [___] und was war… |
| 18 | Substitution | `argen` | `augenschmerzen` | …habe ich keine [___] schmerzen gehabt und… |
| 19 | Löschung | `schmerzen` | *(fehlt)* | …ich keine argen [___] gehabt und am… |
| 20 | Substitution | `mrt` | `mvi` | …ist also war [___] gemacht worden ist… |
| 21 | Substitution | `abgebaut` | `abbaut` | …der muskel komplett [___] und das ja… |
| 22 | Löschung | `u` | *(fehlt)* | …physiotherapeutin dann eingeteilt [___] ja da war… |
| 23 | Substitution | `waren` | `war` | …war eine also [___] sie da in… |
| 24 | Substitution | `sie` | `es` | …eine also waren [___] da in physiotherapeutischer… |
| 25 | Substitution | `in` | `eine` | …waren sie da [___] physiotherapeutischer behandlung ja… |
| 26 | Substitution | `physiotherapeutischer` | `physiotherapeutische` | …sie da in [___] behandlung ja da… |
| 27 | Löschung | `mit` | *(fehlt)* | …eigentlich gehabt habe [___] dem ja und… |
| 28 | Löschung | `dem` | *(fehlt)* | …gehabt habe mit [___] ja und was… |
| 29 | Löschung | `da` | *(fehlt)* | …was haben sie [___] haben sie schon… |
| 30 | Löschung | `haben` | *(fehlt)* | …haben sie da [___] sie schon alles… |
| 31 | Löschung | `sie` | *(fehlt)* | …sie da haben [___] schon alles ja… |
| 32 | Substitution | `also` | `abräumen` | …aufsteigen und das [___] das also dieses… |
| 33 | Löschung | `das` | *(fehlt)* | …und das also [___] also dieses abrollen… |
| 34 | Löschung | `also` | *(fehlt)* | …das also das [___] dieses abrollen mit… |
| 35 | Löschung | `dieses` | *(fehlt)* | …also das also [___] abrollen mit dem… |
| 36 | Löschung | `abrollen` | *(fehlt)* | …das also dieses [___] mit dem fuß… |
| 37 | Löschung | `haben` | *(fehlt)* | …sogar ein bisschen [___] wir treppensteigen dann… |
| 38 | Löschung | `wir` | *(fehlt)* | …ein bisschen haben [___] treppensteigen dann dass… |
| 39 | Löschung | `dann` | *(fehlt)* | …haben wir treppensteigen [___] dass ich halt… |
| 40 | Löschung | `genau` | *(fehlt)* | …wie ich aufsteige [___] das hat ihnen… |
| 41 | Löschung | `ja` | *(fehlt)* | …therapie und ja [___] das war halt… |
| 42 | Substitution | `dann` | `dir` | …wie ist es [___] ergangen dann mit… |
| 43 | Substitution | `ergangen` | `gegangen` | …ist es dann [___] dann mit den… |
| 44 | Löschung | `mit` | *(fehlt)* | …mit den schmerzen [___] den ja schmerzen… |
| 45 | Löschung | `den` | *(fehlt)* | …den schmerzen mit [___] ja schmerzen war… |
| 46 | Löschung | `ja` | *(fehlt)* | …ja schmerzen war [___] war okay sag… |
| 47 | Löschung | `war` | *(fehlt)* | …schmerzen war ja [___] okay sag ich… |
| 48 | Löschung | `die` | *(fehlt)* | …die erste woche [___] letzte zeit und… |
| 49 | Löschung | `letzte` | *(fehlt)* | …erste woche die [___] zeit und dann… |
| 50 | Löschung | `zeit` | *(fehlt)* | …woche die letzte [___] und dann ja… |
| 51 | Einfügung | *(nicht da)* | `darstellen` | …die sich vorstellen [___] und 0 sind… |
| 52 | Löschung | `sind` | *(fehlt)* | …können und 0 [___] ist schmerzfrei wo… |
| 53 | Löschung | `den` | *(fehlt)* | …ich jetzt mit [___] krücken gehe dann… |
| 54 | Löschung | `ansonsten` | *(fehlt)* | …das nicht anders [___] okay und sie… |
| 55 | Einfügung | *(nicht da)* | `ist` | …was das nötigste [___] ja versuche halt… |
| 56 | Einfügung | *(nicht da)* | `zu` | …weit wie möglich [___] und das eigentlich… |
| 57 | Löschung | `ja` | *(fehlt)* | …belastung variiert aber [___] ist noch nicht… |
| 58 | Löschung | `nein` | *(fehlt)* | …richtig schmerzfrei möglich [___] nehmen sie irgendwelche… |
| 59 | Löschung | `nehmen` | *(fehlt)* | …nehme ich nicht [___] sie nicht haben… |
| 60 | Löschung | `sie` | *(fehlt)* | …ich nicht nehmen [___] nicht haben sie… |
| 61 | Löschung | `nicht` | *(fehlt)* | …nicht nehmen sie [___] haben sie am… |
| 62 | Substitution | `anfangs` | `anfang` | …haben sie am [___] aber wahrscheinlich in… |
| 63 | Substitution | `gekriegt` | `bekommen` | …am anfang schmerzmittel [___] am anfang sowieso… |
| 64 | Substitution | `mitgehabt` | `mitbekommen` | …ich noch schmerzmittel [___] für daheim aber… |
| 65 | Substitution | `mussten` | `musst` | …gebraucht also die [___] sie nicht nehmen… |
| 66 | Substitution | `sie` | `du` | …also die mussten [___] nicht nehmen mit… |
| 67 | Löschung | `ne` | *(fehlt)* | …gemacht für übungen [___] keine eigentlich nur… |
| 68 | Löschung | `keine` | *(fehlt)* | …für übungen ne [___] eigentlich nur versucht… |
| 69 | Löschung | `dann` | *(fehlt)* | …haben auch fortschritte [___] bemerkt ja nur… |
| 70 | Löschung | `in` | *(fehlt)* | …in einer wohnung [___] einer wohnung haben… |
| 71 | Löschung | `einer` | *(fehlt)* | …einer wohnung in [___] wohnung haben sie… |
| 72 | Löschung | `wohnung` | *(fehlt)* | …wohnung in einer [___] haben sie da… |
| 73 | Substitution | `dem` | `den` | …es ist in [___] zweiten stock das… |
| 74 | Löschung | `20` | *(fehlt)* | …bis 30 treppen [___] bis 30 treppen… |
| 75 | Löschung | `bis` | *(fehlt)* | …30 treppen 20 [___] 30 treppen und… |
| 76 | Löschung | `30` | *(fehlt)* | …treppen 20 bis [___] treppen und das… |
| 77 | Löschung | `treppen` | *(fehlt)* | …20 bis 30 [___] und das hat… |
| 78 | Löschung | `ja` | *(fehlt)* | …also die so [___] ja ja also… |
| 79 | Löschung | `ja` | *(fehlt)* | …die so ja [___] ja also familie… |
| 80 | Löschung | `ja` | *(fehlt)* | …so ja ja [___] also familie freunde… |
| 81 | Löschung | `nein` | *(fehlt)* | …beispiel diabetes oder [___] dass sie wüssten… |
| 82 | Löschung | `nein` | *(fehlt)* | …sie wüssten narconabhängigkeiten [___] sie stehen ja… |
| 83 | Substitution | `schmerz` | `schmerztabletten` | …abgesehen von den [___] nein tabletten nein… |
| 84 | Löschung | `tabletten` | *(fehlt)* | …den schmerz nein [___] nein die nehme… |
| 85 | Löschung | `nein` | *(fehlt)* | …schmerz nein tabletten [___] die nehme ich… |
| 86 | Substitution | `grasbäutner` | `klaßböckner` | …vielen dank frau [___] und wir treffen… |
| 87 | Einfügung | *(nicht da)* | `what` | …nächsten behandlung danke [___] … |
| 88 | Einfügung | *(nicht da)* | `is` | …nächsten behandlung danke [___] … |
| 89 | Einfügung | *(nicht da)* | `it` | …nächsten behandlung danke [___] … |
| 90 | Einfügung | *(nicht da)* | `boss` | …nächsten behandlung danke [___] … |
