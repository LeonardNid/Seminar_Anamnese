# WER-Analyse: Speechmatics (Cloud) + GPT-4o

> Vollständige Fehleranalyse: STT-Rohausgabe vs. Ground Truth
> Satzzeichen und Groß-/Kleinschreibung werden ignoriert.
> **S** = Substitution, **D** = Löschung (fehlt im STT), **I** = Einfügung (extra im STT)

## Übersicht

| Audiodatei | Ref-Wörter | Hyp-Wörter | S | D | I | Edit-Dist | WER |
|-----------|-----------|-----------|---|---|---|-----------|-----|
| OriginalDC.m4a | 232 | 239 | 6 | 0 | 7 | 13 | **5.6%** |
| OriginalDCWhiteNoise.m4a | 229 | 187 | 26 | 43 | 1 | 70 | **30.6%** |
| OriginalLapInMitte.wav | 231 | 230 | 6 | 4 | 3 | 13 | **5.6%** |
| OriginalLapBeiArzt.wav | 226 | 228 | 6 | 2 | 4 | 12 | **5.3%** |
| SelbstkorrekturLapInMitte.wav | 183 | 185 | 9 | 4 | 6 | 19 | **10.4%** |
| UnterbrechungLapInMitte.wav | 153 | 138 | 11 | 15 | 0 | 26 | **17.0%** |
| GedankenprüngeLapInMitte.wav | 192 | 194 | 4 | 1 | 3 | 8 | **4.2%** |
| MeinungswechselLapinMitte.wav | 179 | 176 | 3 | 3 | 0 | 6 | **3.4%** |
| ChaosLapInMitte.wav | 272 | 256 | 18 | 20 | 4 | 42 | **15.4%** |
| Das Anamnesegespräch.wav | 2317 | 2282 | 51 | 44 | 9 | 104 | **4.5%** |
| Anamnesegesrpäch PWC.mp3 | 1530 | 1452 | 149 | 141 | 63 | 353 | **23.1%** |


---

## OriginalDC.m4a

**WER: 5.6%** — Referenz: 232 Wörter | Hypothese: 239 Wörter | S=6 D=0 I=7 | Edit-Distanz=13

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `das` | `es` | …hoch oder ist [___] eher ein trockener… |
| 2 | Substitution | `hochlaufe` | `hoch` | …den zweiten stock [___] schnaufe ich schon… |
| 3 | Einfügung | *(nicht da)* | `laufe` | …zweiten stock hochlaufe [___] ich schon ganz… |
| 4 | Einfügung | *(nicht da)* | `auch` | …doll und muss [___] der hälfte kurz… |
| 5 | Substitution | `belastungsdyspnoe` | `belastungs` | …erwähnen notiz deutliche [___] aber keine ruhedyspnoe… |
| 6 | Einfügung | *(nicht da)* | `dyspnoe` | …notiz deutliche belastungsdyspnoe [___] keine ruhedyspnoe keine… |
| 7 | Substitution | `ruhedyspnoe` | `ruhe` | …belastungsdyspnoe aber keine [___] keine thorakalen schmerzen… |
| 8 | Einfügung | *(nicht da)* | `dyspnoe` | …aber keine ruhedyspnoe [___] thorakalen schmerzen ja… |
| 9 | Einfügung | *(nicht da)* | `sich` | …ja nehmen sie [___] schon irgendwelche medikamente… |
| 10 | Einfügung | *(nicht da)* | `nehmen` | …ja nehmen sie [___] schon irgendwelche medikamente… |
| 11 | Einfügung | *(nicht da)* | `sie` | …ja nehmen sie [___] schon irgendwelche medikamente… |
| 12 | Substitution | `lutsche` | `lösche` | …medikamente dagegen ich [___] nur diese normalen… |
| 13 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## OriginalDCWhiteNoise.m4a

**WER: 30.6%** — Referenz: 229 Wörter | Hypothese: 187 Wörter | S=26 D=43 I=1 | Edit-Distanz=70

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `huste` | `habe` | …herr doktor ich [___] seit knapp einer… |
| 2 | Substitution | `seit` | `keine` | …doktor ich huste [___] knapp einer woche… |
| 3 | Substitution | `knapp` | `lust` | …ich huste seit [___] einer woche ununterbrochen… |
| 4 | Löschung | `einer` | *(fehlt)* | …huste seit knapp [___] woche ununterbrochen und… |
| 5 | Löschung | `woche` | *(fehlt)* | …seit knapp einer [___] ununterbrochen und mir… |
| 6 | Löschung | `ununterbrochen` | *(fehlt)* | …knapp einer woche [___] und mir ist… |
| 7 | Löschung | `und` | *(fehlt)* | …einer woche ununterbrochen [___] mir ist ständig… |
| 8 | Löschung | `mir` | *(fehlt)* | …woche ununterbrochen und [___] ist ständig heiß… |
| 9 | Löschung | `ist` | *(fehlt)* | …ununterbrochen und mir [___] ständig heiß und… |
| 10 | Substitution | `denn` | `schon` | …haben sie das [___] mal gemessen ja… |
| 11 | Substitution | `8` | `5` | …waren es 38 [___] grad auf dem… |
| 12 | Löschung | `auf` | *(fehlt)* | …38 8 grad [___] dem thermometer okay… |
| 13 | Löschung | `dem` | *(fehlt)* | …8 grad auf [___] thermometer okay notiz… |
| 14 | Löschung | `thermometer` | *(fehlt)* | …grad auf dem [___] okay notiz subfebrile… |
| 15 | Substitution | `da` | `der` | …sie husten kommt [___] schleim mit hoch… |
| 16 | Substitution | `das` | `es` | …hoch oder ist [___] eher ein trockener… |
| 17 | Löschung | `da` | *(fehlt)* | …ein trockener reizhusten [___] kommt richtig viel… |
| 18 | Löschung | `kommt` | *(fehlt)* | …trockener reizhusten da [___] richtig viel hoch… |
| 19 | Substitution | `viel` | `also` | …da kommt richtig [___] hoch das ist… |
| 20 | Löschung | `hoch` | *(fehlt)* | …kommt richtig viel [___] das ist so… |
| 21 | Löschung | `das` | *(fehlt)* | …richtig viel hoch [___] ist so dickflüssig… |
| 22 | Löschung | `ist` | *(fehlt)* | …viel hoch das [___] so dickflüssig und… |
| 23 | Löschung | `dickflüssig` | *(fehlt)* | …das ist so [___] und eher gelblich… |
| 24 | Löschung | `und` | *(fehlt)* | …ist so dickflüssig [___] eher gelblich grün… |
| 25 | Löschung | `eher` | *(fehlt)* | …so dickflüssig und [___] gelblich grün würde… |
| 26 | Löschung | `grün` | *(fehlt)* | …und eher gelblich [___] würde sagen würde… |
| 27 | Löschung | `sagen` | *(fehlt)* | …gelblich grün würde [___] würde ich sagen… |
| 28 | Löschung | `würde` | *(fehlt)* | …grün würde sagen [___] ich sagen das… |
| 29 | Substitution | `purulentem` | `poulentem` | …produktiver husten mit [___] sputum haben sie… |
| 30 | Substitution | `weh` | `bloß` | …luft bekommen also [___] tut es in… |
| 31 | Löschung | `tut` | *(fehlt)* | …bekommen also weh [___] es in der… |
| 32 | Löschung | `es` | *(fehlt)* | …also weh tut [___] in der brust… |
| 33 | Löschung | `in` | *(fehlt)* | …weh tut es [___] der brust nicht… |
| 34 | Löschung | `der` | *(fehlt)* | …tut es in [___] brust nicht direkt… |
| 35 | Löschung | `brust` | *(fehlt)* | …es in der [___] nicht direkt aber… |
| 36 | Löschung | `direkt` | *(fehlt)* | …der brust nicht [___] aber wenn ich… |
| 37 | Löschung | `aber` | *(fehlt)* | …brust nicht direkt [___] wenn ich die… |
| 38 | Substitution | `treppen` | `zweite` | …wenn ich die [___] in den zweiten… |
| 39 | Substitution | `in` | `staffel` | …ich die treppen [___] den zweiten stock… |
| 40 | Substitution | `den` | `laufen` | …die treppen in [___] zweiten stock hochlaufe… |
| 41 | Substitution | `zweiten` | `lassen` | …treppen in den [___] stock hochlaufe schnaufe… |
| 42 | Substitution | `stock` | `soll` | …in den zweiten [___] hochlaufe schnaufe ich… |
| 43 | Substitution | `hochlaufe` | `erhält` | …den zweiten stock [___] schnaufe ich schon… |
| 44 | Substitution | `schnaufe` | `auch` | …zweiten stock hochlaufe [___] ich schon ganz… |
| 45 | Substitution | `ich` | `bezahlt` | …stock hochlaufe schnaufe [___] schon ganz schön… |
| 46 | Löschung | `schon` | *(fehlt)* | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 47 | Löschung | `ganz` | *(fehlt)* | …schnaufe ich schon [___] schön doll und… |
| 48 | Löschung | `schön` | *(fehlt)* | …ich schon ganz [___] doll und muss… |
| 49 | Löschung | `doll` | *(fehlt)* | …schon ganz schön [___] und muss auf… |
| 50 | Löschung | `und` | *(fehlt)* | …ganz schön doll [___] muss auf der… |
| 51 | Löschung | `muss` | *(fehlt)* | …schön doll und [___] auf der hälfte… |
| 52 | Löschung | `auf` | *(fehlt)* | …doll und muss [___] der hälfte kurz… |
| 53 | Löschung | `der` | *(fehlt)* | …und muss auf [___] hälfte kurz anhalten… |
| 54 | Löschung | `hälfte` | *(fehlt)* | …muss auf der [___] kurz anhalten gut… |
| 55 | Löschung | `kurz` | *(fehlt)* | …auf der hälfte [___] anhalten gut dass… |
| 56 | Löschung | `anhalten` | *(fehlt)* | …der hälfte kurz [___] gut dass sie… |
| 57 | Löschung | `gut` | *(fehlt)* | …hälfte kurz anhalten [___] dass sie das… |
| 58 | Substitution | `ruhedyspnoe` | `ruhe` | …deutliche belastungsdyspnoe keine [___] keine thorakalen schmerzen… |
| 59 | Einfügung | *(nicht da)* | `dyspnoe` | …belastungsdyspnoe keine ruhedyspnoe [___] thorakalen schmerzen ja… |
| 60 | Substitution | `lutsche` | `bin` | …medikamente dagegen ich [___] nur diese normalen… |
| 61 | Substitution | `nur` | `schon` | …dagegen ich lutsche [___] diese normalen hustenbonbons… |
| 62 | Substitution | `diese` | `mal` | …ich lutsche nur [___] normalen hustenbonbons aus… |
| 63 | Substitution | `normalen` | `raus` | …lutsche nur diese [___] hustenbonbons aus der… |
| 64 | Löschung | `hustenbonbons` | *(fehlt)* | …nur diese normalen [___] aus der drogerie… |
| 65 | Substitution | `drogerie` | `pflege` | …hustenbonbons aus der [___] und trinke viel… |
| 66 | Löschung | `kamillentee` | *(fehlt)* | …und trinke viel [___] alles klar notiz… |
| 67 | Substitution | `vormedikation` | `medikation` | …notiz keine spezifische [___] lediglich supportive hausmittel… |
| 68 | Löschung | `dafür` | *(fehlt)* | …bitte atmen sie [___] tief durch den… |
| 69 | Löschung | `offen` | *(fehlt)* | …durch den mund [___] ein und aus… |
| 70 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## OriginalLapInMitte.wav

**WER: 5.6%** — Referenz: 231 Wörter | Hypothese: 230 Wörter | S=6 D=4 I=3 | Edit-Distanz=13

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `da` | `das` | …sie husten kommt [___] schleim mit hoch… |
| 2 | Substitution | `schleim` | `schleimetuch` | …husten kommt da [___] mit hoch oder… |
| 3 | Löschung | `mit` | *(fehlt)* | …kommt da schleim [___] hoch oder ist… |
| 4 | Löschung | `hoch` | *(fehlt)* | …da schleim mit [___] oder ist das… |
| 5 | Substitution | `das` | `es` | …hoch oder ist [___] eher ein trockener… |
| 6 | Löschung | `eher` | *(fehlt)* | …dickflüssig dickflüssig und [___] gelblich grün würde… |
| 7 | Substitution | `hochlaufe` | `hoch` | …den zweiten stock [___] schnaufe ich schon… |
| 8 | Einfügung | *(nicht da)* | `laufe` | …zweiten stock hochlaufe [___] ich schon ganz… |
| 9 | Löschung | `schon` | *(fehlt)* | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 10 | Einfügung | *(nicht da)* | `auch` | …doll und muss [___] der hälfte kurz… |
| 11 | Substitution | `ruhedyspnoe` | `ruhe` | …deutliche belastungsdyspnoe keine [___] keine thorakalen schmerzen… |
| 12 | Einfügung | *(nicht da)* | `dyspnoe` | …belastungsdyspnoe keine ruhedyspnoe [___] thorakalen schmerzen nehmen… |
| 13 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## OriginalLapBeiArzt.wav

**WER: 5.3%** — Referenz: 226 Wörter | Hypothese: 228 Wörter | S=6 D=2 I=4 | Edit-Distanz=12

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `huste` | `wusste` | …herr doktor ich [___] seit knapp einer… |
| 2 | Einfügung | *(nicht da)* | `und` | …bis febrile temperaturen [___] 38 8 grad… |
| 3 | Löschung | `eher` | *(fehlt)* | …so dickflüssig und [___] gelblich grün würde… |
| 4 | Substitution | `hochlaufe` | `hoch` | …den zweiten stock [___] schnaufe ich schon… |
| 5 | Einfügung | *(nicht da)* | `laufe` | …zweiten stock hochlaufe [___] ich schon ganz… |
| 6 | Löschung | `schon` | *(fehlt)* | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 7 | Substitution | `auf` | `auch` | …doll und muss [___] der hälfte kurz… |
| 8 | Einfügung | *(nicht da)* | `in` | …und muss auf [___] hälfte kurz anhalten… |
| 9 | Substitution | `ruhedyspnoe` | `ruhe` | …deutliche belastungsdyspnoe keine [___] keine thorakalen schmerzen… |
| 10 | Einfügung | *(nicht da)* | `dyspnoe` | …belastungsdyspnoe keine ruhedyspnoe [___] thorakalen schmerzen nehmen… |
| 11 | Substitution | `lutsche` | `lösche` | …medikamente dagegen ich [___] nur diese normalen… |
| 12 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## SelbstkorrekturLapInMitte.wav

**WER: 10.4%** — Referenz: 183 Wörter | Hypothese: 185 Wörter | S=9 D=4 I=6 | Edit-Distanz=19

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Löschung | `ne` | *(fehlt)* | …habe seit dienstag [___] warten sie mal… |
| 2 | Löschung | `äh` | *(fehlt)* | …linke schläfe aus [___] quatsch ich zeig… |
| 3 | Substitution | `zeig` | `zeige` | …äh quatsch ich [___] s gerade falsch… |
| 4 | Substitution | `s` | `es` | …quatsch ich zeig [___] gerade falsch rum… |
| 5 | Substitution | `rum` | `herum` | …s gerade falsch [___] in die rechte… |
| 6 | Einfügung | *(nicht da)* | `es` | …es ganz schlimm [___] es ein pochender… |
| 7 | Löschung | `es` | *(fehlt)* | …ganz schlimm ist [___] ein pochender oder… |
| 8 | Substitution | `hab` | `habe` | …nicht geholfen dann [___] ich am nachmittag… |
| 9 | Löschung | `nee` | *(fehlt)* | …ich am nachmittag [___] es war erst… |
| 10 | Einfügung | *(nicht da)* | `habe` | …vor dem schlaf [___] eine ibuprofen 400… |
| 11 | Einfügung | *(nicht da)* | `ich` | …vor dem schlaf [___] eine ibuprofen 400… |
| 12 | Substitution | `waren` | `war` | …400 genommen oder [___] das 600 die… |
| 13 | Substitution | `das` | `es` | …genommen oder waren [___] 600 die rosafarbenen… |
| 14 | Substitution | `600` | `doch` | …oder waren das [___] die rosafarbenen aus… |
| 15 | Substitution | `rosafarbenen` | `sechs` | …das 600 die [___] aus der großen… |
| 16 | Einfügung | *(nicht da)* | `hundertste` | …600 die rosafarbenen [___] der großen packung… |
| 17 | Einfügung | *(nicht da)* | `die` | …600 die rosafarbenen [___] der großen packung… |
| 18 | Einfügung | *(nicht da)* | `rosafarbene` | …600 die rosafarbenen [___] der großen packung… |
| 19 | Substitution | `rechtsseitig` | `rechtzeitig` | …notiz pochender kopfschmerz [___] seit mittwoch keine… |


---

## UnterbrechungLapInMitte.wav

**WER: 17.0%** — Referenz: 153 Wörter | Hypothese: 138 Wörter | S=11 D=15 I=0 | Edit-Distanz=26

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `drei` | `3` | …ich bin um [___] uhr aufgewacht und… |
| 2 | Substitution | `uhr` | `00` | …bin um drei [___] aufgewacht und hatte… |
| 3 | Löschung | `wo` | *(fehlt)* | …dass ich direkt [___] genau krampft es… |
| 4 | Löschung | `genau` | *(fehlt)* | …ich direkt wo [___] krampft es denn… |
| 5 | Löschung | `krampft` | *(fehlt)* | …direkt wo genau [___] es denn auf… |
| 6 | Löschung | `es` | *(fehlt)* | …wo genau krampft [___] denn auf die… |
| 7 | Löschung | `denn` | *(fehlt)* | …genau krampft es [___] auf die toilette… |
| 8 | Substitution | `dem` | `den` | …ist direkt über [___] bauchnabel strahlt der… |
| 9 | Löschung | `es` | *(fehlt)* | …in die leiste [___] tut einfach überall… |
| 10 | Löschung | `tut` | *(fehlt)* | …die leiste es [___] einfach überall weh… |
| 11 | Löschung | `einfach` | *(fehlt)* | …leiste es tut [___] überall weh aber… |
| 12 | Löschung | `da` | *(fehlt)* | …ja heute morgen [___] war es bei… |
| 13 | Substitution | `grad` | `30` | …es bei 38 [___] 38 was komma… |
| 14 | Löschung | `38` | *(fehlt)* | …bei 38 grad [___] was komma zwei… |
| 15 | Substitution | `komma` | `2` | …grad 38 was [___] zwei komma fünf… |
| 16 | Substitution | `zwei` | `5` | …38 was komma [___] komma fünf komma… |
| 17 | Substitution | `komma` | `0` | …was komma zwei [___] fünf komma drei… |
| 18 | Substitution | `fünf` | `3` | …komma zwei komma [___] komma drei und… |
| 19 | Löschung | `komma` | *(fehlt)* | …zwei komma fünf [___] drei und ich… |
| 20 | Löschung | `drei` | *(fehlt)* | …komma fünf komma [___] und ich habe… |
| 21 | Substitution | `epigastrischer` | `epigastrische` | …erbrochen heute notiz [___] schmerz erbrechen temperatur… |
| 22 | Substitution | `schmerz` | `schmerzen` | …heute notiz epigastrischer [___] erbrechen temperatur bei… |
| 23 | Löschung | `mit` | *(fehlt)* | …haben wir gegrillt [___] den nachbarn vielleicht… |
| 24 | Löschung | `den` | *(fehlt)* | …wir gegrillt mit [___] nachbarn vielleicht etwas… |
| 25 | Löschung | `nachbarn` | *(fehlt)* | …gegrillt mit den [___] vielleicht etwas vom… |
| 26 | Substitution | `das` | `was` | …etwas vom grill [___] nicht ganz durch… |


---

## GedankenprüngeLapInMitte.wav

**WER: 4.2%** — Referenz: 192 Wörter | Hypothese: 194 Wörter | S=4 D=1 I=3 | Edit-Distanz=8

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `ne` | `eine` | …entzündet das war [___] totale katastrophe der… |
| 2 | Einfügung | *(nicht da)* | `da` | …ne totale katastrophe [___] lag ewig im… |
| 3 | Einfügung | *(nicht da)* | `lag` | …ne totale katastrophe [___] lag ewig im… |
| 4 | Substitution | `nachoperiert` | `operiert` | …und musste dreimal [___] werden aber bei… |
| 5 | Substitution | `tennisspielen` | `tennis` | …ja genau beim [___] vorgestern fing das… |
| 6 | Einfügung | *(nicht da)* | `spielen` | …genau beim tennisspielen [___] fing das nämlich… |
| 7 | Substitution | `rotatorenmanschetten` | `rotatorenmanschettenruptur` | …notiz verdacht auf [___] ruptur nach sporttrauma… |
| 8 | Löschung | `ruptur` | *(fehlt)* | …verdacht auf rotatorenmanschetten [___] nach sporttrauma… |


---

## MeinungswechselLapinMitte.wav

**WER: 3.4%** — Referenz: 179 Wörter | Hypothese: 176 Wörter | S=3 D=3 I=0 | Edit-Distanz=6

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `retrosternaler` | `retrosternale` | …ernst notiz leitsymptom [___] schmerz verdacht auf… |
| 2 | Substitution | `kardiales` | `kardinales` | …angina pectoris eventuell [___] ereignis strahlt das… |
| 3 | Löschung | `in` | *(fehlt)* | …linken arm oder [___] den unterkiefer aus… |
| 4 | Löschung | `ah` | *(fehlt)* | …muss ständig aufstoßen [___] warten sie nach… |
| 5 | Substitution | `das` | `dass` | …ihre speiseröhre auf [___] verursacht diesen schmerz… |
| 6 | Löschung | `verursacht` | *(fehlt)* | …speiseröhre auf das [___] diesen schmerz… |


---

## ChaosLapInMitte.wav

**WER: 15.4%** — Referenz: 272 Wörter | Hypothese: 256 Wörter | S=18 D=20 I=4 | Edit-Distanz=42

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `nee` | `im` | …seit seit samstagabend [___] moment samstag war… |
| 2 | Substitution | `im` | `ein` | …sich alles wie [___] karussell oder schwanken… |
| 3 | Löschung | `genau` | *(fehlt)* | …von der kopfbewegung [___] so hat das… |
| 4 | Löschung | `so` | *(fehlt)* | …der kopfbewegung genau [___] hat das nämlich… |
| 5 | Löschung | `hat` | *(fehlt)* | …kopfbewegung genau so [___] das nämlich auch… |
| 6 | Löschung | `das` | *(fehlt)* | …genau so hat [___] nämlich auch bei… |
| 7 | Löschung | `nämlich` | *(fehlt)* | …so hat das [___] auch bei meiner… |
| 8 | Löschung | `auch` | *(fehlt)* | …hat das nämlich [___] bei meiner cousine… |
| 9 | Löschung | `bei` | *(fehlt)* | …das nämlich auch [___] meiner cousine angefangen… |
| 10 | Löschung | `meiner` | *(fehlt)* | …nämlich auch bei [___] cousine angefangen die… |
| 11 | Löschung | `cousine` | *(fehlt)* | …auch bei meiner [___] angefangen die hat… |
| 12 | Löschung | `angefangen` | *(fehlt)* | …bei meiner cousine [___] die hat dann… |
| 13 | Substitution | `glaub` | `glaube` | …hieß das akustikusneurinom [___] ich die musste… |
| 14 | Substitution | `hab` | `habe` | …und operiert werden [___] ich jetzt auch… |
| 15 | Substitution | `dass` | `ich` | …patient äußert karzinophobie [___] es ein tumor… |
| 16 | Substitution | `es` | `habe` | …äußert karzinophobie dass [___] ein tumor ist… |
| 17 | Substitution | `ein` | `einen` | …karzinophobie dass es [___] tumor ist ist… |
| 18 | Substitution | `ist` | `es` | …es ein tumor [___] ist äußerst unwahrscheinlich… |
| 19 | Substitution | `hab` | `habe` | …äußerst unwahrscheinlich ich [___] da echt panik… |
| 20 | Löschung | `da` | *(fehlt)* | …unwahrscheinlich ich hab [___] echt panik wissen… |
| 21 | Löschung | `echt` | *(fehlt)* | …ich hab da [___] panik wissen sie… |
| 22 | Löschung | `panik` | *(fehlt)* | …hab da echt [___] wissen sie ich… |
| 23 | Löschung | `wissen` | *(fehlt)* | …da echt panik [___] sie ich hab… |
| 24 | Löschung | `sie` | *(fehlt)* | …echt panik wissen [___] ich hab ja… |
| 25 | Löschung | `ich` | *(fehlt)* | …panik wissen sie [___] hab ja zwei… |
| 26 | Löschung | `hab` | *(fehlt)* | …wissen sie ich [___] ja zwei kleine… |
| 27 | Substitution | `mal` | `erstmal` | …lassen sie mich [___] in ihr rechtes… |
| 28 | Substitution | `pfropf` | `pfropfen` | …da einen massiven [___] aus ohrenschmalz der… |
| 29 | Löschung | `drückt` | *(fehlt)* | …aus ohrenschmalz der [___] richtig fest auf… |
| 30 | Löschung | `und` | *(fehlt)* | …jetzt kurz aus [___] dann sollte der… |
| 31 | Substitution | `400er` | `vier` | …waren es zwei [___] ja es waren… |
| 32 | Einfügung | *(nicht da)* | `hunderter` | …es zwei 400er [___] es waren sie… |
| 33 | Substitution | `sie` | `die` | …ja es waren [___] zwei kleinen notiz… |
| 34 | Substitution | `cerumen` | `zero` | …operative intervention nötig [___] obturans rechtsseitig vormedikation… |
| 35 | Substitution | `obturans` | `menopotrans` | …intervention nötig cerumen [___] rechtsseitig vormedikation 800mg… |
| 36 | Substitution | `rechtsseitig` | `rechtzeitig` | …nötig cerumen obturans [___] vormedikation 800mg ibuprofen… |
| 37 | Substitution | `vormedikation` | `vor` | …cerumen obturans rechtsseitig [___] 800mg ibuprofen genau… |
| 38 | Substitution | `800mg` | `medikation` | …obturans rechtsseitig vormedikation [___] ibuprofen genau das… |
| 39 | Einfügung | *(nicht da)* | `800` | …rechtsseitig vormedikation 800mg [___] genau das können… |
| 40 | Einfügung | *(nicht da)* | `milligramm` | …rechtsseitig vormedikation 800mg [___] genau das können… |
| 41 | Einfügung | *(nicht da)* | `dann` | …das können sie [___] morgen dann wieder… |
| 42 | Löschung | `dann` | *(fehlt)* | …sie ab morgen [___] wieder weglassen… |


---

## Das Anamnesegespräch.wav

**WER: 4.5%** — Referenz: 2317 Wörter | Hypothese: 2282 Wörter | S=51 D=44 I=9 | Edit-Distanz=104

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `becken` | `becker` | …ich heiße julia [___] westphalen julia becken… |
| 2 | Substitution | `becken` | `becker` | …becken westphalen julia [___] westphalen können sie… |
| 3 | Löschung | `c` | *(fehlt)* | …becken b e [___] k e n… |
| 4 | Substitution | `westphalen` | `westfalenveste` | …e n bindestrich [___] w e s… |
| 5 | Löschung | `w` | *(fehlt)* | …n bindestrich westphalen [___] e s t… |
| 6 | Löschung | `e` | *(fehlt)* | …bindestrich westphalen w [___] s t p… |
| 7 | Löschung | `s` | *(fehlt)* | …westphalen w e [___] t p h… |
| 8 | Löschung | `t` | *(fehlt)* | …w e s [___] p h a… |
| 9 | Substitution | `l` | `liter` | …p h a [___] e n alles… |
| 10 | Substitution | `becken` | `becker` | …danke schön frau [___] westphalen wie alt… |
| 11 | Substitution | `westphalen` | `westfalen` | …schön frau becken [___] wie alt sind… |
| 12 | Einfügung | *(nicht da)* | `20` | …geburtstag am 27 [___] 1987 ah schön… |
| 13 | Substitution | `ah` | `na` | …27 märz 1987 [___] schön herzlichen glückwunsch… |
| 14 | Substitution | `westphalen` | `westfalen` | …dank frau becken [___] wie groß sind… |
| 15 | Substitution | `1` | `170170` | …sind sie denn [___] 70 1 70… |
| 16 | Löschung | `70` | *(fehlt)* | …sie denn 1 [___] 1 70 alles… |
| 17 | Löschung | `1` | *(fehlt)* | …denn 1 70 [___] 70 alles klar… |
| 18 | Löschung | `70` | *(fehlt)* | …1 70 1 [___] alles klar und… |
| 19 | Substitution | `becken` | `becker` | …klar gut frau [___] westphalen sie wurden… |
| 20 | Substitution | `westphalen` | `westfalen` | …gut frau becken [___] sie wurden ja… |
| 21 | Substitution | `beschmerzen` | `schmerzen` | …denn jetzt für [___] beschmerzen beschwerden schuldigung… |
| 22 | Substitution | `beschmerzen` | `schmerzen` | …jetzt für beschmerzen [___] beschwerden schuldigung haben… |
| 23 | Substitution | `schuldigung` | `entschuldigung` | …beschmerzen beschmerzen beschwerden [___] haben sie schmerzen… |
| 24 | Löschung | `leicht` | *(fehlt)* | …tatsächlich den kopf [___] gestoßen ich habe… |
| 25 | Substitution | `becken` | `beck` | …sie genau frau [___] westphalen haben sie… |
| 26 | Substitution | `westphalen` | `in` | …genau frau becken [___] haben sie denn… |
| 27 | Einfügung | *(nicht da)* | `westfalen` | …frau becken westphalen [___] sie denn einen… |
| 28 | Substitution | `becken` | `beck` | …vor aussehen frau [___] westphalen bitte bitte… |
| 29 | Substitution | `westphalen` | `in` | …aussehen frau becken [___] bitte bitte tragen… |
| 30 | Einfügung | *(nicht da)* | `westfalen` | …frau becken westphalen [___] bitte tragen sie… |
| 31 | Löschung | `am` | *(fehlt)* | …sie haben hinten [___] auf der linken… |
| 32 | Löschung | `ja` | *(fehlt)* | …das ist richtig [___] haben sie irgendeine… |
| 33 | Löschung | `recht` | *(fehlt)* | …gar nicht mehr [___] bewegen wenn wir… |
| 34 | Substitution | `den` | `ihn` | …wenn ich versuche [___] zu bewegen okay… |
| 35 | Substitution | `selbst` | `sehr` | …bewegen es tut [___] weh wenn ich… |
| 36 | Substitution | `becken` | `beck` | …unfall erinnern frau [___] westphalen ich kann… |
| 37 | Substitution | `westphalen` | `in` | …erinnern frau becken [___] ich kann mich… |
| 38 | Einfügung | *(nicht da)* | `westfalen` | …frau becken westphalen [___] kann mich gut… |
| 39 | Substitution | `dran` | `daran` | …kann mich gut [___] erinnern ja ich… |
| 40 | Substitution | `becken` | `beck` | …sehr gut frau [___] westphalen haben sie… |
| 41 | Substitution | `westphalen` | `in` | …gut frau becken [___] haben sie irgendwelche… |
| 42 | Einfügung | *(nicht da)* | `westfalen` | …frau becken westphalen [___] sie irgendwelche vorerkrankungen… |
| 43 | Substitution | `westphalen` | `westfalen` | …gut frau becken [___] sind sie schon… |
| 44 | Substitution | `ich` | `das` | …nein sehr gut [___] konnte ganz bald… |
| 45 | Substitution | `becken` | `becker` | …gut gelaufen frau [___] westphalen nehmen sie… |
| 46 | Substitution | `600` | `608` | …sind das 400 [___] 800 also meistens… |
| 47 | Substitution | `800` | `100` | …das 400 600 [___] also meistens das… |
| 48 | Substitution | `600er` | `600` | …ich glaube 600 [___] alles klar sind… |
| 49 | Löschung | `hätte` | *(fehlt)* | …muss ins krankenhaus [___] ich gewusst dass… |
| 50 | Löschung | `ich` | *(fehlt)* | …ins krankenhaus hätte [___] gewusst dass ich… |
| 51 | Löschung | `gewusst` | *(fehlt)* | …krankenhaus hätte ich [___] dass ich ins… |
| 52 | Löschung | `dass` | *(fehlt)* | …hätte ich gewusst [___] ich ins krankenhaus… |
| 53 | Löschung | `ich` | *(fehlt)* | …ich gewusst dass [___] ins krankenhaus muss… |
| 54 | Löschung | `ins` | *(fehlt)* | …gewusst dass ich [___] krankenhaus muss hätte… |
| 55 | Löschung | `krankenhaus` | *(fehlt)* | …dass ich ins [___] muss hätte ich… |
| 56 | Löschung | `hätte` | *(fehlt)* | …ins krankenhaus muss [___] ich ihn mitgenommen… |
| 57 | Löschung | `ich` | *(fehlt)* | …krankenhaus muss hätte [___] ihn mitgenommen ja… |
| 58 | Löschung | `ihn` | *(fehlt)* | …muss hätte ich [___] mitgenommen ja ich… |
| 59 | Löschung | `mitgenommen` | *(fehlt)* | …hätte ich ihn [___] ja ich muss… |
| 60 | Löschung | `ich` | *(fehlt)* | …ihn mitgenommen ja [___] muss auch gestehen… |
| 61 | Löschung | `muss` | *(fehlt)* | …mitgenommen ja ich [___] auch gestehen dass… |
| 62 | Löschung | `gestehen` | *(fehlt)* | …ich muss auch [___] dass ich ihn… |
| 63 | Löschung | `dass` | *(fehlt)* | …muss auch gestehen [___] ich ihn nicht… |
| 64 | Löschung | `ich` | *(fehlt)* | …auch gestehen dass [___] ihn nicht bei… |
| 65 | Löschung | `ihn` | *(fehlt)* | …gestehen dass ich [___] nicht bei mir… |
| 66 | Substitution | `becken` | `becker` | …gut okay frau [___] westphalen wie geht… |
| 67 | Substitution | `hab` | `habe` | …zeit nein ich [___] gar keine sonstigen… |
| 68 | Löschung | `bis` | *(fehlt)* | …worüber ich mir [___] jetzt sorgen gemacht… |
| 69 | Löschung | `jetzt` | *(fehlt)* | …ich mir bis [___] sorgen gemacht habe… |
| 70 | Substitution | `kennen` | `können` | …ich glaube das [___] wir auch wirklich… |
| 71 | Substitution | `becken` | `beck` | …okay wunderbar frau [___] westphalen rauchen sie… |
| 72 | Substitution | `westphalen` | `in` | …wunderbar frau becken [___] rauchen sie nein… |
| 73 | Einfügung | *(nicht da)* | `westfalen` | …frau becken westphalen [___] sie nein ich… |
| 74 | Löschung | `ja` | *(fehlt)* | …ist ja aber [___] wann haben sie… |
| 75 | Löschung | `wann` | *(fehlt)* | …ja aber ja [___] haben sie damit… |
| 76 | Löschung | `haben` | *(fehlt)* | …aber ja wann [___] sie damit aufgehört… |
| 77 | Löschung | `sie` | *(fehlt)* | …ja wann haben [___] damit aufgehört das… |
| 78 | Löschung | `damit` | *(fehlt)* | …wann haben sie [___] aufgehört das müssten… |
| 79 | Substitution | `becken` | `beck` | …okay wunderbar frau [___] westphalen nehmen sie… |
| 80 | Substitution | `westphalen` | `in` | …wunderbar frau becken [___] nehmen sie das… |
| 81 | Einfügung | *(nicht da)* | `westfalen` | …frau becken westphalen [___] sie das jetzt… |
| 82 | Löschung | `ja` | *(fehlt)* | …lange arbeiten und [___] habe zwei drei… |
| 83 | Substitution | `wars` | `war` | …genommen okay das [___] aber aber jetzt… |
| 84 | Löschung | `aber` | *(fehlt)* | …okay das wars [___] aber jetzt schon… |
| 85 | Löschung | `aber` | *(fehlt)* | …das wars aber [___] jetzt schon ja… |
| 86 | Löschung | `jetzt` | *(fehlt)* | …wars aber aber [___] schon ja sehr… |
| 87 | Löschung | `schon` | *(fehlt)* | …aber aber jetzt [___] ja sehr gut… |
| 88 | Löschung | `okay` | *(fehlt)* | …großvater hatte leberzirrhose [___] und ist leider… |
| 89 | Substitution | `becken` | `böken` | …sie geschwister frau [___] westphalen ich habe… |
| 90 | Substitution | `becken` | `becker` | …sie kinder frau [___] westphalen nein ich… |
| 91 | Löschung | `ja` | *(fehlt)* | …das ist richtig [___] okay prima wir… |
| 92 | Substitution | `bin` | `wenn` | …in einer marketingagentur [___] da für größere… |
| 93 | Substitution | `becken` | `becker` | …frage noch frau [___] westphalen waren sie… |
| 94 | Substitution | `westphalen` | `westfalen` | …noch frau becken [___] waren sie in… |
| 95 | Substitution | `becken` | `becker` | …okay gut frau [___] westphalen von meiner… |
| 96 | Substitution | `westphalen` | `westfalen` | …gut frau becken [___] von meiner seite… |
| 97 | Substitution | `drüber` | `darüber` | …keinerlei positive auskunft [___] geben das was… |
| 98 | Substitution | `erstmal` | `erst` | …müssen wir wirklich [___] mrt bilder von… |
| 99 | Einfügung | *(nicht da)* | `mal` | …wir wirklich erstmal [___] bilder von machen… |
| 100 | Löschung | `auch` | *(fehlt)* | …von machen und [___] röntgenbilder von machen… |
| 101 | Substitution | `nochmal` | `noch` | …ja alles klar [___] kurz zum abgleich… |
| 102 | Einfügung | *(nicht da)* | `mal` | …alles klar nochmal [___] zum abgleich sie… |
| 103 | Substitution | `becken` | `becker` | …sehr gut frau [___] westphalen dann war… |
| 104 | Substitution | `westphalen` | `westfalen` | …gut frau becken [___] dann war es… |


---

## Anamnesegesrpäch PWC.mp3

**WER: 23.1%** — Referenz: 1530 Wörter | Hypothese: 1452 Wörter | S=149 D=141 I=63 | Edit-Distanz=353

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `grasbäutner` | `glasbildner` | …grüß gott frau [___] ich bin die… |
| 2 | Löschung | `ich` | *(fehlt)* | …gott frau grasbäutner [___] bin die eisel… |
| 3 | Substitution | `eisel` | `eisl` | …ich bin die [___] theresa ihre physiotherapeutin… |
| 4 | Substitution | `theresa` | `teresa` | …bin die eisel [___] ihre physiotherapeutin wir… |
| 5 | Substitution | `ihre` | `eine` | …die eisel theresa [___] physiotherapeutin wir dürfen… |
| 6 | Substitution | `gerne` | `gern` | …wir dürfen sie [___] hinsetzen wir beginnen… |
| 7 | Substitution | `wir` | `sie` | …sie gerne hinsetzen [___] beginnen heute so… |
| 8 | Substitution | `heute` | `heit` | …hinsetzen wir beginnen [___] so dass wir… |
| 9 | Substitution | `wir` | `man` | …heute so dass [___] eine anamnese zuerst… |
| 10 | Löschung | `eine` | *(fehlt)* | …so dass wir [___] anamnese zuerst machen… |
| 11 | Substitution | `wir` | `man` | …beinhaltet einfach dass [___] über ein problem… |
| 12 | Substitution | `möchte` | `müssen` | …werden zu beginn [___] ich ihnen ein… |
| 13 | Substitution | `ich` | `sie` | …zu beginn möchte [___] ihnen ein paar… |
| 14 | Substitution | `ihnen` | `noch` | …beginn möchte ich [___] ein paar fragen… |
| 15 | Substitution | `ihr` | `sehr` | …wäre am anfang [___] alter sie sagen… |
| 16 | Substitution | `sagen` | `sind` | …ihr alter sie [___] 27 27 jahre… |
| 17 | Löschung | `heute` | *(fehlt)* | …ja ist das [___] ihr erster besuch… |
| 18 | Löschung | `ja` | *(fehlt)* | …besuch beim physiotherapeuten [___] sie waren noch… |
| 19 | Löschung | `ja` | *(fehlt)* | …dann können wir [___] schon starten warum… |
| 20 | Löschung | `einen` | *(fehlt)* | …also sie haben [___] kreuzbandriss gehabt wie… |
| 21 | Substitution | `also` | `gesprungen` | …training bin ich [___] ich bin gesprungen… |
| 22 | Substitution | `ich` | `am` | …bin ich also [___] bin gesprungen beim… |
| 23 | Substitution | `bin` | `netz` | …ich also ich [___] gesprungen beim netzfahren… |
| 24 | Substitution | `gesprungen` | `und` | …also ich bin [___] beim netzfahren und… |
| 25 | Substitution | `beim` | `war` | …ich bin gesprungen [___] netzfahren und war… |
| 26 | Substitution | `netzfahren` | `also` | …bin gesprungen beim [___] und war zu… |
| 27 | Substitution | `und` | `ich` | …gesprungen beim netzfahren [___] war zu spät… |
| 28 | Löschung | `zu` | *(fehlt)* | …netzfahren und war [___] spät dran zum… |
| 29 | Substitution | `zum` | `zu` | …zu spät dran [___] block und bin… |
| 30 | Einfügung | *(nicht da)* | `einem` | …spät dran zum [___] und bin schief… |
| 31 | Löschung | `schon` | *(fehlt)* | …bin schief weggesprungen [___] und dann irgendwie… |
| 32 | Substitution | `haben` | `sind` | …dann ja sie [___] dann schief gelandet… |
| 33 | Löschung | `dann` | *(fehlt)* | …ja sie haben [___] schief gelandet und… |
| 34 | Einfügung | *(nicht da)* | `sie` | …schief gelandet und [___] sie dann aufgestanden… |
| 35 | Löschung | `sie` | *(fehlt)* | …gelandet und sind [___] dann aufgestanden oder… |
| 36 | Löschung | `dann` | *(fehlt)* | …nein ich bin [___] eigentlich dagelegen und… |
| 37 | Substitution | `dagelegen` | `tackling` | …bin dann eigentlich [___] und habe einen… |
| 38 | Einfügung | *(nicht da)* | `heute` | …dagelegen und habe [___] wirklich argen schmerz… |
| 39 | Substitution | `minute` | `minutenminuten` | …so eine halbe [___] minute circa dass… |
| 40 | Löschung | `minute` | *(fehlt)* | …eine halbe minute [___] circa dass ich… |
| 41 | Substitution | `mein` | `meine` | …circa dass ich [___] bein nicht ausstrecken… |
| 42 | Substitution | `bein` | `beine` | …dass ich mein [___] nicht ausstrecken können… |
| 43 | Einfügung | *(nicht da)* | `also` | …ich mein bein [___] ausstrecken können habe… |
| 44 | Einfügung | *(nicht da)* | `das` | …ich mein bein [___] ausstrecken können habe… |
| 45 | Einfügung | *(nicht da)* | `ist` | …ich mein bein [___] ausstrecken können habe… |
| 46 | Löschung | `habe` | *(fehlt)* | …nicht ausstrecken können [___] und gar nichts… |
| 47 | Einfügung | *(nicht da)* | `ähm` | …nichts eigentlich und [___] ich habe es… |
| 48 | Löschung | `ein` | *(fehlt)* | …gleich gekühlt und [___] bisschen hoch gelagert… |
| 49 | Löschung | `bisschen` | *(fehlt)* | …gekühlt und ein [___] hoch gelagert und… |
| 50 | Löschung | `bisschen` | *(fehlt)* | …es war ein [___] ein komisches gefühl… |
| 51 | Löschung | `ein` | *(fehlt)* | …war ein bisschen [___] komisches gefühl aber… |
| 52 | Substitution | `ja` | `bin` | …bin ich eigentlich [___] duschen gegangen und… |
| 53 | Einfügung | *(nicht da)* | `ich` | …ich eigentlich ja [___] gegangen und dann… |
| 54 | Substitution | `dann` | `hab` | …duschen gegangen und [___] habe ich mir… |
| 55 | Löschung | `habe` | *(fehlt)* | …gegangen und dann [___] ich mir gedacht… |
| 56 | Löschung | `ich` | *(fehlt)* | …und dann habe [___] mir gedacht ja… |
| 57 | Substitution | `gedacht` | `dort` | …habe ich mir [___] ja zur sicherheit… |
| 58 | Löschung | `ja` | *(fehlt)* | …ich mir gedacht [___] zur sicherheit fahre… |
| 59 | Löschung | `fahre` | *(fehlt)* | …ja zur sicherheit [___] ich halt noch… |
| 60 | Löschung | `ich` | *(fehlt)* | …zur sicherheit fahre [___] halt noch ins… |
| 61 | Löschung | `halt` | *(fehlt)* | …sicherheit fahre ich [___] noch ins krankenhaus… |
| 62 | Löschung | `noch` | *(fehlt)* | …fahre ich halt [___] ins krankenhaus aber… |
| 63 | Löschung | `vorher` | *(fehlt)* | …gehabt sie haben [___] gesagt sie spielen… |
| 64 | Substitution | `spielen` | `spüren` | …vorher gesagt sie [___] also es war… |
| 65 | Substitution | `die` | `in` | …woche ja zweimal [___] woche zweimal die… |
| 66 | Einfügung | *(nicht da)* | `der` | …ja zweimal die [___] zweimal die woche… |
| 67 | Substitution | `ja` | `vier` | …machen sie das [___] ich glaube 4… |
| 68 | Löschung | `ich` | *(fehlt)* | …sie das ja [___] glaube 4 jahre… |
| 69 | Löschung | `glaube` | *(fehlt)* | …das ja ich [___] 4 jahre jetzt… |
| 70 | Löschung | `4` | *(fehlt)* | …ja ich glaube [___] jahre jetzt 4… |
| 71 | Substitution | `jetzt` | `vier` | …glaube 4 jahre [___] 4 jahre und… |
| 72 | Löschung | `4` | *(fehlt)* | …4 jahre jetzt [___] jahre und das… |
| 73 | Löschung | `ja` | *(fehlt)* | …ein hobby ist [___] genau und dass… |
| 74 | Löschung | `genau` | *(fehlt)* | …hobby ist ja [___] und dass sie… |
| 75 | Substitution | `einmal` | `immer` | …sie das letztendlich [___] wieder machen wollen… |
| 76 | Substitution | `auf` | `genau` | …machen wollen ja [___] jeden fall ja… |
| 77 | Löschung | `jeden` | *(fehlt)* | …wollen ja auf [___] fall ja und… |
| 78 | Löschung | `fall` | *(fehlt)* | …ja auf jeden [___] ja und was… |
| 79 | Einfügung | *(nicht da)* | `dann` | …wie das dann [___] dann die schmerzen… |
| 80 | Substitution | `dann` | `da` | …das dann waren [___] die schmerzen dann… |
| 81 | Substitution | `dann` | `da` | …dann haben sie [___] bin ich ins… |
| 82 | Substitution | `einmal` | `da` | …ich ins krankenhaus [___] noch zur sicherheit… |
| 83 | Einfügung | *(nicht da)* | `haben` | …ins krankenhaus einmal [___] zur sicherheit weil… |
| 84 | Einfügung | *(nicht da)* | `wir` | …ins krankenhaus einmal [___] zur sicherheit weil… |
| 85 | Löschung | `getan` | *(fehlt)* | …doch wirklich weh [___] hat und dann… |
| 86 | Substitution | `ein` | `auch` | …und dann war [___] röntgen und der… |
| 87 | Löschung | `der` | *(fehlt)* | …und der arzt [___] mich untersucht hat… |
| 88 | Löschung | `mich` | *(fehlt)* | …der arzt der [___] untersucht hat hat… |
| 89 | Substitution | `hat` | `und` | …der mich untersucht [___] hat dann gesagt… |
| 90 | Löschung | `dann` | *(fehlt)* | …untersucht hat hat [___] gesagt nein ich… |
| 91 | Substitution | `und` | `dann` | …habe eigentlich nichts [___] bin wieder heimgefahren… |
| 92 | Einfügung | *(nicht da)* | `ich` | …nichts und bin [___] heimgefahren und da… |
| 93 | Substitution | `heimgefahren` | `hingefahren` | …und bin wieder [___] und da war… |
| 94 | Löschung | `mit` | *(fehlt)* | …da war das [___] da habe ich… |
| 95 | Substitution | `habe` | `hab` | …das mit da [___] ich keine argen… |
| 96 | Löschung | `argen` | *(fehlt)* | …habe ich keine [___] schmerzen gehabt und… |
| 97 | Substitution | `habe` | `hab` | …in der früh [___] ich nicht mehr… |
| 98 | Löschung | `ich` | *(fehlt)* | …aufstehen können weil [___] eben solche schmerzen… |
| 99 | Substitution | `habe` | `haben` | …solche schmerzen gehabt [___] und dann war… |
| 100 | Löschung | `auch` | *(fehlt)* | …dann war es [___] angeschwollen das knie… |
| 101 | Löschung | `dann` | *(fehlt)* | …arzt der mich [___] untersucht hat hat… |
| 102 | Löschung | `ja` | *(fehlt)* | …hat gleich gesagt [___] dass das eindeutig… |
| 103 | Substitution | `also` | `es` | …kreuzband gerissen ist [___] war mrt gemacht… |
| 104 | Substitution | `mrt` | `im` | …ist also war [___] gemacht worden ist… |
| 105 | Einfügung | *(nicht da)* | `mai` | …also war mrt [___] worden ist und… |
| 106 | Löschung | `dass` | *(fehlt)* | …sie wissen auch [___] das vordere kreuzband… |
| 107 | Substitution | `habe` | `hab` | …ja und dann [___] ich mit dem… |
| 108 | Substitution | `mit` | `und` | …dann habe ich [___] dem arzt ausgemacht… |
| 109 | Substitution | `dem` | `dann` | …habe ich mit [___] arzt ausgemacht mit… |
| 110 | Substitution | `arzt` | `hat` | …ich mit dem [___] ausgemacht mit einem… |
| 111 | Einfügung | *(nicht da)* | `es` | …mit dem arzt [___] mit einem anderen… |
| 112 | Einfügung | *(nicht da)* | `mich` | …mit dem arzt [___] mit einem anderen… |
| 113 | Einfügung | *(nicht da)* | `nicht` | …mit dem arzt [___] mit einem anderen… |
| 114 | Einfügung | *(nicht da)* | `mehr` | …mit dem arzt [___] mit einem anderen… |
| 115 | Löschung | `gleich` | *(fehlt)* | …operationstermin eigentlich auch [___] vereinbart ah ja… |
| 116 | Löschung | `jetzt` | *(fehlt)* | …die war dann [___] vor ca einem… |
| 117 | Substitution | `ca` | `circa` | …dann jetzt vor [___] einem monat am… |
| 118 | Löschung | `wie` | *(fehlt)* | …sie das mitbekommen [___] war der heilungsprozess… |
| 119 | Löschung | `war` | *(fehlt)* | …das mitbekommen wie [___] der heilungsprozess also… |
| 120 | Löschung | `also` | *(fehlt)* | …war der heilungsprozess [___] es ist mir… |
| 121 | Löschung | `es` | *(fehlt)* | …der heilungsprozess also [___] ist mir gesagt… |
| 122 | Einfügung | *(nicht da)* | `ähm` | …bei der operation [___] ich bin halt… |
| 123 | Löschung | `halt` | *(fehlt)* | …ja ich bin [___] dann noch glaube… |
| 124 | Einfügung | *(nicht da)* | `nachher` | …bin halt dann [___] glaube ich eine… |
| 125 | Löschung | `glaube` | *(fehlt)* | …halt dann noch [___] ich eine woche… |
| 126 | Löschung | `ich` | *(fehlt)* | …dann noch glaube [___] eine woche im… |
| 127 | Substitution | `hab` | `habe` | …im krankenhaus gewesen [___] halt nicht viel… |
| 128 | Löschung | `halt` | *(fehlt)* | …krankenhaus gewesen hab [___] nicht viel bewegung… |
| 129 | Löschung | `einfach` | *(fehlt)* | …viel bewegung gehabt [___] und durch das… |
| 130 | Löschung | `halt` | *(fehlt)* | …ganze liegen ist [___] der muskel komplett… |
| 131 | Einfügung | *(nicht da)* | `ähm` | …und das ja [___] war das dann… |
| 132 | Substitution | `dann` | `denn` | …wie war das [___] im krankenhaus war… |
| 133 | Substitution | `eine` | `ein` | …krankenhaus war da [___] physiotherapeutin dann eingeteilt… |
| 134 | Substitution | `physiotherapeutin` | `physiotherapeut` | …war da eine [___] dann eingeteilt u… |
| 135 | Substitution | `dann` | `bei` | …da eine physiotherapeutin [___] eingeteilt u ja… |
| 136 | Einfügung | *(nicht da)* | `ihnen` | …eine physiotherapeutin dann [___] u ja da… |
| 137 | Löschung | `u` | *(fehlt)* | …physiotherapeutin dann eingeteilt [___] ja da war… |
| 138 | Löschung | `eine` | *(fehlt)* | …ja da war [___] also waren sie… |
| 139 | Einfügung | *(nicht da)* | `ähm` | …es so dass [___] das ich glaube… |
| 140 | Löschung | `ich` | *(fehlt)* | …dass aber das [___] glaube der ist… |
| 141 | Löschung | `glaube` | *(fehlt)* | …aber das ich [___] der ist halt… |
| 142 | Löschung | `der` | *(fehlt)* | …das ich glaube [___] ist halt auch… |
| 143 | Substitution | `halt` | `an` | …glaube der ist [___] auch mit dem… |
| 144 | Löschung | `auch` | *(fehlt)* | …der ist halt [___] mit dem doc… |
| 145 | Löschung | `mit` | *(fehlt)* | …ist halt auch [___] dem doc gekommen… |
| 146 | Substitution | `doc` | `tag` | …auch mit dem [___] gekommen und das… |
| 147 | Einfügung | *(nicht da)* | `für` | …doc gekommen und [___] war da die… |
| 148 | Substitution | `da` | `auch` | …und das war [___] die einzige bewegung… |
| 149 | Löschung | `eigentlich` | *(fehlt)* | …die ich dort [___] gehabt habe mit… |
| 150 | Einfügung | *(nicht da)* | `da` | …da haben sie [___] alles ja hauptsächlich… |
| 151 | Substitution | `alles` | `was` | …haben sie schon [___] ja hauptsächlich hat… |
| 152 | Substitution | `ja` | `ihr` | …sie schon alles [___] hauptsächlich hat man… |
| 153 | Einfügung | *(nicht da)* | `mit` | …alles ja hauptsächlich [___] man gesagt wie… |
| 154 | Einfügung | *(nicht da)* | `ja` | …alles ja hauptsächlich [___] man gesagt wie… |
| 155 | Substitution | `gesagt` | `sagt` | …hauptsächlich hat man [___] wie man mit… |
| 156 | Löschung | `wie` | *(fehlt)* | …hat man gesagt [___] man mit den… |
| 157 | Löschung | `man` | *(fehlt)* | …man gesagt wie [___] mit den krücken… |
| 158 | Löschung | `also` | *(fehlt)* | …aufsteigen und das [___] das also dieses… |
| 159 | Löschung | `das` | *(fehlt)* | …und das also [___] also dieses abrollen… |
| 160 | Löschung | `also` | *(fehlt)* | …das also das [___] dieses abrollen mit… |
| 161 | Löschung | `dieses` | *(fehlt)* | …also das also [___] abrollen mit dem… |
| 162 | Substitution | `einlernen` | `ein` | …fuß auch richtig [___] wieder ich glaube… |
| 163 | Einfügung | *(nicht da)* | `lernen` | …auch richtig einlernen [___] ich glaube am… |
| 164 | Löschung | `ein` | *(fehlt)* | …am schluss sogar [___] bisschen haben wir… |
| 165 | Löschung | `bisschen` | *(fehlt)* | …schluss sogar ein [___] haben wir treppensteigen… |
| 166 | Löschung | `haben` | *(fehlt)* | …sogar ein bisschen [___] wir treppensteigen dann… |
| 167 | Löschung | `wir` | *(fehlt)* | …ein bisschen haben [___] treppensteigen dann dass… |
| 168 | Löschung | `halt` | *(fehlt)* | …dann dass ich [___] ja und ein… |
| 169 | Löschung | `so` | *(fehlt)* | …und ein bisschen [___] beugen üben so… |
| 170 | Löschung | `halt` | *(fehlt)* | …so weit es [___] gegangen ist und… |
| 171 | Substitution | `da` | `ja` | …sie haben das [___] schon eben sie… |
| 172 | Löschung | `sie` | *(fehlt)* | …da schon eben [___] haben gesagt sie… |
| 173 | Löschung | `haben` | *(fehlt)* | …schon eben sie [___] gesagt sie haben… |
| 174 | Substitution | `eben` | `immer` | …gesagt sie haben [___] mit den stützen… |
| 175 | Substitution | `mit` | `die` | …sie haben eben [___] den stützen das… |
| 176 | Löschung | `den` | *(fehlt)* | …haben eben mit [___] stützen das gelernt… |
| 177 | Substitution | `das` | `des` | …mit den stützen [___] gelernt zum gehen… |
| 178 | Löschung | `dann` | *(fehlt)* | …den sie jetzt [___] wieder haben integriert… |
| 179 | Einfügung | *(nicht da)* | `ist` | …schon also das [___] jeden fall dass… |
| 180 | Substitution | `ich` | `sie` | …jeden fall dass [___] darauf achte wie… |
| 181 | Substitution | `achte` | `achten` | …dass ich darauf [___] wie ich aufsteige… |
| 182 | Substitution | `ich` | `aufsteigen` | …darauf achte wie [___] aufsteige genau das… |
| 183 | Löschung | `aufsteige` | *(fehlt)* | …achte wie ich [___] genau das hat… |
| 184 | Löschung | `genau` | *(fehlt)* | …wie ich aufsteige [___] das hat ihnen… |
| 185 | Löschung | `ihnen` | *(fehlt)* | …genau das hat [___] zum beispiel schon… |
| 186 | Substitution | `also` | `weil` | …beispiel schon geholfen [___] sie haben da… |
| 187 | Löschung | `ja` | *(fehlt)* | …therapie und ja [___] das war halt… |
| 188 | Substitution | `es` | `denn` | …und wie ist [___] dann ergangen dann… |
| 189 | Löschung | `ergangen` | *(fehlt)* | …ist es dann [___] dann mit den… |
| 190 | Löschung | `dann` | *(fehlt)* | …es dann ergangen [___] mit den schmerzen… |
| 191 | Substitution | `den` | `schmerzen` | …den schmerzen mit [___] ja schmerzen war… |
| 192 | Substitution | `ja` | `wo` | …schmerzen mit den [___] schmerzen war ja… |
| 193 | Substitution | `schmerzen` | `er` | …mit den ja [___] war ja war… |
| 194 | Löschung | `war` | *(fehlt)* | …den ja schmerzen [___] ja war okay… |
| 195 | Einfügung | *(nicht da)* | `war` | …sag ich mal [___] nachdem je nach… |
| 196 | Einfügung | *(nicht da)* | `okay` | …sag ich mal [___] nachdem je nach… |
| 197 | Einfügung | *(nicht da)* | `belastung` | …je nach belastung [___] habe mich halt… |
| 198 | Einfügung | *(nicht da)* | `es` | …je nach belastung [___] habe mich halt… |
| 199 | Einfügung | *(nicht da)* | `war` | …je nach belastung [___] habe mich halt… |
| 200 | Einfügung | *(nicht da)* | `halt` | …je nach belastung [___] habe mich halt… |
| 201 | Substitution | `habe` | `hab` | …nach belastung ich [___] mich halt nicht… |
| 202 | Einfügung | *(nicht da)* | `ich` | …viel bewegen können [___] ja eigentlich nur… |
| 203 | Substitution | `die` | `in` | …die erste woche [___] letzte zeit und… |
| 204 | Einfügung | *(nicht da)* | `der` | …erste woche die [___] zeit und dann… |
| 205 | Substitution | `halt` | `herumgehen` | …mit den krücken [___] herumgegangen aber halt… |
| 206 | Substitution | `herumgegangen` | `ein` | …den krücken halt [___] aber halt auch… |
| 207 | Einfügung | *(nicht da)* | `bisschen` | …krücken halt herumgegangen [___] halt auch minimal… |
| 208 | Löschung | `jetzt` | *(fehlt)* | …dann sind wir [___] schon so weit… |
| 209 | Substitution | `wir` | `man` | …so weit dass [___] mal darüber reden… |
| 210 | Einfügung | *(nicht da)* | `wie` | …jetzt geht wie [___] es ihnen wenn… |
| 211 | Substitution | `sie` | `es` | …es ihnen wenn [___] an die schmerzen… |
| 212 | Substitution | `10` | `zehn` | …bis 10 und [___] sind die schlimmsten… |
| 213 | Einfügung | *(nicht da)* | `sie` | …schlimmsten schmerzen die [___] vorstellen können und… |
| 214 | Substitution | `0` | `null` | …vorstellen können und [___] sind ist schmerzfrei… |
| 215 | Löschung | `ist` | *(fehlt)* | …und 0 sind [___] schmerzfrei wo würden… |
| 216 | Substitution | `sich` | `sie` | …wo würden sie [___] da eingliedern ja… |
| 217 | Substitution | `eigentlich` | `darauf` | …gesagt es kommt [___] drauf an auf… |
| 218 | Löschung | `drauf` | *(fehlt)* | …es kommt eigentlich [___] an auf die… |
| 219 | Substitution | `drauf` | `darauf` | …auf die belastung [___] an wenn ich… |
| 220 | Substitution | `1` | `eins` | …sage ich vielleicht [___] aber wenn ich… |
| 221 | Löschung | `den` | *(fehlt)* | …ich jetzt mit [___] krücken gehe dann… |
| 222 | Substitution | `3` | `drei` | …dann keine ahnung [___] und wenn ich… |
| 223 | Einfügung | *(nicht da)* | `wenn` | …und wenn ich [___] ohne stützen probiere… |
| 224 | Einfügung | *(nicht da)* | `ich` | …und wenn ich [___] ohne stützen probiere… |
| 225 | Substitution | `probiere` | `probiert` | …wirklich ohne stützen [___] zu gehen dann… |
| 226 | Substitution | `zu` | `zum` | …ohne stützen probiere [___] gehen dann bin… |
| 227 | Einfügung | *(nicht da)* | `gehen` | …probiere zu gehen [___] bin ich sicher… |
| 228 | Substitution | `6` | `sechs` | …ich sicher bei [___] oder 7 3… |
| 229 | Substitution | `7` | `sieben` | …bei 6 oder [___] 3 6 7… |
| 230 | Substitution | `3` | `7367` | …6 oder 7 [___] 6 7 aber… |
| 231 | Löschung | `6` | *(fehlt)* | …oder 7 3 [___] 7 aber es… |
| 232 | Löschung | `7` | *(fehlt)* | …7 3 6 [___] aber es ist… |
| 233 | Löschung | `halt` | *(fehlt)* | …je nach belastung [___] und das ist… |
| 234 | Einfügung | *(nicht da)* | `ja` | …und das ist [___] einzige faktor die… |
| 235 | Substitution | `ihnen` | `einem` | …die belastung der [___] da einfällt wenn… |
| 236 | Substitution | `sie` | `es` | …da einfällt wenn [___] an den schmerz… |
| 237 | Substitution | `an` | `um` | …einfällt wenn sie [___] den schmerz denken… |
| 238 | Löschung | `den` | *(fehlt)* | …wenn sie an [___] schmerz denken dass… |
| 239 | Substitution | `sich` | `sie` | …schmerz denken dass [___] der da verändert… |
| 240 | Substitution | `ich` | `wie` | …eigentlich ja also [___] sehe das nicht… |
| 241 | Substitution | `sehe` | `ist` | …ja also ich [___] das nicht anders… |
| 242 | Substitution | `nicht` | `mit` | …ich sehe das [___] anders ansonsten okay… |
| 243 | Substitution | `anders` | `und` | …sehe das nicht [___] ansonsten okay und… |
| 244 | Substitution | `ansonsten` | `was` | …das nicht anders [___] okay und sie… |
| 245 | Einfügung | *(nicht da)* | `sonst` | …nicht anders ansonsten [___] und sie haben… |
| 246 | Einfügung | *(nicht da)* | `noch` | …nicht anders ansonsten [___] und sie haben… |
| 247 | Substitution | `den` | `die` | …dem gehen mit [___] stützen das funktioniert… |
| 248 | Substitution | `sind` | `haben` | …vorstellen also sie [___] ab und zu… |
| 249 | Einfügung | *(nicht da)* | `dann` | …also sie sind [___] und zu rausgegangen… |
| 250 | Substitution | `auch` | `einen` | …das ist jetzt [___] ein monat her… |
| 251 | Löschung | `ein` | *(fehlt)* | …ist jetzt auch [___] monat her also… |
| 252 | Einfügung | *(nicht da)* | `und` | …ein monat her [___] nicht wirklich ich… |
| 253 | Substitution | `ich` | `immer` | …also nicht wirklich [___] meine minimal einfluss… |
| 254 | Löschung | `meine` | *(fehlt)* | …nicht wirklich ich [___] minimal einfluss aber… |
| 255 | Substitution | `einfluss` | `einfach` | …ich meine minimal [___] aber ich kann… |
| 256 | Einfügung | *(nicht da)* | `mir` | …aber ich kann [___] wirklich sagen ich… |
| 257 | Einfügung | *(nicht da)* | `dass` | …kann nicht wirklich [___] ich gehe jetzt… |
| 258 | Einfügung | *(nicht da)* | `ich` | …kann nicht wirklich [___] ich gehe jetzt… |
| 259 | Einfügung | *(nicht da)* | `kann` | …nicht wirklich sagen [___] gehe jetzt spazieren… |
| 260 | Löschung | `jetzt` | *(fehlt)* | …sagen ich gehe [___] spazieren oder so… |
| 261 | Löschung | `halt` | *(fehlt)* | …ich bewege mich [___] in der wohnung… |
| 262 | Substitution | `versuche` | `versucht` | …nötigste und ja [___] halt am heimtrainer… |
| 263 | Substitution | `halt` | `auf` | …und ja versuche [___] am heimtrainer ab… |
| 264 | Substitution | `am` | `dem` | …ja versuche halt [___] heimtrainer ab und… |
| 265 | Substitution | `das` | `wir` | …möglich beugen und [___] eigentlich immer unter… |
| 266 | Einfügung | *(nicht da)* | `sind` | …beugen und das [___] immer unter schmerzen… |
| 267 | Löschung | `nein` | *(fehlt)* | …richtig schmerzfrei möglich [___] nehmen sie irgendwelche… |
| 268 | Löschung | `nehme` | *(fehlt)* | …irgendwelche medikamente nein [___] ich nicht nehmen… |
| 269 | Löschung | `ich` | *(fehlt)* | …medikamente nein nehme [___] nicht nehmen sie… |
| 270 | Löschung | `nicht` | *(fehlt)* | …nein nehme ich [___] nehmen sie nicht… |
| 271 | Löschung | `am` | *(fehlt)* | …nicht haben sie [___] anfangs aber wahrscheinlich… |
| 272 | Löschung | `schmerzmittel` | *(fehlt)* | …in der behandlung [___] bekommen ja ich… |
| 273 | Löschung | `bekommen` | *(fehlt)* | …der behandlung schmerzmittel [___] ja ich habe… |
| 274 | Substitution | `am` | `anfangs` | …ja ich habe [___] anfang schmerzmittel gekriegt… |
| 275 | Löschung | `anfang` | *(fehlt)* | …ich habe am [___] schmerzmittel gekriegt am… |
| 276 | Löschung | `noch` | *(fehlt)* | …anfang sowieso infusionen [___] dann hätte ich… |
| 277 | Löschung | `noch` | *(fehlt)* | …dann hätte ich [___] schmerzmittel mitgehabt für… |
| 278 | Substitution | `mitgehabt` | `mitgekriegt` | …ich noch schmerzmittel [___] für daheim aber… |
| 279 | Substitution | `für` | `haben` | …noch schmerzmittel mitgehabt [___] daheim aber die… |
| 280 | Löschung | `daheim` | *(fehlt)* | …schmerzmittel mitgehabt für [___] aber die habe… |
| 281 | Substitution | `habe` | `hab` | …daheim aber die [___] ich dann eigentlich… |
| 282 | Substitution | `gebraucht` | `benötigt` | …eigentlich nicht mehr [___] also die mussten… |
| 283 | Substitution | `also` | `gemacht` | …nicht mehr gebraucht [___] die mussten sie… |
| 284 | Löschung | `die` | *(fehlt)* | …mehr gebraucht also [___] mussten sie nicht… |
| 285 | Löschung | `mussten` | *(fehlt)* | …gebraucht also die [___] sie nicht nehmen… |
| 286 | Löschung | `sie` | *(fehlt)* | …also die mussten [___] nicht nehmen mit… |
| 287 | Löschung | `nicht` | *(fehlt)* | …die mussten sie [___] nehmen mit dem… |
| 288 | Löschung | `nehmen` | *(fehlt)* | …mussten sie nicht [___] mit dem home… |
| 289 | Substitution | `home` | `hometrainer` | …nehmen mit dem [___] trainer haben sie… |
| 290 | Löschung | `trainer` | *(fehlt)* | …mit dem home [___] haben sie erwähnt… |
| 291 | Substitution | `ne` | `nein` | …gemacht für übungen [___] keine eigentlich nur… |
| 292 | Löschung | `keine` | *(fehlt)* | …für übungen ne [___] eigentlich nur versucht… |
| 293 | Einfügung | *(nicht da)* | `also` | …habe drinnen halt [___] da ist ihnen… |
| 294 | Substitution | `ihnen` | `schon` | …und da ist [___] aber aufgefallen dass… |
| 295 | Löschung | `aber` | *(fehlt)* | …da ist ihnen [___] aufgefallen dass es… |
| 296 | Substitution | `ja` | `die` | …bisschen weitergegangen ist [___] auf jeden fall… |
| 297 | Einfügung | *(nicht da)* | `bewegung` | …weitergegangen ist ja [___] jeden fall besser… |
| 298 | Einfügung | *(nicht da)* | `ist` | …weitergegangen ist ja [___] jeden fall besser… |
| 299 | Löschung | `auch` | *(fehlt)* | …also sie haben [___] fortschritte dann bemerkt… |
| 300 | Löschung | `dann` | *(fehlt)* | …haben auch fortschritte [___] bemerkt ja nur… |
| 301 | Substitution | `ja` | `nun` | …fortschritte dann bemerkt [___] nur zu ihrer… |
| 302 | Löschung | `nur` | *(fehlt)* | …dann bemerkt ja [___] zu ihrer wohnsituation… |
| 303 | Substitution | `es` | `sie` | …zweiten stock also [___] ist in dem… |
| 304 | Substitution | `ist` | `sind` | …stock also es [___] in dem zweiten… |
| 305 | Substitution | `in` | `im` | …also es ist [___] dem zweiten stock… |
| 306 | Löschung | `dem` | *(fehlt)* | …es ist in [___] zweiten stock das… |
| 307 | Substitution | `viele` | `viel` | …das heißt wie [___] treppen werden das… |
| 308 | Löschung | `treppen` | *(fehlt)* | …20 bis 30 [___] 20 bis 30… |
| 309 | Substitution | `nie` | `nicht` | …bis jetzt noch [___] so funktioniert haben… |
| 310 | Substitution | `ich` | `man` | …es ist halt [___] mein sicher funktioniert… |
| 311 | Löschung | `mein` | *(fehlt)* | …ist halt ich [___] sicher funktioniert s… |
| 312 | Substitution | `s` | `es` | …mein sicher funktioniert [___] aber ich überleg… |
| 313 | Substitution | `überleg` | `überlege` | …s aber ich [___] mir halt ob… |
| 314 | Löschung | `halt` | *(fehlt)* | …ich überleg mir [___] ob ich jetzt… |
| 315 | Substitution | `muss` | `laufen` | …jetzt wirklich runter [___] oder rauf muss… |
| 316 | Löschung | `oder` | *(fehlt)* | …wirklich runter muss [___] rauf muss haben… |
| 317 | Löschung | `rauf` | *(fehlt)* | …runter muss oder [___] muss haben sie… |
| 318 | Löschung | `die` | *(fehlt)* | …ihnen hilft also [___] so ja ja… |
| 319 | Löschung | `so` | *(fehlt)* | …hilft also die [___] ja ja ja… |
| 320 | Löschung | `ja` | *(fehlt)* | …die so ja [___] ja also familie… |
| 321 | Löschung | `ja` | *(fehlt)* | …so ja ja [___] also familie freunde… |
| 322 | Substitution | `nein` | `na` | …beispiel diabetes oder [___] dass sie wüssten… |
| 323 | Löschung | `dass` | *(fehlt)* | …diabetes oder nein [___] sie wüssten narconabhängigkeiten… |
| 324 | Substitution | `wüssten` | `wissen` | …nein dass sie [___] narconabhängigkeiten nein sie… |
| 325 | Substitution | `narconabhängigkeiten` | `schon` | …dass sie wüssten [___] nein sie stehen… |
| 326 | Substitution | `nein` | `auffälligkeiten` | …sie wüssten narconabhängigkeiten [___] sie stehen ja… |
| 327 | Einfügung | *(nicht da)* | `und` | …wüssten narconabhängigkeiten nein [___] stehen ja sonst… |
| 328 | Substitution | `stehen` | `denken` | …narconabhängigkeiten nein sie [___] ja sonst nicht… |
| 329 | Substitution | `ja` | `dass` | …nein sie stehen [___] sonst nicht unter… |
| 330 | Substitution | `sonst` | `uns` | …sie stehen ja [___] nicht unter medikamenten… |
| 331 | Substitution | `nicht` | `mitunter` | …stehen ja sonst [___] unter medikamenten also… |
| 332 | Substitution | `unter` | `medikamente` | …ja sonst nicht [___] medikamenten also abgesehen… |
| 333 | Löschung | `medikamenten` | *(fehlt)* | …sonst nicht unter [___] also abgesehen von… |
| 334 | Substitution | `schmerz` | `schmerzen` | …abgesehen von den [___] nein tabletten nein… |
| 335 | Löschung | `nein` | *(fehlt)* | …von den schmerz [___] tabletten nein die… |
| 336 | Substitution | `nein` | `nadine` | …schmerz nein tabletten [___] die nehme ich… |
| 337 | Substitution | `die` | `ja` | …nein tabletten nein [___] nehme ich nicht… |
| 338 | Substitution | `nehme` | `ja` | …tabletten nein die [___] ich nicht mehr… |
| 339 | Löschung | `ich` | *(fehlt)* | …nein die nehme [___] nicht mehr okay… |
| 340 | Löschung | `nicht` | *(fehlt)* | …die nehme ich [___] mehr okay und… |
| 341 | Löschung | `mehr` | *(fehlt)* | …nehme ich nicht [___] okay und dann… |
| 342 | Substitution | `sich` | `sie` | …was erwarten sie [___] dass ich halt… |
| 343 | Löschung | `halt` | *(fehlt)* | …sich dass ich [___] alles wieder normal… |
| 344 | Substitution | `sport` | `sporteln` | …dass ich normal [___] machen kann also… |
| 345 | Löschung | `machen` | *(fehlt)* | …ich normal sport [___] kann also der… |
| 346 | Löschung | `wieder` | *(fehlt)* | …allem möchte ich [___] normal gehen können… |
| 347 | Einfügung | *(nicht da)* | `können` | …normal gehen können [___] vielen dank frau… |
| 348 | Einfügung | *(nicht da)* | `ja` | …normal gehen können [___] vielen dank frau… |
| 349 | Substitution | `grasbäutner` | `gerstner` | …vielen dank frau [___] und wir treffen… |
| 350 | Einfügung | *(nicht da)* | `ob` | …nächsten behandlung danke [___] … |
| 351 | Einfügung | *(nicht da)* | `das` | …nächsten behandlung danke [___] … |
| 352 | Einfügung | *(nicht da)* | `jetzt` | …nächsten behandlung danke [___] … |
| 353 | Einfügung | *(nicht da)* | `passt` | …nächsten behandlung danke [___] … |
