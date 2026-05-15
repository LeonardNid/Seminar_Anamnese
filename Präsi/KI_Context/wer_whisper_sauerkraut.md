# WER-Analyse: Whisper Large-v3-turbo + SauerkrautLM 8b

> Vollständige Fehleranalyse: STT-Rohausgabe vs. Ground Truth
> Satzzeichen und Groß-/Kleinschreibung werden ignoriert.
>
> **S** = Substitution, **D** = Löschung (fehlt im STT), **I** = Einfügung (extra im STT)
>
> Ref-Wörter (Referenz) = Anzahl der Wörter im Ground-Truth-Text — das ist die korrekte Vorlage, gegen die verglichen wird.  
>             
> Hyp-Wörter (Hypothese) = Anzahl der Wörter im STT-Rohausgabe — das ist was Whisper tatsächlich erkannt hat.                           
>                                                                                                                                           
> Die WER-Formel ist:
>                                                                                                                                           
> WER = Edit-Distanz / Ref-Wörter × 100%                                                                                                    
>                                                                                                                                           
> Also: wie viele Korrekturen (Substitutionen + Löschungen + Einfügungen) braucht man, um aus der STT-Ausgabe den Ground-Truth-Text zu machen — geteilt durch die Länge des 
> Ground-Truth-Textes.                                                                                 
>                                                                                                                                      
> Beispiel aus dem Report: OriginalLapInMitte hat 231 Ref-Wörter und 226 Hyp-Wörter → Whisper hat 5 Wörter weniger erkannt als gesprochen wurden, und WER = 15/231 = 6.5%.

## Übersicht

| Audiodatei | Ref-Wörter | Hyp-Wörter | S | D | I | Edit-Dist | WER |
|-----------|-----------|-----------|---|---|---|-----------|-----|
| OriginalDC.m4a | 232 | 238 | 8 | 3 | 9 | 20 | **8.6%** |
| OriginalDCWhiteNoise.m4a | 229 | 210 | 59 | 27 | 8 | 94 | **41.0%** |
| OriginalLapInMitte.wav | 231 | 226 | 8 | 6 | 1 | 15 | **6.5%** |
| OriginalLapBeiArzt.wav | 226 | 229 | 12 | 2 | 5 | 19 | **8.4%** |
| SelbstkorrekturLapInMitte.wav | 183 | 211 | 11 | 0 | 28 | 39 | **21.3%** |
| UnterbrechungLapInMitte.wav | 153 | 143 | 8 | 11 | 1 | 20 | **13.1%** |
| GedankenprüngeLapInMitte.wav | 192 | 190 | 3 | 2 | 0 | 5 | **2.6%** |
| MeinungswechselLapinMitte.wav | 179 | 186 | 3 | 2 | 9 | 14 | **7.8%** |
| ChaosLapInMitte.wav | 272 | 252 | 15 | 24 | 4 | 43 | **15.8%** |
| Das Anamnesegespräch.wav | 2317 | 2270 | 77 | 60 | 13 | 150 | **6.5%** |
| Anamnesegesrpäch PWC.mp3 | 1530 | 1516 | 133 | 102 | 88 | 323 | **21.1%** |


---

## OriginalDC.m4a

**WER: 8.6%** — Referenz: 232 Wörter | Hypothese: 238 Wörter | S=8 D=3 I=9 | Edit-Distanz=20

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `subfebrile` | `subfibrile` | …thermometer okay notiz [___] bis febrile temperaturen… |
| 2 | Substitution | `febrile` | `fibrile` | …notiz subfebrile bis [___] temperaturen maximum 38… |
| 3 | Substitution | `das` | `es` | …hoch oder ist [___] eher ein trockener… |
| 4 | Substitution | `schlechter` | `schlechte` | …gefühl dass sie [___] luft bekommen also… |
| 5 | Einfügung | *(nicht da)* | `auch` | …doll und muss [___] der hälfte kurz… |
| 6 | Substitution | `ruhedyspnoe` | `ruhrdyspnoe` | …belastungsdyspnoe aber keine [___] keine thorakalen schmerzen… |
| 7 | Substitution | `kamillentee` | `kamellentee` | …und trinke viel [___] alles klar notiz… |
| 8 | Substitution | `supportive` | `suprative` | …spezifische vormedikation lediglich [___] hausmittel ich werde… |
| 9 | Löschung | `die` | *(fehlt)* | …ich werde jetzt [___] lunge abhören ihre… |
| 10 | Löschung | `lunge` | *(fehlt)* | …werde jetzt die [___] abhören ihre lunge… |
| 11 | Löschung | `abhören` | *(fehlt)* | …jetzt die lunge [___] ihre lunge abhören… |
| 12 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |
| 13 | Einfügung | *(nicht da)* | `speaker_??` | …auskultation der pulmones [___] … |
| 14 | Einfügung | *(nicht da)* | `auskultation` | …auskultation der pulmones [___] … |
| 15 | Einfügung | *(nicht da)* | `der` | …auskultation der pulmones [___] … |
| 16 | Einfügung | *(nicht da)* | `pulmonisse` | …auskultation der pulmones [___] … |
| 17 | Einfügung | *(nicht da)* | `untertitelung` | …auskultation der pulmones [___] … |
| 18 | Einfügung | *(nicht da)* | `des` | …auskultation der pulmones [___] … |
| 19 | Einfügung | *(nicht da)* | `zdf` | …auskultation der pulmones [___] … |
| 20 | Einfügung | *(nicht da)* | `2020` | …auskultation der pulmones [___] … |


---

## OriginalDCWhiteNoise.m4a

**WER: 41.0%** — Referenz: 229 Wörter | Hypothese: 210 Wörter | S=59 D=27 I=8 | Edit-Distanz=94

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `huste` | `bin` | …herr doktor ich [___] seit knapp einer… |
| 2 | Substitution | `seit` | `heute` | …doktor ich huste [___] knapp einer woche… |
| 3 | Substitution | `knapp` | `in` | …ich huste seit [___] einer woche ununterbrochen… |
| 4 | Substitution | `ununterbrochen` | `unterbrochen` | …knapp einer woche [___] und mir ist… |
| 5 | Substitution | `und` | `es` | …einer woche ununterbrochen [___] mir ist ständig… |
| 6 | Löschung | `mir` | *(fehlt)* | …woche ununterbrochen und [___] ist ständig heiß… |
| 7 | Substitution | `rezidivierendes` | `präzidivierende` | …tagen verdacht auf [___] fieber haben sie… |
| 8 | Substitution | `fieber` | `tiber` | …verdacht auf rezidivierendes [___] haben sie das… |
| 9 | Substitution | `waren` | `war` | …ja gestern abend [___] es 38 8… |
| 10 | Substitution | `38` | `nachgekreist` | …abend waren es [___] 8 grad auf… |
| 11 | Substitution | `8` | `ich` | …waren es 38 [___] grad auf dem… |
| 12 | Substitution | `grad` | `war` | …es 38 8 [___] auf dem thermometer… |
| 13 | Substitution | `auf` | `gerade` | …38 8 grad [___] dem thermometer okay… |
| 14 | Substitution | `dem` | `in` | …8 grad auf [___] thermometer okay notiz… |
| 15 | Substitution | `thermometer` | `der` | …grad auf dem [___] okay notiz subfebrile… |
| 16 | Einfügung | *(nicht da)* | `karte` | …auf dem thermometer [___] notiz subfebrile bis… |
| 17 | Substitution | `subfebrile` | `subfibrile` | …thermometer okay notiz [___] bis febrile temperaturen… |
| 18 | Substitution | `febrile` | `fibride` | …notiz subfebrile bis [___] temperaturen maximum 38… |
| 19 | Substitution | `da` | `der` | …sie husten kommt [___] schleim mit hoch… |
| 20 | Substitution | `reizhusten` | `reiz` | …eher ein trockener [___] da kommt richtig… |
| 21 | Löschung | `da` | *(fehlt)* | …ein trockener reizhusten [___] kommt richtig viel… |
| 22 | Löschung | `kommt` | *(fehlt)* | …trockener reizhusten da [___] richtig viel hoch… |
| 23 | Löschung | `richtig` | *(fehlt)* | …reizhusten da kommt [___] viel hoch das… |
| 24 | Löschung | `viel` | *(fehlt)* | …da kommt richtig [___] hoch das ist… |
| 25 | Löschung | `hoch` | *(fehlt)* | …kommt richtig viel [___] das ist so… |
| 26 | Löschung | `so` | *(fehlt)* | …hoch das ist [___] dickflüssig und eher… |
| 27 | Löschung | `dickflüssig` | *(fehlt)* | …das ist so [___] und eher gelblich… |
| 28 | Löschung | `und` | *(fehlt)* | …ist so dickflüssig [___] eher gelblich grün… |
| 29 | Löschung | `eher` | *(fehlt)* | …so dickflüssig und [___] gelblich grün würde… |
| 30 | Löschung | `gelblich` | *(fehlt)* | …dickflüssig und eher [___] grün würde sagen… |
| 31 | Löschung | `grün` | *(fehlt)* | …und eher gelblich [___] würde sagen würde… |
| 32 | Löschung | `würde` | *(fehlt)* | …eher gelblich grün [___] sagen würde ich… |
| 33 | Löschung | `sagen` | *(fehlt)* | …gelblich grün würde [___] würde ich sagen… |
| 34 | Löschung | `würde` | *(fehlt)* | …grün würde sagen [___] ich sagen das… |
| 35 | Löschung | `ich` | *(fehlt)* | …würde sagen würde [___] sagen das ist… |
| 36 | Löschung | `sagen` | *(fehlt)* | …sagen würde ich [___] das ist echt… |
| 37 | Löschung | `das` | *(fehlt)* | …würde ich sagen [___] ist echt eklig… |
| 38 | Löschung | `ist` | *(fehlt)* | …ich sagen das [___] echt eklig das… |
| 39 | Löschung | `echt` | *(fehlt)* | …sagen das ist [___] eklig das ist… |
| 40 | Löschung | `eklig` | *(fehlt)* | …das ist echt [___] das ist wichtig… |
| 41 | Löschung | `das` | *(fehlt)* | …ist echt eklig [___] ist wichtig zu… |
| 42 | Löschung | `ist` | *(fehlt)* | …echt eklig das [___] wichtig zu wissen… |
| 43 | Substitution | `purulentem` | `kudelndem` | …produktiver husten mit [___] sputum haben sie… |
| 44 | Substitution | `schlechter` | `schlechte` | …gefühl dass sie [___] luft bekommen also… |
| 45 | Substitution | `weh` | `den` | …luft bekommen also [___] tut es in… |
| 46 | Substitution | `tut` | `muss` | …bekommen also weh [___] es in der… |
| 47 | Substitution | `es` | `man` | …also weh tut [___] in der brust… |
| 48 | Substitution | `in` | `das` | …weh tut es [___] der brust nicht… |
| 49 | Löschung | `der` | *(fehlt)* | …tut es in [___] brust nicht direkt… |
| 50 | Löschung | `brust` | *(fehlt)* | …es in der [___] nicht direkt aber… |
| 51 | Substitution | `direkt` | `trennen` | …der brust nicht [___] aber wenn ich… |
| 52 | Substitution | `die` | `treffe` | …aber wenn ich [___] treppen in den… |
| 53 | Substitution | `treppen` | `jetzt` | …wenn ich die [___] in den zweiten… |
| 54 | Substitution | `in` | `habe` | …ich die treppen [___] den zweiten stock… |
| 55 | Substitution | `den` | `ich` | …die treppen in [___] zweiten stock hochlaufe… |
| 56 | Substitution | `zweiten` | `eine` | …treppen in den [___] stock hochlaufe schnaufe… |
| 57 | Substitution | `stock` | `schleimung` | …in den zweiten [___] hochlaufe schnaufe ich… |
| 58 | Substitution | `hochlaufe` | `und` | …den zweiten stock [___] schnaufe ich schon… |
| 59 | Substitution | `schnaufe` | `brauche` | …zweiten stock hochlaufe [___] ich schon ganz… |
| 60 | Substitution | `ich` | `und` | …stock hochlaufe schnaufe [___] schon ganz schön… |
| 61 | Einfügung | *(nicht da)* | `schneide` | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 62 | Einfügung | *(nicht da)* | `mich` | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 63 | Substitution | `doll` | `das` | …schon ganz schön [___] und muss auf… |
| 64 | Substitution | `und` | `ist` | …ganz schön doll [___] muss auf der… |
| 65 | Substitution | `muss` | `dann` | …schön doll und [___] auf der hälfte… |
| 66 | Substitution | `auf` | `eben` | …doll und muss [___] der hälfte kurz… |
| 67 | Substitution | `der` | `auch` | …und muss auf [___] hälfte kurz anhalten… |
| 68 | Löschung | `hälfte` | *(fehlt)* | …muss auf der [___] kurz anhalten gut… |
| 69 | Substitution | `anhalten` | `halb` | …der hälfte kurz [___] gut dass sie… |
| 70 | Substitution | `belastungsdyspnoe` | `belastungsdyspnö` | …erwähnen notiz deutliche [___] keine ruhedyspnoe keine… |
| 71 | Substitution | `ruhedyspnoe` | `ruhedyspnö` | …deutliche belastungsdyspnoe keine [___] keine thorakalen schmerzen… |
| 72 | Substitution | `lutsche` | `bin` | …medikamente dagegen ich [___] nur diese normalen… |
| 73 | Substitution | `nur` | `so` | …dagegen ich lutsche [___] diese normalen hustenbonbons… |
| 74 | Substitution | `diese` | `gut` | …ich lutsche nur [___] normalen hustenbonbons aus… |
| 75 | Substitution | `normalen` | `dass` | …lutsche nur diese [___] hustenbonbons aus der… |
| 76 | Substitution | `hustenbonbons` | `sie` | …nur diese normalen [___] aus der drogerie… |
| 77 | Substitution | `aus` | `dann` | …diese normalen hustenbonbons [___] der drogerie und… |
| 78 | Substitution | `der` | `mal` | …normalen hustenbonbons aus [___] drogerie und trinke… |
| 79 | Substitution | `drogerie` | `ein` | …hustenbonbons aus der [___] und trinke viel… |
| 80 | Substitution | `und` | `bisschen` | …aus der drogerie [___] trinke viel kamillentee… |
| 81 | Substitution | `trinke` | `mehr` | …der drogerie und [___] viel kamillentee alles… |
| 82 | Substitution | `viel` | `brauchen` | …drogerie und trinke [___] kamillentee alles klar… |
| 83 | Substitution | `kamillentee` | `das` | …und trinke viel [___] alles klar notiz… |
| 84 | Einfügung | *(nicht da)* | `ist` | …trinke viel kamillentee [___] klar notiz keine… |
| 85 | Einfügung | *(nicht da)* | `ja` | …trinke viel kamillentee [___] klar notiz keine… |
| 86 | Einfügung | *(nicht da)* | `ein` | …trinke viel kamillentee [___] klar notiz keine… |
| 87 | Einfügung | *(nicht da)* | `trinkgefühl` | …trinke viel kamillentee [___] klar notiz keine… |
| 88 | Substitution | `vormedikation` | `formmedikation` | …notiz keine spezifische [___] lediglich supportive hausmittel… |
| 89 | Substitution | `supportive` | `supertive` | …spezifische vormedikation lediglich [___] hausmittel ich werde… |
| 90 | Substitution | `werde` | `will` | …supportive hausmittel ich [___] jetzt ihre lunge… |
| 91 | Einfügung | *(nicht da)* | `so` | …atmen sie dafür [___] durch den mund… |
| 92 | Substitution | `notiz` | `notizbeginn` | …ein und aus [___] beginn der auskultation… |
| 93 | Löschung | `beginn` | *(fehlt)* | …und aus notiz [___] der auskultation der… |
| 94 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## OriginalLapInMitte.wav

**WER: 6.5%** — Referenz: 231 Wörter | Hypothese: 226 Wörter | S=8 D=6 I=1 | Edit-Distanz=15

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `führt` | `fühlt` | …frau weber was [___] sie heute zu… |
| 2 | Substitution | `sie` | `ihr` | …weber was führt [___] heute zu mir… |
| 3 | Substitution | `da` | `das` | …sie husten kommt [___] schleim mit hoch… |
| 4 | Substitution | `das` | `es` | …hoch oder ist [___] eher ein trockener… |
| 5 | Substitution | `reizhusten` | `heizhusten` | …eher ein trockener [___] da kommt richtig… |
| 6 | Löschung | `das` | *(fehlt)* | …das ist so [___] ist so dickflüssig… |
| 7 | Löschung | `ist` | *(fehlt)* | …ist so das [___] so dickflüssig dickflüssig… |
| 8 | Löschung | `so` | *(fehlt)* | …so das ist [___] dickflüssig dickflüssig und… |
| 9 | Löschung | `dickflüssig` | *(fehlt)* | …ist so dickflüssig [___] und eher gelblich… |
| 10 | Substitution | `weh` | `wehtut` | …luft bekommen also [___] tut es direkt… |
| 11 | Löschung | `tut` | *(fehlt)* | …bekommen also weh [___] es direkt nicht… |
| 12 | Löschung | `schon` | *(fehlt)* | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 13 | Einfügung | *(nicht da)* | `auch` | …doll und muss [___] der hälfte kurz… |
| 14 | Substitution | `ihre` | `ihrer` | …ich werde jetzt [___] lunge abhören bitte… |
| 15 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |


---

## OriginalLapBeiArzt.wav

**WER: 8.4%** — Referenz: 226 Wörter | Hypothese: 229 Wörter | S=12 D=2 I=5 | Edit-Distanz=19

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `huste` | `fühlste` | …herr doktor ich [___] seit knapp einer… |
| 2 | Substitution | `waren` | `war` | …ja gestern abend [___] es 38 8… |
| 3 | Substitution | `es` | `und` | …gestern abend waren [___] 38 8 grad… |
| 4 | Einfügung | *(nicht da)* | `ist` | …abend waren es [___] 8 grad auf… |
| 5 | Substitution | `gelblich` | `geltlich` | …dickflüssig und eher [___] grün würde ich… |
| 6 | Substitution | `zweiten` | `zeitstock` | …treppen in den [___] stock hochlaufe schnaufe… |
| 7 | Substitution | `stock` | `hoch` | …in den zweiten [___] hochlaufe schnaufe ich… |
| 8 | Substitution | `hochlaufe` | `laufe` | …den zweiten stock [___] schnaufe ich schon… |
| 9 | Löschung | `schon` | *(fehlt)* | …hochlaufe schnaufe ich [___] ganz schön doll… |
| 10 | Substitution | `auf` | `auch` | …doll und muss [___] der hälfte kurz… |
| 11 | Einfügung | *(nicht da)* | `in` | …und muss auf [___] hälfte kurz anhalten… |
| 12 | Substitution | `lutsche` | `lösche` | …medikamente dagegen ich [___] nur diese normalen… |
| 13 | Substitution | `drogerie` | `brügerie` | …hustenbonbons aus der [___] und trinke viel… |
| 14 | Substitution | `viel` | `vielfamilien` | …drogerie und trinke [___] kamillentee alles klar… |
| 15 | Löschung | `kamillentee` | *(fehlt)* | …und trinke viel [___] alles klar notiz… |
| 16 | Substitution | `pulmones` | `pulmonis` | …der auskultation der [___] … |
| 17 | Einfügung | *(nicht da)* | `gut` | …auskultation der pulmones [___] … |
| 18 | Einfügung | *(nicht da)* | `bitte` | …auskultation der pulmones [___] … |
| 19 | Einfügung | *(nicht da)* | `auch` | …auskultation der pulmones [___] … |


---

## SelbstkorrekturLapInMitte.wav

**WER: 21.3%** — Referenz: 183 Wörter | Hypothese: 211 Wörter | S=11 D=0 I=28 | Edit-Distanz=39

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `ne` | `nee` | …habe seit dienstag [___] warten sie mal… |
| 2 | Substitution | `äh` | `ah` | …linke schläfe aus [___] quatsch ich zeig… |
| 3 | Substitution | `zeig` | `zeige` | …äh quatsch ich [___] s gerade falsch… |
| 4 | Substitution | `s` | `es` | …quatsch ich zeig [___] gerade falsch rum… |
| 5 | Substitution | `hab` | `habe` | …nicht geholfen dann [___] ich am nachmittag… |
| 6 | Einfügung | *(nicht da)* | `habe` | …vor dem schlaf [___] eine ibuprofen 400… |
| 7 | Einfügung | *(nicht da)* | `ich` | …vor dem schlaf [___] eine ibuprofen 400… |
| 8 | Substitution | `ibuprofen` | `ibuprofene` | …schlaf noch eine [___] 400 genommen oder… |
| 9 | Substitution | `waren` | `war` | …400 genommen oder [___] das 600 die… |
| 10 | Substitution | `das` | `es` | …genommen oder waren [___] 600 die rosafarbenen… |
| 11 | Einfügung | *(nicht da)* | `doch` | …oder waren das [___] die rosafarbenen aus… |
| 12 | Einfügung | *(nicht da)* | `die` | …oder waren das [___] die rosafarbenen aus… |
| 13 | Substitution | `rosafarbenen` | `rosafarbene` | …das 600 die [___] aus der großen… |
| 14 | Substitution | `rechtsseitig` | `rechtzeitig` | …notiz pochender kopfschmerz [___] seit mittwoch keine… |
| 15 | Substitution | `ibuprofen` | `ibuprofene` | …besserung fraglich nach [___] 600 wurde ihnen… |
| 16 | Einfügung | *(nicht da)* | `dass` | …bin extrem lichtempfindlich [___] … |
| 17 | Einfügung | *(nicht da)* | `meine` | …bin extrem lichtempfindlich [___] … |
| 18 | Einfügung | *(nicht da)* | `inャ` | …bin extrem lichtempfindlich [___] … |
| 19 | Einfügung | *(nicht da)* | `nil` | …bin extrem lichtempfindlich [___] … |
| 20 | Einfügung | *(nicht da)* | `r` | …bin extrem lichtempfindlich [___] … |
| 21 | Einfügung | *(nicht da)* | `sigt` | …bin extrem lichtempfindlich [___] … |
| 22 | Einfügung | *(nicht da)* | `speaker_` | …bin extrem lichtempfindlich [___] … |
| 23 | Einfügung | *(nicht da)* | `und` | …bin extrem lichtempfindlich [___] … |
| 24 | Einfügung | *(nicht da)* | `nochинاء` | …bin extrem lichtempfindlich [___] … |
| 25 | Einfügung | *(nicht da)* | `sarah` | …bin extrem lichtempfindlich [___] … |
| 26 | Einfügung | *(nicht da)* | `hätte` | …bin extrem lichtempfindlich [___] … |
| 27 | Einfügung | *(nicht da)* | `hierzu` | …bin extrem lichtempfindlich [___] … |
| 28 | Einfügung | *(nicht da)* | `für` | …bin extrem lichtempfindlich [___] … |
| 29 | Einfügung | *(nicht da)* | `mich` | …bin extrem lichtempfindlich [___] … |
| 30 | Einfügung | *(nicht da)* | `auch` | …bin extrem lichtempfindlich [___] … |
| 31 | Einfügung | *(nicht da)* | `waar` | …bin extrem lichtempfindlich [___] … |
| 32 | Einfügung | *(nicht da)* | `es` | …bin extrem lichtempfindlich [___] … |
| 33 | Einfügung | *(nicht da)* | `die` | …bin extrem lichtempfindlich [___] … |
| 34 | Einfügung | *(nicht da)* | `fetam` | …bin extrem lichtempfindlich [___] … |
| 35 | Einfügung | *(nicht da)* | `vor` | …bin extrem lichtempfindlich [___] … |
| 36 | Einfügung | *(nicht da)* | `collar` | …bin extrem lichtempfindlich [___] … |
| 37 | Einfügung | *(nicht da)* | `an` | …bin extrem lichtempfindlich [___] … |
| 38 | Einfügung | *(nicht da)* | `knight` | …bin extrem lichtempfindlich [___] … |
| 39 | Einfügung | *(nicht da)* | `besserhemen` | …bin extrem lichtempfindlich [___] … |


---

## UnterbrechungLapInMitte.wav

**WER: 13.1%** — Referenz: 153 Wörter | Hypothese: 143 Wörter | S=8 D=11 I=1 | Edit-Distanz=20

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `drei` | `3` | …ich bin um [___] uhr aufgewacht und… |
| 2 | Löschung | `wo` | *(fehlt)* | …dass ich direkt [___] genau krampft es… |
| 3 | Löschung | `genau` | *(fehlt)* | …ich direkt wo [___] krampft es denn… |
| 4 | Löschung | `krampft` | *(fehlt)* | …direkt wo genau [___] es denn auf… |
| 5 | Löschung | `es` | *(fehlt)* | …wo genau krampft [___] denn auf die… |
| 6 | Löschung | `denn` | *(fehlt)* | …genau krampft es [___] auf die toilette… |
| 7 | Substitution | `dem` | `den` | …ist direkt über [___] bauchnabel strahlt der… |
| 8 | Löschung | `da` | *(fehlt)* | …ja heute morgen [___] war es bei… |
| 9 | Löschung | `grad` | *(fehlt)* | …es bei 38 [___] 38 was komma… |
| 10 | Substitution | `zwei` | `2` | …38 was komma [___] komma fünf komma… |
| 11 | Substitution | `fünf` | `5` | …komma zwei komma [___] komma drei und… |
| 12 | Substitution | `drei` | `3` | …komma fünf komma [___] und ich habe… |
| 13 | Einfügung | *(nicht da)* | `und` | …epigastrischer schmerz erbrechen [___] bei 38 3… |
| 14 | Löschung | `mit` | *(fehlt)* | …haben wir gegrillt [___] den nachbarn vielleicht… |
| 15 | Löschung | `den` | *(fehlt)* | …wir gegrillt mit [___] nachbarn vielleicht etwas… |
| 16 | Löschung | `nachbarn` | *(fehlt)* | …gegrillt mit den [___] vielleicht etwas vom… |
| 17 | Substitution | `das` | `was` | …etwas vom grill [___] nicht ganz durch… |
| 18 | Löschung | `nicht` | *(fehlt)* | …an einer stelle [___] vielleicht nicht noch… |
| 19 | Substitution | `wo` | `muss` | …etwas rosa jetzt [___] sie es sagen… |
| 20 | Substitution | `sie` | `ich` | …rosa jetzt wo [___] es sagen… |


---

## GedankenprüngeLapInMitte.wav

**WER: 2.6%** — Referenz: 192 Wörter | Hypothese: 190 Wörter | S=3 D=2 I=0 | Edit-Distanz=5

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `yilmaz` | `hielmanns` | …herr [___] sie klagen über… |
| 2 | Substitution | `ne` | `eine` | …entzündet das war [___] totale katastrophe der… |
| 3 | Löschung | `laut` | *(fehlt)* | …hat es laut [___] geknackt und seitdem… |
| 4 | Substitution | `rotatorenmanschetten` | `rotatorenmanschettenruptur` | …notiz verdacht auf [___] ruptur nach sporttrauma… |
| 5 | Löschung | `ruptur` | *(fehlt)* | …verdacht auf rotatorenmanschetten [___] nach sporttrauma… |


---

## MeinungswechselLapinMitte.wav

**WER: 7.8%** — Referenz: 179 Wörter | Hypothese: 186 Wörter | S=3 D=2 I=9 | Edit-Distanz=14

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `kardiales` | `kardinales` | …angina pectoris eventuell [___] ereignis strahlt das… |
| 2 | Löschung | `in` | *(fehlt)* | …linken arm oder [___] den unterkiefer aus… |
| 3 | Löschung | `atemnot` | *(fehlt)* | …haben sie atemnot [___] oder kalten schweiß… |
| 4 | Substitution | `ekligen` | `ekeligen` | …zeit so einen [___] sauren geschmack im… |
| 5 | Substitution | `tomatensoße` | `tomatensauce` | …abend nach der [___] war es extrem… |
| 6 | Einfügung | *(nicht da)* | `das` | …verursacht diesen schmerz [___] … |
| 7 | Einfügung | *(nicht da)* | `war` | …verursacht diesen schmerz [___] … |
| 8 | Einfügung | *(nicht da)* | `s` | …verursacht diesen schmerz [___] … |
| 9 | Einfügung | *(nicht da)* | `speaker_` | …verursacht diesen schmerz [___] … |
| 10 | Einfügung | *(nicht da)* | `das` | …verursacht diesen schmerz [___] … |
| 11 | Einfügung | *(nicht da)* | `war` | …verursacht diesen schmerz [___] … |
| 12 | Einfügung | *(nicht da)* | `s` | …verursacht diesen schmerz [___] … |
| 13 | Einfügung | *(nicht da)* | `grund` | …verursacht diesen schmerz [___] … |
| 14 | Einfügung | *(nicht da)* | `了吧` | …verursacht diesen schmerz [___] … |


---

## ChaosLapInMitte.wav

**WER: 15.8%** — Referenz: 272 Wörter | Hypothese: 252 Wörter | S=15 D=24 I=4 | Edit-Distanz=43

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Löschung | `seit` | *(fehlt)* | …so schwindelig seit [___] samstagabend nee moment… |
| 2 | Substitution | `nee` | `ehm` | …seit seit samstagabend [___] moment samstag war… |
| 3 | Substitution | `schwankschwindel` | `schwankschwindeln` | …so komisch notiz [___] und tinnitus rechts… |
| 4 | Löschung | `genau` | *(fehlt)* | …von der kopfbewegung [___] so hat das… |
| 5 | Löschung | `so` | *(fehlt)* | …der kopfbewegung genau [___] hat das nämlich… |
| 6 | Löschung | `hat` | *(fehlt)* | …kopfbewegung genau so [___] das nämlich auch… |
| 7 | Löschung | `das` | *(fehlt)* | …genau so hat [___] nämlich auch bei… |
| 8 | Löschung | `nämlich` | *(fehlt)* | …so hat das [___] auch bei meiner… |
| 9 | Löschung | `auch` | *(fehlt)* | …hat das nämlich [___] bei meiner cousine… |
| 10 | Löschung | `bei` | *(fehlt)* | …das nämlich auch [___] meiner cousine angefangen… |
| 11 | Löschung | `meiner` | *(fehlt)* | …nämlich auch bei [___] cousine angefangen die… |
| 12 | Löschung | `cousine` | *(fehlt)* | …auch bei meiner [___] angefangen die hat… |
| 13 | Löschung | `angefangen` | *(fehlt)* | …bei meiner cousine [___] die hat dann… |
| 14 | Substitution | `akustikusneurinom` | `akustikus` | …wie hieß das [___] glaub ich die… |
| 15 | Substitution | `glaub` | `neuringum` | …hieß das akustikusneurinom [___] ich die musste… |
| 16 | Einfügung | *(nicht da)* | `glaube` | …das akustikusneurinom glaub [___] die musste direkt… |
| 17 | Substitution | `hab` | `habe` | …und operiert werden [___] ich jetzt auch… |
| 18 | Substitution | `hab` | `habe` | …äußerst unwahrscheinlich ich [___] da echt panik… |
| 19 | Löschung | `da` | *(fehlt)* | …unwahrscheinlich ich hab [___] echt panik wissen… |
| 20 | Löschung | `echt` | *(fehlt)* | …ich hab da [___] panik wissen sie… |
| 21 | Löschung | `panik` | *(fehlt)* | …hab da echt [___] wissen sie ich… |
| 22 | Löschung | `wissen` | *(fehlt)* | …da echt panik [___] sie ich hab… |
| 23 | Löschung | `sie` | *(fehlt)* | …echt panik wissen [___] ich hab ja… |
| 24 | Löschung | `ich` | *(fehlt)* | …panik wissen sie [___] hab ja zwei… |
| 25 | Löschung | `hab` | *(fehlt)* | …wissen sie ich [___] ja zwei kleine… |
| 26 | Substitution | `mal` | `erstmal` | …lassen sie mich [___] in ihr rechtes… |
| 27 | Substitution | `pfropf` | `fropf` | …da einen massiven [___] aus ohrenschmalz der… |
| 28 | Substitution | `drückt` | `riecht` | …aus ohrenschmalz der [___] richtig fest auf… |
| 29 | Substitution | `richtig` | `sich` | …ohrenschmalz der drückt [___] fest auf das… |
| 30 | Löschung | `oft` | *(fehlt)* | …pfeifen und oft [___] oft auch den… |
| 31 | Löschung | `oft` | *(fehlt)* | …und oft oft [___] auch den schwindel… |
| 32 | Löschung | `und` | *(fehlt)* | …jetzt kurz aus [___] dann sollte der… |
| 33 | Löschung | `habe` | *(fehlt)* | …sei dank ich [___] ich hatte schon… |
| 34 | Löschung | `ich` | *(fehlt)* | …dank ich habe [___] hatte schon wieder… |
| 35 | Substitution | `cerumen` | `zeromen` | …operative intervention nötig [___] obturans rechtsseitig vormedikation… |
| 36 | Substitution | `obturans` | `obutrans` | …intervention nötig cerumen [___] rechtsseitig vormedikation 800mg… |
| 37 | Substitution | `rechtsseitig` | `rechtzeitig` | …nötig cerumen obturans [___] vormedikation 800mg ibuprofen… |
| 38 | Substitution | `vormedikation` | `vor` | …cerumen obturans rechtsseitig [___] 800mg ibuprofen genau… |
| 39 | Substitution | `800mg` | `medikation` | …obturans rechtsseitig vormedikation [___] ibuprofen genau das… |
| 40 | Einfügung | *(nicht da)* | `800` | …rechtsseitig vormedikation 800mg [___] genau das können… |
| 41 | Einfügung | *(nicht da)* | `milligramm` | …rechtsseitig vormedikation 800mg [___] genau das können… |
| 42 | Einfügung | *(nicht da)* | `dann` | …das können sie [___] morgen dann wieder… |
| 43 | Löschung | `dann` | *(fehlt)* | …sie ab morgen [___] wieder weglassen… |


---

## Das Anamnesegespräch.wav

**WER: 6.5%** — Referenz: 2317 Wörter | Hypothese: 2270 Wörter | S=77 D=60 I=13 | Edit-Distanz=150

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `colette` | `colett` | …name ist nina [___] und ich bin… |
| 2 | Substitution | `westphalen` | `westfalen` | …heiße julia becken [___] julia becken westphalen… |
| 3 | Substitution | `westphalen` | `westfalen` | …westphalen julia becken [___] können sie mir… |
| 4 | Substitution | `b` | `westfalen` | …gerne ja becken [___] e c k… |
| 5 | Substitution | `e` | `west` | …ja becken b [___] c k e… |
| 6 | Substitution | `c` | `palen` | …becken b e [___] k e n… |
| 7 | Löschung | `k` | *(fehlt)* | …b e c [___] e n bindestrich… |
| 8 | Löschung | `e` | *(fehlt)* | …e c k [___] n bindestrich westphalen… |
| 9 | Löschung | `n` | *(fehlt)* | …c k e [___] bindestrich westphalen w… |
| 10 | Löschung | `bindestrich` | *(fehlt)* | …k e n [___] westphalen w e… |
| 11 | Löschung | `westphalen` | *(fehlt)* | …e n bindestrich [___] w e s… |
| 12 | Löschung | `w` | *(fehlt)* | …n bindestrich westphalen [___] e s t… |
| 13 | Löschung | `e` | *(fehlt)* | …bindestrich westphalen w [___] s t p… |
| 14 | Löschung | `s` | *(fehlt)* | …westphalen w e [___] t p h… |
| 15 | Löschung | `t` | *(fehlt)* | …w e s [___] p h a… |
| 16 | Löschung | `p` | *(fehlt)* | …e s t [___] h a l… |
| 17 | Löschung | `h` | *(fehlt)* | …s t p [___] a l e… |
| 18 | Löschung | `a` | *(fehlt)* | …t p h [___] l e n… |
| 19 | Löschung | `l` | *(fehlt)* | …p h a [___] e n alles… |
| 20 | Löschung | `e` | *(fehlt)* | …h a l [___] n alles klar… |
| 21 | Löschung | `n` | *(fehlt)* | …a l e [___] alles klar danke… |
| 22 | Substitution | `danke` | `dankeschön` | …n alles klar [___] schön frau becken… |
| 23 | Löschung | `schön` | *(fehlt)* | …alles klar danke [___] frau becken westphalen… |
| 24 | Substitution | `westphalen` | `westfalen` | …schön frau becken [___] wie alt sind… |
| 25 | Löschung | `33` | *(fehlt)* | …sie denn 33 [___] und wann ist… |
| 26 | Löschung | `ah` | *(fehlt)* | …27 märz 1987 [___] schön herzlichen glückwunsch… |
| 27 | Substitution | `westphalen` | `westfalen` | …dank frau becken [___] wie groß sind… |
| 28 | Einfügung | *(nicht da)* | `speaker_` | …kilo glaube ich [___] gut können sie… |
| 29 | Substitution | `doktor` | `dr` | …ist der herr [___] becker der herr… |
| 30 | Substitution | `doktor` | `dr` | …becker der herr [___] becker wie der… |
| 31 | Substitution | `westphalen` | `westfalen` | …gut frau becken [___] sie wurden ja… |
| 32 | Substitution | `beschmerzen` | `beschwerden` | …denn jetzt für [___] beschmerzen beschwerden schuldigung… |
| 33 | Löschung | `beschmerzen` | *(fehlt)* | …jetzt für beschmerzen [___] beschwerden schuldigung haben… |
| 34 | Substitution | `schuldigung` | `entschuldigung` | …beschmerzen beschmerzen beschwerden [___] haben sie schmerzen… |
| 35 | Substitution | `in` | `an` | …kopf im oberkörper [___] den beinen ja… |
| 36 | Einfügung | *(nicht da)* | `wirklich` | …bewegen weil ich [___] schmerzen habe am… |
| 37 | Löschung | `ja` | *(fehlt)* | …habe am daumen [___] und er ist… |
| 38 | Substitution | `beides` | `weil` | …richtig stark verletzt [___] richtig geschwollen ist… |
| 39 | Einfügung | *(nicht da)* | `es` | …stark verletzt beides [___] geschwollen ist und… |
| 40 | Substitution | `wehtut` | `weh` | …und auch sehr [___] okay knie ist… |
| 41 | Einfügung | *(nicht da)* | `tut` | …auch sehr wehtut [___] knie ist auch… |
| 42 | Substitution | `westphalen` | `westfalen` | …genau frau becken [___] haben sie denn… |
| 43 | Substitution | `ungerne` | `ungern` | …dass ich sehr [___] einen fahrradhelm trage… |
| 44 | Substitution | `westphalen` | `westfalen` | …aussehen frau becken [___] bitte bitte tragen… |
| 45 | Löschung | `am` | *(fehlt)* | …sie haben hinten [___] auf der linken… |
| 46 | Substitution | `eins` | `1` | …einer schmerzskala wobei [___] sehr leichten schmerzen… |
| 47 | Substitution | `zehn` | `10` | …schmerzen entspricht und [___] sehr starken schmerzen… |
| 48 | Löschung | `den` | *(fehlt)* | …wo würden sie [___] daumen die schmerzen… |
| 49 | Löschung | `daumen` | *(fehlt)* | …würden sie den [___] die schmerzen des… |
| 50 | Substitution | `sieben` | `7` | …so auf die [___] zu vor allem… |
| 51 | Substitution | `acht` | `8` | …ich sagen bei [___] wenn ich sitze… |
| 52 | Löschung | `das` | *(fehlt)* | …aus nein das [___] zum glück nicht… |
| 53 | Substitution | `an` | `in` | …handgelenk aus oder [___] andere finger auch… |
| 54 | Substitution | `becken` | `beckenwestfalen` | …unfall erinnern frau [___] westphalen ich kann… |
| 55 | Löschung | `westphalen` | *(fehlt)* | …erinnern frau becken [___] ich kann mich… |
| 56 | Substitution | `dran` | `daran` | …kann mich gut [___] erinnern ja ich… |
| 57 | Einfügung | *(nicht da)* | `es` | …und mir war [___] schwindelig aber ich… |
| 58 | Substitution | `noch` | `nochmal` | …oder vielleicht doch [___] mal schwarz vor… |
| 59 | Löschung | `mal` | *(fehlt)* | …vielleicht doch noch [___] schwarz vor augen… |
| 60 | Löschung | `wie` | *(fehlt)* | …und dass ich [___] gesagt am anfang… |
| 61 | Löschung | `gesagt` | *(fehlt)* | …dass ich wie [___] am anfang nur… |
| 62 | Substitution | `becken` | `beckenwestfalen` | …sehr gut frau [___] westphalen haben sie… |
| 63 | Löschung | `westphalen` | *(fehlt)* | …gut frau becken [___] haben sie irgendwelche… |
| 64 | Substitution | `laktose` | `laktoseintoleranz` | …ich hatte eine [___] intoleranz vor einigen… |
| 65 | Löschung | `intoleranz` | *(fehlt)* | …hatte eine laktose [___] vor einigen jahren… |
| 66 | Löschung | `ja` | *(fehlt)* | …allerdings schon weg [___] und jetzt wurde… |
| 67 | Substitution | `dekollete` | `dekolleté` | …ausschlag hier im [___] bereich okay sonst… |
| 68 | Einfügung | *(nicht da)* | `nein` | …keine vorerkrankungen nein [___] sehr gut frau… |
| 69 | Substitution | `becken` | `beckenwestfalen` | …sehr gut frau [___] westphalen sind sie… |
| 70 | Löschung | `westphalen` | *(fehlt)* | …gut frau becken [___] sind sie schon… |
| 71 | Substitution | `hallux` | `halux` | …mir wurde ein [___] valgus entfernt ein… |
| 72 | Substitution | `hallux` | `halux` | …valgus entfernt ein [___] valgus und welcher… |
| 73 | Substitution | `becken` | `beckenwestfalen` | …gut gelaufen frau [___] westphalen nehmen sie… |
| 74 | Löschung | `westphalen` | *(fehlt)* | …gelaufen frau becken [___] nehmen sie regelmäßig… |
| 75 | Substitution | `viel` | `viele` | …kopfschmerzen haben wie [___] milligramm sind das… |
| 76 | Substitution | `600er` | `600` | …ich glaube 600 [___] alles klar sind… |
| 77 | Einfügung | *(nicht da)* | `ja` | …glaube 600 600er [___] klar sind sie… |
| 78 | Löschung | `ja` | *(fehlt)* | …nein eher nicht [___] ich muss ins… |
| 79 | Löschung | `ich` | *(fehlt)* | …eher nicht ja [___] muss ins krankenhaus… |
| 80 | Löschung | `muss` | *(fehlt)* | …nicht ja ich [___] ins krankenhaus hätte… |
| 81 | Löschung | `ins` | *(fehlt)* | …ja ich muss [___] krankenhaus hätte ich… |
| 82 | Löschung | `krankenhaus` | *(fehlt)* | …ich muss ins [___] hätte ich gewusst… |
| 83 | Löschung | `hätte` | *(fehlt)* | …ins krankenhaus muss [___] ich ihn mitgenommen… |
| 84 | Löschung | `ich` | *(fehlt)* | …krankenhaus muss hätte [___] ihn mitgenommen ja… |
| 85 | Löschung | `ihn` | *(fehlt)* | …muss hätte ich [___] mitgenommen ja ich… |
| 86 | Löschung | `mitgenommen` | *(fehlt)* | …hätte ich ihn [___] ja ich muss… |
| 87 | Substitution | `gestehen` | `wissen` | …ich muss auch [___] dass ich ihn… |
| 88 | Substitution | `ihn` | `den` | …gestehen dass ich [___] nicht bei mir… |
| 89 | Löschung | `ja` | *(fehlt)* | …bei mir trage [___] sehr gut okay… |
| 90 | Substitution | `becken` | `beckenwestfalen` | …gut okay frau [___] westphalen wie geht… |
| 91 | Löschung | `westphalen` | *(fehlt)* | …okay frau becken [___] wie geht es… |
| 92 | Substitution | `hab` | `habe` | …zeit nein ich [___] gar keine sonstigen… |
| 93 | Substitution | `becken` | `beckenwestfalen` | …okay wunderbar frau [___] westphalen rauchen sie… |
| 94 | Löschung | `westphalen` | *(fehlt)* | …wunderbar frau becken [___] rauchen sie nein… |
| 95 | Substitution | `aber` | `wie` | …relevant ist ja [___] ja wann haben… |
| 96 | Substitution | `ja` | `lange` | …ist ja aber [___] wann haben sie… |
| 97 | Löschung | `wann` | *(fehlt)* | …ja aber ja [___] haben sie damit… |
| 98 | Löschung | `damit` | *(fehlt)* | …wann haben sie [___] aufgehört das müssten… |
| 99 | Einfügung | *(nicht da)* | `ach` | …sie damit aufgehört [___] müssten jetzt schon… |
| 100 | Substitution | `gläser` | `gäser` | …zwei oder drei [___] okay dieses gläschen… |
| 101 | Substitution | `becken` | `beckenwestfalen` | …okay wunderbar frau [___] westphalen nehmen sie… |
| 102 | Löschung | `westphalen` | *(fehlt)* | …wunderbar frau becken [___] nehmen sie das… |
| 103 | Substitution | `drei` | `dreimal` | …ja habe zwei [___] mal ritalin genommen… |
| 104 | Löschung | `mal` | *(fehlt)* | …habe zwei drei [___] ritalin genommen okay… |
| 105 | Substitution | `wars` | `war` | …genommen okay das [___] aber aber jetzt… |
| 106 | Substitution | `aber` | `es` | …okay das wars [___] aber jetzt schon… |
| 107 | Substitution | `jetzt` | `ja` | …wars aber aber [___] schon ja sehr… |
| 108 | Löschung | `schon` | *(fehlt)* | …aber aber jetzt [___] ja sehr gut… |
| 109 | Löschung | `okay` | *(fehlt)* | …großvater hatte leberzirrhose [___] und ist leider… |
| 110 | Substitution | `gestorben` | `geschrauben` | …leider auch daran [___] oh das tut… |
| 111 | Substitution | `brustkrebs` | `großkrebs` | …meine großmutter hatte [___] aber sie lebt… |
| 112 | Substitution | `becken` | `böcken` | …sie geschwister frau [___] westphalen ich habe… |
| 113 | Substitution | `westphalen` | `westfalen` | …geschwister frau becken [___] ich habe eine… |
| 114 | Substitution | `es` | `sie` | …ja sonst geht [___] ihr gut sehr… |
| 115 | Löschung | `ihr` | *(fehlt)* | …sonst geht es [___] gut sehr gut… |
| 116 | Substitution | `becken` | `böcken` | …sie kinder frau [___] westphalen nein ich… |
| 117 | Substitution | `westphalen` | `westfalen` | …kinder frau becken [___] nein ich habe… |
| 118 | Substitution | `bin` | `wenn` | …in einer marketingagentur [___] da für größere… |
| 119 | Einfügung | *(nicht da)* | `sind` | …und marketingprojekte zuständig [___] sehr gut eine… |
| 120 | Substitution | `becken` | `böcken` | …frage noch frau [___] westphalen waren sie… |
| 121 | Substitution | `westphalen` | `westfalen` | …noch frau becken [___] waren sie in… |
| 122 | Substitution | `becken` | `böcken` | …okay gut frau [___] westphalen von meiner… |
| 123 | Substitution | `westphalen` | `westfalen` | …gut frau becken [___] von meiner seite… |
| 124 | Substitution | `noch` | `nochmal` | …würde das gleich [___] mal mit ihnen… |
| 125 | Löschung | `mal` | *(fehlt)* | …das gleich noch [___] mit ihnen durchgehen… |
| 126 | Einfügung | *(nicht da)* | `so` | …sie dass es [___] schlimm wird oder… |
| 127 | Substitution | `noch` | `nochmal` | …würde auch gerne [___] mal eine untersuchung… |
| 128 | Löschung | `mal` | *(fehlt)* | …auch gerne noch [___] eine untersuchung mit… |
| 129 | Löschung | `dem` | *(fehlt)* | …wenn wir mit [___] mit der aufnahme… |
| 130 | Löschung | `mit` | *(fehlt)* | …wir mit dem [___] der aufnahme fertig… |
| 131 | Löschung | `ja` | *(fehlt)* | …auch gleich los [___] alles klar nochmal… |
| 132 | Substitution | `nochmal` | `noch` | …ja alles klar [___] kurz zum abgleich… |
| 133 | Einfügung | *(nicht da)* | `mal` | …alles klar nochmal [___] zum abgleich sie… |
| 134 | Substitution | `sieben` | `7` | …dort mit einer [___] beschrieben und haben… |
| 135 | Substitution | `fürs` | `für` | …und gleiches gilt [___] knie auch das… |
| 136 | Einfügung | *(nicht da)* | `das` | …gleiches gilt fürs [___] auch das knie… |
| 137 | Substitution | `acht` | `8` | …schmerzintensität mit einer [___] beschrieben bei bewegung… |
| 138 | Substitution | `zehn` | `10` | …bewegung unerträglich also [___] oder mehr als… |
| 139 | Substitution | `zehn` | `10` | …oder mehr als [___] auch dieser schmerz… |
| 140 | Löschung | `richtig` | *(fehlt)* | …knie geschwollen richtig [___] sie haben gesagt… |
| 141 | Substitution | `leicht` | `recht` | …nur kurz danach [___] schwindelig war das… |
| 142 | Substitution | `war` | `waren` | …danach leicht schwindelig [___] das sei aber… |
| 143 | Substitution | `hallux` | `halux` | …da wurde der [___] valgus operiert ansonsten… |
| 144 | Löschung | `genau` | *(fehlt)* | …gott sei dank [___] bis auf die… |
| 145 | Substitution | `histaminunverträglichkeit` | `kistaminunverträglichkeit` | …bis auf die [___] genau das hätte… |
| 146 | Substitution | `noch` | `nochmal` | …eingebracht vielen dank [___] mal dafür habe… |
| 147 | Löschung | `mal` | *(fehlt)* | …vielen dank noch [___] dafür habe ich… |
| 148 | Substitution | `westphalen` | `westfalen` | …gut frau becken [___] dann war es… |
| 149 | Substitution | `danke` | `vielen` | …warte dann hier [___] ihnen super besten… |
| 150 | Einfügung | *(nicht da)* | `dank` | …dann hier danke [___] super besten dank… |


---

## Anamnesegesrpäch PWC.mp3

**WER: 21.1%** — Referenz: 1530 Wörter | Hypothese: 1516 Wörter | S=133 D=102 I=88 | Edit-Distanz=323

| # | Typ | Ground Truth | STT-Output | Kontext (Referenz) |
|---|-----|-------------|-----------|-------------------|
| 1 | Substitution | `grasbäutner` | `grasbeutner` | …grüß gott frau [___] ich bin die… |
| 2 | Substitution | `eisel` | `eisley` | …ich bin die [___] theresa ihre physiotherapeutin… |
| 3 | Substitution | `theresa` | `teresa` | …bin die eisel [___] ihre physiotherapeutin wir… |
| 4 | Substitution | `ihre` | `eine` | …die eisel theresa [___] physiotherapeutin wir dürfen… |
| 5 | Substitution | `hinsetzen` | `hier` | …dürfen sie gerne [___] wir beginnen heute… |
| 6 | Einfügung | *(nicht da)* | `sitzen` | …sie gerne hinsetzen [___] beginnen heute so… |
| 7 | Löschung | `heute` | *(fehlt)* | …hinsetzen wir beginnen [___] so dass wir… |
| 8 | Substitution | `anamnese` | `lese` | …dass wir eine [___] zuerst machen und… |
| 9 | Einfügung | *(nicht da)* | `noch` | …möchte ich ihnen [___] paar fragen zu… |
| 10 | Substitution | `sagen` | `sind` | …ihr alter sie [___] 27 27 jahre… |
| 11 | Löschung | `27` | *(fehlt)* | …sie sagen 27 [___] jahre und sie… |
| 12 | Löschung | `büroangestellte` | *(fehlt)* | …arbeiten als büroangestellte [___] und da arbeiten… |
| 13 | Löschung | `ja` | *(fehlt)* | …arbeiten sie vollzeit [___] das heißt ganz… |
| 14 | Löschung | `ja` | *(fehlt)* | …machen sie gern [___] ist das heute… |
| 15 | Substitution | `ja` | `ich` | …besuch beim physiotherapeuten [___] sie waren noch… |
| 16 | Substitution | `sie` | `war` | …beim physiotherapeuten ja [___] waren noch nie… |
| 17 | Löschung | `waren` | *(fehlt)* | …physiotherapeuten ja sie [___] noch nie also… |
| 18 | Löschung | `also` | *(fehlt)* | …waren noch nie [___] sie kennen das… |
| 19 | Substitution | `aus` | `als` | …kennen das nur [___] erzählungen aber selber… |
| 20 | Einfügung | *(nicht da)* | `noch` | …erfahrungen haben sie [___] gemacht dann können… |
| 21 | Substitution | `anderthalb` | `einer` | …ich habe vor [___] monaten circa einen… |
| 22 | Substitution | `monaten` | `und` | …habe vor anderthalb [___] circa einen kreuzbandriss… |
| 23 | Substitution | `circa` | `einem` | …vor anderthalb monaten [___] einen kreuzbandriss gehabt… |
| 24 | Einfügung | *(nicht da)* | `monat` | …anderthalb monaten circa [___] kreuzbandriss gehabt also… |
| 25 | Einfügung | *(nicht da)* | `ca` | …anderthalb monaten circa [___] kreuzbandriss gehabt also… |
| 26 | Substitution | `kreuzbandriss` | `kreuzkontress` | …monaten circa einen [___] gehabt also sie… |
| 27 | Substitution | `kreuzbandriss` | `kreuzkontress` | …sie haben einen [___] gehabt wie ist… |
| 28 | Substitution | `denn` | `ihnen` | …gehabt wie ist [___] das passiert beim… |
| 29 | Löschung | `also` | *(fehlt)* | …training bin ich [___] ich bin gesprungen… |
| 30 | Löschung | `ich` | *(fehlt)* | …bin ich also [___] bin gesprungen beim… |
| 31 | Löschung | `bin` | *(fehlt)* | …ich also ich [___] gesprungen beim netzfahren… |
| 32 | Substitution | `beim` | `am` | …ich bin gesprungen [___] netzfahren und war… |
| 33 | Substitution | `netzfahren` | `netz` | …bin gesprungen beim [___] und war zu… |
| 34 | Löschung | `und` | *(fehlt)* | …gesprungen beim netzfahren [___] war zu spät… |
| 35 | Substitution | `zum` | `zu` | …zu spät dran [___] block und bin… |
| 36 | Einfügung | *(nicht da)* | `einem` | …spät dran zum [___] und bin schief… |
| 37 | Substitution | `haben` | `sind` | …dann ja sie [___] dann schief gelandet… |
| 38 | Einfügung | *(nicht da)* | `sie` | …schief gelandet und [___] sie dann aufgestanden… |
| 39 | Substitution | `sie` | `sich` | …gelandet und sind [___] dann aufgestanden oder… |
| 40 | Substitution | `dagelegen` | `da` | …bin dann eigentlich [___] und habe einen… |
| 41 | Einfügung | *(nicht da)* | `gelegen` | …dann eigentlich dagelegen [___] habe einen wirklich… |
| 42 | Einfügung | *(nicht da)* | `eine` | …eine halbe minute [___] circa dass ich… |
| 43 | Substitution | `circa` | `ca` | …halbe minute minute [___] dass ich mein… |
| 44 | Substitution | `mein` | `es` | …circa dass ich [___] bein nicht ausstrecken… |
| 45 | Löschung | `bein` | *(fehlt)* | …dass ich mein [___] nicht ausstrecken können… |
| 46 | Substitution | `können` | `konnte` | …bein nicht ausstrecken [___] habe und gar… |
| 47 | Löschung | `habe` | *(fehlt)* | …nicht ausstrecken können [___] und gar nichts… |
| 48 | Substitution | `gekühlt` | `kühlt` | …es dann gleich [___] und ein bisschen… |
| 49 | Substitution | `hoch` | `hochgelabert` | …und ein bisschen [___] gelagert und dann… |
| 50 | Löschung | `gelagert` | *(fehlt)* | …ein bisschen hoch [___] und dann hat… |
| 51 | Löschung | `dann` | *(fehlt)* | …schnell wieder nachgelassen [___] es war ein… |
| 52 | Einfügung | *(nicht da)* | `noch` | …dann es war [___] bisschen ein komisches… |
| 53 | Löschung | `ja` | *(fehlt)* | …bin ich eigentlich [___] duschen gegangen und… |
| 54 | Löschung | `dann` | *(fehlt)* | …duschen gegangen und [___] habe ich mir… |
| 55 | Löschung | `ich` | *(fehlt)* | …und dann habe [___] mir gedacht ja… |
| 56 | Substitution | `halt` | `heute` | …sicherheit fahre ich [___] noch ins krankenhaus… |
| 57 | Einfügung | *(nicht da)* | `auch` | …trainieren sie da [___] die woche ja… |
| 58 | Substitution | `die` | `in` | …woche ja zweimal [___] woche zweimal die… |
| 59 | Einfügung | *(nicht da)* | `der` | …ja zweimal die [___] zweimal die woche… |
| 60 | Substitution | `4` | `vier` | …ja ich glaube [___] jahre jetzt 4… |
| 61 | Löschung | `jetzt` | *(fehlt)* | …glaube 4 jahre [___] 4 jahre und… |
| 62 | Löschung | `4` | *(fehlt)* | …4 jahre jetzt [___] jahre und das… |
| 63 | Löschung | `jahre` | *(fehlt)* | …jahre jetzt 4 [___] und das kann… |
| 64 | Einfügung | *(nicht da)* | `auch` | …das kann man [___] dass das ein… |
| 65 | Einfügung | *(nicht da)* | `auch` | …sagen dass das [___] hobby ist ja… |
| 66 | Löschung | `ja` | *(fehlt)* | …ein hobby ist [___] genau und dass… |
| 67 | Löschung | `genau` | *(fehlt)* | …hobby ist ja [___] und dass sie… |
| 68 | Einfügung | *(nicht da)* | `dann` | …dass sie das [___] einmal wieder machen… |
| 69 | Substitution | `einmal` | `auch` | …sie das letztendlich [___] wieder machen wollen… |
| 70 | Einfügung | *(nicht da)* | `mal` | …das letztendlich einmal [___] machen wollen ja… |
| 71 | Substitution | `auf` | `okay` | …machen wollen ja [___] jeden fall ja… |
| 72 | Substitution | `jeden` | `genau` | …wollen ja auf [___] fall ja und… |
| 73 | Löschung | `fall` | *(fehlt)* | …ja auf jeden [___] ja und was… |
| 74 | Substitution | `dann` | `da` | …das dann waren [___] die schmerzen dann… |
| 75 | Löschung | `haben` | *(fehlt)* | …die schmerzen dann [___] sie dann bin… |
| 76 | Löschung | `sie` | *(fehlt)* | …schmerzen dann haben [___] dann bin ich… |
| 77 | Löschung | `dann` | *(fehlt)* | …dann haben sie [___] bin ich ins… |
| 78 | Substitution | `einmal` | `dann` | …ich ins krankenhaus [___] noch zur sicherheit… |
| 79 | Substitution | `noch` | `haben` | …ins krankenhaus einmal [___] zur sicherheit weil… |
| 80 | Substitution | `zur` | `wir` | …krankenhaus einmal noch [___] sicherheit weil es… |
| 81 | Einfügung | *(nicht da)* | `nach` | …einmal noch zur [___] weil es eben… |
| 82 | Einfügung | *(nicht da)* | `der` | …einmal noch zur [___] weil es eben… |
| 83 | Substitution | `weh` | `wehner` | …eben doch wirklich [___] getan hat und… |
| 84 | Löschung | `getan` | *(fehlt)* | …doch wirklich weh [___] hat und dann… |
| 85 | Substitution | `ein` | `auch` | …und dann war [___] röntgen und der… |
| 86 | Substitution | `und` | `dann` | …habe eigentlich nichts [___] bin wieder heimgefahren… |
| 87 | Einfügung | *(nicht da)* | `ich` | …nichts und bin [___] heimgefahren und da… |
| 88 | Löschung | `war` | *(fehlt)* | …heimgefahren und da [___] das mit da… |
| 89 | Löschung | `das` | *(fehlt)* | …und da war [___] mit da habe… |
| 90 | Löschung | `mit` | *(fehlt)* | …da war das [___] da habe ich… |
| 91 | Löschung | `da` | *(fehlt)* | …war das mit [___] habe ich keine… |
| 92 | Einfügung | *(nicht da)* | `auch` | …da habe ich [___] argen schmerzen gehabt… |
| 93 | Löschung | `argen` | *(fehlt)* | …habe ich keine [___] schmerzen gehabt und… |
| 94 | Substitution | `aufstehen` | `aufstecken` | …ich nicht mehr [___] können weil ich… |
| 95 | Substitution | `angeschwollen` | `angeschwohlen` | …war es auch [___] das knie dann… |
| 96 | Substitution | `kreuzband` | `kreuzbandel` | …eindeutig das vordere [___] gerissen ist also… |
| 97 | Substitution | `also` | `bevor` | …kreuzband gerissen ist [___] war mrt gemacht… |
| 98 | Substitution | `war` | `mbi` | …gerissen ist also [___] mrt gemacht worden… |
| 99 | Löschung | `mrt` | *(fehlt)* | …ist also war [___] gemacht worden ist… |
| 100 | Substitution | `kreuzband` | `kreuzbandel` | …dass das vordere [___] betroffen ist ja… |
| 101 | Substitution | `mit` | `den` | …dann habe ich [___] dem arzt ausgemacht… |
| 102 | Substitution | `dem` | `arztabendel` | …habe ich mit [___] arzt ausgemacht mit… |
| 103 | Substitution | `arzt` | `nicht` | …ich mit dem [___] ausgemacht mit einem… |
| 104 | Einfügung | *(nicht da)* | `mehr` | …mit dem arzt [___] mit einem anderen… |
| 105 | Substitution | `operationstermin` | `operationstabendel` | …arzt und dann [___] eigentlich auch gleich… |
| 106 | Löschung | `jetzt` | *(fehlt)* | …die war dann [___] vor ca einem… |
| 107 | Löschung | `vor` | *(fehlt)* | …war dann jetzt [___] ca einem monat… |
| 108 | Substitution | `einem` | `mal` | …jetzt vor ca [___] monat am 15… |
| 109 | Löschung | `monat` | *(fehlt)* | …vor ca einem [___] am 15 november… |
| 110 | Einfügung | *(nicht da)* | `immer` | …am 15 november [___] die operation ist… |
| 111 | Löschung | `wie` | *(fehlt)* | …sie das mitbekommen [___] war der heilungsprozess… |
| 112 | Substitution | `noch` | `nach` | …bin halt dann [___] glaube ich eine… |
| 113 | Substitution | `glaube` | `einer` | …halt dann noch [___] ich eine woche… |
| 114 | Löschung | `ich` | *(fehlt)* | …dann noch glaube [___] eine woche im… |
| 115 | Löschung | `eine` | *(fehlt)* | …noch glaube ich [___] woche im krankenhaus… |
| 116 | Substitution | `hab` | `ich` | …im krankenhaus gewesen [___] halt nicht viel… |
| 117 | Einfügung | *(nicht da)* | `habe` | …krankenhaus gewesen hab [___] nicht viel bewegung… |
| 118 | Substitution | `ist` | `hat` | …das ganze liegen [___] halt der muskel… |
| 119 | Löschung | `halt` | *(fehlt)* | …ganze liegen ist [___] der muskel komplett… |
| 120 | Substitution | `da` | `dann` | …im krankenhaus war [___] eine physiotherapeutin dann… |
| 121 | Substitution | `eine` | `der` | …krankenhaus war da [___] physiotherapeutin dann eingeteilt… |
| 122 | Substitution | `physiotherapeutin` | `physiotherapeut` | …war da eine [___] dann eingeteilt u… |
| 123 | Substitution | `dann` | `bei` | …da eine physiotherapeutin [___] eingeteilt u ja… |
| 124 | Einfügung | *(nicht da)* | `ihnen` | …eine physiotherapeutin dann [___] u ja da… |
| 125 | Löschung | `u` | *(fehlt)* | …physiotherapeutin dann eingeteilt [___] ja da war… |
| 126 | Löschung | `da` | *(fehlt)* | …eingeteilt u ja [___] war eine also… |
| 127 | Löschung | `war` | *(fehlt)* | …u ja da [___] eine also waren… |
| 128 | Löschung | `eine` | *(fehlt)* | …ja da war [___] also waren sie… |
| 129 | Löschung | `ja` | *(fehlt)* | …in physiotherapeutischer behandlung [___] da war es… |
| 130 | Substitution | `auch` | `einmal` | …der ist halt [___] mit dem doc… |
| 131 | Substitution | `mit` | `am` | …ist halt auch [___] dem doc gekommen… |
| 132 | Substitution | `dem` | `tag` | …halt auch mit [___] doc gekommen und… |
| 133 | Löschung | `doc` | *(fehlt)* | …auch mit dem [___] gekommen und das… |
| 134 | Substitution | `da` | `halt` | …und das war [___] die einzige bewegung… |
| 135 | Einfügung | *(nicht da)* | `auch` | …das war da [___] einzige bewegung die… |
| 136 | Substitution | `dort` | `da` | …bewegung die ich [___] eigentlich gehabt habe… |
| 137 | Einfügung | *(nicht da)* | `dann` | …die ich dort [___] gehabt habe mit… |
| 138 | Löschung | `haben` | *(fehlt)* | …haben sie da [___] sie schon alles… |
| 139 | Löschung | `sie` | *(fehlt)* | …sie da haben [___] schon alles ja… |
| 140 | Substitution | `alles` | `gemacht` | …haben sie schon [___] ja hauptsächlich hat… |
| 141 | Einfügung | *(nicht da)* | `mit` | …alles ja hauptsächlich [___] man gesagt wie… |
| 142 | Einfügung | *(nicht da)* | `ja` | …alles ja hauptsächlich [___] man gesagt wie… |
| 143 | Einfügung | *(nicht da)* | `dann` | …alles ja hauptsächlich [___] man gesagt wie… |
| 144 | Einfügung | *(nicht da)* | `halt` | …hauptsächlich hat man [___] wie man mit… |
| 145 | Substitution | `man` | `ich` | …man gesagt wie [___] mit den krücken… |
| 146 | Löschung | `also` | *(fehlt)* | …aufsteigen und das [___] das also dieses… |
| 147 | Löschung | `das` | *(fehlt)* | …und das also [___] also dieses abrollen… |
| 148 | Löschung | `also` | *(fehlt)* | …das also das [___] dieses abrollen mit… |
| 149 | Löschung | `dieses` | *(fehlt)* | …also das also [___] abrollen mit dem… |
| 150 | Substitution | `fuß` | `furs` | …abrollen mit dem [___] auch richtig einlernen… |
| 151 | Löschung | `ein` | *(fehlt)* | …am schluss sogar [___] bisschen haben wir… |
| 152 | Löschung | `bisschen` | *(fehlt)* | …schluss sogar ein [___] haben wir treppensteigen… |
| 153 | Substitution | `dass` | `also` | …wir treppensteigen dann [___] ich halt ja… |
| 154 | Löschung | `ich` | *(fehlt)* | …treppensteigen dann dass [___] halt ja und… |
| 155 | Löschung | `halt` | *(fehlt)* | …dann dass ich [___] ja und ein… |
| 156 | Substitution | `weit` | `war` | …beugen üben so [___] es halt gegangen… |
| 157 | Löschung | `ist` | *(fehlt)* | …es halt gegangen [___] und sie haben… |
| 158 | Löschung | `sie` | *(fehlt)* | …da schon eben [___] haben gesagt sie… |
| 159 | Löschung | `haben` | *(fehlt)* | …schon eben sie [___] gesagt sie haben… |
| 160 | Substitution | `abrollen` | `auch` | …sie haben das [___] gelernt können sie… |
| 161 | Einfügung | *(nicht da)* | `freuen` | …haben das abrollen [___] können sie das… |
| 162 | Substitution | `sich` | `sie` | …sie das für [___] sagen dass sie… |
| 163 | Einfügung | *(nicht da)* | `ja` | …auf jeden fall [___] ich darauf achte… |
| 164 | Substitution | `ich` | `sie` | …jeden fall dass [___] darauf achte wie… |
| 165 | Substitution | `achte` | `achten` | …dass ich darauf [___] wie ich aufsteige… |
| 166 | Einfügung | *(nicht da)* | `einfach` | …ich darauf achte [___] ich aufsteige genau… |
| 167 | Substitution | `ich` | `aufsteigen` | …darauf achte wie [___] aufsteige genau das… |
| 168 | Löschung | `aufsteige` | *(fehlt)* | …achte wie ich [___] genau das hat… |
| 169 | Einfügung | *(nicht da)* | `und` | …ich aufsteige genau [___] hat ihnen zum… |
| 170 | Substitution | `gern` | `gerne` | …sie haben da [___] mitgemacht in der… |
| 171 | Substitution | `ja` | `speaker_` | …der therapie und [___] ja das war… |
| 172 | Einfügung | *(nicht da)* | `okay` | …therapie und ja [___] das war halt… |
| 173 | Substitution | `es` | `ihnen` | …und wie ist [___] dann ergangen dann… |
| 174 | Substitution | `ergangen` | `da` | …ist es dann [___] dann mit den… |
| 175 | Substitution | `dann` | `gegangen` | …es dann ergangen [___] mit den schmerzen… |
| 176 | Löschung | `mit` | *(fehlt)* | …mit den schmerzen [___] den ja schmerzen… |
| 177 | Löschung | `den` | *(fehlt)* | …den schmerzen mit [___] ja schmerzen war… |
| 178 | Einfügung | *(nicht da)* | `war` | …sag ich mal [___] nachdem je nach… |
| 179 | Einfügung | *(nicht da)* | `okay` | …sag ich mal [___] nachdem je nach… |
| 180 | Einfügung | *(nicht da)* | `je` | …je nach belastung [___] habe mich halt… |
| 181 | Einfügung | *(nicht da)* | `nach` | …je nach belastung [___] habe mich halt… |
| 182 | Einfügung | *(nicht da)* | `belastung` | …je nach belastung [___] habe mich halt… |
| 183 | Einfügung | *(nicht da)* | `es` | …je nach belastung [___] habe mich halt… |
| 184 | Einfügung | *(nicht da)* | `war` | …je nach belastung [___] habe mich halt… |
| 185 | Einfügung | *(nicht da)* | `halt` | …je nach belastung [___] habe mich halt… |
| 186 | Einfügung | *(nicht da)* | `ich` | …viel bewegen können [___] ja eigentlich nur… |
| 187 | Einfügung | *(nicht da)* | `okay` | …eigentlich nur gelegen [___] erste woche die… |
| 188 | Substitution | `die` | `deine` | …die erste woche [___] letzte zeit und… |
| 189 | Substitution | `krücken` | `grücken` | …ja mit den [___] halt herumgegangen aber… |
| 190 | Substitution | `herumgegangen` | `herumgehen` | …den krücken halt [___] aber halt auch… |
| 191 | Einfügung | *(nicht da)* | `ein` | …krücken halt herumgegangen [___] halt auch minimal… |
| 192 | Einfügung | *(nicht da)* | `bisschen` | …krücken halt herumgegangen [___] halt auch minimal… |
| 193 | Löschung | `mal` | *(fehlt)* | …weit dass wir [___] darüber reden wie… |
| 194 | Einfügung | *(nicht da)* | `ich` | …schlimmsten schmerzen die [___] vorstellen können und… |
| 195 | Substitution | `können` | `könnte` | …die sich vorstellen [___] und 0 sind… |
| 196 | Löschung | `sind` | *(fehlt)* | …können und 0 [___] ist schmerzfrei wo… |
| 197 | Löschung | `drauf` | *(fehlt)* | …es kommt eigentlich [___] an auf die… |
| 198 | Löschung | `an` | *(fehlt)* | …kommt eigentlich drauf [___] auf die belastung… |
| 199 | Substitution | `drauf` | `darauf` | …auf die belastung [___] an wenn ich… |
| 200 | Substitution | `krücken` | `grücken` | …jetzt mit den [___] gehe dann keine… |
| 201 | Löschung | `zu` | *(fehlt)* | …ohne stützen probiere [___] gehen dann bin… |
| 202 | Löschung | `gehen` | *(fehlt)* | …stützen probiere zu [___] dann bin ich… |
| 203 | Substitution | `3` | `bei` | …6 oder 7 [___] 6 7 aber… |
| 204 | Einfügung | *(nicht da)* | `oder` | …7 3 6 [___] aber es ist… |
| 205 | Einfügung | *(nicht da)* | `ja` | …und das ist [___] einzige faktor die… |
| 206 | Substitution | `ihnen` | `einem` | …die belastung der [___] da einfällt wenn… |
| 207 | Substitution | `sehe` | `weiß` | …ja also ich [___] das nicht anders… |
| 208 | Einfügung | *(nicht da)* | `ja` | …also ich sehe [___] nicht anders ansonsten… |
| 209 | Substitution | `anders` | `an` | …sehe das nicht [___] ansonsten okay und… |
| 210 | Substitution | `ansonsten` | `was` | …das nicht anders [___] okay und sie… |
| 211 | Einfügung | *(nicht da)* | `sonst` | …nicht anders ansonsten [___] und sie haben… |
| 212 | Einfügung | *(nicht da)* | `noch` | …nicht anders ansonsten [___] und sie haben… |
| 213 | Einfügung | *(nicht da)* | `haben` | …haben gesagt sie [___] mit dem gehen… |
| 214 | Substitution | `kann` | `können` | …nur kurz was [___] ich mir da… |
| 215 | Substitution | `ich` | `wir` | …kurz was kann [___] mir da vorstellen… |
| 216 | Löschung | `mir` | *(fehlt)* | …was kann ich [___] da vorstellen also… |
| 217 | Substitution | `vorstellen` | `forschen` | …ich mir da [___] also sie sind… |
| 218 | Einfügung | *(nicht da)* | `sind` | …da vorstellen also [___] sind ab und… |
| 219 | Substitution | `sind` | `auf` | …vorstellen also sie [___] ab und zu… |
| 220 | Löschung | `ab` | *(fehlt)* | …also sie sind [___] und zu rausgegangen… |
| 221 | Löschung | `oder` | *(fehlt)* | …und zu rausgegangen [___] nein jetzt nicht… |
| 222 | Substitution | `einfluss` | `einfach` | …ich meine minimal [___] aber ich kann… |
| 223 | Substitution | `sagen` | `zusammenkriegen` | …kann nicht wirklich [___] ich gehe jetzt… |
| 224 | Löschung | `ich` | *(fehlt)* | …nicht wirklich sagen [___] gehe jetzt spazieren… |
| 225 | Löschung | `gehe` | *(fehlt)* | …wirklich sagen ich [___] jetzt spazieren oder… |
| 226 | Einfügung | *(nicht da)* | `zu` | …weit wie möglich [___] und das eigentlich… |
| 227 | Löschung | `ja` | *(fehlt)* | …belastung variiert aber [___] ist noch nicht… |
| 228 | Substitution | `nehme` | `nehmen` | …irgendwelche medikamente nein [___] ich nicht nehmen… |
| 229 | Substitution | `ich` | `sie` | …medikamente nein nehme [___] nicht nehmen sie… |
| 230 | Substitution | `nicht` | `nichts` | …nein nehme ich [___] nehmen sie nicht… |
| 231 | Substitution | `nicht` | `nichts` | …nicht nehmen sie [___] haben sie am… |
| 232 | Löschung | `am` | *(fehlt)* | …nicht haben sie [___] anfangs aber wahrscheinlich… |
| 233 | Substitution | `in` | `eine` | …anfangs aber wahrscheinlich [___] der behandlung schmerzmittel… |
| 234 | Löschung | `der` | *(fehlt)* | …aber wahrscheinlich in [___] behandlung schmerzmittel bekommen… |
| 235 | Löschung | `schmerzmittel` | *(fehlt)* | …in der behandlung [___] bekommen ja ich… |
| 236 | Löschung | `bekommen` | *(fehlt)* | …der behandlung schmerzmittel [___] ja ich habe… |
| 237 | Substitution | `am` | `manchmal` | …ja ich habe [___] anfang schmerzmittel gekriegt… |
| 238 | Löschung | `anfang` | *(fehlt)* | …ich habe am [___] schmerzmittel gekriegt am… |
| 239 | Substitution | `noch` | `gemacht` | …anfang sowieso infusionen [___] dann hätte ich… |
| 240 | Substitution | `noch` | `nochmal` | …dann hätte ich [___] schmerzmittel mitgehabt für… |
| 241 | Substitution | `gebraucht` | `braucht` | …eigentlich nicht mehr [___] also die mussten… |
| 242 | Substitution | `die` | `haben` | …mehr gebraucht also [___] mussten sie nicht… |
| 243 | Löschung | `mussten` | *(fehlt)* | …gebraucht also die [___] sie nicht nehmen… |
| 244 | Substitution | `nicht` | `das` | …die mussten sie [___] nehmen mit dem… |
| 245 | Substitution | `nehmen` | `benötigt` | …mussten sie nicht [___] mit dem home… |
| 246 | Substitution | `home` | `hometrainer` | …nehmen mit dem [___] trainer haben sie… |
| 247 | Löschung | `trainer` | *(fehlt)* | …mit dem home [___] haben sie erwähnt… |
| 248 | Substitution | `ne` | `nein` | …gemacht für übungen [___] keine eigentlich nur… |
| 249 | Löschung | `keine` | *(fehlt)* | …für übungen ne [___] eigentlich nur versucht… |
| 250 | Substitution | `wie` | `weil` | …eigentlich nur versucht [___] weit ich eben… |
| 251 | Löschung | `weit` | *(fehlt)* | …nur versucht wie [___] ich eben schon… |
| 252 | Einfügung | *(nicht da)* | `schon` | …und da ist [___] aber aufgefallen dass… |
| 253 | Einfügung | *(nicht da)* | `ist` | …und da ist [___] aber aufgefallen dass… |
| 254 | Substitution | `aber` | `da` | …da ist ihnen [___] aufgefallen dass es… |
| 255 | Einfügung | *(nicht da)* | `auch` | …ist ihnen aber [___] dass es einfach… |
| 256 | Einfügung | *(nicht da)* | `die` | …weitergegangen ist ja [___] jeden fall besser… |
| 257 | Einfügung | *(nicht da)* | `bewegung` | …weitergegangen ist ja [___] jeden fall besser… |
| 258 | Löschung | `dann` | *(fehlt)* | …haben auch fortschritte [___] bemerkt ja nur… |
| 259 | Substitution | `es` | `sie` | …zweiten stock also [___] ist in dem… |
| 260 | Substitution | `ist` | `sind` | …stock also es [___] in dem zweiten… |
| 261 | Substitution | `in` | `im` | …also es ist [___] dem zweiten stock… |
| 262 | Löschung | `dem` | *(fehlt)* | …es ist in [___] zweiten stock das… |
| 263 | Löschung | `treppen` | *(fehlt)* | …20 bis 30 [___] 20 bis 30… |
| 264 | Substitution | `mein` | `meine` | …ist halt ich [___] sicher funktioniert s… |
| 265 | Substitution | `s` | `es` | …mein sicher funktioniert [___] aber ich überleg… |
| 266 | Substitution | `überleg` | `überlege` | …s aber ich [___] mir halt ob… |
| 267 | Substitution | `oder` | `ja` | …wirklich runter muss [___] rauf muss haben… |
| 268 | Löschung | `rauf` | *(fehlt)* | …runter muss oder [___] muss haben sie… |
| 269 | Löschung | `muss` | *(fehlt)* | …muss oder rauf [___] haben sie irgendeine… |
| 270 | Löschung | `also` | *(fehlt)* | …die ihnen hilft [___] die so ja… |
| 271 | Löschung | `die` | *(fehlt)* | …ihnen hilft also [___] so ja ja… |
| 272 | Löschung | `so` | *(fehlt)* | …hilft also die [___] ja ja ja… |
| 273 | Löschung | `ja` | *(fehlt)* | …die so ja [___] ja also familie… |
| 274 | Löschung | `ja` | *(fehlt)* | …so ja ja [___] also familie freunde… |
| 275 | Löschung | `nur` | *(fehlt)* | …irgendwelche nebendiagnosen wie [___] ein beispiel diabetes… |
| 276 | Löschung | `nein` | *(fehlt)* | …beispiel diabetes oder [___] dass sie wüssten… |
| 277 | Substitution | `narconabhängigkeiten` | `auch` | …dass sie wüssten [___] nein sie stehen… |
| 278 | Substitution | `nein` | `keine` | …sie wüssten narconabhängigkeiten [___] sie stehen ja… |
| 279 | Einfügung | *(nicht da)* | `auffälligkeiten` | …wüssten narconabhängigkeiten nein [___] stehen ja sonst… |
| 280 | Einfügung | *(nicht da)* | `und` | …wüssten narconabhängigkeiten nein [___] stehen ja sonst… |
| 281 | Substitution | `stehen` | `stängern` | …narconabhängigkeiten nein sie [___] ja sonst nicht… |
| 282 | Substitution | `ja` | `auch` | …nein sie stehen [___] sonst nicht unter… |
| 283 | Substitution | `medikamenten` | `medikamente` | …sonst nicht unter [___] also abgesehen von… |
| 284 | Substitution | `den` | `der` | …also abgesehen von [___] schmerz nein tabletten… |
| 285 | Substitution | `schmerz` | `schmerztabletten` | …abgesehen von den [___] nein tabletten nein… |
| 286 | Löschung | `tabletten` | *(fehlt)* | …den schmerz nein [___] nein die nehme… |
| 287 | Löschung | `nehme` | *(fehlt)* | …tabletten nein die [___] ich nicht mehr… |
| 288 | Löschung | `ich` | *(fehlt)* | …nein die nehme [___] nicht mehr okay… |
| 289 | Substitution | `sich` | `sie` | …was würden sie [___] wünschen was erwarten… |
| 290 | Substitution | `sich` | `sie` | …was erwarten sie [___] dass ich halt… |
| 291 | Einfügung | *(nicht da)* | `auch` | …normal machen kann [___] ich eben wieder… |
| 292 | Substitution | `sport` | `sportlen` | …dass ich normal [___] machen kann also… |
| 293 | Löschung | `machen` | *(fehlt)* | …ich normal sport [___] kann also der… |
| 294 | Einfügung | *(nicht da)* | `auch` | …sport steht da [___] vordergrund ja in… |
| 295 | Einfügung | *(nicht da)* | `ja` | …im vordergrund ja [___] weiterer folge natürlich… |
| 296 | Substitution | `vor` | `vorher` | …weiterer folge natürlich [___] allem möchte ich… |
| 297 | Löschung | `allem` | *(fehlt)* | …folge natürlich vor [___] möchte ich wieder… |
| 298 | Substitution | `wieder` | `mir` | …allem möchte ich [___] normal gehen können… |
| 299 | Einfügung | *(nicht da)* | `keine` | …normal gehen können [___] vielen dank frau… |
| 300 | Einfügung | *(nicht da)* | `lüge` | …normal gehen können [___] vielen dank frau… |
| 301 | Einfügung | *(nicht da)* | `speaker_` | …normal gehen können [___] vielen dank frau… |
| 302 | Einfügung | *(nicht da)* | `und` | …normal gehen können [___] vielen dank frau… |
| 303 | Einfügung | *(nicht da)* | `ja` | …normal gehen können [___] vielen dank frau… |
| 304 | Substitution | `grasbäutner` | `krebspartner` | …vielen dank frau [___] und wir treffen… |
| 305 | Substitution | `zu` | `zur` | …treffen uns dann [___] der nächsten behandlung… |
| 306 | Löschung | `der` | *(fehlt)* | …uns dann zu [___] nächsten behandlung danke… |
| 307 | Einfügung | *(nicht da)* | `ich` | …nächsten behandlung danke [___] … |
| 308 | Einfügung | *(nicht da)* | `hoffe` | …nächsten behandlung danke [___] … |
| 309 | Einfügung | *(nicht da)* | `dass` | …nächsten behandlung danke [___] … |
| 310 | Einfügung | *(nicht da)* | `sie` | …nächsten behandlung danke [___] … |
| 311 | Einfügung | *(nicht da)* | `das` | …nächsten behandlung danke [___] … |
| 312 | Einfügung | *(nicht da)* | `passen` | …nächsten behandlung danke [___] … |
| 313 | Einfügung | *(nicht da)* | `speaker_` | …nächsten behandlung danke [___] … |
| 314 | Einfügung | *(nicht da)* | `ich` | …nächsten behandlung danke [___] … |
| 315 | Einfügung | *(nicht da)* | `hoffe` | …nächsten behandlung danke [___] … |
| 316 | Einfügung | *(nicht da)* | `dass` | …nächsten behandlung danke [___] … |
| 317 | Einfügung | *(nicht da)* | `sie` | …nächsten behandlung danke [___] … |
| 318 | Einfügung | *(nicht da)* | `das` | …nächsten behandlung danke [___] … |
| 319 | Einfügung | *(nicht da)* | `passen` | …nächsten behandlung danke [___] … |
| 320 | Einfügung | *(nicht da)* | `danke` | …nächsten behandlung danke [___] … |
| 321 | Einfügung | *(nicht da)* | `danke` | …nächsten behandlung danke [___] … |
| 322 | Einfügung | *(nicht da)* | `danke` | …nächsten behandlung danke [___] … |
| 323 | Einfügung | *(nicht da)* | `danke` | …nächsten behandlung danke [___] … |
