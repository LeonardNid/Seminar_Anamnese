# LLM-Fehleranalyse: Whisper large-v3-turbo + llama3.2

> RAW STT → Formatted — Satzzeichen und Groß-/Kleinschreibung ignoriert.
> Speaker-Label-Änderungen sind bereits aus der JSON entfernt.
> **S** = Substitution | **D** = Löschung (im RAW, fehlt im FMT) | **I** = Einfügung (im FMT, nicht im RAW)

---

## Übersicht

| Szenario | RAW-Wörter | FMT-Wörter | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 247 | 269 | 5 | 6 | 28 | 39 | 15.8% |
| OriginalDC+Noise | 210 | 210 | 2 | 0 | 0 | 2 | 1.0% |
| LapInMitte | 226 | 226 | 0 | 0 | 0 | 0 | 0.0% |
| LapBeiArzt | 229 | 252 | 6 | 1 | 24 | 31 | 13.5% |
| Selbstkorrekturen | 190 | 193 | 0 | 0 | 3 | 3 | 1.6% |
| Unterbrechungen | 143 | 159 | 4 | 0 | 16 | 20 | 14.0% |
| Gedankensprünge | 190 | 190 | 0 | 0 | 0 | 0 | 0.0% |
| Meinungswechsel | 183 | 185 | 0 | 0 | 2 | 2 | 1.1% |
| Chaos | 252 | 253 | 1 | 0 | 1 | 2 | 0.8% |
| Anamnesegespräch | 2269 | 149 | 82 | 2121 | 1 | 2204 | 97.1% |
| PWC | 1511 | 1203 | 4 | 309 | 1 | 314 | 20.8% |

---

## OriginalDC

**Fehlerrate: 15.8%** — RAW: 247 Wörter | FMT: 269 Wörter | S=5 D=6 I=28 | Fehler=39

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `1` | (FMT) …[___] der arzt ist… |
| 2 | Einfügung | `*(nicht da)*` | `der` | (FMT) …1 [___] arzt ist dr… |
| 3 | Einfügung | `*(nicht da)*` | `arzt` | (FMT) …1 der [___] ist dr weber… |
| 4 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …1 der arzt [___] dr weber nicht… |
| 5 | Einfügung | `*(nicht da)*` | `dr` | (FMT) …der arzt ist [___] weber nicht genannt… |
| 6 | Einfügung | `*(nicht da)*` | `weber` | (FMT) …arzt ist dr [___] nicht genannt und… |
| 7 | Einfügung | `*(nicht da)*` | `nicht` | (FMT) …ist dr weber [___] genannt und der… |
| 8 | Einfügung | `*(nicht da)*` | `genannt` | (FMT) …dr weber nicht [___] und der patient… |
| 9 | Einfügung | `*(nicht da)*` | `und` | (FMT) …weber nicht genannt [___] der patient ist… |
| 10 | Einfügung | `*(nicht da)*` | `der` | (FMT) …nicht genannt und [___] patient ist frau… |
| 11 | Einfügung | `*(nicht da)*` | `patient` | (FMT) …genannt und der [___] ist frau weber… |
| 12 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …und der patient [___] frau weber 2… |
| 13 | Einfügung | `*(nicht da)*` | `frau` | (FMT) …der patient ist [___] weber 2 der… |
| 14 | Einfügung | `*(nicht da)*` | `weber` | (FMT) …patient ist frau [___] 2 der name… |
| 15 | Einfügung | `*(nicht da)*` | `2` | (FMT) …ist frau weber [___] der name des… |
| 16 | Einfügung | `*(nicht da)*` | `der` | (FMT) …frau weber 2 [___] name des patienten… |
| 17 | Einfügung | `*(nicht da)*` | `name` | (FMT) …weber 2 der [___] des patienten ist… |
| 18 | Einfügung | `*(nicht da)*` | `des` | (FMT) …2 der name [___] patienten ist frau… |
| 19 | Einfügung | `*(nicht da)*` | `patienten` | (FMT) …der name des [___] ist frau weber… |
| 20 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …name des patienten [___] frau weber 3… |
| 21 | Einfügung | `*(nicht da)*` | `frau` | (FMT) …des patienten ist [___] weber 3 hier… |
| 22 | Einfügung | `*(nicht da)*` | `weber` | (FMT) …patienten ist frau [___] 3 hier ist… |
| 23 | Einfügung | `*(nicht da)*` | `3` | (FMT) …ist frau weber [___] hier ist das… |
| 24 | Einfügung | `*(nicht da)*` | `hier` | (FMT) …frau weber 3 [___] ist das formatierte… |
| 25 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …weber 3 hier [___] das formatierte transkript… |
| 26 | Einfügung | `*(nicht da)*` | `das` | (FMT) …3 hier ist [___] formatierte transkript guten… |
| 27 | Einfügung | `*(nicht da)*` | `formatierte` | (FMT) …hier ist das [___] transkript guten morgen… |
| 28 | Einfügung | `*(nicht da)*` | `transkript` | (FMT) …ist das formatierte [___] guten morgen frau… |
| 29 | Löschung | `notiz` | `*(nicht da)*` | …und kalt verstehe [___] leitsymptom husten seit… |
| 30 | Substitution | `notiz` | `leitsymptom` | …dem thermometer okay [___] subfibrile bis fibrile… |
| 31 | Substitution | `notiz` | `leitsymptom` | …wichtig zu wissen [___] stark produktiver husten… |
| 32 | Substitution | `notiz` | `leitsymptom` | …sie das erwähnen [___] deutliche belastungsdyspnoe aber… |
| 33 | Substitution | `notiz` | `leitsymptom` | …kamellentee alles klar [___] keine spezifische vormedikation… |
| 34 | Substitution | `notiz` | `leitsymptom` | …ein und aus [___] beginn der auskultation… |
| 35 | Löschung | `chicks` | `*(nicht da)*` | …auskultation der pulmonis [___] da dass ihr… |
| 36 | Löschung | `untertitelung` | `*(nicht da)*` | …kann sie dir [___] des zdf 2020… |
| 37 | Löschung | `des` | `*(nicht da)*` | …sie dir untertitelung [___] zdf 2020… |
| 38 | Löschung | `zdf` | `*(nicht da)*` | …dir untertitelung des [___] 2020… |
| 39 | Löschung | `2020` | `*(nicht da)*` | …untertitelung des zdf [___]… |

---

## OriginalDC+Noise

**Fehlerrate: 1.0%** — RAW: 210 Wörter | FMT: 210 Wörter | S=2 D=0 I=0 | Fehler=2

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Substitution | `tiber` | `tuberkulose` | …verdacht auf präzidivierende [___] haben sie das… |
| 2 | Substitution | `karte` | `küche` | …gerade in der [___] okay notiz subfibrile… |

---

## LapInMitte

**Fehlerrate: 0.0%** — RAW: 226 Wörter | FMT: 226 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## LapBeiArzt

**Fehlerrate: 13.5%** — RAW: 229 Wörter | FMT: 252 Wörter | S=6 D=1 I=24 | Fehler=31

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `1` | (FMT) …[___] der arzt ist… |
| 2 | Einfügung | `*(nicht da)*` | `der` | (FMT) …1 [___] arzt ist herr… |
| 3 | Einfügung | `*(nicht da)*` | `arzt` | (FMT) …1 der [___] ist herr doktor… |
| 4 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …1 der arzt [___] herr doktor also… |
| 5 | Einfügung | `*(nicht da)*` | `herr` | (FMT) …der arzt ist [___] doktor also arzt… |
| 6 | Einfügung | `*(nicht da)*` | `doktor` | (FMT) …arzt ist herr [___] also arzt 2… |
| 7 | Einfügung | `*(nicht da)*` | `also` | (FMT) …ist herr doktor [___] arzt 2 der… |
| 8 | Einfügung | `*(nicht da)*` | `arzt` | (FMT) …herr doktor also [___] 2 der patient… |
| 9 | Einfügung | `*(nicht da)*` | `2` | (FMT) …doktor also arzt [___] der patient heißt… |
| 10 | Einfügung | `*(nicht da)*` | `der` | (FMT) …also arzt 2 [___] patient heißt frau… |
| 11 | Einfügung | `*(nicht da)*` | `patient` | (FMT) …arzt 2 der [___] heißt frau weber… |
| 12 | Einfügung | `*(nicht da)*` | `heißt` | (FMT) …2 der patient [___] frau weber also… |
| 13 | Einfügung | `*(nicht da)*` | `frau` | (FMT) …der patient heißt [___] weber also frau… |
| 14 | Einfügung | `*(nicht da)*` | `weber` | (FMT) …patient heißt frau [___] also frau weber… |
| 15 | Einfügung | `*(nicht da)*` | `also` | (FMT) …heißt frau weber [___] frau weber 3… |
| 16 | Einfügung | `*(nicht da)*` | `frau` | (FMT) …frau weber also [___] weber 3 hier… |
| 17 | Einfügung | `*(nicht da)*` | `weber` | (FMT) …weber also frau [___] 3 hier ist… |
| 18 | Einfügung | `*(nicht da)*` | `3` | (FMT) …also frau weber [___] hier ist das… |
| 19 | Einfügung | `*(nicht da)*` | `hier` | (FMT) …frau weber 3 [___] ist das formatierte… |
| 20 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …weber 3 hier [___] das formatierte transkript… |
| 21 | Einfügung | `*(nicht da)*` | `das` | (FMT) …3 hier ist [___] formatierte transkript guten… |
| 22 | Einfügung | `*(nicht da)*` | `formatierte` | (FMT) …hier ist das [___] transkript guten morgen… |
| 23 | Einfügung | `*(nicht da)*` | `transkript` | (FMT) …ist das formatierte [___] guten morgen frau… |
| 24 | Löschung | `notiz` | `*(nicht da)*` | …und kalt verstehe [___] leitsymptom husten seit… |
| 25 | Substitution | `notiz` | `leitsymptom` | …dem thermometer okay [___] subfebrile bis febrile… |
| 26 | Substitution | `notiz` | `leitsymptom` | …wichtig zu wissen [___] stark produktiver husten… |
| 27 | Substitution | `notiz` | `leitsymptom` | …sie das erwähnen [___] deutliche belastungsdyspnoe keine… |
| 28 | Substitution | `notiz` | `leitsymptom` | …vielfamilien alles klar [___] keine spezifische vormedikation… |
| 29 | Substitution | `notiz` | `leitsymptom` | …ein und aus [___] beginn der auskultation… |
| 30 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …der pulmonis das [___] vielen dank… |
| 31 | Substitution | `istvielen` | `vielen` | …der pulmonis das [___] dank… |

---

## Selbstkorrekturen

**Fehlerrate: 1.6%** — RAW: 190 Wörter | FMT: 193 Wörter | S=0 D=0 I=3 | Fehler=3

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `patient` | (FMT) …ja das ist [___] herr berger… |
| 2 | Einfügung | `*(nicht da)*` | `herr` | (FMT) …das ist patient [___] berger… |
| 3 | Einfügung | `*(nicht da)*` | `berger` | (FMT) …ist patient herr [___]… |

---

## Unterbrechungen

**Fehlerrate: 14.0%** — RAW: 143 Wörter | FMT: 159 Wörter | S=4 D=0 I=16 | Fehler=20

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `3` | (FMT) …es bei 38 [___] 38 3 ist… |
| 2 | Einfügung | `*(nicht da)*` | `3` | (FMT) …38 3 38 [___] ist eine temperatur… |
| 3 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …3 38 3 [___] eine temperatur von… |
| 4 | Einfügung | `*(nicht da)*` | `eine` | (FMT) …38 3 ist [___] temperatur von dreiundachtzig… |
| 5 | Einfügung | `*(nicht da)*` | `temperatur` | (FMT) …3 ist eine [___] von dreiundachtzig grad… |
| 6 | Einfügung | `*(nicht da)*` | `von` | (FMT) …ist eine temperatur [___] dreiundachtzig grad celsius… |
| 7 | Einfügung | `*(nicht da)*` | `dreiundachtzig` | (FMT) …eine temperatur von [___] grad celsius komma… |
| 8 | Einfügung | `*(nicht da)*` | `grad` | (FMT) …temperatur von dreiundachtzig [___] celsius komma zwei… |
| 9 | Substitution | `was` | `celsius` | …bei 38 38 [___] komma 2 komma… |
| 10 | Substitution | `2` | `zwei` | …38 was komma [___] komma 5 komma… |
| 11 | Einfügung | `*(nicht da)*` | `fünf` | (FMT) …komma zwei komma [___] oder komma drei… |
| 12 | Substitution | `5` | `oder` | …komma 2 komma [___] komma 3 und… |
| 13 | Einfügung | `*(nicht da)*` | `drei` | (FMT) …fünf oder komma [___] sind keine spezifischen… |
| 14 | Einfügung | `*(nicht da)*` | `sind` | (FMT) …oder komma drei [___] keine spezifischen werte… |
| 15 | Einfügung | `*(nicht da)*` | `keine` | (FMT) …komma drei sind [___] spezifischen werte für… |
| 16 | Einfügung | `*(nicht da)*` | `spezifischen` | (FMT) …drei sind keine [___] werte für die… |
| 17 | Einfügung | `*(nicht da)*` | `werte` | (FMT) …sind keine spezifischen [___] für die temperatur… |
| 18 | Einfügung | `*(nicht da)*` | `für` | (FMT) …keine spezifischen werte [___] die temperatur und… |
| 19 | Einfügung | `*(nicht da)*` | `die` | (FMT) …spezifischen werte für [___] temperatur und ich… |
| 20 | Substitution | `3` | `temperatur` | …komma 5 komma [___] und ich habe… |

---

## Gedankensprünge

**Fehlerrate: 0.0%** — RAW: 190 Wörter | FMT: 190 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Meinungswechsel

**Fehlerrate: 1.1%** — RAW: 183 Wörter | FMT: 185 Wörter | S=0 D=0 I=2 | Fehler=2

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `patient` | (FMT) …das war s [___] in das war… |
| 2 | Einfügung | `*(nicht da)*` | `in` | (FMT) …war s patient [___] das war s… |

---

## Chaos

**Fehlerrate: 0.8%** — RAW: 252 Wörter | FMT: 253 Wörter | S=1 D=0 I=1 | Fehler=2

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `herr` | (FMT) …auf den beinen [___] schuster mir ist… |
| 2 | Substitution | `ja` | `schuster` | …auf den beinen [___] mir ist so… |

---

## Anamnesegespräch

**Fehlerrate: 97.1%** — RAW: 2269 Wörter | FMT: 149 Wörter | S=82 D=2121 I=1 | Fehler=2204

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Löschung | `schönen` | `*(nicht da)*` | …[___] guten tag mein… |
| 2 | Löschung | `guten` | `*(nicht da)*` | …schönen [___] tag mein name… |
| 3 | Löschung | `tag` | `*(nicht da)*` | …schönen guten [___] mein name ist… |
| 4 | Löschung | `mein` | `*(nicht da)*` | …schönen guten tag [___] name ist nina… |
| 5 | Löschung | `name` | `*(nicht da)*` | …guten tag mein [___] ist nina colett… |
| 6 | Löschung | `ist` | `*(nicht da)*` | …tag mein name [___] nina colett und… |
| 7 | Löschung | `nina` | `*(nicht da)*` | …mein name ist [___] colett und ich… |
| 8 | Löschung | `colett` | `*(nicht da)*` | …name ist nina [___] und ich bin… |
| 9 | Löschung | `und` | `*(nicht da)*` | …ist nina colett [___] ich bin als… |
| 10 | Löschung | `ich` | `*(nicht da)*` | …nina colett und [___] bin als assistenzarzt… |
| 11 | Löschung | `bin` | `*(nicht da)*` | …colett und ich [___] als assistenzarzt hier… |
| 12 | Löschung | `als` | `*(nicht da)*` | …und ich bin [___] assistenzarzt hier auf… |
| 13 | Löschung | `assistenzarzt` | `*(nicht da)*` | …ich bin als [___] hier auf der… |
| 14 | Löschung | `hier` | `*(nicht da)*` | …bin als assistenzarzt [___] auf der station… |
| 15 | Löschung | `auf` | `*(nicht da)*` | …als assistenzarzt hier [___] der station tätig… |
| 16 | Löschung | `der` | `*(nicht da)*` | …assistenzarzt hier auf [___] station tätig ich… |
| 17 | Löschung | `station` | `*(nicht da)*` | …hier auf der [___] tätig ich würde… |
| 18 | Löschung | `tätig` | `*(nicht da)*` | …auf der station [___] ich würde gerne… |
| 19 | Löschung | `ich` | `*(nicht da)*` | …der station tätig [___] würde gerne das… |
| 20 | Löschung | `würde` | `*(nicht da)*` | …station tätig ich [___] gerne das aufnahmegespräch… |
| 21 | Löschung | `gerne` | `*(nicht da)*` | …tätig ich würde [___] das aufnahmegespräch mit… |
| 22 | Löschung | `das` | `*(nicht da)*` | …ich würde gerne [___] aufnahmegespräch mit ihnen… |
| 23 | Löschung | `aufnahmegespräch` | `*(nicht da)*` | …würde gerne das [___] mit ihnen durchführen… |
| 24 | Löschung | `mit` | `*(nicht da)*` | …gerne das aufnahmegespräch [___] ihnen durchführen sind… |
| 25 | Löschung | `ihnen` | `*(nicht da)*` | …das aufnahmegespräch mit [___] durchführen sind sie… |
| 26 | Löschung | `durchführen` | `*(nicht da)*` | …aufnahmegespräch mit ihnen [___] sind sie damit… |
| 27 | Löschung | `sind` | `*(nicht da)*` | …mit ihnen durchführen [___] sie damit einverstanden… |
| 28 | Löschung | `sie` | `*(nicht da)*` | …ihnen durchführen sind [___] damit einverstanden guten… |
| 29 | Löschung | `damit` | `*(nicht da)*` | …durchführen sind sie [___] einverstanden guten tag… |
| 30 | Löschung | `einverstanden` | `*(nicht da)*` | …sind sie damit [___] guten tag ja… |
| 31 | Löschung | `guten` | `*(nicht da)*` | …sie damit einverstanden [___] tag ja natürlich… |
| 32 | Löschung | `tag` | `*(nicht da)*` | …damit einverstanden guten [___] ja natürlich sehr… |
| 33 | Löschung | `ja` | `*(nicht da)*` | …einverstanden guten tag [___] natürlich sehr gerne… |
| 34 | Löschung | `natürlich` | `*(nicht da)*` | …guten tag ja [___] sehr gerne wunderbar… |
| 35 | Löschung | `sehr` | `*(nicht da)*` | …tag ja natürlich [___] gerne wunderbar wie… |
| 36 | Löschung | `gerne` | `*(nicht da)*` | …ja natürlich sehr [___] wunderbar wie heißen… |
| 37 | Löschung | `wunderbar` | `*(nicht da)*` | …natürlich sehr gerne [___] wie heißen sie… |
| 38 | Löschung | `wie` | `*(nicht da)*` | …sehr gerne wunderbar [___] heißen sie denn… |
| 39 | Löschung | `heißen` | `*(nicht da)*` | …gerne wunderbar wie [___] sie denn ich… |
| 40 | Löschung | `sie` | `*(nicht da)*` | …wunderbar wie heißen [___] denn ich heiße… |
| 41 | Löschung | `denn` | `*(nicht da)*` | …wie heißen sie [___] ich heiße julia… |
| 42 | Löschung | `ich` | `*(nicht da)*` | …heißen sie denn [___] heiße julia becken… |
| 43 | Löschung | `heiße` | `*(nicht da)*` | …sie denn ich [___] julia becken westfalen… |
| 44 | Löschung | `julia` | `*(nicht da)*` | …denn ich heiße [___] becken westfalen julia… |
| 45 | Löschung | `becken` | `*(nicht da)*` | …ich heiße julia [___] westfalen julia becken… |
| 46 | Löschung | `westfalen` | `*(nicht da)*` | …heiße julia becken [___] julia becken westfalen… |
| 47 | Löschung | `julia` | `*(nicht da)*` | …julia becken westfalen [___] becken westfalen können… |
| 48 | Löschung | `becken` | `*(nicht da)*` | …becken westfalen julia [___] westfalen können sie… |
| 49 | Löschung | `westfalen` | `*(nicht da)*` | …westfalen julia becken [___] können sie mir… |
| 50 | Löschung | `können` | `*(nicht da)*` | …julia becken westfalen [___] sie mir ihren… |
| 51 | Löschung | `sie` | `*(nicht da)*` | …becken westfalen können [___] mir ihren nachnamen… |
| 52 | Löschung | `mir` | `*(nicht da)*` | …westfalen können sie [___] ihren nachnamen bitte… |
| 53 | Löschung | `ihren` | `*(nicht da)*` | …können sie mir [___] nachnamen bitte einmal… |
| 54 | Löschung | `nachnamen` | `*(nicht da)*` | …sie mir ihren [___] bitte einmal langsam… |
| 55 | Löschung | `bitte` | `*(nicht da)*` | …mir ihren nachnamen [___] einmal langsam buchstabieren… |
| 56 | Löschung | `einmal` | `*(nicht da)*` | …ihren nachnamen bitte [___] langsam buchstabieren gerne… |
| 57 | Löschung | `langsam` | `*(nicht da)*` | …nachnamen bitte einmal [___] buchstabieren gerne ja… |
| 58 | Löschung | `buchstabieren` | `*(nicht da)*` | …bitte einmal langsam [___] gerne ja becken… |
| 59 | Löschung | `gerne` | `*(nicht da)*` | …einmal langsam buchstabieren [___] ja becken westfalen… |
| 60 | Löschung | `ja` | `*(nicht da)*` | …langsam buchstabieren gerne [___] becken westfalen westfalen… |
| 61 | Löschung | `becken` | `*(nicht da)*` | …buchstabieren gerne ja [___] westfalen westfalen westphalen… |
| 62 | Löschung | `westfalen` | `*(nicht da)*` | …gerne ja becken [___] westfalen westphalen alles… |
| 63 | Löschung | `westfalen` | `*(nicht da)*` | …ja becken westfalen [___] westphalen alles klar… |
| 64 | Löschung | `westphalen` | `*(nicht da)*` | …becken westfalen westfalen [___] alles klar dankeschön… |
| 65 | Löschung | `alles` | `*(nicht da)*` | …westfalen westfalen westphalen [___] klar dankeschön frau… |
| 66 | Löschung | `klar` | `*(nicht da)*` | …westfalen westphalen alles [___] dankeschön frau becken… |
| 67 | Löschung | `dankeschön` | `*(nicht da)*` | …westphalen alles klar [___] frau becken westfalen… |
| 68 | Löschung | `frau` | `*(nicht da)*` | …alles klar dankeschön [___] becken westfalen wie… |
| 69 | Löschung | `becken` | `*(nicht da)*` | …klar dankeschön frau [___] westfalen wie alt… |
| 70 | Löschung | `westfalen` | `*(nicht da)*` | …dankeschön frau becken [___] wie alt sind… |
| 71 | Löschung | `wie` | `*(nicht da)*` | …frau becken westfalen [___] alt sind sie… |
| 72 | Löschung | `alt` | `*(nicht da)*` | …becken westfalen wie [___] sind sie denn… |
| 73 | Löschung | `sind` | `*(nicht da)*` | …westfalen wie alt [___] sie denn 33… |
| 74 | Löschung | `sie` | `*(nicht da)*` | …wie alt sind [___] denn 33 und… |
| 75 | Löschung | `denn` | `*(nicht da)*` | …alt sind sie [___] 33 und wann… |
| 76 | Löschung | `33` | `*(nicht da)*` | …sind sie denn [___] und wann ist… |
| 77 | Löschung | `und` | `*(nicht da)*` | …sie denn 33 [___] wann ist ihr… |
| 78 | Löschung | `wann` | `*(nicht da)*` | …denn 33 und [___] ist ihr geburtstag… |
| 79 | Löschung | `ist` | `*(nicht da)*` | …33 und wann [___] ihr geburtstag am… |
| 80 | Löschung | `ihr` | `*(nicht da)*` | …und wann ist [___] geburtstag am 27… |
| 81 | Löschung | `geburtstag` | `*(nicht da)*` | …wann ist ihr [___] am 27 märz… |
| 82 | Löschung | `am` | `*(nicht da)*` | …ist ihr geburtstag [___] 27 märz 1987… |
| 83 | Löschung | `27` | `*(nicht da)*` | …ihr geburtstag am [___] märz 1987 schön… |
| 84 | Löschung | `märz` | `*(nicht da)*` | …geburtstag am 27 [___] 1987 schön herzlichen… |
| 85 | Löschung | `1987` | `*(nicht da)*` | …am 27 märz [___] schön herzlichen glückwunsch… |
| 86 | Löschung | `schön` | `*(nicht da)*` | …27 märz 1987 [___] herzlichen glückwunsch nachträglich… |
| 87 | Löschung | `herzlichen` | `*(nicht da)*` | …märz 1987 schön [___] glückwunsch nachträglich vielen… |
| 88 | Löschung | `glückwunsch` | `*(nicht da)*` | …1987 schön herzlichen [___] nachträglich vielen dank… |
| 89 | Löschung | `nachträglich` | `*(nicht da)*` | …schön herzlichen glückwunsch [___] vielen dank frau… |
| 90 | Löschung | `vielen` | `*(nicht da)*` | …herzlichen glückwunsch nachträglich [___] dank frau becken… |
| 91 | Löschung | `dank` | `*(nicht da)*` | …glückwunsch nachträglich vielen [___] frau becken westfalen… |
| 92 | Löschung | `frau` | `*(nicht da)*` | …nachträglich vielen dank [___] becken westfalen wie… |
| 93 | Löschung | `becken` | `*(nicht da)*` | …vielen dank frau [___] westfalen wie groß… |
| 94 | Löschung | `westfalen` | `*(nicht da)*` | …dank frau becken [___] wie groß sind… |
| 95 | Löschung | `wie` | `*(nicht da)*` | …frau becken westfalen [___] groß sind sie… |
| 96 | Löschung | `groß` | `*(nicht da)*` | …becken westfalen wie [___] sind sie denn… |
| 97 | Löschung | `sind` | `*(nicht da)*` | …westfalen wie groß [___] sie denn 1… |
| 98 | Löschung | `sie` | `*(nicht da)*` | …wie groß sind [___] denn 1 70… |
| 99 | Löschung | `denn` | `*(nicht da)*` | …groß sind sie [___] 1 70 1… |
| 100 | Löschung | `1` | `*(nicht da)*` | …sind sie denn [___] 70 1 70… |
| 101 | Löschung | `70` | `*(nicht da)*` | …sie denn 1 [___] 1 70 alles… |
| 102 | Löschung | `1` | `*(nicht da)*` | …denn 1 70 [___] 70 alles klar… |
| 103 | Löschung | `70` | `*(nicht da)*` | …1 70 1 [___] alles klar und… |
| 104 | Löschung | `alles` | `*(nicht da)*` | …70 1 70 [___] klar und wie… |
| 105 | Löschung | `klar` | `*(nicht da)*` | …1 70 alles [___] und wie viel… |
| 106 | Löschung | `und` | `*(nicht da)*` | …70 alles klar [___] wie viel wiegen… |
| 107 | Löschung | `wie` | `*(nicht da)*` | …alles klar und [___] viel wiegen sie… |
| 108 | Löschung | `viel` | `*(nicht da)*` | …klar und wie [___] wiegen sie zurzeit… |
| 109 | Löschung | `wiegen` | `*(nicht da)*` | …und wie viel [___] sie zurzeit 60… |
| 110 | Löschung | `sie` | `*(nicht da)*` | …wie viel wiegen [___] zurzeit 60 kilo… |
| 111 | Löschung | `zurzeit` | `*(nicht da)*` | …viel wiegen sie [___] 60 kilo glaube… |
| 112 | Löschung | `60` | `*(nicht da)*` | …wiegen sie zurzeit [___] kilo glaube ich… |
| 113 | Löschung | `kilo` | `*(nicht da)*` | …sie zurzeit 60 [___] glaube ich okay… |
| 114 | Löschung | `glaube` | `*(nicht da)*` | …zurzeit 60 kilo [___] ich okay gut… |
| 115 | Löschung | `ich` | `*(nicht da)*` | …60 kilo glaube [___] okay gut können… |
| 116 | Löschung | `okay` | `*(nicht da)*` | …kilo glaube ich [___] gut können sie… |
| 117 | Löschung | `gut` | `*(nicht da)*` | …glaube ich okay [___] können sie mir… |
| 118 | Löschung | `können` | `*(nicht da)*` | …ich okay gut [___] sie mir noch… |
| 119 | Löschung | `sie` | `*(nicht da)*` | …okay gut können [___] mir noch den… |
| 120 | Löschung | `mir` | `*(nicht da)*` | …gut können sie [___] noch den namen… |
| 121 | Löschung | `noch` | `*(nicht da)*` | …können sie mir [___] den namen ihres… |
| 122 | Löschung | `den` | `*(nicht da)*` | …sie mir noch [___] namen ihres hausarztes… |
| 123 | Löschung | `namen` | `*(nicht da)*` | …mir noch den [___] ihres hausarztes verraten… |
| 124 | Löschung | `ihres` | `*(nicht da)*` | …noch den namen [___] hausarztes verraten ja… |
| 125 | Löschung | `hausarztes` | `*(nicht da)*` | …den namen ihres [___] verraten ja das… |
| 126 | Löschung | `verraten` | `*(nicht da)*` | …namen ihres hausarztes [___] ja das ist… |
| 127 | Löschung | `ja` | `*(nicht da)*` | …ihres hausarztes verraten [___] das ist der… |
| 128 | Löschung | `ist` | `*(nicht da)*` | …verraten ja das [___] der herr dr… |
| 129 | Löschung | `der` | `*(nicht da)*` | …ja das ist [___] herr dr becker… |
| 130 | Löschung | `herr` | `*(nicht da)*` | …das ist der [___] dr becker der… |
| 131 | Löschung | `dr` | `*(nicht da)*` | …ist der herr [___] becker der herr… |
| 132 | Löschung | `becker` | `*(nicht da)*` | …der herr dr [___] der herr dr… |
| 133 | Löschung | `der` | `*(nicht da)*` | …herr dr becker [___] herr dr becker… |
| 134 | Löschung | `herr` | `*(nicht da)*` | …dr becker der [___] dr becker wie… |
| 135 | Löschung | `dr` | `*(nicht da)*` | …becker der herr [___] becker wie der… |
| 136 | Löschung | `becker` | `*(nicht da)*` | …der herr dr [___] wie der beruf… |
| 137 | Löschung | `wie` | `*(nicht da)*` | …herr dr becker [___] der beruf oder… |
| 138 | Löschung | `der` | `*(nicht da)*` | …dr becker wie [___] beruf oder mit… |
| 139 | Löschung | `beruf` | `*(nicht da)*` | …becker wie der [___] oder mit e… |
| 140 | Löschung | `oder` | `*(nicht da)*` | …wie der beruf [___] mit e mit… |
| 141 | Löschung | `mit` | `*(nicht da)*` | …der beruf oder [___] e mit e… |
| 142 | Löschung | `e` | `*(nicht da)*` | …beruf oder mit [___] mit e mit… |
| 143 | Löschung | `mit` | `*(nicht da)*` | …oder mit e [___] e mit e… |
| 144 | Löschung | `e` | `*(nicht da)*` | …mit e mit [___] mit e alles… |
| 145 | Löschung | `mit` | `*(nicht da)*` | …e mit e [___] e alles klar… |
| 146 | Löschung | `e` | `*(nicht da)*` | …mit e mit [___] alles klar gut… |
| 147 | Löschung | `alles` | `*(nicht da)*` | …e mit e [___] klar gut frau… |
| 148 | Löschung | `klar` | `*(nicht da)*` | …mit e alles [___] gut frau becken… |
| 149 | Löschung | `gut` | `*(nicht da)*` | …e alles klar [___] frau becken westfalen… |
| 150 | Löschung | `frau` | `*(nicht da)*` | …alles klar gut [___] becken westfalen sie… |
| 151 | Löschung | `becken` | `*(nicht da)*` | …klar gut frau [___] westfalen sie wurden… |
| 152 | Löschung | `westfalen` | `*(nicht da)*` | …gut frau becken [___] sie wurden ja… |
| 153 | Löschung | `sie` | `*(nicht da)*` | …frau becken westfalen [___] wurden ja soeben… |
| 154 | Löschung | `wurden` | `*(nicht da)*` | …becken westfalen sie [___] ja soeben mit… |
| 155 | Löschung | `ja` | `*(nicht da)*` | …westfalen sie wurden [___] soeben mit dem… |
| 156 | Löschung | `soeben` | `*(nicht da)*` | …sie wurden ja [___] mit dem rettungswagen… |
| 157 | Löschung | `mit` | `*(nicht da)*` | …wurden ja soeben [___] dem rettungswagen zu… |
| 158 | Löschung | `dem` | `*(nicht da)*` | …ja soeben mit [___] rettungswagen zu uns… |
| 159 | Löschung | `rettungswagen` | `*(nicht da)*` | …soeben mit dem [___] zu uns gebracht… |
| 160 | Löschung | `zu` | `*(nicht da)*` | …mit dem rettungswagen [___] uns gebracht was… |
| 161 | Löschung | `uns` | `*(nicht da)*` | …dem rettungswagen zu [___] gebracht was ist… |
| 162 | Löschung | `gebracht` | `*(nicht da)*` | …rettungswagen zu uns [___] was ist denn… |
| 163 | Löschung | `was` | `*(nicht da)*` | …zu uns gebracht [___] ist denn passiert… |
| 164 | Löschung | `ist` | `*(nicht da)*` | …uns gebracht was [___] denn passiert ja… |
| 165 | Löschung | `denn` | `*(nicht da)*` | …gebracht was ist [___] passiert ja ich… |
| 166 | Löschung | `passiert` | `*(nicht da)*` | …was ist denn [___] ja ich bin… |
| 167 | Löschung | `ja` | `*(nicht da)*` | …ist denn passiert [___] ich bin unvorsichtig… |
| 168 | Löschung | `ich` | `*(nicht da)*` | …denn passiert ja [___] bin unvorsichtig mit… |
| 169 | Löschung | `bin` | `*(nicht da)*` | …passiert ja ich [___] unvorsichtig mit meinem… |
| 170 | Löschung | `unvorsichtig` | `*(nicht da)*` | …ja ich bin [___] mit meinem fahrrad… |
| 171 | Löschung | `mit` | `*(nicht da)*` | …ich bin unvorsichtig [___] meinem fahrrad nach… |
| 172 | Löschung | `meinem` | `*(nicht da)*` | …bin unvorsichtig mit [___] fahrrad nach hause… |
| 173 | Löschung | `fahrrad` | `*(nicht da)*` | …unvorsichtig mit meinem [___] nach hause gefahren… |
| 174 | Löschung | `nach` | `*(nicht da)*` | …mit meinem fahrrad [___] hause gefahren von… |
| 175 | Löschung | `hause` | `*(nicht da)*` | …meinem fahrrad nach [___] gefahren von der… |
| 176 | Löschung | `gefahren` | `*(nicht da)*` | …fahrrad nach hause [___] von der arbeit… |
| 177 | Löschung | `von` | `*(nicht da)*` | …nach hause gefahren [___] der arbeit und… |
| 178 | Löschung | `der` | `*(nicht da)*` | …hause gefahren von [___] arbeit und hatte… |
| 179 | Löschung | `arbeit` | `*(nicht da)*` | …gefahren von der [___] und hatte leider… |
| 180 | Löschung | `und` | `*(nicht da)*` | …von der arbeit [___] hatte leider einen… |
| 181 | Löschung | `hatte` | `*(nicht da)*` | …der arbeit und [___] leider einen unfall… |
| 182 | Löschung | `leider` | `*(nicht da)*` | …arbeit und hatte [___] einen unfall okay… |
| 183 | Löschung | `einen` | `*(nicht da)*` | …und hatte leider [___] unfall okay dabei… |
| 184 | Löschung | `unfall` | `*(nicht da)*` | …hatte leider einen [___] okay dabei habe… |
| 185 | Löschung | `okay` | `*(nicht da)*` | …leider einen unfall [___] dabei habe ich… |
| 186 | Löschung | `dabei` | `*(nicht da)*` | …einen unfall okay [___] habe ich mich… |
| 187 | Löschung | `habe` | `*(nicht da)*` | …unfall okay dabei [___] ich mich verletzt… |
| 188 | Löschung | `ich` | `*(nicht da)*` | …okay dabei habe [___] mich verletzt den… |
| 189 | Löschung | `mich` | `*(nicht da)*` | …dabei habe ich [___] verletzt den krankenwagen… |
| 190 | Löschung | `verletzt` | `*(nicht da)*` | …habe ich mich [___] den krankenwagen gerufen… |
| 191 | Löschung | `den` | `*(nicht da)*` | …ich mich verletzt [___] krankenwagen gerufen und… |
| 192 | Löschung | `krankenwagen` | `*(nicht da)*` | …mich verletzt den [___] gerufen und da… |
| 193 | Löschung | `gerufen` | `*(nicht da)*` | …verletzt den krankenwagen [___] und da bin… |
| 194 | Löschung | `und` | `*(nicht da)*` | …den krankenwagen gerufen [___] da bin ich… |
| 195 | Löschung | `da` | `*(nicht da)*` | …krankenwagen gerufen und [___] bin ich jetzt… |
| 196 | Löschung | `bin` | `*(nicht da)*` | …gerufen und da [___] ich jetzt da… |
| 197 | Löschung | `ich` | `*(nicht da)*` | …und da bin [___] jetzt da sind… |
| 198 | Löschung | `jetzt` | `*(nicht da)*` | …da bin ich [___] da sind sie… |
| 199 | Löschung | `da` | `*(nicht da)*` | …bin ich jetzt [___] sind sie jetzt… |
| 200 | Löschung | `sind` | `*(nicht da)*` | …ich jetzt da [___] sie jetzt was… |
| 201 | Löschung | `sie` | `*(nicht da)*` | …jetzt da sind [___] jetzt was haben… |
| 202 | Löschung | `jetzt` | `*(nicht da)*` | …da sind sie [___] was haben sie… |
| 203 | Löschung | `was` | `*(nicht da)*` | …sind sie jetzt [___] haben sie denn… |
| 204 | Löschung | `haben` | `*(nicht da)*` | …sie jetzt was [___] sie denn jetzt… |
| 205 | Löschung | `sie` | `*(nicht da)*` | …jetzt was haben [___] denn jetzt für… |
| 206 | Löschung | `denn` | `*(nicht da)*` | …was haben sie [___] jetzt für beschwerden… |
| 207 | Löschung | `jetzt` | `*(nicht da)*` | …haben sie denn [___] für beschwerden beschwerden… |
| 208 | Löschung | `für` | `*(nicht da)*` | …sie denn jetzt [___] beschwerden beschwerden entschuldigung… |
| 209 | Löschung | `beschwerden` | `*(nicht da)*` | …denn jetzt für [___] beschwerden entschuldigung haben… |
| 210 | Löschung | `beschwerden` | `*(nicht da)*` | …jetzt für beschwerden [___] entschuldigung haben sie… |
| 211 | Löschung | `entschuldigung` | `*(nicht da)*` | …für beschwerden beschwerden [___] haben sie schmerzen… |
| 212 | Löschung | `haben` | `*(nicht da)*` | …beschwerden beschwerden entschuldigung [___] sie schmerzen am… |
| 213 | Löschung | `sie` | `*(nicht da)*` | …beschwerden entschuldigung haben [___] schmerzen am kopf… |
| 214 | Löschung | `schmerzen` | `*(nicht da)*` | …entschuldigung haben sie [___] am kopf im… |
| 215 | Löschung | `am` | `*(nicht da)*` | …haben sie schmerzen [___] kopf im oberkörper… |
| 216 | Löschung | `kopf` | `*(nicht da)*` | …sie schmerzen am [___] im oberkörper an… |
| 217 | Löschung | `im` | `*(nicht da)*` | …schmerzen am kopf [___] oberkörper an den… |
| 218 | Löschung | `oberkörper` | `*(nicht da)*` | …am kopf im [___] an den beinen… |
| 219 | Löschung | `an` | `*(nicht da)*` | …kopf im oberkörper [___] den beinen ja… |
| 220 | Löschung | `den` | `*(nicht da)*` | …im oberkörper an [___] beinen ja ich… |
| 221 | Löschung | `beinen` | `*(nicht da)*` | …oberkörper an den [___] ja ich bin… |
| 222 | Löschung | `ja` | `*(nicht da)*` | …an den beinen [___] ich bin auf… |
| 223 | Löschung | `ich` | `*(nicht da)*` | …den beinen ja [___] bin auf meine… |
| 224 | Löschung | `bin` | `*(nicht da)*` | …beinen ja ich [___] auf meine linke… |
| 225 | Löschung | `auf` | `*(nicht da)*` | …ja ich bin [___] meine linke seite… |
| 226 | Löschung | `meine` | `*(nicht da)*` | …ich bin auf [___] linke seite gefallen… |
| 227 | Löschung | `linke` | `*(nicht da)*` | …bin auf meine [___] seite gefallen und… |
| 228 | Löschung | `seite` | `*(nicht da)*` | …auf meine linke [___] gefallen und habe… |
| 229 | Löschung | `gefallen` | `*(nicht da)*` | …meine linke seite [___] und habe mir… |
| 230 | Löschung | `und` | `*(nicht da)*` | …linke seite gefallen [___] habe mir dabei… |
| 231 | Löschung | `habe` | `*(nicht da)*` | …seite gefallen und [___] mir dabei auch… |
| 232 | Löschung | `mir` | `*(nicht da)*` | …gefallen und habe [___] dabei auch tatsächlich… |
| 233 | Löschung | `dabei` | `*(nicht da)*` | …und habe mir [___] auch tatsächlich den… |
| 234 | Löschung | `auch` | `*(nicht da)*` | …habe mir dabei [___] tatsächlich den kopf… |
| 235 | Löschung | `tatsächlich` | `*(nicht da)*` | …mir dabei auch [___] den kopf leicht… |
| 236 | Löschung | `den` | `*(nicht da)*` | …dabei auch tatsächlich [___] kopf leicht gestoßen… |
| 237 | Löschung | `kopf` | `*(nicht da)*` | …auch tatsächlich den [___] leicht gestoßen ich… |
| 238 | Löschung | `leicht` | `*(nicht da)*` | …tatsächlich den kopf [___] gestoßen ich habe… |
| 239 | Löschung | `gestoßen` | `*(nicht da)*` | …den kopf leicht [___] ich habe leichte… |
| 240 | Löschung | `ich` | `*(nicht da)*` | …kopf leicht gestoßen [___] habe leichte schmerzen… |
| 241 | Löschung | `habe` | `*(nicht da)*` | …leicht gestoßen ich [___] leichte schmerzen am… |
| 242 | Löschung | `leichte` | `*(nicht da)*` | …gestoßen ich habe [___] schmerzen am hinterkopf… |
| 243 | Löschung | `schmerzen` | `*(nicht da)*` | …ich habe leichte [___] am hinterkopf auf… |
| 244 | Löschung | `am` | `*(nicht da)*` | …habe leichte schmerzen [___] hinterkopf auf der… |
| 245 | Substitution | `hinterkopf` | `gespräch` | …leichte schmerzen am [___] auf der linken… |
| 246 | Substitution | `auf` | `zwischen` | …schmerzen am hinterkopf [___] der linken seite… |
| 247 | Löschung | `linken` | `*(nicht da)*` | …hinterkopf auf der [___] seite ich kann… |
| 248 | Löschung | `seite` | `*(nicht da)*` | …auf der linken [___] ich kann außerdem… |
| 249 | Löschung | `ich` | `*(nicht da)*` | …der linken seite [___] kann außerdem meinen… |
| 250 | Löschung | `kann` | `*(nicht da)*` | …linken seite ich [___] außerdem meinen linken… |
| 251 | Löschung | `außerdem` | `*(nicht da)*` | …seite ich kann [___] meinen linken daumen… |
| 252 | Löschung | `meinen` | `*(nicht da)*` | …ich kann außerdem [___] linken daumen überhaupt… |
| 253 | Löschung | `linken` | `*(nicht da)*` | …kann außerdem meinen [___] daumen überhaupt nicht… |
| 254 | Löschung | `daumen` | `*(nicht da)*` | …außerdem meinen linken [___] überhaupt nicht bewegen… |
| 255 | Löschung | `überhaupt` | `*(nicht da)*` | …meinen linken daumen [___] nicht bewegen weil… |
| 256 | Löschung | `nicht` | `*(nicht da)*` | …linken daumen überhaupt [___] bewegen weil ich… |
| 257 | Löschung | `bewegen` | `*(nicht da)*` | …daumen überhaupt nicht [___] weil ich wirklich… |
| 258 | Löschung | `weil` | `*(nicht da)*` | …überhaupt nicht bewegen [___] ich wirklich starke… |
| 259 | Löschung | `ich` | `*(nicht da)*` | …nicht bewegen weil [___] wirklich starke schmerzen… |
| 260 | Löschung | `wirklich` | `*(nicht da)*` | …bewegen weil ich [___] starke schmerzen habe… |
| 261 | Löschung | `starke` | `*(nicht da)*` | …weil ich wirklich [___] schmerzen habe am… |
| 262 | Löschung | `schmerzen` | `*(nicht da)*` | …ich wirklich starke [___] habe am daumen… |
| 263 | Löschung | `habe` | `*(nicht da)*` | …wirklich starke schmerzen [___] am daumen und… |
| 264 | Löschung | `am` | `*(nicht da)*` | …starke schmerzen habe [___] daumen und er… |
| 265 | Löschung | `daumen` | `*(nicht da)*` | …schmerzen habe am [___] und er ist… |
| 266 | Löschung | `und` | `*(nicht da)*` | …habe am daumen [___] er ist auch… |
| 267 | Löschung | `er` | `*(nicht da)*` | …am daumen und [___] ist auch etwas… |
| 268 | Löschung | `ist` | `*(nicht da)*` | …daumen und er [___] auch etwas geschwollen… |
| 269 | Löschung | `auch` | `*(nicht da)*` | …und er ist [___] etwas geschwollen und… |
| 270 | Löschung | `etwas` | `*(nicht da)*` | …er ist auch [___] geschwollen und irgendwie… |
| 271 | Löschung | `geschwollen` | `*(nicht da)*` | …ist auch etwas [___] und irgendwie habe… |
| 272 | Löschung | `und` | `*(nicht da)*` | …auch etwas geschwollen [___] irgendwie habe ich… |
| 273 | Löschung | `irgendwie` | `*(nicht da)*` | …etwas geschwollen und [___] habe ich auch… |
| 274 | Löschung | `habe` | `*(nicht da)*` | …geschwollen und irgendwie [___] ich auch mein… |
| 275 | Löschung | `ich` | `*(nicht da)*` | …und irgendwie habe [___] auch mein knie… |
| 276 | Löschung | `auch` | `*(nicht da)*` | …irgendwie habe ich [___] mein knie richtig… |
| 277 | Löschung | `mein` | `*(nicht da)*` | …habe ich auch [___] knie richtig stark… |
| 278 | Löschung | `knie` | `*(nicht da)*` | …ich auch mein [___] richtig stark verletzt… |
| 279 | Löschung | `richtig` | `*(nicht da)*` | …auch mein knie [___] stark verletzt weil… |
| 280 | Löschung | `stark` | `*(nicht da)*` | …mein knie richtig [___] verletzt weil es… |
| 281 | Löschung | `verletzt` | `*(nicht da)*` | …knie richtig stark [___] weil es richtig… |
| 282 | Löschung | `weil` | `*(nicht da)*` | …richtig stark verletzt [___] es richtig geschwollen… |
| 283 | Löschung | `es` | `*(nicht da)*` | …stark verletzt weil [___] richtig geschwollen ist… |
| 284 | Löschung | `richtig` | `*(nicht da)*` | …verletzt weil es [___] geschwollen ist und… |
| 285 | Löschung | `geschwollen` | `*(nicht da)*` | …weil es richtig [___] ist und auch… |
| 286 | Löschung | `ist` | `*(nicht da)*` | …es richtig geschwollen [___] und auch sehr… |
| 287 | Löschung | `und` | `*(nicht da)*` | …richtig geschwollen ist [___] auch sehr weh… |
| 288 | Löschung | `auch` | `*(nicht da)*` | …geschwollen ist und [___] sehr weh tut… |
| 289 | Löschung | `sehr` | `*(nicht da)*` | …ist und auch [___] weh tut okay… |
| 290 | Löschung | `weh` | `*(nicht da)*` | …und auch sehr [___] tut okay knie… |
| 291 | Löschung | `tut` | `*(nicht da)*` | …auch sehr weh [___] okay knie ist… |
| 292 | Löschung | `okay` | `*(nicht da)*` | …sehr weh tut [___] knie ist auch… |
| 293 | Löschung | `knie` | `*(nicht da)*` | …weh tut okay [___] ist auch geschwollen… |
| 294 | Löschung | `ist` | `*(nicht da)*` | …tut okay knie [___] auch geschwollen und… |
| 295 | Löschung | `auch` | `*(nicht da)*` | …okay knie ist [___] geschwollen und starke… |
| 296 | Löschung | `geschwollen` | `*(nicht da)*` | …knie ist auch [___] und starke schmerzen… |
| 297 | Löschung | `und` | `*(nicht da)*` | …ist auch geschwollen [___] starke schmerzen sagen… |
| 298 | Löschung | `starke` | `*(nicht da)*` | …auch geschwollen und [___] schmerzen sagen sie… |
| 299 | Löschung | `schmerzen` | `*(nicht da)*` | …geschwollen und starke [___] sagen sie genau… |
| 300 | Löschung | `sagen` | `*(nicht da)*` | …und starke schmerzen [___] sie genau frau… |
| 301 | Löschung | `sie` | `*(nicht da)*` | …starke schmerzen sagen [___] genau frau becken… |
| 302 | Löschung | `genau` | `*(nicht da)*` | …schmerzen sagen sie [___] frau becken westfalen… |
| 303 | Löschung | `frau` | `*(nicht da)*` | …sagen sie genau [___] becken westfalen haben… |
| 304 | Löschung | `becken` | `*(nicht da)*` | …sie genau frau [___] westfalen haben sie… |
| 305 | Löschung | `westfalen` | `*(nicht da)*` | …genau frau becken [___] haben sie denn… |
| 306 | Löschung | `haben` | `*(nicht da)*` | …frau becken westfalen [___] sie denn einen… |
| 307 | Löschung | `sie` | `*(nicht da)*` | …becken westfalen haben [___] denn einen fahrradhelm… |
| 308 | Löschung | `denn` | `*(nicht da)*` | …westfalen haben sie [___] einen fahrradhelm getragen… |
| 309 | Löschung | `einen` | `*(nicht da)*` | …haben sie denn [___] fahrradhelm getragen leider… |
| 310 | Löschung | `fahrradhelm` | `*(nicht da)*` | …sie denn einen [___] getragen leider nein… |
| 311 | Löschung | `getragen` | `*(nicht da)*` | …denn einen fahrradhelm [___] leider nein ich… |
| 312 | Löschung | `leider` | `*(nicht da)*` | …einen fahrradhelm getragen [___] nein ich muss… |
| 313 | Löschung | `nein` | `*(nicht da)*` | …fahrradhelm getragen leider [___] ich muss auch… |
| 314 | Löschung | `ich` | `*(nicht da)*` | …getragen leider nein [___] muss auch zugeben… |
| 315 | Löschung | `muss` | `*(nicht da)*` | …leider nein ich [___] auch zugeben dass… |
| 316 | Löschung | `auch` | `*(nicht da)*` | …nein ich muss [___] zugeben dass ich… |
| 317 | Löschung | `zugeben` | `*(nicht da)*` | …ich muss auch [___] dass ich sehr… |
| 318 | Löschung | `dass` | `*(nicht da)*` | …muss auch zugeben [___] ich sehr ungern… |
| 319 | Löschung | `ich` | `*(nicht da)*` | …auch zugeben dass [___] sehr ungern einen… |
| 320 | Löschung | `sehr` | `*(nicht da)*` | …zugeben dass ich [___] ungern einen fahrradhelm… |
| 321 | Löschung | `ungern` | `*(nicht da)*` | …dass ich sehr [___] einen fahrradhelm trage… |
| 322 | Löschung | `einen` | `*(nicht da)*` | …ich sehr ungern [___] fahrradhelm trage weil… |
| 323 | Löschung | `fahrradhelm` | `*(nicht da)*` | …sehr ungern einen [___] trage weil sie… |
| 324 | Löschung | `trage` | `*(nicht da)*` | …ungern einen fahrradhelm [___] weil sie mir… |
| 325 | Löschung | `weil` | `*(nicht da)*` | …einen fahrradhelm trage [___] sie mir so… |
| 326 | Löschung | `sie` | `*(nicht da)*` | …fahrradhelm trage weil [___] mir so unbequem… |
| 327 | Löschung | `mir` | `*(nicht da)*` | …trage weil sie [___] so unbequem sind… |
| 328 | Löschung | `so` | `*(nicht da)*` | …weil sie mir [___] unbequem sind und… |
| 329 | Löschung | `unbequem` | `*(nicht da)*` | …sie mir so [___] sind und es… |
| 330 | Löschung | `sind` | `*(nicht da)*` | …mir so unbequem [___] und es sieht… |
| 331 | Löschung | `und` | `*(nicht da)*` | …so unbequem sind [___] es sieht auch… |
| 332 | Löschung | `es` | `*(nicht da)*` | …unbequem sind und [___] sieht auch so… |
| 333 | Löschung | `sieht` | `*(nicht da)*` | …sind und es [___] auch so bescheuert… |
| 334 | Löschung | `auch` | `*(nicht da)*` | …und es sieht [___] so bescheuert aus… |
| 335 | Löschung | `so` | `*(nicht da)*` | …es sieht auch [___] bescheuert aus sie… |
| 336 | Löschung | `bescheuert` | `*(nicht da)*` | …sieht auch so [___] aus sie als… |
| 337 | Löschung | `aus` | `*(nicht da)*` | …auch so bescheuert [___] sie als frau… |
| 338 | Löschung | `sie` | `*(nicht da)*` | …so bescheuert aus [___] als frau würden… |
| 339 | Löschung | `als` | `*(nicht da)*` | …bescheuert aus sie [___] frau würden mich… |
| 340 | Löschung | `frau` | `*(nicht da)*` | …aus sie als [___] würden mich sicherlich… |
| 341 | Löschung | `würden` | `*(nicht da)*` | …sie als frau [___] mich sicherlich verstehen… |
| 342 | Löschung | `mich` | `*(nicht da)*` | …als frau würden [___] sicherlich verstehen ich… |
| 343 | Löschung | `sicherlich` | `*(nicht da)*` | …frau würden mich [___] verstehen ich verstehe… |
| 344 | Löschung | `verstehen` | `*(nicht da)*` | …würden mich sicherlich [___] ich verstehe sie… |
| 345 | Löschung | `ich` | `*(nicht da)*` | …mich sicherlich verstehen [___] verstehe sie voll… |
| 346 | Löschung | `verstehe` | `*(nicht da)*` | …sicherlich verstehen ich [___] sie voll und… |
| 347 | Löschung | `sie` | `*(nicht da)*` | …verstehen ich verstehe [___] voll und ganz… |
| 348 | Löschung | `voll` | `*(nicht da)*` | …ich verstehe sie [___] und ganz meiner… |
| 349 | Löschung | `und` | `*(nicht da)*` | …verstehe sie voll [___] ganz meiner frisur… |
| 350 | Löschung | `ganz` | `*(nicht da)*` | …sie voll und [___] meiner frisur tut… |
| 351 | Löschung | `meiner` | `*(nicht da)*` | …voll und ganz [___] frisur tut das… |
| 352 | Löschung | `frisur` | `*(nicht da)*` | …und ganz meiner [___] tut das auch… |
| 353 | Löschung | `tut` | `*(nicht da)*` | …ganz meiner frisur [___] das auch nicht… |
| 354 | Löschung | `das` | `*(nicht da)*` | …meiner frisur tut [___] auch nicht gut… |
| 355 | Löschung | `auch` | `*(nicht da)*` | …frisur tut das [___] nicht gut aber… |
| 356 | Löschung | `nicht` | `*(nicht da)*` | …tut das auch [___] gut aber da… |
| 357 | Löschung | `gut` | `*(nicht da)*` | …das auch nicht [___] aber da muss… |
| 358 | Löschung | `aber` | `*(nicht da)*` | …auch nicht gut [___] da muss ich… |
| 359 | Löschung | `da` | `*(nicht da)*` | …nicht gut aber [___] muss ich ihnen… |
| 360 | Löschung | `muss` | `*(nicht da)*` | …gut aber da [___] ich ihnen leider… |
| 361 | Löschung | `ich` | `*(nicht da)*` | …aber da muss [___] ihnen leider sagen… |
| 362 | Löschung | `ihnen` | `*(nicht da)*` | …da muss ich [___] leider sagen in… |
| 363 | Löschung | `leider` | `*(nicht da)*` | …muss ich ihnen [___] sagen in diesem… |
| 364 | Löschung | `sagen` | `*(nicht da)*` | …ich ihnen leider [___] in diesem fall… |
| 365 | Löschung | `in` | `*(nicht da)*` | …ihnen leider sagen [___] diesem fall gehen… |
| 366 | Löschung | `diesem` | `*(nicht da)*` | …leider sagen in [___] fall gehen sicherheit… |
| 367 | Löschung | `fall` | `*(nicht da)*` | …sagen in diesem [___] gehen sicherheit und… |
| 368 | Löschung | `gehen` | `*(nicht da)*` | …in diesem fall [___] sicherheit und gesundheit… |
| 369 | Substitution | `sicherheit` | `ärztin` | …diesem fall gehen [___] und gesundheit definitiv… |
| 370 | Löschung | `gesundheit` | `*(nicht da)*` | …gehen sicherheit und [___] definitiv vor aussehen… |
| 371 | Löschung | `definitiv` | `*(nicht da)*` | …sicherheit und gesundheit [___] vor aussehen frau… |
| 372 | Löschung | `vor` | `*(nicht da)*` | …und gesundheit definitiv [___] aussehen frau becken… |
| 373 | Löschung | `aussehen` | `*(nicht da)*` | …gesundheit definitiv vor [___] frau becken westfalen… |
| 374 | Löschung | `bitte` | `*(nicht da)*` | …frau becken westfalen [___] bitte tragen sie… |
| 375 | Löschung | `bitte` | `*(nicht da)*` | …becken westfalen bitte [___] tragen sie beim… |
| 376 | Löschung | `tragen` | `*(nicht da)*` | …westfalen bitte bitte [___] sie beim nächsten… |
| 377 | Löschung | `sie` | `*(nicht da)*` | …bitte bitte tragen [___] beim nächsten mal… |
| 378 | Löschung | `beim` | `*(nicht da)*` | …bitte tragen sie [___] nächsten mal einen… |
| 379 | Löschung | `nächsten` | `*(nicht da)*` | …tragen sie beim [___] mal einen helm… |
| 380 | Löschung | `mal` | `*(nicht da)*` | …sie beim nächsten [___] einen helm da… |
| 381 | Löschung | `einen` | `*(nicht da)*` | …beim nächsten mal [___] helm da haben… |
| 382 | Löschung | `helm` | `*(nicht da)*` | …nächsten mal einen [___] da haben sie… |
| 383 | Löschung | `da` | `*(nicht da)*` | …mal einen helm [___] haben sie diesmal… |
| 384 | Löschung | `haben` | `*(nicht da)*` | …einen helm da [___] sie diesmal wirklich… |
| 385 | Löschung | `sie` | `*(nicht da)*` | …helm da haben [___] diesmal wirklich noch… |
| 386 | Löschung | `diesmal` | `*(nicht da)*` | …da haben sie [___] wirklich noch glück… |
| 387 | Löschung | `wirklich` | `*(nicht da)*` | …haben sie diesmal [___] noch glück gehabt… |
| 388 | Löschung | `noch` | `*(nicht da)*` | …sie diesmal wirklich [___] glück gehabt dass… |
| 389 | Löschung | `glück` | `*(nicht da)*` | …diesmal wirklich noch [___] gehabt dass nichts… |
| 390 | Löschung | `gehabt` | `*(nicht da)*` | …wirklich noch glück [___] dass nichts passiert… |
| 391 | Löschung | `dass` | `*(nicht da)*` | …noch glück gehabt [___] nichts passiert ist… |
| 392 | Löschung | `nichts` | `*(nicht da)*` | …glück gehabt dass [___] passiert ist da… |
| 393 | Löschung | `passiert` | `*(nicht da)*` | …gehabt dass nichts [___] ist da haben… |
| 394 | Löschung | `ist` | `*(nicht da)*` | …dass nichts passiert [___] da haben sie… |
| 395 | Löschung | `da` | `*(nicht da)*` | …nichts passiert ist [___] haben sie auf… |
| 396 | Löschung | `haben` | `*(nicht da)*` | …passiert ist da [___] sie auf jeden… |
| 397 | Löschung | `sie` | `*(nicht da)*` | …ist da haben [___] auf jeden fall… |
| 398 | Löschung | `auf` | `*(nicht da)*` | …da haben sie [___] jeden fall recht… |
| 399 | Löschung | `jeden` | `*(nicht da)*` | …haben sie auf [___] fall recht ich… |
| 400 | Löschung | `fall` | `*(nicht da)*` | …sie auf jeden [___] recht ich habe… |
| 401 | Löschung | `recht` | `*(nicht da)*` | …auf jeden fall [___] ich habe jetzt… |
| 402 | Löschung | `ich` | `*(nicht da)*` | …jeden fall recht [___] habe jetzt daraus… |
| 403 | Löschung | `habe` | `*(nicht da)*` | …fall recht ich [___] jetzt daraus gelernt… |
| 404 | Löschung | `jetzt` | `*(nicht da)*` | …recht ich habe [___] daraus gelernt und… |
| 405 | Löschung | `daraus` | `*(nicht da)*` | …ich habe jetzt [___] gelernt und werde… |
| 406 | Löschung | `gelernt` | `*(nicht da)*` | …habe jetzt daraus [___] und werde mir… |
| 407 | Löschung | `und` | `*(nicht da)*` | …jetzt daraus gelernt [___] werde mir auch… |
| 408 | Löschung | `werde` | `*(nicht da)*` | …daraus gelernt und [___] mir auch einen… |
| 409 | Löschung | `mir` | `*(nicht da)*` | …gelernt und werde [___] auch einen besorgen… |
| 410 | Löschung | `auch` | `*(nicht da)*` | …und werde mir [___] einen besorgen okay… |
| 411 | Löschung | `einen` | `*(nicht da)*` | …werde mir auch [___] besorgen okay sehr… |
| 412 | Löschung | `besorgen` | `*(nicht da)*` | …mir auch einen [___] okay sehr gut… |
| 413 | Löschung | `okay` | `*(nicht da)*` | …auch einen besorgen [___] sehr gut sie… |
| 414 | Löschung | `sehr` | `*(nicht da)*` | …einen besorgen okay [___] gut sie hatten… |
| 415 | Löschung | `gut` | `*(nicht da)*` | …besorgen okay sehr [___] sie hatten gesagt… |
| 416 | Löschung | `sie` | `*(nicht da)*` | …okay sehr gut [___] hatten gesagt sie… |
| 417 | Löschung | `hatten` | `*(nicht da)*` | …sehr gut sie [___] gesagt sie haben… |
| 418 | Löschung | `gesagt` | `*(nicht da)*` | …gut sie hatten [___] sie haben hinten… |
| 419 | Löschung | `sie` | `*(nicht da)*` | …sie hatten gesagt [___] haben hinten auf… |
| 420 | Löschung | `haben` | `*(nicht da)*` | …hatten gesagt sie [___] hinten auf der… |
| 421 | Löschung | `hinten` | `*(nicht da)*` | …gesagt sie haben [___] auf der linken… |
| 422 | Löschung | `auf` | `*(nicht da)*` | …sie haben hinten [___] der linken seite… |
| 423 | Löschung | `der` | `*(nicht da)*` | …haben hinten auf [___] linken seite des… |
| 424 | Löschung | `linken` | `*(nicht da)*` | …hinten auf der [___] seite des hinterkopfes… |
| 425 | Löschung | `seite` | `*(nicht da)*` | …auf der linken [___] des hinterkopfes eine… |
| 426 | Löschung | `des` | `*(nicht da)*` | …der linken seite [___] hinterkopfes eine kleine… |
| 427 | Löschung | `hinterkopfes` | `*(nicht da)*` | …linken seite des [___] eine kleine beule… |
| 428 | Löschung | `eine` | `*(nicht da)*` | …seite des hinterkopfes [___] kleine beule richtig… |
| 429 | Löschung | `kleine` | `*(nicht da)*` | …des hinterkopfes eine [___] beule richtig genau… |
| 430 | Löschung | `beule` | `*(nicht da)*` | …hinterkopfes eine kleine [___] richtig genau das… |
| 431 | Löschung | `richtig` | `*(nicht da)*` | …eine kleine beule [___] genau das ist… |
| 432 | Löschung | `genau` | `*(nicht da)*` | …kleine beule richtig [___] das ist richtig… |
| 433 | Löschung | `das` | `*(nicht da)*` | …beule richtig genau [___] ist richtig ja… |
| 434 | Löschung | `ist` | `*(nicht da)*` | …richtig genau das [___] richtig ja haben… |
| 435 | Löschung | `richtig` | `*(nicht da)*` | …genau das ist [___] ja haben sie… |
| 436 | Löschung | `ja` | `*(nicht da)*` | …das ist richtig [___] haben sie irgendeine… |
| 437 | Löschung | `haben` | `*(nicht da)*` | …ist richtig ja [___] sie irgendeine blutige… |
| 438 | Löschung | `sie` | `*(nicht da)*` | …richtig ja haben [___] irgendeine blutige verletzung… |
| 439 | Löschung | `irgendeine` | `*(nicht da)*` | …ja haben sie [___] blutige verletzung am… |
| 440 | Löschung | `blutige` | `*(nicht da)*` | …haben sie irgendeine [___] verletzung am kopf… |
| 441 | Löschung | `verletzung` | `*(nicht da)*` | …sie irgendeine blutige [___] am kopf oder… |
| 442 | Löschung | `am` | `*(nicht da)*` | …irgendeine blutige verletzung [___] kopf oder ist… |
| 443 | Löschung | `kopf` | `*(nicht da)*` | …blutige verletzung am [___] oder ist das… |
| 444 | Löschung | `oder` | `*(nicht da)*` | …verletzung am kopf [___] ist das alles… |
| 445 | Löschung | `ist` | `*(nicht da)*` | …am kopf oder [___] das alles es… |
| 446 | Löschung | `das` | `*(nicht da)*` | …kopf oder ist [___] alles es ist… |
| 447 | Löschung | `alles` | `*(nicht da)*` | …oder ist das [___] es ist mir… |
| 448 | Löschung | `es` | `*(nicht da)*` | …ist das alles [___] ist mir nichts… |
| 449 | Löschung | `ist` | `*(nicht da)*` | …das alles es [___] mir nichts anderes… |
| 450 | Löschung | `mir` | `*(nicht da)*` | …alles es ist [___] nichts anderes aufgefallen… |
| 451 | Löschung | `nichts` | `*(nicht da)*` | …es ist mir [___] anderes aufgefallen zum… |
| 452 | Löschung | `anderes` | `*(nicht da)*` | …ist mir nichts [___] aufgefallen zum glück… |
| 453 | Löschung | `aufgefallen` | `*(nicht da)*` | …mir nichts anderes [___] zum glück ist… |
| 454 | Löschung | `zum` | `*(nicht da)*` | …nichts anderes aufgefallen [___] glück ist es… |
| 455 | Löschung | `glück` | `*(nicht da)*` | …anderes aufgefallen zum [___] ist es glaube… |
| 456 | Löschung | `ist` | `*(nicht da)*` | …aufgefallen zum glück [___] es glaube ich… |
| 457 | Löschung | `es` | `*(nicht da)*` | …zum glück ist [___] glaube ich nur… |
| 458 | Löschung | `glaube` | `*(nicht da)*` | …glück ist es [___] ich nur die… |
| 459 | Löschung | `ich` | `*(nicht da)*` | …ist es glaube [___] nur die beule… |
| 460 | Löschung | `nur` | `*(nicht da)*` | …es glaube ich [___] die beule okay… |
| 461 | Löschung | `die` | `*(nicht da)*` | …glaube ich nur [___] beule okay sehr… |
| 462 | Löschung | `beule` | `*(nicht da)*` | …ich nur die [___] okay sehr gut… |
| 463 | Löschung | `okay` | `*(nicht da)*` | …nur die beule [___] sehr gut die… |
| 464 | Löschung | `sehr` | `*(nicht da)*` | …die beule okay [___] gut die schmerzen… |
| 465 | Löschung | `gut` | `*(nicht da)*` | …beule okay sehr [___] die schmerzen sind… |
| 466 | Löschung | `die` | `*(nicht da)*` | …okay sehr gut [___] schmerzen sind die… |
| 467 | Löschung | `schmerzen` | `*(nicht da)*` | …sehr gut die [___] sind die stark… |
| 468 | Löschung | `sind` | `*(nicht da)*` | …gut die schmerzen [___] die stark oder… |
| 469 | Löschung | `die` | `*(nicht da)*` | …die schmerzen sind [___] stark oder geht… |
| 470 | Löschung | `stark` | `*(nicht da)*` | …schmerzen sind die [___] oder geht es… |
| 471 | Löschung | `oder` | `*(nicht da)*` | …sind die stark [___] geht es die… |
| 472 | Löschung | `geht` | `*(nicht da)*` | …die stark oder [___] es die sind… |
| 473 | Löschung | `es` | `*(nicht da)*` | …stark oder geht [___] die sind nicht… |
| 474 | Löschung | `die` | `*(nicht da)*` | …oder geht es [___] sind nicht so… |
| 475 | Löschung | `sind` | `*(nicht da)*` | …geht es die [___] nicht so stark… |
| 476 | Löschung | `nicht` | `*(nicht da)*` | …es die sind [___] so stark das… |
| 477 | Löschung | `so` | `*(nicht da)*` | …die sind nicht [___] stark das geht… |
| 478 | Löschung | `stark` | `*(nicht da)*` | …sind nicht so [___] das geht tatsächlich… |
| 479 | Löschung | `das` | `*(nicht da)*` | …nicht so stark [___] geht tatsächlich am… |
| 480 | Löschung | `geht` | `*(nicht da)*` | …so stark das [___] tatsächlich am kopf… |
| 481 | Löschung | `tatsächlich` | `*(nicht da)*` | …stark das geht [___] am kopf sind… |
| 482 | Löschung | `am` | `*(nicht da)*` | …das geht tatsächlich [___] kopf sind die… |
| 483 | Löschung | `kopf` | `*(nicht da)*` | …geht tatsächlich am [___] sind die am… |
| 484 | Löschung | `sind` | `*(nicht da)*` | …tatsächlich am kopf [___] die am schwächsten… |
| 485 | Löschung | `die` | `*(nicht da)*` | …am kopf sind [___] am schwächsten okay… |
| 486 | Löschung | `am` | `*(nicht da)*` | …kopf sind die [___] schwächsten okay alles… |
| 487 | Löschung | `schwächsten` | `*(nicht da)*` | …sind die am [___] okay alles klar… |
| 488 | Löschung | `okay` | `*(nicht da)*` | …die am schwächsten [___] alles klar der… |
| 489 | Löschung | `alles` | `*(nicht da)*` | …am schwächsten okay [___] klar der daumen… |
| 490 | Löschung | `klar` | `*(nicht da)*` | …schwächsten okay alles [___] der daumen sie… |
| 491 | Löschung | `der` | `*(nicht da)*` | …okay alles klar [___] daumen sie haben… |
| 492 | Löschung | `daumen` | `*(nicht da)*` | …alles klar der [___] sie haben jetzt… |
| 493 | Löschung | `sie` | `*(nicht da)*` | …klar der daumen [___] haben jetzt gesagt… |
| 494 | Löschung | `haben` | `*(nicht da)*` | …der daumen sie [___] jetzt gesagt sie… |
| 495 | Löschung | `jetzt` | `*(nicht da)*` | …daumen sie haben [___] gesagt sie können… |
| 496 | Löschung | `gesagt` | `*(nicht da)*` | …sie haben jetzt [___] sie können den… |
| 497 | Löschung | `sie` | `*(nicht da)*` | …haben jetzt gesagt [___] können den daumen… |
| 498 | Löschung | `können` | `*(nicht da)*` | …jetzt gesagt sie [___] den daumen gar… |
| 499 | Löschung | `den` | `*(nicht da)*` | …gesagt sie können [___] daumen gar nicht… |
| 500 | Löschung | `daumen` | `*(nicht da)*` | …sie können den [___] gar nicht mehr… |
| 501 | Löschung | `gar` | `*(nicht da)*` | …können den daumen [___] nicht mehr recht… |
| 502 | Löschung | `nicht` | `*(nicht da)*` | …den daumen gar [___] mehr recht bewegen… |
| 503 | Löschung | `mehr` | `*(nicht da)*` | …daumen gar nicht [___] recht bewegen wenn… |
| 504 | Löschung | `recht` | `*(nicht da)*` | …gar nicht mehr [___] bewegen wenn wir… |
| 505 | Löschung | `bewegen` | `*(nicht da)*` | …nicht mehr recht [___] wenn wir jetzt… |
| 506 | Löschung | `wenn` | `*(nicht da)*` | …mehr recht bewegen [___] wir jetzt die… |
| 507 | Löschung | `wir` | `*(nicht da)*` | …recht bewegen wenn [___] jetzt die schmerzen… |
| 508 | Löschung | `jetzt` | `*(nicht da)*` | …bewegen wenn wir [___] die schmerzen einschätzen… |
| 509 | Löschung | `die` | `*(nicht da)*` | …wenn wir jetzt [___] schmerzen einschätzen an… |
| 510 | Löschung | `schmerzen` | `*(nicht da)*` | …wir jetzt die [___] einschätzen an einer… |
| 511 | Löschung | `einschätzen` | `*(nicht da)*` | …jetzt die schmerzen [___] an einer schmerzskala… |
| 512 | Löschung | `an` | `*(nicht da)*` | …die schmerzen einschätzen [___] einer schmerzskala wobei… |
| 513 | Löschung | `einer` | `*(nicht da)*` | …schmerzen einschätzen an [___] schmerzskala wobei 1… |
| 514 | Löschung | `schmerzskala` | `*(nicht da)*` | …einschätzen an einer [___] wobei 1 sehr… |
| 515 | Löschung | `wobei` | `*(nicht da)*` | …an einer schmerzskala [___] 1 sehr leichten… |
| 516 | Löschung | `1` | `*(nicht da)*` | …einer schmerzskala wobei [___] sehr leichten schmerzen… |
| 517 | Löschung | `sehr` | `*(nicht da)*` | …schmerzskala wobei 1 [___] leichten schmerzen entspricht… |
| 518 | Löschung | `leichten` | `*(nicht da)*` | …wobei 1 sehr [___] schmerzen entspricht und… |
| 519 | Löschung | `schmerzen` | `*(nicht da)*` | …1 sehr leichten [___] entspricht und 10… |
| 520 | Löschung | `entspricht` | `*(nicht da)*` | …sehr leichten schmerzen [___] und 10 sehr… |
| 521 | Löschung | `und` | `*(nicht da)*` | …leichten schmerzen entspricht [___] 10 sehr starken… |
| 522 | Löschung | `10` | `*(nicht da)*` | …schmerzen entspricht und [___] sehr starken schmerzen… |
| 523 | Löschung | `sehr` | `*(nicht da)*` | …entspricht und 10 [___] starken schmerzen wo… |
| 524 | Löschung | `starken` | `*(nicht da)*` | …und 10 sehr [___] schmerzen wo würden… |
| 525 | Löschung | `schmerzen` | `*(nicht da)*` | …10 sehr starken [___] wo würden sie… |
| 526 | Löschung | `wo` | `*(nicht da)*` | …sehr starken schmerzen [___] würden sie die… |
| 527 | Löschung | `würden` | `*(nicht da)*` | …starken schmerzen wo [___] sie die schmerzen… |
| 528 | Löschung | `sie` | `*(nicht da)*` | …schmerzen wo würden [___] die schmerzen des… |
| 529 | Löschung | `die` | `*(nicht da)*` | …wo würden sie [___] schmerzen des daumens… |
| 530 | Löschung | `schmerzen` | `*(nicht da)*` | …würden sie die [___] des daumens einstufen… |
| 531 | Löschung | `des` | `*(nicht da)*` | …sie die schmerzen [___] daumens einstufen beim… |
| 532 | Löschung | `daumens` | `*(nicht da)*` | …die schmerzen des [___] einstufen beim daumen… |
| 533 | Löschung | `einstufen` | `*(nicht da)*` | …schmerzen des daumens [___] beim daumen würde… |
| 534 | Löschung | `beim` | `*(nicht da)*` | …des daumens einstufen [___] daumen würde ich… |
| 535 | Löschung | `daumen` | `*(nicht da)*` | …daumens einstufen beim [___] würde ich schon… |
| 536 | Löschung | `würde` | `*(nicht da)*` | …einstufen beim daumen [___] ich schon sagen… |
| 537 | Löschung | `ich` | `*(nicht da)*` | …beim daumen würde [___] schon sagen geht… |
| 538 | Löschung | `schon` | `*(nicht da)*` | …daumen würde ich [___] sagen geht es… |
| 539 | Löschung | `sagen` | `*(nicht da)*` | …würde ich schon [___] geht es so… |
| 540 | Löschung | `geht` | `*(nicht da)*` | …ich schon sagen [___] es so auf… |
| 541 | Löschung | `es` | `*(nicht da)*` | …schon sagen geht [___] so auf die… |
| 542 | Löschung | `so` | `*(nicht da)*` | …sagen geht es [___] auf die 7… |
| 543 | Löschung | `auf` | `*(nicht da)*` | …geht es so [___] die 7 zu… |
| 544 | Löschung | `die` | `*(nicht da)*` | …es so auf [___] 7 zu vor… |
| 545 | Löschung | `7` | `*(nicht da)*` | …so auf die [___] zu vor allem… |
| 546 | Löschung | `zu` | `*(nicht da)*` | …auf die 7 [___] vor allem wenn… |
| 547 | Löschung | `vor` | `*(nicht da)*` | …die 7 zu [___] allem wenn ich… |
| 548 | Löschung | `allem` | `*(nicht da)*` | …7 zu vor [___] wenn ich versuche… |
| 549 | Löschung | `wenn` | `*(nicht da)*` | …zu vor allem [___] ich versuche ihn… |
| 550 | Löschung | `ich` | `*(nicht da)*` | …vor allem wenn [___] versuche ihn zu… |
| 551 | Löschung | `versuche` | `*(nicht da)*` | …allem wenn ich [___] ihn zu bewegen… |
| 552 | Löschung | `ihn` | `*(nicht da)*` | …wenn ich versuche [___] zu bewegen okay… |
| 553 | Löschung | `zu` | `*(nicht da)*` | …ich versuche ihn [___] bewegen okay was… |
| 554 | Löschung | `bewegen` | `*(nicht da)*` | …versuche ihn zu [___] okay was ist… |
| 555 | Löschung | `okay` | `*(nicht da)*` | …ihn zu bewegen [___] was ist das… |
| 556 | Löschung | `was` | `*(nicht da)*` | …zu bewegen okay [___] ist das denn… |
| 557 | Löschung | `ist` | `*(nicht da)*` | …bewegen okay was [___] das denn für… |
| 558 | Löschung | `das` | `*(nicht da)*` | …okay was ist [___] denn für ein… |
| 559 | Löschung | `denn` | `*(nicht da)*` | …was ist das [___] für ein schmerz… |
| 560 | Löschung | `für` | `*(nicht da)*` | …ist das denn [___] ein schmerz ist… |
| 561 | Löschung | `ein` | `*(nicht da)*` | …das denn für [___] schmerz ist das… |
| 562 | Löschung | `schmerz` | `*(nicht da)*` | …denn für ein [___] ist das ein… |
| 563 | Löschung | `ist` | `*(nicht da)*` | …für ein schmerz [___] das ein stechender… |
| 564 | Löschung | `das` | `*(nicht da)*` | …ein schmerz ist [___] ein stechender schmerz… |
| 565 | Löschung | `ein` | `*(nicht da)*` | …schmerz ist das [___] stechender schmerz ein… |
| 566 | Löschung | `stechender` | `*(nicht da)*` | …ist das ein [___] schmerz ein ziehender… |
| 567 | Löschung | `schmerz` | `*(nicht da)*` | …das ein stechender [___] ein ziehender schmerz… |
| 568 | Löschung | `ein` | `*(nicht da)*` | …ein stechender schmerz [___] ziehender schmerz ein… |
| 569 | Löschung | `ziehender` | `*(nicht da)*` | …stechender schmerz ein [___] schmerz ein brennender… |
| 570 | Löschung | `schmerz` | `*(nicht da)*` | …schmerz ein ziehender [___] ein brennender schmerz… |
| 571 | Löschung | `ein` | `*(nicht da)*` | …ein ziehender schmerz [___] brennender schmerz das… |
| 572 | Löschung | `brennender` | `*(nicht da)*` | …ziehender schmerz ein [___] schmerz das ist… |
| 573 | Löschung | `schmerz` | `*(nicht da)*` | …schmerz ein brennender [___] das ist ein… |
| 574 | Löschung | `das` | `*(nicht da)*` | …ein brennender schmerz [___] ist ein stechender… |
| 575 | Löschung | `ist` | `*(nicht da)*` | …brennender schmerz das [___] ein stechender schmerz… |
| 576 | Löschung | `ein` | `*(nicht da)*` | …schmerz das ist [___] stechender schmerz würde… |
| 577 | Löschung | `stechender` | `*(nicht da)*` | …das ist ein [___] schmerz würde ich… |
| 578 | Löschung | `schmerz` | `*(nicht da)*` | …ist ein stechender [___] würde ich sagen… |
| 579 | Löschung | `würde` | `*(nicht da)*` | …ein stechender schmerz [___] ich sagen sehr… |
| 580 | Löschung | `ich` | `*(nicht da)*` | …stechender schmerz würde [___] sagen sehr stark… |
| 581 | Löschung | `sagen` | `*(nicht da)*` | …schmerz würde ich [___] sehr stark stechend… |
| 582 | Löschung | `sehr` | `*(nicht da)*` | …würde ich sagen [___] stark stechend wenn… |
| 583 | Löschung | `stark` | `*(nicht da)*` | …ich sagen sehr [___] stechend wenn ich… |
| 584 | Löschung | `stechend` | `*(nicht da)*` | …sagen sehr stark [___] wenn ich versuche… |
| 585 | Löschung | `wenn` | `*(nicht da)*` | …sehr stark stechend [___] ich versuche den… |
| 586 | Löschung | `ich` | `*(nicht da)*` | …stark stechend wenn [___] versuche den zu… |
| 587 | Löschung | `versuche` | `*(nicht da)*` | …stechend wenn ich [___] den zu bewegen… |
| 588 | Löschung | `den` | `*(nicht da)*` | …wenn ich versuche [___] zu bewegen okay… |
| 589 | Löschung | `zu` | `*(nicht da)*` | …ich versuche den [___] bewegen okay und… |
| 590 | Löschung | `bewegen` | `*(nicht da)*` | …versuche den zu [___] okay und wie… |
| 591 | Löschung | `okay` | `*(nicht da)*` | …den zu bewegen [___] und wie sieht… |
| 592 | Löschung | `und` | `*(nicht da)*` | …zu bewegen okay [___] wie sieht es… |
| 593 | Löschung | `wie` | `*(nicht da)*` | …bewegen okay und [___] sieht es am… |
| 594 | Löschung | `sieht` | `*(nicht da)*` | …okay und wie [___] es am knie… |
| 595 | Löschung | `es` | `*(nicht da)*` | …und wie sieht [___] am knie aus… |
| 596 | Löschung | `am` | `*(nicht da)*` | …wie sieht es [___] knie aus können… |
| 597 | Löschung | `knie` | `*(nicht da)*` | …sieht es am [___] aus können sie… |
| 598 | Löschung | `aus` | `*(nicht da)*` | …es am knie [___] können sie das… |
| 599 | Löschung | `können` | `*(nicht da)*` | …am knie aus [___] sie das knie… |
| 600 | Löschung | `sie` | `*(nicht da)*` | …knie aus können [___] das knie bewegen… |
| 601 | Löschung | `das` | `*(nicht da)*` | …aus können sie [___] knie bewegen sehr… |
| 602 | Löschung | `knie` | `*(nicht da)*` | …können sie das [___] bewegen sehr sehr… |
| 603 | Löschung | `bewegen` | `*(nicht da)*` | …sie das knie [___] sehr sehr schwer… |
| 604 | Löschung | `sehr` | `*(nicht da)*` | …das knie bewegen [___] sehr schwer da… |
| 605 | Löschung | `sehr` | `*(nicht da)*` | …knie bewegen sehr [___] schwer da tut… |
| 606 | Löschung | `schwer` | `*(nicht da)*` | …bewegen sehr sehr [___] da tut es… |
| 607 | Löschung | `da` | `*(nicht da)*` | …sehr sehr schwer [___] tut es wirklich… |
| 608 | Löschung | `tut` | `*(nicht da)*` | …sehr schwer da [___] es wirklich sehr… |
| 609 | Löschung | `es` | `*(nicht da)*` | …schwer da tut [___] wirklich sehr stark… |
| 610 | Löschung | `wirklich` | `*(nicht da)*` | …da tut es [___] sehr stark weh… |
| 611 | Löschung | `sehr` | `*(nicht da)*` | …tut es wirklich [___] stark weh wenn… |
| 612 | Löschung | `stark` | `*(nicht da)*` | …es wirklich sehr [___] weh wenn ich… |
| 613 | Löschung | `weh` | `*(nicht da)*` | …wirklich sehr stark [___] wenn ich versuche… |
| 614 | Löschung | `wenn` | `*(nicht da)*` | …sehr stark weh [___] ich versuche mein… |
| 615 | Löschung | `ich` | `*(nicht da)*` | …stark weh wenn [___] versuche mein knie… |
| 616 | Löschung | `versuche` | `*(nicht da)*` | …weh wenn ich [___] mein knie zu… |
| 617 | Löschung | `mein` | `*(nicht da)*` | …wenn ich versuche [___] knie zu bewegen… |
| 618 | Löschung | `knie` | `*(nicht da)*` | …ich versuche mein [___] zu bewegen es… |
| 619 | Löschung | `zu` | `*(nicht da)*` | …versuche mein knie [___] bewegen es tut… |
| 620 | Löschung | `bewegen` | `*(nicht da)*` | …mein knie zu [___] es tut selbst… |
| 621 | Löschung | `es` | `*(nicht da)*` | …knie zu bewegen [___] tut selbst weh… |
| 622 | Löschung | `tut` | `*(nicht da)*` | …zu bewegen es [___] selbst weh wenn… |
| 623 | Löschung | `selbst` | `*(nicht da)*` | …bewegen es tut [___] weh wenn ich… |
| 624 | Löschung | `weh` | `*(nicht da)*` | …es tut selbst [___] wenn ich gerade… |
| 625 | Löschung | `wenn` | `*(nicht da)*` | …tut selbst weh [___] ich gerade einfach… |
| 626 | Löschung | `ich` | `*(nicht da)*` | …selbst weh wenn [___] gerade einfach so… |
| 627 | Löschung | `gerade` | `*(nicht da)*` | …weh wenn ich [___] einfach so hier… |
| 628 | Löschung | `einfach` | `*(nicht da)*` | …wenn ich gerade [___] so hier sitze… |
| 629 | Löschung | `so` | `*(nicht da)*` | …ich gerade einfach [___] hier sitze okay… |
| 630 | Löschung | `hier` | `*(nicht da)*` | …gerade einfach so [___] sitze okay sogar… |
| 631 | Löschung | `sitze` | `*(nicht da)*` | …einfach so hier [___] okay sogar im… |
| 632 | Löschung | `okay` | `*(nicht da)*` | …so hier sitze [___] sogar im ruhezustand… |
| 633 | Löschung | `sogar` | `*(nicht da)*` | …hier sitze okay [___] im ruhezustand ja… |
| 634 | Löschung | `im` | `*(nicht da)*` | …sitze okay sogar [___] ruhezustand ja wo… |
| 635 | Löschung | `ruhezustand` | `*(nicht da)*` | …okay sogar im [___] ja wo würden… |
| 636 | Löschung | `ja` | `*(nicht da)*` | …sogar im ruhezustand [___] wo würden sie… |
| 637 | Löschung | `wo` | `*(nicht da)*` | …im ruhezustand ja [___] würden sie die… |
| 638 | Löschung | `würden` | `*(nicht da)*` | …ruhezustand ja wo [___] sie die schmerzen… |
| 639 | Löschung | `sie` | `*(nicht da)*` | …ja wo würden [___] die schmerzen hier… |
| 640 | Löschung | `die` | `*(nicht da)*` | …wo würden sie [___] schmerzen hier einstufen… |
| 641 | Löschung | `schmerzen` | `*(nicht da)*` | …würden sie die [___] hier einstufen da… |
| 642 | Löschung | `hier` | `*(nicht da)*` | …sie die schmerzen [___] einstufen da würde… |
| 643 | Löschung | `einstufen` | `*(nicht da)*` | …die schmerzen hier [___] da würde ich… |
| 644 | Löschung | `da` | `*(nicht da)*` | …schmerzen hier einstufen [___] würde ich sagen… |
| 645 | Löschung | `würde` | `*(nicht da)*` | …hier einstufen da [___] ich sagen bei… |
| 646 | Löschung | `ich` | `*(nicht da)*` | …einstufen da würde [___] sagen bei 8… |
| 647 | Löschung | `sagen` | `*(nicht da)*` | …da würde ich [___] bei 8 wenn… |
| 648 | Löschung | `bei` | `*(nicht da)*` | …würde ich sagen [___] 8 wenn ich… |
| 649 | Löschung | `8` | `*(nicht da)*` | …ich sagen bei [___] wenn ich sitze… |
| 650 | Löschung | `wenn` | `*(nicht da)*` | …sagen bei 8 [___] ich sitze und… |
| 651 | Löschung | `ich` | `*(nicht da)*` | …bei 8 wenn [___] sitze und wenn… |
| 652 | Löschung | `sitze` | `*(nicht da)*` | …8 wenn ich [___] und wenn ich… |
| 653 | Löschung | `und` | `*(nicht da)*` | …wenn ich sitze [___] wenn ich versuche… |
| 654 | Löschung | `wenn` | `*(nicht da)*` | …ich sitze und [___] ich versuche mein… |
| 655 | Löschung | `ich` | `*(nicht da)*` | …sitze und wenn [___] versuche mein knie… |
| 656 | Löschung | `versuche` | `*(nicht da)*` | …und wenn ich [___] mein knie zu… |
| 657 | Löschung | `mein` | `*(nicht da)*` | …wenn ich versuche [___] knie zu bewegen… |
| 658 | Löschung | `knie` | `*(nicht da)*` | …ich versuche mein [___] zu bewegen ist… |
| 659 | Löschung | `zu` | `*(nicht da)*` | …versuche mein knie [___] bewegen ist es… |
| 660 | Löschung | `bewegen` | `*(nicht da)*` | …mein knie zu [___] ist es wirklich… |
| 661 | Löschung | `ist` | `*(nicht da)*` | …knie zu bewegen [___] es wirklich unerträglich… |
| 662 | Löschung | `es` | `*(nicht da)*` | …zu bewegen ist [___] wirklich unerträglich okay… |
| 663 | Löschung | `wirklich` | `*(nicht da)*` | …bewegen ist es [___] unerträglich okay okay… |
| 664 | Löschung | `unerträglich` | `*(nicht da)*` | …ist es wirklich [___] okay okay gut… |
| 665 | Löschung | `okay` | `*(nicht da)*` | …es wirklich unerträglich [___] okay gut strahlen… |
| 666 | Löschung | `okay` | `*(nicht da)*` | …wirklich unerträglich okay [___] gut strahlen die… |
| 667 | Löschung | `gut` | `*(nicht da)*` | …unerträglich okay okay [___] strahlen die schmerzen… |
| 668 | Löschung | `strahlen` | `*(nicht da)*` | …okay okay gut [___] die schmerzen noch… |
| 669 | Löschung | `die` | `*(nicht da)*` | …okay gut strahlen [___] schmerzen noch in… |
| 670 | Löschung | `schmerzen` | `*(nicht da)*` | …gut strahlen die [___] noch in andere… |
| 671 | Löschung | `noch` | `*(nicht da)*` | …strahlen die schmerzen [___] in andere körperregionen… |
| 672 | Löschung | `in` | `*(nicht da)*` | …die schmerzen noch [___] andere körperregionen aus… |
| 673 | Löschung | `andere` | `*(nicht da)*` | …schmerzen noch in [___] körperregionen aus nein… |
| 674 | Löschung | `körperregionen` | `*(nicht da)*` | …noch in andere [___] aus nein das… |
| 675 | Löschung | `aus` | `*(nicht da)*` | …in andere körperregionen [___] nein das zum… |
| 676 | Löschung | `nein` | `*(nicht da)*` | …andere körperregionen aus [___] das zum glück… |
| 677 | Löschung | `das` | `*(nicht da)*` | …körperregionen aus nein [___] zum glück nicht… |
| 678 | Löschung | `zum` | `*(nicht da)*` | …aus nein das [___] glück nicht okay… |
| 679 | Löschung | `glück` | `*(nicht da)*` | …nein das zum [___] nicht okay wie… |
| 680 | Löschung | `nicht` | `*(nicht da)*` | …das zum glück [___] okay wie sieht… |
| 681 | Löschung | `okay` | `*(nicht da)*` | …zum glück nicht [___] wie sieht es… |
| 682 | Löschung | `wie` | `*(nicht da)*` | …glück nicht okay [___] sieht es an… |
| 683 | Löschung | `sieht` | `*(nicht da)*` | …nicht okay wie [___] es an der… |
| 684 | Löschung | `es` | `*(nicht da)*` | …okay wie sieht [___] an der hand… |
| 685 | Löschung | `an` | `*(nicht da)*` | …wie sieht es [___] der hand aus… |
| 686 | Löschung | `der` | `*(nicht da)*` | …sieht es an [___] hand aus am… |
| 687 | Löschung | `hand` | `*(nicht da)*` | …es an der [___] aus am daumen… |
| 688 | Löschung | `aus` | `*(nicht da)*` | …an der hand [___] am daumen strahlen… |
| 689 | Löschung | `am` | `*(nicht da)*` | …der hand aus [___] daumen strahlen die… |
| 690 | Löschung | `daumen` | `*(nicht da)*` | …hand aus am [___] strahlen die schmerzen… |
| 691 | Löschung | `strahlen` | `*(nicht da)*` | …aus am daumen [___] die schmerzen da… |
| 692 | Löschung | `die` | `*(nicht da)*` | …am daumen strahlen [___] schmerzen da irgendwie… |
| 693 | Löschung | `schmerzen` | `*(nicht da)*` | …daumen strahlen die [___] da irgendwie ins… |
| 694 | Löschung | `da` | `*(nicht da)*` | …strahlen die schmerzen [___] irgendwie ins handgelenk… |
| 695 | Löschung | `irgendwie` | `*(nicht da)*` | …die schmerzen da [___] ins handgelenk aus… |
| 696 | Löschung | `ins` | `*(nicht da)*` | …schmerzen da irgendwie [___] handgelenk aus oder… |
| 697 | Löschung | `handgelenk` | `*(nicht da)*` | …da irgendwie ins [___] aus oder in… |
| 698 | Löschung | `aus` | `*(nicht da)*` | …irgendwie ins handgelenk [___] oder in andere… |
| 699 | Löschung | `oder` | `*(nicht da)*` | …ins handgelenk aus [___] in andere finger… |
| 700 | Löschung | `in` | `*(nicht da)*` | …handgelenk aus oder [___] andere finger auch… |
| 701 | Löschung | `andere` | `*(nicht da)*` | …aus oder in [___] finger auch nicht… |
| 702 | Löschung | `finger` | `*(nicht da)*` | …oder in andere [___] auch nicht nein… |
| 703 | Löschung | `auch` | `*(nicht da)*` | …in andere finger [___] nicht nein okay… |
| 704 | Löschung | `nicht` | `*(nicht da)*` | …andere finger auch [___] nein okay sehr… |
| 705 | Löschung | `nein` | `*(nicht da)*` | …finger auch nicht [___] okay sehr sehr… |
| 706 | Löschung | `okay` | `*(nicht da)*` | …auch nicht nein [___] sehr sehr gut… |
| 707 | Löschung | `sehr` | `*(nicht da)*` | …nicht nein okay [___] sehr gut können… |
| 708 | Löschung | `sehr` | `*(nicht da)*` | …nein okay sehr [___] gut können sie… |
| 709 | Löschung | `gut` | `*(nicht da)*` | …okay sehr sehr [___] können sie sich… |
| 710 | Löschung | `können` | `*(nicht da)*` | …sehr sehr gut [___] sie sich an… |
| 711 | Löschung | `sie` | `*(nicht da)*` | …sehr gut können [___] sich an den… |
| 712 | Löschung | `sich` | `*(nicht da)*` | …gut können sie [___] an den unfall… |
| 713 | Löschung | `an` | `*(nicht da)*` | …können sie sich [___] den unfall erinnern… |
| 714 | Löschung | `den` | `*(nicht da)*` | …sie sich an [___] unfall erinnern frau… |
| 715 | Löschung | `unfall` | `*(nicht da)*` | …sich an den [___] erinnern frau beckenwestfalen… |
| 716 | Löschung | `erinnern` | `*(nicht da)*` | …an den unfall [___] frau beckenwestfalen ich… |
| 717 | Löschung | `frau` | `*(nicht da)*` | …den unfall erinnern [___] beckenwestfalen ich kann… |
| 718 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …unfall erinnern frau [___] ich kann mich… |
| 719 | Löschung | `ich` | `*(nicht da)*` | …erinnern frau beckenwestfalen [___] kann mich gut… |
| 720 | Löschung | `kann` | `*(nicht da)*` | …frau beckenwestfalen ich [___] mich gut daran… |
| 721 | Löschung | `mich` | `*(nicht da)*` | …beckenwestfalen ich kann [___] gut daran erinnern… |
| 722 | Löschung | `gut` | `*(nicht da)*` | …ich kann mich [___] daran erinnern ja… |
| 723 | Löschung | `daran` | `*(nicht da)*` | …kann mich gut [___] erinnern ja ich… |
| 724 | Löschung | `erinnern` | `*(nicht da)*` | …mich gut daran [___] ja ich war… |
| 725 | Löschung | `ja` | `*(nicht da)*` | …gut daran erinnern [___] ich war am… |
| 726 | Löschung | `ich` | `*(nicht da)*` | …daran erinnern ja [___] war am anfang… |
| 727 | Löschung | `war` | `*(nicht da)*` | …erinnern ja ich [___] am anfang zwar… |
| 728 | Löschung | `am` | `*(nicht da)*` | …ja ich war [___] anfang zwar etwas… |
| 729 | Löschung | `anfang` | `*(nicht da)*` | …ich war am [___] zwar etwas benebelt… |
| 730 | Löschung | `zwar` | `*(nicht da)*` | …war am anfang [___] etwas benebelt und… |
| 731 | Löschung | `etwas` | `*(nicht da)*` | …am anfang zwar [___] benebelt und mir… |
| 732 | Löschung | `benebelt` | `*(nicht da)*` | …anfang zwar etwas [___] und mir war… |
| 733 | Löschung | `und` | `*(nicht da)*` | …zwar etwas benebelt [___] mir war es… |
| 734 | Löschung | `mir` | `*(nicht da)*` | …etwas benebelt und [___] war es ziemlich… |
| 735 | Löschung | `war` | `*(nicht da)*` | …benebelt und mir [___] es ziemlich schwindelig… |
| 736 | Löschung | `es` | `*(nicht da)*` | …und mir war [___] ziemlich schwindelig aber… |
| 737 | Löschung | `ziemlich` | `*(nicht da)*` | …mir war es [___] schwindelig aber ich… |
| 738 | Löschung | `schwindelig` | `*(nicht da)*` | …war es ziemlich [___] aber ich denke… |
| 739 | Löschung | `aber` | `*(nicht da)*` | …es ziemlich schwindelig [___] ich denke das… |
| 740 | Löschung | `ich` | `*(nicht da)*` | …ziemlich schwindelig aber [___] denke das lag… |
| 741 | Löschung | `denke` | `*(nicht da)*` | …schwindelig aber ich [___] das lag vielleicht… |
| 742 | Löschung | `das` | `*(nicht da)*` | …aber ich denke [___] lag vielleicht am… |
| 743 | Löschung | `lag` | `*(nicht da)*` | …ich denke das [___] vielleicht am schock… |
| 744 | Löschung | `vielleicht` | `*(nicht da)*` | …denke das lag [___] am schock im… |
| 745 | Löschung | `am` | `*(nicht da)*` | …das lag vielleicht [___] schock im ersten… |
| 746 | Löschung | `schock` | `*(nicht da)*` | …lag vielleicht am [___] im ersten moment… |
| 747 | Löschung | `im` | `*(nicht da)*` | …vielleicht am schock [___] ersten moment okay… |
| 748 | Löschung | `ersten` | `*(nicht da)*` | …am schock im [___] moment okay gibt… |
| 749 | Löschung | `moment` | `*(nicht da)*` | …schock im ersten [___] okay gibt es… |
| 750 | Löschung | `okay` | `*(nicht da)*` | …im ersten moment [___] gibt es sonst… |
| 751 | Löschung | `gibt` | `*(nicht da)*` | …ersten moment okay [___] es sonst etwas… |
| 752 | Löschung | `es` | `*(nicht da)*` | …moment okay gibt [___] sonst etwas was… |
| 753 | Löschung | `sonst` | `*(nicht da)*` | …okay gibt es [___] etwas was ihnen… |
| 754 | Löschung | `etwas` | `*(nicht da)*` | …gibt es sonst [___] was ihnen aufgefallen… |
| 755 | Löschung | `was` | `*(nicht da)*` | …es sonst etwas [___] ihnen aufgefallen ist… |
| 756 | Löschung | `ihnen` | `*(nicht da)*` | …sonst etwas was [___] aufgefallen ist seit… |
| 757 | Löschung | `aufgefallen` | `*(nicht da)*` | …etwas was ihnen [___] ist seit dem… |
| 758 | Löschung | `ist` | `*(nicht da)*` | …was ihnen aufgefallen [___] seit dem unfall… |
| 759 | Löschung | `seit` | `*(nicht da)*` | …ihnen aufgefallen ist [___] dem unfall was… |
| 760 | Löschung | `dem` | `*(nicht da)*` | …aufgefallen ist seit [___] unfall was ich… |
| 761 | Löschung | `unfall` | `*(nicht da)*` | …ist seit dem [___] was ich wissen… |
| 762 | Löschung | `was` | `*(nicht da)*` | …seit dem unfall [___] ich wissen sollte… |
| 763 | Löschung | `ich` | `*(nicht da)*` | …dem unfall was [___] wissen sollte ist… |
| 764 | Löschung | `wissen` | `*(nicht da)*` | …unfall was ich [___] sollte ist ihnen… |
| 765 | Löschung | `sollte` | `*(nicht da)*` | …was ich wissen [___] ist ihnen übel… |
| 766 | Löschung | `ist` | `*(nicht da)*` | …ich wissen sollte [___] ihnen übel geworden… |
| 767 | Löschung | `ihnen` | `*(nicht da)*` | …wissen sollte ist [___] übel geworden oder… |
| 768 | Löschung | `übel` | `*(nicht da)*` | …sollte ist ihnen [___] geworden oder vielleicht… |
| 769 | Löschung | `geworden` | `*(nicht da)*` | …ist ihnen übel [___] oder vielleicht doch… |
| 770 | Löschung | `oder` | `*(nicht da)*` | …ihnen übel geworden [___] vielleicht doch nochmal… |
| 771 | Löschung | `vielleicht` | `*(nicht da)*` | …übel geworden oder [___] doch nochmal schwarz… |
| 772 | Löschung | `doch` | `*(nicht da)*` | …geworden oder vielleicht [___] nochmal schwarz vor… |
| 773 | Löschung | `nochmal` | `*(nicht da)*` | …oder vielleicht doch [___] schwarz vor augen… |
| 774 | Löschung | `schwarz` | `*(nicht da)*` | …vielleicht doch nochmal [___] vor augen oder… |
| 775 | Löschung | `vor` | `*(nicht da)*` | …doch nochmal schwarz [___] augen oder fühlen… |
| 776 | Löschung | `augen` | `*(nicht da)*` | …nochmal schwarz vor [___] oder fühlen sie… |
| 777 | Löschung | `oder` | `*(nicht da)*` | …schwarz vor augen [___] fühlen sie sich… |
| 778 | Löschung | `fühlen` | `*(nicht da)*` | …vor augen oder [___] sie sich seltsam… |
| 779 | Löschung | `sie` | `*(nicht da)*` | …augen oder fühlen [___] sich seltsam seitdem… |
| 780 | Löschung | `sich` | `*(nicht da)*` | …oder fühlen sie [___] seltsam seitdem nein… |
| 781 | Löschung | `seltsam` | `*(nicht da)*` | …fühlen sie sich [___] seitdem nein außer… |
| 782 | Löschung | `seitdem` | `*(nicht da)*` | …sie sich seltsam [___] nein außer dass… |
| 783 | Löschung | `nein` | `*(nicht da)*` | …sich seltsam seitdem [___] außer dass ich… |
| 784 | Löschung | `außer` | `*(nicht da)*` | …seltsam seitdem nein [___] dass ich sehr… |
| 785 | Löschung | `dass` | `*(nicht da)*` | …seitdem nein außer [___] ich sehr starke… |
| 786 | Löschung | `ich` | `*(nicht da)*` | …nein außer dass [___] sehr starke schmerzen… |
| 787 | Löschung | `sehr` | `*(nicht da)*` | …außer dass ich [___] starke schmerzen habe… |
| 788 | Löschung | `starke` | `*(nicht da)*` | …dass ich sehr [___] schmerzen habe ist… |
| 789 | Löschung | `schmerzen` | `*(nicht da)*` | …ich sehr starke [___] habe ist mir… |
| 790 | Löschung | `habe` | `*(nicht da)*` | …sehr starke schmerzen [___] ist mir nichts… |
| 791 | Löschung | `ist` | `*(nicht da)*` | …starke schmerzen habe [___] mir nichts anderes… |
| 792 | Löschung | `mir` | `*(nicht da)*` | …schmerzen habe ist [___] nichts anderes aufgefallen… |
| 793 | Löschung | `nichts` | `*(nicht da)*` | …habe ist mir [___] anderes aufgefallen und… |
| 794 | Löschung | `anderes` | `*(nicht da)*` | …ist mir nichts [___] aufgefallen und dass… |
| 795 | Löschung | `aufgefallen` | `*(nicht da)*` | …mir nichts anderes [___] und dass ich… |
| 796 | Löschung | `und` | `*(nicht da)*` | …nichts anderes aufgefallen [___] dass ich am… |
| 797 | Löschung | `dass` | `*(nicht da)*` | …anderes aufgefallen und [___] ich am anfang… |
| 798 | Löschung | `ich` | `*(nicht da)*` | …aufgefallen und dass [___] am anfang nur… |
| 799 | Löschung | `am` | `*(nicht da)*` | …und dass ich [___] anfang nur etwas… |
| 800 | Löschung | `anfang` | `*(nicht da)*` | …dass ich am [___] nur etwas benebelt… |
| 801 | Löschung | `nur` | `*(nicht da)*` | …ich am anfang [___] etwas benebelt war… |
| 802 | Löschung | `etwas` | `*(nicht da)*` | …am anfang nur [___] benebelt war aber… |
| 803 | Löschung | `benebelt` | `*(nicht da)*` | …anfang nur etwas [___] war aber jetzt… |
| 804 | Löschung | `war` | `*(nicht da)*` | …nur etwas benebelt [___] aber jetzt bin… |
| 805 | Löschung | `aber` | `*(nicht da)*` | …etwas benebelt war [___] jetzt bin ich… |
| 806 | Löschung | `jetzt` | `*(nicht da)*` | …benebelt war aber [___] bin ich ganz… |
| 807 | Löschung | `bin` | `*(nicht da)*` | …war aber jetzt [___] ich ganz klar… |
| 808 | Löschung | `ich` | `*(nicht da)*` | …aber jetzt bin [___] ganz klar okay… |
| 809 | Löschung | `ganz` | `*(nicht da)*` | …jetzt bin ich [___] klar okay gut… |
| 810 | Löschung | `klar` | `*(nicht da)*` | …bin ich ganz [___] okay gut sehr… |
| 811 | Löschung | `okay` | `*(nicht da)*` | …ich ganz klar [___] gut sehr sehr… |
| 812 | Löschung | `gut` | `*(nicht da)*` | …ganz klar okay [___] sehr sehr gut… |
| 813 | Löschung | `sehr` | `*(nicht da)*` | …klar okay gut [___] sehr gut frau… |
| 814 | Löschung | `sehr` | `*(nicht da)*` | …okay gut sehr [___] gut frau beckenwestfalen… |
| 815 | Löschung | `gut` | `*(nicht da)*` | …gut sehr sehr [___] frau beckenwestfalen haben… |
| 816 | Löschung | `frau` | `*(nicht da)*` | …sehr sehr gut [___] beckenwestfalen haben sie… |
| 817 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …sehr gut frau [___] haben sie irgendwelche… |
| 818 | Löschung | `haben` | `*(nicht da)*` | …gut frau beckenwestfalen [___] sie irgendwelche vorerkrankungen… |
| 819 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen haben [___] irgendwelche vorerkrankungen von… |
| 820 | Löschung | `irgendwelche` | `*(nicht da)*` | …beckenwestfalen haben sie [___] vorerkrankungen von denen… |
| 821 | Substitution | `vorerkrankungen` | `handelt` | …haben sie irgendwelche [___] von denen ich… |
| 822 | Löschung | `denen` | `*(nicht da)*` | …irgendwelche vorerkrankungen von [___] ich wissen sollte… |
| 823 | Löschung | `ich` | `*(nicht da)*` | …vorerkrankungen von denen [___] wissen sollte wie… |
| 824 | Löschung | `wissen` | `*(nicht da)*` | …von denen ich [___] sollte wie zum… |
| 825 | Löschung | `sollte` | `*(nicht da)*` | …denen ich wissen [___] wie zum beispiel… |
| 826 | Löschung | `wie` | `*(nicht da)*` | …ich wissen sollte [___] zum beispiel erhöhten… |
| 827 | Löschung | `zum` | `*(nicht da)*` | …wissen sollte wie [___] beispiel erhöhten blutdruck… |
| 828 | Löschung | `beispiel` | `*(nicht da)*` | …sollte wie zum [___] erhöhten blutdruck oder… |
| 829 | Löschung | `erhöhten` | `*(nicht da)*` | …wie zum beispiel [___] blutdruck oder diabetes… |
| 830 | Löschung | `blutdruck` | `*(nicht da)*` | …zum beispiel erhöhten [___] oder diabetes oder… |
| 831 | Löschung | `oder` | `*(nicht da)*` | …beispiel erhöhten blutdruck [___] diabetes oder etwas… |
| 832 | Löschung | `diabetes` | `*(nicht da)*` | …erhöhten blutdruck oder [___] oder etwas anderes… |
| 833 | Löschung | `oder` | `*(nicht da)*` | …blutdruck oder diabetes [___] etwas anderes nichts… |
| 834 | Löschung | `etwas` | `*(nicht da)*` | …oder diabetes oder [___] anderes nichts ernsthaftes… |
| 835 | Löschung | `anderes` | `*(nicht da)*` | …diabetes oder etwas [___] nichts ernsthaftes ich… |
| 836 | Löschung | `nichts` | `*(nicht da)*` | …oder etwas anderes [___] ernsthaftes ich hatte… |
| 837 | Löschung | `ernsthaftes` | `*(nicht da)*` | …etwas anderes nichts [___] ich hatte eine… |
| 838 | Löschung | `ich` | `*(nicht da)*` | …anderes nichts ernsthaftes [___] hatte eine laktoseintoleranz… |
| 839 | Löschung | `hatte` | `*(nicht da)*` | …nichts ernsthaftes ich [___] eine laktoseintoleranz vor… |
| 840 | Löschung | `eine` | `*(nicht da)*` | …ernsthaftes ich hatte [___] laktoseintoleranz vor einigen… |
| 841 | Löschung | `laktoseintoleranz` | `*(nicht da)*` | …ich hatte eine [___] vor einigen jahren… |
| 842 | Löschung | `vor` | `*(nicht da)*` | …hatte eine laktoseintoleranz [___] einigen jahren sie… |
| 843 | Löschung | `einigen` | `*(nicht da)*` | …eine laktoseintoleranz vor [___] jahren sie ist… |
| 844 | Löschung | `jahren` | `*(nicht da)*` | …laktoseintoleranz vor einigen [___] sie ist allerdings… |
| 845 | Löschung | `sie` | `*(nicht da)*` | …vor einigen jahren [___] ist allerdings schon… |
| 846 | Löschung | `ist` | `*(nicht da)*` | …einigen jahren sie [___] allerdings schon weg… |
| 847 | Löschung | `allerdings` | `*(nicht da)*` | …jahren sie ist [___] schon weg und… |
| 848 | Löschung | `schon` | `*(nicht da)*` | …sie ist allerdings [___] weg und jetzt… |
| 849 | Löschung | `weg` | `*(nicht da)*` | …ist allerdings schon [___] und jetzt wurde… |
| 850 | Löschung | `und` | `*(nicht da)*` | …allerdings schon weg [___] jetzt wurde bei… |
| 851 | Löschung | `jetzt` | `*(nicht da)*` | …schon weg und [___] wurde bei mir… |
| 852 | Löschung | `wurde` | `*(nicht da)*` | …weg und jetzt [___] bei mir vor… |
| 853 | Löschung | `bei` | `*(nicht da)*` | …und jetzt wurde [___] mir vor drei… |
| 854 | Löschung | `mir` | `*(nicht da)*` | …jetzt wurde bei [___] vor drei wochen… |
| 855 | Löschung | `vor` | `*(nicht da)*` | …wurde bei mir [___] drei wochen eine… |
| 856 | Löschung | `drei` | `*(nicht da)*` | …bei mir vor [___] wochen eine histaminunverträglichkeit… |
| 857 | Löschung | `wochen` | `*(nicht da)*` | …mir vor drei [___] eine histaminunverträglichkeit festgestellt… |
| 858 | Löschung | `eine` | `*(nicht da)*` | …vor drei wochen [___] histaminunverträglichkeit festgestellt wie… |
| 859 | Löschung | `histaminunverträglichkeit` | `*(nicht da)*` | …drei wochen eine [___] festgestellt wie äußert… |
| 860 | Löschung | `festgestellt` | `*(nicht da)*` | …wochen eine histaminunverträglichkeit [___] wie äußert sich… |
| 861 | Löschung | `wie` | `*(nicht da)*` | …eine histaminunverträglichkeit festgestellt [___] äußert sich die… |
| 862 | Löschung | `äußert` | `*(nicht da)*` | …histaminunverträglichkeit festgestellt wie [___] sich die unverträglichkeit… |
| 863 | Löschung | `sich` | `*(nicht da)*` | …festgestellt wie äußert [___] die unverträglichkeit wenn… |
| 864 | Löschung | `die` | `*(nicht da)*` | …wie äußert sich [___] unverträglichkeit wenn ich… |
| 865 | Löschung | `unverträglichkeit` | `*(nicht da)*` | …äußert sich die [___] wenn ich bestimmte… |
| 866 | Löschung | `wenn` | `*(nicht da)*` | …sich die unverträglichkeit [___] ich bestimmte sachen… |
| 867 | Löschung | `ich` | `*(nicht da)*` | …die unverträglichkeit wenn [___] bestimmte sachen esse… |
| 868 | Löschung | `bestimmte` | `*(nicht da)*` | …unverträglichkeit wenn ich [___] sachen esse oder… |
| 869 | Löschung | `sachen` | `*(nicht da)*` | …wenn ich bestimmte [___] esse oder trinke… |
| 870 | Löschung | `esse` | `*(nicht da)*` | …ich bestimmte sachen [___] oder trinke vor… |
| 871 | Löschung | `oder` | `*(nicht da)*` | …bestimmte sachen esse [___] trinke vor allem… |
| 872 | Löschung | `trinke` | `*(nicht da)*` | …sachen esse oder [___] vor allem in… |
| 873 | Löschung | `vor` | `*(nicht da)*` | …esse oder trinke [___] allem in kombination… |
| 874 | Löschung | `allem` | `*(nicht da)*` | …oder trinke vor [___] in kombination dann… |
| 875 | Löschung | `in` | `*(nicht da)*` | …trinke vor allem [___] kombination dann bekomme… |
| 876 | Löschung | `kombination` | `*(nicht da)*` | …vor allem in [___] dann bekomme ich… |
| 877 | Löschung | `dann` | `*(nicht da)*` | …allem in kombination [___] bekomme ich starke… |
| 878 | Löschung | `bekomme` | `*(nicht da)*` | …in kombination dann [___] ich starke bauchschmerzen… |
| 879 | Löschung | `ich` | `*(nicht da)*` | …kombination dann bekomme [___] starke bauchschmerzen übelkeit… |
| 880 | Löschung | `starke` | `*(nicht da)*` | …dann bekomme ich [___] bauchschmerzen übelkeit manchmal… |
| 881 | Löschung | `bauchschmerzen` | `*(nicht da)*` | …bekomme ich starke [___] übelkeit manchmal und… |
| 882 | Löschung | `übelkeit` | `*(nicht da)*` | …ich starke bauchschmerzen [___] manchmal und manchmal… |
| 883 | Löschung | `manchmal` | `*(nicht da)*` | …starke bauchschmerzen übelkeit [___] und manchmal auch… |
| 884 | Löschung | `und` | `*(nicht da)*` | …bauchschmerzen übelkeit manchmal [___] manchmal auch einen… |
| 885 | Löschung | `manchmal` | `*(nicht da)*` | …übelkeit manchmal und [___] auch einen ausschlag… |
| 886 | Löschung | `auch` | `*(nicht da)*` | …manchmal und manchmal [___] einen ausschlag hier… |
| 887 | Löschung | `einen` | `*(nicht da)*` | …und manchmal auch [___] ausschlag hier im… |
| 888 | Löschung | `ausschlag` | `*(nicht da)*` | …manchmal auch einen [___] hier im dekolleté… |
| 889 | Löschung | `hier` | `*(nicht da)*` | …auch einen ausschlag [___] im dekolleté bereich… |
| 890 | Löschung | `im` | `*(nicht da)*` | …einen ausschlag hier [___] dekolleté bereich okay… |
| 891 | Löschung | `dekolleté` | `*(nicht da)*` | …ausschlag hier im [___] bereich okay sonst… |
| 892 | Löschung | `bereich` | `*(nicht da)*` | …hier im dekolleté [___] okay sonst gibt… |
| 893 | Löschung | `okay` | `*(nicht da)*` | …im dekolleté bereich [___] sonst gibt es… |
| 894 | Löschung | `sonst` | `*(nicht da)*` | …dekolleté bereich okay [___] gibt es aber… |
| 895 | Löschung | `gibt` | `*(nicht da)*` | …bereich okay sonst [___] es aber keine… |
| 896 | Löschung | `es` | `*(nicht da)*` | …okay sonst gibt [___] aber keine vorerkrankungen… |
| 897 | Löschung | `aber` | `*(nicht da)*` | …sonst gibt es [___] keine vorerkrankungen nein… |
| 898 | Löschung | `keine` | `*(nicht da)*` | …gibt es aber [___] vorerkrankungen nein nein… |
| 899 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …es aber keine [___] nein nein okay… |
| 900 | Löschung | `nein` | `*(nicht da)*` | …aber keine vorerkrankungen [___] nein okay sehr… |
| 901 | Löschung | `nein` | `*(nicht da)*` | …keine vorerkrankungen nein [___] okay sehr gut… |
| 902 | Löschung | `okay` | `*(nicht da)*` | …vorerkrankungen nein nein [___] sehr gut frau… |
| 903 | Löschung | `sehr` | `*(nicht da)*` | …nein nein okay [___] gut frau beckenwestfalen… |
| 904 | Löschung | `gut` | `*(nicht da)*` | …nein okay sehr [___] frau beckenwestfalen sind… |
| 905 | Löschung | `frau` | `*(nicht da)*` | …okay sehr gut [___] beckenwestfalen sind sie… |
| 906 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …sehr gut frau [___] sind sie schon… |
| 907 | Löschung | `sind` | `*(nicht da)*` | …gut frau beckenwestfalen [___] sie schon einmal… |
| 908 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen sind [___] schon einmal operiert… |
| 909 | Löschung | `schon` | `*(nicht da)*` | …beckenwestfalen sind sie [___] einmal operiert worden… |
| 910 | Löschung | `einmal` | `*(nicht da)*` | …sind sie schon [___] operiert worden ja… |
| 911 | Löschung | `operiert` | `*(nicht da)*` | …sie schon einmal [___] worden ja ich… |
| 912 | Löschung | `worden` | `*(nicht da)*` | …schon einmal operiert [___] ja ich wurde… |
| 913 | Löschung | `ja` | `*(nicht da)*` | …einmal operiert worden [___] ich wurde vor… |
| 914 | Löschung | `ich` | `*(nicht da)*` | …operiert worden ja [___] wurde vor zwei… |
| 915 | Löschung | `wurde` | `*(nicht da)*` | …worden ja ich [___] vor zwei jahren… |
| 916 | Löschung | `vor` | `*(nicht da)*` | …ja ich wurde [___] zwei jahren am… |
| 917 | Löschung | `zwei` | `*(nicht da)*` | …ich wurde vor [___] jahren am fuß… |
| 918 | Löschung | `jahren` | `*(nicht da)*` | …wurde vor zwei [___] am fuß operiert… |
| 919 | Löschung | `am` | `*(nicht da)*` | …vor zwei jahren [___] fuß operiert mir… |
| 920 | Löschung | `fuß` | `*(nicht da)*` | …zwei jahren am [___] operiert mir wurde… |
| 921 | Löschung | `operiert` | `*(nicht da)*` | …jahren am fuß [___] mir wurde ein… |
| 922 | Löschung | `mir` | `*(nicht da)*` | …am fuß operiert [___] wurde ein halux… |
| 923 | Löschung | `wurde` | `*(nicht da)*` | …fuß operiert mir [___] ein halux valgus… |
| 924 | Löschung | `ein` | `*(nicht da)*` | …operiert mir wurde [___] halux valgus entfernt… |
| 925 | Löschung | `halux` | `*(nicht da)*` | …mir wurde ein [___] valgus entfernt ein… |
| 926 | Löschung | `valgus` | `*(nicht da)*` | …wurde ein halux [___] entfernt ein halux… |
| 927 | Löschung | `entfernt` | `*(nicht da)*` | …ein halux valgus [___] ein halux valgus… |
| 928 | Löschung | `ein` | `*(nicht da)*` | …halux valgus entfernt [___] halux valgus und… |
| 929 | Löschung | `halux` | `*(nicht da)*` | …valgus entfernt ein [___] valgus und welcher… |
| 930 | Löschung | `valgus` | `*(nicht da)*` | …entfernt ein halux [___] und welcher fuß… |
| 931 | Löschung | `und` | `*(nicht da)*` | …ein halux valgus [___] welcher fuß war… |
| 932 | Löschung | `welcher` | `*(nicht da)*` | …halux valgus und [___] fuß war das… |
| 933 | Löschung | `fuß` | `*(nicht da)*` | …valgus und welcher [___] war das der… |
| 934 | Löschung | `war` | `*(nicht da)*` | …und welcher fuß [___] das der rechte… |
| 935 | Löschung | `das` | `*(nicht da)*` | …welcher fuß war [___] der rechte fuß… |
| 936 | Löschung | `der` | `*(nicht da)*` | …fuß war das [___] rechte fuß der… |
| 937 | Löschung | `rechte` | `*(nicht da)*` | …war das der [___] fuß der rechte… |
| 938 | Löschung | `fuß` | `*(nicht da)*` | …das der rechte [___] der rechte fuß… |
| 939 | Löschung | `der` | `*(nicht da)*` | …der rechte fuß [___] rechte fuß sind… |
| 940 | Löschung | `rechte` | `*(nicht da)*` | …rechte fuß der [___] fuß sind irgendwelche… |
| 941 | Löschung | `fuß` | `*(nicht da)*` | …fuß der rechte [___] sind irgendwelche komplikationen… |
| 942 | Löschung | `sind` | `*(nicht da)*` | …der rechte fuß [___] irgendwelche komplikationen während… |
| 943 | Löschung | `irgendwelche` | `*(nicht da)*` | …rechte fuß sind [___] komplikationen während oder… |
| 944 | Löschung | `komplikationen` | `*(nicht da)*` | …fuß sind irgendwelche [___] während oder nach… |
| 945 | Löschung | `während` | `*(nicht da)*` | …sind irgendwelche komplikationen [___] oder nach der… |
| 946 | Löschung | `oder` | `*(nicht da)*` | …irgendwelche komplikationen während [___] nach der operation… |
| 947 | Löschung | `nach` | `*(nicht da)*` | …komplikationen während oder [___] der operation aufgetreten… |
| 948 | Löschung | `der` | `*(nicht da)*` | …während oder nach [___] operation aufgetreten nein… |
| 949 | Löschung | `operation` | `*(nicht da)*` | …oder nach der [___] aufgetreten nein zum… |
| 950 | Löschung | `aufgetreten` | `*(nicht da)*` | …nach der operation [___] nein zum glück… |
| 951 | Löschung | `nein` | `*(nicht da)*` | …der operation aufgetreten [___] zum glück nicht… |
| 952 | Löschung | `zum` | `*(nicht da)*` | …operation aufgetreten nein [___] glück nicht nein… |
| 953 | Löschung | `glück` | `*(nicht da)*` | …aufgetreten nein zum [___] nicht nein sehr… |
| 954 | Löschung | `nicht` | `*(nicht da)*` | …nein zum glück [___] nein sehr gut… |
| 955 | Löschung | `nein` | `*(nicht da)*` | …zum glück nicht [___] sehr gut ich… |
| 956 | Löschung | `sehr` | `*(nicht da)*` | …glück nicht nein [___] gut ich konnte… |
| 957 | Löschung | `gut` | `*(nicht da)*` | …nicht nein sehr [___] ich konnte ganz… |
| 958 | Löschung | `ich` | `*(nicht da)*` | …nein sehr gut [___] konnte ganz bald… |
| 959 | Löschung | `konnte` | `*(nicht da)*` | …sehr gut ich [___] ganz bald wieder… |
| 960 | Löschung | `ganz` | `*(nicht da)*` | …gut ich konnte [___] bald wieder meine… |
| 961 | Löschung | `bald` | `*(nicht da)*` | …ich konnte ganz [___] wieder meine hohen… |
| 962 | Löschung | `wieder` | `*(nicht da)*` | …konnte ganz bald [___] meine hohen schuhe… |
| 963 | Löschung | `meine` | `*(nicht da)*` | …ganz bald wieder [___] hohen schuhe tragen… |
| 964 | Löschung | `hohen` | `*(nicht da)*` | …bald wieder meine [___] schuhe tragen perfekt… |
| 965 | Löschung | `schuhe` | `*(nicht da)*` | …wieder meine hohen [___] tragen perfekt dann… |
| 966 | Löschung | `tragen` | `*(nicht da)*` | …meine hohen schuhe [___] perfekt dann ist… |
| 967 | Löschung | `perfekt` | `*(nicht da)*` | …hohen schuhe tragen [___] dann ist wirklich… |
| 968 | Löschung | `dann` | `*(nicht da)*` | …schuhe tragen perfekt [___] ist wirklich alles… |
| 969 | Löschung | `ist` | `*(nicht da)*` | …tragen perfekt dann [___] wirklich alles gut… |
| 970 | Löschung | `wirklich` | `*(nicht da)*` | …perfekt dann ist [___] alles gut gelaufen… |
| 971 | Löschung | `alles` | `*(nicht da)*` | …dann ist wirklich [___] gut gelaufen frau… |
| 972 | Löschung | `gut` | `*(nicht da)*` | …ist wirklich alles [___] gelaufen frau beckenwestfalen… |
| 973 | Löschung | `gelaufen` | `*(nicht da)*` | …wirklich alles gut [___] frau beckenwestfalen nehmen… |
| 974 | Löschung | `frau` | `*(nicht da)*` | …alles gut gelaufen [___] beckenwestfalen nehmen sie… |
| 975 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …gut gelaufen frau [___] nehmen sie regelmäßig… |
| 976 | Löschung | `nehmen` | `*(nicht da)*` | …gelaufen frau beckenwestfalen [___] sie regelmäßig oder… |
| 977 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen nehmen [___] regelmäßig oder bei… |
| 978 | Löschung | `regelmäßig` | `*(nicht da)*` | …beckenwestfalen nehmen sie [___] oder bei bedarf… |
| 979 | Löschung | `oder` | `*(nicht da)*` | …nehmen sie regelmäßig [___] bei bedarf medikamente… |
| 980 | Löschung | `bei` | `*(nicht da)*` | …sie regelmäßig oder [___] bedarf medikamente ein… |
| 981 | Löschung | `bedarf` | `*(nicht da)*` | …regelmäßig oder bei [___] medikamente ein ich… |
| 982 | Löschung | `medikamente` | `*(nicht da)*` | …oder bei bedarf [___] ein ich nehme… |
| 983 | Löschung | `ein` | `*(nicht da)*` | …bei bedarf medikamente [___] ich nehme gelegentlich… |
| 984 | Löschung | `ich` | `*(nicht da)*` | …bedarf medikamente ein [___] nehme gelegentlich ein… |
| 985 | Löschung | `nehme` | `*(nicht da)*` | …medikamente ein ich [___] gelegentlich ein ibuprofen… |
| 986 | Löschung | `gelegentlich` | `*(nicht da)*` | …ein ich nehme [___] ein ibuprofen wenn… |
| 987 | Löschung | `ein` | `*(nicht da)*` | …ich nehme gelegentlich [___] ibuprofen wenn ich… |
| 988 | Löschung | `ibuprofen` | `*(nicht da)*` | …nehme gelegentlich ein [___] wenn ich kopfschmerzen… |
| 989 | Löschung | `wenn` | `*(nicht da)*` | …gelegentlich ein ibuprofen [___] ich kopfschmerzen habe… |
| 990 | Löschung | `ich` | `*(nicht da)*` | …ein ibuprofen wenn [___] kopfschmerzen habe und… |
| 991 | Löschung | `kopfschmerzen` | `*(nicht da)*` | …ibuprofen wenn ich [___] habe und ansonsten… |
| 992 | Löschung | `habe` | `*(nicht da)*` | …wenn ich kopfschmerzen [___] und ansonsten nehme… |
| 993 | Löschung | `und` | `*(nicht da)*` | …ich kopfschmerzen habe [___] ansonsten nehme ich… |
| 994 | Löschung | `ansonsten` | `*(nicht da)*` | …kopfschmerzen habe und [___] nehme ich die… |
| 995 | Löschung | `nehme` | `*(nicht da)*` | …habe und ansonsten [___] ich die pille… |
| 996 | Löschung | `ich` | `*(nicht da)*` | …und ansonsten nehme [___] die pille die… |
| 997 | Löschung | `die` | `*(nicht da)*` | …ansonsten nehme ich [___] pille die pille… |
| 998 | Löschung | `pille` | `*(nicht da)*` | …nehme ich die [___] die pille seit… |
| 999 | Löschung | `die` | `*(nicht da)*` | …ich die pille [___] pille seit wann… |
| 1000 | Löschung | `pille` | `*(nicht da)*` | …die pille die [___] seit wann nehmen… |
| 1001 | Löschung | `seit` | `*(nicht da)*` | …pille die pille [___] wann nehmen sie… |
| 1002 | Löschung | `wann` | `*(nicht da)*` | …die pille seit [___] nehmen sie die… |
| 1003 | Löschung | `nehmen` | `*(nicht da)*` | …pille seit wann [___] sie die pille… |
| 1004 | Löschung | `sie` | `*(nicht da)*` | …seit wann nehmen [___] die pille sieben… |
| 1005 | Löschung | `die` | `*(nicht da)*` | …wann nehmen sie [___] pille sieben oder… |
| 1006 | Löschung | `pille` | `*(nicht da)*` | …nehmen sie die [___] sieben oder acht… |
| 1007 | Löschung | `sieben` | `*(nicht da)*` | …sie die pille [___] oder acht jahren… |
| 1008 | Löschung | `oder` | `*(nicht da)*` | …die pille sieben [___] acht jahren okay… |
| 1009 | Löschung | `acht` | `*(nicht da)*` | …pille sieben oder [___] jahren okay die… |
| 1010 | Löschung | `jahren` | `*(nicht da)*` | …sieben oder acht [___] okay die ibuprofen… |
| 1011 | Löschung | `okay` | `*(nicht da)*` | …oder acht jahren [___] die ibuprofen wenn… |
| 1012 | Löschung | `die` | `*(nicht da)*` | …acht jahren okay [___] ibuprofen wenn sie… |
| 1013 | Löschung | `ibuprofen` | `*(nicht da)*` | …jahren okay die [___] wenn sie kopfschmerzen… |
| 1014 | Löschung | `wenn` | `*(nicht da)*` | …okay die ibuprofen [___] sie kopfschmerzen haben… |
| 1015 | Löschung | `sie` | `*(nicht da)*` | …die ibuprofen wenn [___] kopfschmerzen haben wie… |
| 1016 | Löschung | `kopfschmerzen` | `*(nicht da)*` | …ibuprofen wenn sie [___] haben wie viele… |
| 1017 | Löschung | `haben` | `*(nicht da)*` | …wenn sie kopfschmerzen [___] wie viele milligramm… |
| 1018 | Löschung | `wie` | `*(nicht da)*` | …sie kopfschmerzen haben [___] viele milligramm sind… |
| 1019 | Löschung | `viele` | `*(nicht da)*` | …kopfschmerzen haben wie [___] milligramm sind das… |
| 1020 | Löschung | `milligramm` | `*(nicht da)*` | …haben wie viele [___] sind das 400… |
| 1021 | Löschung | `sind` | `*(nicht da)*` | …wie viele milligramm [___] das 400 600… |
| 1022 | Löschung | `das` | `*(nicht da)*` | …viele milligramm sind [___] 400 600 800… |
| 1023 | Löschung | `400` | `*(nicht da)*` | …milligramm sind das [___] 600 800 also… |
| 1024 | Löschung | `600` | `*(nicht da)*` | …sind das 400 [___] 800 also meistens… |
| 1025 | Löschung | `800` | `*(nicht da)*` | …das 400 600 [___] also meistens das… |
| 1026 | Löschung | `also` | `*(nicht da)*` | …400 600 800 [___] meistens das was… |
| 1027 | Löschung | `meistens` | `*(nicht da)*` | …600 800 also [___] das was ich… |
| 1028 | Löschung | `das` | `*(nicht da)*` | …800 also meistens [___] was ich gerade… |
| 1029 | Löschung | `was` | `*(nicht da)*` | …also meistens das [___] ich gerade zu… |
| 1030 | Löschung | `ich` | `*(nicht da)*` | …meistens das was [___] gerade zu hause… |
| 1031 | Löschung | `gerade` | `*(nicht da)*` | …das was ich [___] zu hause habe… |
| 1032 | Löschung | `zu` | `*(nicht da)*` | …was ich gerade [___] hause habe aber… |
| 1033 | Löschung | `hause` | `*(nicht da)*` | …ich gerade zu [___] habe aber ich… |
| 1034 | Löschung | `habe` | `*(nicht da)*` | …gerade zu hause [___] aber ich glaube… |
| 1035 | Löschung | `aber` | `*(nicht da)*` | …zu hause habe [___] ich glaube 600… |
| 1036 | Löschung | `ich` | `*(nicht da)*` | …hause habe aber [___] glaube 600 600… |
| 1037 | Löschung | `glaube` | `*(nicht da)*` | …habe aber ich [___] 600 600 ja… |
| 1038 | Löschung | `600` | `*(nicht da)*` | …aber ich glaube [___] 600 ja alles… |
| 1039 | Löschung | `600` | `*(nicht da)*` | …ich glaube 600 [___] ja alles klar… |
| 1040 | Löschung | `ja` | `*(nicht da)*` | …glaube 600 600 [___] alles klar sind… |
| 1041 | Löschung | `alles` | `*(nicht da)*` | …600 600 ja [___] klar sind sie… |
| 1042 | Löschung | `klar` | `*(nicht da)*` | …600 ja alles [___] sind sie geimpft… |
| 1043 | Löschung | `sind` | `*(nicht da)*` | …ja alles klar [___] sie geimpft ich… |
| 1044 | Löschung | `sie` | `*(nicht da)*` | …alles klar sind [___] geimpft ich bin… |
| 1045 | Löschung | `geimpft` | `*(nicht da)*` | …klar sind sie [___] ich bin geimpft… |
| 1046 | Löschung | `ich` | `*(nicht da)*` | …sind sie geimpft [___] bin geimpft ja… |
| 1047 | Löschung | `bin` | `*(nicht da)*` | …sie geimpft ich [___] geimpft ja haben… |
| 1048 | Löschung | `geimpft` | `*(nicht da)*` | …geimpft ich bin [___] ja haben sie… |
| 1049 | Löschung | `ja` | `*(nicht da)*` | …ich bin geimpft [___] haben sie ganz… |
| 1050 | Löschung | `haben` | `*(nicht da)*` | …bin geimpft ja [___] sie ganz zufällig… |
| 1051 | Löschung | `sie` | `*(nicht da)*` | …geimpft ja haben [___] ganz zufällig ihren… |
| 1052 | Löschung | `ganz` | `*(nicht da)*` | …ja haben sie [___] zufällig ihren impfpass… |
| 1053 | Löschung | `zufällig` | `*(nicht da)*` | …haben sie ganz [___] ihren impfpass dabei… |
| 1054 | Löschung | `ihren` | `*(nicht da)*` | …sie ganz zufällig [___] impfpass dabei oh… |
| 1055 | Löschung | `impfpass` | `*(nicht da)*` | …ganz zufällig ihren [___] dabei oh leider… |
| 1056 | Löschung | `dabei` | `*(nicht da)*` | …zufällig ihren impfpass [___] oh leider nein… |
| 1057 | Löschung | `oh` | `*(nicht da)*` | …ihren impfpass dabei [___] leider nein eher… |
| 1058 | Löschung | `leider` | `*(nicht da)*` | …impfpass dabei oh [___] nein eher nicht… |
| 1059 | Löschung | `nein` | `*(nicht da)*` | …dabei oh leider [___] eher nicht hätte… |
| 1060 | Löschung | `eher` | `*(nicht da)*` | …oh leider nein [___] nicht hätte ich… |
| 1061 | Löschung | `nicht` | `*(nicht da)*` | …leider nein eher [___] hätte ich gewusst… |
| 1062 | Löschung | `hätte` | `*(nicht da)*` | …nein eher nicht [___] ich gewusst dass… |
| 1063 | Löschung | `ich` | `*(nicht da)*` | …eher nicht hätte [___] gewusst dass ich… |
| 1064 | Löschung | `gewusst` | `*(nicht da)*` | …nicht hätte ich [___] dass ich ins… |
| 1065 | Löschung | `dass` | `*(nicht da)*` | …hätte ich gewusst [___] ich ins krankenhaus… |
| 1066 | Löschung | `ich` | `*(nicht da)*` | …ich gewusst dass [___] ins krankenhaus muss… |
| 1067 | Löschung | `ins` | `*(nicht da)*` | …gewusst dass ich [___] krankenhaus muss ja… |
| 1068 | Löschung | `krankenhaus` | `*(nicht da)*` | …dass ich ins [___] muss ja ich… |
| 1069 | Löschung | `muss` | `*(nicht da)*` | …ich ins krankenhaus [___] ja ich muss… |
| 1070 | Löschung | `ja` | `*(nicht da)*` | …ins krankenhaus muss [___] ich muss auch… |
| 1071 | Löschung | `ich` | `*(nicht da)*` | …krankenhaus muss ja [___] muss auch wissen… |
| 1072 | Löschung | `muss` | `*(nicht da)*` | …muss ja ich [___] auch wissen dass… |
| 1073 | Löschung | `auch` | `*(nicht da)*` | …ja ich muss [___] wissen dass ich… |
| 1074 | Löschung | `wissen` | `*(nicht da)*` | …ich muss auch [___] dass ich den… |
| 1075 | Löschung | `dass` | `*(nicht da)*` | …muss auch wissen [___] ich den nicht… |
| 1076 | Löschung | `ich` | `*(nicht da)*` | …auch wissen dass [___] den nicht bei… |
| 1077 | Substitution | `den` | `einem` | …wissen dass ich [___] nicht bei mir… |
| 1078 | Substitution | `nicht` | `fahrradunfall` | …dass ich den [___] bei mir trage… |
| 1079 | Löschung | `mir` | `*(nicht da)*` | …den nicht bei [___] trage sehr gut… |
| 1080 | Löschung | `trage` | `*(nicht da)*` | …nicht bei mir [___] sehr gut okay… |
| 1081 | Löschung | `sehr` | `*(nicht da)*` | …bei mir trage [___] gut okay frau… |
| 1082 | Löschung | `gut` | `*(nicht da)*` | …mir trage sehr [___] okay frau beckenwestfalen… |
| 1083 | Löschung | `okay` | `*(nicht da)*` | …trage sehr gut [___] frau beckenwestfalen wie… |
| 1084 | Löschung | `frau` | `*(nicht da)*` | …sehr gut okay [___] beckenwestfalen wie geht… |
| 1085 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …gut okay frau [___] wie geht es… |
| 1086 | Löschung | `wie` | `*(nicht da)*` | …okay frau beckenwestfalen [___] geht es ihnen… |
| 1087 | Löschung | `geht` | `*(nicht da)*` | …frau beckenwestfalen wie [___] es ihnen denn… |
| 1088 | Löschung | `es` | `*(nicht da)*` | …beckenwestfalen wie geht [___] ihnen denn sonst… |
| 1089 | Löschung | `ihnen` | `*(nicht da)*` | …wie geht es [___] denn sonst körperlich… |
| 1090 | Löschung | `denn` | `*(nicht da)*` | …geht es ihnen [___] sonst körperlich haben… |
| 1091 | Löschung | `sonst` | `*(nicht da)*` | …es ihnen denn [___] körperlich haben sie… |
| 1092 | Löschung | `körperlich` | `*(nicht da)*` | …ihnen denn sonst [___] haben sie in… |
| 1093 | Löschung | `haben` | `*(nicht da)*` | …denn sonst körperlich [___] sie in letzter… |
| 1094 | Löschung | `sie` | `*(nicht da)*` | …sonst körperlich haben [___] in letzter zeit… |
| 1095 | Löschung | `in` | `*(nicht da)*` | …körperlich haben sie [___] letzter zeit fieber… |
| 1096 | Löschung | `letzter` | `*(nicht da)*` | …haben sie in [___] zeit fieber gehabt… |
| 1097 | Löschung | `zeit` | `*(nicht da)*` | …sie in letzter [___] fieber gehabt oder… |
| 1098 | Löschung | `fieber` | `*(nicht da)*` | …in letzter zeit [___] gehabt oder schüttelfrost… |
| 1099 | Löschung | `gehabt` | `*(nicht da)*` | …letzter zeit fieber [___] oder schüttelfrost oder… |
| 1100 | Löschung | `oder` | `*(nicht da)*` | …zeit fieber gehabt [___] schüttelfrost oder nachtschweiß… |
| 1101 | Löschung | `schüttelfrost` | `*(nicht da)*` | …fieber gehabt oder [___] oder nachtschweiß oder… |
| 1102 | Löschung | `oder` | `*(nicht da)*` | …gehabt oder schüttelfrost [___] nachtschweiß oder fühlen… |
| 1103 | Löschung | `nachtschweiß` | `*(nicht da)*` | …oder schüttelfrost oder [___] oder fühlen sie… |
| 1104 | Löschung | `oder` | `*(nicht da)*` | …schüttelfrost oder nachtschweiß [___] fühlen sie sich… |
| 1105 | Löschung | `fühlen` | `*(nicht da)*` | …oder nachtschweiß oder [___] sie sich irgendwie… |
| 1106 | Löschung | `sie` | `*(nicht da)*` | …nachtschweiß oder fühlen [___] sich irgendwie ungut… |
| 1107 | Löschung | `sich` | `*(nicht da)*` | …oder fühlen sie [___] irgendwie ungut in… |
| 1108 | Löschung | `irgendwie` | `*(nicht da)*` | …fühlen sie sich [___] ungut in letzter… |
| 1109 | Löschung | `ungut` | `*(nicht da)*` | …sie sich irgendwie [___] in letzter zeit… |
| 1110 | Löschung | `in` | `*(nicht da)*` | …sich irgendwie ungut [___] letzter zeit nein… |
| 1111 | Löschung | `letzter` | `*(nicht da)*` | …irgendwie ungut in [___] zeit nein ich… |
| 1112 | Löschung | `zeit` | `*(nicht da)*` | …ungut in letzter [___] nein ich habe… |
| 1113 | Löschung | `nein` | `*(nicht da)*` | …in letzter zeit [___] ich habe gar… |
| 1114 | Löschung | `ich` | `*(nicht da)*` | …letzter zeit nein [___] habe gar keine… |
| 1115 | Löschung | `habe` | `*(nicht da)*` | …zeit nein ich [___] gar keine sonstigen… |
| 1116 | Löschung | `gar` | `*(nicht da)*` | …nein ich habe [___] keine sonstigen gesundheitlichen… |
| 1117 | Löschung | `keine` | `*(nicht da)*` | …ich habe gar [___] sonstigen gesundheitlichen probleme… |
| 1118 | Löschung | `sonstigen` | `*(nicht da)*` | …habe gar keine [___] gesundheitlichen probleme ich… |
| 1119 | Löschung | `gesundheitlichen` | `*(nicht da)*` | …gar keine sonstigen [___] probleme ich habe… |
| 1120 | Löschung | `probleme` | `*(nicht da)*` | …keine sonstigen gesundheitlichen [___] ich habe manchmal… |
| 1121 | Löschung | `ich` | `*(nicht da)*` | …sonstigen gesundheitlichen probleme [___] habe manchmal schwierigkeiten… |
| 1122 | Löschung | `habe` | `*(nicht da)*` | …gesundheitlichen probleme ich [___] manchmal schwierigkeiten beim… |
| 1123 | Löschung | `manchmal` | `*(nicht da)*` | …probleme ich habe [___] schwierigkeiten beim einschlafen… |
| 1124 | Löschung | `schwierigkeiten` | `*(nicht da)*` | …ich habe manchmal [___] beim einschlafen aber… |
| 1125 | Löschung | `beim` | `*(nicht da)*` | …habe manchmal schwierigkeiten [___] einschlafen aber das… |
| 1126 | Löschung | `einschlafen` | `*(nicht da)*` | …manchmal schwierigkeiten beim [___] aber das ist… |
| 1127 | Löschung | `aber` | `*(nicht da)*` | …schwierigkeiten beim einschlafen [___] das ist oft… |
| 1128 | Löschung | `das` | `*(nicht da)*` | …beim einschlafen aber [___] ist oft der… |
| 1129 | Löschung | `ist` | `*(nicht da)*` | …einschlafen aber das [___] oft der fall… |
| 1130 | Löschung | `oft` | `*(nicht da)*` | …aber das ist [___] der fall wenn… |
| 1131 | Löschung | `der` | `*(nicht da)*` | …das ist oft [___] fall wenn ich… |
| 1132 | Löschung | `fall` | `*(nicht da)*` | …ist oft der [___] wenn ich auf… |
| 1133 | Löschung | `wenn` | `*(nicht da)*` | …oft der fall [___] ich auf der… |
| 1134 | Löschung | `ich` | `*(nicht da)*` | …der fall wenn [___] auf der arbeit… |
| 1135 | Löschung | `auf` | `*(nicht da)*` | …fall wenn ich [___] der arbeit viel… |
| 1136 | Löschung | `der` | `*(nicht da)*` | …wenn ich auf [___] arbeit viel zu… |
| 1137 | Löschung | `arbeit` | `*(nicht da)*` | …ich auf der [___] viel zu tun… |
| 1138 | Löschung | `viel` | `*(nicht da)*` | …auf der arbeit [___] zu tun habe… |
| 1139 | Löschung | `zu` | `*(nicht da)*` | …der arbeit viel [___] tun habe oder… |
| 1140 | Löschung | `tun` | `*(nicht da)*` | …arbeit viel zu [___] habe oder zu… |
| 1141 | Löschung | `habe` | `*(nicht da)*` | …viel zu tun [___] oder zu viel… |
| 1142 | Löschung | `oder` | `*(nicht da)*` | …zu tun habe [___] zu viel nachdenke… |
| 1143 | Löschung | `zu` | `*(nicht da)*` | …tun habe oder [___] viel nachdenke also… |
| 1144 | Löschung | `viel` | `*(nicht da)*` | …habe oder zu [___] nachdenke also nichts… |
| 1145 | Löschung | `nachdenke` | `*(nicht da)*` | …oder zu viel [___] also nichts worüber… |
| 1146 | Löschung | `also` | `*(nicht da)*` | …zu viel nachdenke [___] nichts worüber ich… |
| 1147 | Löschung | `nichts` | `*(nicht da)*` | …viel nachdenke also [___] worüber ich mir… |
| 1148 | Löschung | `worüber` | `*(nicht da)*` | …nachdenke also nichts [___] ich mir bis… |
| 1149 | Löschung | `ich` | `*(nicht da)*` | …also nichts worüber [___] mir bis jetzt… |
| 1150 | Löschung | `mir` | `*(nicht da)*` | …nichts worüber ich [___] bis jetzt sorgen… |
| 1151 | Löschung | `bis` | `*(nicht da)*` | …worüber ich mir [___] jetzt sorgen gemacht… |
| 1152 | Löschung | `jetzt` | `*(nicht da)*` | …ich mir bis [___] sorgen gemacht habe… |
| 1153 | Löschung | `sorgen` | `*(nicht da)*` | …mir bis jetzt [___] gemacht habe okay… |
| 1154 | Löschung | `gemacht` | `*(nicht da)*` | …bis jetzt sorgen [___] habe okay prima… |
| 1155 | Löschung | `habe` | `*(nicht da)*` | …jetzt sorgen gemacht [___] okay prima ich… |
| 1156 | Löschung | `okay` | `*(nicht da)*` | …sorgen gemacht habe [___] prima ich glaube… |
| 1157 | Löschung | `prima` | `*(nicht da)*` | …gemacht habe okay [___] ich glaube das… |
| 1158 | Löschung | `ich` | `*(nicht da)*` | …habe okay prima [___] glaube das kennen… |
| 1159 | Löschung | `glaube` | `*(nicht da)*` | …okay prima ich [___] das kennen wir… |
| 1160 | Löschung | `das` | `*(nicht da)*` | …prima ich glaube [___] kennen wir auch… |
| 1161 | Löschung | `kennen` | `*(nicht da)*` | …ich glaube das [___] wir auch wirklich… |
| 1162 | Löschung | `wir` | `*(nicht da)*` | …glaube das kennen [___] auch wirklich alle… |
| 1163 | Löschung | `auch` | `*(nicht da)*` | …das kennen wir [___] wirklich alle ja… |
| 1164 | Löschung | `wirklich` | `*(nicht da)*` | …kennen wir auch [___] alle ja wie… |
| 1165 | Löschung | `alle` | `*(nicht da)*` | …wir auch wirklich [___] ja wie sieht… |
| 1166 | Löschung | `ja` | `*(nicht da)*` | …auch wirklich alle [___] wie sieht es… |
| 1167 | Löschung | `wie` | `*(nicht da)*` | …wirklich alle ja [___] sieht es denn… |
| 1168 | Löschung | `sieht` | `*(nicht da)*` | …alle ja wie [___] es denn aus… |
| 1169 | Löschung | `es` | `*(nicht da)*` | …ja wie sieht [___] denn aus mit… |
| 1170 | Löschung | `denn` | `*(nicht da)*` | …wie sieht es [___] aus mit ihrer… |
| 1171 | Löschung | `aus` | `*(nicht da)*` | …sieht es denn [___] mit ihrer periode… |
| 1172 | Löschung | `mit` | `*(nicht da)*` | …es denn aus [___] ihrer periode bekommen… |
| 1173 | Löschung | `ihrer` | `*(nicht da)*` | …denn aus mit [___] periode bekommen sie… |
| 1174 | Löschung | `periode` | `*(nicht da)*` | …aus mit ihrer [___] bekommen sie die… |
| 1175 | Löschung | `bekommen` | `*(nicht da)*` | …mit ihrer periode [___] sie die regelmäßig… |
| 1176 | Löschung | `sie` | `*(nicht da)*` | …ihrer periode bekommen [___] die regelmäßig ich… |
| 1177 | Löschung | `die` | `*(nicht da)*` | …periode bekommen sie [___] regelmäßig ich bekomme… |
| 1178 | Löschung | `regelmäßig` | `*(nicht da)*` | …bekommen sie die [___] ich bekomme sie… |
| 1179 | Löschung | `ich` | `*(nicht da)*` | …sie die regelmäßig [___] bekomme sie regelmäßig… |
| 1180 | Löschung | `bekomme` | `*(nicht da)*` | …die regelmäßig ich [___] sie regelmäßig ja… |
| 1181 | Löschung | `sie` | `*(nicht da)*` | …regelmäßig ich bekomme [___] regelmäßig ja seitdem… |
| 1182 | Löschung | `regelmäßig` | `*(nicht da)*` | …ich bekomme sie [___] ja seitdem ich… |
| 1183 | Löschung | `ja` | `*(nicht da)*` | …bekomme sie regelmäßig [___] seitdem ich die… |
| 1184 | Löschung | `seitdem` | `*(nicht da)*` | …sie regelmäßig ja [___] ich die pille… |
| 1185 | Löschung | `ich` | `*(nicht da)*` | …regelmäßig ja seitdem [___] die pille nehme… |
| 1186 | Löschung | `die` | `*(nicht da)*` | …ja seitdem ich [___] pille nehme bekomme… |
| 1187 | Löschung | `pille` | `*(nicht da)*` | …seitdem ich die [___] nehme bekomme ich… |
| 1188 | Löschung | `nehme` | `*(nicht da)*` | …ich die pille [___] bekomme ich sie… |
| 1189 | Löschung | `bekomme` | `*(nicht da)*` | …die pille nehme [___] ich sie ganz… |
| 1190 | Löschung | `ich` | `*(nicht da)*` | …pille nehme bekomme [___] sie ganz regelmäßig… |
| 1191 | Löschung | `sie` | `*(nicht da)*` | …nehme bekomme ich [___] ganz regelmäßig okay… |
| 1192 | Löschung | `ganz` | `*(nicht da)*` | …bekomme ich sie [___] regelmäßig okay wunderbar… |
| 1193 | Löschung | `regelmäßig` | `*(nicht da)*` | …ich sie ganz [___] okay wunderbar frau… |
| 1194 | Löschung | `okay` | `*(nicht da)*` | …sie ganz regelmäßig [___] wunderbar frau beckenwestfalen… |
| 1195 | Löschung | `wunderbar` | `*(nicht da)*` | …ganz regelmäßig okay [___] frau beckenwestfalen rauchen… |
| 1196 | Löschung | `frau` | `*(nicht da)*` | …regelmäßig okay wunderbar [___] beckenwestfalen rauchen sie… |
| 1197 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …okay wunderbar frau [___] rauchen sie nein… |
| 1198 | Löschung | `rauchen` | `*(nicht da)*` | …wunderbar frau beckenwestfalen [___] sie nein ich… |
| 1199 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen rauchen [___] nein ich habe… |
| 1200 | Löschung | `nein` | `*(nicht da)*` | …beckenwestfalen rauchen sie [___] ich habe früher… |
| 1201 | Löschung | `ich` | `*(nicht da)*` | …rauchen sie nein [___] habe früher geraucht… |
| 1202 | Löschung | `habe` | `*(nicht da)*` | …sie nein ich [___] früher geraucht falls… |
| 1203 | Löschung | `früher` | `*(nicht da)*` | …nein ich habe [___] geraucht falls das… |
| 1204 | Löschung | `geraucht` | `*(nicht da)*` | …ich habe früher [___] falls das relevant… |
| 1205 | Löschung | `falls` | `*(nicht da)*` | …habe früher geraucht [___] das relevant ist… |
| 1206 | Löschung | `das` | `*(nicht da)*` | …früher geraucht falls [___] relevant ist ja… |
| 1207 | Löschung | `relevant` | `*(nicht da)*` | …geraucht falls das [___] ist ja wie… |
| 1208 | Löschung | `ist` | `*(nicht da)*` | …falls das relevant [___] ja wie lange… |
| 1209 | Löschung | `ja` | `*(nicht da)*` | …das relevant ist [___] wie lange haben… |
| 1210 | Löschung | `wie` | `*(nicht da)*` | …relevant ist ja [___] lange haben sie… |
| 1211 | Löschung | `lange` | `*(nicht da)*` | …ist ja wie [___] haben sie aufgehört… |
| 1212 | Löschung | `haben` | `*(nicht da)*` | …ja wie lange [___] sie aufgehört ach… |
| 1213 | Löschung | `sie` | `*(nicht da)*` | …wie lange haben [___] aufgehört ach das… |
| 1214 | Löschung | `aufgehört` | `*(nicht da)*` | …lange haben sie [___] ach das müssten… |
| 1215 | Löschung | `ach` | `*(nicht da)*` | …haben sie aufgehört [___] das müssten jetzt… |
| 1216 | Löschung | `das` | `*(nicht da)*` | …sie aufgehört ach [___] müssten jetzt schon… |
| 1217 | Löschung | `müssten` | `*(nicht da)*` | …aufgehört ach das [___] jetzt schon acht… |
| 1218 | Löschung | `jetzt` | `*(nicht da)*` | …ach das müssten [___] schon acht jahre… |
| 1219 | Löschung | `schon` | `*(nicht da)*` | …das müssten jetzt [___] acht jahre sein… |
| 1220 | Löschung | `acht` | `*(nicht da)*` | …müssten jetzt schon [___] jahre sein seitdem… |
| 1221 | Löschung | `jahre` | `*(nicht da)*` | …jetzt schon acht [___] sein seitdem ich… |
| 1222 | Löschung | `sein` | `*(nicht da)*` | …schon acht jahre [___] seitdem ich aufgehört… |
| 1223 | Löschung | `seitdem` | `*(nicht da)*` | …acht jahre sein [___] ich aufgehört habe… |
| 1224 | Löschung | `ich` | `*(nicht da)*` | …jahre sein seitdem [___] aufgehört habe zum… |
| 1225 | Löschung | `aufgehört` | `*(nicht da)*` | …sein seitdem ich [___] habe zum glück… |
| 1226 | Löschung | `habe` | `*(nicht da)*` | …seitdem ich aufgehört [___] zum glück und… |
| 1227 | Löschung | `zum` | `*(nicht da)*` | …ich aufgehört habe [___] glück und wie… |
| 1228 | Löschung | `glück` | `*(nicht da)*` | …aufgehört habe zum [___] und wie lange… |
| 1229 | Löschung | `und` | `*(nicht da)*` | …habe zum glück [___] wie lange haben… |
| 1230 | Löschung | `wie` | `*(nicht da)*` | …zum glück und [___] lange haben sie… |
| 1231 | Löschung | `lange` | `*(nicht da)*` | …glück und wie [___] haben sie geraucht… |
| 1232 | Löschung | `haben` | `*(nicht da)*` | …und wie lange [___] sie geraucht damals… |
| 1233 | Löschung | `sie` | `*(nicht da)*` | …wie lange haben [___] geraucht damals sechs… |
| 1234 | Löschung | `geraucht` | `*(nicht da)*` | …lange haben sie [___] damals sechs sieben… |
| 1235 | Löschung | `damals` | `*(nicht da)*` | …haben sie geraucht [___] sechs sieben jahre… |
| 1236 | Löschung | `sechs` | `*(nicht da)*` | …sie geraucht damals [___] sieben jahre sechs… |
| 1237 | Löschung | `sieben` | `*(nicht da)*` | …geraucht damals sechs [___] jahre sechs sieben… |
| 1238 | Löschung | `jahre` | `*(nicht da)*` | …damals sechs sieben [___] sechs sieben jahre… |
| 1239 | Löschung | `sechs` | `*(nicht da)*` | …sechs sieben jahre [___] sieben jahre okay… |
| 1240 | Löschung | `sieben` | `*(nicht da)*` | …sieben jahre sechs [___] jahre okay gut… |
| 1241 | Löschung | `jahre` | `*(nicht da)*` | …jahre sechs sieben [___] okay gut trinken… |
| 1242 | Löschung | `okay` | `*(nicht da)*` | …sechs sieben jahre [___] gut trinken sie… |
| 1243 | Löschung | `gut` | `*(nicht da)*` | …sieben jahre okay [___] trinken sie alkohol… |
| 1244 | Löschung | `trinken` | `*(nicht da)*` | …jahre okay gut [___] sie alkohol ja… |
| 1245 | Löschung | `sie` | `*(nicht da)*` | …okay gut trinken [___] alkohol ja nicht… |
| 1246 | Löschung | `alkohol` | `*(nicht da)*` | …gut trinken sie [___] ja nicht viel… |
| 1247 | Löschung | `ja` | `*(nicht da)*` | …trinken sie alkohol [___] nicht viel aber… |
| 1248 | Löschung | `nicht` | `*(nicht da)*` | …sie alkohol ja [___] viel aber schon… |
| 1249 | Löschung | `viel` | `*(nicht da)*` | …alkohol ja nicht [___] aber schon abends… |
| 1250 | Löschung | `aber` | `*(nicht da)*` | …ja nicht viel [___] schon abends nach… |
| 1251 | Löschung | `schon` | `*(nicht da)*` | …nicht viel aber [___] abends nach der… |
| 1252 | Löschung | `abends` | `*(nicht da)*` | …viel aber schon [___] nach der arbeit… |
| 1253 | Löschung | `nach` | `*(nicht da)*` | …aber schon abends [___] der arbeit gerne… |
| 1254 | Löschung | `der` | `*(nicht da)*` | …schon abends nach [___] arbeit gerne ein… |
| 1255 | Löschung | `arbeit` | `*(nicht da)*` | …abends nach der [___] gerne ein glas… |
| 1256 | Löschung | `gerne` | `*(nicht da)*` | …nach der arbeit [___] ein glas wein… |
| 1257 | Löschung | `ein` | `*(nicht da)*` | …der arbeit gerne [___] glas wein und… |
| 1258 | Löschung | `glas` | `*(nicht da)*` | …arbeit gerne ein [___] wein und am… |
| 1259 | Löschung | `wein` | `*(nicht da)*` | …gerne ein glas [___] und am wochenende… |
| 1260 | Löschung | `und` | `*(nicht da)*` | …ein glas wein [___] am wochenende wenn… |
| 1261 | Löschung | `am` | `*(nicht da)*` | …glas wein und [___] wochenende wenn wir… |
| 1262 | Löschung | `wochenende` | `*(nicht da)*` | …wein und am [___] wenn wir mit… |
| 1263 | Löschung | `wenn` | `*(nicht da)*` | …und am wochenende [___] wir mit freunden… |
| 1264 | Löschung | `wir` | `*(nicht da)*` | …am wochenende wenn [___] mit freunden unterwegs… |
| 1265 | Löschung | `mit` | `*(nicht da)*` | …wochenende wenn wir [___] freunden unterwegs sind… |
| 1266 | Löschung | `freunden` | `*(nicht da)*` | …wenn wir mit [___] unterwegs sind dann… |
| 1267 | Löschung | `unterwegs` | `*(nicht da)*` | …wir mit freunden [___] sind dann gerne… |
| 1268 | Löschung | `sind` | `*(nicht da)*` | …mit freunden unterwegs [___] dann gerne auch… |
| 1269 | Löschung | `dann` | `*(nicht da)*` | …freunden unterwegs sind [___] gerne auch zwei… |
| 1270 | Löschung | `gerne` | `*(nicht da)*` | …unterwegs sind dann [___] auch zwei oder… |
| 1271 | Löschung | `auch` | `*(nicht da)*` | …sind dann gerne [___] zwei oder drei… |
| 1272 | Löschung | `zwei` | `*(nicht da)*` | …dann gerne auch [___] oder drei gäser… |
| 1273 | Löschung | `oder` | `*(nicht da)*` | …gerne auch zwei [___] drei gäser okay… |
| 1274 | Löschung | `drei` | `*(nicht da)*` | …auch zwei oder [___] gäser okay dieses… |
| 1275 | Löschung | `gäser` | `*(nicht da)*` | …zwei oder drei [___] okay dieses gläschen… |
| 1276 | Löschung | `okay` | `*(nicht da)*` | …oder drei gäser [___] dieses gläschen wein… |
| 1277 | Löschung | `dieses` | `*(nicht da)*` | …drei gäser okay [___] gläschen wein nach… |
| 1278 | Löschung | `gläschen` | `*(nicht da)*` | …gäser okay dieses [___] wein nach der… |
| 1279 | Löschung | `wein` | `*(nicht da)*` | …okay dieses gläschen [___] nach der arbeit… |
| 1280 | Löschung | `nach` | `*(nicht da)*` | …dieses gläschen wein [___] der arbeit ist… |
| 1281 | Löschung | `der` | `*(nicht da)*` | …gläschen wein nach [___] arbeit ist das… |
| 1282 | Löschung | `arbeit` | `*(nicht da)*` | …wein nach der [___] ist das so… |
| 1283 | Löschung | `ist` | `*(nicht da)*` | …nach der arbeit [___] das so einmal… |
| 1284 | Löschung | `das` | `*(nicht da)*` | …der arbeit ist [___] so einmal die… |
| 1285 | Löschung | `so` | `*(nicht da)*` | …arbeit ist das [___] einmal die woche… |
| 1286 | Löschung | `einmal` | `*(nicht da)*` | …ist das so [___] die woche zweimal… |
| 1287 | Löschung | `die` | `*(nicht da)*` | …das so einmal [___] woche zweimal oder… |
| 1288 | Löschung | `woche` | `*(nicht da)*` | …so einmal die [___] zweimal oder doch… |
| 1289 | Löschung | `zweimal` | `*(nicht da)*` | …einmal die woche [___] oder doch öfter… |
| 1290 | Löschung | `oder` | `*(nicht da)*` | …die woche zweimal [___] doch öfter ach… |
| 1291 | Löschung | `doch` | `*(nicht da)*` | …woche zweimal oder [___] öfter ach das… |
| 1292 | Löschung | `öfter` | `*(nicht da)*` | …zweimal oder doch [___] ach das ist… |
| 1293 | Löschung | `ach` | `*(nicht da)*` | …oder doch öfter [___] das ist schon… |
| 1294 | Löschung | `das` | `*(nicht da)*` | …doch öfter ach [___] ist schon fast… |
| 1295 | Löschung | `ist` | `*(nicht da)*` | …öfter ach das [___] schon fast jeden… |
| 1296 | Löschung | `schon` | `*(nicht da)*` | …ach das ist [___] fast jeden abend… |
| 1297 | Löschung | `fast` | `*(nicht da)*` | …das ist schon [___] jeden abend aber… |
| 1298 | Löschung | `jeden` | `*(nicht da)*` | …ist schon fast [___] abend aber ein… |
| 1299 | Löschung | `abend` | `*(nicht da)*` | …schon fast jeden [___] aber ein kleines… |
| 1300 | Löschung | `aber` | `*(nicht da)*` | …fast jeden abend [___] ein kleines gläschen… |
| 1301 | Löschung | `ein` | `*(nicht da)*` | …jeden abend aber [___] kleines gläschen okay… |
| 1302 | Löschung | `kleines` | `*(nicht da)*` | …abend aber ein [___] gläschen okay wunderbar… |
| 1303 | Löschung | `gläschen` | `*(nicht da)*` | …aber ein kleines [___] okay wunderbar frau… |
| 1304 | Löschung | `okay` | `*(nicht da)*` | …ein kleines gläschen [___] wunderbar frau beckenwestfalen… |
| 1305 | Löschung | `wunderbar` | `*(nicht da)*` | …kleines gläschen okay [___] frau beckenwestfalen nehmen… |
| 1306 | Löschung | `frau` | `*(nicht da)*` | …gläschen okay wunderbar [___] beckenwestfalen nehmen sie… |
| 1307 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …okay wunderbar frau [___] nehmen sie das… |
| 1308 | Löschung | `nehmen` | `*(nicht da)*` | …wunderbar frau beckenwestfalen [___] sie das jetzt… |
| 1309 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen nehmen [___] das jetzt bitte… |
| 1310 | Löschung | `das` | `*(nicht da)*` | …beckenwestfalen nehmen sie [___] jetzt bitte nicht… |
| 1311 | Löschung | `jetzt` | `*(nicht da)*` | …nehmen sie das [___] bitte nicht persönlich… |
| 1312 | Löschung | `bitte` | `*(nicht da)*` | …sie das jetzt [___] nicht persönlich das… |
| 1313 | Löschung | `nicht` | `*(nicht da)*` | …das jetzt bitte [___] persönlich das ist… |
| 1314 | Löschung | `persönlich` | `*(nicht da)*` | …jetzt bitte nicht [___] das ist eine… |
| 1315 | Löschung | `das` | `*(nicht da)*` | …bitte nicht persönlich [___] ist eine reine… |
| 1316 | Löschung | `ist` | `*(nicht da)*` | …nicht persönlich das [___] eine reine routinefrage… |
| 1317 | Löschung | `eine` | `*(nicht da)*` | …persönlich das ist [___] reine routinefrage die… |
| 1318 | Löschung | `reine` | `*(nicht da)*` | …das ist eine [___] routinefrage die ich… |
| 1319 | Löschung | `routinefrage` | `*(nicht da)*` | …ist eine reine [___] die ich aber… |
| 1320 | Löschung | `die` | `*(nicht da)*` | …eine reine routinefrage [___] ich aber natürlich… |
| 1321 | Löschung | `ich` | `*(nicht da)*` | …reine routinefrage die [___] aber natürlich auch… |
| 1322 | Löschung | `aber` | `*(nicht da)*` | …routinefrage die ich [___] natürlich auch ihnen… |
| 1323 | Löschung | `natürlich` | `*(nicht da)*` | …die ich aber [___] auch ihnen stellen… |
| 1324 | Löschung | `auch` | `*(nicht da)*` | …ich aber natürlich [___] ihnen stellen muss… |
| 1325 | Löschung | `ihnen` | `*(nicht da)*` | …aber natürlich auch [___] stellen muss und… |
| 1326 | Löschung | `stellen` | `*(nicht da)*` | …natürlich auch ihnen [___] muss und zwar… |
| 1327 | Löschung | `muss` | `*(nicht da)*` | …auch ihnen stellen [___] und zwar nehmen… |
| 1328 | Löschung | `und` | `*(nicht da)*` | …ihnen stellen muss [___] zwar nehmen sie… |
| 1329 | Löschung | `zwar` | `*(nicht da)*` | …stellen muss und [___] nehmen sie drogen… |
| 1330 | Löschung | `nehmen` | `*(nicht da)*` | …muss und zwar [___] sie drogen nein… |
| 1331 | Löschung | `sie` | `*(nicht da)*` | …und zwar nehmen [___] drogen nein ich… |
| 1332 | Löschung | `drogen` | `*(nicht da)*` | …zwar nehmen sie [___] nein ich nehme… |
| 1333 | Löschung | `nein` | `*(nicht da)*` | …nehmen sie drogen [___] ich nehme keine… |
| 1334 | Löschung | `ich` | `*(nicht da)*` | …sie drogen nein [___] nehme keine drogen… |
| 1335 | Löschung | `nehme` | `*(nicht da)*` | …drogen nein ich [___] keine drogen wobei… |
| 1336 | Löschung | `keine` | `*(nicht da)*` | …nein ich nehme [___] drogen wobei ich… |
| 1337 | Löschung | `drogen` | `*(nicht da)*` | …ich nehme keine [___] wobei ich zugeben… |
| 1338 | Löschung | `wobei` | `*(nicht da)*` | …nehme keine drogen [___] ich zugeben muss… |
| 1339 | Löschung | `ich` | `*(nicht da)*` | …keine drogen wobei [___] zugeben muss dass… |
| 1340 | Löschung | `zugeben` | `*(nicht da)*` | …drogen wobei ich [___] muss dass ich… |
| 1341 | Löschung | `muss` | `*(nicht da)*` | …wobei ich zugeben [___] dass ich vor… |
| 1342 | Löschung | `dass` | `*(nicht da)*` | …ich zugeben muss [___] ich vor einiger… |
| 1343 | Löschung | `ich` | `*(nicht da)*` | …zugeben muss dass [___] vor einiger zeit… |
| 1344 | Löschung | `vor` | `*(nicht da)*` | …muss dass ich [___] einiger zeit ab… |
| 1345 | Löschung | `einiger` | `*(nicht da)*` | …dass ich vor [___] zeit ab und… |
| 1346 | Löschung | `zeit` | `*(nicht da)*` | …ich vor einiger [___] ab und zu… |
| 1347 | Löschung | `ab` | `*(nicht da)*` | …vor einiger zeit [___] und zu mal… |
| 1348 | Löschung | `und` | `*(nicht da)*` | …einiger zeit ab [___] zu mal ritalin… |
| 1349 | Löschung | `zu` | `*(nicht da)*` | …zeit ab und [___] mal ritalin genommen… |
| 1350 | Löschung | `mal` | `*(nicht da)*` | …ab und zu [___] ritalin genommen habe… |
| 1351 | Löschung | `ritalin` | `*(nicht da)*` | …und zu mal [___] genommen habe einfach… |
| 1352 | Löschung | `genommen` | `*(nicht da)*` | …zu mal ritalin [___] habe einfach weil… |
| 1353 | Löschung | `habe` | `*(nicht da)*` | …mal ritalin genommen [___] einfach weil wir… |
| 1354 | Löschung | `einfach` | `*(nicht da)*` | …ritalin genommen habe [___] weil wir ein… |
| 1355 | Löschung | `weil` | `*(nicht da)*` | …genommen habe einfach [___] wir ein paar… |
| 1356 | Löschung | `wir` | `*(nicht da)*` | …habe einfach weil [___] ein paar wirklich… |
| 1357 | Löschung | `ein` | `*(nicht da)*` | …einfach weil wir [___] paar wirklich große… |
| 1358 | Löschung | `paar` | `*(nicht da)*` | …weil wir ein [___] wirklich große projekte… |
| 1359 | Löschung | `wirklich` | `*(nicht da)*` | …wir ein paar [___] große projekte auf… |
| 1360 | Löschung | `große` | `*(nicht da)*` | …ein paar wirklich [___] projekte auf der… |
| 1361 | Löschung | `projekte` | `*(nicht da)*` | …paar wirklich große [___] auf der arbeit… |
| 1362 | Löschung | `auf` | `*(nicht da)*` | …wirklich große projekte [___] der arbeit hatten… |
| 1363 | Löschung | `der` | `*(nicht da)*` | …große projekte auf [___] arbeit hatten für… |
| 1364 | Löschung | `arbeit` | `*(nicht da)*` | …projekte auf der [___] hatten für die… |
| 1365 | Löschung | `hatten` | `*(nicht da)*` | …auf der arbeit [___] für die ich… |
| 1366 | Löschung | `für` | `*(nicht da)*` | …der arbeit hatten [___] die ich zuständig… |
| 1367 | Löschung | `die` | `*(nicht da)*` | …arbeit hatten für [___] ich zuständig war… |
| 1368 | Löschung | `ich` | `*(nicht da)*` | …hatten für die [___] zuständig war und… |
| 1369 | Löschung | `zuständig` | `*(nicht da)*` | …für die ich [___] war und ich… |
| 1370 | Löschung | `war` | `*(nicht da)*` | …die ich zuständig [___] und ich musste… |
| 1371 | Löschung | `und` | `*(nicht da)*` | …ich zuständig war [___] ich musste wirklich… |
| 1372 | Löschung | `ich` | `*(nicht da)*` | …zuständig war und [___] musste wirklich sehr… |
| 1373 | Löschung | `musste` | `*(nicht da)*` | …war und ich [___] wirklich sehr lange… |
| 1374 | Löschung | `wirklich` | `*(nicht da)*` | …und ich musste [___] sehr lange arbeiten… |
| 1375 | Löschung | `sehr` | `*(nicht da)*` | …ich musste wirklich [___] lange arbeiten und… |
| 1376 | Löschung | `lange` | `*(nicht da)*` | …musste wirklich sehr [___] arbeiten und ja… |
| 1377 | Löschung | `arbeiten` | `*(nicht da)*` | …wirklich sehr lange [___] und ja habe… |
| 1378 | Löschung | `und` | `*(nicht da)*` | …sehr lange arbeiten [___] ja habe zwei… |
| 1379 | Löschung | `ja` | `*(nicht da)*` | …lange arbeiten und [___] habe zwei dreimal… |
| 1380 | Löschung | `habe` | `*(nicht da)*` | …arbeiten und ja [___] zwei dreimal ritalin… |
| 1381 | Löschung | `zwei` | `*(nicht da)*` | …und ja habe [___] dreimal ritalin genommen… |
| 1382 | Löschung | `dreimal` | `*(nicht da)*` | …ja habe zwei [___] ritalin genommen okay… |
| 1383 | Löschung | `ritalin` | `*(nicht da)*` | …habe zwei dreimal [___] genommen okay das… |
| 1384 | Löschung | `genommen` | `*(nicht da)*` | …zwei dreimal ritalin [___] okay das war… |
| 1385 | Löschung | `okay` | `*(nicht da)*` | …dreimal ritalin genommen [___] das war es… |
| 1386 | Löschung | `das` | `*(nicht da)*` | …ritalin genommen okay [___] war es aber… |
| 1387 | Löschung | `war` | `*(nicht da)*` | …genommen okay das [___] es aber ja… |
| 1388 | Löschung | `es` | `*(nicht da)*` | …okay das war [___] aber ja ja… |
| 1389 | Löschung | `aber` | `*(nicht da)*` | …das war es [___] ja ja sehr… |
| 1390 | Löschung | `ja` | `*(nicht da)*` | …war es aber [___] ja sehr gut… |
| 1391 | Löschung | `ja` | `*(nicht da)*` | …es aber ja [___] sehr gut okay… |
| 1392 | Löschung | `sehr` | `*(nicht da)*` | …aber ja ja [___] gut okay prima… |
| 1393 | Löschung | `gut` | `*(nicht da)*` | …ja ja sehr [___] okay prima gut… |
| 1394 | Löschung | `okay` | `*(nicht da)*` | …ja sehr gut [___] prima gut kurz… |
| 1395 | Löschung | `prima` | `*(nicht da)*` | …sehr gut okay [___] gut kurz zu… |
| 1396 | Löschung | `gut` | `*(nicht da)*` | …gut okay prima [___] kurz zu ihrer… |
| 1397 | Löschung | `kurz` | `*(nicht da)*` | …okay prima gut [___] zu ihrer familie… |
| 1398 | Löschung | `zu` | `*(nicht da)*` | …prima gut kurz [___] ihrer familie gibt… |
| 1399 | Löschung | `ihrer` | `*(nicht da)*` | …gut kurz zu [___] familie gibt es… |
| 1400 | Löschung | `familie` | `*(nicht da)*` | …kurz zu ihrer [___] gibt es in… |
| 1401 | Löschung | `gibt` | `*(nicht da)*` | …zu ihrer familie [___] es in ihrer… |
| 1402 | Löschung | `es` | `*(nicht da)*` | …ihrer familie gibt [___] in ihrer familie… |
| 1403 | Löschung | `in` | `*(nicht da)*` | …familie gibt es [___] ihrer familie eltern… |
| 1404 | Löschung | `ihrer` | `*(nicht da)*` | …gibt es in [___] familie eltern großeltern… |
| 1405 | Löschung | `familie` | `*(nicht da)*` | …es in ihrer [___] eltern großeltern geschwister… |
| 1406 | Löschung | `eltern` | `*(nicht da)*` | …in ihrer familie [___] großeltern geschwister irgendwelche… |
| 1407 | Löschung | `großeltern` | `*(nicht da)*` | …ihrer familie eltern [___] geschwister irgendwelche vorerkrankungen… |
| 1408 | Löschung | `geschwister` | `*(nicht da)*` | …familie eltern großeltern [___] irgendwelche vorerkrankungen oder… |
| 1409 | Löschung | `irgendwelche` | `*(nicht da)*` | …eltern großeltern geschwister [___] vorerkrankungen oder chronische… |
| 1410 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …großeltern geschwister irgendwelche [___] oder chronische erkrankungen… |
| 1411 | Löschung | `oder` | `*(nicht da)*` | …geschwister irgendwelche vorerkrankungen [___] chronische erkrankungen wie… |
| 1412 | Löschung | `chronische` | `*(nicht da)*` | …irgendwelche vorerkrankungen oder [___] erkrankungen wie zum… |
| 1413 | Löschung | `erkrankungen` | `*(nicht da)*` | …vorerkrankungen oder chronische [___] wie zum beispiel… |
| 1414 | Löschung | `wie` | `*(nicht da)*` | …oder chronische erkrankungen [___] zum beispiel krebs… |
| 1415 | Löschung | `zum` | `*(nicht da)*` | …chronische erkrankungen wie [___] beispiel krebs oder… |
| 1416 | Löschung | `beispiel` | `*(nicht da)*` | …erkrankungen wie zum [___] krebs oder diabetes… |
| 1417 | Löschung | `krebs` | `*(nicht da)*` | …wie zum beispiel [___] oder diabetes oder… |
| 1418 | Löschung | `oder` | `*(nicht da)*` | …zum beispiel krebs [___] diabetes oder einen… |
| 1419 | Löschung | `diabetes` | `*(nicht da)*` | …beispiel krebs oder [___] oder einen herzinfarkt… |
| 1420 | Löschung | `oder` | `*(nicht da)*` | …krebs oder diabetes [___] einen herzinfarkt irgendetwas… |
| 1421 | Löschung | `einen` | `*(nicht da)*` | …oder diabetes oder [___] herzinfarkt irgendetwas was… |
| 1422 | Löschung | `herzinfarkt` | `*(nicht da)*` | …diabetes oder einen [___] irgendetwas was ihnen… |
| 1423 | Löschung | `irgendetwas` | `*(nicht da)*` | …oder einen herzinfarkt [___] was ihnen bekannt… |
| 1424 | Löschung | `was` | `*(nicht da)*` | …einen herzinfarkt irgendetwas [___] ihnen bekannt ist… |
| 1425 | Löschung | `ihnen` | `*(nicht da)*` | …herzinfarkt irgendetwas was [___] bekannt ist das… |
| 1426 | Löschung | `bekannt` | `*(nicht da)*` | …irgendetwas was ihnen [___] ist das gibt… |
| 1427 | Löschung | `ist` | `*(nicht da)*` | …was ihnen bekannt [___] das gibt es… |
| 1428 | Löschung | `das` | `*(nicht da)*` | …ihnen bekannt ist [___] gibt es ja… |
| 1429 | Löschung | `gibt` | `*(nicht da)*` | …bekannt ist das [___] es ja großeltern… |
| 1430 | Löschung | `es` | `*(nicht da)*` | …ist das gibt [___] ja großeltern auch… |
| 1431 | Löschung | `ja` | `*(nicht da)*` | …das gibt es [___] großeltern auch ja… |
| 1432 | Löschung | `großeltern` | `*(nicht da)*` | …gibt es ja [___] auch ja klar… |
| 1433 | Löschung | `auch` | `*(nicht da)*` | …es ja großeltern [___] ja klar mein… |
| 1434 | Löschung | `ja` | `*(nicht da)*` | …ja großeltern auch [___] klar mein großvater… |
| 1435 | Löschung | `klar` | `*(nicht da)*` | …großeltern auch ja [___] mein großvater hatte… |
| 1436 | Löschung | `mein` | `*(nicht da)*` | …auch ja klar [___] großvater hatte leberzirrhose… |
| 1437 | Löschung | `großvater` | `*(nicht da)*` | …ja klar mein [___] hatte leberzirrhose und… |
| 1438 | Löschung | `hatte` | `*(nicht da)*` | …klar mein großvater [___] leberzirrhose und ist… |
| 1439 | Löschung | `leberzirrhose` | `*(nicht da)*` | …mein großvater hatte [___] und ist leider… |
| 1440 | Löschung | `und` | `*(nicht da)*` | …großvater hatte leberzirrhose [___] ist leider auch… |
| 1441 | Löschung | `ist` | `*(nicht da)*` | …hatte leberzirrhose und [___] leider auch daran… |
| 1442 | Löschung | `leider` | `*(nicht da)*` | …leberzirrhose und ist [___] auch daran geschrauben… |
| 1443 | Löschung | `auch` | `*(nicht da)*` | …und ist leider [___] daran geschrauben oh… |
| 1444 | Löschung | `daran` | `*(nicht da)*` | …ist leider auch [___] geschrauben oh das… |
| 1445 | Löschung | `geschrauben` | `*(nicht da)*` | …leider auch daran [___] oh das tut… |
| 1446 | Löschung | `oh` | `*(nicht da)*` | …auch daran geschrauben [___] das tut mir… |
| 1447 | Löschung | `das` | `*(nicht da)*` | …daran geschrauben oh [___] tut mir leid… |
| 1448 | Löschung | `tut` | `*(nicht da)*` | …geschrauben oh das [___] mir leid danke… |
| 1449 | Löschung | `mir` | `*(nicht da)*` | …oh das tut [___] leid danke ist… |
| 1450 | Löschung | `leid` | `*(nicht da)*` | …das tut mir [___] danke ist schon… |
| 1451 | Löschung | `danke` | `*(nicht da)*` | …tut mir leid [___] ist schon lange… |
| 1452 | Löschung | `ist` | `*(nicht da)*` | …mir leid danke [___] schon lange her… |
| 1453 | Löschung | `schon` | `*(nicht da)*` | …leid danke ist [___] lange her und… |
| 1454 | Löschung | `lange` | `*(nicht da)*` | …danke ist schon [___] her und meine… |
| 1455 | Löschung | `her` | `*(nicht da)*` | …ist schon lange [___] und meine großmutter… |
| 1456 | Löschung | `und` | `*(nicht da)*` | …schon lange her [___] meine großmutter hatte… |
| 1457 | Löschung | `meine` | `*(nicht da)*` | …lange her und [___] großmutter hatte großkrebs… |
| 1458 | Löschung | `großmutter` | `*(nicht da)*` | …her und meine [___] hatte großkrebs aber… |
| 1459 | Löschung | `hatte` | `*(nicht da)*` | …und meine großmutter [___] großkrebs aber sie… |
| 1460 | Löschung | `großkrebs` | `*(nicht da)*` | …meine großmutter hatte [___] aber sie lebt… |
| 1461 | Löschung | `aber` | `*(nicht da)*` | …großmutter hatte großkrebs [___] sie lebt noch… |
| 1462 | Löschung | `sie` | `*(nicht da)*` | …hatte großkrebs aber [___] lebt noch okay… |
| 1463 | Löschung | `lebt` | `*(nicht da)*` | …großkrebs aber sie [___] noch okay sehr… |
| 1464 | Löschung | `noch` | `*(nicht da)*` | …aber sie lebt [___] okay sehr gut… |
| 1465 | Löschung | `okay` | `*(nicht da)*` | …sie lebt noch [___] sehr gut ihre… |
| 1466 | Löschung | `sehr` | `*(nicht da)*` | …lebt noch okay [___] gut ihre eltern… |
| 1467 | Löschung | `gut` | `*(nicht da)*` | …noch okay sehr [___] ihre eltern sind… |
| 1468 | Löschung | `ihre` | `*(nicht da)*` | …okay sehr gut [___] eltern sind gesund… |
| 1469 | Löschung | `eltern` | `*(nicht da)*` | …sehr gut ihre [___] sind gesund meine… |
| 1470 | Löschung | `sind` | `*(nicht da)*` | …gut ihre eltern [___] gesund meine eltern… |
| 1471 | Löschung | `gesund` | `*(nicht da)*` | …ihre eltern sind [___] meine eltern sind… |
| 1472 | Löschung | `meine` | `*(nicht da)*` | …eltern sind gesund [___] eltern sind zum… |
| 1473 | Löschung | `eltern` | `*(nicht da)*` | …sind gesund meine [___] sind zum glück… |
| 1474 | Löschung | `sind` | `*(nicht da)*` | …gesund meine eltern [___] zum glück gesund… |
| 1475 | Löschung | `zum` | `*(nicht da)*` | …meine eltern sind [___] glück gesund ja… |
| 1476 | Löschung | `glück` | `*(nicht da)*` | …eltern sind zum [___] gesund ja sehr… |
| 1477 | Löschung | `gesund` | `*(nicht da)*` | …sind zum glück [___] ja sehr schön… |
| 1478 | Löschung | `ja` | `*(nicht da)*` | …zum glück gesund [___] sehr schön haben… |
| 1479 | Löschung | `sehr` | `*(nicht da)*` | …glück gesund ja [___] schön haben sie… |
| 1480 | Löschung | `schön` | `*(nicht da)*` | …gesund ja sehr [___] haben sie geschwister… |
| 1481 | Löschung | `haben` | `*(nicht da)*` | …ja sehr schön [___] sie geschwister frau… |
| 1482 | Löschung | `sie` | `*(nicht da)*` | …sehr schön haben [___] geschwister frau böcken… |
| 1483 | Löschung | `geschwister` | `*(nicht da)*` | …schön haben sie [___] frau böcken westfalen… |
| 1484 | Löschung | `frau` | `*(nicht da)*` | …haben sie geschwister [___] böcken westfalen ich… |
| 1485 | Löschung | `böcken` | `*(nicht da)*` | …sie geschwister frau [___] westfalen ich habe… |
| 1486 | Löschung | `westfalen` | `*(nicht da)*` | …geschwister frau böcken [___] ich habe eine… |
| 1487 | Löschung | `ich` | `*(nicht da)*` | …frau böcken westfalen [___] habe eine schwester… |
| 1488 | Löschung | `habe` | `*(nicht da)*` | …böcken westfalen ich [___] eine schwester und… |
| 1489 | Löschung | `eine` | `*(nicht da)*` | …westfalen ich habe [___] schwester und sie… |
| 1490 | Löschung | `schwester` | `*(nicht da)*` | …ich habe eine [___] und sie hat… |
| 1491 | Löschung | `und` | `*(nicht da)*` | …habe eine schwester [___] sie hat auch… |
| 1492 | Löschung | `sie` | `*(nicht da)*` | …eine schwester und [___] hat auch ein… |
| 1493 | Löschung | `hat` | `*(nicht da)*` | …schwester und sie [___] auch ein paar… |
| 1494 | Löschung | `auch` | `*(nicht da)*` | …und sie hat [___] ein paar problemchen… |
| 1495 | Löschung | `ein` | `*(nicht da)*` | …sie hat auch [___] paar problemchen und… |
| 1496 | Löschung | `paar` | `*(nicht da)*` | …hat auch ein [___] problemchen und zwar… |
| 1497 | Löschung | `problemchen` | `*(nicht da)*` | …auch ein paar [___] und zwar hat… |
| 1498 | Löschung | `und` | `*(nicht da)*` | …ein paar problemchen [___] zwar hat sie… |
| 1499 | Löschung | `zwar` | `*(nicht da)*` | …paar problemchen und [___] hat sie asthma… |
| 1500 | Löschung | `hat` | `*(nicht da)*` | …problemchen und zwar [___] sie asthma und… |
| 1501 | Löschung | `sie` | `*(nicht da)*` | …und zwar hat [___] asthma und neurodermitis… |
| 1502 | Löschung | `asthma` | `*(nicht da)*` | …zwar hat sie [___] und neurodermitis asthma… |
| 1503 | Löschung | `und` | `*(nicht da)*` | …hat sie asthma [___] neurodermitis asthma und… |
| 1504 | Löschung | `neurodermitis` | `*(nicht da)*` | …sie asthma und [___] asthma und neurodermitis… |
| 1505 | Löschung | `asthma` | `*(nicht da)*` | …asthma und neurodermitis [___] und neurodermitis okay… |
| 1506 | Löschung | `und` | `*(nicht da)*` | …und neurodermitis asthma [___] neurodermitis okay aber… |
| 1507 | Löschung | `neurodermitis` | `*(nicht da)*` | …neurodermitis asthma und [___] okay aber sonst… |
| 1508 | Löschung | `okay` | `*(nicht da)*` | …asthma und neurodermitis [___] aber sonst ist… |
| 1509 | Löschung | `aber` | `*(nicht da)*` | …und neurodermitis okay [___] sonst ist auch… |
| 1510 | Löschung | `sonst` | `*(nicht da)*` | …neurodermitis okay aber [___] ist auch sie… |
| 1511 | Löschung | `ist` | `*(nicht da)*` | …okay aber sonst [___] auch sie gesund… |
| 1512 | Löschung | `auch` | `*(nicht da)*` | …aber sonst ist [___] sie gesund ja… |
| 1513 | Löschung | `sie` | `*(nicht da)*` | …sonst ist auch [___] gesund ja sonst… |
| 1514 | Löschung | `gesund` | `*(nicht da)*` | …ist auch sie [___] ja sonst geht… |
| 1515 | Löschung | `ja` | `*(nicht da)*` | …auch sie gesund [___] sonst geht sie… |
| 1516 | Löschung | `sonst` | `*(nicht da)*` | …sie gesund ja [___] geht sie gut… |
| 1517 | Löschung | `geht` | `*(nicht da)*` | …gesund ja sonst [___] sie gut sehr… |
| 1518 | Löschung | `sie` | `*(nicht da)*` | …ja sonst geht [___] gut sehr gut… |
| 1519 | Löschung | `gut` | `*(nicht da)*` | …sonst geht sie [___] sehr gut haben… |
| 1520 | Löschung | `sehr` | `*(nicht da)*` | …geht sie gut [___] gut haben sie… |
| 1521 | Löschung | `gut` | `*(nicht da)*` | …sie gut sehr [___] haben sie kinder… |
| 1522 | Löschung | `haben` | `*(nicht da)*` | …gut sehr gut [___] sie kinder frau… |
| 1523 | Löschung | `sie` | `*(nicht da)*` | …sehr gut haben [___] kinder frau böcken… |
| 1524 | Löschung | `kinder` | `*(nicht da)*` | …gut haben sie [___] frau böcken westfalen… |
| 1525 | Löschung | `frau` | `*(nicht da)*` | …haben sie kinder [___] böcken westfalen nein… |
| 1526 | Löschung | `böcken` | `*(nicht da)*` | …sie kinder frau [___] westfalen nein ich… |
| 1527 | Löschung | `westfalen` | `*(nicht da)*` | …kinder frau böcken [___] nein ich habe… |
| 1528 | Löschung | `nein` | `*(nicht da)*` | …frau böcken westfalen [___] ich habe keine… |
| 1529 | Löschung | `ich` | `*(nicht da)*` | …böcken westfalen nein [___] habe keine kinder… |
| 1530 | Löschung | `habe` | `*(nicht da)*` | …westfalen nein ich [___] keine kinder okay… |
| 1531 | Löschung | `keine` | `*(nicht da)*` | …nein ich habe [___] kinder okay wie… |
| 1532 | Löschung | `kinder` | `*(nicht da)*` | …ich habe keine [___] okay wie sieht… |
| 1533 | Löschung | `okay` | `*(nicht da)*` | …habe keine kinder [___] wie sieht es… |
| 1534 | Löschung | `wie` | `*(nicht da)*` | …keine kinder okay [___] sieht es denn… |
| 1535 | Löschung | `sieht` | `*(nicht da)*` | …kinder okay wie [___] es denn in… |
| 1536 | Löschung | `es` | `*(nicht da)*` | …okay wie sieht [___] denn in ihrem… |
| 1537 | Löschung | `denn` | `*(nicht da)*` | …wie sieht es [___] in ihrem sozialleben… |
| 1538 | Löschung | `in` | `*(nicht da)*` | …sieht es denn [___] ihrem sozialleben aus… |
| 1539 | Löschung | `ihrem` | `*(nicht da)*` | …es denn in [___] sozialleben aus sind… |
| 1540 | Löschung | `sozialleben` | `*(nicht da)*` | …denn in ihrem [___] aus sind sie… |
| 1541 | Löschung | `aus` | `*(nicht da)*` | …in ihrem sozialleben [___] sind sie verheiratet… |
| 1542 | Löschung | `sind` | `*(nicht da)*` | …ihrem sozialleben aus [___] sie verheiratet ich… |
| 1543 | Löschung | `sie` | `*(nicht da)*` | …sozialleben aus sind [___] verheiratet ich bin… |
| 1544 | Löschung | `verheiratet` | `*(nicht da)*` | …aus sind sie [___] ich bin frisch… |
| 1545 | Löschung | `ich` | `*(nicht da)*` | …sind sie verheiratet [___] bin frisch verheiratet… |
| 1546 | Löschung | `bin` | `*(nicht da)*` | …sie verheiratet ich [___] frisch verheiratet ja… |
| 1547 | Löschung | `frisch` | `*(nicht da)*` | …verheiratet ich bin [___] verheiratet ja seit… |
| 1548 | Löschung | `verheiratet` | `*(nicht da)*` | …ich bin frisch [___] ja seit fünf… |
| 1549 | Löschung | `ja` | `*(nicht da)*` | …bin frisch verheiratet [___] seit fünf monaten… |
| 1550 | Löschung | `seit` | `*(nicht da)*` | …frisch verheiratet ja [___] fünf monaten wie… |
| 1551 | Löschung | `fünf` | `*(nicht da)*` | …verheiratet ja seit [___] monaten wie schön… |
| 1552 | Löschung | `monaten` | `*(nicht da)*` | …ja seit fünf [___] wie schön herzlichen… |
| 1553 | Löschung | `wie` | `*(nicht da)*` | …seit fünf monaten [___] schön herzlichen glückwunsch… |
| 1554 | Löschung | `schön` | `*(nicht da)*` | …fünf monaten wie [___] herzlichen glückwunsch auch… |
| 1555 | Löschung | `herzlichen` | `*(nicht da)*` | …monaten wie schön [___] glückwunsch auch dazu… |
| 1556 | Löschung | `glückwunsch` | `*(nicht da)*` | …wie schön herzlichen [___] auch dazu herzlichen… |
| 1557 | Löschung | `auch` | `*(nicht da)*` | …schön herzlichen glückwunsch [___] dazu herzlichen dank… |
| 1558 | Löschung | `dazu` | `*(nicht da)*` | …herzlichen glückwunsch auch [___] herzlichen dank sehr… |
| 1559 | Löschung | `herzlichen` | `*(nicht da)*` | …glückwunsch auch dazu [___] dank sehr schön… |
| 1560 | Löschung | `dank` | `*(nicht da)*` | …auch dazu herzlichen [___] sehr schön dann… |
| 1561 | Löschung | `sehr` | `*(nicht da)*` | …dazu herzlichen dank [___] schön dann gehe… |
| 1562 | Löschung | `schön` | `*(nicht da)*` | …herzlichen dank sehr [___] dann gehe ich… |
| 1563 | Löschung | `dann` | `*(nicht da)*` | …dank sehr schön [___] gehe ich davon… |
| 1564 | Löschung | `gehe` | `*(nicht da)*` | …sehr schön dann [___] ich davon aus… |
| 1565 | Löschung | `ich` | `*(nicht da)*` | …schön dann gehe [___] davon aus sie… |
| 1566 | Löschung | `davon` | `*(nicht da)*` | …dann gehe ich [___] aus sie leben… |
| 1567 | Löschung | `aus` | `*(nicht da)*` | …gehe ich davon [___] sie leben auch… |
| 1568 | Löschung | `sie` | `*(nicht da)*` | …ich davon aus [___] leben auch mit… |
| 1569 | Löschung | `leben` | `*(nicht da)*` | …davon aus sie [___] auch mit ihrem… |
| 1570 | Löschung | `auch` | `*(nicht da)*` | …aus sie leben [___] mit ihrem ehemann… |
| 1571 | Löschung | `mit` | `*(nicht da)*` | …sie leben auch [___] ihrem ehemann zusammen… |
| 1572 | Löschung | `ihrem` | `*(nicht da)*` | …leben auch mit [___] ehemann zusammen das… |
| 1573 | Löschung | `ehemann` | `*(nicht da)*` | …auch mit ihrem [___] zusammen das ist… |
| 1574 | Löschung | `zusammen` | `*(nicht da)*` | …mit ihrem ehemann [___] das ist richtig… |
| 1575 | Löschung | `das` | `*(nicht da)*` | …ihrem ehemann zusammen [___] ist richtig ja… |
| 1576 | Löschung | `ist` | `*(nicht da)*` | …ehemann zusammen das [___] richtig ja okay… |
| 1577 | Löschung | `richtig` | `*(nicht da)*` | …zusammen das ist [___] ja okay prima… |
| 1578 | Löschung | `ja` | `*(nicht da)*` | …das ist richtig [___] okay prima wir… |
| 1579 | Löschung | `okay` | `*(nicht da)*` | …ist richtig ja [___] prima wir hatten… |
| 1580 | Löschung | `prima` | `*(nicht da)*` | …richtig ja okay [___] wir hatten zwar… |
| 1581 | Löschung | `wir` | `*(nicht da)*` | …ja okay prima [___] hatten zwar eben… |
| 1582 | Löschung | `hatten` | `*(nicht da)*` | …okay prima wir [___] zwar eben schon… |
| 1583 | Löschung | `zwar` | `*(nicht da)*` | …prima wir hatten [___] eben schon mal… |
| 1584 | Löschung | `eben` | `*(nicht da)*` | …wir hatten zwar [___] schon mal kurz… |
| 1585 | Löschung | `schon` | `*(nicht da)*` | …hatten zwar eben [___] mal kurz über… |
| 1586 | Löschung | `mal` | `*(nicht da)*` | …zwar eben schon [___] kurz über ihre… |
| 1587 | Löschung | `kurz` | `*(nicht da)*` | …eben schon mal [___] über ihre arbeit… |
| 1588 | Löschung | `über` | `*(nicht da)*` | …schon mal kurz [___] ihre arbeit gesprochen… |
| 1589 | Löschung | `ihre` | `*(nicht da)*` | …mal kurz über [___] arbeit gesprochen aber… |
| 1590 | Löschung | `arbeit` | `*(nicht da)*` | …kurz über ihre [___] gesprochen aber ich… |
| 1591 | Löschung | `gesprochen` | `*(nicht da)*` | …über ihre arbeit [___] aber ich habe… |
| 1592 | Löschung | `aber` | `*(nicht da)*` | …ihre arbeit gesprochen [___] ich habe es… |
| 1593 | Löschung | `ich` | `*(nicht da)*` | …arbeit gesprochen aber [___] habe es nicht… |
| 1594 | Löschung | `habe` | `*(nicht da)*` | …gesprochen aber ich [___] es nicht ganz… |
| 1595 | Löschung | `es` | `*(nicht da)*` | …aber ich habe [___] nicht ganz auf… |
| 1596 | Löschung | `nicht` | `*(nicht da)*` | …ich habe es [___] ganz auf dem… |
| 1597 | Löschung | `ganz` | `*(nicht da)*` | …habe es nicht [___] auf dem schirm… |
| 1598 | Löschung | `auf` | `*(nicht da)*` | …es nicht ganz [___] dem schirm ob… |
| 1599 | Löschung | `schirm` | `*(nicht da)*` | …ganz auf dem [___] ob ich sie… |
| 1600 | Löschung | `ob` | `*(nicht da)*` | …auf dem schirm [___] ich sie schon… |
| 1601 | Löschung | `ich` | `*(nicht da)*` | …dem schirm ob [___] sie schon gefragt… |
| 1602 | Löschung | `sie` | `*(nicht da)*` | …schirm ob ich [___] schon gefragt habe… |
| 1603 | Löschung | `schon` | `*(nicht da)*` | …ob ich sie [___] gefragt habe was… |
| 1604 | Löschung | `gefragt` | `*(nicht da)*` | …ich sie schon [___] habe was sie… |
| 1605 | Löschung | `habe` | `*(nicht da)*` | …sie schon gefragt [___] was sie denn… |
| 1606 | Löschung | `was` | `*(nicht da)*` | …schon gefragt habe [___] sie denn beruflich… |
| 1607 | Löschung | `sie` | `*(nicht da)*` | …gefragt habe was [___] denn beruflich machen… |
| 1608 | Löschung | `denn` | `*(nicht da)*` | …habe was sie [___] beruflich machen ich… |
| 1609 | Löschung | `beruflich` | `*(nicht da)*` | …was sie denn [___] machen ich arbeite… |
| 1610 | Löschung | `machen` | `*(nicht da)*` | …sie denn beruflich [___] ich arbeite in… |
| 1611 | Löschung | `ich` | `*(nicht da)*` | …denn beruflich machen [___] arbeite in einer… |
| 1612 | Löschung | `arbeite` | `*(nicht da)*` | …beruflich machen ich [___] in einer marketingagentur… |
| 1613 | Löschung | `in` | `*(nicht da)*` | …machen ich arbeite [___] einer marketingagentur wenn… |
| 1614 | Löschung | `einer` | `*(nicht da)*` | …ich arbeite in [___] marketingagentur wenn da… |
| 1615 | Löschung | `marketingagentur` | `*(nicht da)*` | …arbeite in einer [___] wenn da für… |
| 1616 | Löschung | `wenn` | `*(nicht da)*` | …in einer marketingagentur [___] da für größere… |
| 1617 | Löschung | `da` | `*(nicht da)*` | …einer marketingagentur wenn [___] für größere kunden… |
| 1618 | Löschung | `für` | `*(nicht da)*` | …marketingagentur wenn da [___] größere kunden und… |
| 1619 | Löschung | `größere` | `*(nicht da)*` | …wenn da für [___] kunden und für… |
| 1620 | Löschung | `kunden` | `*(nicht da)*` | …da für größere [___] und für größere… |
| 1621 | Löschung | `und` | `*(nicht da)*` | …für größere kunden [___] für größere firmenkunden… |
| 1622 | Löschung | `für` | `*(nicht da)*` | …größere kunden und [___] größere firmenkunden und… |
| 1623 | Löschung | `größere` | `*(nicht da)*` | …kunden und für [___] firmenkunden und marketingprojekte… |
| 1624 | Löschung | `firmenkunden` | `*(nicht da)*` | …und für größere [___] und marketingprojekte zuständig… |
| 1625 | Löschung | `und` | `*(nicht da)*` | …für größere firmenkunden [___] marketingprojekte zuständig sind… |
| 1626 | Löschung | `marketingprojekte` | `*(nicht da)*` | …größere firmenkunden und [___] zuständig sind okay… |
| 1627 | Löschung | `zuständig` | `*(nicht da)*` | …firmenkunden und marketingprojekte [___] sind okay sehr… |
| 1628 | Löschung | `sind` | `*(nicht da)*` | …und marketingprojekte zuständig [___] okay sehr gut… |
| 1629 | Löschung | `okay` | `*(nicht da)*` | …marketingprojekte zuständig sind [___] sehr gut eine… |
| 1630 | Löschung | `sehr` | `*(nicht da)*` | …zuständig sind okay [___] gut eine letzte… |
| 1631 | Löschung | `gut` | `*(nicht da)*` | …sind okay sehr [___] eine letzte frage… |
| 1632 | Löschung | `eine` | `*(nicht da)*` | …okay sehr gut [___] letzte frage noch… |
| 1633 | Löschung | `letzte` | `*(nicht da)*` | …sehr gut eine [___] frage noch frau… |
| 1634 | Löschung | `frage` | `*(nicht da)*` | …gut eine letzte [___] noch frau böcken… |
| 1635 | Löschung | `noch` | `*(nicht da)*` | …eine letzte frage [___] frau böcken westfalen… |
| 1636 | Löschung | `frau` | `*(nicht da)*` | …letzte frage noch [___] böcken westfalen waren… |
| 1637 | Löschung | `böcken` | `*(nicht da)*` | …frage noch frau [___] westfalen waren sie… |
| 1638 | Löschung | `westfalen` | `*(nicht da)*` | …noch frau böcken [___] waren sie in… |
| 1639 | Löschung | `waren` | `*(nicht da)*` | …frau böcken westfalen [___] sie in der… |
| 1640 | Löschung | `sie` | `*(nicht da)*` | …böcken westfalen waren [___] in der letzten… |
| 1641 | Löschung | `in` | `*(nicht da)*` | …westfalen waren sie [___] der letzten zeit… |
| 1642 | Löschung | `der` | `*(nicht da)*` | …waren sie in [___] letzten zeit im… |
| 1643 | Löschung | `letzten` | `*(nicht da)*` | …sie in der [___] zeit im ausland… |
| 1644 | Löschung | `zeit` | `*(nicht da)*` | …in der letzten [___] im ausland ja… |
| 1645 | Löschung | `im` | `*(nicht da)*` | …der letzten zeit [___] ausland ja ich… |
| 1646 | Löschung | `ausland` | `*(nicht da)*` | …letzten zeit im [___] ja ich war… |
| 1647 | Löschung | `ja` | `*(nicht da)*` | …zeit im ausland [___] ich war vor… |
| 1648 | Löschung | `ich` | `*(nicht da)*` | …im ausland ja [___] war vor zwei… |
| 1649 | Löschung | `war` | `*(nicht da)*` | …ausland ja ich [___] vor zwei monaten… |
| 1650 | Löschung | `vor` | `*(nicht da)*` | …ja ich war [___] zwei monaten geschäftlich… |
| 1651 | Löschung | `zwei` | `*(nicht da)*` | …ich war vor [___] monaten geschäftlich in… |
| 1652 | Löschung | `monaten` | `*(nicht da)*` | …war vor zwei [___] geschäftlich in singapur… |
| 1653 | Löschung | `geschäftlich` | `*(nicht da)*` | …vor zwei monaten [___] in singapur okay… |
| 1654 | Löschung | `in` | `*(nicht da)*` | …zwei monaten geschäftlich [___] singapur okay und… |
| 1655 | Löschung | `singapur` | `*(nicht da)*` | …monaten geschäftlich in [___] okay und wie… |
| 1656 | Löschung | `okay` | `*(nicht da)*` | …geschäftlich in singapur [___] und wie lange… |
| 1657 | Löschung | `und` | `*(nicht da)*` | …in singapur okay [___] wie lange waren… |
| 1658 | Löschung | `wie` | `*(nicht da)*` | …singapur okay und [___] lange waren sie… |
| 1659 | Löschung | `lange` | `*(nicht da)*` | …okay und wie [___] waren sie da… |
| 1660 | Löschung | `waren` | `*(nicht da)*` | …und wie lange [___] sie da zwei… |
| 1661 | Löschung | `sie` | `*(nicht da)*` | …wie lange waren [___] da zwei wochen… |
| 1662 | Löschung | `da` | `*(nicht da)*` | …lange waren sie [___] zwei wochen insgesamt… |
| 1663 | Löschung | `zwei` | `*(nicht da)*` | …waren sie da [___] wochen insgesamt zwei… |
| 1664 | Löschung | `wochen` | `*(nicht da)*` | …sie da zwei [___] insgesamt zwei wochen… |
| 1665 | Löschung | `insgesamt` | `*(nicht da)*` | …da zwei wochen [___] zwei wochen insgesamt… |
| 1666 | Löschung | `zwei` | `*(nicht da)*` | …zwei wochen insgesamt [___] wochen insgesamt okay… |
| 1667 | Löschung | `wochen` | `*(nicht da)*` | …wochen insgesamt zwei [___] insgesamt okay gut… |
| 1668 | Löschung | `insgesamt` | `*(nicht da)*` | …insgesamt zwei wochen [___] okay gut frau… |
| 1669 | Löschung | `okay` | `*(nicht da)*` | …zwei wochen insgesamt [___] gut frau böcken… |
| 1670 | Löschung | `gut` | `*(nicht da)*` | …wochen insgesamt okay [___] frau böcken westfalen… |
| 1671 | Löschung | `frau` | `*(nicht da)*` | …insgesamt okay gut [___] böcken westfalen von… |
| 1672 | Löschung | `böcken` | `*(nicht da)*` | …okay gut frau [___] westfalen von meiner… |
| 1673 | Löschung | `westfalen` | `*(nicht da)*` | …gut frau böcken [___] von meiner seite… |
| 1674 | Löschung | `von` | `*(nicht da)*` | …frau böcken westfalen [___] meiner seite war… |
| 1675 | Löschung | `meiner` | `*(nicht da)*` | …böcken westfalen von [___] seite war es… |
| 1676 | Löschung | `seite` | `*(nicht da)*` | …westfalen von meiner [___] war es das… |
| 1677 | Löschung | `war` | `*(nicht da)*` | …von meiner seite [___] es das ich… |
| 1678 | Löschung | `es` | `*(nicht da)*` | …meiner seite war [___] das ich habe… |
| 1679 | Löschung | `das` | `*(nicht da)*` | …seite war es [___] ich habe alle… |
| 1680 | Löschung | `ich` | `*(nicht da)*` | …war es das [___] habe alle fragen… |
| 1681 | Löschung | `habe` | `*(nicht da)*` | …es das ich [___] alle fragen gestellt… |
| 1682 | Löschung | `alle` | `*(nicht da)*` | …das ich habe [___] fragen gestellt ich… |
| 1683 | Löschung | `fragen` | `*(nicht da)*` | …ich habe alle [___] gestellt ich würde… |
| 1684 | Löschung | `gestellt` | `*(nicht da)*` | …habe alle fragen [___] ich würde das… |
| 1685 | Löschung | `ich` | `*(nicht da)*` | …alle fragen gestellt [___] würde das gleich… |
| 1686 | Löschung | `würde` | `*(nicht da)*` | …fragen gestellt ich [___] das gleich nochmal… |
| 1687 | Löschung | `das` | `*(nicht da)*` | …gestellt ich würde [___] gleich nochmal mit… |
| 1688 | Löschung | `gleich` | `*(nicht da)*` | …ich würde das [___] nochmal mit ihnen… |
| 1689 | Löschung | `nochmal` | `*(nicht da)*` | …würde das gleich [___] mit ihnen durchgehen… |
| 1690 | Löschung | `mit` | `*(nicht da)*` | …das gleich nochmal [___] ihnen durchgehen um… |
| 1691 | Löschung | `ihnen` | `*(nicht da)*` | …gleich nochmal mit [___] durchgehen um abzugleichen… |
| 1692 | Löschung | `durchgehen` | `*(nicht da)*` | …nochmal mit ihnen [___] um abzugleichen dass… |
| 1693 | Löschung | `um` | `*(nicht da)*` | …mit ihnen durchgehen [___] abzugleichen dass ich… |
| 1694 | Löschung | `abzugleichen` | `*(nicht da)*` | …ihnen durchgehen um [___] dass ich auch… |
| 1695 | Löschung | `dass` | `*(nicht da)*` | …durchgehen um abzugleichen [___] ich auch wirklich… |
| 1696 | Löschung | `ich` | `*(nicht da)*` | …um abzugleichen dass [___] auch wirklich alles… |
| 1697 | Löschung | `auch` | `*(nicht da)*` | …abzugleichen dass ich [___] wirklich alles richtig… |
| 1698 | Löschung | `wirklich` | `*(nicht da)*` | …dass ich auch [___] alles richtig notiert… |
| 1699 | Löschung | `alles` | `*(nicht da)*` | …ich auch wirklich [___] richtig notiert habe… |
| 1700 | Löschung | `richtig` | `*(nicht da)*` | …auch wirklich alles [___] notiert habe vorher… |
| 1701 | Löschung | `notiert` | `*(nicht da)*` | …wirklich alles richtig [___] habe vorher möchte… |
| 1702 | Löschung | `habe` | `*(nicht da)*` | …alles richtig notiert [___] vorher möchte ich… |
| 1703 | Löschung | `vorher` | `*(nicht da)*` | …richtig notiert habe [___] möchte ich sie… |
| 1704 | Löschung | `möchte` | `*(nicht da)*` | …notiert habe vorher [___] ich sie aber… |
| 1705 | Löschung | `ich` | `*(nicht da)*` | …habe vorher möchte [___] sie aber noch… |
| 1706 | Löschung | `sie` | `*(nicht da)*` | …vorher möchte ich [___] aber noch kurz… |
| 1707 | Löschung | `aber` | `*(nicht da)*` | …möchte ich sie [___] noch kurz fragen… |
| 1708 | Löschung | `noch` | `*(nicht da)*` | …ich sie aber [___] kurz fragen haben… |
| 1709 | Löschung | `kurz` | `*(nicht da)*` | …sie aber noch [___] fragen haben sie… |
| 1710 | Löschung | `fragen` | `*(nicht da)*` | …aber noch kurz [___] haben sie fragen… |
| 1711 | Löschung | `haben` | `*(nicht da)*` | …noch kurz fragen [___] sie fragen an… |
| 1712 | Löschung | `sie` | `*(nicht da)*` | …kurz fragen haben [___] fragen an mich… |
| 1713 | Löschung | `fragen` | `*(nicht da)*` | …fragen haben sie [___] an mich ja… |
| 1714 | Löschung | `an` | `*(nicht da)*` | …haben sie fragen [___] mich ja eine… |
| 1715 | Löschung | `mich` | `*(nicht da)*` | …sie fragen an [___] ja eine meinen… |
| 1716 | Löschung | `ja` | `*(nicht da)*` | …fragen an mich [___] eine meinen sie… |
| 1717 | Löschung | `eine` | `*(nicht da)*` | …an mich ja [___] meinen sie dass… |
| 1718 | Löschung | `meinen` | `*(nicht da)*` | …mich ja eine [___] sie dass es… |
| 1719 | Löschung | `sie` | `*(nicht da)*` | …ja eine meinen [___] dass es so… |
| 1720 | Löschung | `dass` | `*(nicht da)*` | …eine meinen sie [___] es so sehr… |
| 1721 | Löschung | `es` | `*(nicht da)*` | …meinen sie dass [___] so sehr schlimm… |
| 1722 | Löschung | `so` | `*(nicht da)*` | …sie dass es [___] sehr schlimm wird… |
| 1723 | Löschung | `sehr` | `*(nicht da)*` | …dass es so [___] schlimm wird oder… |
| 1724 | Löschung | `schlimm` | `*(nicht da)*` | …es so sehr [___] wird oder meinen… |
| 1725 | Löschung | `wird` | `*(nicht da)*` | …so sehr schlimm [___] oder meinen sie… |
| 1726 | Löschung | `oder` | `*(nicht da)*` | …sehr schlimm wird [___] meinen sie dass… |
| 1727 | Löschung | `meinen` | `*(nicht da)*` | …schlimm wird oder [___] sie dass ich… |
| 1728 | Löschung | `sie` | `*(nicht da)*` | …wird oder meinen [___] dass ich morgen… |
| 1729 | Löschung | `dass` | `*(nicht da)*` | …oder meinen sie [___] ich morgen wieder… |
| 1730 | Löschung | `ich` | `*(nicht da)*` | …meinen sie dass [___] morgen wieder arbeiten… |
| 1731 | Löschung | `morgen` | `*(nicht da)*` | …sie dass ich [___] wieder arbeiten gehen… |
| 1732 | Löschung | `wieder` | `*(nicht da)*` | …dass ich morgen [___] arbeiten gehen kann… |
| 1733 | Löschung | `arbeiten` | `*(nicht da)*` | …ich morgen wieder [___] gehen kann weil… |
| 1734 | Löschung | `gehen` | `*(nicht da)*` | …morgen wieder arbeiten [___] kann weil ich… |
| 1735 | Löschung | `kann` | `*(nicht da)*` | …wieder arbeiten gehen [___] weil ich ein… |
| 1736 | Löschung | `weil` | `*(nicht da)*` | …arbeiten gehen kann [___] ich ein sehr… |
| 1737 | Löschung | `ich` | `*(nicht da)*` | …gehen kann weil [___] ein sehr wichtiges… |
| 1738 | Löschung | `ein` | `*(nicht da)*` | …kann weil ich [___] sehr wichtiges meeting… |
| 1739 | Löschung | `sehr` | `*(nicht da)*` | …weil ich ein [___] wichtiges meeting habe… |
| 1740 | Löschung | `wichtiges` | `*(nicht da)*` | …ich ein sehr [___] meeting habe morgen… |
| 1741 | Löschung | `meeting` | `*(nicht da)*` | …ein sehr wichtiges [___] habe morgen und… |
| 1742 | Löschung | `habe` | `*(nicht da)*` | …sehr wichtiges meeting [___] morgen und wenn… |
| 1743 | Löschung | `morgen` | `*(nicht da)*` | …wichtiges meeting habe [___] und wenn ich… |
| 1744 | Löschung | `und` | `*(nicht da)*` | …meeting habe morgen [___] wenn ich nicht… |
| 1745 | Löschung | `wenn` | `*(nicht da)*` | …habe morgen und [___] ich nicht selbst… |
| 1746 | Löschung | `ich` | `*(nicht da)*` | …morgen und wenn [___] nicht selbst hingehen… |
| 1747 | Löschung | `nicht` | `*(nicht da)*` | …und wenn ich [___] selbst hingehen kann… |
| 1748 | Löschung | `selbst` | `*(nicht da)*` | …wenn ich nicht [___] hingehen kann dann… |
| 1749 | Löschung | `hingehen` | `*(nicht da)*` | …ich nicht selbst [___] kann dann müsste… |
| 1750 | Löschung | `kann` | `*(nicht da)*` | …nicht selbst hingehen [___] dann müsste ich… |
| 1751 | Löschung | `dann` | `*(nicht da)*` | …selbst hingehen kann [___] müsste ich mich… |
| 1752 | Löschung | `müsste` | `*(nicht da)*` | …hingehen kann dann [___] ich mich darum… |
| 1753 | Löschung | `ich` | `*(nicht da)*` | …kann dann müsste [___] mich darum kümmern… |
| 1754 | Löschung | `mich` | `*(nicht da)*` | …dann müsste ich [___] darum kümmern und… |
| 1755 | Löschung | `darum` | `*(nicht da)*` | …müsste ich mich [___] kümmern und eine… |
| 1756 | Löschung | `kümmern` | `*(nicht da)*` | …ich mich darum [___] und eine vertretung… |
| 1757 | Löschung | `und` | `*(nicht da)*` | …mich darum kümmern [___] eine vertretung finden… |
| 1758 | Löschung | `eine` | `*(nicht da)*` | …darum kümmern und [___] vertretung finden ich… |
| 1759 | Löschung | `vertretung` | `*(nicht da)*` | …kümmern und eine [___] finden ich werde… |
| 1760 | Löschung | `finden` | `*(nicht da)*` | …und eine vertretung [___] ich werde ihnen… |
| 1761 | Löschung | `ich` | `*(nicht da)*` | …eine vertretung finden [___] werde ihnen jetzt… |
| 1762 | Löschung | `werde` | `*(nicht da)*` | …vertretung finden ich [___] ihnen jetzt etwas… |
| 1763 | Löschung | `ihnen` | `*(nicht da)*` | …finden ich werde [___] jetzt etwas sagen… |
| 1764 | Löschung | `jetzt` | `*(nicht da)*` | …ich werde ihnen [___] etwas sagen was… |
| 1765 | Löschung | `etwas` | `*(nicht da)*` | …werde ihnen jetzt [___] sagen was sie… |
| 1766 | Löschung | `sagen` | `*(nicht da)*` | …ihnen jetzt etwas [___] was sie wahrscheinlich… |
| 1767 | Löschung | `was` | `*(nicht da)*` | …jetzt etwas sagen [___] sie wahrscheinlich nicht… |
| 1768 | Löschung | `sie` | `*(nicht da)*` | …etwas sagen was [___] wahrscheinlich nicht so… |
| 1769 | Löschung | `wahrscheinlich` | `*(nicht da)*` | …sagen was sie [___] nicht so gerne… |
| 1770 | Löschung | `nicht` | `*(nicht da)*` | …was sie wahrscheinlich [___] so gerne hören… |
| 1771 | Löschung | `so` | `*(nicht da)*` | …sie wahrscheinlich nicht [___] gerne hören möchten… |
| 1772 | Löschung | `gerne` | `*(nicht da)*` | …wahrscheinlich nicht so [___] hören möchten aber… |
| 1773 | Löschung | `hören` | `*(nicht da)*` | …nicht so gerne [___] möchten aber da… |
| 1774 | Löschung | `möchten` | `*(nicht da)*` | …so gerne hören [___] aber da kann… |
| 1775 | Löschung | `aber` | `*(nicht da)*` | …gerne hören möchten [___] da kann ich… |
| 1776 | Löschung | `da` | `*(nicht da)*` | …hören möchten aber [___] kann ich ihnen… |
| 1777 | Löschung | `kann` | `*(nicht da)*` | …möchten aber da [___] ich ihnen leider… |
| 1778 | Löschung | `ich` | `*(nicht da)*` | …aber da kann [___] ihnen leider gerade… |
| 1779 | Löschung | `ihnen` | `*(nicht da)*` | …da kann ich [___] leider gerade noch… |
| 1780 | Löschung | `leider` | `*(nicht da)*` | …kann ich ihnen [___] gerade noch keinerlei… |
| 1781 | Löschung | `gerade` | `*(nicht da)*` | …ich ihnen leider [___] noch keinerlei positive… |
| 1782 | Löschung | `noch` | `*(nicht da)*` | …ihnen leider gerade [___] keinerlei positive auskunft… |
| 1783 | Löschung | `keinerlei` | `*(nicht da)*` | …leider gerade noch [___] positive auskunft drüber… |
| 1784 | Löschung | `positive` | `*(nicht da)*` | …gerade noch keinerlei [___] auskunft drüber geben… |
| 1785 | Löschung | `auskunft` | `*(nicht da)*` | …noch keinerlei positive [___] drüber geben das… |
| 1786 | Löschung | `drüber` | `*(nicht da)*` | …keinerlei positive auskunft [___] geben das was… |
| 1787 | Löschung | `geben` | `*(nicht da)*` | …positive auskunft drüber [___] das was sie… |
| 1788 | Löschung | `das` | `*(nicht da)*` | …auskunft drüber geben [___] was sie beschreiben… |
| 1789 | Löschung | `was` | `*(nicht da)*` | …drüber geben das [___] sie beschreiben bezüglich… |
| 1790 | Löschung | `sie` | `*(nicht da)*` | …geben das was [___] beschreiben bezüglich ihres… |
| 1791 | Löschung | `beschreiben` | `*(nicht da)*` | …das was sie [___] bezüglich ihres knies… |
| 1792 | Löschung | `bezüglich` | `*(nicht da)*` | …was sie beschreiben [___] ihres knies und… |
| 1793 | Löschung | `ihres` | `*(nicht da)*` | …sie beschreiben bezüglich [___] knies und auch… |
| 1794 | Löschung | `knies` | `*(nicht da)*` | …beschreiben bezüglich ihres [___] und auch ihres… |
| 1795 | Löschung | `und` | `*(nicht da)*` | …bezüglich ihres knies [___] auch ihres daumens… |
| 1796 | Löschung | `auch` | `*(nicht da)*` | …ihres knies und [___] ihres daumens da… |
| 1797 | Löschung | `ihres` | `*(nicht da)*` | …knies und auch [___] daumens da müssen… |
| 1798 | Löschung | `daumens` | `*(nicht da)*` | …und auch ihres [___] da müssen wir… |
| 1799 | Löschung | `da` | `*(nicht da)*` | …auch ihres daumens [___] müssen wir wirklich… |
| 1800 | Löschung | `müssen` | `*(nicht da)*` | …ihres daumens da [___] wir wirklich erstmal… |
| 1801 | Löschung | `wir` | `*(nicht da)*` | …daumens da müssen [___] wirklich erstmal mrt… |
| 1802 | Löschung | `wirklich` | `*(nicht da)*` | …da müssen wir [___] erstmal mrt bilder… |
| 1803 | Löschung | `erstmal` | `*(nicht da)*` | …müssen wir wirklich [___] mrt bilder von… |
| 1804 | Löschung | `mrt` | `*(nicht da)*` | …wir wirklich erstmal [___] bilder von machen… |
| 1805 | Löschung | `bilder` | `*(nicht da)*` | …wirklich erstmal mrt [___] von machen und… |
| 1806 | Löschung | `von` | `*(nicht da)*` | …erstmal mrt bilder [___] machen und auch… |
| 1807 | Löschung | `machen` | `*(nicht da)*` | …mrt bilder von [___] und auch röntgenbilder… |
| 1808 | Löschung | `und` | `*(nicht da)*` | …bilder von machen [___] auch röntgenbilder von… |
| 1809 | Löschung | `auch` | `*(nicht da)*` | …von machen und [___] röntgenbilder von machen… |
| 1810 | Löschung | `röntgenbilder` | `*(nicht da)*` | …machen und auch [___] von machen um… |
| 1811 | Löschung | `von` | `*(nicht da)*` | …und auch röntgenbilder [___] machen um wirklich… |
| 1812 | Löschung | `machen` | `*(nicht da)*` | …auch röntgenbilder von [___] um wirklich zu… |
| 1813 | Löschung | `um` | `*(nicht da)*` | …röntgenbilder von machen [___] wirklich zu sehen… |
| 1814 | Löschung | `wirklich` | `*(nicht da)*` | …von machen um [___] zu sehen was… |
| 1815 | Löschung | `zu` | `*(nicht da)*` | …machen um wirklich [___] sehen was da… |
| 1816 | Löschung | `sehen` | `*(nicht da)*` | …um wirklich zu [___] was da los… |
| 1817 | Löschung | `was` | `*(nicht da)*` | …wirklich zu sehen [___] da los ist… |
| 1818 | Löschung | `da` | `*(nicht da)*` | …zu sehen was [___] los ist nicht… |
| 1819 | Löschung | `los` | `*(nicht da)*` | …sehen was da [___] ist nicht dass… |
| 1820 | Löschung | `ist` | `*(nicht da)*` | …was da los [___] nicht dass sie… |
| 1821 | Löschung | `nicht` | `*(nicht da)*` | …da los ist [___] dass sie sich… |
| 1822 | Löschung | `dass` | `*(nicht da)*` | …los ist nicht [___] sie sich etwas… |
| 1823 | Löschung | `sie` | `*(nicht da)*` | …ist nicht dass [___] sich etwas gebrochen… |
| 1824 | Löschung | `sich` | `*(nicht da)*` | …nicht dass sie [___] etwas gebrochen oder… |
| 1825 | Löschung | `etwas` | `*(nicht da)*` | …dass sie sich [___] gebrochen oder gerissen… |
| 1826 | Löschung | `gebrochen` | `*(nicht da)*` | …sie sich etwas [___] oder gerissen haben… |
| 1827 | Löschung | `oder` | `*(nicht da)*` | …sich etwas gebrochen [___] gerissen haben sollte… |
| 1828 | Löschung | `gerissen` | `*(nicht da)*` | …etwas gebrochen oder [___] haben sollte das… |
| 1829 | Löschung | `haben` | `*(nicht da)*` | …gebrochen oder gerissen [___] sollte das der… |
| 1830 | Löschung | `sollte` | `*(nicht da)*` | …oder gerissen haben [___] das der fall… |
| 1831 | Löschung | `das` | `*(nicht da)*` | …gerissen haben sollte [___] der fall sein… |
| 1832 | Löschung | `der` | `*(nicht da)*` | …haben sollte das [___] fall sein muss… |
| 1833 | Löschung | `fall` | `*(nicht da)*` | …sollte das der [___] sein muss man… |
| 1834 | Löschung | `sein` | `*(nicht da)*` | …das der fall [___] muss man abwägen… |
| 1835 | Löschung | `muss` | `*(nicht da)*` | …der fall sein [___] man abwägen ob… |
| 1836 | Löschung | `man` | `*(nicht da)*` | …fall sein muss [___] abwägen ob wir… |
| 1837 | Löschung | `abwägen` | `*(nicht da)*` | …sein muss man [___] ob wir sie… |
| 1838 | Löschung | `ob` | `*(nicht da)*` | …muss man abwägen [___] wir sie operieren… |
| 1839 | Löschung | `wir` | `*(nicht da)*` | …man abwägen ob [___] sie operieren oder… |
| 1840 | Löschung | `sie` | `*(nicht da)*` | …abwägen ob wir [___] operieren oder nicht… |
| 1841 | Löschung | `operieren` | `*(nicht da)*` | …ob wir sie [___] oder nicht das… |
| 1842 | Löschung | `oder` | `*(nicht da)*` | …wir sie operieren [___] nicht das können… |
| 1843 | Löschung | `nicht` | `*(nicht da)*` | …sie operieren oder [___] das können wir… |
| 1844 | Löschung | `das` | `*(nicht da)*` | …operieren oder nicht [___] können wir aber… |
| 1845 | Löschung | `können` | `*(nicht da)*` | …oder nicht das [___] wir aber alles… |
| 1846 | Löschung | `wir` | `*(nicht da)*` | …nicht das können [___] aber alles erst… |
| 1847 | Löschung | `aber` | `*(nicht da)*` | …das können wir [___] alles erst sagen… |
| 1848 | Löschung | `alles` | `*(nicht da)*` | …können wir aber [___] erst sagen wenn… |
| 1849 | Löschung | `erst` | `*(nicht da)*` | …wir aber alles [___] sagen wenn wir… |
| 1850 | Löschung | `sagen` | `*(nicht da)*` | …aber alles erst [___] wenn wir die… |
| 1851 | Löschung | `wenn` | `*(nicht da)*` | …alles erst sagen [___] wir die befunde… |
| 1852 | Löschung | `wir` | `*(nicht da)*` | …erst sagen wenn [___] die befunde da… |
| 1853 | Löschung | `die` | `*(nicht da)*` | …sagen wenn wir [___] befunde da haben… |
| 1854 | Löschung | `befunde` | `*(nicht da)*` | …wenn wir die [___] da haben ich… |
| 1855 | Löschung | `da` | `*(nicht da)*` | …wir die befunde [___] haben ich würde… |
| 1856 | Löschung | `haben` | `*(nicht da)*` | …die befunde da [___] ich würde auch… |
| 1857 | Löschung | `ich` | `*(nicht da)*` | …befunde da haben [___] würde auch gerne… |
| 1858 | Löschung | `würde` | `*(nicht da)*` | …da haben ich [___] auch gerne nochmal… |
| 1859 | Löschung | `auch` | `*(nicht da)*` | …haben ich würde [___] gerne nochmal eine… |
| 1860 | Löschung | `gerne` | `*(nicht da)*` | …ich würde auch [___] nochmal eine untersuchung… |
| 1861 | Löschung | `nochmal` | `*(nicht da)*` | …würde auch gerne [___] eine untersuchung mit… |
| 1862 | Löschung | `eine` | `*(nicht da)*` | …auch gerne nochmal [___] untersuchung mit ihrem… |
| 1863 | Löschung | `untersuchung` | `*(nicht da)*` | …gerne nochmal eine [___] mit ihrem kopf… |
| 1864 | Löschung | `mit` | `*(nicht da)*` | …nochmal eine untersuchung [___] ihrem kopf durchführen… |
| 1865 | Löschung | `ihrem` | `*(nicht da)*` | …eine untersuchung mit [___] kopf durchführen um… |
| 1866 | Löschung | `kopf` | `*(nicht da)*` | …untersuchung mit ihrem [___] durchführen um dort… |
| 1867 | Löschung | `durchführen` | `*(nicht da)*` | …mit ihrem kopf [___] um dort auch… |
| 1868 | Löschung | `um` | `*(nicht da)*` | …ihrem kopf durchführen [___] dort auch zu… |
| 1869 | Löschung | `dort` | `*(nicht da)*` | …kopf durchführen um [___] auch zu checken… |
| 1870 | Löschung | `auch` | `*(nicht da)*` | …durchführen um dort [___] zu checken dass… |
| 1871 | Löschung | `zu` | `*(nicht da)*` | …um dort auch [___] checken dass da… |
| 1872 | Löschung | `checken` | `*(nicht da)*` | …dort auch zu [___] dass da eben… |
| 1873 | Löschung | `dass` | `*(nicht da)*` | …auch zu checken [___] da eben alles… |
| 1874 | Löschung | `da` | `*(nicht da)*` | …zu checken dass [___] eben alles in… |
| 1875 | Löschung | `eben` | `*(nicht da)*` | …checken dass da [___] alles in ordnung… |
| 1876 | Löschung | `alles` | `*(nicht da)*` | …dass da eben [___] in ordnung ist… |
| 1877 | Löschung | `in` | `*(nicht da)*` | …da eben alles [___] ordnung ist und… |
| 1878 | Löschung | `ordnung` | `*(nicht da)*` | …eben alles in [___] ist und ja… |
| 1879 | Löschung | `ist` | `*(nicht da)*` | …alles in ordnung [___] und ja wenn… |
| 1880 | Löschung | `und` | `*(nicht da)*` | …in ordnung ist [___] ja wenn alles… |
| 1881 | Löschung | `ja` | `*(nicht da)*` | …ordnung ist und [___] wenn alles in… |
| 1882 | Löschung | `wenn` | `*(nicht da)*` | …ist und ja [___] alles in ordnung… |
| 1883 | Löschung | `alles` | `*(nicht da)*` | …und ja wenn [___] in ordnung ist… |
| 1884 | Löschung | `in` | `*(nicht da)*` | …ja wenn alles [___] ordnung ist würde… |
| 1885 | Löschung | `ordnung` | `*(nicht da)*` | …wenn alles in [___] ist würde ich… |
| 1886 | Löschung | `ist` | `*(nicht da)*` | …alles in ordnung [___] würde ich ihnen… |
| 1887 | Löschung | `würde` | `*(nicht da)*` | …in ordnung ist [___] ich ihnen trotzdem… |
| 1888 | Löschung | `ich` | `*(nicht da)*` | …ordnung ist würde [___] ihnen trotzdem raten… |
| 1889 | Löschung | `ihnen` | `*(nicht da)*` | …ist würde ich [___] trotzdem raten das… |
| 1890 | Löschung | `trotzdem` | `*(nicht da)*` | …würde ich ihnen [___] raten das meeting… |
| 1891 | Löschung | `raten` | `*(nicht da)*` | …ich ihnen trotzdem [___] das meeting vielleicht… |
| 1892 | Löschung | `das` | `*(nicht da)*` | …ihnen trotzdem raten [___] meeting vielleicht morgen… |
| 1893 | Löschung | `meeting` | `*(nicht da)*` | …trotzdem raten das [___] vielleicht morgen einmal… |
| 1894 | Löschung | `vielleicht` | `*(nicht da)*` | …raten das meeting [___] morgen einmal online… |
| 1895 | Löschung | `morgen` | `*(nicht da)*` | …das meeting vielleicht [___] einmal online durchzuführen… |
| 1896 | Löschung | `einmal` | `*(nicht da)*` | …meeting vielleicht morgen [___] online durchzuführen damit… |
| 1897 | Löschung | `online` | `*(nicht da)*` | …vielleicht morgen einmal [___] durchzuführen damit sie… |
| 1898 | Löschung | `durchzuführen` | `*(nicht da)*` | …morgen einmal online [___] damit sie sich… |
| 1899 | Löschung | `damit` | `*(nicht da)*` | …einmal online durchzuführen [___] sie sich schonen… |
| 1900 | Löschung | `sie` | `*(nicht da)*` | …online durchzuführen damit [___] sich schonen können… |
| 1901 | Löschung | `sich` | `*(nicht da)*` | …durchzuführen damit sie [___] schonen können aber… |
| 1902 | Löschung | `schonen` | `*(nicht da)*` | …damit sie sich [___] können aber genauere… |
| 1903 | Löschung | `können` | `*(nicht da)*` | …sie sich schonen [___] aber genauere auskunft… |
| 1904 | Löschung | `aber` | `*(nicht da)*` | …sich schonen können [___] genauere auskunft wie… |
| 1905 | Löschung | `genauere` | `*(nicht da)*` | …schonen können aber [___] auskunft wie bereits… |
| 1906 | Löschung | `auskunft` | `*(nicht da)*` | …können aber genauere [___] wie bereits gesagt… |
| 1907 | Löschung | `wie` | `*(nicht da)*` | …aber genauere auskunft [___] bereits gesagt kann… |
| 1908 | Löschung | `bereits` | `*(nicht da)*` | …genauere auskunft wie [___] gesagt kann ich… |
| 1909 | Löschung | `gesagt` | `*(nicht da)*` | …auskunft wie bereits [___] kann ich ihnen… |
| 1910 | Löschung | `kann` | `*(nicht da)*` | …wie bereits gesagt [___] ich ihnen erst… |
| 1911 | Löschung | `ich` | `*(nicht da)*` | …bereits gesagt kann [___] ihnen erst geben… |
| 1912 | Löschung | `ihnen` | `*(nicht da)*` | …gesagt kann ich [___] erst geben wenn… |
| 1913 | Löschung | `erst` | `*(nicht da)*` | …kann ich ihnen [___] geben wenn wir… |
| 1914 | Löschung | `geben` | `*(nicht da)*` | …ich ihnen erst [___] wenn wir alle… |
| 1915 | Löschung | `wenn` | `*(nicht da)*` | …ihnen erst geben [___] wir alle befunde… |
| 1916 | Löschung | `wir` | `*(nicht da)*` | …erst geben wenn [___] alle befunde da… |
| 1917 | Löschung | `alle` | `*(nicht da)*` | …geben wenn wir [___] befunde da haben… |
| 1918 | Löschung | `befunde` | `*(nicht da)*` | …wenn wir alle [___] da haben okay… |
| 1919 | Löschung | `da` | `*(nicht da)*` | …wir alle befunde [___] haben okay ich… |
| 1920 | Löschung | `haben` | `*(nicht da)*` | …alle befunde da [___] okay ich danke… |
| 1921 | Löschung | `okay` | `*(nicht da)*` | …befunde da haben [___] ich danke ihnen… |
| 1922 | Löschung | `ich` | `*(nicht da)*` | …da haben okay [___] danke ihnen online… |
| 1923 | Löschung | `danke` | `*(nicht da)*` | …haben okay ich [___] ihnen online wäre… |
| 1924 | Löschung | `ihnen` | `*(nicht da)*` | …okay ich danke [___] online wäre schwierig… |
| 1925 | Löschung | `online` | `*(nicht da)*` | …ich danke ihnen [___] wäre schwierig aber… |
| 1926 | Löschung | `wäre` | `*(nicht da)*` | …danke ihnen online [___] schwierig aber dann… |
| 1927 | Löschung | `schwierig` | `*(nicht da)*` | …ihnen online wäre [___] aber dann werde… |
| 1928 | Löschung | `aber` | `*(nicht da)*` | …online wäre schwierig [___] dann werde ich… |
| 1929 | Löschung | `dann` | `*(nicht da)*` | …wäre schwierig aber [___] werde ich jetzt… |
| 1930 | Löschung | `werde` | `*(nicht da)*` | …schwierig aber dann [___] ich jetzt gleich… |
| 1931 | Löschung | `ich` | `*(nicht da)*` | …aber dann werde [___] jetzt gleich dafür… |
| 1932 | Löschung | `jetzt` | `*(nicht da)*` | …dann werde ich [___] gleich dafür sorgen… |
| 1933 | Löschung | `gleich` | `*(nicht da)*` | …werde ich jetzt [___] dafür sorgen dass… |
| 1934 | Löschung | `dafür` | `*(nicht da)*` | …ich jetzt gleich [___] sorgen dass mich… |
| 1935 | Löschung | `sorgen` | `*(nicht da)*` | …jetzt gleich dafür [___] dass mich jemand… |
| 1936 | Löschung | `dass` | `*(nicht da)*` | …gleich dafür sorgen [___] mich jemand vertritt… |
| 1937 | Löschung | `mich` | `*(nicht da)*` | …dafür sorgen dass [___] jemand vertritt okay… |
| 1938 | Löschung | `jemand` | `*(nicht da)*` | …sorgen dass mich [___] vertritt okay sehr… |
| 1939 | Löschung | `vertritt` | `*(nicht da)*` | …dass mich jemand [___] okay sehr gut… |
| 1940 | Löschung | `okay` | `*(nicht da)*` | …mich jemand vertritt [___] sehr gut wir… |
| 1941 | Löschung | `sehr` | `*(nicht da)*` | …jemand vertritt okay [___] gut wir werden… |
| 1942 | Löschung | `gut` | `*(nicht da)*` | …vertritt okay sehr [___] wir werden auch… |
| 1943 | Löschung | `wir` | `*(nicht da)*` | …okay sehr gut [___] werden auch gleich… |
| 1944 | Löschung | `werden` | `*(nicht da)*` | …sehr gut wir [___] auch gleich die… |
| 1945 | Löschung | `auch` | `*(nicht da)*` | …gut wir werden [___] gleich die untersuchungen… |
| 1946 | Löschung | `gleich` | `*(nicht da)*` | …wir werden auch [___] die untersuchungen direkt… |
| 1947 | Löschung | `die` | `*(nicht da)*` | …werden auch gleich [___] untersuchungen direkt durchführen… |
| 1948 | Löschung | `untersuchungen` | `*(nicht da)*` | …auch gleich die [___] direkt durchführen wenn… |
| 1949 | Löschung | `direkt` | `*(nicht da)*` | …gleich die untersuchungen [___] durchführen wenn wir… |
| 1950 | Löschung | `durchführen` | `*(nicht da)*` | …die untersuchungen direkt [___] wenn wir mit… |
| 1951 | Löschung | `wenn` | `*(nicht da)*` | …untersuchungen direkt durchführen [___] wir mit der… |
| 1952 | Löschung | `wir` | `*(nicht da)*` | …direkt durchführen wenn [___] mit der aufnahme… |
| 1953 | Löschung | `mit` | `*(nicht da)*` | …durchführen wenn wir [___] der aufnahme fertig… |
| 1954 | Löschung | `der` | `*(nicht da)*` | …wenn wir mit [___] aufnahme fertig sind… |
| 1955 | Löschung | `aufnahme` | `*(nicht da)*` | …wir mit der [___] fertig sind dann… |
| 1956 | Löschung | `fertig` | `*(nicht da)*` | …mit der aufnahme [___] sind dann würde… |
| 1957 | Löschung | `sind` | `*(nicht da)*` | …der aufnahme fertig [___] dann würde ich… |
| 1958 | Löschung | `dann` | `*(nicht da)*` | …aufnahme fertig sind [___] würde ich sie… |
| 1959 | Löschung | `würde` | `*(nicht da)*` | …fertig sind dann [___] ich sie bitten… |
| 1960 | Löschung | `ich` | `*(nicht da)*` | …sind dann würde [___] sie bitten schon… |
| 1961 | Löschung | `sie` | `*(nicht da)*` | …dann würde ich [___] bitten schon mal… |
| 1962 | Löschung | `bitten` | `*(nicht da)*` | …würde ich sie [___] schon mal rüber… |
| 1963 | Löschung | `schon` | `*(nicht da)*` | …ich sie bitten [___] mal rüber ins… |
| 1964 | Löschung | `mal` | `*(nicht da)*` | …sie bitten schon [___] rüber ins untersuchungszimmer… |
| 1965 | Löschung | `rüber` | `*(nicht da)*` | …bitten schon mal [___] ins untersuchungszimmer zu… |
| 1966 | Löschung | `ins` | `*(nicht da)*` | …schon mal rüber [___] untersuchungszimmer zu gehen… |
| 1967 | Löschung | `untersuchungszimmer` | `*(nicht da)*` | …mal rüber ins [___] zu gehen und… |
| 1968 | Löschung | `zu` | `*(nicht da)*` | …rüber ins untersuchungszimmer [___] gehen und dann… |
| 1969 | Löschung | `gehen` | `*(nicht da)*` | …ins untersuchungszimmer zu [___] und dann geht… |
| 1970 | Löschung | `und` | `*(nicht da)*` | …untersuchungszimmer zu gehen [___] dann geht es… |
| 1971 | Löschung | `dann` | `*(nicht da)*` | …zu gehen und [___] geht es dort… |
| 1972 | Löschung | `geht` | `*(nicht da)*` | …gehen und dann [___] es dort auch… |
| 1973 | Löschung | `es` | `*(nicht da)*` | …und dann geht [___] dort auch gleich… |
| 1974 | Löschung | `dort` | `*(nicht da)*` | …dann geht es [___] auch gleich los… |
| 1975 | Löschung | `auch` | `*(nicht da)*` | …geht es dort [___] gleich los alles… |
| 1976 | Löschung | `gleich` | `*(nicht da)*` | …es dort auch [___] los alles klar… |
| 1977 | Löschung | `los` | `*(nicht da)*` | …dort auch gleich [___] alles klar noch… |
| 1978 | Löschung | `alles` | `*(nicht da)*` | …auch gleich los [___] klar noch mal… |
| 1979 | Löschung | `klar` | `*(nicht da)*` | …gleich los alles [___] noch mal kurz… |
| 1980 | Löschung | `noch` | `*(nicht da)*` | …los alles klar [___] mal kurz zum… |
| 1981 | Löschung | `mal` | `*(nicht da)*` | …alles klar noch [___] kurz zum abgleich… |
| 1982 | Löschung | `kurz` | `*(nicht da)*` | …klar noch mal [___] zum abgleich sie… |
| 1983 | Löschung | `zum` | `*(nicht da)*` | …noch mal kurz [___] abgleich sie hatten… |
| 1984 | Löschung | `abgleich` | `*(nicht da)*` | …mal kurz zum [___] sie hatten einen… |
| 1985 | Löschung | `sie` | `*(nicht da)*` | …kurz zum abgleich [___] hatten einen fahrradunfall… |
| 1986 | Löschung | `hatten` | `*(nicht da)*` | …zum abgleich sie [___] einen fahrradunfall sind… |
| 1987 | Löschung | `einen` | `*(nicht da)*` | …abgleich sie hatten [___] fahrradunfall sind auf… |
| 1988 | Löschung | `fahrradunfall` | `*(nicht da)*` | …sie hatten einen [___] sind auf die… |
| 1989 | Substitution | `sind` | `diese` | …hatten einen fahrradunfall [___] auf die linke… |
| 1990 | Substitution | `und` | `ist` | …linke seite gestürzt [___] haben seitdem schmerzen… |
| 1991 | Substitution | `haben` | `und` | …seite gestürzt und [___] seitdem schmerzen auf… |
| 1992 | Einfügung | `*(nicht da)*` | `hat` | (FMT) …im linken knie [___] die schmerzen im… |
| 1993 | Löschung | `sie` | `*(nicht da)*` | …schon wesentlich stärker [___] haben die schmerzintensität… |
| 1994 | Löschung | `haben` | `*(nicht da)*` | …wesentlich stärker sie [___] die schmerzintensität dort… |
| 1995 | Löschung | `die` | `*(nicht da)*` | …stärker sie haben [___] schmerzintensität dort mit… |
| 1996 | Substitution | `schmerzintensität` | `und` | …sie haben die [___] dort mit einer… |
| 1997 | Substitution | `dort` | `wurden` | …haben die schmerzintensität [___] mit einer 7… |
| 1998 | Substitution | `mit` | `von` | …die schmerzintensität dort [___] einer 7 beschrieben… |
| 1999 | Substitution | `beschrieben` | `bis` | …mit einer 7 [___] und haben gesagt… |
| 2000 | Substitution | `und` | `einer` | …einer 7 beschrieben [___] haben gesagt dass… |
| 2001 | Substitution | `haben` | `8` | …7 beschrieben und [___] gesagt dass sie… |
| 2002 | Substitution | `gesagt` | `beschrieben` | …beschrieben und haben [___] dass sie den… |
| 2003 | Substitution | `dass` | `sie` | …und haben gesagt [___] sie den daumen… |
| 2004 | Substitution | `sie` | `kann` | …haben gesagt dass [___] den daumen auch… |
| 2005 | Löschung | `auch` | `*(nicht da)*` | …sie den daumen [___] nicht mehr bewegen… |
| 2006 | Substitution | `können` | `und` | …nicht mehr bewegen [___] der schmerz wurde… |
| 2007 | Substitution | `wurde` | `ist` | …können der schmerz [___] stechend beschrieben und… |
| 2008 | Löschung | `beschrieben` | `*(nicht da)*` | …schmerz wurde stechend [___] und gleiches gilt… |
| 2009 | Löschung | `und` | `*(nicht da)*` | …wurde stechend beschrieben [___] gleiches gilt für… |
| 2010 | Löschung | `gleiches` | `*(nicht da)*` | …stechend beschrieben und [___] gilt für das… |
| 2011 | Löschung | `gilt` | `*(nicht da)*` | …beschrieben und gleiches [___] für das knie… |
| 2012 | Löschung | `für` | `*(nicht da)*` | …und gleiches gilt [___] das knie auch… |
| 2013 | Löschung | `das` | `*(nicht da)*` | …gleiches gilt für [___] knie auch das… |
| 2014 | Löschung | `knie` | `*(nicht da)*` | …gilt für das [___] auch das knie… |
| 2015 | Löschung | `auch` | `*(nicht da)*` | …für das knie [___] das knie können… |
| 2016 | Löschung | `das` | `*(nicht da)*` | …das knie auch [___] knie können sie… |
| 2017 | Löschung | `knie` | `*(nicht da)*` | …knie auch das [___] können sie nicht… |
| 2018 | Löschung | `können` | `*(nicht da)*` | …auch das knie [___] sie nicht mehr… |
| 2019 | Löschung | `sie` | `*(nicht da)*` | …das knie können [___] nicht mehr bewegen… |
| 2020 | Löschung | `nicht` | `*(nicht da)*` | …knie können sie [___] mehr bewegen im… |
| 2021 | Löschung | `mehr` | `*(nicht da)*` | …können sie nicht [___] bewegen im ruhezustand… |
| 2022 | Löschung | `bewegen` | `*(nicht da)*` | …sie nicht mehr [___] im ruhezustand wurde… |
| 2023 | Löschung | `im` | `*(nicht da)*` | …nicht mehr bewegen [___] ruhezustand wurde die… |
| 2024 | Löschung | `ruhezustand` | `*(nicht da)*` | …mehr bewegen im [___] wurde die schmerzintensität… |
| 2025 | Löschung | `wurde` | `*(nicht da)*` | …bewegen im ruhezustand [___] die schmerzintensität mit… |
| 2026 | Löschung | `schmerzintensität` | `*(nicht da)*` | …ruhezustand wurde die [___] mit einer 8… |
| 2027 | Löschung | `mit` | `*(nicht da)*` | …wurde die schmerzintensität [___] einer 8 beschrieben… |
| 2028 | Löschung | `einer` | `*(nicht da)*` | …die schmerzintensität mit [___] 8 beschrieben bei… |
| 2029 | Löschung | `8` | `*(nicht da)*` | …schmerzintensität mit einer [___] beschrieben bei bewegung… |
| 2030 | Substitution | `beschrieben` | `ärztin` | …mit einer 8 [___] bei bewegung unerträglich… |
| 2031 | Substitution | `bei` | `möchte` | …einer 8 beschrieben [___] bewegung unerträglich also… |
| 2032 | Substitution | `bewegung` | `noch` | …8 beschrieben bei [___] unerträglich also 10… |
| 2033 | Substitution | `unerträglich` | `einige` | …beschrieben bei bewegung [___] also 10 oder… |
| 2034 | Substitution | `also` | `fragen` | …bei bewegung unerträglich [___] 10 oder mehr… |
| 2035 | Substitution | `10` | `stellen` | …bewegung unerträglich also [___] oder mehr als… |
| 2036 | Substitution | `oder` | `um` | …unerträglich also 10 [___] mehr als 10… |
| 2037 | Löschung | `als` | `*(nicht da)*` | …10 oder mehr [___] 10 auch dieser… |
| 2038 | Löschung | `10` | `*(nicht da)*` | …oder mehr als [___] auch dieser schmerz… |
| 2039 | Löschung | `auch` | `*(nicht da)*` | …mehr als 10 [___] dieser schmerz ist… |
| 2040 | Löschung | `dieser` | `*(nicht da)*` | …als 10 auch [___] schmerz ist stechend… |
| 2041 | Löschung | `schmerz` | `*(nicht da)*` | …10 auch dieser [___] ist stechend ansonsten… |
| 2042 | Löschung | `ist` | `*(nicht da)*` | …auch dieser schmerz [___] stechend ansonsten sind… |
| 2043 | Löschung | `stechend` | `*(nicht da)*` | …dieser schmerz ist [___] ansonsten sind daumen… |
| 2044 | Löschung | `ansonsten` | `*(nicht da)*` | …schmerz ist stechend [___] sind daumen wie… |
| 2045 | Löschung | `sind` | `*(nicht da)*` | …ist stechend ansonsten [___] daumen wie auch… |
| 2046 | Löschung | `daumen` | `*(nicht da)*` | …stechend ansonsten sind [___] wie auch knie… |
| 2047 | Substitution | `wie` | `über` | …ansonsten sind daumen [___] auch knie geschwollen… |
| 2048 | Substitution | `auch` | `die` | …sind daumen wie [___] knie geschwollen richtig… |
| 2049 | Substitution | `knie` | `situation` | …daumen wie auch [___] geschwollen richtig sie… |
| 2050 | Substitution | `geschwollen` | `zu` | …wie auch knie [___] richtig sie haben… |
| 2051 | Substitution | `richtig` | `erfahren` | …auch knie geschwollen [___] sie haben gesagt… |
| 2052 | Substitution | `sie` | `*` | …knie geschwollen richtig [___] haben gesagt dass… |
| 2053 | Löschung | `gesagt` | `*(nicht da)*` | …richtig sie haben [___] dass sie das… |
| 2054 | Löschung | `dass` | `*(nicht da)*` | …sie haben gesagt [___] sie das bewusstsein… |
| 2055 | Löschung | `das` | `*(nicht da)*` | …gesagt dass sie [___] bewusstsein nicht verloren… |
| 2056 | Löschung | `bewusstsein` | `*(nicht da)*` | …dass sie das [___] nicht verloren haben… |
| 2057 | Löschung | `nicht` | `*(nicht da)*` | …sie das bewusstsein [___] verloren haben bei… |
| 2058 | Löschung | `verloren` | `*(nicht da)*` | …das bewusstsein nicht [___] haben bei dem… |
| 2059 | Löschung | `haben` | `*(nicht da)*` | …bewusstsein nicht verloren [___] bei dem unfall… |
| 2060 | Löschung | `bei` | `*(nicht da)*` | …nicht verloren haben [___] dem unfall dass… |
| 2061 | Löschung | `dem` | `*(nicht da)*` | …verloren haben bei [___] unfall dass sie… |
| 2062 | Substitution | `unfall` | `in` | …haben bei dem [___] dass sie nur… |
| 2063 | Substitution | `dass` | `der` | …bei dem unfall [___] sie nur kurz… |
| 2064 | Substitution | `sie` | `letzten` | …dem unfall dass [___] nur kurz danach… |
| 2065 | Substitution | `nur` | `zeit` | …unfall dass sie [___] kurz danach recht… |
| 2066 | Substitution | `kurz` | `im` | …dass sie nur [___] danach recht schwindelig… |
| 2067 | Substitution | `danach` | `ausland` | …sie nur kurz [___] recht schwindelig waren… |
| 2068 | Substitution | `recht` | `gewesen` | …nur kurz danach [___] schwindelig waren das… |
| 2069 | Substitution | `schwindelig` | `*` | …kurz danach recht [___] waren das sei… |
| 2070 | Löschung | `das` | `*(nicht da)*` | …recht schwindelig waren [___] sei aber schon… |
| 2071 | Löschung | `sei` | `*(nicht da)*` | …schwindelig waren das [___] aber schon wieder… |
| 2072 | Löschung | `aber` | `*(nicht da)*` | …waren das sei [___] schon wieder vorbei… |
| 2073 | Löschung | `schon` | `*(nicht da)*` | …das sei aber [___] wieder vorbei genau… |
| 2074 | Löschung | `wieder` | `*(nicht da)*` | …sei aber schon [___] vorbei genau vorerkrankungen… |
| 2075 | Löschung | `vorbei` | `*(nicht da)*` | …aber schon wieder [___] genau vorerkrankungen haben… |
| 2076 | Löschung | `genau` | `*(nicht da)*` | …schon wieder vorbei [___] vorerkrankungen haben sie… |
| 2077 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …wieder vorbei genau [___] haben sie keine… |
| 2078 | Löschung | `haben` | `*(nicht da)*` | …vorbei genau vorerkrankungen [___] sie keine medikamente… |
| 2079 | Löschung | `sie` | `*(nicht da)*` | …genau vorerkrankungen haben [___] keine medikamente nehmen… |
| 2080 | Löschung | `keine` | `*(nicht da)*` | …vorerkrankungen haben sie [___] medikamente nehmen sie… |
| 2081 | Löschung | `medikamente` | `*(nicht da)*` | …haben sie keine [___] nehmen sie auch… |
| 2082 | Löschung | `nehmen` | `*(nicht da)*` | …sie keine medikamente [___] sie auch keine… |
| 2083 | Löschung | `sie` | `*(nicht da)*` | …keine medikamente nehmen [___] auch keine regelmäßig… |
| 2084 | Löschung | `auch` | `*(nicht da)*` | …medikamente nehmen sie [___] keine regelmäßig ein… |
| 2085 | Löschung | `keine` | `*(nicht da)*` | …nehmen sie auch [___] regelmäßig ein außer… |
| 2086 | Löschung | `regelmäßig` | `*(nicht da)*` | …sie auch keine [___] ein außer der… |
| 2087 | Löschung | `ein` | `*(nicht da)*` | …auch keine regelmäßig [___] außer der pille… |
| 2088 | Löschung | `außer` | `*(nicht da)*` | …keine regelmäßig ein [___] der pille sie… |
| 2089 | Löschung | `der` | `*(nicht da)*` | …regelmäßig ein außer [___] pille sie hatten… |
| 2090 | Löschung | `pille` | `*(nicht da)*` | …ein außer der [___] sie hatten eine… |
| 2091 | Löschung | `hatten` | `*(nicht da)*` | …der pille sie [___] eine operation am… |
| 2092 | Löschung | `eine` | `*(nicht da)*` | …pille sie hatten [___] operation am rechten… |
| 2093 | Löschung | `operation` | `*(nicht da)*` | …sie hatten eine [___] am rechten fuß… |
| 2094 | Löschung | `am` | `*(nicht da)*` | …hatten eine operation [___] rechten fuß vor… |
| 2095 | Löschung | `rechten` | `*(nicht da)*` | …eine operation am [___] fuß vor zwei… |
| 2096 | Löschung | `fuß` | `*(nicht da)*` | …operation am rechten [___] vor zwei jahren… |
| 2097 | Substitution | `zwei` | `dem` | …rechten fuß vor [___] jahren da wurde… |
| 2098 | Substitution | `jahren` | `unfall` | …fuß vor zwei [___] da wurde der… |
| 2099 | Substitution | `da` | `gesund` | …vor zwei jahren [___] wurde der halux… |
| 2100 | Substitution | `wurde` | `*` | …zwei jahren da [___] der halux valgus… |
| 2101 | Substitution | `der` | `haben` | …jahren da wurde [___] halux valgus operiert… |
| 2102 | Substitution | `halux` | `sie` | …da wurde der [___] valgus operiert ansonsten… |
| 2103 | Substitution | `valgus` | `fieber` | …wurde der halux [___] operiert ansonsten körperliche… |
| 2104 | Substitution | `operiert` | `schüttelfrost` | …der halux valgus [___] ansonsten körperliche beschwerden… |
| 2105 | Substitution | `ansonsten` | `oder` | …halux valgus operiert [___] körperliche beschwerden gibt… |
| 2106 | Substitution | `körperliche` | `nachtschweiß` | …valgus operiert ansonsten [___] beschwerden gibt es… |
| 2107 | Substitution | `beschwerden` | `*` | …operiert ansonsten körperliche [___] gibt es keine… |
| 2108 | Löschung | `keine` | `*(nicht da)*` | …beschwerden gibt es [___] sie sind ansonsten… |
| 2109 | Löschung | `sie` | `*(nicht da)*` | …gibt es keine [___] sind ansonsten gesund… |
| 2110 | Substitution | `sind` | `andere` | …es keine sie [___] ansonsten gesund gott… |
| 2111 | Substitution | `ansonsten` | `körperliche` | …keine sie sind [___] gesund gott sei… |
| 2112 | Substitution | `gesund` | `beschwerden` | …sie sind ansonsten [___] gott sei dank… |
| 2113 | Substitution | `gott` | `außerhalb` | …sind ansonsten gesund [___] sei dank bis… |
| 2114 | Substitution | `sei` | `des` | …ansonsten gesund gott [___] dank bis auf… |
| 2115 | Substitution | `dank` | `knie` | …gesund gott sei [___] bis auf die… |
| 2116 | Substitution | `bis` | `und` | …gott sei dank [___] auf die kistaminunverträglichkeit… |
| 2117 | Substitution | `auf` | `daumenschmerzes` | …sei dank bis [___] die kistaminunverträglichkeit genau… |
| 2118 | Löschung | `kistaminunverträglichkeit` | `*(nicht da)*` | …bis auf die [___] genau das hätte… |
| 2119 | Löschung | `genau` | `*(nicht da)*` | …auf die kistaminunverträglichkeit [___] das hätte ich… |
| 2120 | Löschung | `das` | `*(nicht da)*` | …die kistaminunverträglichkeit genau [___] hätte ich jetzt… |
| 2121 | Löschung | `hätte` | `*(nicht da)*` | …kistaminunverträglichkeit genau das [___] ich jetzt auch… |
| 2122 | Substitution | `ich` | `ärztin` | …genau das hätte [___] jetzt auch noch… |
| 2123 | Substitution | `jetzt` | `möchte` | …das hätte ich [___] auch noch mit… |
| 2124 | Löschung | `mit` | `*(nicht da)*` | …jetzt auch noch [___] eingebracht vielen dank… |
| 2125 | Löschung | `eingebracht` | `*(nicht da)*` | …auch noch mit [___] vielen dank nochmal… |
| 2126 | Löschung | `vielen` | `*(nicht da)*` | …noch mit eingebracht [___] dank nochmal dafür… |
| 2127 | Löschung | `dank` | `*(nicht da)*` | …mit eingebracht vielen [___] nochmal dafür habe… |
| 2128 | Löschung | `nochmal` | `*(nicht da)*` | …eingebracht vielen dank [___] dafür habe ich… |
| 2129 | Löschung | `dafür` | `*(nicht da)*` | …vielen dank nochmal [___] habe ich mir… |
| 2130 | Löschung | `habe` | `*(nicht da)*` | …dank nochmal dafür [___] ich mir notiert… |
| 2131 | Löschung | `ich` | `*(nicht da)*` | …nochmal dafür habe [___] mir notiert genau… |
| 2132 | Löschung | `mir` | `*(nicht da)*` | …dafür habe ich [___] notiert genau es… |
| 2133 | Löschung | `notiert` | `*(nicht da)*` | …habe ich mir [___] genau es gibt… |
| 2134 | Löschung | `genau` | `*(nicht da)*` | …ich mir notiert [___] es gibt ein… |
| 2135 | Substitution | `es` | `einige` | …mir notiert genau [___] gibt ein paar… |
| 2136 | Substitution | `gibt` | `untersuchungen` | …notiert genau es [___] ein paar vorerkrankungen… |
| 2137 | Substitution | `ein` | `durchführen` | …genau es gibt [___] paar vorerkrankungen in… |
| 2138 | Substitution | `paar` | `um` | …es gibt ein [___] vorerkrankungen in der… |
| 2139 | Substitution | `vorerkrankungen` | `die` | …gibt ein paar [___] in der familiengeschichte… |
| 2140 | Substitution | `in` | `ursache` | …ein paar vorerkrankungen [___] der familiengeschichte sie… |
| 2141 | Löschung | `familiengeschichte` | `*(nicht da)*` | …vorerkrankungen in der [___] sie sind frisch… |
| 2142 | Löschung | `sie` | `*(nicht da)*` | …in der familiengeschichte [___] sind frisch verheiratet… |
| 2143 | Löschung | `sind` | `*(nicht da)*` | …der familiengeschichte sie [___] frisch verheiratet und… |
| 2144 | Löschung | `frisch` | `*(nicht da)*` | …familiengeschichte sie sind [___] verheiratet und arbeiten… |
| 2145 | Löschung | `verheiratet` | `*(nicht da)*` | …sie sind frisch [___] und arbeiten in… |
| 2146 | Löschung | `und` | `*(nicht da)*` | …sind frisch verheiratet [___] arbeiten in einer… |
| 2147 | Löschung | `arbeiten` | `*(nicht da)*` | …frisch verheiratet und [___] in einer marketingagentur… |
| 2148 | Löschung | `in` | `*(nicht da)*` | …verheiratet und arbeiten [___] einer marketingagentur ja… |
| 2149 | Löschung | `einer` | `*(nicht da)*` | …und arbeiten in [___] marketingagentur ja das… |
| 2150 | Löschung | `marketingagentur` | `*(nicht da)*` | …arbeiten in einer [___] ja das ist… |
| 2151 | Löschung | `ja` | `*(nicht da)*` | …in einer marketingagentur [___] das ist alles… |
| 2152 | Löschung | `das` | `*(nicht da)*` | …einer marketingagentur ja [___] ist alles richtig… |
| 2153 | Löschung | `ist` | `*(nicht da)*` | …marketingagentur ja das [___] alles richtig ja… |
| 2154 | Löschung | `alles` | `*(nicht da)*` | …ja das ist [___] richtig ja perfekt… |
| 2155 | Löschung | `richtig` | `*(nicht da)*` | …das ist alles [___] ja perfekt sehr… |
| 2156 | Löschung | `ja` | `*(nicht da)*` | …ist alles richtig [___] perfekt sehr gut… |
| 2157 | Löschung | `perfekt` | `*(nicht da)*` | …alles richtig ja [___] sehr gut frau… |
| 2158 | Löschung | `sehr` | `*(nicht da)*` | …richtig ja perfekt [___] gut frau becken… |
| 2159 | Löschung | `gut` | `*(nicht da)*` | …ja perfekt sehr [___] frau becken westfalen… |
| 2160 | Löschung | `frau` | `*(nicht da)*` | …perfekt sehr gut [___] becken westfalen dann… |
| 2161 | Löschung | `becken` | `*(nicht da)*` | …sehr gut frau [___] westfalen dann war… |
| 2162 | Löschung | `westfalen` | `*(nicht da)*` | …gut frau becken [___] dann war es… |
| 2163 | Substitution | `dann` | `schmerzen` | …frau becken westfalen [___] war es das… |
| 2164 | Substitution | `war` | `zu` | …becken westfalen dann [___] es das jetzt… |
| 2165 | Substitution | `es` | `ermitteln` | …westfalen dann war [___] das jetzt erstmal… |
| 2166 | Substitution | `das` | `dazu` | …dann war es [___] jetzt erstmal von… |
| 2167 | Substitution | `jetzt` | `gehören` | …war es das [___] erstmal von meiner… |
| 2168 | Substitution | `erstmal` | `*` | …es das jetzt [___] von meiner seite… |
| 2169 | Substitution | `von` | `eine` | …das jetzt erstmal [___] meiner seite wir… |
| 2170 | Substitution | `meiner` | `mrt` | …jetzt erstmal von [___] seite wir machen… |
| 2171 | Substitution | `seite` | `bildung*` | …erstmal von meiner [___] wir machen jetzt… |
| 2172 | Substitution | `wir` | `röntgenbilder*` | …von meiner seite [___] machen jetzt mit… |
| 2173 | Substitution | `machen` | `eine` | …meiner seite wir [___] jetzt mit den… |
| 2174 | Substitution | `jetzt` | `untersuchung` | …seite wir machen [___] mit den untersuchungen… |
| 2175 | Löschung | `den` | `*(nicht da)*` | …machen jetzt mit [___] untersuchungen weiter ich… |
| 2176 | Löschung | `untersuchungen` | `*(nicht da)*` | …jetzt mit den [___] weiter ich bin… |
| 2177 | Löschung | `weiter` | `*(nicht da)*` | …mit den untersuchungen [___] ich bin in… |
| 2178 | Löschung | `ich` | `*(nicht da)*` | …den untersuchungen weiter [___] bin in kurzer… |
| 2179 | Löschung | `bin` | `*(nicht da)*` | …untersuchungen weiter ich [___] in kurzer zeit… |
| 2180 | Löschung | `in` | `*(nicht da)*` | …weiter ich bin [___] kurzer zeit wieder… |
| 2181 | Löschung | `kurzer` | `*(nicht da)*` | …ich bin in [___] zeit wieder für… |
| 2182 | Löschung | `zeit` | `*(nicht da)*` | …bin in kurzer [___] wieder für sie… |
| 2183 | Löschung | `wieder` | `*(nicht da)*` | …in kurzer zeit [___] für sie da… |
| 2184 | Löschung | `für` | `*(nicht da)*` | …kurzer zeit wieder [___] sie da okay… |
| 2185 | Löschung | `sie` | `*(nicht da)*` | …zeit wieder für [___] da okay alles… |
| 2186 | Löschung | `da` | `*(nicht da)*` | …wieder für sie [___] okay alles klar… |
| 2187 | Löschung | `okay` | `*(nicht da)*` | …für sie da [___] alles klar ich… |
| 2188 | Löschung | `alles` | `*(nicht da)*` | …sie da okay [___] klar ich warte… |
| 2189 | Löschung | `klar` | `*(nicht da)*` | …da okay alles [___] ich warte dann… |
| 2190 | Löschung | `ich` | `*(nicht da)*` | …okay alles klar [___] warte dann hier… |
| 2191 | Löschung | `warte` | `*(nicht da)*` | …alles klar ich [___] dann hier vielen… |
| 2192 | Löschung | `dann` | `*(nicht da)*` | …klar ich warte [___] hier vielen dank… |
| 2193 | Löschung | `hier` | `*(nicht da)*` | …ich warte dann [___] vielen dank ihnen… |
| 2194 | Löschung | `vielen` | `*(nicht da)*` | …warte dann hier [___] dank ihnen super… |
| 2195 | Löschung | `dank` | `*(nicht da)*` | …dann hier vielen [___] ihnen super besten… |
| 2196 | Löschung | `ihnen` | `*(nicht da)*` | …hier vielen dank [___] super besten dank… |
| 2197 | Löschung | `super` | `*(nicht da)*` | …vielen dank ihnen [___] besten dank und… |
| 2198 | Löschung | `besten` | `*(nicht da)*` | …dank ihnen super [___] dank und bis… |
| 2199 | Löschung | `dank` | `*(nicht da)*` | …ihnen super besten [___] und bis gleich… |
| 2200 | Löschung | `und` | `*(nicht da)*` | …super besten dank [___] bis gleich bis… |
| 2201 | Löschung | `bis` | `*(nicht da)*` | …besten dank und [___] gleich bis gleich… |
| 2202 | Löschung | `gleich` | `*(nicht da)*` | …dank und bis [___] bis gleich… |
| 2203 | Substitution | `bis` | `dem` | …und bis gleich [___] gleich… |
| 2204 | Substitution | `gleich` | `kopf` | …bis gleich bis [___]… |

---

## PWC

**Fehlerrate: 20.8%** — RAW: 1511 Wörter | FMT: 1203 Wörter | S=4 D=309 I=1 | Fehler=314

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Substitution | `sind` | `ist` | …warum sie hier [___] und was wir… |
| 2 | Einfügung | `*(nicht da)*` | `und…` | (FMT) …in der therapie [___] okay ja das… |
| 3 | Substitution | `und…okay` | `okay` | …in der therapie [___] ja das war… |
| 4 | Löschung | `je` | `*(nicht da)*` | …okay je nachdem [___] nach belastung je… |
| 5 | Löschung | `nach` | `*(nicht da)*` | …je nachdem je [___] belastung je nach… |
| 6 | Löschung | `belastung` | `*(nicht da)*` | …nachdem je nach [___] je nach belastung… |
| 7 | Löschung | `je` | `*(nicht da)*` | …je nach belastung [___] nach belastung es… |
| 8 | Löschung | `nach` | `*(nicht da)*` | …nach belastung je [___] belastung es war… |
| 9 | Löschung | `belastung` | `*(nicht da)*` | …belastung je nach [___] es war halt…… |
| 10 | Löschung | `es` | `*(nicht da)*` | …je nach belastung [___] war halt… ich… |
| 11 | Löschung | `war` | `*(nicht da)*` | …nach belastung es [___] halt… ich habe… |
| 12 | Löschung | `halt…` | `*(nicht da)*` | …belastung es war [___] ich habe mich… |
| 13 | Löschung | `ich` | `*(nicht da)*` | …es war halt… [___] habe mich halt… |
| 14 | Löschung | `habe` | `*(nicht da)*` | …war halt… ich [___] mich halt nicht… |
| 15 | Löschung | `mich` | `*(nicht da)*` | …halt… ich habe [___] halt nicht viel… |
| 16 | Löschung | `halt` | `*(nicht da)*` | …ich habe mich [___] nicht viel bewegen… |
| 17 | Löschung | `nicht` | `*(nicht da)*` | …habe mich halt [___] viel bewegen können… |
| 18 | Löschung | `viel` | `*(nicht da)*` | …mich halt nicht [___] bewegen können ich… |
| 19 | Löschung | `bewegen` | `*(nicht da)*` | …halt nicht viel [___] können ich bin… |
| 20 | Löschung | `können` | `*(nicht da)*` | …nicht viel bewegen [___] ich bin ja… |
| 21 | Löschung | `ich` | `*(nicht da)*` | …viel bewegen können [___] bin ja eigentlich… |
| 22 | Löschung | `bin` | `*(nicht da)*` | …bewegen können ich [___] ja eigentlich nur… |
| 23 | Löschung | `ja` | `*(nicht da)*` | …können ich bin [___] eigentlich nur gelegen… |
| 24 | Löschung | `eigentlich` | `*(nicht da)*` | …ich bin ja [___] nur gelegen okay… |
| 25 | Löschung | `nur` | `*(nicht da)*` | …bin ja eigentlich [___] gelegen okay die… |
| 26 | Löschung | `gelegen` | `*(nicht da)*` | …ja eigentlich nur [___] okay die erste… |
| 27 | Löschung | `okay` | `*(nicht da)*` | …eigentlich nur gelegen [___] die erste woche… |
| 28 | Löschung | `die` | `*(nicht da)*` | …nur gelegen okay [___] erste woche deine… |
| 29 | Löschung | `erste` | `*(nicht da)*` | …gelegen okay die [___] woche deine letzte… |
| 30 | Löschung | `woche` | `*(nicht da)*` | …okay die erste [___] deine letzte zeit… |
| 31 | Löschung | `deine` | `*(nicht da)*` | …die erste woche [___] letzte zeit und… |
| 32 | Löschung | `letzte` | `*(nicht da)*` | …erste woche deine [___] zeit und dann… |
| 33 | Löschung | `zeit` | `*(nicht da)*` | …woche deine letzte [___] und dann ja… |
| 34 | Löschung | `und` | `*(nicht da)*` | …deine letzte zeit [___] dann ja mit… |
| 35 | Löschung | `dann` | `*(nicht da)*` | …letzte zeit und [___] ja mit den… |
| 36 | Löschung | `ja` | `*(nicht da)*` | …zeit und dann [___] mit den grücken… |
| 37 | Löschung | `mit` | `*(nicht da)*` | …und dann ja [___] den grücken halt… |
| 38 | Löschung | `den` | `*(nicht da)*` | …dann ja mit [___] grücken halt herumgehen… |
| 39 | Löschung | `grücken` | `*(nicht da)*` | …ja mit den [___] halt herumgehen ein… |
| 40 | Löschung | `halt` | `*(nicht da)*` | …mit den grücken [___] herumgehen ein bisschen… |
| 41 | Löschung | `herumgehen` | `*(nicht da)*` | …den grücken halt [___] ein bisschen aber… |
| 42 | Löschung | `ein` | `*(nicht da)*` | …grücken halt herumgehen [___] bisschen aber halt… |
| 43 | Löschung | `bisschen` | `*(nicht da)*` | …halt herumgehen ein [___] aber halt auch… |
| 44 | Löschung | `aber` | `*(nicht da)*` | …herumgehen ein bisschen [___] halt auch minimal… |
| 45 | Löschung | `halt` | `*(nicht da)*` | …ein bisschen aber [___] auch minimal okay… |
| 46 | Löschung | `auch` | `*(nicht da)*` | …bisschen aber halt [___] minimal okay dann… |
| 47 | Löschung | `minimal` | `*(nicht da)*` | …aber halt auch [___] okay dann sind… |
| 48 | Löschung | `okay` | `*(nicht da)*` | …halt auch minimal [___] dann sind wir… |
| 49 | Löschung | `dann` | `*(nicht da)*` | …auch minimal okay [___] sind wir jetzt… |
| 50 | Löschung | `sind` | `*(nicht da)*` | …minimal okay dann [___] wir jetzt schon… |
| 51 | Löschung | `wir` | `*(nicht da)*` | …okay dann sind [___] jetzt schon so… |
| 52 | Löschung | `jetzt` | `*(nicht da)*` | …dann sind wir [___] schon so weit… |
| 53 | Löschung | `schon` | `*(nicht da)*` | …sind wir jetzt [___] so weit dass… |
| 54 | Löschung | `so` | `*(nicht da)*` | …wir jetzt schon [___] weit dass wir… |
| 55 | Löschung | `weit` | `*(nicht da)*` | …jetzt schon so [___] dass wir darüber… |
| 56 | Löschung | `dass` | `*(nicht da)*` | …schon so weit [___] wir darüber reden… |
| 57 | Löschung | `wir` | `*(nicht da)*` | …so weit dass [___] darüber reden wie… |
| 58 | Löschung | `darüber` | `*(nicht da)*` | …weit dass wir [___] reden wie es… |
| 59 | Löschung | `reden` | `*(nicht da)*` | …dass wir darüber [___] wie es ihnen… |
| 60 | Löschung | `wie` | `*(nicht da)*` | …wir darüber reden [___] es ihnen jetzt… |
| 61 | Löschung | `es` | `*(nicht da)*` | …darüber reden wie [___] ihnen jetzt geht… |
| 62 | Löschung | `ihnen` | `*(nicht da)*` | …reden wie es [___] jetzt geht wie… |
| 63 | Löschung | `jetzt` | `*(nicht da)*` | …wie es ihnen [___] geht wie geht… |
| 64 | Löschung | `geht` | `*(nicht da)*` | …es ihnen jetzt [___] wie geht es… |
| 65 | Löschung | `wie` | `*(nicht da)*` | …ihnen jetzt geht [___] geht es ihnen… |
| 66 | Löschung | `geht` | `*(nicht da)*` | …jetzt geht wie [___] es ihnen wenn… |
| 67 | Löschung | `es` | `*(nicht da)*` | …geht wie geht [___] ihnen wenn sie… |
| 68 | Löschung | `ihnen` | `*(nicht da)*` | …wie geht es [___] wenn sie an… |
| 69 | Löschung | `wenn` | `*(nicht da)*` | …geht es ihnen [___] sie an die… |
| 70 | Löschung | `sie` | `*(nicht da)*` | …es ihnen wenn [___] an die schmerzen… |
| 71 | Löschung | `an` | `*(nicht da)*` | …ihnen wenn sie [___] die schmerzen denken… |
| 72 | Löschung | `die` | `*(nicht da)*` | …wenn sie an [___] schmerzen denken auf… |
| 73 | Löschung | `schmerzen` | `*(nicht da)*` | …sie an die [___] denken auf einer… |
| 74 | Löschung | `denken` | `*(nicht da)*` | …an die schmerzen [___] auf einer skala… |
| 75 | Löschung | `auf` | `*(nicht da)*` | …die schmerzen denken [___] einer skala von… |
| 76 | Löschung | `einer` | `*(nicht da)*` | …schmerzen denken auf [___] skala von 0… |
| 77 | Löschung | `skala` | `*(nicht da)*` | …denken auf einer [___] von 0 bis… |
| 78 | Löschung | `von` | `*(nicht da)*` | …auf einer skala [___] 0 bis 10… |
| 79 | Löschung | `0` | `*(nicht da)*` | …einer skala von [___] bis 10 und… |
| 80 | Löschung | `bis` | `*(nicht da)*` | …skala von 0 [___] 10 und 10… |
| 81 | Löschung | `10` | `*(nicht da)*` | …von 0 bis [___] und 10 sind… |
| 82 | Löschung | `und` | `*(nicht da)*` | …0 bis 10 [___] 10 sind die… |
| 83 | Löschung | `10` | `*(nicht da)*` | …bis 10 und [___] sind die schlimmsten… |
| 84 | Löschung | `sind` | `*(nicht da)*` | …10 und 10 [___] die schlimmsten schmerzen… |
| 85 | Löschung | `die` | `*(nicht da)*` | …und 10 sind [___] schlimmsten schmerzen die… |
| 86 | Löschung | `schlimmsten` | `*(nicht da)*` | …10 sind die [___] schmerzen die ich… |
| 87 | Löschung | `schmerzen` | `*(nicht da)*` | …sind die schlimmsten [___] die ich sich… |
| 88 | Löschung | `die` | `*(nicht da)*` | …die schlimmsten schmerzen [___] ich sich vorstellen… |
| 89 | Löschung | `ich` | `*(nicht da)*` | …schlimmsten schmerzen die [___] sich vorstellen könnte… |
| 90 | Löschung | `sich` | `*(nicht da)*` | …schmerzen die ich [___] vorstellen könnte und… |
| 91 | Löschung | `vorstellen` | `*(nicht da)*` | …die ich sich [___] könnte und 0… |
| 92 | Löschung | `könnte` | `*(nicht da)*` | …ich sich vorstellen [___] und 0 ist… |
| 93 | Löschung | `und` | `*(nicht da)*` | …sich vorstellen könnte [___] 0 ist schmerzfrei… |
| 94 | Löschung | `0` | `*(nicht da)*` | …vorstellen könnte und [___] ist schmerzfrei wo… |
| 95 | Löschung | `ist` | `*(nicht da)*` | …könnte und 0 [___] schmerzfrei wo würden… |
| 96 | Löschung | `schmerzfrei` | `*(nicht da)*` | …und 0 ist [___] wo würden sie… |
| 97 | Löschung | `wo` | `*(nicht da)*` | …0 ist schmerzfrei [___] würden sie sich… |
| 98 | Löschung | `würden` | `*(nicht da)*` | …ist schmerzfrei wo [___] sie sich da… |
| 99 | Löschung | `sie` | `*(nicht da)*` | …schmerzfrei wo würden [___] sich da eingliedern… |
| 100 | Löschung | `sich` | `*(nicht da)*` | …wo würden sie [___] da eingliedern ja… |
| 101 | Löschung | `da` | `*(nicht da)*` | …würden sie sich [___] eingliedern ja wie… |
| 102 | Löschung | `eingliedern` | `*(nicht da)*` | …sie sich da [___] ja wie gesagt… |
| 103 | Löschung | `ja` | `*(nicht da)*` | …sich da eingliedern [___] wie gesagt es… |
| 104 | Löschung | `wie` | `*(nicht da)*` | …da eingliedern ja [___] gesagt es kommt… |
| 105 | Löschung | `gesagt` | `*(nicht da)*` | …eingliedern ja wie [___] es kommt eigentlich… |
| 106 | Löschung | `es` | `*(nicht da)*` | …ja wie gesagt [___] kommt eigentlich auf… |
| 107 | Löschung | `kommt` | `*(nicht da)*` | …wie gesagt es [___] eigentlich auf die… |
| 108 | Löschung | `eigentlich` | `*(nicht da)*` | …gesagt es kommt [___] auf die belastung… |
| 109 | Löschung | `auf` | `*(nicht da)*` | …es kommt eigentlich [___] die belastung darauf… |
| 110 | Löschung | `die` | `*(nicht da)*` | …kommt eigentlich auf [___] belastung darauf an… |
| 111 | Löschung | `belastung` | `*(nicht da)*` | …eigentlich auf die [___] darauf an wenn… |
| 112 | Löschung | `darauf` | `*(nicht da)*` | …auf die belastung [___] an wenn ich… |
| 113 | Löschung | `an` | `*(nicht da)*` | …die belastung darauf [___] wenn ich jetzt… |
| 114 | Löschung | `wenn` | `*(nicht da)*` | …belastung darauf an [___] ich jetzt im… |
| 115 | Löschung | `ich` | `*(nicht da)*` | …darauf an wenn [___] jetzt im ruhezustand… |
| 116 | Löschung | `jetzt` | `*(nicht da)*` | …an wenn ich [___] im ruhezustand bin… |
| 117 | Löschung | `im` | `*(nicht da)*` | …wenn ich jetzt [___] ruhezustand bin und… |
| 118 | Löschung | `ruhezustand` | `*(nicht da)*` | …ich jetzt im [___] bin und mich… |
| 119 | Löschung | `bin` | `*(nicht da)*` | …jetzt im ruhezustand [___] und mich nicht… |
| 120 | Löschung | `und` | `*(nicht da)*` | …im ruhezustand bin [___] mich nicht bewege… |
| 121 | Löschung | `mich` | `*(nicht da)*` | …ruhezustand bin und [___] nicht bewege dann… |
| 122 | Löschung | `nicht` | `*(nicht da)*` | …bin und mich [___] bewege dann sage… |
| 123 | Löschung | `bewege` | `*(nicht da)*` | …und mich nicht [___] dann sage ich… |
| 124 | Löschung | `dann` | `*(nicht da)*` | …mich nicht bewege [___] sage ich vielleicht… |
| 125 | Löschung | `sage` | `*(nicht da)*` | …nicht bewege dann [___] ich vielleicht 1… |
| 126 | Löschung | `ich` | `*(nicht da)*` | …bewege dann sage [___] vielleicht 1 aber… |
| 127 | Löschung | `vielleicht` | `*(nicht da)*` | …dann sage ich [___] 1 aber wenn… |
| 128 | Löschung | `1` | `*(nicht da)*` | …sage ich vielleicht [___] aber wenn ich… |
| 129 | Löschung | `aber` | `*(nicht da)*` | …ich vielleicht 1 [___] wenn ich jetzt… |
| 130 | Löschung | `wenn` | `*(nicht da)*` | …vielleicht 1 aber [___] ich jetzt mit… |
| 131 | Löschung | `ich` | `*(nicht da)*` | …1 aber wenn [___] jetzt mit den… |
| 132 | Löschung | `jetzt` | `*(nicht da)*` | …aber wenn ich [___] mit den grücken… |
| 133 | Löschung | `mit` | `*(nicht da)*` | …wenn ich jetzt [___] den grücken gehe… |
| 134 | Löschung | `den` | `*(nicht da)*` | …ich jetzt mit [___] grücken gehe dann… |
| 135 | Löschung | `grücken` | `*(nicht da)*` | …jetzt mit den [___] gehe dann keine… |
| 136 | Löschung | `gehe` | `*(nicht da)*` | …mit den grücken [___] dann keine ahnung… |
| 137 | Löschung | `dann` | `*(nicht da)*` | …den grücken gehe [___] keine ahnung 3… |
| 138 | Löschung | `keine` | `*(nicht da)*` | …grücken gehe dann [___] ahnung 3 und… |
| 139 | Löschung | `ahnung` | `*(nicht da)*` | …gehe dann keine [___] 3 und wenn… |
| 140 | Löschung | `3` | `*(nicht da)*` | …dann keine ahnung [___] und wenn ich… |
| 141 | Löschung | `und` | `*(nicht da)*` | …keine ahnung 3 [___] wenn ich wirklich… |
| 142 | Löschung | `wenn` | `*(nicht da)*` | …ahnung 3 und [___] ich wirklich ohne… |
| 143 | Löschung | `ich` | `*(nicht da)*` | …3 und wenn [___] wirklich ohne stützen… |
| 144 | Löschung | `wirklich` | `*(nicht da)*` | …und wenn ich [___] ohne stützen probiere… |
| 145 | Löschung | `ohne` | `*(nicht da)*` | …wenn ich wirklich [___] stützen probiere dann… |
| 146 | Löschung | `stützen` | `*(nicht da)*` | …ich wirklich ohne [___] probiere dann bin… |
| 147 | Löschung | `probiere` | `*(nicht da)*` | …wirklich ohne stützen [___] dann bin ich… |
| 148 | Löschung | `dann` | `*(nicht da)*` | …ohne stützen probiere [___] bin ich sicher… |
| 149 | Löschung | `bin` | `*(nicht da)*` | …stützen probiere dann [___] ich sicher bei… |
| 150 | Löschung | `ich` | `*(nicht da)*` | …probiere dann bin [___] sicher bei 6… |
| 151 | Löschung | `sicher` | `*(nicht da)*` | …dann bin ich [___] bei 6 oder… |
| 152 | Löschung | `bei` | `*(nicht da)*` | …bin ich sicher [___] 6 oder 7… |
| 153 | Löschung | `6` | `*(nicht da)*` | …ich sicher bei [___] oder 7 bei… |
| 154 | Löschung | `oder` | `*(nicht da)*` | …sicher bei 6 [___] 7 bei 6… |
| 155 | Löschung | `7` | `*(nicht da)*` | …bei 6 oder [___] bei 6 oder… |
| 156 | Löschung | `bei` | `*(nicht da)*` | …6 oder 7 [___] 6 oder 7… |
| 157 | Löschung | `6` | `*(nicht da)*` | …oder 7 bei [___] oder 7 aber… |
| 158 | Löschung | `oder` | `*(nicht da)*` | …7 bei 6 [___] 7 aber es… |
| 159 | Löschung | `7` | `*(nicht da)*` | …bei 6 oder [___] aber es ist… |
| 160 | Löschung | `aber` | `*(nicht da)*` | …6 oder 7 [___] es ist je… |
| 161 | Löschung | `es` | `*(nicht da)*` | …oder 7 aber [___] ist je nach… |
| 162 | Löschung | `ist` | `*(nicht da)*` | …7 aber es [___] je nach belastung… |
| 163 | Löschung | `je` | `*(nicht da)*` | …aber es ist [___] nach belastung halt… |
| 164 | Löschung | `nach` | `*(nicht da)*` | …es ist je [___] belastung halt und… |
| 165 | Löschung | `belastung` | `*(nicht da)*` | …ist je nach [___] halt und das… |
| 166 | Löschung | `halt` | `*(nicht da)*` | …je nach belastung [___] und das ist… |
| 167 | Löschung | `und` | `*(nicht da)*` | …nach belastung halt [___] das ist ja… |
| 168 | Löschung | `das` | `*(nicht da)*` | …belastung halt und [___] ist ja der… |
| 169 | Löschung | `ist` | `*(nicht da)*` | …halt und das [___] ja der einzige… |
| 170 | Löschung | `ja` | `*(nicht da)*` | …und das ist [___] der einzige faktor… |
| 171 | Löschung | `der` | `*(nicht da)*` | …das ist ja [___] einzige faktor die… |
| 172 | Löschung | `einzige` | `*(nicht da)*` | …ist ja der [___] faktor die belastung… |
| 173 | Löschung | `faktor` | `*(nicht da)*` | …ja der einzige [___] die belastung der… |
| 174 | Löschung | `die` | `*(nicht da)*` | …der einzige faktor [___] belastung der einem… |
| 175 | Löschung | `belastung` | `*(nicht da)*` | …einzige faktor die [___] der einem da… |
| 176 | Löschung | `der` | `*(nicht da)*` | …faktor die belastung [___] einem da einfällt… |
| 177 | Löschung | `einem` | `*(nicht da)*` | …die belastung der [___] da einfällt wenn… |
| 178 | Löschung | `da` | `*(nicht da)*` | …belastung der einem [___] einfällt wenn sie… |
| 179 | Löschung | `einfällt` | `*(nicht da)*` | …der einem da [___] wenn sie an… |
| 180 | Löschung | `wenn` | `*(nicht da)*` | …einem da einfällt [___] sie an den… |
| 181 | Löschung | `sie` | `*(nicht da)*` | …da einfällt wenn [___] an den schmerz… |
| 182 | Löschung | `an` | `*(nicht da)*` | …einfällt wenn sie [___] den schmerz denken… |
| 183 | Löschung | `den` | `*(nicht da)*` | …wenn sie an [___] schmerz denken dass… |
| 184 | Löschung | `schmerz` | `*(nicht da)*` | …sie an den [___] denken dass sich… |
| 185 | Löschung | `denken` | `*(nicht da)*` | …an den schmerz [___] dass sich der… |
| 186 | Löschung | `dass` | `*(nicht da)*` | …den schmerz denken [___] sich der da… |
| 187 | Löschung | `sich` | `*(nicht da)*` | …schmerz denken dass [___] der da verändert… |
| 188 | Löschung | `der` | `*(nicht da)*` | …denken dass sich [___] da verändert ja… |
| 189 | Löschung | `da` | `*(nicht da)*` | …dass sich der [___] verändert ja eigentlich… |
| 190 | Löschung | `verändert` | `*(nicht da)*` | …sich der da [___] ja eigentlich ja… |
| 191 | Löschung | `ja` | `*(nicht da)*` | …der da verändert [___] eigentlich ja also… |
| 192 | Löschung | `eigentlich` | `*(nicht da)*` | …da verändert ja [___] ja also ich… |
| 193 | Löschung | `ja` | `*(nicht da)*` | …verändert ja eigentlich [___] also ich weiß… |
| 194 | Löschung | `also` | `*(nicht da)*` | …ja eigentlich ja [___] ich weiß ja… |
| 195 | Löschung | `ich` | `*(nicht da)*` | …eigentlich ja also [___] weiß ja das… |
| 196 | Löschung | `weiß` | `*(nicht da)*` | …ja also ich [___] ja das nicht… |
| 197 | Löschung | `ja` | `*(nicht da)*` | …also ich weiß [___] das nicht an… |
| 198 | Löschung | `das` | `*(nicht da)*` | …ich weiß ja [___] nicht an was… |
| 199 | Löschung | `nicht` | `*(nicht da)*` | …weiß ja das [___] an was sonst… |
| 200 | Löschung | `an` | `*(nicht da)*` | …ja das nicht [___] was sonst noch… |
| 201 | Löschung | `was` | `*(nicht da)*` | …das nicht an [___] sonst noch okay… |
| 202 | Löschung | `sonst` | `*(nicht da)*` | …nicht an was [___] noch okay und… |
| 203 | Löschung | `noch` | `*(nicht da)*` | …an was sonst [___] okay und sie… |
| 204 | Löschung | `okay` | `*(nicht da)*` | …was sonst noch [___] und sie haben… |
| 205 | Löschung | `und` | `*(nicht da)*` | …sonst noch okay [___] sie haben gesagt… |
| 206 | Löschung | `sie` | `*(nicht da)*` | …noch okay und [___] haben gesagt sie… |
| 207 | Löschung | `haben` | `*(nicht da)*` | …okay und sie [___] gesagt sie haben… |
| 208 | Löschung | `gesagt` | `*(nicht da)*` | …und sie haben [___] sie haben eben… |
| 209 | Löschung | `sie` | `*(nicht da)*` | …sie haben gesagt [___] haben eben mit… |
| 210 | Löschung | `haben` | `*(nicht da)*` | …haben gesagt sie [___] eben mit dem… |
| 211 | Löschung | `eben` | `*(nicht da)*` | …gesagt sie haben [___] mit dem gehen… |
| 212 | Löschung | `mit` | `*(nicht da)*` | …sie haben eben [___] dem gehen mit… |
| 213 | Löschung | `dem` | `*(nicht da)*` | …haben eben mit [___] gehen mit den… |
| 214 | Löschung | `gehen` | `*(nicht da)*` | …eben mit dem [___] mit den stützen… |
| 215 | Löschung | `mit` | `*(nicht da)*` | …mit dem gehen [___] den stützen das… |
| 216 | Löschung | `den` | `*(nicht da)*` | …dem gehen mit [___] stützen das funktioniert… |
| 217 | Löschung | `stützen` | `*(nicht da)*` | …gehen mit den [___] das funktioniert nur… |
| 218 | Löschung | `das` | `*(nicht da)*` | …mit den stützen [___] funktioniert nur kurz… |
| 219 | Löschung | `funktioniert` | `*(nicht da)*` | …den stützen das [___] nur kurz was… |
| 220 | Löschung | `nur` | `*(nicht da)*` | …stützen das funktioniert [___] kurz was können… |
| 221 | Löschung | `kurz` | `*(nicht da)*` | …das funktioniert nur [___] was können wir… |
| 222 | Löschung | `was` | `*(nicht da)*` | …funktioniert nur kurz [___] können wir da… |
| 223 | Löschung | `können` | `*(nicht da)*` | …nur kurz was [___] wir da forschen… |
| 224 | Löschung | `wir` | `*(nicht da)*` | …kurz was können [___] da forschen also… |
| 225 | Löschung | `da` | `*(nicht da)*` | …was können wir [___] forschen also sind… |
| 226 | Löschung | `forschen` | `*(nicht da)*` | …können wir da [___] also sind sie… |
| 227 | Löschung | `also` | `*(nicht da)*` | …wir da forschen [___] sind sie auf… |
| 228 | Löschung | `sind` | `*(nicht da)*` | …da forschen also [___] sie auf und… |
| 229 | Löschung | `sie` | `*(nicht da)*` | …forschen also sind [___] auf und zu… |
| 230 | Löschung | `auf` | `*(nicht da)*` | …also sind sie [___] und zu rausgegangen… |
| 231 | Löschung | `und` | `*(nicht da)*` | …sind sie auf [___] zu rausgegangen nein… |
| 232 | Löschung | `zu` | `*(nicht da)*` | …sie auf und [___] rausgegangen nein jetzt… |
| 233 | Löschung | `rausgegangen` | `*(nicht da)*` | …auf und zu [___] nein jetzt nicht… |
| 234 | Löschung | `nein` | `*(nicht da)*` | …und zu rausgegangen [___] jetzt nicht also… |
| 235 | Löschung | `jetzt` | `*(nicht da)*` | …zu rausgegangen nein [___] nicht also das… |
| 236 | Löschung | `nicht` | `*(nicht da)*` | …rausgegangen nein jetzt [___] also das ist… |
| 237 | Löschung | `also` | `*(nicht da)*` | …nein jetzt nicht [___] das ist jetzt… |
| 238 | Löschung | `das` | `*(nicht da)*` | …jetzt nicht also [___] ist jetzt auch… |
| 239 | Löschung | `ist` | `*(nicht da)*` | …nicht also das [___] jetzt auch ein… |
| 240 | Löschung | `jetzt` | `*(nicht da)*` | …also das ist [___] auch ein monat… |
| 241 | Löschung | `auch` | `*(nicht da)*` | …das ist jetzt [___] ein monat her… |
| 242 | Löschung | `ein` | `*(nicht da)*` | …ist jetzt auch [___] monat her also… |
| 243 | Löschung | `monat` | `*(nicht da)*` | …jetzt auch ein [___] her also nicht… |
| 244 | Löschung | `her` | `*(nicht da)*` | …auch ein monat [___] also nicht wirklich… |
| 245 | Löschung | `also` | `*(nicht da)*` | …ein monat her [___] nicht wirklich ich… |
| 246 | Löschung | `nicht` | `*(nicht da)*` | …monat her also [___] wirklich ich meine… |
| 247 | Löschung | `wirklich` | `*(nicht da)*` | …her also nicht [___] ich meine minimal… |
| 248 | Löschung | `ich` | `*(nicht da)*` | …also nicht wirklich [___] meine minimal einfach… |
| 249 | Löschung | `meine` | `*(nicht da)*` | …nicht wirklich ich [___] minimal einfach aber… |
| 250 | Löschung | `minimal` | `*(nicht da)*` | …wirklich ich meine [___] einfach aber ich… |
| 251 | Löschung | `einfach` | `*(nicht da)*` | …ich meine minimal [___] aber ich kann… |
| 252 | Löschung | `aber` | `*(nicht da)*` | …meine minimal einfach [___] ich kann nicht… |
| 253 | Löschung | `ich` | `*(nicht da)*` | …minimal einfach aber [___] kann nicht wirklich… |
| 254 | Löschung | `kann` | `*(nicht da)*` | …einfach aber ich [___] nicht wirklich zusammenkriegen… |
| 255 | Löschung | `nicht` | `*(nicht da)*` | …aber ich kann [___] wirklich zusammenkriegen jetzt… |
| 256 | Löschung | `wirklich` | `*(nicht da)*` | …ich kann nicht [___] zusammenkriegen jetzt spazieren… |
| 257 | Löschung | `zusammenkriegen` | `*(nicht da)*` | …kann nicht wirklich [___] jetzt spazieren oder… |
| 258 | Löschung | `jetzt` | `*(nicht da)*` | …nicht wirklich zusammenkriegen [___] spazieren oder so… |
| 259 | Löschung | `spazieren` | `*(nicht da)*` | …wirklich zusammenkriegen jetzt [___] oder so also… |
| 260 | Löschung | `oder` | `*(nicht da)*` | …zusammenkriegen jetzt spazieren [___] so also ich… |
| 261 | Löschung | `so` | `*(nicht da)*` | …jetzt spazieren oder [___] also ich bewege… |
| 262 | Löschung | `also` | `*(nicht da)*` | …spazieren oder so [___] ich bewege mich… |
| 263 | Löschung | `ich` | `*(nicht da)*` | …oder so also [___] bewege mich halt… |
| 264 | Löschung | `bewege` | `*(nicht da)*` | …so also ich [___] mich halt in… |
| 265 | Löschung | `mich` | `*(nicht da)*` | …also ich bewege [___] halt in der… |
| 266 | Löschung | `halt` | `*(nicht da)*` | …ich bewege mich [___] in der wohnung… |
| 267 | Löschung | `in` | `*(nicht da)*` | …bewege mich halt [___] der wohnung was… |
| 268 | Löschung | `der` | `*(nicht da)*` | …mich halt in [___] wohnung was das… |
| 269 | Löschung | `wohnung` | `*(nicht da)*` | …halt in der [___] was das nötigste… |
| 270 | Löschung | `was` | `*(nicht da)*` | …in der wohnung [___] das nötigste und… |
| 271 | Löschung | `das` | `*(nicht da)*` | …der wohnung was [___] nötigste und ja… |
| 272 | Löschung | `nötigste` | `*(nicht da)*` | …wohnung was das [___] und ja versuche… |
| 273 | Löschung | `und` | `*(nicht da)*` | …was das nötigste [___] ja versuche halt… |
| 274 | Löschung | `ja` | `*(nicht da)*` | …das nötigste und [___] versuche halt am… |
| 275 | Löschung | `versuche` | `*(nicht da)*` | …nötigste und ja [___] halt am heimtrainer… |
| 276 | Löschung | `halt` | `*(nicht da)*` | …und ja versuche [___] am heimtrainer ab… |
| 277 | Löschung | `am` | `*(nicht da)*` | …ja versuche halt [___] heimtrainer ab und… |
| 278 | Löschung | `heimtrainer` | `*(nicht da)*` | …versuche halt am [___] ab und zu… |
| 279 | Löschung | `ab` | `*(nicht da)*` | …halt am heimtrainer [___] und zu so… |
| 280 | Löschung | `und` | `*(nicht da)*` | …am heimtrainer ab [___] zu so weit… |
| 281 | Löschung | `zu` | `*(nicht da)*` | …heimtrainer ab und [___] so weit wie… |
| 282 | Löschung | `so` | `*(nicht da)*` | …ab und zu [___] weit wie möglich… |
| 283 | Löschung | `weit` | `*(nicht da)*` | …und zu so [___] wie möglich zu… |
| 284 | Löschung | `wie` | `*(nicht da)*` | …zu so weit [___] möglich zu beugen… |
| 285 | Löschung | `möglich` | `*(nicht da)*` | …so weit wie [___] zu beugen und… |
| 286 | Löschung | `zu` | `*(nicht da)*` | …weit wie möglich [___] beugen und das… |
| 287 | Löschung | `beugen` | `*(nicht da)*` | …wie möglich zu [___] und das eigentlich… |
| 288 | Löschung | `und` | `*(nicht da)*` | …möglich zu beugen [___] das eigentlich immer… |
| 289 | Löschung | `das` | `*(nicht da)*` | …zu beugen und [___] eigentlich immer unter… |
| 290 | Löschung | `eigentlich` | `*(nicht da)*` | …beugen und das [___] immer unter schmerzen… |
| 291 | Löschung | `immer` | `*(nicht da)*` | …und das eigentlich [___] unter schmerzen dann… |
| 292 | Löschung | `unter` | `*(nicht da)*` | …das eigentlich immer [___] schmerzen dann wenn… |
| 293 | Löschung | `schmerzen` | `*(nicht da)*` | …eigentlich immer unter [___] dann wenn man… |
| 294 | Löschung | `dann` | `*(nicht da)*` | …immer unter schmerzen [___] wenn man sagt… |
| 295 | Löschung | `wenn` | `*(nicht da)*` | …unter schmerzen dann [___] man sagt mit… |
| 296 | Löschung | `man` | `*(nicht da)*` | …schmerzen dann wenn [___] sagt mit der… |
| 297 | Löschung | `sagt` | `*(nicht da)*` | …dann wenn man [___] mit der belastung… |
| 298 | Löschung | `mit` | `*(nicht da)*` | …wenn man sagt [___] der belastung variiert… |
| 299 | Löschung | `der` | `*(nicht da)*` | …man sagt mit [___] belastung variiert aber… |
| 300 | Löschung | `belastung` | `*(nicht da)*` | …sagt mit der [___] variiert aber ist… |
| 301 | Löschung | `variiert` | `*(nicht da)*` | …mit der belastung [___] aber ist noch… |
| 302 | Löschung | `aber` | `*(nicht da)*` | …der belastung variiert [___] ist noch nicht… |
| 303 | Löschung | `ist` | `*(nicht da)*` | …belastung variiert aber [___] noch nicht richtig… |
| 304 | Löschung | `noch` | `*(nicht da)*` | …variiert aber ist [___] nicht richtig schmerzfrei… |
| 305 | Löschung | `nicht` | `*(nicht da)*` | …aber ist noch [___] richtig schmerzfrei möglich… |
| 306 | Löschung | `richtig` | `*(nicht da)*` | …ist noch nicht [___] schmerzfrei möglich nein… |
| 307 | Löschung | `schmerzfrei` | `*(nicht da)*` | …noch nicht richtig [___] möglich nein nehmen… |
| 308 | Löschung | `möglich` | `*(nicht da)*` | …nicht richtig schmerzfrei [___] nein nehmen sie… |
| 309 | Löschung | `nein` | `*(nicht da)*` | …richtig schmerzfrei möglich [___] nehmen sie irgendwelche… |
| 310 | Löschung | `nehmen` | `*(nicht da)*` | …irgendwelche medikamente nein [___] sie nichts nehmen… |
| 311 | Löschung | `sie` | `*(nicht da)*` | …medikamente nein nehmen [___] nichts nehmen sie… |
| 312 | Löschung | `nichts` | `*(nicht da)*` | …nein nehmen sie [___] nehmen sie nichts… |
| 313 | Substitution | `medikamente` | `medikamen` | …sonst nicht unter [___] also abgesehen von… |
| 314 | Substitution | `krebspartner` | `grasbeutner` | …vielen dank frau [___] und wir treffen… |
