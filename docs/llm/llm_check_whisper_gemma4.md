# LLM-Fehleranalyse: Whisper large-v3-turbo + gemma4

> RAW STT → Formatted — Satzzeichen und Groß-/Kleinschreibung ignoriert.
> Speaker-Label-Änderungen sind bereits aus der JSON entfernt.
> **S** = Substitution | **D** = Löschung (im RAW, fehlt im FMT) | **I** = Einfügung (im FMT, nicht im RAW)

---

## Übersicht

| Szenario | RAW-Wörter | FMT-Wörter | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 235 | 234 | 0 | 1 | 0 | 1 | 0.4% |
| OriginalDC+Noise | 210 | 210 | 0 | 0 | 0 | 0 | 0.0% |
| LapInMitte | 226 | 226 | 0 | 0 | 0 | 0 | 0.0% |
| LapBeiArzt | 229 | 229 | 0 | 0 | 0 | 0 | 0.0% |
| Selbstkorrekturen | 200 | 200 | 0 | 0 | 0 | 0 | 0.0% |
| Unterbrechungen | 143 | 143 | 0 | 0 | 0 | 0 | 0.0% |
| Gedankensprünge | 190 | 190 | 0 | 0 | 0 | 0 | 0.0% |
| Meinungswechsel | 183 | 183 | 0 | 0 | 0 | 0 | 0.0% |
| Chaos | 252 | 252 | 0 | 0 | 0 | 0 | 0.0% |
| Anamnesegespräch | 2269 | 478 | 338 | 1791 | 0 | 2129 | 93.8% |
| PWC | 1512 | 944 | 228 | 568 | 0 | 796 | 52.6% |

---

## OriginalDC

**Fehlerrate: 0.4%** — RAW: 235 Wörter | FMT: 234 Wörter | S=0 D=1 I=0 | Fehler=1

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Löschung | `years` | `*(nicht da)*` | …kommt jetzt cosechs [___]… |

---

## OriginalDC+Noise

**Fehlerrate: 0.0%** — RAW: 210 Wörter | FMT: 210 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## LapInMitte

**Fehlerrate: 0.0%** — RAW: 226 Wörter | FMT: 226 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## LapBeiArzt

**Fehlerrate: 0.0%** — RAW: 229 Wörter | FMT: 229 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Selbstkorrekturen

**Fehlerrate: 0.0%** — RAW: 200 Wörter | FMT: 200 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Unterbrechungen

**Fehlerrate: 0.0%** — RAW: 143 Wörter | FMT: 143 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Gedankensprünge

**Fehlerrate: 0.0%** — RAW: 190 Wörter | FMT: 190 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Meinungswechsel

**Fehlerrate: 0.0%** — RAW: 183 Wörter | FMT: 183 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Chaos

**Fehlerrate: 0.0%** — RAW: 252 Wörter | FMT: 252 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Anamnesegespräch

**Fehlerrate: 93.8%** — RAW: 2269 Wörter | FMT: 478 Wörter | S=338 D=1791 I=0 | Fehler=2129

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
| 31 | Löschung | `ja` | `*(nicht da)*` | …einverstanden guten tag [___] natürlich sehr gerne… |
| 32 | Löschung | `natürlich` | `*(nicht da)*` | …guten tag ja [___] sehr gerne wunderbar… |
| 33 | Löschung | `sehr` | `*(nicht da)*` | …tag ja natürlich [___] gerne wunderbar wie… |
| 34 | Löschung | `gerne` | `*(nicht da)*` | …ja natürlich sehr [___] wunderbar wie heißen… |
| 35 | Löschung | `wunderbar` | `*(nicht da)*` | …natürlich sehr gerne [___] wie heißen sie… |
| 36 | Löschung | `heißen` | `*(nicht da)*` | …gerne wunderbar wie [___] sie denn ich… |
| 37 | Löschung | `sie` | `*(nicht da)*` | …wunderbar wie heißen [___] denn ich heiße… |
| 38 | Löschung | `denn` | `*(nicht da)*` | …wie heißen sie [___] ich heiße julia… |
| 39 | Löschung | `ich` | `*(nicht da)*` | …heißen sie denn [___] heiße julia becken… |
| 40 | Löschung | `heiße` | `*(nicht da)*` | …sie denn ich [___] julia becken westfalen… |
| 41 | Löschung | `julia` | `*(nicht da)*` | …denn ich heiße [___] becken westfalen julia… |
| 42 | Löschung | `becken` | `*(nicht da)*` | …ich heiße julia [___] westfalen julia becken… |
| 43 | Löschung | `westfalen` | `*(nicht da)*` | …heiße julia becken [___] julia becken westfalen… |
| 44 | Löschung | `julia` | `*(nicht da)*` | …julia becken westfalen [___] becken westfalen können… |
| 45 | Löschung | `becken` | `*(nicht da)*` | …becken westfalen julia [___] westfalen können sie… |
| 46 | Substitution | `westfalen` | `geht` | …westfalen julia becken [___] können sie mir… |
| 47 | Substitution | `können` | `es` | …julia becken westfalen [___] sie mir ihren… |
| 48 | Substitution | `sie` | `ihnen` | …becken westfalen können [___] mir ihren nachnamen… |
| 49 | Löschung | `ihren` | `*(nicht da)*` | …können sie mir [___] nachnamen bitte einmal… |
| 50 | Löschung | `nachnamen` | `*(nicht da)*` | …sie mir ihren [___] bitte einmal langsam… |
| 51 | Löschung | `bitte` | `*(nicht da)*` | …mir ihren nachnamen [___] einmal langsam buchstabieren… |
| 52 | Löschung | `einmal` | `*(nicht da)*` | …ihren nachnamen bitte [___] langsam buchstabieren gerne… |
| 53 | Löschung | `langsam` | `*(nicht da)*` | …nachnamen bitte einmal [___] buchstabieren gerne ja… |
| 54 | Löschung | `buchstabieren` | `*(nicht da)*` | …bitte einmal langsam [___] gerne ja becken… |
| 55 | Löschung | `gerne` | `*(nicht da)*` | …einmal langsam buchstabieren [___] ja becken westfalen… |
| 56 | Löschung | `ja` | `*(nicht da)*` | …langsam buchstabieren gerne [___] becken westfalen westfalen… |
| 57 | Löschung | `becken` | `*(nicht da)*` | …buchstabieren gerne ja [___] westfalen westfalen westphalen… |
| 58 | Löschung | `westfalen` | `*(nicht da)*` | …gerne ja becken [___] westfalen westphalen alles… |
| 59 | Löschung | `westfalen` | `*(nicht da)*` | …ja becken westfalen [___] westphalen alles klar… |
| 60 | Löschung | `westphalen` | `*(nicht da)*` | …becken westfalen westfalen [___] alles klar dankeschön… |
| 61 | Löschung | `alles` | `*(nicht da)*` | …westfalen westfalen westphalen [___] klar dankeschön frau… |
| 62 | Löschung | `klar` | `*(nicht da)*` | …westfalen westphalen alles [___] dankeschön frau becken… |
| 63 | Löschung | `dankeschön` | `*(nicht da)*` | …westphalen alles klar [___] frau becken westfalen… |
| 64 | Löschung | `frau` | `*(nicht da)*` | …alles klar dankeschön [___] becken westfalen wie… |
| 65 | Löschung | `becken` | `*(nicht da)*` | …klar dankeschön frau [___] westfalen wie alt… |
| 66 | Löschung | `westfalen` | `*(nicht da)*` | …dankeschön frau becken [___] wie alt sind… |
| 67 | Löschung | `wie` | `*(nicht da)*` | …frau becken westfalen [___] alt sind sie… |
| 68 | Löschung | `alt` | `*(nicht da)*` | …becken westfalen wie [___] sind sie denn… |
| 69 | Löschung | `sind` | `*(nicht da)*` | …westfalen wie alt [___] sie denn 33… |
| 70 | Löschung | `sie` | `*(nicht da)*` | …wie alt sind [___] denn 33 33… |
| 71 | Löschung | `denn` | `*(nicht da)*` | …alt sind sie [___] 33 33 und… |
| 72 | Löschung | `33` | `*(nicht da)*` | …sind sie denn [___] 33 und wann… |
| 73 | Löschung | `33` | `*(nicht da)*` | …sie denn 33 [___] und wann ist… |
| 74 | Löschung | `und` | `*(nicht da)*` | …denn 33 33 [___] wann ist ihr… |
| 75 | Löschung | `wann` | `*(nicht da)*` | …33 33 und [___] ist ihr geburtstag… |
| 76 | Löschung | `ist` | `*(nicht da)*` | …33 und wann [___] ihr geburtstag am… |
| 77 | Löschung | `ihr` | `*(nicht da)*` | …und wann ist [___] geburtstag am 27… |
| 78 | Löschung | `geburtstag` | `*(nicht da)*` | …wann ist ihr [___] am 27 märz… |
| 79 | Löschung | `am` | `*(nicht da)*` | …ist ihr geburtstag [___] 27 märz 1987… |
| 80 | Löschung | `27` | `*(nicht da)*` | …ihr geburtstag am [___] märz 1987 oh… |
| 81 | Löschung | `märz` | `*(nicht da)*` | …geburtstag am 27 [___] 1987 oh schön… |
| 82 | Löschung | `1987` | `*(nicht da)*` | …am 27 märz [___] oh schön herzlichen… |
| 83 | Löschung | `oh` | `*(nicht da)*` | …27 märz 1987 [___] schön herzlichen glückwunsch… |
| 84 | Löschung | `schön` | `*(nicht da)*` | …märz 1987 oh [___] herzlichen glückwunsch nachträglich… |
| 85 | Löschung | `herzlichen` | `*(nicht da)*` | …1987 oh schön [___] glückwunsch nachträglich vielen… |
| 86 | Löschung | `glückwunsch` | `*(nicht da)*` | …oh schön herzlichen [___] nachträglich vielen dank… |
| 87 | Löschung | `nachträglich` | `*(nicht da)*` | …schön herzlichen glückwunsch [___] vielen dank frau… |
| 88 | Löschung | `vielen` | `*(nicht da)*` | …herzlichen glückwunsch nachträglich [___] dank frau becken… |
| 89 | Löschung | `dank` | `*(nicht da)*` | …glückwunsch nachträglich vielen [___] frau becken westfalen… |
| 90 | Löschung | `frau` | `*(nicht da)*` | …nachträglich vielen dank [___] becken westfalen wie… |
| 91 | Löschung | `becken` | `*(nicht da)*` | …vielen dank frau [___] westfalen wie groß… |
| 92 | Löschung | `westfalen` | `*(nicht da)*` | …dank frau becken [___] wie groß sind… |
| 93 | Löschung | `wie` | `*(nicht da)*` | …frau becken westfalen [___] groß sind sie… |
| 94 | Löschung | `groß` | `*(nicht da)*` | …becken westfalen wie [___] sind sie denn… |
| 95 | Löschung | `sind` | `*(nicht da)*` | …westfalen wie groß [___] sie denn 1… |
| 96 | Löschung | `sie` | `*(nicht da)*` | …wie groß sind [___] denn 1 70… |
| 97 | Löschung | `denn` | `*(nicht da)*` | …groß sind sie [___] 1 70 1… |
| 98 | Löschung | `1` | `*(nicht da)*` | …sind sie denn [___] 70 1 70… |
| 99 | Löschung | `70` | `*(nicht da)*` | …sie denn 1 [___] 1 70 alles… |
| 100 | Löschung | `1` | `*(nicht da)*` | …denn 1 70 [___] 70 alles klar… |
| 101 | Löschung | `70` | `*(nicht da)*` | …1 70 1 [___] alles klar und… |
| 102 | Löschung | `alles` | `*(nicht da)*` | …70 1 70 [___] klar und wie… |
| 103 | Löschung | `klar` | `*(nicht da)*` | …1 70 alles [___] und wie viel… |
| 104 | Löschung | `und` | `*(nicht da)*` | …70 alles klar [___] wie viel wiegen… |
| 105 | Löschung | `wie` | `*(nicht da)*` | …alles klar und [___] viel wiegen sie… |
| 106 | Löschung | `viel` | `*(nicht da)*` | …klar und wie [___] wiegen sie zurzeit… |
| 107 | Löschung | `wiegen` | `*(nicht da)*` | …und wie viel [___] sie zurzeit 60… |
| 108 | Löschung | `sie` | `*(nicht da)*` | …wie viel wiegen [___] zurzeit 60 kilo… |
| 109 | Löschung | `zurzeit` | `*(nicht da)*` | …viel wiegen sie [___] 60 kilo glaube… |
| 110 | Löschung | `60` | `*(nicht da)*` | …wiegen sie zurzeit [___] kilo glaube ich… |
| 111 | Löschung | `kilo` | `*(nicht da)*` | …sie zurzeit 60 [___] glaube ich okay… |
| 112 | Löschung | `glaube` | `*(nicht da)*` | …zurzeit 60 kilo [___] ich okay gut… |
| 113 | Substitution | `ich` | `geht` | …60 kilo glaube [___] okay gut können… |
| 114 | Substitution | `okay` | `es` | …kilo glaube ich [___] gut können sie… |
| 115 | Löschung | `können` | `*(nicht da)*` | …ich okay gut [___] sie mir noch… |
| 116 | Löschung | `sie` | `*(nicht da)*` | …okay gut können [___] mir noch den… |
| 117 | Löschung | `mir` | `*(nicht da)*` | …gut können sie [___] noch den namen… |
| 118 | Löschung | `noch` | `*(nicht da)*` | …können sie mir [___] den namen ihres… |
| 119 | Löschung | `den` | `*(nicht da)*` | …sie mir noch [___] namen ihres hausarztes… |
| 120 | Löschung | `namen` | `*(nicht da)*` | …mir noch den [___] ihres hausarztes verraten… |
| 121 | Löschung | `ihres` | `*(nicht da)*` | …noch den namen [___] hausarztes verraten ja… |
| 122 | Löschung | `hausarztes` | `*(nicht da)*` | …den namen ihres [___] verraten ja das… |
| 123 | Löschung | `verraten` | `*(nicht da)*` | …namen ihres hausarztes [___] ja das ist… |
| 124 | Löschung | `ja` | `*(nicht da)*` | …ihres hausarztes verraten [___] das ist der… |
| 125 | Löschung | `das` | `*(nicht da)*` | …hausarztes verraten ja [___] ist der herr… |
| 126 | Löschung | `ist` | `*(nicht da)*` | …verraten ja das [___] der herr dr… |
| 127 | Löschung | `der` | `*(nicht da)*` | …ja das ist [___] herr dr becker… |
| 128 | Löschung | `herr` | `*(nicht da)*` | …das ist der [___] dr becker der… |
| 129 | Löschung | `dr` | `*(nicht da)*` | …ist der herr [___] becker der herr… |
| 130 | Löschung | `becker` | `*(nicht da)*` | …der herr dr [___] der herr dr… |
| 131 | Löschung | `der` | `*(nicht da)*` | …herr dr becker [___] herr dr becker… |
| 132 | Löschung | `herr` | `*(nicht da)*` | …dr becker der [___] dr becker wie… |
| 133 | Löschung | `dr` | `*(nicht da)*` | …becker der herr [___] becker wie der… |
| 134 | Löschung | `becker` | `*(nicht da)*` | …der herr dr [___] wie der beruf… |
| 135 | Löschung | `wie` | `*(nicht da)*` | …herr dr becker [___] der beruf oder… |
| 136 | Löschung | `der` | `*(nicht da)*` | …dr becker wie [___] beruf oder mit… |
| 137 | Löschung | `beruf` | `*(nicht da)*` | …becker wie der [___] oder mit e… |
| 138 | Löschung | `oder` | `*(nicht da)*` | …wie der beruf [___] mit e mit… |
| 139 | Löschung | `mit` | `*(nicht da)*` | …der beruf oder [___] e mit e… |
| 140 | Löschung | `e` | `*(nicht da)*` | …beruf oder mit [___] mit e mit… |
| 141 | Löschung | `mit` | `*(nicht da)*` | …oder mit e [___] e mit e… |
| 142 | Löschung | `e` | `*(nicht da)*` | …mit e mit [___] mit e alles… |
| 143 | Löschung | `mit` | `*(nicht da)*` | …e mit e [___] e alles klar… |
| 144 | Löschung | `e` | `*(nicht da)*` | …mit e mit [___] alles klar gut… |
| 145 | Löschung | `alles` | `*(nicht da)*` | …e mit e [___] klar gut frau… |
| 146 | Substitution | `klar` | `danke` | …mit e alles [___] gut frau becken… |
| 147 | Löschung | `frau` | `*(nicht da)*` | …alles klar gut [___] becken westfalen sie… |
| 148 | Löschung | `becken` | `*(nicht da)*` | …klar gut frau [___] westfalen sie wurden… |
| 149 | Löschung | `westfalen` | `*(nicht da)*` | …gut frau becken [___] sie wurden ja… |
| 150 | Löschung | `sie` | `*(nicht da)*` | …frau becken westfalen [___] wurden ja soeben… |
| 151 | Löschung | `wurden` | `*(nicht da)*` | …becken westfalen sie [___] ja soeben mit… |
| 152 | Löschung | `ja` | `*(nicht da)*` | …westfalen sie wurden [___] soeben mit dem… |
| 153 | Löschung | `soeben` | `*(nicht da)*` | …sie wurden ja [___] mit dem rettungswagen… |
| 154 | Löschung | `mit` | `*(nicht da)*` | …wurden ja soeben [___] dem rettungswagen zu… |
| 155 | Löschung | `dem` | `*(nicht da)*` | …ja soeben mit [___] rettungswagen zu uns… |
| 156 | Löschung | `rettungswagen` | `*(nicht da)*` | …soeben mit dem [___] zu uns gebracht… |
| 157 | Löschung | `zu` | `*(nicht da)*` | …mit dem rettungswagen [___] uns gebracht was… |
| 158 | Löschung | `uns` | `*(nicht da)*` | …dem rettungswagen zu [___] gebracht was ist… |
| 159 | Substitution | `gebracht` | `und` | …rettungswagen zu uns [___] was ist denn… |
| 160 | Löschung | `denn` | `*(nicht da)*` | …gebracht was ist [___] passiert ja ich… |
| 161 | Löschung | `passiert` | `*(nicht da)*` | …was ist denn [___] ja ich bin… |
| 162 | Löschung | `ja` | `*(nicht da)*` | …ist denn passiert [___] ich bin unvorsichtig… |
| 163 | Löschung | `ich` | `*(nicht da)*` | …denn passiert ja [___] bin unvorsichtig mit… |
| 164 | Löschung | `bin` | `*(nicht da)*` | …passiert ja ich [___] unvorsichtig mit meinem… |
| 165 | Löschung | `unvorsichtig` | `*(nicht da)*` | …ja ich bin [___] mit meinem fahrrad… |
| 166 | Löschung | `mit` | `*(nicht da)*` | …ich bin unvorsichtig [___] meinem fahrrad nach… |
| 167 | Löschung | `meinem` | `*(nicht da)*` | …bin unvorsichtig mit [___] fahrrad nach hause… |
| 168 | Löschung | `fahrrad` | `*(nicht da)*` | …unvorsichtig mit meinem [___] nach hause gefahren… |
| 169 | Löschung | `nach` | `*(nicht da)*` | …mit meinem fahrrad [___] hause gefahren von… |
| 170 | Löschung | `hause` | `*(nicht da)*` | …meinem fahrrad nach [___] gefahren von der… |
| 171 | Löschung | `gefahren` | `*(nicht da)*` | …fahrrad nach hause [___] von der arbeit… |
| 172 | Substitution | `von` | `heute` | …nach hause gefahren [___] der arbeit und… |
| 173 | Löschung | `arbeit` | `*(nicht da)*` | …gefahren von der [___] und hatte leider… |
| 174 | Löschung | `und` | `*(nicht da)*` | …von der arbeit [___] hatte leider einen… |
| 175 | Löschung | `hatte` | `*(nicht da)*` | …der arbeit und [___] leider einen unfall… |
| 176 | Löschung | `leider` | `*(nicht da)*` | …arbeit und hatte [___] einen unfall dabei… |
| 177 | Löschung | `einen` | `*(nicht da)*` | …und hatte leider [___] unfall dabei habe… |
| 178 | Löschung | `unfall` | `*(nicht da)*` | …hatte leider einen [___] dabei habe ich… |
| 179 | Löschung | `dabei` | `*(nicht da)*` | …leider einen unfall [___] habe ich mich… |
| 180 | Löschung | `habe` | `*(nicht da)*` | …einen unfall dabei [___] ich mich verletzt… |
| 181 | Löschung | `ich` | `*(nicht da)*` | …unfall dabei habe [___] mich verletzt okay… |
| 182 | Löschung | `mich` | `*(nicht da)*` | …dabei habe ich [___] verletzt okay dabei… |
| 183 | Substitution | `verletzt` | `anlass` | …habe ich mich [___] okay dabei habe… |
| 184 | Substitution | `okay` | `für` | …ich mich verletzt [___] dabei habe ich… |
| 185 | Substitution | `dabei` | `ihren` | …mich verletzt okay [___] habe ich mich… |
| 186 | Substitution | `habe` | `besuch` | …verletzt okay dabei [___] ich mich verletzt… |
| 187 | Löschung | `mich` | `*(nicht da)*` | …dabei habe ich [___] verletzt den krankenwagen… |
| 188 | Substitution | `verletzt` | `bin` | …habe ich mich [___] den krankenwagen gerufen… |
| 189 | Substitution | `den` | `heute` | …ich mich verletzt [___] krankenwagen gerufen und… |
| 190 | Substitution | `krankenwagen` | `wegen` | …mich verletzt den [___] gerufen und da… |
| 191 | Substitution | `gerufen` | `meiner` | …verletzt den krankenwagen [___] und da bin… |
| 192 | Substitution | `und` | `knieprobleme` | …den krankenwagen gerufen [___] da bin ich… |
| 193 | Substitution | `da` | `hier` | …krankenwagen gerufen und [___] bin ich jetzt… |
| 194 | Substitution | `bin` | `ah` | …gerufen und da [___] ich jetzt da… |
| 195 | Löschung | `jetzt` | `*(nicht da)*` | …da bin ich [___] da sind sie… |
| 196 | Löschung | `da` | `*(nicht da)*` | …bin ich jetzt [___] sind sie jetzt… |
| 197 | Löschung | `sind` | `*(nicht da)*` | …ich jetzt da [___] sie jetzt was… |
| 198 | Substitution | `sie` | `verstehe` | …jetzt da sind [___] jetzt was haben… |
| 199 | Substitution | `jetzt` | `seit` | …da sind sie [___] was haben sie… |
| 200 | Substitution | `was` | `wann` | …sind sie jetzt [___] haben sie denn… |
| 201 | Löschung | `jetzt` | `*(nicht da)*` | …haben sie denn [___] für beschwerden entschuldigung… |
| 202 | Substitution | `für` | `diese` | …sie denn jetzt [___] beschwerden entschuldigung haben… |
| 203 | Löschung | `entschuldigung` | `*(nicht da)*` | …jetzt für beschwerden [___] haben sie schmerzen… |
| 204 | Löschung | `haben` | `*(nicht da)*` | …für beschwerden entschuldigung [___] sie schmerzen am… |
| 205 | Löschung | `sie` | `*(nicht da)*` | …beschwerden entschuldigung haben [___] schmerzen am kopf… |
| 206 | Löschung | `schmerzen` | `*(nicht da)*` | …entschuldigung haben sie [___] am kopf im… |
| 207 | Löschung | `am` | `*(nicht da)*` | …haben sie schmerzen [___] kopf im oberkörper… |
| 208 | Löschung | `kopf` | `*(nicht da)*` | …sie schmerzen am [___] im oberkörper in… |
| 209 | Löschung | `im` | `*(nicht da)*` | …schmerzen am kopf [___] oberkörper in den… |
| 210 | Löschung | `oberkörper` | `*(nicht da)*` | …am kopf im [___] in den beinen… |
| 211 | Löschung | `in` | `*(nicht da)*` | …kopf im oberkörper [___] den beinen ja… |
| 212 | Löschung | `den` | `*(nicht da)*` | …im oberkörper in [___] beinen ja ich… |
| 213 | Löschung | `beinen` | `*(nicht da)*` | …oberkörper in den [___] ja ich bin… |
| 214 | Löschung | `ja` | `*(nicht da)*` | …in den beinen [___] ich bin auf… |
| 215 | Löschung | `ich` | `*(nicht da)*` | …den beinen ja [___] bin auf meine… |
| 216 | Löschung | `bin` | `*(nicht da)*` | …beinen ja ich [___] auf meine linke… |
| 217 | Löschung | `auf` | `*(nicht da)*` | …ja ich bin [___] meine linke seite… |
| 218 | Substitution | `meine` | `seit` | …ich bin auf [___] linke seite gefallen… |
| 219 | Substitution | `linke` | `ungefähr` | …bin auf meine [___] seite gefallen und… |
| 220 | Substitution | `seite` | `zwei` | …auf meine linke [___] gefallen und habe… |
| 221 | Substitution | `gefallen` | `monaten` | …meine linke seite [___] und habe mir… |
| 222 | Substitution | `und` | `es` | …linke seite gefallen [___] habe mir dabei… |
| 223 | Substitution | `habe` | `tut` | …seite gefallen und [___] mir dabei auch… |
| 224 | Löschung | `dabei` | `*(nicht da)*` | …und habe mir [___] auch tatsächlich den… |
| 225 | Löschung | `auch` | `*(nicht da)*` | …habe mir dabei [___] tatsächlich den kopf… |
| 226 | Löschung | `tatsächlich` | `*(nicht da)*` | …mir dabei auch [___] den kopf leicht… |
| 227 | Löschung | `den` | `*(nicht da)*` | …dabei auch tatsächlich [___] kopf leicht gestoßen… |
| 228 | Substitution | `kopf` | `immer` | …auch tatsächlich den [___] leicht gestoßen ich… |
| 229 | Substitution | `leicht` | `wieder` | …tatsächlich den kopf [___] gestoßen ich habe… |
| 230 | Substitution | `gestoßen` | `weh` | …den kopf leicht [___] ich habe leichte… |
| 231 | Substitution | `ich` | `besonders` | …kopf leicht gestoßen [___] habe leichte schmerzen… |
| 232 | Substitution | `habe` | `morgens` | …leicht gestoßen ich [___] leichte schmerzen am… |
| 233 | Substitution | `leichte` | `und` | …gestoßen ich habe [___] schmerzen am hinterkopf… |
| 234 | Substitution | `schmerzen` | `wie` | …ich habe leichte [___] am hinterkopf auf… |
| 235 | Substitution | `am` | `genau` | …habe leichte schmerzen [___] hinterkopf auf der… |
| 236 | Substitution | `hinterkopf` | `äußert` | …leichte schmerzen am [___] auf der linken… |
| 237 | Substitution | `auf` | `sich` | …schmerzen am hinterkopf [___] der linken seite… |
| 238 | Löschung | `linken` | `*(nicht da)*` | …hinterkopf auf der [___] seite ich kann… |
| 239 | Löschung | `seite` | `*(nicht da)*` | …auf der linken [___] ich kann außerdem… |
| 240 | Löschung | `ich` | `*(nicht da)*` | …der linken seite [___] kann außerdem meinen… |
| 241 | Löschung | `kann` | `*(nicht da)*` | …linken seite ich [___] außerdem meinen linken… |
| 242 | Löschung | `außerdem` | `*(nicht da)*` | …seite ich kann [___] meinen linken daumen… |
| 243 | Löschung | `meinen` | `*(nicht da)*` | …ich kann außerdem [___] linken daumen überhaupt… |
| 244 | Löschung | `linken` | `*(nicht da)*` | …kann außerdem meinen [___] daumen überhaupt nicht… |
| 245 | Löschung | `daumen` | `*(nicht da)*` | …außerdem meinen linken [___] überhaupt nicht bewegen… |
| 246 | Löschung | `überhaupt` | `*(nicht da)*` | …meinen linken daumen [___] nicht bewegen weil… |
| 247 | Löschung | `nicht` | `*(nicht da)*` | …linken daumen überhaupt [___] bewegen weil ich… |
| 248 | Löschung | `bewegen` | `*(nicht da)*` | …daumen überhaupt nicht [___] weil ich wirklich… |
| 249 | Löschung | `weil` | `*(nicht da)*` | …überhaupt nicht bewegen [___] ich wirklich starke… |
| 250 | Löschung | `ich` | `*(nicht da)*` | …nicht bewegen weil [___] wirklich starke schmerzen… |
| 251 | Löschung | `wirklich` | `*(nicht da)*` | …bewegen weil ich [___] starke schmerzen habe… |
| 252 | Löschung | `starke` | `*(nicht da)*` | …weil ich wirklich [___] schmerzen habe am… |
| 253 | Löschung | `schmerzen` | `*(nicht da)*` | …ich wirklich starke [___] habe am daumen… |
| 254 | Löschung | `habe` | `*(nicht da)*` | …wirklich starke schmerzen [___] am daumen und… |
| 255 | Löschung | `am` | `*(nicht da)*` | …starke schmerzen habe [___] daumen und er… |
| 256 | Löschung | `daumen` | `*(nicht da)*` | …schmerzen habe am [___] und er ist… |
| 257 | Löschung | `und` | `*(nicht da)*` | …habe am daumen [___] er ist auch… |
| 258 | Substitution | `er` | `schmerz` | …am daumen und [___] ist auch etwas… |
| 259 | Löschung | `auch` | `*(nicht da)*` | …und er ist [___] etwas geschwollen und… |
| 260 | Löschung | `etwas` | `*(nicht da)*` | …er ist auch [___] geschwollen und irgendwie… |
| 261 | Löschung | `geschwollen` | `*(nicht da)*` | …ist auch etwas [___] und irgendwie habe… |
| 262 | Löschung | `und` | `*(nicht da)*` | …auch etwas geschwollen [___] irgendwie habe ich… |
| 263 | Löschung | `irgendwie` | `*(nicht da)*` | …etwas geschwollen und [___] habe ich auch… |
| 264 | Löschung | `habe` | `*(nicht da)*` | …geschwollen und irgendwie [___] ich auch mein… |
| 265 | Substitution | `ich` | `es` | …und irgendwie habe [___] auch mein knie… |
| 266 | Substitution | `auch` | `ein` | …irgendwie habe ich [___] mein knie richtig… |
| 267 | Substitution | `mein` | `dumpfer` | …habe ich auch [___] knie richtig stark… |
| 268 | Substitution | `knie` | `schmerz` | …ich auch mein [___] richtig stark verletzt… |
| 269 | Substitution | `richtig` | `oder` | …auch mein knie [___] stark verletzt weil… |
| 270 | Substitution | `stark` | `eher` | …mein knie richtig [___] verletzt weil es… |
| 271 | Substitution | `verletzt` | `stechend` | …knie richtig stark [___] weil es richtig… |
| 272 | Substitution | `weil` | `eher` | …richtig stark verletzt [___] es richtig geschwollen… |
| 273 | Substitution | `es` | `ein` | …stark verletzt weil [___] richtig geschwollen ist… |
| 274 | Substitution | `richtig` | `dumpfer` | …verletzt weil es [___] geschwollen ist und… |
| 275 | Substitution | `geschwollen` | `schmerz` | …weil es richtig [___] ist und auch… |
| 276 | Substitution | `ist` | `der` | …es richtig geschwollen [___] und auch sehr… |
| 277 | Substitution | `und` | `manchmal` | …richtig geschwollen ist [___] auch sehr weh… |
| 278 | Löschung | `sehr` | `*(nicht da)*` | …ist und auch [___] weh tut okay… |
| 279 | Löschung | `weh` | `*(nicht da)*` | …und auch sehr [___] tut okay knie… |
| 280 | Substitution | `tut` | `ein` | …auch sehr weh [___] okay knie ist… |
| 281 | Substitution | `okay` | `bisschen` | …sehr weh tut [___] knie ist auch… |
| 282 | Substitution | `knie` | `ziehend` | …weh tut okay [___] ist auch geschwollen… |
| 283 | Löschung | `auch` | `*(nicht da)*` | …okay knie ist [___] geschwollen und starke… |
| 284 | Löschung | `geschwollen` | `*(nicht da)*` | …knie ist auch [___] und starke schmerzen… |
| 285 | Löschung | `starke` | `*(nicht da)*` | …auch geschwollen und [___] schmerzen sagen sie… |
| 286 | Löschung | `schmerzen` | `*(nicht da)*` | …geschwollen und starke [___] sagen sie genau… |
| 287 | Löschung | `sagen` | `*(nicht da)*` | …und starke schmerzen [___] sie genau frau… |
| 288 | Löschung | `sie` | `*(nicht da)*` | …starke schmerzen sagen [___] genau frau becken… |
| 289 | Löschung | `genau` | `*(nicht da)*` | …schmerzen sagen sie [___] frau becken westfalen… |
| 290 | Löschung | `frau` | `*(nicht da)*` | …sagen sie genau [___] becken westfalen haben… |
| 291 | Löschung | `becken` | `*(nicht da)*` | …sie genau frau [___] westfalen haben sie… |
| 292 | Löschung | `westfalen` | `*(nicht da)*` | …genau frau becken [___] haben sie denn… |
| 293 | Löschung | `haben` | `*(nicht da)*` | …frau becken westfalen [___] sie denn einen… |
| 294 | Löschung | `sie` | `*(nicht da)*` | …becken westfalen haben [___] denn einen fahrradhelm… |
| 295 | Löschung | `denn` | `*(nicht da)*` | …westfalen haben sie [___] einen fahrradhelm getragen… |
| 296 | Löschung | `einen` | `*(nicht da)*` | …haben sie denn [___] fahrradhelm getragen leider… |
| 297 | Löschung | `fahrradhelm` | `*(nicht da)*` | …sie denn einen [___] getragen leider nein… |
| 298 | Löschung | `getragen` | `*(nicht da)*` | …denn einen fahrradhelm [___] leider nein ich… |
| 299 | Löschung | `leider` | `*(nicht da)*` | …einen fahrradhelm getragen [___] nein ich muss… |
| 300 | Löschung | `nein` | `*(nicht da)*` | …fahrradhelm getragen leider [___] ich muss auch… |
| 301 | Löschung | `ich` | `*(nicht da)*` | …getragen leider nein [___] muss auch zugeben… |
| 302 | Substitution | `muss` | `es` | …leider nein ich [___] auch zugeben dass… |
| 303 | Substitution | `auch` | `ist` | …nein ich muss [___] zugeben dass ich… |
| 304 | Substitution | `zugeben` | `schlimmer` | …ich muss auch [___] dass ich sehr… |
| 305 | Substitution | `dass` | `wenn` | …muss auch zugeben [___] ich sehr ungern… |
| 306 | Löschung | `sehr` | `*(nicht da)*` | …zugeben dass ich [___] ungern einen fahrradhelm… |
| 307 | Löschung | `ungern` | `*(nicht da)*` | …dass ich sehr [___] einen fahrradhelm trage… |
| 308 | Löschung | `einen` | `*(nicht da)*` | …ich sehr ungern [___] fahrradhelm trage weil… |
| 309 | Löschung | `fahrradhelm` | `*(nicht da)*` | …sehr ungern einen [___] trage weil sie… |
| 310 | Löschung | `trage` | `*(nicht da)*` | …ungern einen fahrradhelm [___] weil sie mir… |
| 311 | Löschung | `weil` | `*(nicht da)*` | …einen fahrradhelm trage [___] sie mir so… |
| 312 | Löschung | `sie` | `*(nicht da)*` | …fahrradhelm trage weil [___] mir so unbequem… |
| 313 | Löschung | `mir` | `*(nicht da)*` | …trage weil sie [___] so unbequem sind… |
| 314 | Löschung | `so` | `*(nicht da)*` | …weil sie mir [___] unbequem sind und… |
| 315 | Löschung | `unbequem` | `*(nicht da)*` | …sie mir so [___] sind und es… |
| 316 | Löschung | `sind` | `*(nicht da)*` | …mir so unbequem [___] und es sieht… |
| 317 | Löschung | `und` | `*(nicht da)*` | …so unbequem sind [___] es sieht auch… |
| 318 | Löschung | `es` | `*(nicht da)*` | …unbequem sind und [___] sieht auch so… |
| 319 | Löschung | `sieht` | `*(nicht da)*` | …sind und es [___] auch so bescheuert… |
| 320 | Löschung | `auch` | `*(nicht da)*` | …und es sieht [___] so bescheuert aus… |
| 321 | Substitution | `so` | `treppen` | …es sieht auch [___] bescheuert aus sie… |
| 322 | Substitution | `bescheuert` | `steige` | …sieht auch so [___] aus sie als… |
| 323 | Substitution | `aus` | `haben` | …auch so bescheuert [___] sie als frau… |
| 324 | Löschung | `als` | `*(nicht da)*` | …bescheuert aus sie [___] frau würden mich… |
| 325 | Löschung | `frau` | `*(nicht da)*` | …aus sie als [___] würden mich sicherlich… |
| 326 | Löschung | `würden` | `*(nicht da)*` | …sie als frau [___] mich sicherlich verstehen… |
| 327 | Löschung | `mich` | `*(nicht da)*` | …als frau würden [___] sicherlich verstehen ich… |
| 328 | Substitution | `sicherlich` | `irgendwelche` | …frau würden mich [___] verstehen ich verstehe… |
| 329 | Substitution | `verstehen` | `vorerkrankungen` | …würden mich sicherlich [___] ich verstehe sie… |
| 330 | Löschung | `verstehe` | `*(nicht da)*` | …sicherlich verstehen ich [___] sie voll und… |
| 331 | Löschung | `sie` | `*(nicht da)*` | …verstehen ich verstehe [___] voll und ganz… |
| 332 | Löschung | `voll` | `*(nicht da)*` | …ich verstehe sie [___] und ganz meiner… |
| 333 | Löschung | `und` | `*(nicht da)*` | …verstehe sie voll [___] ganz meiner frisur… |
| 334 | Löschung | `ganz` | `*(nicht da)*` | …sie voll und [___] meiner frisur tut… |
| 335 | Löschung | `meiner` | `*(nicht da)*` | …voll und ganz [___] frisur tut das… |
| 336 | Substitution | `frisur` | `habe` | …und ganz meiner [___] tut das auch… |
| 337 | Substitution | `tut` | `bluthochdruck` | …ganz meiner frisur [___] das auch nicht… |
| 338 | Löschung | `auch` | `*(nicht da)*` | …frisur tut das [___] nicht gut aber… |
| 339 | Löschung | `nicht` | `*(nicht da)*` | …tut das auch [___] gut aber da… |
| 340 | Löschung | `gut` | `*(nicht da)*` | …das auch nicht [___] aber da muss… |
| 341 | Löschung | `aber` | `*(nicht da)*` | …auch nicht gut [___] da muss ich… |
| 342 | Löschung | `da` | `*(nicht da)*` | …nicht gut aber [___] muss ich ihnen… |
| 343 | Substitution | `muss` | `nehme` | …gut aber da [___] ich ihnen leider… |
| 344 | Löschung | `ihnen` | `*(nicht da)*` | …da muss ich [___] leider sagen in… |
| 345 | Löschung | `leider` | `*(nicht da)*` | …muss ich ihnen [___] sagen in diesem… |
| 346 | Löschung | `sagen` | `*(nicht da)*` | …ich ihnen leider [___] in diesem fall… |
| 347 | Löschung | `in` | `*(nicht da)*` | …ihnen leider sagen [___] diesem fall gehen… |
| 348 | Löschung | `diesem` | `*(nicht da)*` | …leider sagen in [___] fall gehen sicherheit… |
| 349 | Substitution | `fall` | `regelmäßig` | …sagen in diesem [___] gehen sicherheit und… |
| 350 | Substitution | `gehen` | `mit` | …in diesem fall [___] sicherheit und gesundheit… |
| 351 | Substitution | `sicherheit` | `tabletten` | …diesem fall gehen [___] und gesundheit definitiv… |
| 352 | Löschung | `gesundheit` | `*(nicht da)*` | …gehen sicherheit und [___] definitiv vor aussehen… |
| 353 | Löschung | `definitiv` | `*(nicht da)*` | …sicherheit und gesundheit [___] vor aussehen frau… |
| 354 | Löschung | `vor` | `*(nicht da)*` | …und gesundheit definitiv [___] aussehen frau becken… |
| 355 | Löschung | `aussehen` | `*(nicht da)*` | …gesundheit definitiv vor [___] frau becken westfalen… |
| 356 | Löschung | `frau` | `*(nicht da)*` | …definitiv vor aussehen [___] becken westfalen bitte… |
| 357 | Löschung | `becken` | `*(nicht da)*` | …vor aussehen frau [___] westfalen bitte bitte… |
| 358 | Löschung | `westfalen` | `*(nicht da)*` | …aussehen frau becken [___] bitte bitte tragen… |
| 359 | Löschung | `bitte` | `*(nicht da)*` | …frau becken westfalen [___] bitte tragen sie… |
| 360 | Löschung | `bitte` | `*(nicht da)*` | …becken westfalen bitte [___] tragen sie beim… |
| 361 | Löschung | `tragen` | `*(nicht da)*` | …westfalen bitte bitte [___] sie beim nächsten… |
| 362 | Löschung | `sie` | `*(nicht da)*` | …bitte bitte tragen [___] beim nächsten mal… |
| 363 | Löschung | `beim` | `*(nicht da)*` | …bitte tragen sie [___] nächsten mal einen… |
| 364 | Löschung | `nächsten` | `*(nicht da)*` | …tragen sie beim [___] mal einen helm… |
| 365 | Löschung | `mal` | `*(nicht da)*` | …sie beim nächsten [___] einen helm da… |
| 366 | Löschung | `einen` | `*(nicht da)*` | …beim nächsten mal [___] helm da haben… |
| 367 | Löschung | `helm` | `*(nicht da)*` | …nächsten mal einen [___] da haben sie… |
| 368 | Löschung | `da` | `*(nicht da)*` | …mal einen helm [___] haben sie diesmal… |
| 369 | Löschung | `haben` | `*(nicht da)*` | …einen helm da [___] sie diesmal wirklich… |
| 370 | Löschung | `sie` | `*(nicht da)*` | …helm da haben [___] diesmal wirklich noch… |
| 371 | Löschung | `diesmal` | `*(nicht da)*` | …da haben sie [___] wirklich noch glück… |
| 372 | Löschung | `wirklich` | `*(nicht da)*` | …haben sie diesmal [___] noch glück gehabt… |
| 373 | Löschung | `noch` | `*(nicht da)*` | …sie diesmal wirklich [___] glück gehabt dass… |
| 374 | Löschung | `glück` | `*(nicht da)*` | …diesmal wirklich noch [___] gehabt dass nichts… |
| 375 | Löschung | `gehabt` | `*(nicht da)*` | …wirklich noch glück [___] dass nichts passiert… |
| 376 | Löschung | `dass` | `*(nicht da)*` | …noch glück gehabt [___] nichts passiert ist… |
| 377 | Löschung | `nichts` | `*(nicht da)*` | …glück gehabt dass [___] passiert ist da… |
| 378 | Löschung | `passiert` | `*(nicht da)*` | …gehabt dass nichts [___] ist da haben… |
| 379 | Löschung | `ist` | `*(nicht da)*` | …dass nichts passiert [___] da haben sie… |
| 380 | Löschung | `da` | `*(nicht da)*` | …nichts passiert ist [___] haben sie auf… |
| 381 | Substitution | `haben` | `nehmen` | …passiert ist da [___] sie auf jeden… |
| 382 | Löschung | `auf` | `*(nicht da)*` | …da haben sie [___] jeden fall recht… |
| 383 | Löschung | `jeden` | `*(nicht da)*` | …haben sie auf [___] fall recht ich… |
| 384 | Löschung | `fall` | `*(nicht da)*` | …sie auf jeden [___] recht ich habe… |
| 385 | Löschung | `recht` | `*(nicht da)*` | …auf jeden fall [___] ich habe jetzt… |
| 386 | Löschung | `ich` | `*(nicht da)*` | …jeden fall recht [___] habe jetzt draus… |
| 387 | Löschung | `habe` | `*(nicht da)*` | …fall recht ich [___] jetzt draus gelernt… |
| 388 | Löschung | `jetzt` | `*(nicht da)*` | …recht ich habe [___] draus gelernt und… |
| 389 | Löschung | `draus` | `*(nicht da)*` | …ich habe jetzt [___] gelernt und werde… |
| 390 | Löschung | `gelernt` | `*(nicht da)*` | …habe jetzt draus [___] und werde mir… |
| 391 | Löschung | `und` | `*(nicht da)*` | …jetzt draus gelernt [___] werde mir auch… |
| 392 | Löschung | `werde` | `*(nicht da)*` | …draus gelernt und [___] mir auch einen… |
| 393 | Löschung | `mir` | `*(nicht da)*` | …gelernt und werde [___] auch einen besorgen… |
| 394 | Löschung | `einen` | `*(nicht da)*` | …werde mir auch [___] besorgen okay sehr… |
| 395 | Löschung | `besorgen` | `*(nicht da)*` | …mir auch einen [___] okay sehr gut… |
| 396 | Löschung | `okay` | `*(nicht da)*` | …auch einen besorgen [___] sehr gut sie… |
| 397 | Löschung | `sehr` | `*(nicht da)*` | …einen besorgen okay [___] gut sie hatten… |
| 398 | Löschung | `gut` | `*(nicht da)*` | …besorgen okay sehr [___] sie hatten gesagt… |
| 399 | Löschung | `sie` | `*(nicht da)*` | …okay sehr gut [___] hatten gesagt sie… |
| 400 | Löschung | `hatten` | `*(nicht da)*` | …sehr gut sie [___] gesagt sie haben… |
| 401 | Löschung | `gesagt` | `*(nicht da)*` | …gut sie hatten [___] sie haben hinten… |
| 402 | Löschung | `sie` | `*(nicht da)*` | …sie hatten gesagt [___] haben hinten auf… |
| 403 | Löschung | `haben` | `*(nicht da)*` | …hatten gesagt sie [___] hinten auf der… |
| 404 | Löschung | `hinten` | `*(nicht da)*` | …gesagt sie haben [___] auf der linken… |
| 405 | Löschung | `auf` | `*(nicht da)*` | …sie haben hinten [___] der linken seite… |
| 406 | Löschung | `der` | `*(nicht da)*` | …haben hinten auf [___] linken seite des… |
| 407 | Löschung | `linken` | `*(nicht da)*` | …hinten auf der [___] seite des hinterkopfes… |
| 408 | Löschung | `seite` | `*(nicht da)*` | …auf der linken [___] des hinterkopfes eine… |
| 409 | Löschung | `des` | `*(nicht da)*` | …der linken seite [___] hinterkopfes eine kleine… |
| 410 | Löschung | `hinterkopfes` | `*(nicht da)*` | …linken seite des [___] eine kleine beule… |
| 411 | Löschung | `eine` | `*(nicht da)*` | …seite des hinterkopfes [___] kleine beule richtig… |
| 412 | Löschung | `kleine` | `*(nicht da)*` | …des hinterkopfes eine [___] beule richtig genau… |
| 413 | Löschung | `beule` | `*(nicht da)*` | …hinterkopfes eine kleine [___] richtig genau haben… |
| 414 | Löschung | `richtig` | `*(nicht da)*` | …eine kleine beule [___] genau haben sie… |
| 415 | Löschung | `genau` | `*(nicht da)*` | …kleine beule richtig [___] haben sie irgendeine… |
| 416 | Löschung | `haben` | `*(nicht da)*` | …beule richtig genau [___] sie irgendeine blutige… |
| 417 | Löschung | `sie` | `*(nicht da)*` | …richtig genau haben [___] irgendeine blutige verletzung… |
| 418 | Löschung | `irgendeine` | `*(nicht da)*` | …genau haben sie [___] blutige verletzung am… |
| 419 | Löschung | `blutige` | `*(nicht da)*` | …haben sie irgendeine [___] verletzung am kopf… |
| 420 | Löschung | `verletzung` | `*(nicht da)*` | …sie irgendeine blutige [___] am kopf oder… |
| 421 | Löschung | `am` | `*(nicht da)*` | …irgendeine blutige verletzung [___] kopf oder ist… |
| 422 | Löschung | `kopf` | `*(nicht da)*` | …blutige verletzung am [___] oder ist das… |
| 423 | Löschung | `oder` | `*(nicht da)*` | …verletzung am kopf [___] ist das alles… |
| 424 | Löschung | `ist` | `*(nicht da)*` | …am kopf oder [___] das alles es… |
| 425 | Löschung | `das` | `*(nicht da)*` | …kopf oder ist [___] alles es ist… |
| 426 | Löschung | `alles` | `*(nicht da)*` | …oder ist das [___] es ist mir… |
| 427 | Löschung | `es` | `*(nicht da)*` | …ist das alles [___] ist mir nichts… |
| 428 | Löschung | `ist` | `*(nicht da)*` | …das alles es [___] mir nichts anderes… |
| 429 | Löschung | `mir` | `*(nicht da)*` | …alles es ist [___] nichts anderes aufgefallen… |
| 430 | Löschung | `nichts` | `*(nicht da)*` | …es ist mir [___] anderes aufgefallen zum… |
| 431 | Löschung | `anderes` | `*(nicht da)*` | …ist mir nichts [___] aufgefallen zum glück… |
| 432 | Löschung | `aufgefallen` | `*(nicht da)*` | …mir nichts anderes [___] zum glück ist… |
| 433 | Löschung | `zum` | `*(nicht da)*` | …nichts anderes aufgefallen [___] glück ist es… |
| 434 | Löschung | `glück` | `*(nicht da)*` | …anderes aufgefallen zum [___] ist es glaube… |
| 435 | Löschung | `ist` | `*(nicht da)*` | …aufgefallen zum glück [___] es glaube ich… |
| 436 | Substitution | `es` | `andere` | …zum glück ist [___] glaube ich nur… |
| 437 | Substitution | `glaube` | `medikamente` | …glück ist es [___] ich nur die… |
| 438 | Substitution | `ich` | `nein` | …ist es glaube [___] nur die beule… |
| 439 | Löschung | `die` | `*(nicht da)*` | …glaube ich nur [___] beule okay sehr… |
| 440 | Löschung | `beule` | `*(nicht da)*` | …ich nur die [___] okay sehr gut… |
| 441 | Löschung | `okay` | `*(nicht da)*` | …nur die beule [___] sehr gut die… |
| 442 | Löschung | `sehr` | `*(nicht da)*` | …die beule okay [___] gut die schmerzen… |
| 443 | Löschung | `gut` | `*(nicht da)*` | …beule okay sehr [___] die schmerzen sind… |
| 444 | Löschung | `die` | `*(nicht da)*` | …okay sehr gut [___] schmerzen sind die… |
| 445 | Löschung | `schmerzen` | `*(nicht da)*` | …sehr gut die [___] sind die stark… |
| 446 | Löschung | `sind` | `*(nicht da)*` | …gut die schmerzen [___] die stark oder… |
| 447 | Löschung | `die` | `*(nicht da)*` | …die schmerzen sind [___] stark oder geht… |
| 448 | Löschung | `stark` | `*(nicht da)*` | …schmerzen sind die [___] oder geht es… |
| 449 | Löschung | `oder` | `*(nicht da)*` | …sind die stark [___] geht es die… |
| 450 | Löschung | `geht` | `*(nicht da)*` | …die stark oder [___] es die sind… |
| 451 | Löschung | `es` | `*(nicht da)*` | …stark oder geht [___] die sind nicht… |
| 452 | Löschung | `die` | `*(nicht da)*` | …oder geht es [___] sind nicht so… |
| 453 | Löschung | `sind` | `*(nicht da)*` | …geht es die [___] nicht so stark… |
| 454 | Löschung | `nicht` | `*(nicht da)*` | …es die sind [___] so stark das… |
| 455 | Löschung | `so` | `*(nicht da)*` | …die sind nicht [___] stark das geht… |
| 456 | Löschung | `stark` | `*(nicht da)*` | …sind nicht so [___] das geht tatsächlich… |
| 457 | Löschung | `das` | `*(nicht da)*` | …nicht so stark [___] geht tatsächlich am… |
| 458 | Löschung | `geht` | `*(nicht da)*` | …so stark das [___] tatsächlich am kopf… |
| 459 | Löschung | `tatsächlich` | `*(nicht da)*` | …stark das geht [___] am kopf sind… |
| 460 | Löschung | `am` | `*(nicht da)*` | …das geht tatsächlich [___] kopf sind die… |
| 461 | Löschung | `kopf` | `*(nicht da)*` | …geht tatsächlich am [___] sind die am… |
| 462 | Löschung | `sind` | `*(nicht da)*` | …tatsächlich am kopf [___] die am schwächsten… |
| 463 | Löschung | `am` | `*(nicht da)*` | …kopf sind die [___] schwächsten okay alles… |
| 464 | Löschung | `schwächsten` | `*(nicht da)*` | …sind die am [___] okay alles klar… |
| 465 | Löschung | `okay` | `*(nicht da)*` | …die am schwächsten [___] alles klar der… |
| 466 | Löschung | `alles` | `*(nicht da)*` | …am schwächsten okay [___] klar der daumen… |
| 467 | Löschung | `klar` | `*(nicht da)*` | …schwächsten okay alles [___] der daumen sie… |
| 468 | Löschung | `der` | `*(nicht da)*` | …okay alles klar [___] daumen sie haben… |
| 469 | Löschung | `daumen` | `*(nicht da)*` | …alles klar der [___] sie haben jetzt… |
| 470 | Löschung | `sie` | `*(nicht da)*` | …klar der daumen [___] haben jetzt gesagt… |
| 471 | Löschung | `haben` | `*(nicht da)*` | …der daumen sie [___] jetzt gesagt sie… |
| 472 | Löschung | `jetzt` | `*(nicht da)*` | …daumen sie haben [___] gesagt sie können… |
| 473 | Löschung | `gesagt` | `*(nicht da)*` | …sie haben jetzt [___] sie können den… |
| 474 | Löschung | `sie` | `*(nicht da)*` | …haben jetzt gesagt [___] können den daumen… |
| 475 | Substitution | `können` | `für` | …jetzt gesagt sie [___] den daumen gar… |
| 476 | Löschung | `daumen` | `*(nicht da)*` | …sie können den [___] gar nicht mehr… |
| 477 | Löschung | `gar` | `*(nicht da)*` | …können den daumen [___] nicht mehr recht… |
| 478 | Löschung | `nicht` | `*(nicht da)*` | …den daumen gar [___] mehr recht bewegen… |
| 479 | Löschung | `mehr` | `*(nicht da)*` | …daumen gar nicht [___] recht bewegen wenn… |
| 480 | Löschung | `recht` | `*(nicht da)*` | …gar nicht mehr [___] bewegen wenn wir… |
| 481 | Löschung | `bewegen` | `*(nicht da)*` | …nicht mehr recht [___] wenn wir jetzt… |
| 482 | Löschung | `wenn` | `*(nicht da)*` | …mehr recht bewegen [___] wir jetzt die… |
| 483 | Löschung | `wir` | `*(nicht da)*` | …recht bewegen wenn [___] jetzt die schmerzen… |
| 484 | Löschung | `jetzt` | `*(nicht da)*` | …bewegen wenn wir [___] die schmerzen einschätzen… |
| 485 | Löschung | `die` | `*(nicht da)*` | …wenn wir jetzt [___] schmerzen einschätzen an… |
| 486 | Löschung | `schmerzen` | `*(nicht da)*` | …wir jetzt die [___] einschätzen an einer… |
| 487 | Löschung | `einschätzen` | `*(nicht da)*` | …jetzt die schmerzen [___] an einer schmerzskala… |
| 488 | Löschung | `an` | `*(nicht da)*` | …die schmerzen einschätzen [___] einer schmerzskala wobei… |
| 489 | Löschung | `einer` | `*(nicht da)*` | …schmerzen einschätzen an [___] schmerzskala wobei 1… |
| 490 | Löschung | `schmerzskala` | `*(nicht da)*` | …einschätzen an einer [___] wobei 1 sehr… |
| 491 | Löschung | `wobei` | `*(nicht da)*` | …an einer schmerzskala [___] 1 sehr leichten… |
| 492 | Löschung | `1` | `*(nicht da)*` | …einer schmerzskala wobei [___] sehr leichten schmerzen… |
| 493 | Löschung | `sehr` | `*(nicht da)*` | …schmerzskala wobei 1 [___] leichten schmerzen entspricht… |
| 494 | Löschung | `leichten` | `*(nicht da)*` | …wobei 1 sehr [___] schmerzen entspricht und… |
| 495 | Löschung | `schmerzen` | `*(nicht da)*` | …1 sehr leichten [___] entspricht und 10… |
| 496 | Löschung | `entspricht` | `*(nicht da)*` | …sehr leichten schmerzen [___] und 10 sehr… |
| 497 | Löschung | `und` | `*(nicht da)*` | …leichten schmerzen entspricht [___] 10 sehr starken… |
| 498 | Löschung | `10` | `*(nicht da)*` | …schmerzen entspricht und [___] sehr starken schmerzen… |
| 499 | Löschung | `sehr` | `*(nicht da)*` | …entspricht und 10 [___] starken schmerzen wo… |
| 500 | Löschung | `starken` | `*(nicht da)*` | …und 10 sehr [___] schmerzen wo würden… |
| 501 | Löschung | `schmerzen` | `*(nicht da)*` | …10 sehr starken [___] wo würden sie… |
| 502 | Substitution | `wo` | `blutdruck` | …sehr starken schmerzen [___] würden sie die… |
| 503 | Substitution | `würden` | `haben` | …starken schmerzen wo [___] sie die schmerzen… |
| 504 | Löschung | `die` | `*(nicht da)*` | …wo würden sie [___] schmerzen des daumens… |
| 505 | Löschung | `schmerzen` | `*(nicht da)*` | …würden sie die [___] des daumens einstufen… |
| 506 | Löschung | `des` | `*(nicht da)*` | …sie die schmerzen [___] daumens einstufen beim… |
| 507 | Löschung | `daumens` | `*(nicht da)*` | …die schmerzen des [___] einstufen beim daumen… |
| 508 | Löschung | `einstufen` | `*(nicht da)*` | …schmerzen des daumens [___] beim daumen würde… |
| 509 | Löschung | `beim` | `*(nicht da)*` | …des daumens einstufen [___] daumen würde ich… |
| 510 | Löschung | `daumen` | `*(nicht da)*` | …daumens einstufen beim [___] würde ich schon… |
| 511 | Löschung | `würde` | `*(nicht da)*` | …einstufen beim daumen [___] ich schon sagen… |
| 512 | Löschung | `ich` | `*(nicht da)*` | …beim daumen würde [___] schon sagen geht… |
| 513 | Löschung | `schon` | `*(nicht da)*` | …daumen würde ich [___] sagen geht es… |
| 514 | Löschung | `sagen` | `*(nicht da)*` | …würde ich schon [___] geht es so… |
| 515 | Löschung | `geht` | `*(nicht da)*` | …ich schon sagen [___] es so auf… |
| 516 | Löschung | `es` | `*(nicht da)*` | …schon sagen geht [___] so auf die… |
| 517 | Löschung | `so` | `*(nicht da)*` | …sagen geht es [___] auf die 7… |
| 518 | Löschung | `auf` | `*(nicht da)*` | …geht es so [___] die 7 zu… |
| 519 | Löschung | `die` | `*(nicht da)*` | …es so auf [___] 7 zu vor… |
| 520 | Löschung | `7` | `*(nicht da)*` | …so auf die [___] zu vor allem… |
| 521 | Löschung | `zu` | `*(nicht da)*` | …auf die 7 [___] vor allem wenn… |
| 522 | Substitution | `vor` | `irgendwelche` | …die 7 zu [___] allem wenn ich… |
| 523 | Substitution | `allem` | `allergien` | …7 zu vor [___] wenn ich versuche… |
| 524 | Substitution | `wenn` | `nein` | …zu vor allem [___] ich versuche ihn… |
| 525 | Löschung | `versuche` | `*(nicht da)*` | …allem wenn ich [___] ihn zu bewegen… |
| 526 | Löschung | `ihn` | `*(nicht da)*` | …wenn ich versuche [___] zu bewegen okay… |
| 527 | Löschung | `zu` | `*(nicht da)*` | …ich versuche ihn [___] bewegen okay was… |
| 528 | Löschung | `bewegen` | `*(nicht da)*` | …versuche ihn zu [___] okay was ist… |
| 529 | Löschung | `okay` | `*(nicht da)*` | …ihn zu bewegen [___] was ist das… |
| 530 | Löschung | `was` | `*(nicht da)*` | …zu bewegen okay [___] ist das denn… |
| 531 | Löschung | `ist` | `*(nicht da)*` | …bewegen okay was [___] das denn für… |
| 532 | Löschung | `das` | `*(nicht da)*` | …okay was ist [___] denn für ein… |
| 533 | Löschung | `denn` | `*(nicht da)*` | …was ist das [___] für ein schmerz… |
| 534 | Löschung | `für` | `*(nicht da)*` | …ist das denn [___] ein schmerz ist… |
| 535 | Löschung | `ein` | `*(nicht da)*` | …das denn für [___] schmerz ist das… |
| 536 | Löschung | `schmerz` | `*(nicht da)*` | …denn für ein [___] ist das ein… |
| 537 | Löschung | `ist` | `*(nicht da)*` | …für ein schmerz [___] das ein stechender… |
| 538 | Löschung | `das` | `*(nicht da)*` | …ein schmerz ist [___] ein stechender schmerz… |
| 539 | Löschung | `ein` | `*(nicht da)*` | …schmerz ist das [___] stechender schmerz ein… |
| 540 | Löschung | `stechender` | `*(nicht da)*` | …ist das ein [___] schmerz ein ziehender… |
| 541 | Löschung | `schmerz` | `*(nicht da)*` | …das ein stechender [___] ein ziehender schmerz… |
| 542 | Löschung | `ein` | `*(nicht da)*` | …ein stechender schmerz [___] ziehender schmerz ein… |
| 543 | Substitution | `ziehender` | `bin` | …stechender schmerz ein [___] schmerz ein brennender… |
| 544 | Substitution | `schmerz` | `nicht` | …schmerz ein ziehender [___] ein brennender schmerz… |
| 545 | Substitution | `ein` | `allergisch` | …ein ziehender schmerz [___] brennender schmerz das… |
| 546 | Substitution | `brennender` | `gut` | …ziehender schmerz ein [___] schmerz das ist… |
| 547 | Substitution | `schmerz` | `und` | …schmerz ein brennender [___] das ist ein… |
| 548 | Substitution | `das` | `wie` | …ein brennender schmerz [___] ist ein stechender… |
| 549 | Substitution | `ein` | `ihre` | …schmerz das ist [___] stechender schmerz würde… |
| 550 | Substitution | `stechender` | `allgemeine` | …das ist ein [___] schmerz würde ich… |
| 551 | Substitution | `schmerz` | `körperliche` | …ist ein stechender [___] würde ich sagen… |
| 552 | Substitution | `würde` | `verfassung` | …ein stechender schmerz [___] ich sagen sehr… |
| 553 | Löschung | `sagen` | `*(nicht da)*` | …schmerz würde ich [___] sehr stark stechen… |
| 554 | Löschung | `sehr` | `*(nicht da)*` | …würde ich sagen [___] stark stechen wenn… |
| 555 | Substitution | `stark` | `bin` | …ich sagen sehr [___] stechen wenn ich… |
| 556 | Substitution | `stechen` | `relativ` | …sagen sehr stark [___] wenn ich versuche… |
| 557 | Substitution | `wenn` | `fit` | …sehr stark stechen [___] ich versuche ihn… |
| 558 | Löschung | `versuche` | `*(nicht da)*` | …stechen wenn ich [___] ihn zu bewegen… |
| 559 | Löschung | `ihn` | `*(nicht da)*` | …wenn ich versuche [___] zu bewegen okay… |
| 560 | Substitution | `zu` | `gehe` | …ich versuche ihn [___] bewegen okay und… |
| 561 | Substitution | `bewegen` | `regelmäßig` | …versuche ihn zu [___] okay und wie… |
| 562 | Substitution | `okay` | `spazieren` | …ihn zu bewegen [___] und wie sieht… |
| 563 | Löschung | `sieht` | `*(nicht da)*` | …okay und wie [___] es am knie… |
| 564 | Löschung | `es` | `*(nicht da)*` | …und wie sieht [___] am knie aus… |
| 565 | Löschung | `am` | `*(nicht da)*` | …wie sieht es [___] knie aus können… |
| 566 | Löschung | `knie` | `*(nicht da)*` | …sieht es am [___] aus können sie… |
| 567 | Löschung | `aus` | `*(nicht da)*` | …es am knie [___] können sie das… |
| 568 | Löschung | `können` | `*(nicht da)*` | …am knie aus [___] sie das knie… |
| 569 | Löschung | `sie` | `*(nicht da)*` | …knie aus können [___] das knie bewegen… |
| 570 | Löschung | `das` | `*(nicht da)*` | …aus können sie [___] knie bewegen sehr… |
| 571 | Löschung | `knie` | `*(nicht da)*` | …können sie das [___] bewegen sehr sehr… |
| 572 | Löschung | `bewegen` | `*(nicht da)*` | …sie das knie [___] sehr sehr schwer… |
| 573 | Löschung | `sehr` | `*(nicht da)*` | …das knie bewegen [___] sehr schwer da… |
| 574 | Löschung | `sehr` | `*(nicht da)*` | …knie bewegen sehr [___] schwer da tut… |
| 575 | Löschung | `schwer` | `*(nicht da)*` | …bewegen sehr sehr [___] da tut es… |
| 576 | Löschung | `da` | `*(nicht da)*` | …sehr sehr schwer [___] tut es wirklich… |
| 577 | Löschung | `tut` | `*(nicht da)*` | …sehr schwer da [___] es wirklich sehr… |
| 578 | Löschung | `es` | `*(nicht da)*` | …schwer da tut [___] wirklich sehr stark… |
| 579 | Löschung | `wirklich` | `*(nicht da)*` | …da tut es [___] sehr stark weh… |
| 580 | Löschung | `sehr` | `*(nicht da)*` | …tut es wirklich [___] stark weh wenn… |
| 581 | Löschung | `stark` | `*(nicht da)*` | …es wirklich sehr [___] weh wenn ich… |
| 582 | Löschung | `weh` | `*(nicht da)*` | …wirklich sehr stark [___] wenn ich versuche… |
| 583 | Löschung | `wenn` | `*(nicht da)*` | …sehr stark weh [___] ich versuche mein… |
| 584 | Löschung | `ich` | `*(nicht da)*` | …stark weh wenn [___] versuche mein knie… |
| 585 | Löschung | `versuche` | `*(nicht da)*` | …weh wenn ich [___] mein knie zu… |
| 586 | Löschung | `mein` | `*(nicht da)*` | …wenn ich versuche [___] knie zu bewegen… |
| 587 | Löschung | `knie` | `*(nicht da)*` | …ich versuche mein [___] zu bewegen es… |
| 588 | Löschung | `zu` | `*(nicht da)*` | …versuche mein knie [___] bewegen es tut… |
| 589 | Löschung | `bewegen` | `*(nicht da)*` | …mein knie zu [___] es tut selbst… |
| 590 | Löschung | `es` | `*(nicht da)*` | …knie zu bewegen [___] tut selbst weh… |
| 591 | Löschung | `tut` | `*(nicht da)*` | …zu bewegen es [___] selbst weh wenn… |
| 592 | Löschung | `selbst` | `*(nicht da)*` | …bewegen es tut [___] weh wenn ich… |
| 593 | Substitution | `weh` | `oft` | …es tut selbst [___] wenn ich gerade… |
| 594 | Substitution | `wenn` | `ungefähr` | …tut selbst weh [___] ich gerade einfach… |
| 595 | Substitution | `ich` | `mehrmals` | …selbst weh wenn [___] gerade einfach so… |
| 596 | Substitution | `gerade` | `pro` | …weh wenn ich [___] einfach so hier… |
| 597 | Substitution | `einfach` | `woche` | …wenn ich gerade [___] so hier sitze… |
| 598 | Löschung | `hier` | `*(nicht da)*` | …gerade einfach so [___] sitze okay sogar… |
| 599 | Löschung | `sitze` | `*(nicht da)*` | …einfach so hier [___] okay sogar im… |
| 600 | Löschung | `okay` | `*(nicht da)*` | …so hier sitze [___] sogar im ruhezustand… |
| 601 | Löschung | `sogar` | `*(nicht da)*` | …hier sitze okay [___] im ruhezustand ja… |
| 602 | Löschung | `im` | `*(nicht da)*` | …sitze okay sogar [___] ruhezustand ja wo… |
| 603 | Löschung | `ruhezustand` | `*(nicht da)*` | …okay sogar im [___] ja wo würden… |
| 604 | Löschung | `ja` | `*(nicht da)*` | …sogar im ruhezustand [___] wo würden sie… |
| 605 | Löschung | `wo` | `*(nicht da)*` | …im ruhezustand ja [___] würden sie die… |
| 606 | Löschung | `würden` | `*(nicht da)*` | …ruhezustand ja wo [___] sie die schmerzen… |
| 607 | Löschung | `sie` | `*(nicht da)*` | …ja wo würden [___] die schmerzen hier… |
| 608 | Löschung | `die` | `*(nicht da)*` | …wo würden sie [___] schmerzen hier einstufen… |
| 609 | Löschung | `schmerzen` | `*(nicht da)*` | …würden sie die [___] hier einstufen da… |
| 610 | Löschung | `hier` | `*(nicht da)*` | …sie die schmerzen [___] einstufen da würde… |
| 611 | Löschung | `einstufen` | `*(nicht da)*` | …die schmerzen hier [___] da würde ich… |
| 612 | Löschung | `da` | `*(nicht da)*` | …schmerzen hier einstufen [___] würde ich sagen… |
| 613 | Löschung | `würde` | `*(nicht da)*` | …hier einstufen da [___] ich sagen bei… |
| 614 | Löschung | `ich` | `*(nicht da)*` | …einstufen da würde [___] sagen bei 8… |
| 615 | Löschung | `sagen` | `*(nicht da)*` | …da würde ich [___] bei 8 wenn… |
| 616 | Löschung | `bei` | `*(nicht da)*` | …würde ich sagen [___] 8 wenn ich… |
| 617 | Substitution | `8` | `drei` | …ich sagen bei [___] wenn ich sitze… |
| 618 | Substitution | `wenn` | `bis` | …sagen bei 8 [___] ich sitze und… |
| 619 | Substitution | `ich` | `vier` | …bei 8 wenn [___] sitze und wenn… |
| 620 | Substitution | `sitze` | `mal` | …8 wenn ich [___] und wenn ich… |
| 621 | Löschung | `wenn` | `*(nicht da)*` | …ich sitze und [___] ich versuche mein… |
| 622 | Löschung | `ich` | `*(nicht da)*` | …sitze und wenn [___] versuche mein knie… |
| 623 | Löschung | `versuche` | `*(nicht da)*` | …und wenn ich [___] mein knie zu… |
| 624 | Löschung | `mein` | `*(nicht da)*` | …wenn ich versuche [___] knie zu bewegen… |
| 625 | Löschung | `knie` | `*(nicht da)*` | …ich versuche mein [___] zu bewegen ist… |
| 626 | Löschung | `zu` | `*(nicht da)*` | …versuche mein knie [___] bewegen ist es… |
| 627 | Löschung | `bewegen` | `*(nicht da)*` | …mein knie zu [___] ist es wirklich… |
| 628 | Löschung | `ist` | `*(nicht da)*` | …knie zu bewegen [___] es wirklich unerträglich… |
| 629 | Löschung | `es` | `*(nicht da)*` | …zu bewegen ist [___] wirklich unerträglich okay… |
| 630 | Löschung | `wirklich` | `*(nicht da)*` | …bewegen ist es [___] unerträglich okay okay… |
| 631 | Löschung | `unerträglich` | `*(nicht da)*` | …ist es wirklich [___] okay okay gut… |
| 632 | Löschung | `okay` | `*(nicht da)*` | …es wirklich unerträglich [___] okay gut strahlen… |
| 633 | Löschung | `okay` | `*(nicht da)*` | …wirklich unerträglich okay [___] gut strahlen die… |
| 634 | Löschung | `gut` | `*(nicht da)*` | …unerträglich okay okay [___] strahlen die schmerzen… |
| 635 | Löschung | `strahlen` | `*(nicht da)*` | …okay okay gut [___] die schmerzen noch… |
| 636 | Löschung | `die` | `*(nicht da)*` | …okay gut strahlen [___] schmerzen noch in… |
| 637 | Löschung | `schmerzen` | `*(nicht da)*` | …gut strahlen die [___] noch in andere… |
| 638 | Löschung | `noch` | `*(nicht da)*` | …strahlen die schmerzen [___] in andere körperregionen… |
| 639 | Löschung | `in` | `*(nicht da)*` | …die schmerzen noch [___] andere körperregionen aus… |
| 640 | Löschung | `andere` | `*(nicht da)*` | …schmerzen noch in [___] körperregionen aus nein… |
| 641 | Löschung | `körperregionen` | `*(nicht da)*` | …noch in andere [___] aus nein das… |
| 642 | Löschung | `aus` | `*(nicht da)*` | …in andere körperregionen [___] nein das zum… |
| 643 | Löschung | `nein` | `*(nicht da)*` | …andere körperregionen aus [___] das zum glück… |
| 644 | Löschung | `das` | `*(nicht da)*` | …körperregionen aus nein [___] zum glück nicht… |
| 645 | Löschung | `zum` | `*(nicht da)*` | …aus nein das [___] glück nicht okay… |
| 646 | Löschung | `glück` | `*(nicht da)*` | …nein das zum [___] nicht okay wie… |
| 647 | Löschung | `nicht` | `*(nicht da)*` | …das zum glück [___] okay wie sieht… |
| 648 | Löschung | `okay` | `*(nicht da)*` | …zum glück nicht [___] wie sieht es… |
| 649 | Löschung | `sieht` | `*(nicht da)*` | …nicht okay wie [___] es an der… |
| 650 | Löschung | `es` | `*(nicht da)*` | …okay wie sieht [___] an der hand… |
| 651 | Löschung | `an` | `*(nicht da)*` | …wie sieht es [___] der hand aus… |
| 652 | Löschung | `der` | `*(nicht da)*` | …sieht es an [___] hand aus am… |
| 653 | Löschung | `hand` | `*(nicht da)*` | …es an der [___] aus am daumen… |
| 654 | Löschung | `aus` | `*(nicht da)*` | …an der hand [___] am daumen strahlen… |
| 655 | Löschung | `am` | `*(nicht da)*` | …der hand aus [___] daumen strahlen die… |
| 656 | Löschung | `daumen` | `*(nicht da)*` | …hand aus am [___] strahlen die schmerzen… |
| 657 | Löschung | `strahlen` | `*(nicht da)*` | …aus am daumen [___] die schmerzen da… |
| 658 | Löschung | `die` | `*(nicht da)*` | …am daumen strahlen [___] schmerzen da irgendwie… |
| 659 | Löschung | `schmerzen` | `*(nicht da)*` | …daumen strahlen die [___] da irgendwie ins… |
| 660 | Löschung | `da` | `*(nicht da)*` | …strahlen die schmerzen [___] irgendwie ins handgelenk… |
| 661 | Löschung | `irgendwie` | `*(nicht da)*` | …die schmerzen da [___] ins handgelenk aus… |
| 662 | Löschung | `ins` | `*(nicht da)*` | …schmerzen da irgendwie [___] handgelenk aus oder… |
| 663 | Löschung | `handgelenk` | `*(nicht da)*` | …da irgendwie ins [___] aus oder in… |
| 664 | Löschung | `aus` | `*(nicht da)*` | …irgendwie ins handgelenk [___] oder in andere… |
| 665 | Löschung | `oder` | `*(nicht da)*` | …ins handgelenk aus [___] in andere finger… |
| 666 | Löschung | `in` | `*(nicht da)*` | …handgelenk aus oder [___] andere finger auch… |
| 667 | Löschung | `andere` | `*(nicht da)*` | …aus oder in [___] finger auch nicht… |
| 668 | Löschung | `finger` | `*(nicht da)*` | …oder in andere [___] auch nicht nein… |
| 669 | Löschung | `auch` | `*(nicht da)*` | …in andere finger [___] nicht nein okay… |
| 670 | Löschung | `nicht` | `*(nicht da)*` | …andere finger auch [___] nein okay sehr… |
| 671 | Löschung | `nein` | `*(nicht da)*` | …finger auch nicht [___] okay sehr sehr… |
| 672 | Löschung | `okay` | `*(nicht da)*` | …auch nicht nein [___] sehr sehr gut… |
| 673 | Löschung | `sehr` | `*(nicht da)*` | …nicht nein okay [___] sehr gut können… |
| 674 | Löschung | `sehr` | `*(nicht da)*` | …nein okay sehr [___] gut können sie… |
| 675 | Löschung | `gut` | `*(nicht da)*` | …okay sehr sehr [___] können sie sich… |
| 676 | Löschung | `können` | `*(nicht da)*` | …sehr sehr gut [___] sie sich an… |
| 677 | Löschung | `sie` | `*(nicht da)*` | …sehr gut können [___] sich an den… |
| 678 | Löschung | `sich` | `*(nicht da)*` | …gut können sie [___] an den unfall… |
| 679 | Löschung | `an` | `*(nicht da)*` | …können sie sich [___] den unfall erinnern… |
| 680 | Substitution | `den` | `lange` | …sie sich an [___] unfall erinnern frau… |
| 681 | Substitution | `unfall` | `je` | …sich an den [___] erinnern frau beckenwestfalen… |
| 682 | Substitution | `erinnern` | `nach` | …an den unfall [___] frau beckenwestfalen ich… |
| 683 | Substitution | `frau` | `wetter` | …den unfall erinnern [___] beckenwestfalen ich kann… |
| 684 | Substitution | `beckenwestfalen` | `meistens` | …unfall erinnern frau [___] ich kann mich… |
| 685 | Substitution | `ich` | `so` | …erinnern frau beckenwestfalen [___] kann mich gut… |
| 686 | Substitution | `kann` | `eine` | …frau beckenwestfalen ich [___] mich gut daran… |
| 687 | Substitution | `mich` | `stunde` | …beckenwestfalen ich kann [___] gut daran erinnern… |
| 688 | Löschung | `daran` | `*(nicht da)*` | …kann mich gut [___] erinnern ja ich… |
| 689 | Löschung | `erinnern` | `*(nicht da)*` | …mich gut daran [___] ja ich war… |
| 690 | Löschung | `ja` | `*(nicht da)*` | …gut daran erinnern [___] ich war am… |
| 691 | Löschung | `ich` | `*(nicht da)*` | …daran erinnern ja [___] war am anfang… |
| 692 | Löschung | `war` | `*(nicht da)*` | …erinnern ja ich [___] am anfang zwar… |
| 693 | Löschung | `am` | `*(nicht da)*` | …ja ich war [___] anfang zwar etwas… |
| 694 | Löschung | `anfang` | `*(nicht da)*` | …ich war am [___] zwar etwas benebelt… |
| 695 | Löschung | `zwar` | `*(nicht da)*` | …war am anfang [___] etwas benebelt und… |
| 696 | Löschung | `etwas` | `*(nicht da)*` | …am anfang zwar [___] benebelt und mir… |
| 697 | Löschung | `benebelt` | `*(nicht da)*` | …anfang zwar etwas [___] und mir war… |
| 698 | Löschung | `mir` | `*(nicht da)*` | …etwas benebelt und [___] war es ziemlich… |
| 699 | Löschung | `war` | `*(nicht da)*` | …benebelt und mir [___] es ziemlich schwindelig… |
| 700 | Löschung | `es` | `*(nicht da)*` | …und mir war [___] ziemlich schwindelig aber… |
| 701 | Löschung | `ziemlich` | `*(nicht da)*` | …mir war es [___] schwindelig aber ich… |
| 702 | Löschung | `schwindelig` | `*(nicht da)*` | …war es ziemlich [___] aber ich denke… |
| 703 | Löschung | `aber` | `*(nicht da)*` | …es ziemlich schwindelig [___] ich denke das… |
| 704 | Löschung | `ich` | `*(nicht da)*` | …ziemlich schwindelig aber [___] denke das lag… |
| 705 | Löschung | `denke` | `*(nicht da)*` | …schwindelig aber ich [___] das lag vielleicht… |
| 706 | Löschung | `das` | `*(nicht da)*` | …aber ich denke [___] lag vielleicht am… |
| 707 | Löschung | `lag` | `*(nicht da)*` | …ich denke das [___] vielleicht am schock… |
| 708 | Löschung | `vielleicht` | `*(nicht da)*` | …denke das lag [___] am schock im… |
| 709 | Löschung | `am` | `*(nicht da)*` | …das lag vielleicht [___] schock im ersten… |
| 710 | Löschung | `schock` | `*(nicht da)*` | …lag vielleicht am [___] im ersten moment… |
| 711 | Löschung | `im` | `*(nicht da)*` | …vielleicht am schock [___] ersten moment okay… |
| 712 | Löschung | `ersten` | `*(nicht da)*` | …am schock im [___] moment okay gibt… |
| 713 | Löschung | `moment` | `*(nicht da)*` | …schock im ersten [___] okay gibt es… |
| 714 | Löschung | `okay` | `*(nicht da)*` | …im ersten moment [___] gibt es sonst… |
| 715 | Löschung | `gibt` | `*(nicht da)*` | …ersten moment okay [___] es sonst etwas… |
| 716 | Löschung | `es` | `*(nicht da)*` | …moment okay gibt [___] sonst etwas was… |
| 717 | Löschung | `sonst` | `*(nicht da)*` | …okay gibt es [___] etwas was ihnen… |
| 718 | Löschung | `etwas` | `*(nicht da)*` | …gibt es sonst [___] was ihnen aufgefallen… |
| 719 | Löschung | `was` | `*(nicht da)*` | …es sonst etwas [___] ihnen aufgefallen ist… |
| 720 | Löschung | `ihnen` | `*(nicht da)*` | …sonst etwas was [___] aufgefallen ist seit… |
| 721 | Substitution | `aufgefallen` | `wie` | …etwas was ihnen [___] ist seit dem… |
| 722 | Löschung | `seit` | `*(nicht da)*` | …ihnen aufgefallen ist [___] dem unfall was… |
| 723 | Löschung | `dem` | `*(nicht da)*` | …aufgefallen ist seit [___] unfall was ich… |
| 724 | Substitution | `unfall` | `ihr` | …ist seit dem [___] was ich wissen… |
| 725 | Substitution | `was` | `beruf` | …seit dem unfall [___] ich wissen sollte… |
| 726 | Löschung | `wissen` | `*(nicht da)*` | …unfall was ich [___] sollte ist ihnen… |
| 727 | Löschung | `sollte` | `*(nicht da)*` | …was ich wissen [___] ist ihnen übel… |
| 728 | Löschung | `ist` | `*(nicht da)*` | …ich wissen sollte [___] ihnen übel geworden… |
| 729 | Löschung | `ihnen` | `*(nicht da)*` | …wissen sollte ist [___] übel geworden oder… |
| 730 | Löschung | `übel` | `*(nicht da)*` | …sollte ist ihnen [___] geworden oder vielleicht… |
| 731 | Löschung | `geworden` | `*(nicht da)*` | …ist ihnen übel [___] oder vielleicht doch… |
| 732 | Löschung | `oder` | `*(nicht da)*` | …ihnen übel geworden [___] vielleicht doch noch… |
| 733 | Löschung | `vielleicht` | `*(nicht da)*` | …übel geworden oder [___] doch noch mal… |
| 734 | Löschung | `doch` | `*(nicht da)*` | …geworden oder vielleicht [___] noch mal schwarz… |
| 735 | Löschung | `noch` | `*(nicht da)*` | …oder vielleicht doch [___] mal schwarz vor… |
| 736 | Löschung | `mal` | `*(nicht da)*` | …vielleicht doch noch [___] schwarz vor augen… |
| 737 | Löschung | `schwarz` | `*(nicht da)*` | …doch noch mal [___] vor augen oder… |
| 738 | Löschung | `vor` | `*(nicht da)*` | …noch mal schwarz [___] augen oder fühlen… |
| 739 | Löschung | `augen` | `*(nicht da)*` | …mal schwarz vor [___] oder fühlen sie… |
| 740 | Löschung | `oder` | `*(nicht da)*` | …schwarz vor augen [___] fühlen sie sich… |
| 741 | Löschung | `fühlen` | `*(nicht da)*` | …vor augen oder [___] sie sich seltsam… |
| 742 | Löschung | `sie` | `*(nicht da)*` | …augen oder fühlen [___] sich seltsam seitdem… |
| 743 | Löschung | `sich` | `*(nicht da)*` | …oder fühlen sie [___] seltsam seitdem nein… |
| 744 | Löschung | `seltsam` | `*(nicht da)*` | …fühlen sie sich [___] seitdem nein außer… |
| 745 | Löschung | `seitdem` | `*(nicht da)*` | …sie sich seltsam [___] nein außer dass… |
| 746 | Substitution | `nein` | `arbeite` | …sich seltsam seitdem [___] außer dass ich… |
| 747 | Substitution | `außer` | `im` | …seltsam seitdem nein [___] dass ich sehr… |
| 748 | Substitution | `dass` | `büro` | …seitdem nein außer [___] ich sehr starke… |
| 749 | Löschung | `sehr` | `*(nicht da)*` | …außer dass ich [___] starke schmerzen habe… |
| 750 | Löschung | `starke` | `*(nicht da)*` | …dass ich sehr [___] schmerzen habe ist… |
| 751 | Löschung | `schmerzen` | `*(nicht da)*` | …ich sehr starke [___] habe ist mir… |
| 752 | Löschung | `habe` | `*(nicht da)*` | …sehr starke schmerzen [___] ist mir nichts… |
| 753 | Löschung | `ist` | `*(nicht da)*` | …starke schmerzen habe [___] mir nichts anderes… |
| 754 | Substitution | `mir` | `sitze` | …schmerzen habe ist [___] nichts anderes aufgefallen… |
| 755 | Substitution | `nichts` | `den` | …habe ist mir [___] anderes aufgefallen und… |
| 756 | Substitution | `anderes` | `ganzen` | …ist mir nichts [___] aufgefallen und dass… |
| 757 | Substitution | `aufgefallen` | `tag` | …mir nichts anderes [___] und dass ich… |
| 758 | Löschung | `dass` | `*(nicht da)*` | …anderes aufgefallen und [___] ich wie gesagt… |
| 759 | Löschung | `ich` | `*(nicht da)*` | …aufgefallen und dass [___] wie gesagt am… |
| 760 | Löschung | `gesagt` | `*(nicht da)*` | …dass ich wie [___] am anfang nur… |
| 761 | Löschung | `am` | `*(nicht da)*` | …ich wie gesagt [___] anfang nur etwas… |
| 762 | Löschung | `anfang` | `*(nicht da)*` | …wie gesagt am [___] nur etwas benebelt… |
| 763 | Löschung | `nur` | `*(nicht da)*` | …gesagt am anfang [___] etwas benebelt war… |
| 764 | Löschung | `etwas` | `*(nicht da)*` | …am anfang nur [___] benebelt war aber… |
| 765 | Löschung | `benebelt` | `*(nicht da)*` | …anfang nur etwas [___] war aber jetzt… |
| 766 | Löschung | `war` | `*(nicht da)*` | …nur etwas benebelt [___] aber jetzt bin… |
| 767 | Löschung | `aber` | `*(nicht da)*` | …etwas benebelt war [___] jetzt bin ich… |
| 768 | Löschung | `jetzt` | `*(nicht da)*` | …benebelt war aber [___] bin ich ganz… |
| 769 | Löschung | `bin` | `*(nicht da)*` | …war aber jetzt [___] ich ganz klar… |
| 770 | Löschung | `ich` | `*(nicht da)*` | …aber jetzt bin [___] ganz klar okay… |
| 771 | Löschung | `ganz` | `*(nicht da)*` | …jetzt bin ich [___] klar okay gut… |
| 772 | Löschung | `klar` | `*(nicht da)*` | …bin ich ganz [___] okay gut sehr… |
| 773 | Löschung | `okay` | `*(nicht da)*` | …ich ganz klar [___] gut sehr sehr… |
| 774 | Löschung | `gut` | `*(nicht da)*` | …ganz klar okay [___] sehr sehr gut… |
| 775 | Löschung | `sehr` | `*(nicht da)*` | …klar okay gut [___] sehr gut frau… |
| 776 | Löschung | `sehr` | `*(nicht da)*` | …okay gut sehr [___] gut frau beckenwestfalen… |
| 777 | Löschung | `gut` | `*(nicht da)*` | …gut sehr sehr [___] frau beckenwestfalen haben… |
| 778 | Löschung | `frau` | `*(nicht da)*` | …sehr sehr gut [___] beckenwestfalen haben sie… |
| 779 | Substitution | `beckenwestfalen` | `lange` | …sehr gut frau [___] haben sie irgendwelche… |
| 780 | Substitution | `haben` | `arbeiten` | …gut frau beckenwestfalen [___] sie irgendwelche vorerkrankungen… |
| 781 | Löschung | `irgendwelche` | `*(nicht da)*` | …beckenwestfalen haben sie [___] vorerkrankungen von denen… |
| 782 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …haben sie irgendwelche [___] von denen ich… |
| 783 | Löschung | `von` | `*(nicht da)*` | …sie irgendwelche vorerkrankungen [___] denen ich wissen… |
| 784 | Löschung | `denen` | `*(nicht da)*` | …irgendwelche vorerkrankungen von [___] ich wissen sollte… |
| 785 | Löschung | `ich` | `*(nicht da)*` | …vorerkrankungen von denen [___] wissen sollte wie… |
| 786 | Löschung | `wissen` | `*(nicht da)*` | …von denen ich [___] sollte wie zum… |
| 787 | Löschung | `sollte` | `*(nicht da)*` | …denen ich wissen [___] wie zum beispiel… |
| 788 | Löschung | `wie` | `*(nicht da)*` | …ich wissen sollte [___] zum beispiel erhöhten… |
| 789 | Löschung | `zum` | `*(nicht da)*` | …wissen sollte wie [___] beispiel erhöhten blutdruck… |
| 790 | Löschung | `beispiel` | `*(nicht da)*` | …sollte wie zum [___] erhöhten blutdruck oder… |
| 791 | Löschung | `erhöhten` | `*(nicht da)*` | …wie zum beispiel [___] blutdruck oder diabetes… |
| 792 | Löschung | `blutdruck` | `*(nicht da)*` | …zum beispiel erhöhten [___] oder diabetes oder… |
| 793 | Löschung | `oder` | `*(nicht da)*` | …beispiel erhöhten blutdruck [___] diabetes oder etwas… |
| 794 | Löschung | `diabetes` | `*(nicht da)*` | …erhöhten blutdruck oder [___] oder etwas anderes… |
| 795 | Löschung | `oder` | `*(nicht da)*` | …blutdruck oder diabetes [___] etwas anderes nichts… |
| 796 | Löschung | `etwas` | `*(nicht da)*` | …oder diabetes oder [___] anderes nichts ernsthaftes… |
| 797 | Löschung | `anderes` | `*(nicht da)*` | …diabetes oder etwas [___] nichts ernsthaftes ich… |
| 798 | Löschung | `nichts` | `*(nicht da)*` | …oder etwas anderes [___] ernsthaftes ich hatte… |
| 799 | Substitution | `ernsthaftes` | `schon` | …etwas anderes nichts [___] ich hatte eine… |
| 800 | Substitution | `ich` | `in` | …anderes nichts ernsthaftes [___] hatte eine laktoseintoleranz… |
| 801 | Substitution | `hatte` | `diesem` | …nichts ernsthaftes ich [___] eine laktoseintoleranz vor… |
| 802 | Substitution | `eine` | `beruf` | …ernsthaftes ich hatte [___] laktoseintoleranz vor einigen… |
| 803 | Substitution | `laktoseintoleranz` | `seit` | …ich hatte eine [___] vor einigen jahren… |
| 804 | Substitution | `vor` | `ungefähr` | …hatte eine laktoseintoleranz [___] einigen jahren sie… |
| 805 | Substitution | `einigen` | `zehn` | …eine laktoseintoleranz vor [___] jahren sie ist… |
| 806 | Löschung | `sie` | `*(nicht da)*` | …vor einigen jahren [___] ist allerdings schon… |
| 807 | Löschung | `ist` | `*(nicht da)*` | …einigen jahren sie [___] allerdings schon weg… |
| 808 | Löschung | `allerdings` | `*(nicht da)*` | …jahren sie ist [___] schon weg ja… |
| 809 | Löschung | `schon` | `*(nicht da)*` | …sie ist allerdings [___] weg ja und… |
| 810 | Löschung | `weg` | `*(nicht da)*` | …ist allerdings schon [___] ja und jetzt… |
| 811 | Substitution | `ja` | `gut` | …allerdings schon weg [___] und jetzt wurde… |
| 812 | Löschung | `jetzt` | `*(nicht da)*` | …weg ja und [___] wurde bei mir… |
| 813 | Löschung | `wurde` | `*(nicht da)*` | …ja und jetzt [___] bei mir vor… |
| 814 | Löschung | `bei` | `*(nicht da)*` | …und jetzt wurde [___] mir vor drei… |
| 815 | Löschung | `mir` | `*(nicht da)*` | …jetzt wurde bei [___] vor drei wochen… |
| 816 | Löschung | `vor` | `*(nicht da)*` | …wurde bei mir [___] drei wochen eine… |
| 817 | Löschung | `drei` | `*(nicht da)*` | …bei mir vor [___] wochen eine histaminunverträglichkeit… |
| 818 | Löschung | `wochen` | `*(nicht da)*` | …mir vor drei [___] eine histaminunverträglichkeit festgestellt… |
| 819 | Löschung | `eine` | `*(nicht da)*` | …vor drei wochen [___] histaminunverträglichkeit festgestellt wie… |
| 820 | Löschung | `histaminunverträglichkeit` | `*(nicht da)*` | …drei wochen eine [___] festgestellt wie äußert… |
| 821 | Löschung | `festgestellt` | `*(nicht da)*` | …wochen eine histaminunverträglichkeit [___] wie äußert sich… |
| 822 | Löschung | `äußert` | `*(nicht da)*` | …histaminunverträglichkeit festgestellt wie [___] sich die unverträglichkeit… |
| 823 | Löschung | `sich` | `*(nicht da)*` | …festgestellt wie äußert [___] die unverträglichkeit wenn… |
| 824 | Löschung | `die` | `*(nicht da)*` | …wie äußert sich [___] unverträglichkeit wenn ich… |
| 825 | Löschung | `unverträglichkeit` | `*(nicht da)*` | …äußert sich die [___] wenn ich bestimmte… |
| 826 | Löschung | `wenn` | `*(nicht da)*` | …sich die unverträglichkeit [___] ich bestimmte sachen… |
| 827 | Löschung | `ich` | `*(nicht da)*` | …die unverträglichkeit wenn [___] bestimmte sachen esse… |
| 828 | Löschung | `bestimmte` | `*(nicht da)*` | …unverträglichkeit wenn ich [___] sachen esse oder… |
| 829 | Löschung | `sachen` | `*(nicht da)*` | …wenn ich bestimmte [___] esse oder trinke… |
| 830 | Löschung | `esse` | `*(nicht da)*` | …ich bestimmte sachen [___] oder trinke vor… |
| 831 | Löschung | `oder` | `*(nicht da)*` | …bestimmte sachen esse [___] trinke vor allem… |
| 832 | Löschung | `trinke` | `*(nicht da)*` | …sachen esse oder [___] vor allem in… |
| 833 | Löschung | `vor` | `*(nicht da)*` | …esse oder trinke [___] allem in kombination… |
| 834 | Löschung | `allem` | `*(nicht da)*` | …oder trinke vor [___] in kombination dann… |
| 835 | Löschung | `in` | `*(nicht da)*` | …trinke vor allem [___] kombination dann bekomme… |
| 836 | Substitution | `kombination` | `ist` | …vor allem in [___] dann bekomme ich… |
| 837 | Substitution | `dann` | `ihre` | …allem in kombination [___] bekomme ich starkebauchschmerzen… |
| 838 | Substitution | `bekomme` | `ernährung` | …in kombination dann [___] ich starkebauchschmerzen übelkeit… |
| 839 | Löschung | `starkebauchschmerzen` | `*(nicht da)*` | …dann bekomme ich [___] übelkeit manchmal und… |
| 840 | Löschung | `übelkeit` | `*(nicht da)*` | …bekomme ich starkebauchschmerzen [___] manchmal und manchmal… |
| 841 | Löschung | `manchmal` | `*(nicht da)*` | …ich starkebauchschmerzen übelkeit [___] und manchmal auch… |
| 842 | Löschung | `und` | `*(nicht da)*` | …starkebauchschmerzen übelkeit manchmal [___] manchmal auch einen… |
| 843 | Löschung | `manchmal` | `*(nicht da)*` | …übelkeit manchmal und [___] auch einen ausschlag… |
| 844 | Löschung | `auch` | `*(nicht da)*` | …manchmal und manchmal [___] einen ausschlag hier… |
| 845 | Löschung | `einen` | `*(nicht da)*` | …und manchmal auch [___] ausschlag hier im… |
| 846 | Löschung | `ausschlag` | `*(nicht da)*` | …manchmal auch einen [___] hier im dekolleté… |
| 847 | Löschung | `hier` | `*(nicht da)*` | …auch einen ausschlag [___] im dekolleté bereich… |
| 848 | Löschung | `im` | `*(nicht da)*` | …einen ausschlag hier [___] dekolleté bereich okay… |
| 849 | Löschung | `dekolleté` | `*(nicht da)*` | …ausschlag hier im [___] bereich okay sonst… |
| 850 | Löschung | `bereich` | `*(nicht da)*` | …hier im dekolleté [___] okay sonst gibt… |
| 851 | Substitution | `okay` | `versuche` | …im dekolleté bereich [___] sonst gibt es… |
| 852 | Substitution | `sonst` | `ausgewogen` | …dekolleté bereich okay [___] gibt es aber… |
| 853 | Substitution | `gibt` | `zu` | …bereich okay sonst [___] es aber keine… |
| 854 | Substitution | `es` | `essen` | …okay sonst gibt [___] aber keine vorerkrankungen… |
| 855 | Löschung | `keine` | `*(nicht da)*` | …gibt es aber [___] vorerkrankungen nein okay… |
| 856 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …es aber keine [___] nein okay sehr… |
| 857 | Löschung | `nein` | `*(nicht da)*` | …aber keine vorerkrankungen [___] okay sehr gut… |
| 858 | Löschung | `okay` | `*(nicht da)*` | …keine vorerkrankungen nein [___] sehr gut frau… |
| 859 | Löschung | `sehr` | `*(nicht da)*` | …vorerkrankungen nein okay [___] gut frau beckenwestfalen… |
| 860 | Löschung | `gut` | `*(nicht da)*` | …nein okay sehr [___] frau beckenwestfalen sind… |
| 861 | Löschung | `frau` | `*(nicht da)*` | …okay sehr gut [___] beckenwestfalen sind sie… |
| 862 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …sehr gut frau [___] sind sie schon… |
| 863 | Löschung | `sind` | `*(nicht da)*` | …gut frau beckenwestfalen [___] sie schon einmal… |
| 864 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen sind [___] schon einmal operiert… |
| 865 | Löschung | `schon` | `*(nicht da)*` | …beckenwestfalen sind sie [___] einmal operiert worden… |
| 866 | Löschung | `einmal` | `*(nicht da)*` | …sind sie schon [___] operiert worden ja… |
| 867 | Löschung | `operiert` | `*(nicht da)*` | …sie schon einmal [___] worden ja ich… |
| 868 | Substitution | `worden` | `manchmal` | …schon einmal operiert [___] ja ich wurde… |
| 869 | Substitution | `ja` | `esse` | …einmal operiert worden [___] ich wurde vor… |
| 870 | Löschung | `wurde` | `*(nicht da)*` | …worden ja ich [___] vor zwei jahren… |
| 871 | Löschung | `vor` | `*(nicht da)*` | …ja ich wurde [___] zwei jahren am… |
| 872 | Löschung | `zwei` | `*(nicht da)*` | …ich wurde vor [___] jahren am fuß… |
| 873 | Löschung | `jahren` | `*(nicht da)*` | …wurde vor zwei [___] am fuß operiert… |
| 874 | Löschung | `am` | `*(nicht da)*` | …vor zwei jahren [___] fuß operiert mir… |
| 875 | Löschung | `fuß` | `*(nicht da)*` | …zwei jahren am [___] operiert mir wurde… |
| 876 | Löschung | `operiert` | `*(nicht da)*` | …jahren am fuß [___] mir wurde ein… |
| 877 | Löschung | `mir` | `*(nicht da)*` | …am fuß operiert [___] wurde ein halux… |
| 878 | Löschung | `wurde` | `*(nicht da)*` | …fuß operiert mir [___] ein halux valgus… |
| 879 | Löschung | `ein` | `*(nicht da)*` | …operiert mir wurde [___] halux valgus entfernt… |
| 880 | Löschung | `halux` | `*(nicht da)*` | …mir wurde ein [___] valgus entfernt ein… |
| 881 | Löschung | `valgus` | `*(nicht da)*` | …wurde ein halux [___] entfernt ein halux… |
| 882 | Löschung | `entfernt` | `*(nicht da)*` | …ein halux valgus [___] ein halux valgus… |
| 883 | Löschung | `ein` | `*(nicht da)*` | …halux valgus entfernt [___] halux valgus und… |
| 884 | Löschung | `halux` | `*(nicht da)*` | …valgus entfernt ein [___] valgus und welcher… |
| 885 | Löschung | `valgus` | `*(nicht da)*` | …entfernt ein halux [___] und welcher fuß… |
| 886 | Substitution | `und` | `auch` | …ein halux valgus [___] welcher fuß war… |
| 887 | Substitution | `welcher` | `etwas` | …halux valgus und [___] fuß war das… |
| 888 | Substitution | `fuß` | `zu` | …valgus und welcher [___] war das der… |
| 889 | Substitution | `war` | `viel` | …und welcher fuß [___] das der rechte… |
| 890 | Löschung | `der` | `*(nicht da)*` | …fuß war das [___] rechte fuß der… |
| 891 | Löschung | `rechte` | `*(nicht da)*` | …war das der [___] fuß der rechte… |
| 892 | Löschung | `fuß` | `*(nicht da)*` | …das der rechte [___] der rechte fuß… |
| 893 | Löschung | `der` | `*(nicht da)*` | …der rechte fuß [___] rechte fuß sind… |
| 894 | Löschung | `rechte` | `*(nicht da)*` | …rechte fuß der [___] fuß sind irgendwelche… |
| 895 | Löschung | `fuß` | `*(nicht da)*` | …fuß der rechte [___] sind irgendwelche komplikationen… |
| 896 | Löschung | `sind` | `*(nicht da)*` | …der rechte fuß [___] irgendwelche komplikationen während… |
| 897 | Löschung | `irgendwelche` | `*(nicht da)*` | …rechte fuß sind [___] komplikationen während oder… |
| 898 | Löschung | `komplikationen` | `*(nicht da)*` | …fuß sind irgendwelche [___] während oder nach… |
| 899 | Löschung | `während` | `*(nicht da)*` | …sind irgendwelche komplikationen [___] oder nach der… |
| 900 | Löschung | `oder` | `*(nicht da)*` | …irgendwelche komplikationen während [___] nach der operation… |
| 901 | Löschung | `nach` | `*(nicht da)*` | …komplikationen während oder [___] der operation aufgetreten… |
| 902 | Löschung | `der` | `*(nicht da)*` | …während oder nach [___] operation aufgetreten nein… |
| 903 | Löschung | `operation` | `*(nicht da)*` | …oder nach der [___] aufgetreten nein zum… |
| 904 | Löschung | `aufgetreten` | `*(nicht da)*` | …nach der operation [___] nein zum glück… |
| 905 | Löschung | `nein` | `*(nicht da)*` | …der operation aufgetreten [___] zum glück nicht… |
| 906 | Löschung | `zum` | `*(nicht da)*` | …operation aufgetreten nein [___] glück nicht nein… |
| 907 | Löschung | `glück` | `*(nicht da)*` | …aufgetreten nein zum [___] nicht nein sehr… |
| 908 | Löschung | `nicht` | `*(nicht da)*` | …nein zum glück [___] nein sehr gut… |
| 909 | Löschung | `nein` | `*(nicht da)*` | …zum glück nicht [___] sehr gut ich… |
| 910 | Löschung | `sehr` | `*(nicht da)*` | …glück nicht nein [___] gut ich konnte… |
| 911 | Löschung | `gut` | `*(nicht da)*` | …nicht nein sehr [___] ich konnte ganz… |
| 912 | Löschung | `ich` | `*(nicht da)*` | …nein sehr gut [___] konnte ganz bald… |
| 913 | Löschung | `konnte` | `*(nicht da)*` | …sehr gut ich [___] ganz bald wieder… |
| 914 | Löschung | `ganz` | `*(nicht da)*` | …gut ich konnte [___] bald wieder meine… |
| 915 | Löschung | `bald` | `*(nicht da)*` | …ich konnte ganz [___] wieder meine hohen… |
| 916 | Löschung | `wieder` | `*(nicht da)*` | …konnte ganz bald [___] meine hohen schuhe… |
| 917 | Löschung | `meine` | `*(nicht da)*` | …ganz bald wieder [___] hohen schuhe tragen… |
| 918 | Löschung | `hohen` | `*(nicht da)*` | …bald wieder meine [___] schuhe tragen perfekt… |
| 919 | Löschung | `schuhe` | `*(nicht da)*` | …wieder meine hohen [___] tragen perfekt dann… |
| 920 | Löschung | `tragen` | `*(nicht da)*` | …meine hohen schuhe [___] perfekt dann ist… |
| 921 | Löschung | `perfekt` | `*(nicht da)*` | …hohen schuhe tragen [___] dann ist wirklich… |
| 922 | Löschung | `dann` | `*(nicht da)*` | …schuhe tragen perfekt [___] ist wirklich alles… |
| 923 | Löschung | `wirklich` | `*(nicht da)*` | …perfekt dann ist [___] alles gut gelaufen… |
| 924 | Löschung | `alles` | `*(nicht da)*` | …dann ist wirklich [___] gut gelaufen frau… |
| 925 | Löschung | `gut` | `*(nicht da)*` | …ist wirklich alles [___] gelaufen frau beckenwestfalen… |
| 926 | Löschung | `gelaufen` | `*(nicht da)*` | …wirklich alles gut [___] frau beckenwestfalen nehmen… |
| 927 | Löschung | `frau` | `*(nicht da)*` | …alles gut gelaufen [___] beckenwestfalen nehmen sie… |
| 928 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …gut gelaufen frau [___] nehmen sie regelmäßig… |
| 929 | Löschung | `nehmen` | `*(nicht da)*` | …gelaufen frau beckenwestfalen [___] sie regelmäßig oder… |
| 930 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen nehmen [___] regelmäßig oder bei… |
| 931 | Löschung | `regelmäßig` | `*(nicht da)*` | …beckenwestfalen nehmen sie [___] oder bei bedarf… |
| 932 | Löschung | `oder` | `*(nicht da)*` | …nehmen sie regelmäßig [___] bei bedarf medikamente… |
| 933 | Löschung | `bei` | `*(nicht da)*` | …sie regelmäßig oder [___] bedarf medikamente ein… |
| 934 | Löschung | `bedarf` | `*(nicht da)*` | …regelmäßig oder bei [___] medikamente ein ich… |
| 935 | Löschung | `medikamente` | `*(nicht da)*` | …oder bei bedarf [___] ein ich nehme… |
| 936 | Löschung | `ein` | `*(nicht da)*` | …bei bedarf medikamente [___] ich nehme gelegentlich… |
| 937 | Löschung | `ich` | `*(nicht da)*` | …bedarf medikamente ein [___] nehme gelegentlich ein… |
| 938 | Löschung | `nehme` | `*(nicht da)*` | …medikamente ein ich [___] gelegentlich ein ibuprofen… |
| 939 | Löschung | `gelegentlich` | `*(nicht da)*` | …ein ich nehme [___] ein ibuprofen wenn… |
| 940 | Löschung | `ein` | `*(nicht da)*` | …ich nehme gelegentlich [___] ibuprofen wenn ich… |
| 941 | Löschung | `ibuprofen` | `*(nicht da)*` | …nehme gelegentlich ein [___] wenn ich kopfschmerzen… |
| 942 | Löschung | `wenn` | `*(nicht da)*` | …gelegentlich ein ibuprofen [___] ich kopfschmerzen habe… |
| 943 | Löschung | `ich` | `*(nicht da)*` | …ein ibuprofen wenn [___] kopfschmerzen habe und… |
| 944 | Löschung | `kopfschmerzen` | `*(nicht da)*` | …ibuprofen wenn ich [___] habe und ansonsten… |
| 945 | Substitution | `habe` | `normal` | …wenn ich kopfschmerzen [___] und ansonsten nehme… |
| 946 | Löschung | `ansonsten` | `*(nicht da)*` | …kopfschmerzen habe und [___] nehme ich die… |
| 947 | Löschung | `nehme` | `*(nicht da)*` | …habe und ansonsten [___] ich die pille… |
| 948 | Löschung | `ich` | `*(nicht da)*` | …und ansonsten nehme [___] die pille die… |
| 949 | Löschung | `die` | `*(nicht da)*` | …ansonsten nehme ich [___] pille die pille… |
| 950 | Löschung | `pille` | `*(nicht da)*` | …nehme ich die [___] die pille seit… |
| 951 | Löschung | `die` | `*(nicht da)*` | …ich die pille [___] pille seit wann… |
| 952 | Löschung | `pille` | `*(nicht da)*` | …die pille die [___] seit wann nehmen… |
| 953 | Löschung | `seit` | `*(nicht da)*` | …pille die pille [___] wann nehmen sie… |
| 954 | Löschung | `wann` | `*(nicht da)*` | …die pille seit [___] nehmen sie die… |
| 955 | Löschung | `nehmen` | `*(nicht da)*` | …pille seit wann [___] sie die pille… |
| 956 | Löschung | `sie` | `*(nicht da)*` | …seit wann nehmen [___] die pille sieben… |
| 957 | Löschung | `die` | `*(nicht da)*` | …wann nehmen sie [___] pille sieben oder… |
| 958 | Löschung | `pille` | `*(nicht da)*` | …nehmen sie die [___] sieben oder acht… |
| 959 | Löschung | `sieben` | `*(nicht da)*` | …sie die pille [___] oder acht jahren… |
| 960 | Löschung | `oder` | `*(nicht da)*` | …die pille sieben [___] acht jahren okay… |
| 961 | Löschung | `acht` | `*(nicht da)*` | …pille sieben oder [___] jahren okay die… |
| 962 | Löschung | `jahren` | `*(nicht da)*` | …sieben oder acht [___] okay die ibuprofen… |
| 963 | Löschung | `okay` | `*(nicht da)*` | …oder acht jahren [___] die ibuprofen wenn… |
| 964 | Löschung | `die` | `*(nicht da)*` | …acht jahren okay [___] ibuprofen wenn sie… |
| 965 | Löschung | `ibuprofen` | `*(nicht da)*` | …jahren okay die [___] wenn sie kopfschmerzen… |
| 966 | Löschung | `wenn` | `*(nicht da)*` | …okay die ibuprofen [___] sie kopfschmerzen haben… |
| 967 | Löschung | `sie` | `*(nicht da)*` | …die ibuprofen wenn [___] kopfschmerzen haben wie… |
| 968 | Löschung | `kopfschmerzen` | `*(nicht da)*` | …ibuprofen wenn sie [___] haben wie viele… |
| 969 | Löschung | `haben` | `*(nicht da)*` | …wenn sie kopfschmerzen [___] wie viele milligramm… |
| 970 | Löschung | `viele` | `*(nicht da)*` | …kopfschmerzen haben wie [___] milligramm sind das… |
| 971 | Löschung | `milligramm` | `*(nicht da)*` | …haben wie viele [___] sind das 400… |
| 972 | Löschung | `sind` | `*(nicht da)*` | …wie viele milligramm [___] das 400 600… |
| 973 | Löschung | `das` | `*(nicht da)*` | …viele milligramm sind [___] 400 600 800… |
| 974 | Löschung | `400` | `*(nicht da)*` | …milligramm sind das [___] 600 800 also… |
| 975 | Substitution | `600` | `ist` | …sind das 400 [___] 800 also meistens… |
| 976 | Substitution | `800` | `ihr` | …das 400 600 [___] also meistens das… |
| 977 | Substitution | `also` | `schlaf` | …400 600 800 [___] meistens das was… |
| 978 | Löschung | `das` | `*(nicht da)*` | …800 also meistens [___] was ich gerade… |
| 979 | Substitution | `was` | `gut` | …also meistens das [___] ich gerade zu… |
| 980 | Löschung | `gerade` | `*(nicht da)*` | …das was ich [___] zu hause habe… |
| 981 | Substitution | `zu` | `schlafe` | …was ich gerade [___] hause habe aber… |
| 982 | Substitution | `hause` | `sieben` | …ich gerade zu [___] habe aber ich… |
| 983 | Substitution | `habe` | `bis` | …gerade zu hause [___] aber ich glaube… |
| 984 | Substitution | `aber` | `acht` | …zu hause habe [___] ich glaube 600… |
| 985 | Substitution | `ich` | `stunden` | …hause habe aber [___] glaube 600 600… |
| 986 | Substitution | `glaube` | `gut` | …habe aber ich [___] 600 600 ja… |
| 987 | Substitution | `600` | `und` | …aber ich glaube [___] 600 ja sind… |
| 988 | Substitution | `600` | `wie` | …ich glaube 600 [___] ja sind sie… |
| 989 | Substitution | `ja` | `ist` | …glaube 600 600 [___] sind sie geimpft… |
| 990 | Substitution | `sind` | `ihre` | …600 600 ja [___] sie geimpft ich… |
| 991 | Substitution | `sie` | `stimmung` | …600 ja sind [___] geimpft ich bin… |
| 992 | Substitution | `geimpft` | `allgemein` | …ja sind sie [___] ich bin geimpft… |
| 993 | Löschung | `geimpft` | `*(nicht da)*` | …geimpft ich bin [___] ja haben sie… |
| 994 | Löschung | `ja` | `*(nicht da)*` | …ich bin geimpft [___] haben sie ganz… |
| 995 | Löschung | `haben` | `*(nicht da)*` | …bin geimpft ja [___] sie ganz zufällig… |
| 996 | Löschung | `sie` | `*(nicht da)*` | …geimpft ja haben [___] ganz zufällig ihren… |
| 997 | Löschung | `ganz` | `*(nicht da)*` | …ja haben sie [___] zufällig ihren impfpass… |
| 998 | Löschung | `zufällig` | `*(nicht da)*` | …haben sie ganz [___] ihren impfpass dabei… |
| 999 | Löschung | `ihren` | `*(nicht da)*` | …sie ganz zufällig [___] impfpass dabei oh… |
| 1000 | Löschung | `impfpass` | `*(nicht da)*` | …ganz zufällig ihren [___] dabei oh leider… |
| 1001 | Löschung | `dabei` | `*(nicht da)*` | …zufällig ihren impfpass [___] oh leider nein… |
| 1002 | Löschung | `oh` | `*(nicht da)*` | …ihren impfpass dabei [___] leider nein eher… |
| 1003 | Löschung | `leider` | `*(nicht da)*` | …impfpass dabei oh [___] nein eher nicht… |
| 1004 | Löschung | `nein` | `*(nicht da)*` | …dabei oh leider [___] eher nicht hätte… |
| 1005 | Löschung | `eher` | `*(nicht da)*` | …oh leider nein [___] nicht hätte ich… |
| 1006 | Löschung | `nicht` | `*(nicht da)*` | …leider nein eher [___] hätte ich gewusst… |
| 1007 | Löschung | `hätte` | `*(nicht da)*` | …nein eher nicht [___] ich gewusst dass… |
| 1008 | Löschung | `ich` | `*(nicht da)*` | …eher nicht hätte [___] gewusst dass ich… |
| 1009 | Löschung | `gewusst` | `*(nicht da)*` | …nicht hätte ich [___] dass ich ins… |
| 1010 | Löschung | `dass` | `*(nicht da)*` | …hätte ich gewusst [___] ich ins krankenhaus… |
| 1011 | Löschung | `ich` | `*(nicht da)*` | …ich gewusst dass [___] ins krankenhaus muss… |
| 1012 | Löschung | `ins` | `*(nicht da)*` | …gewusst dass ich [___] krankenhaus muss ja… |
| 1013 | Löschung | `krankenhaus` | `*(nicht da)*` | …dass ich ins [___] muss ja ich… |
| 1014 | Löschung | `muss` | `*(nicht da)*` | …ich ins krankenhaus [___] ja ich brauche… |
| 1015 | Löschung | `ja` | `*(nicht da)*` | …ins krankenhaus muss [___] ich brauche den… |
| 1016 | Löschung | `ich` | `*(nicht da)*` | …krankenhaus muss ja [___] brauche den den… |
| 1017 | Löschung | `brauche` | `*(nicht da)*` | …muss ja ich [___] den den ich… |
| 1018 | Löschung | `den` | `*(nicht da)*` | …ja ich brauche [___] den ich nicht… |
| 1019 | Löschung | `den` | `*(nicht da)*` | …ich brauche den [___] ich nicht bei… |
| 1020 | Löschung | `ich` | `*(nicht da)*` | …brauche den den [___] nicht bei mir… |
| 1021 | Löschung | `nicht` | `*(nicht da)*` | …den den ich [___] bei mir trage… |
| 1022 | Löschung | `bei` | `*(nicht da)*` | …den ich nicht [___] mir trage ja… |
| 1023 | Löschung | `mir` | `*(nicht da)*` | …ich nicht bei [___] trage ja sehr… |
| 1024 | Löschung | `trage` | `*(nicht da)*` | …nicht bei mir [___] ja sehr gut… |
| 1025 | Löschung | `ja` | `*(nicht da)*` | …bei mir trage [___] sehr gut okay… |
| 1026 | Substitution | `sehr` | `zufrieden` | …mir trage ja [___] gut okay frau… |
| 1027 | Löschung | `okay` | `*(nicht da)*` | …ja sehr gut [___] frau beckenwestfalen wie… |
| 1028 | Löschung | `frau` | `*(nicht da)*` | …sehr gut okay [___] beckenwestfalen wie geht… |
| 1029 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …gut okay frau [___] wie geht es… |
| 1030 | Substitution | `wie` | `und` | …okay frau beckenwestfalen [___] geht es ihnen… |
| 1031 | Substitution | `geht` | `gibt` | …frau beckenwestfalen wie [___] es ihnen denn… |
| 1032 | Löschung | `ihnen` | `*(nicht da)*` | …wie geht es [___] denn sonst körperlich… |
| 1033 | Löschung | `denn` | `*(nicht da)*` | …geht es ihnen [___] sonst körperlich haben… |
| 1034 | Substitution | `sonst` | `irgendwelche` | …es ihnen denn [___] körperlich haben sie… |
| 1035 | Substitution | `körperlich` | `dinge` | …ihnen denn sonst [___] haben sie in… |
| 1036 | Substitution | `haben` | `die` | …denn sonst körperlich [___] sie in letzter… |
| 1037 | Löschung | `fieber` | `*(nicht da)*` | …in letzter zeit [___] gehabt oder schüttelfrost… |
| 1038 | Löschung | `gehabt` | `*(nicht da)*` | …letzter zeit fieber [___] oder schüttelfrost oder… |
| 1039 | Löschung | `oder` | `*(nicht da)*` | …zeit fieber gehabt [___] schüttelfrost oder nachtschweiß… |
| 1040 | Löschung | `schüttelfrost` | `*(nicht da)*` | …fieber gehabt oder [___] oder nachtschweiß oder… |
| 1041 | Löschung | `oder` | `*(nicht da)*` | …gehabt oder schüttelfrost [___] nachtschweiß oder fühlen… |
| 1042 | Löschung | `nachtschweiß` | `*(nicht da)*` | …oder schüttelfrost oder [___] oder fühlen sie… |
| 1043 | Löschung | `oder` | `*(nicht da)*` | …schüttelfrost oder nachtschweiß [___] fühlen sie sich… |
| 1044 | Löschung | `fühlen` | `*(nicht da)*` | …oder nachtschweiß oder [___] sie sich irgendwie… |
| 1045 | Löschung | `sie` | `*(nicht da)*` | …nachtschweiß oder fühlen [___] sich irgendwie ungut… |
| 1046 | Löschung | `sich` | `*(nicht da)*` | …oder fühlen sie [___] irgendwie ungut in… |
| 1047 | Löschung | `irgendwie` | `*(nicht da)*` | …fühlen sie sich [___] ungut in letzter… |
| 1048 | Löschung | `ungut` | `*(nicht da)*` | …sie sich irgendwie [___] in letzter zeit… |
| 1049 | Substitution | `in` | `anders` | …sich irgendwie ungut [___] letzter zeit nein… |
| 1050 | Substitution | `letzter` | `gemacht` | …irgendwie ungut in [___] zeit nein ich… |
| 1051 | Substitution | `zeit` | `haben` | …ungut in letzter [___] nein ich habe… |
| 1052 | Löschung | `ich` | `*(nicht da)*` | …letzter zeit nein [___] habe gar keine… |
| 1053 | Löschung | `habe` | `*(nicht da)*` | …zeit nein ich [___] gar keine sonstigen… |
| 1054 | Löschung | `gar` | `*(nicht da)*` | …nein ich habe [___] keine sonstigen gesundheitlichen… |
| 1055 | Löschung | `keine` | `*(nicht da)*` | …ich habe gar [___] sonstigen gesundheitlichen probleme… |
| 1056 | Löschung | `sonstigen` | `*(nicht da)*` | …habe gar keine [___] gesundheitlichen probleme ich… |
| 1057 | Löschung | `gesundheitlichen` | `*(nicht da)*` | …gar keine sonstigen [___] probleme ich habe… |
| 1058 | Löschung | `probleme` | `*(nicht da)*` | …keine sonstigen gesundheitlichen [___] ich habe manchmal… |
| 1059 | Löschung | `ich` | `*(nicht da)*` | …sonstigen gesundheitlichen probleme [___] habe manchmal schwierigkeiten… |
| 1060 | Löschung | `habe` | `*(nicht da)*` | …gesundheitlichen probleme ich [___] manchmal schwierigkeiten beim… |
| 1061 | Löschung | `manchmal` | `*(nicht da)*` | …probleme ich habe [___] schwierigkeiten beim einschlafen… |
| 1062 | Löschung | `schwierigkeiten` | `*(nicht da)*` | …ich habe manchmal [___] beim einschlafen aber… |
| 1063 | Löschung | `beim` | `*(nicht da)*` | …habe manchmal schwierigkeiten [___] einschlafen aber das… |
| 1064 | Löschung | `einschlafen` | `*(nicht da)*` | …manchmal schwierigkeiten beim [___] aber das ist… |
| 1065 | Löschung | `aber` | `*(nicht da)*` | …schwierigkeiten beim einschlafen [___] das ist oft… |
| 1066 | Löschung | `das` | `*(nicht da)*` | …beim einschlafen aber [___] ist oft der… |
| 1067 | Löschung | `ist` | `*(nicht da)*` | …einschlafen aber das [___] oft der fall… |
| 1068 | Löschung | `oft` | `*(nicht da)*` | …aber das ist [___] der fall wenn… |
| 1069 | Löschung | `der` | `*(nicht da)*` | …das ist oft [___] fall wenn ich… |
| 1070 | Löschung | `fall` | `*(nicht da)*` | …ist oft der [___] wenn ich auf… |
| 1071 | Löschung | `wenn` | `*(nicht da)*` | …oft der fall [___] ich auf der… |
| 1072 | Löschung | `ich` | `*(nicht da)*` | …der fall wenn [___] auf der arbeit… |
| 1073 | Löschung | `auf` | `*(nicht da)*` | …fall wenn ich [___] der arbeit viel… |
| 1074 | Löschung | `der` | `*(nicht da)*` | …wenn ich auf [___] arbeit viel zu… |
| 1075 | Löschung | `arbeit` | `*(nicht da)*` | …ich auf der [___] viel zu tun… |
| 1076 | Löschung | `viel` | `*(nicht da)*` | …auf der arbeit [___] zu tun habe… |
| 1077 | Löschung | `zu` | `*(nicht da)*` | …der arbeit viel [___] tun habe oder… |
| 1078 | Löschung | `tun` | `*(nicht da)*` | …arbeit viel zu [___] habe oder zu… |
| 1079 | Löschung | `habe` | `*(nicht da)*` | …viel zu tun [___] oder zu viel… |
| 1080 | Löschung | `oder` | `*(nicht da)*` | …zu tun habe [___] zu viel nachdenke… |
| 1081 | Löschung | `zu` | `*(nicht da)*` | …tun habe oder [___] viel nachdenke also… |
| 1082 | Löschung | `viel` | `*(nicht da)*` | …habe oder zu [___] nachdenke also nichts… |
| 1083 | Löschung | `nachdenke` | `*(nicht da)*` | …oder zu viel [___] also nichts worüber… |
| 1084 | Substitution | `also` | `eigentlich` | …zu viel nachdenke [___] nichts worüber ich… |
| 1085 | Löschung | `worüber` | `*(nicht da)*` | …nachdenke also nichts [___] ich mir bis… |
| 1086 | Löschung | `ich` | `*(nicht da)*` | …also nichts worüber [___] mir bis jetzt… |
| 1087 | Löschung | `mir` | `*(nicht da)*` | …nichts worüber ich [___] bis jetzt sorgen… |
| 1088 | Löschung | `bis` | `*(nicht da)*` | …worüber ich mir [___] jetzt sorgen gemacht… |
| 1089 | Löschung | `jetzt` | `*(nicht da)*` | …ich mir bis [___] sorgen gemacht habe… |
| 1090 | Löschung | `sorgen` | `*(nicht da)*` | …mir bis jetzt [___] gemacht habe okay… |
| 1091 | Löschung | `gemacht` | `*(nicht da)*` | …bis jetzt sorgen [___] habe okay prima… |
| 1092 | Substitution | `habe` | `besonderes` | …jetzt sorgen gemacht [___] okay prima ich… |
| 1093 | Löschung | `prima` | `*(nicht da)*` | …gemacht habe okay [___] ich glaube das… |
| 1094 | Löschung | `ich` | `*(nicht da)*` | …habe okay prima [___] glaube das kennen… |
| 1095 | Löschung | `glaube` | `*(nicht da)*` | …okay prima ich [___] das kennen wir… |
| 1096 | Löschung | `das` | `*(nicht da)*` | …prima ich glaube [___] kennen wir auch… |
| 1097 | Löschung | `kennen` | `*(nicht da)*` | …ich glaube das [___] wir auch wirklich… |
| 1098 | Löschung | `wir` | `*(nicht da)*` | …glaube das kennen [___] auch wirklich alle… |
| 1099 | Löschung | `auch` | `*(nicht da)*` | …das kennen wir [___] wirklich alle ja… |
| 1100 | Löschung | `wirklich` | `*(nicht da)*` | …kennen wir auch [___] alle ja wie… |
| 1101 | Löschung | `alle` | `*(nicht da)*` | …wir auch wirklich [___] ja wie sieht… |
| 1102 | Substitution | `ja` | `und` | …auch wirklich alle [___] wie sieht es… |
| 1103 | Löschung | `denn` | `*(nicht da)*` | …wie sieht es [___] aus mit ihrer… |
| 1104 | Löschung | `aus` | `*(nicht da)*` | …sieht es denn [___] mit ihrer periode… |
| 1105 | Löschung | `ihrer` | `*(nicht da)*` | …denn aus mit [___] periode bekommen sie… |
| 1106 | Löschung | `periode` | `*(nicht da)*` | …aus mit ihrer [___] bekommen sie die… |
| 1107 | Löschung | `bekommen` | `*(nicht da)*` | …mit ihrer periode [___] sie die regelmäßig… |
| 1108 | Löschung | `sie` | `*(nicht da)*` | …ihrer periode bekommen [___] die regelmäßig ich… |
| 1109 | Löschung | `die` | `*(nicht da)*` | …periode bekommen sie [___] regelmäßig ich bekomme… |
| 1110 | Löschung | `regelmäßig` | `*(nicht da)*` | …bekommen sie die [___] ich bekomme sie… |
| 1111 | Löschung | `ich` | `*(nicht da)*` | …sie die regelmäßig [___] bekomme sie regelmäßig… |
| 1112 | Löschung | `bekomme` | `*(nicht da)*` | …die regelmäßig ich [___] sie regelmäßig ja… |
| 1113 | Löschung | `sie` | `*(nicht da)*` | …regelmäßig ich bekomme [___] regelmäßig ja seitdem… |
| 1114 | Löschung | `regelmäßig` | `*(nicht da)*` | …ich bekomme sie [___] ja seitdem ich… |
| 1115 | Löschung | `ja` | `*(nicht da)*` | …bekomme sie regelmäßig [___] seitdem ich die… |
| 1116 | Löschung | `seitdem` | `*(nicht da)*` | …sie regelmäßig ja [___] ich die pille… |
| 1117 | Löschung | `ich` | `*(nicht da)*` | …regelmäßig ja seitdem [___] die pille nehme… |
| 1118 | Löschung | `die` | `*(nicht da)*` | …ja seitdem ich [___] pille nehme bekomme… |
| 1119 | Löschung | `pille` | `*(nicht da)*` | …seitdem ich die [___] nehme bekomme ich… |
| 1120 | Löschung | `nehme` | `*(nicht da)*` | …ich die pille [___] bekomme ich sie… |
| 1121 | Löschung | `bekomme` | `*(nicht da)*` | …die pille nehme [___] ich sie ganz… |
| 1122 | Löschung | `ich` | `*(nicht da)*` | …pille nehme bekomme [___] sie ganz regelmäßig… |
| 1123 | Löschung | `sie` | `*(nicht da)*` | …nehme bekomme ich [___] ganz regelmäßig okay… |
| 1124 | Löschung | `ganz` | `*(nicht da)*` | …bekomme ich sie [___] regelmäßig okay wunderbar… |
| 1125 | Löschung | `regelmäßig` | `*(nicht da)*` | …ich sie ganz [___] okay wunderbar frau… |
| 1126 | Löschung | `okay` | `*(nicht da)*` | …sie ganz regelmäßig [___] wunderbar frau beckenwestfalen… |
| 1127 | Löschung | `wunderbar` | `*(nicht da)*` | …ganz regelmäßig okay [___] frau beckenwestfalen rauchen… |
| 1128 | Löschung | `frau` | `*(nicht da)*` | …regelmäßig okay wunderbar [___] beckenwestfalen rauchen sie… |
| 1129 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …okay wunderbar frau [___] rauchen sie nein… |
| 1130 | Löschung | `rauchen` | `*(nicht da)*` | …wunderbar frau beckenwestfalen [___] sie nein ich… |
| 1131 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen rauchen [___] nein ich habe… |
| 1132 | Löschung | `nein` | `*(nicht da)*` | …beckenwestfalen rauchen sie [___] ich habe früher… |
| 1133 | Löschung | `ich` | `*(nicht da)*` | …rauchen sie nein [___] habe früher geraucht… |
| 1134 | Löschung | `habe` | `*(nicht da)*` | …sie nein ich [___] früher geraucht falls… |
| 1135 | Löschung | `früher` | `*(nicht da)*` | …nein ich habe [___] geraucht falls das… |
| 1136 | Löschung | `geraucht` | `*(nicht da)*` | …ich habe früher [___] falls das relevant… |
| 1137 | Löschung | `falls` | `*(nicht da)*` | …habe früher geraucht [___] das relevant ist… |
| 1138 | Löschung | `das` | `*(nicht da)*` | …früher geraucht falls [___] relevant ist ja… |
| 1139 | Löschung | `relevant` | `*(nicht da)*` | …geraucht falls das [___] ist ja aber… |
| 1140 | Löschung | `ist` | `*(nicht da)*` | …falls das relevant [___] ja aber haben… |
| 1141 | Löschung | `ja` | `*(nicht da)*` | …das relevant ist [___] aber haben sie… |
| 1142 | Löschung | `aber` | `*(nicht da)*` | …relevant ist ja [___] haben sie aufgehört… |
| 1143 | Löschung | `haben` | `*(nicht da)*` | …ist ja aber [___] sie aufgehört ach… |
| 1144 | Löschung | `sie` | `*(nicht da)*` | …ja aber haben [___] aufgehört ach das… |
| 1145 | Löschung | `aufgehört` | `*(nicht da)*` | …aber haben sie [___] ach das müssten… |
| 1146 | Löschung | `ach` | `*(nicht da)*` | …haben sie aufgehört [___] das müssten jetzt… |
| 1147 | Löschung | `das` | `*(nicht da)*` | …sie aufgehört ach [___] müssten jetzt schon… |
| 1148 | Löschung | `müssten` | `*(nicht da)*` | …aufgehört ach das [___] jetzt schon acht… |
| 1149 | Löschung | `jetzt` | `*(nicht da)*` | …ach das müssten [___] schon acht jahre… |
| 1150 | Löschung | `schon` | `*(nicht da)*` | …das müssten jetzt [___] acht jahre sein… |
| 1151 | Löschung | `acht` | `*(nicht da)*` | …müssten jetzt schon [___] jahre sein seitdem… |
| 1152 | Substitution | `jahre` | `ihrem` | …jetzt schon acht [___] sein seitdem ich… |
| 1153 | Substitution | `sein` | `familienstand` | …schon acht jahre [___] seitdem ich aufgehört… |
| 1154 | Substitution | `seitdem` | `aus` | …acht jahre sein [___] ich aufgehört habe… |
| 1155 | Löschung | `aufgehört` | `*(nicht da)*` | …sein seitdem ich [___] habe zum glück… |
| 1156 | Löschung | `habe` | `*(nicht da)*` | …seitdem ich aufgehört [___] zum glück und… |
| 1157 | Substitution | `zum` | `bin` | …ich aufgehört habe [___] glück und wie… |
| 1158 | Substitution | `glück` | `verheiratet` | …aufgehört habe zum [___] und wie lange… |
| 1159 | Löschung | `wie` | `*(nicht da)*` | …zum glück und [___] lange haben sie… |
| 1160 | Substitution | `lange` | `wir` | …glück und wie [___] haben sie geraucht… |
| 1161 | Löschung | `sie` | `*(nicht da)*` | …wie lange haben [___] geraucht damals sechs… |
| 1162 | Löschung | `geraucht` | `*(nicht da)*` | …lange haben sie [___] damals sechs sieben… |
| 1163 | Löschung | `damals` | `*(nicht da)*` | …haben sie geraucht [___] sechs sieben jahre… |
| 1164 | Löschung | `sechs` | `*(nicht da)*` | …sie geraucht damals [___] sieben jahre sechs… |
| 1165 | Löschung | `sieben` | `*(nicht da)*` | …geraucht damals sechs [___] jahre sechs sieben… |
| 1166 | Löschung | `jahre` | `*(nicht da)*` | …damals sechs sieben [___] sechs sieben jahre… |
| 1167 | Löschung | `sechs` | `*(nicht da)*` | …sechs sieben jahre [___] sieben jahre okay… |
| 1168 | Löschung | `sieben` | `*(nicht da)*` | …sieben jahre sechs [___] jahre okay gut… |
| 1169 | Substitution | `jahre` | `zwei` | …jahre sechs sieben [___] okay gut trinken… |
| 1170 | Substitution | `okay` | `kinder` | …sechs sieben jahre [___] gut trinken sie… |
| 1171 | Löschung | `trinken` | `*(nicht da)*` | …jahre okay gut [___] sie alkohol ja… |
| 1172 | Löschung | `sie` | `*(nicht da)*` | …okay gut trinken [___] alkohol ja nicht… |
| 1173 | Löschung | `alkohol` | `*(nicht da)*` | …gut trinken sie [___] ja nicht viel… |
| 1174 | Löschung | `ja` | `*(nicht da)*` | …trinken sie alkohol [___] nicht viel aber… |
| 1175 | Löschung | `nicht` | `*(nicht da)*` | …sie alkohol ja [___] viel aber schon… |
| 1176 | Löschung | `viel` | `*(nicht da)*` | …alkohol ja nicht [___] aber schon abends… |
| 1177 | Löschung | `aber` | `*(nicht da)*` | …ja nicht viel [___] schon abends nach… |
| 1178 | Löschung | `schon` | `*(nicht da)*` | …nicht viel aber [___] abends nach der… |
| 1179 | Löschung | `abends` | `*(nicht da)*` | …viel aber schon [___] nach der arbeit… |
| 1180 | Löschung | `nach` | `*(nicht da)*` | …aber schon abends [___] der arbeit gerne… |
| 1181 | Löschung | `der` | `*(nicht da)*` | …schon abends nach [___] arbeit gerne ein… |
| 1182 | Löschung | `arbeit` | `*(nicht da)*` | …abends nach der [___] gerne ein glas… |
| 1183 | Löschung | `gerne` | `*(nicht da)*` | …nach der arbeit [___] ein glas wein… |
| 1184 | Löschung | `ein` | `*(nicht da)*` | …der arbeit gerne [___] glas wein und… |
| 1185 | Löschung | `glas` | `*(nicht da)*` | …arbeit gerne ein [___] wein und am… |
| 1186 | Löschung | `wein` | `*(nicht da)*` | …gerne ein glas [___] und am wochenende… |
| 1187 | Löschung | `am` | `*(nicht da)*` | …glas wein und [___] wochenende wenn wir… |
| 1188 | Löschung | `wochenende` | `*(nicht da)*` | …wein und am [___] wenn wir mit… |
| 1189 | Löschung | `wenn` | `*(nicht da)*` | …und am wochenende [___] wir mit freunden… |
| 1190 | Löschung | `wir` | `*(nicht da)*` | …am wochenende wenn [___] mit freunden unterwegs… |
| 1191 | Löschung | `mit` | `*(nicht da)*` | …wochenende wenn wir [___] freunden unterwegs sind… |
| 1192 | Löschung | `freunden` | `*(nicht da)*` | …wenn wir mit [___] unterwegs sind dann… |
| 1193 | Löschung | `unterwegs` | `*(nicht da)*` | …wir mit freunden [___] sind dann gerne… |
| 1194 | Löschung | `sind` | `*(nicht da)*` | …mit freunden unterwegs [___] dann gerne auch… |
| 1195 | Löschung | `dann` | `*(nicht da)*` | …freunden unterwegs sind [___] gerne auch zwei… |
| 1196 | Löschung | `gerne` | `*(nicht da)*` | …unterwegs sind dann [___] auch zwei oder… |
| 1197 | Löschung | `auch` | `*(nicht da)*` | …sind dann gerne [___] zwei oder drei… |
| 1198 | Löschung | `zwei` | `*(nicht da)*` | …dann gerne auch [___] oder drei läser… |
| 1199 | Löschung | `oder` | `*(nicht da)*` | …gerne auch zwei [___] drei läser okay… |
| 1200 | Löschung | `drei` | `*(nicht da)*` | …auch zwei oder [___] läser okay dieses… |
| 1201 | Löschung | `läser` | `*(nicht da)*` | …zwei oder drei [___] okay dieses gläschen… |
| 1202 | Löschung | `okay` | `*(nicht da)*` | …oder drei läser [___] dieses gläschen wein… |
| 1203 | Löschung | `dieses` | `*(nicht da)*` | …drei läser okay [___] gläschen wein nach… |
| 1204 | Löschung | `gläschen` | `*(nicht da)*` | …läser okay dieses [___] wein nach der… |
| 1205 | Löschung | `wein` | `*(nicht da)*` | …okay dieses gläschen [___] nach der arbeit… |
| 1206 | Löschung | `nach` | `*(nicht da)*` | …dieses gläschen wein [___] der arbeit ist… |
| 1207 | Löschung | `der` | `*(nicht da)*` | …gläschen wein nach [___] arbeit ist das… |
| 1208 | Löschung | `arbeit` | `*(nicht da)*` | …wein nach der [___] ist das so… |
| 1209 | Löschung | `ist` | `*(nicht da)*` | …nach der arbeit [___] das so einmal… |
| 1210 | Löschung | `das` | `*(nicht da)*` | …der arbeit ist [___] so einmal die… |
| 1211 | Löschung | `so` | `*(nicht da)*` | …arbeit ist das [___] einmal die woche… |
| 1212 | Löschung | `einmal` | `*(nicht da)*` | …ist das so [___] die woche zweimal… |
| 1213 | Löschung | `die` | `*(nicht da)*` | …das so einmal [___] woche zweimal oder… |
| 1214 | Löschung | `woche` | `*(nicht da)*` | …so einmal die [___] zweimal oder doch… |
| 1215 | Löschung | `zweimal` | `*(nicht da)*` | …einmal die woche [___] oder doch öfter… |
| 1216 | Löschung | `oder` | `*(nicht da)*` | …die woche zweimal [___] doch öfter ach… |
| 1217 | Löschung | `doch` | `*(nicht da)*` | …woche zweimal oder [___] öfter ach das… |
| 1218 | Löschung | `öfter` | `*(nicht da)*` | …zweimal oder doch [___] ach das ist… |
| 1219 | Löschung | `ach` | `*(nicht da)*` | …oder doch öfter [___] das ist schon… |
| 1220 | Substitution | `das` | `wie` | …doch öfter ach [___] ist schon fast… |
| 1221 | Löschung | `schon` | `*(nicht da)*` | …ach das ist [___] fast jeden abend… |
| 1222 | Löschung | `fast` | `*(nicht da)*` | …das ist schon [___] jeden abend aber… |
| 1223 | Löschung | `jeden` | `*(nicht da)*` | …ist schon fast [___] abend aber ein… |
| 1224 | Löschung | `abend` | `*(nicht da)*` | …schon fast jeden [___] aber ein kleines… |
| 1225 | Löschung | `aber` | `*(nicht da)*` | …fast jeden abend [___] ein kleines gläschen… |
| 1226 | Löschung | `ein` | `*(nicht da)*` | …jeden abend aber [___] kleines gläschen okay… |
| 1227 | Löschung | `kleines` | `*(nicht da)*` | …abend aber ein [___] gläschen okay wunderbar… |
| 1228 | Löschung | `gläschen` | `*(nicht da)*` | …aber ein kleines [___] okay wunderbar frau… |
| 1229 | Löschung | `okay` | `*(nicht da)*` | …ein kleines gläschen [___] wunderbar frau beckenwestfalen… |
| 1230 | Löschung | `wunderbar` | `*(nicht da)*` | …kleines gläschen okay [___] frau beckenwestfalen nehmen… |
| 1231 | Löschung | `frau` | `*(nicht da)*` | …gläschen okay wunderbar [___] beckenwestfalen nehmen sie… |
| 1232 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …okay wunderbar frau [___] nehmen sie das… |
| 1233 | Löschung | `nehmen` | `*(nicht da)*` | …wunderbar frau beckenwestfalen [___] sie das jetzt… |
| 1234 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen nehmen [___] das jetzt bitte… |
| 1235 | Löschung | `das` | `*(nicht da)*` | …beckenwestfalen nehmen sie [___] jetzt bitte nicht… |
| 1236 | Löschung | `jetzt` | `*(nicht da)*` | …nehmen sie das [___] bitte nicht persönlich… |
| 1237 | Löschung | `bitte` | `*(nicht da)*` | …sie das jetzt [___] nicht persönlich das… |
| 1238 | Substitution | `nicht` | `ihr` | …das jetzt bitte [___] persönlich das ist… |
| 1239 | Substitution | `persönlich` | `einkommen` | …jetzt bitte nicht [___] das ist eine… |
| 1240 | Substitution | `das` | `es` | …bitte nicht persönlich [___] ist eine reine… |
| 1241 | Löschung | `eine` | `*(nicht da)*` | …persönlich das ist [___] reine routinefrage die… |
| 1242 | Löschung | `reine` | `*(nicht da)*` | …das ist eine [___] routinefrage die ich… |
| 1243 | Löschung | `routinefrage` | `*(nicht da)*` | …ist eine reine [___] die ich aber… |
| 1244 | Löschung | `die` | `*(nicht da)*` | …eine reine routinefrage [___] ich aber natürlich… |
| 1245 | Löschung | `ich` | `*(nicht da)*` | …reine routinefrage die [___] aber natürlich auch… |
| 1246 | Löschung | `aber` | `*(nicht da)*` | …routinefrage die ich [___] natürlich auch ihnen… |
| 1247 | Löschung | `natürlich` | `*(nicht da)*` | …die ich aber [___] auch ihnen stellen… |
| 1248 | Löschung | `auch` | `*(nicht da)*` | …ich aber natürlich [___] ihnen stellen muss… |
| 1249 | Löschung | `ihnen` | `*(nicht da)*` | …aber natürlich auch [___] stellen muss und… |
| 1250 | Löschung | `stellen` | `*(nicht da)*` | …natürlich auch ihnen [___] muss und zwar… |
| 1251 | Löschung | `muss` | `*(nicht da)*` | …auch ihnen stellen [___] und zwar nehmen… |
| 1252 | Löschung | `und` | `*(nicht da)*` | …ihnen stellen muss [___] zwar nehmen sie… |
| 1253 | Löschung | `zwar` | `*(nicht da)*` | …stellen muss und [___] nehmen sie drogen… |
| 1254 | Löschung | `nehmen` | `*(nicht da)*` | …muss und zwar [___] sie drogen nein… |
| 1255 | Löschung | `sie` | `*(nicht da)*` | …und zwar nehmen [___] drogen nein ich… |
| 1256 | Löschung | `drogen` | `*(nicht da)*` | …zwar nehmen sie [___] nein ich nehme… |
| 1257 | Löschung | `nein` | `*(nicht da)*` | …nehmen sie drogen [___] ich nehme keine… |
| 1258 | Löschung | `ich` | `*(nicht da)*` | …sie drogen nein [___] nehme keine drogen… |
| 1259 | Löschung | `nehme` | `*(nicht da)*` | …drogen nein ich [___] keine drogen wobei… |
| 1260 | Löschung | `keine` | `*(nicht da)*` | …nein ich nehme [___] drogen wobei ich… |
| 1261 | Löschung | `drogen` | `*(nicht da)*` | …ich nehme keine [___] wobei ich zugeben… |
| 1262 | Löschung | `wobei` | `*(nicht da)*` | …nehme keine drogen [___] ich zugeben muss… |
| 1263 | Löschung | `ich` | `*(nicht da)*` | …keine drogen wobei [___] zugeben muss dass… |
| 1264 | Löschung | `zugeben` | `*(nicht da)*` | …drogen wobei ich [___] muss dass ich… |
| 1265 | Löschung | `muss` | `*(nicht da)*` | …wobei ich zugeben [___] dass ich vor… |
| 1266 | Löschung | `dass` | `*(nicht da)*` | …ich zugeben muss [___] ich vor einiger… |
| 1267 | Löschung | `ich` | `*(nicht da)*` | …zugeben muss dass [___] vor einiger zeit… |
| 1268 | Löschung | `vor` | `*(nicht da)*` | …muss dass ich [___] einiger zeit ab… |
| 1269 | Löschung | `einiger` | `*(nicht da)*` | …dass ich vor [___] zeit ab und… |
| 1270 | Löschung | `zeit` | `*(nicht da)*` | …ich vor einiger [___] ab und zu… |
| 1271 | Löschung | `ab` | `*(nicht da)*` | …vor einiger zeit [___] und zu mal… |
| 1272 | Löschung | `und` | `*(nicht da)*` | …einiger zeit ab [___] zu mal ritalin… |
| 1273 | Löschung | `zu` | `*(nicht da)*` | …zeit ab und [___] mal ritalin genommen… |
| 1274 | Löschung | `mal` | `*(nicht da)*` | …ab und zu [___] ritalin genommen habe… |
| 1275 | Löschung | `ritalin` | `*(nicht da)*` | …und zu mal [___] genommen habe okay… |
| 1276 | Löschung | `genommen` | `*(nicht da)*` | …zu mal ritalin [___] habe okay einfach… |
| 1277 | Löschung | `habe` | `*(nicht da)*` | …mal ritalin genommen [___] okay einfach weil… |
| 1278 | Löschung | `okay` | `*(nicht da)*` | …ritalin genommen habe [___] einfach weil wir… |
| 1279 | Löschung | `einfach` | `*(nicht da)*` | …genommen habe okay [___] weil wir ein… |
| 1280 | Löschung | `weil` | `*(nicht da)*` | …habe okay einfach [___] wir ein paar… |
| 1281 | Löschung | `wir` | `*(nicht da)*` | …okay einfach weil [___] ein paar wirklich… |
| 1282 | Löschung | `ein` | `*(nicht da)*` | …einfach weil wir [___] paar wirklich große… |
| 1283 | Löschung | `paar` | `*(nicht da)*` | …weil wir ein [___] wirklich große projekte… |
| 1284 | Löschung | `wirklich` | `*(nicht da)*` | …wir ein paar [___] große projekte auf… |
| 1285 | Löschung | `große` | `*(nicht da)*` | …ein paar wirklich [___] projekte auf der… |
| 1286 | Löschung | `projekte` | `*(nicht da)*` | …paar wirklich große [___] auf der arbeit… |
| 1287 | Löschung | `auf` | `*(nicht da)*` | …wirklich große projekte [___] der arbeit hatten… |
| 1288 | Löschung | `der` | `*(nicht da)*` | …große projekte auf [___] arbeit hatten für… |
| 1289 | Löschung | `arbeit` | `*(nicht da)*` | …projekte auf der [___] hatten für die… |
| 1290 | Löschung | `hatten` | `*(nicht da)*` | …auf der arbeit [___] für die ich… |
| 1291 | Löschung | `für` | `*(nicht da)*` | …der arbeit hatten [___] die ich zuständig… |
| 1292 | Löschung | `die` | `*(nicht da)*` | …arbeit hatten für [___] ich zuständig war… |
| 1293 | Löschung | `ich` | `*(nicht da)*` | …hatten für die [___] zuständig war und… |
| 1294 | Löschung | `zuständig` | `*(nicht da)*` | …für die ich [___] war und ich… |
| 1295 | Löschung | `war` | `*(nicht da)*` | …die ich zuständig [___] und ich musste… |
| 1296 | Löschung | `und` | `*(nicht da)*` | …ich zuständig war [___] ich musste wirklich… |
| 1297 | Löschung | `ich` | `*(nicht da)*` | …zuständig war und [___] musste wirklich sehr… |
| 1298 | Löschung | `musste` | `*(nicht da)*` | …war und ich [___] wirklich sehr lange… |
| 1299 | Löschung | `wirklich` | `*(nicht da)*` | …und ich musste [___] sehr lange arbeiten… |
| 1300 | Substitution | `sehr` | `ausreichend` | …ich musste wirklich [___] lange arbeiten und… |
| 1301 | Substitution | `lange` | `alles` | …musste wirklich sehr [___] arbeiten und ja… |
| 1302 | Substitution | `arbeiten` | `klar` | …wirklich sehr lange [___] und ja habe… |
| 1303 | Löschung | `ja` | `*(nicht da)*` | …lange arbeiten und [___] habe zwei dreimal… |
| 1304 | Löschung | `habe` | `*(nicht da)*` | …arbeiten und ja [___] zwei dreimal ritalin… |
| 1305 | Löschung | `zwei` | `*(nicht da)*` | …und ja habe [___] dreimal ritalin genommen… |
| 1306 | Löschung | `dreimal` | `*(nicht da)*` | …ja habe zwei [___] ritalin genommen okay… |
| 1307 | Löschung | `ritalin` | `*(nicht da)*` | …habe zwei dreimal [___] genommen okay das… |
| 1308 | Löschung | `genommen` | `*(nicht da)*` | …zwei dreimal ritalin [___] okay das war… |
| 1309 | Löschung | `okay` | `*(nicht da)*` | …dreimal ritalin genommen [___] das war es… |
| 1310 | Löschung | `das` | `*(nicht da)*` | …ritalin genommen okay [___] war es aber… |
| 1311 | Löschung | `war` | `*(nicht da)*` | …genommen okay das [___] es aber ja… |
| 1312 | Löschung | `es` | `*(nicht da)*` | …okay das war [___] aber ja sehr… |
| 1313 | Löschung | `aber` | `*(nicht da)*` | …das war es [___] ja sehr gut… |
| 1314 | Löschung | `ja` | `*(nicht da)*` | …war es aber [___] sehr gut okay… |
| 1315 | Löschung | `sehr` | `*(nicht da)*` | …es aber ja [___] gut okay prima… |
| 1316 | Löschung | `gut` | `*(nicht da)*` | …aber ja sehr [___] okay prima kurz… |
| 1317 | Löschung | `okay` | `*(nicht da)*` | …ja sehr gut [___] prima kurz zu… |
| 1318 | Löschung | `prima` | `*(nicht da)*` | …sehr gut okay [___] kurz zu ihrer… |
| 1319 | Löschung | `kurz` | `*(nicht da)*` | …gut okay prima [___] zu ihrer familie… |
| 1320 | Löschung | `zu` | `*(nicht da)*` | …okay prima kurz [___] ihrer familie gibt… |
| 1321 | Löschung | `ihrer` | `*(nicht da)*` | …prima kurz zu [___] familie gibt es… |
| 1322 | Löschung | `familie` | `*(nicht da)*` | …kurz zu ihrer [___] gibt es in… |
| 1323 | Löschung | `in` | `*(nicht da)*` | …familie gibt es [___] ihrer familie eltern… |
| 1324 | Löschung | `ihrer` | `*(nicht da)*` | …gibt es in [___] familie eltern großeltern… |
| 1325 | Löschung | `familie` | `*(nicht da)*` | …es in ihrer [___] eltern großeltern geschwister… |
| 1326 | Löschung | `eltern` | `*(nicht da)*` | …in ihrer familie [___] großeltern geschwister irgendwelche… |
| 1327 | Löschung | `großeltern` | `*(nicht da)*` | …ihrer familie eltern [___] geschwister irgendwelche vorerkrankungen… |
| 1328 | Löschung | `geschwister` | `*(nicht da)*` | …familie eltern großeltern [___] irgendwelche vorerkrankungen oder… |
| 1329 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …großeltern geschwister irgendwelche [___] oder chronische erkrankungen… |
| 1330 | Löschung | `oder` | `*(nicht da)*` | …geschwister irgendwelche vorerkrankungen [___] chronische erkrankungen wie… |
| 1331 | Löschung | `chronische` | `*(nicht da)*` | …irgendwelche vorerkrankungen oder [___] erkrankungen wie zum… |
| 1332 | Löschung | `erkrankungen` | `*(nicht da)*` | …vorerkrankungen oder chronische [___] wie zum beispiel… |
| 1333 | Löschung | `wie` | `*(nicht da)*` | …oder chronische erkrankungen [___] zum beispiel krebs… |
| 1334 | Löschung | `zum` | `*(nicht da)*` | …chronische erkrankungen wie [___] beispiel krebs oder… |
| 1335 | Löschung | `beispiel` | `*(nicht da)*` | …erkrankungen wie zum [___] krebs oder diabetes… |
| 1336 | Löschung | `krebs` | `*(nicht da)*` | …wie zum beispiel [___] oder diabetes oder… |
| 1337 | Löschung | `oder` | `*(nicht da)*` | …zum beispiel krebs [___] diabetes oder einen… |
| 1338 | Löschung | `diabetes` | `*(nicht da)*` | …beispiel krebs oder [___] oder einen herzinfarkt… |
| 1339 | Löschung | `oder` | `*(nicht da)*` | …krebs oder diabetes [___] einen herzinfarkt irgendetwas… |
| 1340 | Löschung | `einen` | `*(nicht da)*` | …oder diabetes oder [___] herzinfarkt irgendetwas was… |
| 1341 | Löschung | `herzinfarkt` | `*(nicht da)*` | …diabetes oder einen [___] irgendetwas was ihnen… |
| 1342 | Löschung | `irgendetwas` | `*(nicht da)*` | …oder einen herzinfarkt [___] was ihnen bekannt… |
| 1343 | Löschung | `was` | `*(nicht da)*` | …einen herzinfarkt irgendetwas [___] ihnen bekannt ist… |
| 1344 | Löschung | `ihnen` | `*(nicht da)*` | …herzinfarkt irgendetwas was [___] bekannt ist das… |
| 1345 | Löschung | `bekannt` | `*(nicht da)*` | …irgendetwas was ihnen [___] ist das gibt… |
| 1346 | Löschung | `ist` | `*(nicht da)*` | …was ihnen bekannt [___] das gibt es… |
| 1347 | Löschung | `das` | `*(nicht da)*` | …ihnen bekannt ist [___] gibt es ja… |
| 1348 | Löschung | `gibt` | `*(nicht da)*` | …bekannt ist das [___] es ja großeltern… |
| 1349 | Löschung | `es` | `*(nicht da)*` | …ist das gibt [___] ja großeltern auch… |
| 1350 | Löschung | `ja` | `*(nicht da)*` | …das gibt es [___] großeltern auch ja… |
| 1351 | Löschung | `großeltern` | `*(nicht da)*` | …gibt es ja [___] auch ja klar… |
| 1352 | Löschung | `auch` | `*(nicht da)*` | …es ja großeltern [___] ja klar mein… |
| 1353 | Löschung | `ja` | `*(nicht da)*` | …ja großeltern auch [___] klar mein großvater… |
| 1354 | Löschung | `klar` | `*(nicht da)*` | …großeltern auch ja [___] mein großvater hatte… |
| 1355 | Löschung | `mein` | `*(nicht da)*` | …auch ja klar [___] großvater hatte leberzirrhose… |
| 1356 | Löschung | `großvater` | `*(nicht da)*` | …ja klar mein [___] hatte leberzirrhose und… |
| 1357 | Löschung | `hatte` | `*(nicht da)*` | …klar mein großvater [___] leberzirrhose und ist… |
| 1358 | Löschung | `leberzirrhose` | `*(nicht da)*` | …mein großvater hatte [___] und ist leider… |
| 1359 | Löschung | `und` | `*(nicht da)*` | …großvater hatte leberzirrhose [___] ist leider auch… |
| 1360 | Löschung | `ist` | `*(nicht da)*` | …hatte leberzirrhose und [___] leider auch daran… |
| 1361 | Löschung | `leider` | `*(nicht da)*` | …leberzirrhose und ist [___] auch daran geschrauben… |
| 1362 | Löschung | `auch` | `*(nicht da)*` | …und ist leider [___] daran geschrauben oh… |
| 1363 | Löschung | `daran` | `*(nicht da)*` | …ist leider auch [___] geschrauben oh das… |
| 1364 | Löschung | `geschrauben` | `*(nicht da)*` | …leider auch daran [___] oh das tut… |
| 1365 | Löschung | `oh` | `*(nicht da)*` | …auch daran geschrauben [___] das tut mir… |
| 1366 | Löschung | `das` | `*(nicht da)*` | …daran geschrauben oh [___] tut mir leid… |
| 1367 | Löschung | `tut` | `*(nicht da)*` | …geschrauben oh das [___] mir leid danke… |
| 1368 | Löschung | `mir` | `*(nicht da)*` | …oh das tut [___] leid danke ist… |
| 1369 | Löschung | `leid` | `*(nicht da)*` | …das tut mir [___] danke ist schon… |
| 1370 | Löschung | `danke` | `*(nicht da)*` | …tut mir leid [___] ist schon lange… |
| 1371 | Löschung | `ist` | `*(nicht da)*` | …mir leid danke [___] schon lange her… |
| 1372 | Löschung | `schon` | `*(nicht da)*` | …leid danke ist [___] lange her und… |
| 1373 | Löschung | `lange` | `*(nicht da)*` | …danke ist schon [___] her und meine… |
| 1374 | Löschung | `her` | `*(nicht da)*` | …ist schon lange [___] und meine großmutter… |
| 1375 | Löschung | `und` | `*(nicht da)*` | …schon lange her [___] meine großmutter hatte… |
| 1376 | Löschung | `meine` | `*(nicht da)*` | …lange her und [___] großmutter hatte großkrebs… |
| 1377 | Löschung | `großmutter` | `*(nicht da)*` | …her und meine [___] hatte großkrebs aber… |
| 1378 | Löschung | `hatte` | `*(nicht da)*` | …und meine großmutter [___] großkrebs aber sie… |
| 1379 | Substitution | `großkrebs` | `dinge` | …meine großmutter hatte [___] aber sie lebt… |
| 1380 | Substitution | `aber` | `die` | …großmutter hatte großkrebs [___] sie lebt noch… |
| 1381 | Substitution | `lebt` | `mir` | …großkrebs aber sie [___] noch okay sehr… |
| 1382 | Löschung | `okay` | `*(nicht da)*` | …sie lebt noch [___] sehr gut ihre… |
| 1383 | Löschung | `sehr` | `*(nicht da)*` | …lebt noch okay [___] gut ihre eltern… |
| 1384 | Löschung | `gut` | `*(nicht da)*` | …noch okay sehr [___] ihre eltern sind… |
| 1385 | Löschung | `ihre` | `*(nicht da)*` | …okay sehr gut [___] eltern sind gesund… |
| 1386 | Löschung | `eltern` | `*(nicht da)*` | …sehr gut ihre [___] sind gesund meine… |
| 1387 | Löschung | `sind` | `*(nicht da)*` | …gut ihre eltern [___] gesund meine eltern… |
| 1388 | Löschung | `gesund` | `*(nicht da)*` | …ihre eltern sind [___] meine eltern sind… |
| 1389 | Löschung | `meine` | `*(nicht da)*` | …eltern sind gesund [___] eltern sind zum… |
| 1390 | Löschung | `eltern` | `*(nicht da)*` | …sind gesund meine [___] sind zum glück… |
| 1391 | Löschung | `sind` | `*(nicht da)*` | …gesund meine eltern [___] zum glück gesund… |
| 1392 | Löschung | `zum` | `*(nicht da)*` | …meine eltern sind [___] glück gesund ja… |
| 1393 | Löschung | `glück` | `*(nicht da)*` | …eltern sind zum [___] gesund ja sehr… |
| 1394 | Löschung | `gesund` | `*(nicht da)*` | …sind zum glück [___] ja sehr schön… |
| 1395 | Löschung | `ja` | `*(nicht da)*` | …zum glück gesund [___] sehr schön haben… |
| 1396 | Löschung | `sehr` | `*(nicht da)*` | …glück gesund ja [___] schön haben sie… |
| 1397 | Löschung | `schön` | `*(nicht da)*` | …gesund ja sehr [___] haben sie geschwister… |
| 1398 | Löschung | `haben` | `*(nicht da)*` | …ja sehr schön [___] sie geschwister frau… |
| 1399 | Löschung | `sie` | `*(nicht da)*` | …sehr schön haben [___] geschwister frau böcken… |
| 1400 | Löschung | `geschwister` | `*(nicht da)*` | …schön haben sie [___] frau böcken westfalen… |
| 1401 | Löschung | `frau` | `*(nicht da)*` | …haben sie geschwister [___] böcken westfalen ich… |
| 1402 | Löschung | `böcken` | `*(nicht da)*` | …sie geschwister frau [___] westfalen ich habe… |
| 1403 | Löschung | `westfalen` | `*(nicht da)*` | …geschwister frau böcken [___] ich habe eine… |
| 1404 | Löschung | `ich` | `*(nicht da)*` | …frau böcken westfalen [___] habe eine schwester… |
| 1405 | Löschung | `habe` | `*(nicht da)*` | …böcken westfalen ich [___] eine schwester und… |
| 1406 | Löschung | `eine` | `*(nicht da)*` | …westfalen ich habe [___] schwester und sie… |
| 1407 | Löschung | `schwester` | `*(nicht da)*` | …ich habe eine [___] und sie hat… |
| 1408 | Löschung | `und` | `*(nicht da)*` | …habe eine schwester [___] sie hat auch… |
| 1409 | Löschung | `sie` | `*(nicht da)*` | …eine schwester und [___] hat auch ein… |
| 1410 | Löschung | `hat` | `*(nicht da)*` | …schwester und sie [___] auch ein paar… |
| 1411 | Löschung | `auch` | `*(nicht da)*` | …und sie hat [___] ein paar problemchen… |
| 1412 | Löschung | `ein` | `*(nicht da)*` | …sie hat auch [___] paar problemchen und… |
| 1413 | Löschung | `paar` | `*(nicht da)*` | …hat auch ein [___] problemchen und zwar… |
| 1414 | Löschung | `problemchen` | `*(nicht da)*` | …auch ein paar [___] und zwar hat… |
| 1415 | Löschung | `und` | `*(nicht da)*` | …ein paar problemchen [___] zwar hat sie… |
| 1416 | Löschung | `zwar` | `*(nicht da)*` | …paar problemchen und [___] hat sie asthma… |
| 1417 | Löschung | `hat` | `*(nicht da)*` | …problemchen und zwar [___] sie asthma und… |
| 1418 | Löschung | `sie` | `*(nicht da)*` | …und zwar hat [___] asthma und neurodermitis… |
| 1419 | Löschung | `asthma` | `*(nicht da)*` | …zwar hat sie [___] und neurodermitis asthma… |
| 1420 | Löschung | `und` | `*(nicht da)*` | …hat sie asthma [___] neurodermitis asthma und… |
| 1421 | Löschung | `neurodermitis` | `*(nicht da)*` | …sie asthma und [___] asthma und neurodermitis… |
| 1422 | Löschung | `asthma` | `*(nicht da)*` | …asthma und neurodermitis [___] und neurodermitis okay… |
| 1423 | Löschung | `und` | `*(nicht da)*` | …und neurodermitis asthma [___] neurodermitis okay aber… |
| 1424 | Löschung | `neurodermitis` | `*(nicht da)*` | …neurodermitis asthma und [___] okay aber sonst… |
| 1425 | Löschung | `okay` | `*(nicht da)*` | …asthma und neurodermitis [___] aber sonst ist… |
| 1426 | Löschung | `aber` | `*(nicht da)*` | …und neurodermitis okay [___] sonst ist auch… |
| 1427 | Löschung | `sonst` | `*(nicht da)*` | …neurodermitis okay aber [___] ist auch sie… |
| 1428 | Löschung | `ist` | `*(nicht da)*` | …okay aber sonst [___] auch sie gesund… |
| 1429 | Löschung | `auch` | `*(nicht da)*` | …aber sonst ist [___] sie gesund ja… |
| 1430 | Löschung | `sie` | `*(nicht da)*` | …sonst ist auch [___] gesund ja sonst… |
| 1431 | Löschung | `gesund` | `*(nicht da)*` | …ist auch sie [___] ja sonst geht… |
| 1432 | Substitution | `ja` | `erzählen` | …auch sie gesund [___] sonst geht sie… |
| 1433 | Substitution | `sonst` | `möchten` | …sie gesund ja [___] geht sie gut… |
| 1434 | Substitution | `geht` | `die` | …gesund ja sonst [___] sie gut sehr… |
| 1435 | Substitution | `sie` | `wir` | …ja sonst geht [___] gut sehr gut… |
| 1436 | Substitution | `gut` | `noch` | …sonst geht sie [___] sehr gut haben… |
| 1437 | Substitution | `sehr` | `nicht` | …geht sie gut [___] gut haben sie… |
| 1438 | Substitution | `gut` | `besprochen` | …sie gut sehr [___] haben sie kinder… |
| 1439 | Löschung | `sie` | `*(nicht da)*` | …sehr gut haben [___] kinder frau böcken… |
| 1440 | Löschung | `kinder` | `*(nicht da)*` | …gut haben sie [___] frau böcken westfalen… |
| 1441 | Löschung | `frau` | `*(nicht da)*` | …haben sie kinder [___] böcken westfalen nein… |
| 1442 | Löschung | `böcken` | `*(nicht da)*` | …sie kinder frau [___] westfalen nein ich… |
| 1443 | Löschung | `westfalen` | `*(nicht da)*` | …kinder frau böcken [___] nein ich habe… |
| 1444 | Löschung | `ich` | `*(nicht da)*` | …böcken westfalen nein [___] habe keine kinder… |
| 1445 | Löschung | `habe` | `*(nicht da)*` | …westfalen nein ich [___] keine kinder wie… |
| 1446 | Löschung | `keine` | `*(nicht da)*` | …nein ich habe [___] kinder wie sieht… |
| 1447 | Löschung | `kinder` | `*(nicht da)*` | …ich habe keine [___] wie sieht es… |
| 1448 | Löschung | `wie` | `*(nicht da)*` | …habe keine kinder [___] sieht es denn… |
| 1449 | Löschung | `sieht` | `*(nicht da)*` | …keine kinder wie [___] es denn in… |
| 1450 | Löschung | `es` | `*(nicht da)*` | …kinder wie sieht [___] denn in ihrem… |
| 1451 | Löschung | `denn` | `*(nicht da)*` | …wie sieht es [___] in ihrem sozialleben… |
| 1452 | Löschung | `in` | `*(nicht da)*` | …sieht es denn [___] ihrem sozialleben aus… |
| 1453 | Löschung | `ihrem` | `*(nicht da)*` | …es denn in [___] sozialleben aus sind… |
| 1454 | Löschung | `sozialleben` | `*(nicht da)*` | …denn in ihrem [___] aus sind sie… |
| 1455 | Löschung | `aus` | `*(nicht da)*` | …in ihrem sozialleben [___] sind sie verheiratet… |
| 1456 | Löschung | `sind` | `*(nicht da)*` | …ihrem sozialleben aus [___] sie verheiratet ich… |
| 1457 | Löschung | `sie` | `*(nicht da)*` | …sozialleben aus sind [___] verheiratet ich bin… |
| 1458 | Löschung | `verheiratet` | `*(nicht da)*` | …aus sind sie [___] ich bin frisch… |
| 1459 | Löschung | `ich` | `*(nicht da)*` | …sind sie verheiratet [___] bin frisch verheiratet… |
| 1460 | Löschung | `bin` | `*(nicht da)*` | …sie verheiratet ich [___] frisch verheiratet ja… |
| 1461 | Löschung | `frisch` | `*(nicht da)*` | …verheiratet ich bin [___] verheiratet ja seit… |
| 1462 | Löschung | `verheiratet` | `*(nicht da)*` | …ich bin frisch [___] ja seit fünf… |
| 1463 | Löschung | `ja` | `*(nicht da)*` | …bin frisch verheiratet [___] seit fünf monaten… |
| 1464 | Löschung | `seit` | `*(nicht da)*` | …frisch verheiratet ja [___] fünf monaten wie… |
| 1465 | Löschung | `fünf` | `*(nicht da)*` | …verheiratet ja seit [___] monaten wie schön… |
| 1466 | Löschung | `monaten` | `*(nicht da)*` | …ja seit fünf [___] wie schön herzlichen… |
| 1467 | Löschung | `wie` | `*(nicht da)*` | …seit fünf monaten [___] schön herzlichen glückwunsch… |
| 1468 | Löschung | `schön` | `*(nicht da)*` | …fünf monaten wie [___] herzlichen glückwunsch auch… |
| 1469 | Löschung | `herzlichen` | `*(nicht da)*` | …monaten wie schön [___] glückwunsch auch dazu… |
| 1470 | Löschung | `glückwunsch` | `*(nicht da)*` | …wie schön herzlichen [___] auch dazu herzlichen… |
| 1471 | Löschung | `auch` | `*(nicht da)*` | …schön herzlichen glückwunsch [___] dazu herzlichen dank… |
| 1472 | Löschung | `dazu` | `*(nicht da)*` | …herzlichen glückwunsch auch [___] herzlichen dank sehr… |
| 1473 | Löschung | `herzlichen` | `*(nicht da)*` | …glückwunsch auch dazu [___] dank sehr schön… |
| 1474 | Löschung | `dank` | `*(nicht da)*` | …auch dazu herzlichen [___] sehr schön dann… |
| 1475 | Löschung | `sehr` | `*(nicht da)*` | …dazu herzlichen dank [___] schön dann gehe… |
| 1476 | Löschung | `schön` | `*(nicht da)*` | …herzlichen dank sehr [___] dann gehe ich… |
| 1477 | Löschung | `dann` | `*(nicht da)*` | …dank sehr schön [___] gehe ich davon… |
| 1478 | Löschung | `gehe` | `*(nicht da)*` | …sehr schön dann [___] ich davon aus… |
| 1479 | Löschung | `davon` | `*(nicht da)*` | …dann gehe ich [___] aus sie leben… |
| 1480 | Löschung | `aus` | `*(nicht da)*` | …gehe ich davon [___] sie leben auch… |
| 1481 | Löschung | `sie` | `*(nicht da)*` | …ich davon aus [___] leben auch mit… |
| 1482 | Löschung | `leben` | `*(nicht da)*` | …davon aus sie [___] auch mit ihrem… |
| 1483 | Löschung | `auch` | `*(nicht da)*` | …aus sie leben [___] mit ihrem ehemann… |
| 1484 | Löschung | `mit` | `*(nicht da)*` | …sie leben auch [___] ihrem ehemann zusammen… |
| 1485 | Löschung | `ihrem` | `*(nicht da)*` | …leben auch mit [___] ehemann zusammen das… |
| 1486 | Löschung | `ehemann` | `*(nicht da)*` | …auch mit ihrem [___] zusammen das ist… |
| 1487 | Substitution | `zusammen` | `glaube` | …mit ihrem ehemann [___] das ist richtig… |
| 1488 | Löschung | `ist` | `*(nicht da)*` | …ehemann zusammen das [___] richtig ja okay… |
| 1489 | Substitution | `richtig` | `war` | …zusammen das ist [___] ja okay prima… |
| 1490 | Substitution | `ja` | `alles` | …das ist richtig [___] okay prima wir… |
| 1491 | Löschung | `prima` | `*(nicht da)*` | …richtig ja okay [___] wir hatten zwar… |
| 1492 | Löschung | `wir` | `*(nicht da)*` | …ja okay prima [___] hatten zwar eben… |
| 1493 | Substitution | `hatten` | `dann` | …okay prima wir [___] zwar eben schon… |
| 1494 | Substitution | `zwar` | `schauen` | …prima wir hatten [___] eben schon mal… |
| 1495 | Substitution | `eben` | `wir` | …wir hatten zwar [___] schon mal kurz… |
| 1496 | Substitution | `schon` | `uns` | …hatten zwar eben [___] mal kurz über… |
| 1497 | Löschung | `kurz` | `*(nicht da)*` | …eben schon mal [___] über ihre arbeit… |
| 1498 | Löschung | `über` | `*(nicht da)*` | …schon mal kurz [___] ihre arbeit gesprochen… |
| 1499 | Löschung | `arbeit` | `*(nicht da)*` | …kurz über ihre [___] gesprochen aber ich… |
| 1500 | Löschung | `gesprochen` | `*(nicht da)*` | …über ihre arbeit [___] aber ich habe… |
| 1501 | Löschung | `aber` | `*(nicht da)*` | …ihre arbeit gesprochen [___] ich habe es… |
| 1502 | Substitution | `ich` | `knie` | …arbeit gesprochen aber [___] habe es nicht… |
| 1503 | Substitution | `habe` | `an` | …gesprochen aber ich [___] es nicht ganz… |
| 1504 | Substitution | `es` | `können` | …aber ich habe [___] nicht ganz auf… |
| 1505 | Substitution | `nicht` | `sie` | …ich habe es [___] ganz auf dem… |
| 1506 | Substitution | `ganz` | `bitte` | …habe es nicht [___] auf dem schirm… |
| 1507 | Löschung | `dem` | `*(nicht da)*` | …nicht ganz auf [___] schirm ob ich… |
| 1508 | Löschung | `schirm` | `*(nicht da)*` | …ganz auf dem [___] ob ich sie… |
| 1509 | Löschung | `ob` | `*(nicht da)*` | …auf dem schirm [___] ich sie schon… |
| 1510 | Substitution | `ich` | `die` | …dem schirm ob [___] sie schon gefragt… |
| 1511 | Substitution | `sie` | `untersuchungsliege` | …schirm ob ich [___] schon gefragt habe… |
| 1512 | Substitution | `schon` | `ja` | …ob ich sie [___] gefragt habe was… |
| 1513 | Substitution | `gefragt` | `natürlich` | …ich sie schon [___] habe was sie… |
| 1514 | Substitution | `habe` | `gut` | …sie schon gefragt [___] was sie denn… |
| 1515 | Substitution | `was` | `können` | …schon gefragt habe [___] sie denn beruflich… |
| 1516 | Löschung | `denn` | `*(nicht da)*` | …habe was sie [___] beruflich machen ich… |
| 1517 | Löschung | `beruflich` | `*(nicht da)*` | …was sie denn [___] machen ich arbeite… |
| 1518 | Löschung | `machen` | `*(nicht da)*` | …sie denn beruflich [___] ich arbeite in… |
| 1519 | Löschung | `ich` | `*(nicht da)*` | …denn beruflich machen [___] arbeite in einer… |
| 1520 | Löschung | `arbeite` | `*(nicht da)*` | …beruflich machen ich [___] in einer marketingagentur… |
| 1521 | Löschung | `in` | `*(nicht da)*` | …machen ich arbeite [___] einer marketingagentur wenn… |
| 1522 | Löschung | `einer` | `*(nicht da)*` | …ich arbeite in [___] marketingagentur wenn ich… |
| 1523 | Löschung | `marketingagentur` | `*(nicht da)*` | …arbeite in einer [___] wenn ich da… |
| 1524 | Löschung | `wenn` | `*(nicht da)*` | …in einer marketingagentur [___] ich da für… |
| 1525 | Löschung | `ich` | `*(nicht da)*` | …einer marketingagentur wenn [___] da für größere… |
| 1526 | Löschung | `da` | `*(nicht da)*` | …marketingagentur wenn ich [___] für größere kunden… |
| 1527 | Löschung | `für` | `*(nicht da)*` | …wenn ich da [___] größere kunden und… |
| 1528 | Löschung | `größere` | `*(nicht da)*` | …ich da für [___] kunden und für… |
| 1529 | Löschung | `kunden` | `*(nicht da)*` | …da für größere [___] und für größere… |
| 1530 | Löschung | `und` | `*(nicht da)*` | …für größere kunden [___] für größere firmenkunden… |
| 1531 | Substitution | `für` | `bitte` | …größere kunden und [___] größere firmenkunden und… |
| 1532 | Substitution | `größere` | `einmal` | …kunden und für [___] firmenkunden und marketingprojekte… |
| 1533 | Substitution | `firmenkunden` | `aufstehen` | …und für größere [___] und marketingprojekte zuständig… |
| 1534 | Löschung | `marketingprojekte` | `*(nicht da)*` | …größere firmenkunden und [___] zuständig bin okay… |
| 1535 | Löschung | `zuständig` | `*(nicht da)*` | …firmenkunden und marketingprojekte [___] bin okay sehr… |
| 1536 | Löschung | `bin` | `*(nicht da)*` | …und marketingprojekte zuständig [___] okay sehr gut… |
| 1537 | Löschung | `okay` | `*(nicht da)*` | …marketingprojekte zuständig bin [___] sehr gut eine… |
| 1538 | Löschung | `sehr` | `*(nicht da)*` | …zuständig bin okay [___] gut eine letzte… |
| 1539 | Substitution | `gut` | `ein` | …bin okay sehr [___] eine letzte frage… |
| 1540 | Substitution | `eine` | `paar` | …okay sehr gut [___] letzte frage noch… |
| 1541 | Substitution | `letzte` | `schritte` | …sehr gut eine [___] frage noch frau… |
| 1542 | Substitution | `frage` | `gehen` | …gut eine letzte [___] noch frau böcken… |
| 1543 | Substitution | `noch` | `gerne` | …eine letzte frage [___] frau böcken westfalen… |
| 1544 | Substitution | `frau` | `und` | …letzte frage noch [___] böcken westfalen waren… |
| 1545 | Substitution | `böcken` | `jetzt` | …frage noch frau [___] westfalen waren sie… |
| 1546 | Substitution | `westfalen` | `bitte` | …noch frau böcken [___] waren sie in… |
| 1547 | Substitution | `waren` | `setzen` | …frau böcken westfalen [___] sie in der… |
| 1548 | Substitution | `in` | `sich` | …westfalen waren sie [___] der letzten zeit… |
| 1549 | Substitution | `der` | `wieder` | …waren sie in [___] letzten zeit im… |
| 1550 | Substitution | `letzten` | `hin` | …sie in der [___] zeit im ausland… |
| 1551 | Substitution | `zeit` | `können` | …in der letzten [___] im ausland ja… |
| 1552 | Substitution | `im` | `sie` | …der letzten zeit [___] ausland ja ich… |
| 1553 | Substitution | `ausland` | `mir` | …letzten zeit im [___] ja ich war… |
| 1554 | Substitution | `ja` | `zeigen` | …zeit im ausland [___] ich war vor… |
| 1555 | Substitution | `ich` | `wo` | …im ausland ja [___] war vor zwei… |
| 1556 | Substitution | `war` | `genau` | …ausland ja ich [___] vor zwei monaten… |
| 1557 | Substitution | `vor` | `es` | …ja ich war [___] zwei monaten geschäftlich… |
| 1558 | Substitution | `zwei` | `wehtut` | …ich war vor [___] monaten geschäftlich in… |
| 1559 | Substitution | `monaten` | `hier` | …war vor zwei [___] geschäftlich in singapur… |
| 1560 | Substitution | `geschäftlich` | `an` | …vor zwei monaten [___] in singapur okay… |
| 1561 | Substitution | `in` | `dieser` | …zwei monaten geschäftlich [___] singapur okay und… |
| 1562 | Substitution | `singapur` | `stelle` | …monaten geschäftlich in [___] okay und wie… |
| 1563 | Substitution | `wie` | `jetzt` | …singapur okay und [___] lange waren sie… |
| 1564 | Substitution | `lange` | `bitte` | …okay und wie [___] waren sie da… |
| 1565 | Substitution | `waren` | `strecken` | …und wie lange [___] sie da zwei… |
| 1566 | Löschung | `da` | `*(nicht da)*` | …lange waren sie [___] zwei wochen insgesamt… |
| 1567 | Löschung | `zwei` | `*(nicht da)*` | …waren sie da [___] wochen insgesamt zwei… |
| 1568 | Löschung | `wochen` | `*(nicht da)*` | …sie da zwei [___] insgesamt zwei wochen… |
| 1569 | Löschung | `insgesamt` | `*(nicht da)*` | …da zwei wochen [___] zwei wochen insgesamt… |
| 1570 | Löschung | `zwei` | `*(nicht da)*` | …zwei wochen insgesamt [___] wochen insgesamt okay… |
| 1571 | Löschung | `wochen` | `*(nicht da)*` | …wochen insgesamt zwei [___] insgesamt okay gut… |
| 1572 | Löschung | `insgesamt` | `*(nicht da)*` | …insgesamt zwei wochen [___] okay gut frau… |
| 1573 | Löschung | `okay` | `*(nicht da)*` | …zwei wochen insgesamt [___] gut frau böcken… |
| 1574 | Löschung | `gut` | `*(nicht da)*` | …wochen insgesamt okay [___] frau böcken westfalen… |
| 1575 | Löschung | `frau` | `*(nicht da)*` | …insgesamt okay gut [___] böcken westfalen von… |
| 1576 | Löschung | `böcken` | `*(nicht da)*` | …okay gut frau [___] westfalen von meiner… |
| 1577 | Löschung | `westfalen` | `*(nicht da)*` | …gut frau böcken [___] von meiner seite… |
| 1578 | Löschung | `von` | `*(nicht da)*` | …frau böcken westfalen [___] meiner seite war… |
| 1579 | Löschung | `meiner` | `*(nicht da)*` | …böcken westfalen von [___] seite war es… |
| 1580 | Löschung | `seite` | `*(nicht da)*` | …westfalen von meiner [___] war es das… |
| 1581 | Löschung | `war` | `*(nicht da)*` | …von meiner seite [___] es das ich… |
| 1582 | Löschung | `es` | `*(nicht da)*` | …meiner seite war [___] das ich habe… |
| 1583 | Löschung | `ich` | `*(nicht da)*` | …war es das [___] habe alle fragen… |
| 1584 | Löschung | `habe` | `*(nicht da)*` | …es das ich [___] alle fragen gestellt… |
| 1585 | Löschung | `alle` | `*(nicht da)*` | …das ich habe [___] fragen gestellt ich… |
| 1586 | Substitution | `fragen` | `bein` | …ich habe alle [___] gestellt ich würde… |
| 1587 | Substitution | `gestellt` | `ganz` | …habe alle fragen [___] ich würde das… |
| 1588 | Substitution | `ich` | `aus` | …alle fragen gestellt [___] würde das gleich… |
| 1589 | Substitution | `würde` | `streckt` | …fragen gestellt ich [___] das gleich nochmal… |
| 1590 | Löschung | `gleich` | `*(nicht da)*` | …ich würde das [___] nochmal mit ihnen… |
| 1591 | Löschung | `nochmal` | `*(nicht da)*` | …würde das gleich [___] mit ihnen durchgehen… |
| 1592 | Löschung | `mit` | `*(nicht da)*` | …das gleich nochmal [___] ihnen durchgehen um… |
| 1593 | Löschung | `ihnen` | `*(nicht da)*` | …gleich nochmal mit [___] durchgehen um abzugleichen… |
| 1594 | Löschung | `durchgehen` | `*(nicht da)*` | …nochmal mit ihnen [___] um abzugleichen dass… |
| 1595 | Löschung | `um` | `*(nicht da)*` | …mit ihnen durchgehen [___] abzugleichen dass ich… |
| 1596 | Löschung | `abzugleichen` | `*(nicht da)*` | …ihnen durchgehen um [___] dass ich auch… |
| 1597 | Löschung | `dass` | `*(nicht da)*` | …durchgehen um abzugleichen [___] ich auch wirklich… |
| 1598 | Löschung | `ich` | `*(nicht da)*` | …um abzugleichen dass [___] auch wirklich alles… |
| 1599 | Löschung | `auch` | `*(nicht da)*` | …abzugleichen dass ich [___] wirklich alles richtig… |
| 1600 | Löschung | `wirklich` | `*(nicht da)*` | …dass ich auch [___] alles richtig notiert… |
| 1601 | Löschung | `alles` | `*(nicht da)*` | …ich auch wirklich [___] richtig notiert habe… |
| 1602 | Löschung | `richtig` | `*(nicht da)*` | …auch wirklich alles [___] notiert habe vorher… |
| 1603 | Löschung | `notiert` | `*(nicht da)*` | …wirklich alles richtig [___] habe vorher möchte… |
| 1604 | Löschung | `habe` | `*(nicht da)*` | …alles richtig notiert [___] vorher möchte ich… |
| 1605 | Löschung | `vorher` | `*(nicht da)*` | …richtig notiert habe [___] möchte ich sie… |
| 1606 | Löschung | `möchte` | `*(nicht da)*` | …notiert habe vorher [___] ich sie aber… |
| 1607 | Löschung | `ich` | `*(nicht da)*` | …habe vorher möchte [___] sie aber noch… |
| 1608 | Löschung | `sie` | `*(nicht da)*` | …vorher möchte ich [___] aber noch kurz… |
| 1609 | Löschung | `aber` | `*(nicht da)*` | …möchte ich sie [___] noch kurz fragen… |
| 1610 | Löschung | `noch` | `*(nicht da)*` | …ich sie aber [___] kurz fragen haben… |
| 1611 | Löschung | `kurz` | `*(nicht da)*` | …sie aber noch [___] fragen haben sie… |
| 1612 | Löschung | `fragen` | `*(nicht da)*` | …aber noch kurz [___] haben sie fragen… |
| 1613 | Löschung | `haben` | `*(nicht da)*` | …noch kurz fragen [___] sie fragen an… |
| 1614 | Löschung | `sie` | `*(nicht da)*` | …kurz fragen haben [___] fragen an mich… |
| 1615 | Löschung | `fragen` | `*(nicht da)*` | …fragen haben sie [___] an mich ja… |
| 1616 | Substitution | `an` | `bein` | …haben sie fragen [___] mich ja eine… |
| 1617 | Substitution | `mich` | `und` | …sie fragen an [___] ja eine meinen… |
| 1618 | Substitution | `ja` | `jetzt` | …fragen an mich [___] eine meinen sie… |
| 1619 | Substitution | `eine` | `bitte` | …an mich ja [___] meinen sie dass… |
| 1620 | Substitution | `meinen` | `beugen` | …mich ja eine [___] sie dass es… |
| 1621 | Löschung | `dass` | `*(nicht da)*` | …eine meinen sie [___] es so sehr… |
| 1622 | Löschung | `so` | `*(nicht da)*` | …sie dass es [___] sehr schlimm wird… |
| 1623 | Löschung | `sehr` | `*(nicht da)*` | …dass es so [___] schlimm wird oder… |
| 1624 | Löschung | `schlimm` | `*(nicht da)*` | …es so sehr [___] wird oder meinen… |
| 1625 | Löschung | `wird` | `*(nicht da)*` | …so sehr schlimm [___] oder meinen sie… |
| 1626 | Löschung | `oder` | `*(nicht da)*` | …sehr schlimm wird [___] meinen sie dass… |
| 1627 | Löschung | `meinen` | `*(nicht da)*` | …schlimm wird oder [___] sie dass ich… |
| 1628 | Löschung | `sie` | `*(nicht da)*` | …wird oder meinen [___] dass ich morgen… |
| 1629 | Löschung | `dass` | `*(nicht da)*` | …oder meinen sie [___] ich morgen wieder… |
| 1630 | Löschung | `ich` | `*(nicht da)*` | …meinen sie dass [___] morgen wieder arbeiten… |
| 1631 | Löschung | `morgen` | `*(nicht da)*` | …sie dass ich [___] wieder arbeiten gehen… |
| 1632 | Löschung | `wieder` | `*(nicht da)*` | …dass ich morgen [___] arbeiten gehen kann… |
| 1633 | Löschung | `arbeiten` | `*(nicht da)*` | …ich morgen wieder [___] gehen kann weil… |
| 1634 | Löschung | `gehen` | `*(nicht da)*` | …morgen wieder arbeiten [___] kann weil ich… |
| 1635 | Löschung | `kann` | `*(nicht da)*` | …wieder arbeiten gehen [___] weil ich ein… |
| 1636 | Löschung | `weil` | `*(nicht da)*` | …arbeiten gehen kann [___] ich ein sehr… |
| 1637 | Löschung | `ich` | `*(nicht da)*` | …gehen kann weil [___] ein sehr wichtiges… |
| 1638 | Löschung | `ein` | `*(nicht da)*` | …kann weil ich [___] sehr wichtiges meeting… |
| 1639 | Löschung | `sehr` | `*(nicht da)*` | …weil ich ein [___] wichtiges meeting habe… |
| 1640 | Löschung | `wichtiges` | `*(nicht da)*` | …ich ein sehr [___] meeting habe morgen… |
| 1641 | Löschung | `meeting` | `*(nicht da)*` | …ein sehr wichtiges [___] habe morgen und… |
| 1642 | Löschung | `habe` | `*(nicht da)*` | …sehr wichtiges meeting [___] morgen und wenn… |
| 1643 | Löschung | `morgen` | `*(nicht da)*` | …wichtiges meeting habe [___] und wenn ich… |
| 1644 | Löschung | `und` | `*(nicht da)*` | …meeting habe morgen [___] wenn ich nicht… |
| 1645 | Löschung | `wenn` | `*(nicht da)*` | …habe morgen und [___] ich nicht selbst… |
| 1646 | Löschung | `ich` | `*(nicht da)*` | …morgen und wenn [___] nicht selbst hingehen… |
| 1647 | Löschung | `nicht` | `*(nicht da)*` | …und wenn ich [___] selbst hingehen kann… |
| 1648 | Löschung | `selbst` | `*(nicht da)*` | …wenn ich nicht [___] hingehen kann dann… |
| 1649 | Löschung | `hingehen` | `*(nicht da)*` | …ich nicht selbst [___] kann dann müsste… |
| 1650 | Löschung | `kann` | `*(nicht da)*` | …nicht selbst hingehen [___] dann müsste ich… |
| 1651 | Löschung | `dann` | `*(nicht da)*` | …selbst hingehen kann [___] müsste ich mich… |
| 1652 | Substitution | `müsste` | `langsam` | …hingehen kann dann [___] ich mich darum… |
| 1653 | Substitution | `ich` | `beugt` | …kann dann müsste [___] mich darum kümmern… |
| 1654 | Substitution | `mich` | `das` | …dann müsste ich [___] darum kümmern und… |
| 1655 | Substitution | `darum` | `bein` | …müsste ich mich [___] kümmern und eine… |
| 1656 | Substitution | `kümmern` | `gut` | …ich mich darum [___] und eine vertretung… |
| 1657 | Löschung | `eine` | `*(nicht da)*` | …darum kümmern und [___] vertretung finden ich… |
| 1658 | Löschung | `vertretung` | `*(nicht da)*` | …kümmern und eine [___] finden ich werde… |
| 1659 | Löschung | `finden` | `*(nicht da)*` | …und eine vertretung [___] ich werde ihnen… |
| 1660 | Löschung | `ich` | `*(nicht da)*` | …eine vertretung finden [___] werde ihnen jetzt… |
| 1661 | Löschung | `werde` | `*(nicht da)*` | …vertretung finden ich [___] ihnen jetzt etwas… |
| 1662 | Löschung | `ihnen` | `*(nicht da)*` | …finden ich werde [___] jetzt etwas sagen… |
| 1663 | Löschung | `etwas` | `*(nicht da)*` | …werde ihnen jetzt [___] sagen was sie… |
| 1664 | Substitution | `sagen` | `bitte` | …ihnen jetzt etwas [___] was sie wahrscheinlich… |
| 1665 | Substitution | `was` | `wechseln` | …jetzt etwas sagen [___] sie wahrscheinlich nicht… |
| 1666 | Löschung | `wahrscheinlich` | `*(nicht da)*` | …sagen was sie [___] nicht so gerne… |
| 1667 | Löschung | `nicht` | `*(nicht da)*` | …was sie wahrscheinlich [___] so gerne hören… |
| 1668 | Löschung | `so` | `*(nicht da)*` | …sie wahrscheinlich nicht [___] gerne hören möchten… |
| 1669 | Löschung | `gerne` | `*(nicht da)*` | …wahrscheinlich nicht so [___] hören möchten aber… |
| 1670 | Löschung | `hören` | `*(nicht da)*` | …nicht so gerne [___] möchten aber da… |
| 1671 | Löschung | `möchten` | `*(nicht da)*` | …so gerne hören [___] aber da kann… |
| 1672 | Löschung | `aber` | `*(nicht da)*` | …gerne hören möchten [___] da kann ich… |
| 1673 | Löschung | `da` | `*(nicht da)*` | …hören möchten aber [___] kann ich ihnen… |
| 1674 | Löschung | `kann` | `*(nicht da)*` | …möchten aber da [___] ich ihnen leider… |
| 1675 | Löschung | `ich` | `*(nicht da)*` | …aber da kann [___] ihnen leider gerade… |
| 1676 | Löschung | `ihnen` | `*(nicht da)*` | …da kann ich [___] leider gerade noch… |
| 1677 | Löschung | `leider` | `*(nicht da)*` | …kann ich ihnen [___] gerade noch keinerlei… |
| 1678 | Löschung | `gerade` | `*(nicht da)*` | …ich ihnen leider [___] noch keinerlei positive… |
| 1679 | Löschung | `noch` | `*(nicht da)*` | …ihnen leider gerade [___] keinerlei positive auskunft… |
| 1680 | Löschung | `keinerlei` | `*(nicht da)*` | …leider gerade noch [___] positive auskunft drüber… |
| 1681 | Löschung | `positive` | `*(nicht da)*` | …gerade noch keinerlei [___] auskunft drüber geben… |
| 1682 | Löschung | `auskunft` | `*(nicht da)*` | …noch keinerlei positive [___] drüber geben das… |
| 1683 | Löschung | `drüber` | `*(nicht da)*` | …keinerlei positive auskunft [___] geben das was… |
| 1684 | Löschung | `geben` | `*(nicht da)*` | …positive auskunft drüber [___] das was sie… |
| 1685 | Löschung | `was` | `*(nicht da)*` | …drüber geben das [___] sie beschreiben bezüglich… |
| 1686 | Löschung | `sie` | `*(nicht da)*` | …geben das was [___] beschreiben bezüglich ihres… |
| 1687 | Löschung | `beschreiben` | `*(nicht da)*` | …das was sie [___] bezüglich ihres knies… |
| 1688 | Löschung | `bezüglich` | `*(nicht da)*` | …was sie beschreiben [___] ihres knies und… |
| 1689 | Löschung | `ihres` | `*(nicht da)*` | …sie beschreiben bezüglich [___] knies und auch… |
| 1690 | Löschung | `knies` | `*(nicht da)*` | …beschreiben bezüglich ihres [___] und auch ihres… |
| 1691 | Löschung | `und` | `*(nicht da)*` | …bezüglich ihres knies [___] auch ihres daumens… |
| 1692 | Löschung | `auch` | `*(nicht da)*` | …ihres knies und [___] ihres daumens da… |
| 1693 | Substitution | `ihres` | `bein` | …knies und auch [___] daumens da müssen… |
| 1694 | Substitution | `daumens` | `wechselt` | …und auch ihres [___] da müssen wir… |
| 1695 | Substitution | `da` | `das` | …auch ihres daumens [___] müssen wir wirklich… |
| 1696 | Substitution | `müssen` | `bein` | …ihres daumens da [___] wir wirklich erst… |
| 1697 | Substitution | `wirklich` | `machen` | …da müssen wir [___] erst mal mrt… |
| 1698 | Substitution | `erst` | `das` | …müssen wir wirklich [___] mal mrt bilder… |
| 1699 | Substitution | `mrt` | `langsam` | …wirklich erst mal [___] bilder von machen… |
| 1700 | Substitution | `bilder` | `ich` | …erst mal mrt [___] von machen und… |
| 1701 | Substitution | `von` | `werde` | …mal mrt bilder [___] machen und auch… |
| 1702 | Substitution | `machen` | `jetzt` | …mrt bilder von [___] und auch röntgenbilder… |
| 1703 | Substitution | `und` | `vorsichtig` | …bilder von machen [___] auch röntgenbilder von… |
| 1704 | Substitution | `auch` | `ein` | …von machen und [___] röntgenbilder von machen… |
| 1705 | Substitution | `röntgenbilder` | `paar` | …machen und auch [___] von machen um… |
| 1706 | Substitution | `von` | `bewegungen` | …und auch röntgenbilder [___] machen um wirklich… |
| 1707 | Löschung | `um` | `*(nicht da)*` | …röntgenbilder von machen [___] wirklich zu sehen… |
| 1708 | Löschung | `wirklich` | `*(nicht da)*` | …von machen um [___] zu sehen was… |
| 1709 | Löschung | `zu` | `*(nicht da)*` | …machen um wirklich [___] sehen was da… |
| 1710 | Löschung | `sehen` | `*(nicht da)*` | …um wirklich zu [___] was da los… |
| 1711 | Löschung | `was` | `*(nicht da)*` | …wirklich zu sehen [___] da los ist… |
| 1712 | Löschung | `da` | `*(nicht da)*` | …zu sehen was [___] los ist nicht… |
| 1713 | Löschung | `los` | `*(nicht da)*` | …sehen was da [___] ist nicht dass… |
| 1714 | Löschung | `ist` | `*(nicht da)*` | …was da los [___] nicht dass sie… |
| 1715 | Löschung | `nicht` | `*(nicht da)*` | …da los ist [___] dass sie sich… |
| 1716 | Löschung | `dass` | `*(nicht da)*` | …los ist nicht [___] sie sich etwas… |
| 1717 | Löschung | `sie` | `*(nicht da)*` | …ist nicht dass [___] sich etwas gebrochen… |
| 1718 | Löschung | `sich` | `*(nicht da)*` | …nicht dass sie [___] etwas gebrochen oder… |
| 1719 | Löschung | `etwas` | `*(nicht da)*` | …dass sie sich [___] gebrochen oder gerissen… |
| 1720 | Löschung | `gebrochen` | `*(nicht da)*` | …sie sich etwas [___] oder gerissen haben… |
| 1721 | Löschung | `oder` | `*(nicht da)*` | …sich etwas gebrochen [___] gerissen haben sollte… |
| 1722 | Löschung | `gerissen` | `*(nicht da)*` | …etwas gebrochen oder [___] haben sollte das… |
| 1723 | Löschung | `haben` | `*(nicht da)*` | …gebrochen oder gerissen [___] sollte das der… |
| 1724 | Löschung | `sollte` | `*(nicht da)*` | …oder gerissen haben [___] das der fall… |
| 1725 | Löschung | `das` | `*(nicht da)*` | …gerissen haben sollte [___] der fall sein… |
| 1726 | Löschung | `der` | `*(nicht da)*` | …haben sollte das [___] fall sein muss… |
| 1727 | Löschung | `fall` | `*(nicht da)*` | …sollte das der [___] sein muss man… |
| 1728 | Löschung | `sein` | `*(nicht da)*` | …das der fall [___] muss man abwägen… |
| 1729 | Löschung | `muss` | `*(nicht da)*` | …der fall sein [___] man abwägen ob… |
| 1730 | Löschung | `man` | `*(nicht da)*` | …fall sein muss [___] abwägen ob sie… |
| 1731 | Substitution | `abwägen` | `bitte` | …sein muss man [___] ob sie operieren… |
| 1732 | Substitution | `ob` | `sagen` | …muss man abwägen [___] sie operieren oder… |
| 1733 | Löschung | `operieren` | `*(nicht da)*` | …abwägen ob sie [___] oder nicht das… |
| 1734 | Löschung | `oder` | `*(nicht da)*` | …ob sie operieren [___] nicht das können… |
| 1735 | Löschung | `nicht` | `*(nicht da)*` | …sie operieren oder [___] das können wir… |
| 1736 | Löschung | `das` | `*(nicht da)*` | …operieren oder nicht [___] können wir aber… |
| 1737 | Löschung | `können` | `*(nicht da)*` | …oder nicht das [___] wir aber alles… |
| 1738 | Löschung | `wir` | `*(nicht da)*` | …nicht das können [___] aber alles erst… |
| 1739 | Löschung | `aber` | `*(nicht da)*` | …das können wir [___] alles erst sagen… |
| 1740 | Löschung | `alles` | `*(nicht da)*` | …können wir aber [___] erst sagen wenn… |
| 1741 | Substitution | `erst` | `mir` | …wir aber alles [___] sagen wenn wir… |
| 1742 | Substitution | `sagen` | `bescheid` | …aber alles erst [___] wenn wir die… |
| 1743 | Löschung | `wir` | `*(nicht da)*` | …erst sagen wenn [___] die befunde da… |
| 1744 | Löschung | `die` | `*(nicht da)*` | …sagen wenn wir [___] befunde da haben… |
| 1745 | Löschung | `befunde` | `*(nicht da)*` | …wenn wir die [___] da haben ich… |
| 1746 | Löschung | `da` | `*(nicht da)*` | …wir die befunde [___] haben ich würde… |
| 1747 | Löschung | `haben` | `*(nicht da)*` | …die befunde da [___] ich würde auch… |
| 1748 | Löschung | `ich` | `*(nicht da)*` | …befunde da haben [___] würde auch gerne… |
| 1749 | Löschung | `würde` | `*(nicht da)*` | …da haben ich [___] auch gerne nochmal… |
| 1750 | Löschung | `auch` | `*(nicht da)*` | …haben ich würde [___] gerne nochmal eine… |
| 1751 | Löschung | `gerne` | `*(nicht da)*` | …ich würde auch [___] nochmal eine untersuchung… |
| 1752 | Löschung | `nochmal` | `*(nicht da)*` | …würde auch gerne [___] eine untersuchung mit… |
| 1753 | Löschung | `eine` | `*(nicht da)*` | …auch gerne nochmal [___] untersuchung mit ihrem… |
| 1754 | Löschung | `untersuchung` | `*(nicht da)*` | …gerne nochmal eine [___] mit ihrem kopf… |
| 1755 | Löschung | `mit` | `*(nicht da)*` | …nochmal eine untersuchung [___] ihrem kopf durchführen… |
| 1756 | Löschung | `ihrem` | `*(nicht da)*` | …eine untersuchung mit [___] kopf durchführen um… |
| 1757 | Löschung | `kopf` | `*(nicht da)*` | …untersuchung mit ihrem [___] durchführen um dort… |
| 1758 | Löschung | `durchführen` | `*(nicht da)*` | …mit ihrem kopf [___] um dort auch… |
| 1759 | Löschung | `um` | `*(nicht da)*` | …ihrem kopf durchführen [___] dort auch zu… |
| 1760 | Löschung | `dort` | `*(nicht da)*` | …kopf durchführen um [___] auch zu checken… |
| 1761 | Substitution | `auch` | `es` | …durchführen um dort [___] zu checken dass… |
| 1762 | Löschung | `checken` | `*(nicht da)*` | …dort auch zu [___] dass da eben… |
| 1763 | Löschung | `dass` | `*(nicht da)*` | …auch zu checken [___] da eben alles… |
| 1764 | Löschung | `da` | `*(nicht da)*` | …zu checken dass [___] eben alles in… |
| 1765 | Löschung | `eben` | `*(nicht da)*` | …checken dass da [___] alles in ordnung… |
| 1766 | Löschung | `alles` | `*(nicht da)*` | …dass da eben [___] in ordnung ist… |
| 1767 | Löschung | `in` | `*(nicht da)*` | …da eben alles [___] ordnung ist und… |
| 1768 | Löschung | `ordnung` | `*(nicht da)*` | …eben alles in [___] ist und ja… |
| 1769 | Löschung | `ist` | `*(nicht da)*` | …alles in ordnung [___] und ja wenn… |
| 1770 | Löschung | `und` | `*(nicht da)*` | …in ordnung ist [___] ja wenn alles… |
| 1771 | Löschung | `ja` | `*(nicht da)*` | …ordnung ist und [___] wenn alles in… |
| 1772 | Löschung | `wenn` | `*(nicht da)*` | …ist und ja [___] alles in ordnung… |
| 1773 | Löschung | `alles` | `*(nicht da)*` | …und ja wenn [___] in ordnung ist… |
| 1774 | Löschung | `in` | `*(nicht da)*` | …ja wenn alles [___] ordnung ist würde… |
| 1775 | Substitution | `ordnung` | `stark` | …wenn alles in [___] ist würde ich… |
| 1776 | Löschung | `würde` | `*(nicht da)*` | …in ordnung ist [___] ich ihnen trotzdem… |
| 1777 | Löschung | `ich` | `*(nicht da)*` | …ordnung ist würde [___] ihnen trotzdem raten… |
| 1778 | Löschung | `ihnen` | `*(nicht da)*` | …ist würde ich [___] trotzdem raten das… |
| 1779 | Löschung | `trotzdem` | `*(nicht da)*` | …würde ich ihnen [___] raten das meeting… |
| 1780 | Löschung | `raten` | `*(nicht da)*` | …ich ihnen trotzdem [___] das meeting vielleicht… |
| 1781 | Löschung | `das` | `*(nicht da)*` | …ihnen trotzdem raten [___] meeting vielleicht morgen… |
| 1782 | Löschung | `meeting` | `*(nicht da)*` | …trotzdem raten das [___] vielleicht morgen einmal… |
| 1783 | Löschung | `vielleicht` | `*(nicht da)*` | …raten das meeting [___] morgen einmal online… |
| 1784 | Löschung | `morgen` | `*(nicht da)*` | …das meeting vielleicht [___] einmal online durchzuführen… |
| 1785 | Löschung | `einmal` | `*(nicht da)*` | …meeting vielleicht morgen [___] online durchzuführen damit… |
| 1786 | Löschung | `online` | `*(nicht da)*` | …vielleicht morgen einmal [___] durchzuführen damit sie… |
| 1787 | Löschung | `durchzuführen` | `*(nicht da)*` | …morgen einmal online [___] damit sie sich… |
| 1788 | Löschung | `damit` | `*(nicht da)*` | …einmal online durchzuführen [___] sie sich schonen… |
| 1789 | Löschung | `sie` | `*(nicht da)*` | …online durchzuführen damit [___] sich schonen können… |
| 1790 | Löschung | `sich` | `*(nicht da)*` | …durchzuführen damit sie [___] schonen können aber… |
| 1791 | Löschung | `schonen` | `*(nicht da)*` | …damit sie sich [___] können aber genauere… |
| 1792 | Löschung | `können` | `*(nicht da)*` | …sie sich schonen [___] aber genauere auskunft… |
| 1793 | Löschung | `aber` | `*(nicht da)*` | …sich schonen können [___] genauere auskunft wie… |
| 1794 | Löschung | `genauere` | `*(nicht da)*` | …schonen können aber [___] auskunft wie bereits… |
| 1795 | Löschung | `auskunft` | `*(nicht da)*` | …können aber genauere [___] wie bereits gesagt… |
| 1796 | Löschung | `wie` | `*(nicht da)*` | …aber genauere auskunft [___] bereits gesagt kann… |
| 1797 | Löschung | `bereits` | `*(nicht da)*` | …genauere auskunft wie [___] gesagt kann ich… |
| 1798 | Löschung | `gesagt` | `*(nicht da)*` | …auskunft wie bereits [___] kann ich ihnen… |
| 1799 | Löschung | `kann` | `*(nicht da)*` | …wie bereits gesagt [___] ich ihnen erst… |
| 1800 | Löschung | `ich` | `*(nicht da)*` | …bereits gesagt kann [___] ihnen erst geben… |
| 1801 | Löschung | `ihnen` | `*(nicht da)*` | …gesagt kann ich [___] erst geben wenn… |
| 1802 | Löschung | `erst` | `*(nicht da)*` | …kann ich ihnen [___] geben wenn wir… |
| 1803 | Löschung | `geben` | `*(nicht da)*` | …ich ihnen erst [___] wenn wir alle… |
| 1804 | Löschung | `wenn` | `*(nicht da)*` | …ihnen erst geben [___] wir alle befunde… |
| 1805 | Löschung | `wir` | `*(nicht da)*` | …erst geben wenn [___] alle befunde da… |
| 1806 | Löschung | `alle` | `*(nicht da)*` | …geben wenn wir [___] befunde da haben… |
| 1807 | Löschung | `befunde` | `*(nicht da)*` | …wenn wir alle [___] da haben okay… |
| 1808 | Löschung | `da` | `*(nicht da)*` | …wir alle befunde [___] haben okay ich… |
| 1809 | Löschung | `haben` | `*(nicht da)*` | …alle befunde da [___] okay ich danke… |
| 1810 | Löschung | `ich` | `*(nicht da)*` | …da haben okay [___] danke ihnen online… |
| 1811 | Löschung | `danke` | `*(nicht da)*` | …haben okay ich [___] ihnen online wäre… |
| 1812 | Löschung | `ihnen` | `*(nicht da)*` | …okay ich danke [___] online wäre schwierig… |
| 1813 | Löschung | `online` | `*(nicht da)*` | …ich danke ihnen [___] wäre schwierig aber… |
| 1814 | Löschung | `wäre` | `*(nicht da)*` | …danke ihnen online [___] schwierig aber dann… |
| 1815 | Löschung | `schwierig` | `*(nicht da)*` | …ihnen online wäre [___] aber dann werde… |
| 1816 | Löschung | `aber` | `*(nicht da)*` | …online wäre schwierig [___] dann werde ich… |
| 1817 | Löschung | `dann` | `*(nicht da)*` | …wäre schwierig aber [___] werde ich jetzt… |
| 1818 | Löschung | `werde` | `*(nicht da)*` | …schwierig aber dann [___] ich jetzt gleich… |
| 1819 | Löschung | `jetzt` | `*(nicht da)*` | …dann werde ich [___] gleich dafür sorgen… |
| 1820 | Löschung | `gleich` | `*(nicht da)*` | …werde ich jetzt [___] dafür sorgen dass… |
| 1821 | Löschung | `dafür` | `*(nicht da)*` | …ich jetzt gleich [___] sorgen dass mich… |
| 1822 | Löschung | `sorgen` | `*(nicht da)*` | …jetzt gleich dafür [___] dass mich jemand… |
| 1823 | Löschung | `dass` | `*(nicht da)*` | …gleich dafür sorgen [___] mich jemand vertritt… |
| 1824 | Löschung | `mich` | `*(nicht da)*` | …dafür sorgen dass [___] jemand vertritt okay… |
| 1825 | Löschung | `jemand` | `*(nicht da)*` | …sorgen dass mich [___] vertritt okay sehr… |
| 1826 | Löschung | `vertritt` | `*(nicht da)*` | …dass mich jemand [___] okay sehr gut… |
| 1827 | Substitution | `okay` | `achte` | …mich jemand vertritt [___] sehr gut wir… |
| 1828 | Substitution | `sehr` | `darauf` | …jemand vertritt okay [___] gut wir werden… |
| 1829 | Löschung | `werden` | `*(nicht da)*` | …sehr gut wir [___] auch gleich die… |
| 1830 | Löschung | `auch` | `*(nicht da)*` | …gut wir werden [___] gleich die untersuchungen… |
| 1831 | Löschung | `gleich` | `*(nicht da)*` | …wir werden auch [___] die untersuchungen direkt… |
| 1832 | Löschung | `die` | `*(nicht da)*` | …werden auch gleich [___] untersuchungen direkt durchführen… |
| 1833 | Löschung | `untersuchungen` | `*(nicht da)*` | …auch gleich die [___] direkt durchführen wenn… |
| 1834 | Substitution | `direkt` | `machen` | …gleich die untersuchungen [___] durchführen wenn wir… |
| 1835 | Substitution | `durchführen` | `das` | …die untersuchungen direkt [___] wenn wir mit… |
| 1836 | Substitution | `wenn` | `jetzt` | …untersuchungen direkt durchführen [___] wir mit der… |
| 1837 | Substitution | `wir` | `mal` | …direkt durchführen wenn [___] mit der aufnahme… |
| 1838 | Löschung | `der` | `*(nicht da)*` | …wenn wir mit [___] aufnahme fertig sind… |
| 1839 | Löschung | `aufnahme` | `*(nicht da)*` | …wir mit der [___] fertig sind dann… |
| 1840 | Löschung | `fertig` | `*(nicht da)*` | …mit der aufnahme [___] sind dann würde… |
| 1841 | Löschung | `sind` | `*(nicht da)*` | …der aufnahme fertig [___] dann würde ich… |
| 1842 | Löschung | `dann` | `*(nicht da)*` | …aufnahme fertig sind [___] würde ich sie… |
| 1843 | Löschung | `würde` | `*(nicht da)*` | …fertig sind dann [___] ich sie bitten… |
| 1844 | Löschung | `ich` | `*(nicht da)*` | …sind dann würde [___] sie bitten schon… |
| 1845 | Löschung | `sie` | `*(nicht da)*` | …dann würde ich [___] bitten schon mal… |
| 1846 | Löschung | `bitten` | `*(nicht da)*` | …würde ich sie [___] schon mal rüber… |
| 1847 | Löschung | `schon` | `*(nicht da)*` | …ich sie bitten [___] mal rüber ins… |
| 1848 | Substitution | `mal` | `dem` | …sie bitten schon [___] rüber ins untersuchungszimmer… |
| 1849 | Substitution | `rüber` | `rechten` | …bitten schon mal [___] ins untersuchungszimmer zu… |
| 1850 | Substitution | `ins` | `knie` | …schon mal rüber [___] untersuchungszimmer zu gehen… |
| 1851 | Substitution | `untersuchungszimmer` | `hört` | …mal rüber ins [___] zu gehen und… |
| 1852 | Löschung | `gehen` | `*(nicht da)*` | …ins untersuchungszimmer zu [___] und dann geht… |
| 1853 | Löschung | `und` | `*(nicht da)*` | …untersuchungszimmer zu gehen [___] dann geht es… |
| 1854 | Löschung | `dann` | `*(nicht da)*` | …zu gehen und [___] geht es dort… |
| 1855 | Löschung | `geht` | `*(nicht da)*` | …gehen und dann [___] es dort auch… |
| 1856 | Löschung | `es` | `*(nicht da)*` | …und dann geht [___] dort auch gleich… |
| 1857 | Löschung | `dort` | `*(nicht da)*` | …dann geht es [___] auch gleich los… |
| 1858 | Löschung | `auch` | `*(nicht da)*` | …geht es dort [___] gleich los ja… |
| 1859 | Löschung | `gleich` | `*(nicht da)*` | …es dort auch [___] los ja alles… |
| 1860 | Löschung | `los` | `*(nicht da)*` | …dort auch gleich [___] ja alles klar… |
| 1861 | Löschung | `ja` | `*(nicht da)*` | …auch gleich los [___] alles klar noch… |
| 1862 | Löschung | `alles` | `*(nicht da)*` | …gleich los ja [___] klar noch mal… |
| 1863 | Löschung | `klar` | `*(nicht da)*` | …los ja alles [___] noch mal kurz… |
| 1864 | Löschung | `noch` | `*(nicht da)*` | …ja alles klar [___] mal kurz zum… |
| 1865 | Löschung | `mal` | `*(nicht da)*` | …alles klar noch [___] kurz zum abgleich… |
| 1866 | Löschung | `kurz` | `*(nicht da)*` | …klar noch mal [___] zum abgleich sie… |
| 1867 | Löschung | `zum` | `*(nicht da)*` | …noch mal kurz [___] abgleich sie hatten… |
| 1868 | Löschung | `abgleich` | `*(nicht da)*` | …mal kurz zum [___] sie hatten einen… |
| 1869 | Löschung | `sie` | `*(nicht da)*` | …kurz zum abgleich [___] hatten einen fahrradunfall… |
| 1870 | Löschung | `hatten` | `*(nicht da)*` | …zum abgleich sie [___] einen fahrradunfall sind… |
| 1871 | Löschung | `einen` | `*(nicht da)*` | …abgleich sie hatten [___] fahrradunfall sind auf… |
| 1872 | Löschung | `fahrradunfall` | `*(nicht da)*` | …sie hatten einen [___] sind auf die… |
| 1873 | Löschung | `sind` | `*(nicht da)*` | …hatten einen fahrradunfall [___] auf die linke… |
| 1874 | Löschung | `auf` | `*(nicht da)*` | …einen fahrradunfall sind [___] die linke seite… |
| 1875 | Löschung | `die` | `*(nicht da)*` | …fahrradunfall sind auf [___] linke seite gestürzt… |
| 1876 | Löschung | `linke` | `*(nicht da)*` | …sind auf die [___] seite gestürzt und… |
| 1877 | Löschung | `seite` | `*(nicht da)*` | …auf die linke [___] gestürzt und haben… |
| 1878 | Löschung | `gestürzt` | `*(nicht da)*` | …die linke seite [___] und haben seitdem… |
| 1879 | Löschung | `haben` | `*(nicht da)*` | …seite gestürzt und [___] seitdem schmerzen auf… |
| 1880 | Löschung | `seitdem` | `*(nicht da)*` | …gestürzt und haben [___] schmerzen auf der… |
| 1881 | Löschung | `schmerzen` | `*(nicht da)*` | …und haben seitdem [___] auf der linken… |
| 1882 | Löschung | `auf` | `*(nicht da)*` | …haben seitdem schmerzen [___] der linken kopfseite… |
| 1883 | Löschung | `der` | `*(nicht da)*` | …seitdem schmerzen auf [___] linken kopfseite im… |
| 1884 | Löschung | `linken` | `*(nicht da)*` | …schmerzen auf der [___] kopfseite im linken… |
| 1885 | Löschung | `kopfseite` | `*(nicht da)*` | …auf der linken [___] im linken daumen… |
| 1886 | Löschung | `im` | `*(nicht da)*` | …der linken kopfseite [___] linken daumen sowie… |
| 1887 | Löschung | `linken` | `*(nicht da)*` | …linken kopfseite im [___] daumen sowie im… |
| 1888 | Substitution | `daumen` | `jetzt` | …kopfseite im linken [___] sowie im linken… |
| 1889 | Substitution | `sowie` | `mit` | …im linken daumen [___] im linken knie… |
| 1890 | Substitution | `im` | `dem` | …linken daumen sowie [___] linken knie die… |
| 1891 | Löschung | `die` | `*(nicht da)*` | …im linken knie [___] schmerzen im kopf… |
| 1892 | Löschung | `schmerzen` | `*(nicht da)*` | …linken knie die [___] im kopf sind… |
| 1893 | Löschung | `im` | `*(nicht da)*` | …knie die schmerzen [___] kopf sind sehr… |
| 1894 | Löschung | `kopf` | `*(nicht da)*` | …die schmerzen im [___] sind sehr leicht… |
| 1895 | Löschung | `sind` | `*(nicht da)*` | …schmerzen im kopf [___] sehr leicht die… |
| 1896 | Löschung | `sehr` | `*(nicht da)*` | …im kopf sind [___] leicht die schmerzen… |
| 1897 | Löschung | `leicht` | `*(nicht da)*` | …kopf sind sehr [___] die schmerzen im… |
| 1898 | Löschung | `die` | `*(nicht da)*` | …sind sehr leicht [___] schmerzen im daumen… |
| 1899 | Löschung | `schmerzen` | `*(nicht da)*` | …sehr leicht die [___] im daumen dagegen… |
| 1900 | Löschung | `im` | `*(nicht da)*` | …leicht die schmerzen [___] daumen dagegen schon… |
| 1901 | Löschung | `daumen` | `*(nicht da)*` | …die schmerzen im [___] dagegen schon wesentlich… |
| 1902 | Löschung | `dagegen` | `*(nicht da)*` | …schmerzen im daumen [___] schon wesentlich stärker… |
| 1903 | Löschung | `schon` | `*(nicht da)*` | …im daumen dagegen [___] wesentlich stärker sie… |
| 1904 | Löschung | `wesentlich` | `*(nicht da)*` | …daumen dagegen schon [___] stärker sie haben… |
| 1905 | Löschung | `stärker` | `*(nicht da)*` | …dagegen schon wesentlich [___] sie haben die… |
| 1906 | Löschung | `sie` | `*(nicht da)*` | …schon wesentlich stärker [___] haben die schmerzintensität… |
| 1907 | Löschung | `haben` | `*(nicht da)*` | …wesentlich stärker sie [___] die schmerzintensität dort… |
| 1908 | Löschung | `die` | `*(nicht da)*` | …stärker sie haben [___] schmerzintensität dort mit… |
| 1909 | Löschung | `schmerzintensität` | `*(nicht da)*` | …sie haben die [___] dort mit einer… |
| 1910 | Löschung | `dort` | `*(nicht da)*` | …haben die schmerzintensität [___] mit einer 7… |
| 1911 | Löschung | `mit` | `*(nicht da)*` | …die schmerzintensität dort [___] einer 7 beschrieben… |
| 1912 | Löschung | `einer` | `*(nicht da)*` | …schmerzintensität dort mit [___] 7 beschrieben und… |
| 1913 | Löschung | `7` | `*(nicht da)*` | …dort mit einer [___] beschrieben und haben… |
| 1914 | Löschung | `beschrieben` | `*(nicht da)*` | …mit einer 7 [___] und haben gesagt… |
| 1915 | Löschung | `und` | `*(nicht da)*` | …einer 7 beschrieben [___] haben gesagt dass… |
| 1916 | Löschung | `haben` | `*(nicht da)*` | …7 beschrieben und [___] gesagt dass sie… |
| 1917 | Löschung | `gesagt` | `*(nicht da)*` | …beschrieben und haben [___] dass sie den… |
| 1918 | Löschung | `dass` | `*(nicht da)*` | …und haben gesagt [___] sie den daumen… |
| 1919 | Löschung | `sie` | `*(nicht da)*` | …haben gesagt dass [___] den daumen auch… |
| 1920 | Löschung | `den` | `*(nicht da)*` | …gesagt dass sie [___] daumen auch nicht… |
| 1921 | Löschung | `daumen` | `*(nicht da)*` | …dass sie den [___] auch nicht mehr… |
| 1922 | Löschung | `auch` | `*(nicht da)*` | …sie den daumen [___] nicht mehr bewegen… |
| 1923 | Löschung | `nicht` | `*(nicht da)*` | …den daumen auch [___] mehr bewegen können… |
| 1924 | Löschung | `mehr` | `*(nicht da)*` | …daumen auch nicht [___] bewegen können der… |
| 1925 | Löschung | `bewegen` | `*(nicht da)*` | …auch nicht mehr [___] können der schmerz… |
| 1926 | Löschung | `können` | `*(nicht da)*` | …nicht mehr bewegen [___] der schmerz wurde… |
| 1927 | Löschung | `der` | `*(nicht da)*` | …mehr bewegen können [___] schmerz wurde stechend… |
| 1928 | Löschung | `schmerz` | `*(nicht da)*` | …bewegen können der [___] wurde stechend beschrieben… |
| 1929 | Löschung | `wurde` | `*(nicht da)*` | …können der schmerz [___] stechend beschrieben und… |
| 1930 | Löschung | `stechend` | `*(nicht da)*` | …der schmerz wurde [___] beschrieben und gleiches… |
| 1931 | Löschung | `beschrieben` | `*(nicht da)*` | …schmerz wurde stechend [___] und gleiches gilt… |
| 1932 | Löschung | `und` | `*(nicht da)*` | …wurde stechend beschrieben [___] gleiches gilt für… |
| 1933 | Löschung | `gleiches` | `*(nicht da)*` | …stechend beschrieben und [___] gilt für das… |
| 1934 | Substitution | `gilt` | `hört` | …beschrieben und gleiches [___] für das knie… |
| 1935 | Substitution | `für` | `zu` | …und gleiches gilt [___] das knie auch… |
| 1936 | Substitution | `das` | `okay` | …gleiches gilt für [___] knie auch das… |
| 1937 | Substitution | `knie` | `wir` | …gilt für das [___] auch das knie… |
| 1938 | Substitution | `auch` | `machen` | …für das knie [___] das knie können… |
| 1939 | Löschung | `knie` | `*(nicht da)*` | …knie auch das [___] können sie nicht… |
| 1940 | Löschung | `können` | `*(nicht da)*` | …auch das knie [___] sie nicht mehr… |
| 1941 | Löschung | `sie` | `*(nicht da)*` | …das knie können [___] nicht mehr bewegen… |
| 1942 | Löschung | `nicht` | `*(nicht da)*` | …knie können sie [___] mehr bewegen im… |
| 1943 | Löschung | `mehr` | `*(nicht da)*` | …können sie nicht [___] bewegen im ruhezustand… |
| 1944 | Löschung | `bewegen` | `*(nicht da)*` | …sie nicht mehr [___] im ruhezustand wurde… |
| 1945 | Löschung | `im` | `*(nicht da)*` | …nicht mehr bewegen [___] ruhezustand wurde die… |
| 1946 | Löschung | `ruhezustand` | `*(nicht da)*` | …mehr bewegen im [___] wurde die schmerzintensität… |
| 1947 | Löschung | `wurde` | `*(nicht da)*` | …bewegen im ruhezustand [___] die schmerzintensität mit… |
| 1948 | Substitution | `die` | `jetzt` | …im ruhezustand wurde [___] schmerzintensität mit einer… |
| 1949 | Substitution | `schmerzintensität` | `mal` | …ruhezustand wurde die [___] mit einer 8… |
| 1950 | Löschung | `einer` | `*(nicht da)*` | …die schmerzintensität mit [___] 8 beschrieben bei… |
| 1951 | Löschung | `8` | `*(nicht da)*` | …schmerzintensität mit einer [___] beschrieben bei bewegung… |
| 1952 | Löschung | `beschrieben` | `*(nicht da)*` | …mit einer 8 [___] bei bewegung unerträglich… |
| 1953 | Löschung | `bei` | `*(nicht da)*` | …einer 8 beschrieben [___] bewegung unerträglich also… |
| 1954 | Löschung | `bewegung` | `*(nicht da)*` | …8 beschrieben bei [___] unerträglich also 10… |
| 1955 | Löschung | `unerträglich` | `*(nicht da)*` | …beschrieben bei bewegung [___] also 10 oder… |
| 1956 | Löschung | `also` | `*(nicht da)*` | …bei bewegung unerträglich [___] 10 oder mehr… |
| 1957 | Löschung | `10` | `*(nicht da)*` | …bewegung unerträglich also [___] oder mehr als… |
| 1958 | Löschung | `oder` | `*(nicht da)*` | …unerträglich also 10 [___] mehr als 10… |
| 1959 | Löschung | `mehr` | `*(nicht da)*` | …also 10 oder [___] als 10 auch… |
| 1960 | Löschung | `als` | `*(nicht da)*` | …10 oder mehr [___] 10 auch dieser… |
| 1961 | Löschung | `10` | `*(nicht da)*` | …oder mehr als [___] auch dieser schmerz… |
| 1962 | Löschung | `auch` | `*(nicht da)*` | …mehr als 10 [___] dieser schmerz ist… |
| 1963 | Löschung | `dieser` | `*(nicht da)*` | …als 10 auch [___] schmerz ist stechend… |
| 1964 | Löschung | `schmerz` | `*(nicht da)*` | …10 auch dieser [___] ist stechend ansonsten… |
| 1965 | Löschung | `ist` | `*(nicht da)*` | …auch dieser schmerz [___] stechend ansonsten sind… |
| 1966 | Löschung | `stechend` | `*(nicht da)*` | …dieser schmerz ist [___] ansonsten sind daumen… |
| 1967 | Löschung | `ansonsten` | `*(nicht da)*` | …schmerz ist stechend [___] sind daumen wie… |
| 1968 | Löschung | `sind` | `*(nicht da)*` | …ist stechend ansonsten [___] daumen wie auch… |
| 1969 | Löschung | `daumen` | `*(nicht da)*` | …stechend ansonsten sind [___] wie auch knie… |
| 1970 | Löschung | `wie` | `*(nicht da)*` | …ansonsten sind daumen [___] auch knie geschwollen… |
| 1971 | Löschung | `auch` | `*(nicht da)*` | …sind daumen wie [___] knie geschwollen richtig… |
| 1972 | Löschung | `knie` | `*(nicht da)*` | …daumen wie auch [___] geschwollen richtig richtig… |
| 1973 | Substitution | `geschwollen` | `einem` | …wie auch knie [___] richtig richtig sie… |
| 1974 | Substitution | `richtig` | `leichten` | …auch knie geschwollen [___] richtig sie haben… |
| 1975 | Substitution | `richtig` | `druck` | …knie geschwollen richtig [___] sie haben gesagt… |
| 1976 | Substitution | `sie` | `murmelt` | …geschwollen richtig richtig [___] haben gesagt dass… |
| 1977 | Substitution | `haben` | `leise` | …richtig richtig sie [___] gesagt dass sie… |
| 1978 | Substitution | `gesagt` | `gut` | …richtig sie haben [___] dass sie das… |
| 1979 | Substitution | `dass` | `wir` | …sie haben gesagt [___] sie das bewusstsein… |
| 1980 | Substitution | `sie` | `machen` | …haben gesagt dass [___] das bewusstsein nicht… |
| 1981 | Löschung | `bewusstsein` | `*(nicht da)*` | …dass sie das [___] nicht verloren haben… |
| 1982 | Löschung | `nicht` | `*(nicht da)*` | …sie das bewusstsein [___] verloren haben bei… |
| 1983 | Löschung | `verloren` | `*(nicht da)*` | …das bewusstsein nicht [___] haben bei dem… |
| 1984 | Löschung | `haben` | `*(nicht da)*` | …bewusstsein nicht verloren [___] bei dem unfall… |
| 1985 | Löschung | `bei` | `*(nicht da)*` | …nicht verloren haben [___] dem unfall dass… |
| 1986 | Löschung | `dem` | `*(nicht da)*` | …verloren haben bei [___] unfall dass sie… |
| 1987 | Substitution | `unfall` | `jetzt` | …haben bei dem [___] dass sie nur… |
| 1988 | Substitution | `dass` | `mal` | …bei dem unfall [___] sie nur kurz… |
| 1989 | Substitution | `sie` | `mit` | …dem unfall dass [___] nur kurz danach… |
| 1990 | Substitution | `nur` | `einem` | …unfall dass sie [___] kurz danach recht… |
| 1991 | Substitution | `kurz` | `leichten` | …dass sie nur [___] danach recht schwindelig… |
| 1992 | Substitution | `danach` | `zug` | …sie nur kurz [___] recht schwindelig waren… |
| 1993 | Substitution | `recht` | `atmet` | …nur kurz danach [___] schwindelig waren das… |
| 1994 | Substitution | `schwindelig` | `aus` | …kurz danach recht [___] waren das sei… |
| 1995 | Substitution | `waren` | `okay` | …danach recht schwindelig [___] das sei aber… |
| 1996 | Löschung | `sei` | `*(nicht da)*` | …schwindelig waren das [___] aber schon wieder… |
| 1997 | Löschung | `aber` | `*(nicht da)*` | …waren das sei [___] schon wieder vorbei… |
| 1998 | Löschung | `schon` | `*(nicht da)*` | …das sei aber [___] wieder vorbei genau… |
| 1999 | Löschung | `wieder` | `*(nicht da)*` | …sei aber schon [___] vorbei genau vor… |
| 2000 | Löschung | `vorbei` | `*(nicht da)*` | …aber schon wieder [___] genau vor erkrankungen… |
| 2001 | Löschung | `genau` | `*(nicht da)*` | …schon wieder vorbei [___] vor erkrankungen haben… |
| 2002 | Löschung | `vor` | `*(nicht da)*` | …wieder vorbei genau [___] erkrankungen haben sie… |
| 2003 | Löschung | `erkrankungen` | `*(nicht da)*` | …vorbei genau vor [___] haben sie keine… |
| 2004 | Löschung | `haben` | `*(nicht da)*` | …genau vor erkrankungen [___] sie keine medikamente… |
| 2005 | Löschung | `sie` | `*(nicht da)*` | …vor erkrankungen haben [___] keine medikamente nehmen… |
| 2006 | Löschung | `keine` | `*(nicht da)*` | …erkrankungen haben sie [___] medikamente nehmen sie… |
| 2007 | Löschung | `medikamente` | `*(nicht da)*` | …haben sie keine [___] nehmen sie auch… |
| 2008 | Löschung | `nehmen` | `*(nicht da)*` | …sie keine medikamente [___] sie auch keine… |
| 2009 | Löschung | `sie` | `*(nicht da)*` | …keine medikamente nehmen [___] auch keine regelmäßig… |
| 2010 | Löschung | `auch` | `*(nicht da)*` | …medikamente nehmen sie [___] keine regelmäßig ein… |
| 2011 | Löschung | `keine` | `*(nicht da)*` | …nehmen sie auch [___] regelmäßig ein außer… |
| 2012 | Löschung | `regelmäßig` | `*(nicht da)*` | …sie auch keine [___] ein außer der… |
| 2013 | Löschung | `ein` | `*(nicht da)*` | …auch keine regelmäßig [___] außer der pille… |
| 2014 | Löschung | `außer` | `*(nicht da)*` | …keine regelmäßig ein [___] der pille sie… |
| 2015 | Löschung | `der` | `*(nicht da)*` | …regelmäßig ein außer [___] pille sie hatten… |
| 2016 | Löschung | `pille` | `*(nicht da)*` | …ein außer der [___] sie hatten eine… |
| 2017 | Löschung | `sie` | `*(nicht da)*` | …außer der pille [___] hatten eine operation… |
| 2018 | Löschung | `hatten` | `*(nicht da)*` | …der pille sie [___] eine operation am… |
| 2019 | Löschung | `eine` | `*(nicht da)*` | …pille sie hatten [___] operation am rechten… |
| 2020 | Löschung | `operation` | `*(nicht da)*` | …sie hatten eine [___] am rechten fuß… |
| 2021 | Löschung | `am` | `*(nicht da)*` | …hatten eine operation [___] rechten fuß vor… |
| 2022 | Löschung | `rechten` | `*(nicht da)*` | …eine operation am [___] fuß vor zwei… |
| 2023 | Löschung | `fuß` | `*(nicht da)*` | …operation am rechten [___] vor zwei jahren… |
| 2024 | Löschung | `vor` | `*(nicht da)*` | …am rechten fuß [___] zwei jahren da… |
| 2025 | Löschung | `zwei` | `*(nicht da)*` | …rechten fuß vor [___] jahren da wurde… |
| 2026 | Löschung | `jahren` | `*(nicht da)*` | …fuß vor zwei [___] da wurde der… |
| 2027 | Löschung | `da` | `*(nicht da)*` | …vor zwei jahren [___] wurde der halux… |
| 2028 | Löschung | `wurde` | `*(nicht da)*` | …zwei jahren da [___] der halux valgus… |
| 2029 | Löschung | `der` | `*(nicht da)*` | …jahren da wurde [___] halux valgus operiert… |
| 2030 | Löschung | `halux` | `*(nicht da)*` | …da wurde der [___] valgus operiert ansonsten… |
| 2031 | Löschung | `valgus` | `*(nicht da)*` | …wurde der halux [___] operiert ansonsten körperliche… |
| 2032 | Löschung | `operiert` | `*(nicht da)*` | …der halux valgus [___] ansonsten körperliche beschwerden… |
| 2033 | Löschung | `ansonsten` | `*(nicht da)*` | …halux valgus operiert [___] körperliche beschwerden gibt… |
| 2034 | Löschung | `körperliche` | `*(nicht da)*` | …valgus operiert ansonsten [___] beschwerden gibt es… |
| 2035 | Löschung | `beschwerden` | `*(nicht da)*` | …operiert ansonsten körperliche [___] gibt es keine… |
| 2036 | Löschung | `gibt` | `*(nicht da)*` | …ansonsten körperliche beschwerden [___] es keine sie… |
| 2037 | Löschung | `es` | `*(nicht da)*` | …körperliche beschwerden gibt [___] keine sie sind… |
| 2038 | Löschung | `keine` | `*(nicht da)*` | …beschwerden gibt es [___] sie sind ansonsten… |
| 2039 | Löschung | `sie` | `*(nicht da)*` | …gibt es keine [___] sind ansonsten gesund… |
| 2040 | Löschung | `sind` | `*(nicht da)*` | …es keine sie [___] ansonsten gesund gott… |
| 2041 | Löschung | `ansonsten` | `*(nicht da)*` | …keine sie sind [___] gesund gott sei… |
| 2042 | Löschung | `gesund` | `*(nicht da)*` | …sie sind ansonsten [___] gott sei dank… |
| 2043 | Löschung | `gott` | `*(nicht da)*` | …sind ansonsten gesund [___] sei dank bis… |
| 2044 | Löschung | `sei` | `*(nicht da)*` | …ansonsten gesund gott [___] dank bis auf… |
| 2045 | Löschung | `dank` | `*(nicht da)*` | …gesund gott sei [___] bis auf die… |
| 2046 | Löschung | `bis` | `*(nicht da)*` | …gott sei dank [___] auf die histaminunverträglichkeit… |
| 2047 | Löschung | `auf` | `*(nicht da)*` | …sei dank bis [___] die histaminunverträglichkeit genau… |
| 2048 | Löschung | `die` | `*(nicht da)*` | …dank bis auf [___] histaminunverträglichkeit genau das… |
| 2049 | Löschung | `histaminunverträglichkeit` | `*(nicht da)*` | …bis auf die [___] genau das hätte… |
| 2050 | Substitution | `genau` | `war` | …auf die histaminunverträglichkeit [___] das hätte ich… |
| 2051 | Substitution | `das` | `s` | …die histaminunverträglichkeit genau [___] hätte ich jetzt… |
| 2052 | Substitution | `hätte` | `erstmal` | …histaminunverträglichkeit genau das [___] ich jetzt auch… |
| 2053 | Löschung | `jetzt` | `*(nicht da)*` | …das hätte ich [___] auch noch mit… |
| 2054 | Löschung | `auch` | `*(nicht da)*` | …hätte ich jetzt [___] noch mit eingebracht… |
| 2055 | Löschung | `noch` | `*(nicht da)*` | …ich jetzt auch [___] mit eingebracht vielen… |
| 2056 | Löschung | `mit` | `*(nicht da)*` | …jetzt auch noch [___] eingebracht vielen dank… |
| 2057 | Löschung | `eingebracht` | `*(nicht da)*` | …auch noch mit [___] vielen dank nochmal… |
| 2058 | Löschung | `vielen` | `*(nicht da)*` | …noch mit eingebracht [___] dank nochmal dafür… |
| 2059 | Löschung | `dank` | `*(nicht da)*` | …mit eingebracht vielen [___] nochmal dafür habe… |
| 2060 | Löschung | `nochmal` | `*(nicht da)*` | …eingebracht vielen dank [___] dafür habe ich… |
| 2061 | Löschung | `dafür` | `*(nicht da)*` | …vielen dank nochmal [___] habe ich mir… |
| 2062 | Löschung | `habe` | `*(nicht da)*` | …dank nochmal dafür [___] ich mir notiert… |
| 2063 | Substitution | `ich` | `werde` | …nochmal dafür habe [___] mir notiert es… |
| 2064 | Löschung | `notiert` | `*(nicht da)*` | …habe ich mir [___] es gibt ein… |
| 2065 | Löschung | `es` | `*(nicht da)*` | …ich mir notiert [___] gibt ein paar… |
| 2066 | Löschung | `gibt` | `*(nicht da)*` | …mir notiert es [___] ein paar vorerkrankungen… |
| 2067 | Löschung | `ein` | `*(nicht da)*` | …notiert es gibt [___] paar vorerkrankungen in… |
| 2068 | Löschung | `paar` | `*(nicht da)*` | …es gibt ein [___] vorerkrankungen in der… |
| 2069 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …gibt ein paar [___] in der familiengeschichte… |
| 2070 | Löschung | `in` | `*(nicht da)*` | …ein paar vorerkrankungen [___] der familiengeschichte sie… |
| 2071 | Löschung | `der` | `*(nicht da)*` | …paar vorerkrankungen in [___] familiengeschichte sie sind… |
| 2072 | Löschung | `familiengeschichte` | `*(nicht da)*` | …vorerkrankungen in der [___] sie sind frisch… |
| 2073 | Löschung | `sie` | `*(nicht da)*` | …in der familiengeschichte [___] sind frisch verheiratet… |
| 2074 | Löschung | `sind` | `*(nicht da)*` | …der familiengeschichte sie [___] frisch verheiratet und… |
| 2075 | Löschung | `frisch` | `*(nicht da)*` | …familiengeschichte sie sind [___] verheiratet und arbeiten… |
| 2076 | Löschung | `verheiratet` | `*(nicht da)*` | …sie sind frisch [___] und arbeiten in… |
| 2077 | Löschung | `und` | `*(nicht da)*` | …sind frisch verheiratet [___] arbeiten in einer… |
| 2078 | Löschung | `arbeiten` | `*(nicht da)*` | …frisch verheiratet und [___] in einer marketingagentur… |
| 2079 | Löschung | `in` | `*(nicht da)*` | …verheiratet und arbeiten [___] einer marketingagentur ja… |
| 2080 | Löschung | `einer` | `*(nicht da)*` | …und arbeiten in [___] marketingagentur ja das… |
| 2081 | Löschung | `marketingagentur` | `*(nicht da)*` | …arbeiten in einer [___] ja das ist… |
| 2082 | Löschung | `ja` | `*(nicht da)*` | …in einer marketingagentur [___] das ist alles… |
| 2083 | Löschung | `ist` | `*(nicht da)*` | …marketingagentur ja das [___] alles richtig ja… |
| 2084 | Löschung | `richtig` | `*(nicht da)*` | …das ist alles [___] ja perfekt sehr… |
| 2085 | Löschung | `ja` | `*(nicht da)*` | …ist alles richtig [___] perfekt sehr gut… |
| 2086 | Löschung | `perfekt` | `*(nicht da)*` | …alles richtig ja [___] sehr gut frau… |
| 2087 | Löschung | `sehr` | `*(nicht da)*` | …richtig ja perfekt [___] gut frau beckenwestfalen… |
| 2088 | Substitution | `gut` | `mal` | …ja perfekt sehr [___] frau beckenwestfalen dann… |
| 2089 | Substitution | `frau` | `anschauen` | …perfekt sehr gut [___] beckenwestfalen dann war… |
| 2090 | Substitution | `beckenwestfalen` | `danke` | …sehr gut frau [___] dann war es… |
| 2091 | Substitution | `dann` | `schön` | …gut frau beckenwestfalen [___] war es das… |
| 2092 | Substitution | `war` | `gern` | …frau beckenwestfalen dann [___] es das jetzt… |
| 2093 | Substitution | `es` | `geschehen` | …beckenwestfalen dann war [___] das jetzt erstmal… |
| 2094 | Substitution | `das` | `ich` | …dann war es [___] jetzt erstmal von… |
| 2095 | Substitution | `jetzt` | `werde` | …war es das [___] erstmal von meiner… |
| 2096 | Substitution | `erstmal` | `ihnen` | …es das jetzt [___] von meiner seite… |
| 2097 | Substitution | `von` | `gleich` | …das jetzt erstmal [___] meiner seite wir… |
| 2098 | Substitution | `meiner` | `sagen` | …jetzt erstmal von [___] seite wir machen… |
| 2099 | Substitution | `seite` | `was` | …erstmal von meiner [___] wir machen jetzt… |
| 2100 | Löschung | `jetzt` | `*(nicht da)*` | …seite wir machen [___] mit den untersuchungen… |
| 2101 | Löschung | `mit` | `*(nicht da)*` | …wir machen jetzt [___] den untersuchungen weiter… |
| 2102 | Löschung | `den` | `*(nicht da)*` | …machen jetzt mit [___] untersuchungen weiter ich… |
| 2103 | Löschung | `untersuchungen` | `*(nicht da)*` | …jetzt mit den [___] weiter ich bin… |
| 2104 | Löschung | `weiter` | `*(nicht da)*` | …mit den untersuchungen [___] ich bin in… |
| 2105 | Löschung | `ich` | `*(nicht da)*` | …den untersuchungen weiter [___] bin in kurzer… |
| 2106 | Löschung | `bin` | `*(nicht da)*` | …untersuchungen weiter ich [___] in kurzer zeit… |
| 2107 | Löschung | `in` | `*(nicht da)*` | …weiter ich bin [___] kurzer zeit wieder… |
| 2108 | Löschung | `kurzer` | `*(nicht da)*` | …ich bin in [___] zeit wieder für… |
| 2109 | Löschung | `zeit` | `*(nicht da)*` | …bin in kurzer [___] wieder für sie… |
| 2110 | Löschung | `wieder` | `*(nicht da)*` | …in kurzer zeit [___] für sie da… |
| 2111 | Löschung | `für` | `*(nicht da)*` | …kurzer zeit wieder [___] sie da okay… |
| 2112 | Löschung | `sie` | `*(nicht da)*` | …zeit wieder für [___] da okay alles… |
| 2113 | Löschung | `da` | `*(nicht da)*` | …wieder für sie [___] okay alles klar… |
| 2114 | Löschung | `okay` | `*(nicht da)*` | …für sie da [___] alles klar ich… |
| 2115 | Löschung | `alles` | `*(nicht da)*` | …sie da okay [___] klar ich warte… |
| 2116 | Löschung | `klar` | `*(nicht da)*` | …da okay alles [___] ich warte dann… |
| 2117 | Löschung | `dann` | `*(nicht da)*` | …klar ich warte [___] hier dagegen super… |
| 2118 | Löschung | `hier` | `*(nicht da)*` | …ich warte dann [___] dagegen super besten… |
| 2119 | Löschung | `dagegen` | `*(nicht da)*` | …warte dann hier [___] super besten dank… |
| 2120 | Löschung | `super` | `*(nicht da)*` | …dann hier dagegen [___] besten dank und… |
| 2121 | Löschung | `besten` | `*(nicht da)*` | …hier dagegen super [___] dank und bis… |
| 2122 | Löschung | `dank` | `*(nicht da)*` | …dagegen super besten [___] und bis gleich… |
| 2123 | Löschung | `und` | `*(nicht da)*` | …super besten dank [___] bis gleich bis… |
| 2124 | Löschung | `bis` | `*(nicht da)*` | …besten dank und [___] gleich bis gleich… |
| 2125 | Löschung | `gleich` | `*(nicht da)*` | …dank und bis [___] bis gleich bis… |
| 2126 | Löschung | `bis` | `*(nicht da)*` | …und bis gleich [___] gleich bis gleich… |
| 2127 | Löschung | `gleich` | `*(nicht da)*` | …bis gleich bis [___] bis gleich… |
| 2128 | Löschung | `bis` | `*(nicht da)*` | …gleich bis gleich [___] gleich… |
| 2129 | Löschung | `gleich` | `*(nicht da)*` | …bis gleich bis [___]… |

---

## PWC

**Fehlerrate: 52.6%** — RAW: 1512 Wörter | FMT: 944 Wörter | S=228 D=568 I=0 | Fehler=796

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Löschung | `sogar` | `*(nicht da)*` | …glaube am schluss [___] haben wir treppensteigen… |
| 2 | Löschung | `haben` | `*(nicht da)*` | …am schluss sogar [___] wir treppensteigen dann… |
| 3 | Löschung | `wir` | `*(nicht da)*` | …schluss sogar haben [___] treppensteigen dann also… |
| 4 | Löschung | `treppensteigen` | `*(nicht da)*` | …sogar haben wir [___] dann also ja… |
| 5 | Löschung | `dann` | `*(nicht da)*` | …haben wir treppensteigen [___] also ja und… |
| 6 | Löschung | `also` | `*(nicht da)*` | …wir treppensteigen dann [___] ja und ein… |
| 7 | Löschung | `ja` | `*(nicht da)*` | …treppensteigen dann also [___] und ein bisschen… |
| 8 | Löschung | `und` | `*(nicht da)*` | …dann also ja [___] ein bisschen so… |
| 9 | Löschung | `ein` | `*(nicht da)*` | …also ja und [___] bisschen so beugen… |
| 10 | Löschung | `bisschen` | `*(nicht da)*` | …ja und ein [___] so beugen üben… |
| 11 | Löschung | `so` | `*(nicht da)*` | …und ein bisschen [___] beugen üben so… |
| 12 | Löschung | `beugen` | `*(nicht da)*` | …ein bisschen so [___] üben so war… |
| 13 | Löschung | `üben` | `*(nicht da)*` | …bisschen so beugen [___] so war es… |
| 14 | Löschung | `so` | `*(nicht da)*` | …so beugen üben [___] war es halt… |
| 15 | Löschung | `war` | `*(nicht da)*` | …beugen üben so [___] es halt gegangen… |
| 16 | Löschung | `es` | `*(nicht da)*` | …üben so war [___] halt gegangen und… |
| 17 | Löschung | `halt` | `*(nicht da)*` | …so war es [___] gegangen und sie… |
| 18 | Löschung | `gegangen` | `*(nicht da)*` | …war es halt [___] und sie haben… |
| 19 | Löschung | `und` | `*(nicht da)*` | …es halt gegangen [___] sie haben das… |
| 20 | Löschung | `sie` | `*(nicht da)*` | …halt gegangen und [___] haben das da… |
| 21 | Löschung | `haben` | `*(nicht da)*` | …gegangen und sie [___] das da schon… |
| 22 | Löschung | `das` | `*(nicht da)*` | …und sie haben [___] da schon eben… |
| 23 | Löschung | `da` | `*(nicht da)*` | …sie haben das [___] schon eben gesagt… |
| 24 | Löschung | `schon` | `*(nicht da)*` | …haben das da [___] eben gesagt sie… |
| 25 | Löschung | `eben` | `*(nicht da)*` | …das da schon [___] gesagt sie haben… |
| 26 | Löschung | `gesagt` | `*(nicht da)*` | …da schon eben [___] sie haben eben… |
| 27 | Löschung | `sie` | `*(nicht da)*` | …schon eben gesagt [___] haben eben mit… |
| 28 | Löschung | `haben` | `*(nicht da)*` | …eben gesagt sie [___] eben mit den… |
| 29 | Löschung | `eben` | `*(nicht da)*` | …gesagt sie haben [___] mit den stützen… |
| 30 | Löschung | `mit` | `*(nicht da)*` | …sie haben eben [___] den stützen das… |
| 31 | Löschung | `den` | `*(nicht da)*` | …haben eben mit [___] stützen das gelernt… |
| 32 | Löschung | `stützen` | `*(nicht da)*` | …eben mit den [___] das gelernt zum… |
| 33 | Löschung | `das` | `*(nicht da)*` | …mit den stützen [___] gelernt zum gehen… |
| 34 | Löschung | `gelernt` | `*(nicht da)*` | …den stützen das [___] zum gehen sie… |
| 35 | Löschung | `zum` | `*(nicht da)*` | …stützen das gelernt [___] gehen sie haben… |
| 36 | Löschung | `gehen` | `*(nicht da)*` | …das gelernt zum [___] sie haben das… |
| 37 | Löschung | `sie` | `*(nicht da)*` | …gelernt zum gehen [___] haben das auch… |
| 38 | Löschung | `haben` | `*(nicht da)*` | …zum gehen sie [___] das auch freuen… |
| 39 | Löschung | `das` | `*(nicht da)*` | …gehen sie haben [___] auch freuen gelernt… |
| 40 | Löschung | `auch` | `*(nicht da)*` | …sie haben das [___] freuen gelernt können… |
| 41 | Löschung | `freuen` | `*(nicht da)*` | …haben das auch [___] gelernt können sie… |
| 42 | Löschung | `gelernt` | `*(nicht da)*` | …das auch freuen [___] können sie das… |
| 43 | Löschung | `können` | `*(nicht da)*` | …auch freuen gelernt [___] sie das für… |
| 44 | Löschung | `sie` | `*(nicht da)*` | …freuen gelernt können [___] das für sie… |
| 45 | Löschung | `das` | `*(nicht da)*` | …gelernt können sie [___] für sie sagen… |
| 46 | Löschung | `für` | `*(nicht da)*` | …können sie das [___] sie sagen dass… |
| 47 | Löschung | `sie` | `*(nicht da)*` | …sie das für [___] sagen dass sie… |
| 48 | Löschung | `sagen` | `*(nicht da)*` | …das für sie [___] dass sie das… |
| 49 | Löschung | `dass` | `*(nicht da)*` | …für sie sagen [___] sie das dann… |
| 50 | Löschung | `sie` | `*(nicht da)*` | …sie sagen dass [___] das dann jetzt… |
| 51 | Löschung | `das` | `*(nicht da)*` | …sagen dass sie [___] dann jetzt in… |
| 52 | Löschung | `dann` | `*(nicht da)*` | …dass sie das [___] jetzt in den… |
| 53 | Löschung | `jetzt` | `*(nicht da)*` | …sie das dann [___] in den alltag… |
| 54 | Löschung | `in` | `*(nicht da)*` | …das dann jetzt [___] den alltag den… |
| 55 | Löschung | `den` | `*(nicht da)*` | …dann jetzt in [___] alltag den sie… |
| 56 | Löschung | `alltag` | `*(nicht da)*` | …jetzt in den [___] den sie jetzt… |
| 57 | Löschung | `den` | `*(nicht da)*` | …in den alltag [___] sie jetzt dann… |
| 58 | Löschung | `sie` | `*(nicht da)*` | …den alltag den [___] jetzt dann wieder… |
| 59 | Löschung | `jetzt` | `*(nicht da)*` | …alltag den sie [___] dann wieder haben… |
| 60 | Löschung | `dann` | `*(nicht da)*` | …den sie jetzt [___] wieder haben integriert… |
| 61 | Löschung | `wieder` | `*(nicht da)*` | …sie jetzt dann [___] haben integriert haben… |
| 62 | Löschung | `haben` | `*(nicht da)*` | …jetzt dann wieder [___] integriert haben ja… |
| 63 | Löschung | `integriert` | `*(nicht da)*` | …dann wieder haben [___] haben ja dass… |
| 64 | Löschung | `haben` | `*(nicht da)*` | …wieder haben integriert [___] ja dass das… |
| 65 | Löschung | `ja` | `*(nicht da)*` | …haben integriert haben [___] dass das auch… |
| 66 | Löschung | `dass` | `*(nicht da)*` | …integriert haben ja [___] das auch gut… |
| 67 | Löschung | `das` | `*(nicht da)*` | …haben ja dass [___] auch gut beherrschen… |
| 68 | Löschung | `auch` | `*(nicht da)*` | …ja dass das [___] gut beherrschen ja… |
| 69 | Löschung | `gut` | `*(nicht da)*` | …dass das auch [___] beherrschen ja schon… |
| 70 | Löschung | `beherrschen` | `*(nicht da)*` | …das auch gut [___] ja schon also… |
| 71 | Löschung | `ja` | `*(nicht da)*` | …auch gut beherrschen [___] schon also das… |
| 72 | Löschung | `schon` | `*(nicht da)*` | …gut beherrschen ja [___] also das auf… |
| 73 | Löschung | `also` | `*(nicht da)*` | …beherrschen ja schon [___] das auf jeden… |
| 74 | Löschung | `das` | `*(nicht da)*` | …ja schon also [___] auf jeden fall… |
| 75 | Löschung | `auf` | `*(nicht da)*` | …schon also das [___] jeden fall ja… |
| 76 | Löschung | `jeden` | `*(nicht da)*` | …also das auf [___] fall ja dass… |
| 77 | Löschung | `fall` | `*(nicht da)*` | …das auf jeden [___] ja dass sie… |
| 78 | Löschung | `ja` | `*(nicht da)*` | …auf jeden fall [___] dass sie darauf… |
| 79 | Löschung | `dass` | `*(nicht da)*` | …jeden fall ja [___] sie darauf achten… |
| 80 | Löschung | `sie` | `*(nicht da)*` | …fall ja dass [___] darauf achten einfach… |
| 81 | Löschung | `darauf` | `*(nicht da)*` | …ja dass sie [___] achten einfach wie… |
| 82 | Löschung | `achten` | `*(nicht da)*` | …dass sie darauf [___] einfach wie aufsteigen… |
| 83 | Löschung | `einfach` | `*(nicht da)*` | …sie darauf achten [___] wie aufsteigen genau… |
| 84 | Löschung | `wie` | `*(nicht da)*` | …darauf achten einfach [___] aufsteigen genau und… |
| 85 | Löschung | `aufsteigen` | `*(nicht da)*` | …achten einfach wie [___] genau und das… |
| 86 | Löschung | `genau` | `*(nicht da)*` | …einfach wie aufsteigen [___] und das hat… |
| 87 | Löschung | `und` | `*(nicht da)*` | …wie aufsteigen genau [___] das hat ihnen… |
| 88 | Löschung | `das` | `*(nicht da)*` | …aufsteigen genau und [___] hat ihnen zum… |
| 89 | Löschung | `hat` | `*(nicht da)*` | …genau und das [___] ihnen zum beispiel… |
| 90 | Löschung | `ihnen` | `*(nicht da)*` | …und das hat [___] zum beispiel schon… |
| 91 | Löschung | `zum` | `*(nicht da)*` | …das hat ihnen [___] beispiel schon geholfen… |
| 92 | Löschung | `beispiel` | `*(nicht da)*` | …hat ihnen zum [___] schon geholfen also… |
| 93 | Löschung | `schon` | `*(nicht da)*` | …ihnen zum beispiel [___] geholfen also sie… |
| 94 | Löschung | `geholfen` | `*(nicht da)*` | …zum beispiel schon [___] also sie haben… |
| 95 | Löschung | `also` | `*(nicht da)*` | …beispiel schon geholfen [___] sie haben da… |
| 96 | Löschung | `sie` | `*(nicht da)*` | …schon geholfen also [___] haben da gerne… |
| 97 | Löschung | `haben` | `*(nicht da)*` | …geholfen also sie [___] da gerne mitgemacht… |
| 98 | Löschung | `da` | `*(nicht da)*` | …also sie haben [___] gerne mitgemacht in… |
| 99 | Löschung | `gerne` | `*(nicht da)*` | …sie haben da [___] mitgemacht in der… |
| 100 | Löschung | `mitgemacht` | `*(nicht da)*` | …haben da gerne [___] in der therapie… |
| 101 | Löschung | `in` | `*(nicht da)*` | …da gerne mitgemacht [___] der therapie und…okay… |
| 102 | Löschung | `der` | `*(nicht da)*` | …gerne mitgemacht in [___] therapie und…okay ja… |
| 103 | Löschung | `therapie` | `*(nicht da)*` | …mitgemacht in der [___] und…okay ja das… |
| 104 | Löschung | `und…okay` | `*(nicht da)*` | …in der therapie [___] ja das war… |
| 105 | Löschung | `ja` | `*(nicht da)*` | …der therapie und…okay [___] das war halt… |
| 106 | Löschung | `das` | `*(nicht da)*` | …therapie und…okay ja [___] war halt der… |
| 107 | Löschung | `war` | `*(nicht da)*` | …und…okay ja das [___] halt der verlauf… |
| 108 | Löschung | `halt` | `*(nicht da)*` | …ja das war [___] der verlauf wie… |
| 109 | Löschung | `der` | `*(nicht da)*` | …das war halt [___] verlauf wie war… |
| 110 | Löschung | `verlauf` | `*(nicht da)*` | …war halt der [___] wie war das… |
| 111 | Löschung | `wie` | `*(nicht da)*` | …halt der verlauf [___] war das dann… |
| 112 | Löschung | `war` | `*(nicht da)*` | …der verlauf wie [___] das dann nach… |
| 113 | Löschung | `das` | `*(nicht da)*` | …verlauf wie war [___] dann nach der… |
| 114 | Löschung | `dann` | `*(nicht da)*` | …wie war das [___] nach der operation… |
| 115 | Löschung | `nach` | `*(nicht da)*` | …war das dann [___] der operation die… |
| 116 | Löschung | `der` | `*(nicht da)*` | …das dann nach [___] operation die woche… |
| 117 | Löschung | `operation` | `*(nicht da)*` | …dann nach der [___] die woche nachher… |
| 118 | Löschung | `die` | `*(nicht da)*` | …nach der operation [___] woche nachher und… |
| 119 | Löschung | `woche` | `*(nicht da)*` | …der operation die [___] nachher und wie… |
| 120 | Löschung | `nachher` | `*(nicht da)*` | …operation die woche [___] und wie ist… |
| 121 | Löschung | `und` | `*(nicht da)*` | …die woche nachher [___] wie ist ihnen… |
| 122 | Löschung | `wie` | `*(nicht da)*` | …woche nachher und [___] ist ihnen dann… |
| 123 | Löschung | `ist` | `*(nicht da)*` | …nachher und wie [___] ihnen dann da… |
| 124 | Löschung | `ihnen` | `*(nicht da)*` | …und wie ist [___] dann da gegangen… |
| 125 | Löschung | `dann` | `*(nicht da)*` | …wie ist ihnen [___] da gegangen mit… |
| 126 | Löschung | `da` | `*(nicht da)*` | …ist ihnen dann [___] gegangen mit den… |
| 127 | Löschung | `gegangen` | `*(nicht da)*` | …ihnen dann da [___] mit den schmerzen… |
| 128 | Löschung | `mit` | `*(nicht da)*` | …dann da gegangen [___] den schmerzen ja… |
| 129 | Löschung | `den` | `*(nicht da)*` | …da gegangen mit [___] schmerzen ja schmerzen… |
| 130 | Löschung | `schmerzen` | `*(nicht da)*` | …gegangen mit den [___] ja schmerzen war… |
| 131 | Löschung | `ja` | `*(nicht da)*` | …mit den schmerzen [___] schmerzen war ja… |
| 132 | Löschung | `schmerzen` | `*(nicht da)*` | …den schmerzen ja [___] war ja war… |
| 133 | Löschung | `war` | `*(nicht da)*` | …schmerzen ja schmerzen [___] ja war okay… |
| 134 | Löschung | `ja` | `*(nicht da)*` | …ja schmerzen war [___] war okay sag… |
| 135 | Löschung | `war` | `*(nicht da)*` | …schmerzen war ja [___] okay sag ich… |
| 136 | Löschung | `okay` | `*(nicht da)*` | …war ja war [___] sag ich mal… |
| 137 | Löschung | `sag` | `*(nicht da)*` | …ja war okay [___] ich mal war… |
| 138 | Löschung | `ich` | `*(nicht da)*` | …war okay sag [___] mal war okay… |
| 139 | Löschung | `mal` | `*(nicht da)*` | …okay sag ich [___] war okay je… |
| 140 | Löschung | `war` | `*(nicht da)*` | …sag ich mal [___] okay je nachdem… |
| 141 | Löschung | `okay` | `*(nicht da)*` | …ich mal war [___] je nachdem je… |
| 142 | Löschung | `je` | `*(nicht da)*` | …mal war okay [___] nachdem je nach… |
| 143 | Löschung | `nachdem` | `*(nicht da)*` | …war okay je [___] je nach belastung… |
| 144 | Löschung | `je` | `*(nicht da)*` | …okay je nachdem [___] nach belastung je… |
| 145 | Löschung | `nach` | `*(nicht da)*` | …je nachdem je [___] belastung je nach… |
| 146 | Löschung | `belastung` | `*(nicht da)*` | …nachdem je nach [___] je nach belastung… |
| 147 | Löschung | `je` | `*(nicht da)*` | …je nach belastung [___] nach belastung es… |
| 148 | Löschung | `nach` | `*(nicht da)*` | …nach belastung je [___] belastung es war… |
| 149 | Löschung | `belastung` | `*(nicht da)*` | …belastung je nach [___] es war halt…… |
| 150 | Löschung | `es` | `*(nicht da)*` | …je nach belastung [___] war halt… ich… |
| 151 | Löschung | `war` | `*(nicht da)*` | …nach belastung es [___] halt… ich habe… |
| 152 | Löschung | `halt…` | `*(nicht da)*` | …belastung es war [___] ich habe mich… |
| 153 | Löschung | `ich` | `*(nicht da)*` | …es war halt… [___] habe mich halt… |
| 154 | Löschung | `habe` | `*(nicht da)*` | …war halt… ich [___] mich halt nicht… |
| 155 | Löschung | `mich` | `*(nicht da)*` | …halt… ich habe [___] halt nicht viel… |
| 156 | Löschung | `halt` | `*(nicht da)*` | …ich habe mich [___] nicht viel bewegen… |
| 157 | Löschung | `nicht` | `*(nicht da)*` | …habe mich halt [___] viel bewegen können… |
| 158 | Löschung | `viel` | `*(nicht da)*` | …mich halt nicht [___] bewegen können ich… |
| 159 | Löschung | `bewegen` | `*(nicht da)*` | …halt nicht viel [___] können ich bin… |
| 160 | Löschung | `können` | `*(nicht da)*` | …nicht viel bewegen [___] ich bin ja… |
| 161 | Löschung | `ich` | `*(nicht da)*` | …viel bewegen können [___] bin ja eigentlich… |
| 162 | Löschung | `bin` | `*(nicht da)*` | …bewegen können ich [___] ja eigentlich nur… |
| 163 | Löschung | `ja` | `*(nicht da)*` | …können ich bin [___] eigentlich nur gelegen… |
| 164 | Löschung | `eigentlich` | `*(nicht da)*` | …ich bin ja [___] nur gelegen okay… |
| 165 | Löschung | `nur` | `*(nicht da)*` | …bin ja eigentlich [___] gelegen okay die… |
| 166 | Löschung | `gelegen` | `*(nicht da)*` | …ja eigentlich nur [___] okay die erste… |
| 167 | Löschung | `okay` | `*(nicht da)*` | …eigentlich nur gelegen [___] die erste woche… |
| 168 | Löschung | `die` | `*(nicht da)*` | …nur gelegen okay [___] erste woche deine… |
| 169 | Löschung | `erste` | `*(nicht da)*` | …gelegen okay die [___] woche deine letzte… |
| 170 | Löschung | `woche` | `*(nicht da)*` | …okay die erste [___] deine letzte zeit… |
| 171 | Löschung | `deine` | `*(nicht da)*` | …die erste woche [___] letzte zeit und… |
| 172 | Löschung | `letzte` | `*(nicht da)*` | …erste woche deine [___] zeit und dann… |
| 173 | Löschung | `zeit` | `*(nicht da)*` | …woche deine letzte [___] und dann ja… |
| 174 | Löschung | `und` | `*(nicht da)*` | …deine letzte zeit [___] dann ja mit… |
| 175 | Löschung | `dann` | `*(nicht da)*` | …letzte zeit und [___] ja mit den… |
| 176 | Löschung | `ja` | `*(nicht da)*` | …zeit und dann [___] mit den grücken… |
| 177 | Löschung | `mit` | `*(nicht da)*` | …und dann ja [___] den grücken halt… |
| 178 | Löschung | `den` | `*(nicht da)*` | …dann ja mit [___] grücken halt herumgehen… |
| 179 | Löschung | `grücken` | `*(nicht da)*` | …ja mit den [___] halt herumgehen ein… |
| 180 | Löschung | `halt` | `*(nicht da)*` | …mit den grücken [___] herumgehen ein bisschen… |
| 181 | Löschung | `herumgehen` | `*(nicht da)*` | …den grücken halt [___] ein bisschen aber… |
| 182 | Löschung | `ein` | `*(nicht da)*` | …grücken halt herumgehen [___] bisschen aber halt… |
| 183 | Löschung | `bisschen` | `*(nicht da)*` | …halt herumgehen ein [___] aber halt auch… |
| 184 | Löschung | `aber` | `*(nicht da)*` | …herumgehen ein bisschen [___] halt auch minimal… |
| 185 | Löschung | `halt` | `*(nicht da)*` | …ein bisschen aber [___] auch minimal okay… |
| 186 | Löschung | `auch` | `*(nicht da)*` | …bisschen aber halt [___] minimal okay dann… |
| 187 | Löschung | `minimal` | `*(nicht da)*` | …aber halt auch [___] okay dann sind… |
| 188 | Löschung | `okay` | `*(nicht da)*` | …halt auch minimal [___] dann sind wir… |
| 189 | Löschung | `dann` | `*(nicht da)*` | …auch minimal okay [___] sind wir jetzt… |
| 190 | Löschung | `sind` | `*(nicht da)*` | …minimal okay dann [___] wir jetzt schon… |
| 191 | Löschung | `wir` | `*(nicht da)*` | …okay dann sind [___] jetzt schon so… |
| 192 | Löschung | `jetzt` | `*(nicht da)*` | …dann sind wir [___] schon so weit… |
| 193 | Löschung | `schon` | `*(nicht da)*` | …sind wir jetzt [___] so weit dass… |
| 194 | Löschung | `so` | `*(nicht da)*` | …wir jetzt schon [___] weit dass wir… |
| 195 | Löschung | `weit` | `*(nicht da)*` | …jetzt schon so [___] dass wir darüber… |
| 196 | Löschung | `dass` | `*(nicht da)*` | …schon so weit [___] wir darüber reden… |
| 197 | Löschung | `wir` | `*(nicht da)*` | …so weit dass [___] darüber reden wie… |
| 198 | Löschung | `darüber` | `*(nicht da)*` | …weit dass wir [___] reden wie es… |
| 199 | Löschung | `reden` | `*(nicht da)*` | …dass wir darüber [___] wie es ihnen… |
| 200 | Löschung | `wie` | `*(nicht da)*` | …wir darüber reden [___] es ihnen jetzt… |
| 201 | Löschung | `es` | `*(nicht da)*` | …darüber reden wie [___] ihnen jetzt geht… |
| 202 | Löschung | `ihnen` | `*(nicht da)*` | …reden wie es [___] jetzt geht wie… |
| 203 | Löschung | `jetzt` | `*(nicht da)*` | …wie es ihnen [___] geht wie geht… |
| 204 | Löschung | `geht` | `*(nicht da)*` | …es ihnen jetzt [___] wie geht es… |
| 205 | Löschung | `wie` | `*(nicht da)*` | …ihnen jetzt geht [___] geht es ihnen… |
| 206 | Löschung | `geht` | `*(nicht da)*` | …jetzt geht wie [___] es ihnen wenn… |
| 207 | Löschung | `es` | `*(nicht da)*` | …geht wie geht [___] ihnen wenn sie… |
| 208 | Löschung | `ihnen` | `*(nicht da)*` | …wie geht es [___] wenn sie an… |
| 209 | Löschung | `wenn` | `*(nicht da)*` | …geht es ihnen [___] sie an die… |
| 210 | Löschung | `sie` | `*(nicht da)*` | …es ihnen wenn [___] an die schmerzen… |
| 211 | Löschung | `an` | `*(nicht da)*` | …ihnen wenn sie [___] die schmerzen denken… |
| 212 | Löschung | `die` | `*(nicht da)*` | …wenn sie an [___] schmerzen denken auf… |
| 213 | Löschung | `schmerzen` | `*(nicht da)*` | …sie an die [___] denken auf einer… |
| 214 | Löschung | `denken` | `*(nicht da)*` | …an die schmerzen [___] auf einer skala… |
| 215 | Löschung | `auf` | `*(nicht da)*` | …die schmerzen denken [___] einer skala von… |
| 216 | Löschung | `einer` | `*(nicht da)*` | …schmerzen denken auf [___] skala von 0… |
| 217 | Löschung | `skala` | `*(nicht da)*` | …denken auf einer [___] von 0 bis… |
| 218 | Löschung | `von` | `*(nicht da)*` | …auf einer skala [___] 0 bis 10… |
| 219 | Löschung | `0` | `*(nicht da)*` | …einer skala von [___] bis 10 und… |
| 220 | Löschung | `bis` | `*(nicht da)*` | …skala von 0 [___] 10 und 10… |
| 221 | Löschung | `10` | `*(nicht da)*` | …von 0 bis [___] und 10 sind… |
| 222 | Löschung | `und` | `*(nicht da)*` | …0 bis 10 [___] 10 sind die… |
| 223 | Löschung | `10` | `*(nicht da)*` | …bis 10 und [___] sind die schlimmsten… |
| 224 | Löschung | `sind` | `*(nicht da)*` | …10 und 10 [___] die schlimmsten schmerzen… |
| 225 | Löschung | `die` | `*(nicht da)*` | …und 10 sind [___] schlimmsten schmerzen die… |
| 226 | Löschung | `schlimmsten` | `*(nicht da)*` | …10 sind die [___] schmerzen die ich… |
| 227 | Löschung | `schmerzen` | `*(nicht da)*` | …sind die schlimmsten [___] die ich sich… |
| 228 | Löschung | `die` | `*(nicht da)*` | …die schlimmsten schmerzen [___] ich sich vorstellen… |
| 229 | Löschung | `ich` | `*(nicht da)*` | …schlimmsten schmerzen die [___] sich vorstellen könnte… |
| 230 | Löschung | `sich` | `*(nicht da)*` | …schmerzen die ich [___] vorstellen könnte und… |
| 231 | Löschung | `vorstellen` | `*(nicht da)*` | …die ich sich [___] könnte und 0… |
| 232 | Löschung | `könnte` | `*(nicht da)*` | …ich sich vorstellen [___] und 0 ist… |
| 233 | Löschung | `und` | `*(nicht da)*` | …sich vorstellen könnte [___] 0 ist schmerzfrei… |
| 234 | Löschung | `0` | `*(nicht da)*` | …vorstellen könnte und [___] ist schmerzfrei wo… |
| 235 | Löschung | `ist` | `*(nicht da)*` | …könnte und 0 [___] schmerzfrei wo würden… |
| 236 | Löschung | `schmerzfrei` | `*(nicht da)*` | …und 0 ist [___] wo würden sie… |
| 237 | Löschung | `wo` | `*(nicht da)*` | …0 ist schmerzfrei [___] würden sie sich… |
| 238 | Löschung | `würden` | `*(nicht da)*` | …ist schmerzfrei wo [___] sie sich da… |
| 239 | Löschung | `sie` | `*(nicht da)*` | …schmerzfrei wo würden [___] sich da eingliedern… |
| 240 | Löschung | `sich` | `*(nicht da)*` | …wo würden sie [___] da eingliedern ja… |
| 241 | Löschung | `da` | `*(nicht da)*` | …würden sie sich [___] eingliedern ja wie… |
| 242 | Löschung | `eingliedern` | `*(nicht da)*` | …sie sich da [___] ja wie gesagt… |
| 243 | Löschung | `ja` | `*(nicht da)*` | …sich da eingliedern [___] wie gesagt es… |
| 244 | Löschung | `wie` | `*(nicht da)*` | …da eingliedern ja [___] gesagt es kommt… |
| 245 | Löschung | `gesagt` | `*(nicht da)*` | …eingliedern ja wie [___] es kommt eigentlich… |
| 246 | Löschung | `es` | `*(nicht da)*` | …ja wie gesagt [___] kommt eigentlich auf… |
| 247 | Löschung | `kommt` | `*(nicht da)*` | …wie gesagt es [___] eigentlich auf die… |
| 248 | Löschung | `eigentlich` | `*(nicht da)*` | …gesagt es kommt [___] auf die belastung… |
| 249 | Löschung | `auf` | `*(nicht da)*` | …es kommt eigentlich [___] die belastung darauf… |
| 250 | Löschung | `die` | `*(nicht da)*` | …kommt eigentlich auf [___] belastung darauf an… |
| 251 | Löschung | `belastung` | `*(nicht da)*` | …eigentlich auf die [___] darauf an wenn… |
| 252 | Löschung | `darauf` | `*(nicht da)*` | …auf die belastung [___] an wenn ich… |
| 253 | Löschung | `an` | `*(nicht da)*` | …die belastung darauf [___] wenn ich jetzt… |
| 254 | Löschung | `wenn` | `*(nicht da)*` | …belastung darauf an [___] ich jetzt im… |
| 255 | Löschung | `ich` | `*(nicht da)*` | …darauf an wenn [___] jetzt im ruhezustand… |
| 256 | Löschung | `jetzt` | `*(nicht da)*` | …an wenn ich [___] im ruhezustand bin… |
| 257 | Löschung | `im` | `*(nicht da)*` | …wenn ich jetzt [___] ruhezustand bin und… |
| 258 | Löschung | `ruhezustand` | `*(nicht da)*` | …ich jetzt im [___] bin und mich… |
| 259 | Löschung | `bin` | `*(nicht da)*` | …jetzt im ruhezustand [___] und mich nicht… |
| 260 | Löschung | `und` | `*(nicht da)*` | …im ruhezustand bin [___] mich nicht bewege… |
| 261 | Löschung | `mich` | `*(nicht da)*` | …ruhezustand bin und [___] nicht bewege dann… |
| 262 | Löschung | `nicht` | `*(nicht da)*` | …bin und mich [___] bewege dann sage… |
| 263 | Löschung | `bewege` | `*(nicht da)*` | …und mich nicht [___] dann sage ich… |
| 264 | Löschung | `dann` | `*(nicht da)*` | …mich nicht bewege [___] sage ich vielleicht… |
| 265 | Löschung | `sage` | `*(nicht da)*` | …nicht bewege dann [___] ich vielleicht 1… |
| 266 | Löschung | `ich` | `*(nicht da)*` | …bewege dann sage [___] vielleicht 1 aber… |
| 267 | Löschung | `vielleicht` | `*(nicht da)*` | …dann sage ich [___] 1 aber wenn… |
| 268 | Löschung | `1` | `*(nicht da)*` | …sage ich vielleicht [___] aber wenn ich… |
| 269 | Löschung | `aber` | `*(nicht da)*` | …ich vielleicht 1 [___] wenn ich jetzt… |
| 270 | Löschung | `wenn` | `*(nicht da)*` | …vielleicht 1 aber [___] ich jetzt mit… |
| 271 | Löschung | `ich` | `*(nicht da)*` | …1 aber wenn [___] jetzt mit den… |
| 272 | Löschung | `jetzt` | `*(nicht da)*` | …aber wenn ich [___] mit den grücken… |
| 273 | Löschung | `mit` | `*(nicht da)*` | …wenn ich jetzt [___] den grücken gehe… |
| 274 | Löschung | `den` | `*(nicht da)*` | …ich jetzt mit [___] grücken gehe dann… |
| 275 | Löschung | `grücken` | `*(nicht da)*` | …jetzt mit den [___] gehe dann keine… |
| 276 | Löschung | `gehe` | `*(nicht da)*` | …mit den grücken [___] dann keine ahnung… |
| 277 | Löschung | `dann` | `*(nicht da)*` | …den grücken gehe [___] keine ahnung 3… |
| 278 | Löschung | `keine` | `*(nicht da)*` | …grücken gehe dann [___] ahnung 3 und… |
| 279 | Löschung | `ahnung` | `*(nicht da)*` | …gehe dann keine [___] 3 und wenn… |
| 280 | Löschung | `3` | `*(nicht da)*` | …dann keine ahnung [___] und wenn ich… |
| 281 | Löschung | `und` | `*(nicht da)*` | …keine ahnung 3 [___] wenn ich wirklich… |
| 282 | Löschung | `wenn` | `*(nicht da)*` | …ahnung 3 und [___] ich wirklich ohne… |
| 283 | Löschung | `ich` | `*(nicht da)*` | …3 und wenn [___] wirklich ohne stützen… |
| 284 | Löschung | `wirklich` | `*(nicht da)*` | …und wenn ich [___] ohne stützen probiere… |
| 285 | Löschung | `ohne` | `*(nicht da)*` | …wenn ich wirklich [___] stützen probiere dann… |
| 286 | Löschung | `stützen` | `*(nicht da)*` | …ich wirklich ohne [___] probiere dann bin… |
| 287 | Löschung | `probiere` | `*(nicht da)*` | …wirklich ohne stützen [___] dann bin ich… |
| 288 | Löschung | `dann` | `*(nicht da)*` | …ohne stützen probiere [___] bin ich sicher… |
| 289 | Löschung | `bin` | `*(nicht da)*` | …stützen probiere dann [___] ich sicher bei… |
| 290 | Löschung | `ich` | `*(nicht da)*` | …probiere dann bin [___] sicher bei 6… |
| 291 | Löschung | `sicher` | `*(nicht da)*` | …dann bin ich [___] bei 6 oder… |
| 292 | Löschung | `bei` | `*(nicht da)*` | …bin ich sicher [___] 6 oder 7… |
| 293 | Löschung | `6` | `*(nicht da)*` | …ich sicher bei [___] oder 7 bei… |
| 294 | Löschung | `oder` | `*(nicht da)*` | …sicher bei 6 [___] 7 bei 6… |
| 295 | Löschung | `7` | `*(nicht da)*` | …bei 6 oder [___] bei 6 oder… |
| 296 | Löschung | `bei` | `*(nicht da)*` | …6 oder 7 [___] 6 oder 7… |
| 297 | Löschung | `6` | `*(nicht da)*` | …oder 7 bei [___] oder 7 aber… |
| 298 | Löschung | `oder` | `*(nicht da)*` | …7 bei 6 [___] 7 aber es… |
| 299 | Löschung | `7` | `*(nicht da)*` | …bei 6 oder [___] aber es ist… |
| 300 | Löschung | `aber` | `*(nicht da)*` | …6 oder 7 [___] es ist je… |
| 301 | Löschung | `es` | `*(nicht da)*` | …oder 7 aber [___] ist je nach… |
| 302 | Löschung | `ist` | `*(nicht da)*` | …7 aber es [___] je nach belastung… |
| 303 | Löschung | `je` | `*(nicht da)*` | …aber es ist [___] nach belastung halt… |
| 304 | Löschung | `nach` | `*(nicht da)*` | …es ist je [___] belastung halt und… |
| 305 | Löschung | `belastung` | `*(nicht da)*` | …ist je nach [___] halt und das… |
| 306 | Löschung | `halt` | `*(nicht da)*` | …je nach belastung [___] und das ist… |
| 307 | Löschung | `und` | `*(nicht da)*` | …nach belastung halt [___] das ist ja… |
| 308 | Löschung | `das` | `*(nicht da)*` | …belastung halt und [___] ist ja der… |
| 309 | Löschung | `ist` | `*(nicht da)*` | …halt und das [___] ja der einzige… |
| 310 | Löschung | `ja` | `*(nicht da)*` | …und das ist [___] der einzige faktor… |
| 311 | Löschung | `der` | `*(nicht da)*` | …das ist ja [___] einzige faktor die… |
| 312 | Löschung | `einzige` | `*(nicht da)*` | …ist ja der [___] faktor die belastung… |
| 313 | Löschung | `faktor` | `*(nicht da)*` | …ja der einzige [___] die belastung der… |
| 314 | Löschung | `die` | `*(nicht da)*` | …der einzige faktor [___] belastung der einem… |
| 315 | Löschung | `belastung` | `*(nicht da)*` | …einzige faktor die [___] der einem da… |
| 316 | Löschung | `der` | `*(nicht da)*` | …faktor die belastung [___] einem da einfällt… |
| 317 | Löschung | `einem` | `*(nicht da)*` | …die belastung der [___] da einfällt wenn… |
| 318 | Löschung | `da` | `*(nicht da)*` | …belastung der einem [___] einfällt wenn sie… |
| 319 | Löschung | `einfällt` | `*(nicht da)*` | …der einem da [___] wenn sie an… |
| 320 | Löschung | `wenn` | `*(nicht da)*` | …einem da einfällt [___] sie an den… |
| 321 | Löschung | `sie` | `*(nicht da)*` | …da einfällt wenn [___] an den schmerz… |
| 322 | Löschung | `an` | `*(nicht da)*` | …einfällt wenn sie [___] den schmerz denken… |
| 323 | Löschung | `den` | `*(nicht da)*` | …wenn sie an [___] schmerz denken dass… |
| 324 | Löschung | `schmerz` | `*(nicht da)*` | …sie an den [___] denken dass sich… |
| 325 | Löschung | `denken` | `*(nicht da)*` | …an den schmerz [___] dass sich der… |
| 326 | Löschung | `dass` | `*(nicht da)*` | …den schmerz denken [___] sich der da… |
| 327 | Löschung | `sich` | `*(nicht da)*` | …schmerz denken dass [___] der da verändert… |
| 328 | Löschung | `der` | `*(nicht da)*` | …denken dass sich [___] da verändert ja… |
| 329 | Löschung | `da` | `*(nicht da)*` | …dass sich der [___] verändert ja eigentlich… |
| 330 | Löschung | `verändert` | `*(nicht da)*` | …sich der da [___] ja eigentlich ja… |
| 331 | Löschung | `ja` | `*(nicht da)*` | …der da verändert [___] eigentlich ja also… |
| 332 | Löschung | `eigentlich` | `*(nicht da)*` | …da verändert ja [___] ja also ich… |
| 333 | Löschung | `ja` | `*(nicht da)*` | …verändert ja eigentlich [___] also ich weiß… |
| 334 | Löschung | `also` | `*(nicht da)*` | …ja eigentlich ja [___] ich weiß ja… |
| 335 | Löschung | `ich` | `*(nicht da)*` | …eigentlich ja also [___] weiß ja das… |
| 336 | Löschung | `weiß` | `*(nicht da)*` | …ja also ich [___] ja das nicht… |
| 337 | Löschung | `ja` | `*(nicht da)*` | …also ich weiß [___] das nicht an… |
| 338 | Löschung | `das` | `*(nicht da)*` | …ich weiß ja [___] nicht an was… |
| 339 | Löschung | `nicht` | `*(nicht da)*` | …weiß ja das [___] an was sonst… |
| 340 | Löschung | `an` | `*(nicht da)*` | …ja das nicht [___] was sonst noch… |
| 341 | Löschung | `was` | `*(nicht da)*` | …das nicht an [___] sonst noch okay… |
| 342 | Löschung | `sonst` | `*(nicht da)*` | …nicht an was [___] noch okay und… |
| 343 | Löschung | `noch` | `*(nicht da)*` | …an was sonst [___] okay und sie… |
| 344 | Löschung | `okay` | `*(nicht da)*` | …was sonst noch [___] und sie haben… |
| 345 | Löschung | `und` | `*(nicht da)*` | …sonst noch okay [___] sie haben gesagt… |
| 346 | Löschung | `sie` | `*(nicht da)*` | …noch okay und [___] haben gesagt sie… |
| 347 | Löschung | `haben` | `*(nicht da)*` | …okay und sie [___] gesagt sie haben… |
| 348 | Löschung | `gesagt` | `*(nicht da)*` | …und sie haben [___] sie haben eben… |
| 349 | Löschung | `sie` | `*(nicht da)*` | …sie haben gesagt [___] haben eben mit… |
| 350 | Löschung | `haben` | `*(nicht da)*` | …haben gesagt sie [___] eben mit dem… |
| 351 | Löschung | `eben` | `*(nicht da)*` | …gesagt sie haben [___] mit dem gehen… |
| 352 | Löschung | `mit` | `*(nicht da)*` | …sie haben eben [___] dem gehen mit… |
| 353 | Löschung | `dem` | `*(nicht da)*` | …haben eben mit [___] gehen mit den… |
| 354 | Löschung | `gehen` | `*(nicht da)*` | …eben mit dem [___] mit den stützen… |
| 355 | Löschung | `mit` | `*(nicht da)*` | …mit dem gehen [___] den stützen das… |
| 356 | Löschung | `den` | `*(nicht da)*` | …dem gehen mit [___] stützen das funktioniert… |
| 357 | Löschung | `stützen` | `*(nicht da)*` | …gehen mit den [___] das funktioniert nur… |
| 358 | Löschung | `das` | `*(nicht da)*` | …mit den stützen [___] funktioniert nur kurz… |
| 359 | Löschung | `funktioniert` | `*(nicht da)*` | …den stützen das [___] nur kurz was… |
| 360 | Löschung | `nur` | `*(nicht da)*` | …stützen das funktioniert [___] kurz was können… |
| 361 | Löschung | `kurz` | `*(nicht da)*` | …das funktioniert nur [___] was können wir… |
| 362 | Löschung | `was` | `*(nicht da)*` | …funktioniert nur kurz [___] können wir da… |
| 363 | Löschung | `können` | `*(nicht da)*` | …nur kurz was [___] wir da forschen… |
| 364 | Löschung | `wir` | `*(nicht da)*` | …kurz was können [___] da forschen also… |
| 365 | Löschung | `da` | `*(nicht da)*` | …was können wir [___] forschen also sind… |
| 366 | Löschung | `forschen` | `*(nicht da)*` | …können wir da [___] also sind sie… |
| 367 | Löschung | `also` | `*(nicht da)*` | …wir da forschen [___] sind sie auf… |
| 368 | Löschung | `sind` | `*(nicht da)*` | …da forschen also [___] sie auf und… |
| 369 | Löschung | `sie` | `*(nicht da)*` | …forschen also sind [___] auf und zu… |
| 370 | Löschung | `auf` | `*(nicht da)*` | …also sind sie [___] und zu rausgegangen… |
| 371 | Löschung | `und` | `*(nicht da)*` | …sind sie auf [___] zu rausgegangen nein… |
| 372 | Löschung | `zu` | `*(nicht da)*` | …sie auf und [___] rausgegangen nein jetzt… |
| 373 | Löschung | `rausgegangen` | `*(nicht da)*` | …auf und zu [___] nein jetzt nicht… |
| 374 | Löschung | `nein` | `*(nicht da)*` | …und zu rausgegangen [___] jetzt nicht also… |
| 375 | Löschung | `jetzt` | `*(nicht da)*` | …zu rausgegangen nein [___] nicht also das… |
| 376 | Löschung | `nicht` | `*(nicht da)*` | …rausgegangen nein jetzt [___] also das ist… |
| 377 | Löschung | `also` | `*(nicht da)*` | …nein jetzt nicht [___] das ist jetzt… |
| 378 | Löschung | `das` | `*(nicht da)*` | …jetzt nicht also [___] ist jetzt auch… |
| 379 | Löschung | `ist` | `*(nicht da)*` | …nicht also das [___] jetzt auch ein… |
| 380 | Löschung | `jetzt` | `*(nicht da)*` | …also das ist [___] auch ein monat… |
| 381 | Löschung | `auch` | `*(nicht da)*` | …das ist jetzt [___] ein monat her… |
| 382 | Löschung | `ein` | `*(nicht da)*` | …ist jetzt auch [___] monat her also… |
| 383 | Löschung | `monat` | `*(nicht da)*` | …jetzt auch ein [___] her also nicht… |
| 384 | Löschung | `her` | `*(nicht da)*` | …auch ein monat [___] also nicht wirklich… |
| 385 | Löschung | `also` | `*(nicht da)*` | …ein monat her [___] nicht wirklich ich… |
| 386 | Löschung | `nicht` | `*(nicht da)*` | …monat her also [___] wirklich ich meine… |
| 387 | Löschung | `wirklich` | `*(nicht da)*` | …her also nicht [___] ich meine minimal… |
| 388 | Löschung | `ich` | `*(nicht da)*` | …also nicht wirklich [___] meine minimal einfach… |
| 389 | Löschung | `meine` | `*(nicht da)*` | …nicht wirklich ich [___] minimal einfach aber… |
| 390 | Löschung | `minimal` | `*(nicht da)*` | …wirklich ich meine [___] einfach aber ich… |
| 391 | Löschung | `einfach` | `*(nicht da)*` | …ich meine minimal [___] aber ich kann… |
| 392 | Löschung | `aber` | `*(nicht da)*` | …meine minimal einfach [___] ich kann nicht… |
| 393 | Löschung | `ich` | `*(nicht da)*` | …minimal einfach aber [___] kann nicht wirklich… |
| 394 | Löschung | `kann` | `*(nicht da)*` | …einfach aber ich [___] nicht wirklich zusammenkriegen… |
| 395 | Löschung | `nicht` | `*(nicht da)*` | …aber ich kann [___] wirklich zusammenkriegen jetzt… |
| 396 | Löschung | `wirklich` | `*(nicht da)*` | …ich kann nicht [___] zusammenkriegen jetzt spazieren… |
| 397 | Löschung | `zusammenkriegen` | `*(nicht da)*` | …kann nicht wirklich [___] jetzt spazieren oder… |
| 398 | Löschung | `jetzt` | `*(nicht da)*` | …nicht wirklich zusammenkriegen [___] spazieren oder so… |
| 399 | Löschung | `spazieren` | `*(nicht da)*` | …wirklich zusammenkriegen jetzt [___] oder so also… |
| 400 | Löschung | `oder` | `*(nicht da)*` | …zusammenkriegen jetzt spazieren [___] so also ich… |
| 401 | Löschung | `so` | `*(nicht da)*` | …jetzt spazieren oder [___] also ich bewege… |
| 402 | Löschung | `also` | `*(nicht da)*` | …spazieren oder so [___] ich bewege mich… |
| 403 | Löschung | `ich` | `*(nicht da)*` | …oder so also [___] bewege mich halt… |
| 404 | Löschung | `bewege` | `*(nicht da)*` | …so also ich [___] mich halt in… |
| 405 | Löschung | `mich` | `*(nicht da)*` | …also ich bewege [___] halt in der… |
| 406 | Löschung | `halt` | `*(nicht da)*` | …ich bewege mich [___] in der wohnung… |
| 407 | Löschung | `in` | `*(nicht da)*` | …bewege mich halt [___] der wohnung was… |
| 408 | Löschung | `der` | `*(nicht da)*` | …mich halt in [___] wohnung was das… |
| 409 | Löschung | `wohnung` | `*(nicht da)*` | …halt in der [___] was das nötigste… |
| 410 | Löschung | `was` | `*(nicht da)*` | …in der wohnung [___] das nötigste und… |
| 411 | Löschung | `das` | `*(nicht da)*` | …der wohnung was [___] nötigste und ja… |
| 412 | Löschung | `nötigste` | `*(nicht da)*` | …wohnung was das [___] und ja versuche… |
| 413 | Löschung | `und` | `*(nicht da)*` | …was das nötigste [___] ja versuche halt… |
| 414 | Löschung | `ja` | `*(nicht da)*` | …das nötigste und [___] versuche halt am… |
| 415 | Löschung | `versuche` | `*(nicht da)*` | …nötigste und ja [___] halt am heimtrainer… |
| 416 | Löschung | `halt` | `*(nicht da)*` | …und ja versuche [___] am heimtrainer ab… |
| 417 | Löschung | `am` | `*(nicht da)*` | …ja versuche halt [___] heimtrainer ab und… |
| 418 | Löschung | `heimtrainer` | `*(nicht da)*` | …versuche halt am [___] ab und zu… |
| 419 | Löschung | `ab` | `*(nicht da)*` | …halt am heimtrainer [___] und zu so… |
| 420 | Löschung | `und` | `*(nicht da)*` | …am heimtrainer ab [___] zu so weit… |
| 421 | Löschung | `zu` | `*(nicht da)*` | …heimtrainer ab und [___] so weit wie… |
| 422 | Löschung | `so` | `*(nicht da)*` | …ab und zu [___] weit wie möglich… |
| 423 | Löschung | `weit` | `*(nicht da)*` | …und zu so [___] wie möglich zu… |
| 424 | Löschung | `wie` | `*(nicht da)*` | …zu so weit [___] möglich zu beugen… |
| 425 | Löschung | `möglich` | `*(nicht da)*` | …so weit wie [___] zu beugen und… |
| 426 | Löschung | `zu` | `*(nicht da)*` | …weit wie möglich [___] beugen und das… |
| 427 | Löschung | `beugen` | `*(nicht da)*` | …wie möglich zu [___] und das eigentlich… |
| 428 | Löschung | `und` | `*(nicht da)*` | …möglich zu beugen [___] das eigentlich immer… |
| 429 | Löschung | `das` | `*(nicht da)*` | …zu beugen und [___] eigentlich immer unter… |
| 430 | Löschung | `eigentlich` | `*(nicht da)*` | …beugen und das [___] immer unter schmerzen… |
| 431 | Löschung | `immer` | `*(nicht da)*` | …und das eigentlich [___] unter schmerzen dann… |
| 432 | Löschung | `unter` | `*(nicht da)*` | …das eigentlich immer [___] schmerzen dann wenn… |
| 433 | Löschung | `schmerzen` | `*(nicht da)*` | …eigentlich immer unter [___] dann wenn man… |
| 434 | Löschung | `dann` | `*(nicht da)*` | …immer unter schmerzen [___] wenn man sagt… |
| 435 | Löschung | `wenn` | `*(nicht da)*` | …unter schmerzen dann [___] man sagt mit… |
| 436 | Löschung | `man` | `*(nicht da)*` | …schmerzen dann wenn [___] sagt mit der… |
| 437 | Löschung | `sagt` | `*(nicht da)*` | …dann wenn man [___] mit der belastung… |
| 438 | Löschung | `mit` | `*(nicht da)*` | …wenn man sagt [___] der belastung variiert… |
| 439 | Löschung | `der` | `*(nicht da)*` | …man sagt mit [___] belastung variiert aber… |
| 440 | Löschung | `belastung` | `*(nicht da)*` | …sagt mit der [___] variiert aber ist… |
| 441 | Löschung | `variiert` | `*(nicht da)*` | …mit der belastung [___] aber ist noch… |
| 442 | Löschung | `aber` | `*(nicht da)*` | …der belastung variiert [___] ist noch nicht… |
| 443 | Löschung | `ist` | `*(nicht da)*` | …belastung variiert aber [___] noch nicht richtig… |
| 444 | Löschung | `noch` | `*(nicht da)*` | …variiert aber ist [___] nicht richtig schmerzfrei… |
| 445 | Löschung | `nicht` | `*(nicht da)*` | …aber ist noch [___] richtig schmerzfrei möglich… |
| 446 | Löschung | `richtig` | `*(nicht da)*` | …ist noch nicht [___] schmerzfrei möglich nein… |
| 447 | Löschung | `schmerzfrei` | `*(nicht da)*` | …noch nicht richtig [___] möglich nein nehmen… |
| 448 | Löschung | `möglich` | `*(nicht da)*` | …nicht richtig schmerzfrei [___] nein nehmen sie… |
| 449 | Löschung | `nein` | `*(nicht da)*` | …richtig schmerzfrei möglich [___] nehmen sie irgendwelche… |
| 450 | Löschung | `nehmen` | `*(nicht da)*` | …schmerzfrei möglich nein [___] sie irgendwelche medikamente… |
| 451 | Löschung | `sie` | `*(nicht da)*` | …möglich nein nehmen [___] irgendwelche medikamente nein… |
| 452 | Löschung | `irgendwelche` | `*(nicht da)*` | …nein nehmen sie [___] medikamente nein nehmen… |
| 453 | Löschung | `medikamente` | `*(nicht da)*` | …nehmen sie irgendwelche [___] nein nehmen sie… |
| 454 | Löschung | `nein` | `*(nicht da)*` | …sie irgendwelche medikamente [___] nehmen sie nichts… |
| 455 | Löschung | `nehmen` | `*(nicht da)*` | …irgendwelche medikamente nein [___] sie nichts nehmen… |
| 456 | Löschung | `sie` | `*(nicht da)*` | …medikamente nein nehmen [___] nichts nehmen sie… |
| 457 | Löschung | `nichts` | `*(nicht da)*` | …nein nehmen sie [___] nehmen sie nichts… |
| 458 | Löschung | `nehmen` | `*(nicht da)*` | …nehmen sie nichts [___] sie nichts haben… |
| 459 | Löschung | `sie` | `*(nicht da)*` | …sie nichts nehmen [___] nichts haben sie… |
| 460 | Löschung | `nichts` | `*(nicht da)*` | …nichts nehmen sie [___] haben sie anfangs… |
| 461 | Löschung | `haben` | `*(nicht da)*` | …nehmen sie nichts [___] sie anfangs aber… |
| 462 | Löschung | `sie` | `*(nicht da)*` | …sie nichts haben [___] anfangs aber wahrscheinlich… |
| 463 | Löschung | `anfangs` | `*(nicht da)*` | …nichts haben sie [___] aber wahrscheinlich eine… |
| 464 | Löschung | `aber` | `*(nicht da)*` | …haben sie anfangs [___] wahrscheinlich eine behandlung… |
| 465 | Löschung | `wahrscheinlich` | `*(nicht da)*` | …sie anfangs aber [___] eine behandlung ja… |
| 466 | Löschung | `eine` | `*(nicht da)*` | …anfangs aber wahrscheinlich [___] behandlung ja ich… |
| 467 | Löschung | `behandlung` | `*(nicht da)*` | …aber wahrscheinlich eine [___] ja ich habe… |
| 468 | Löschung | `ja` | `*(nicht da)*` | …wahrscheinlich eine behandlung [___] ich habe manchmal… |
| 469 | Löschung | `ich` | `*(nicht da)*` | …eine behandlung ja [___] habe manchmal schmerzmittel… |
| 470 | Löschung | `habe` | `*(nicht da)*` | …behandlung ja ich [___] manchmal schmerzmittel gekriegt… |
| 471 | Löschung | `manchmal` | `*(nicht da)*` | …ja ich habe [___] schmerzmittel gekriegt am… |
| 472 | Löschung | `schmerzmittel` | `*(nicht da)*` | …ich habe manchmal [___] gekriegt am anfang… |
| 473 | Löschung | `gekriegt` | `*(nicht da)*` | …habe manchmal schmerzmittel [___] am anfang sowieso… |
| 474 | Löschung | `am` | `*(nicht da)*` | …manchmal schmerzmittel gekriegt [___] anfang sowieso infusionen… |
| 475 | Löschung | `anfang` | `*(nicht da)*` | …schmerzmittel gekriegt am [___] sowieso infusionen gemacht… |
| 476 | Löschung | `sowieso` | `*(nicht da)*` | …gekriegt am anfang [___] infusionen gemacht dann… |
| 477 | Löschung | `infusionen` | `*(nicht da)*` | …am anfang sowieso [___] gemacht dann hätte… |
| 478 | Löschung | `gemacht` | `*(nicht da)*` | …anfang sowieso infusionen [___] dann hätte ich… |
| 479 | Löschung | `dann` | `*(nicht da)*` | …sowieso infusionen gemacht [___] hätte ich nochmal… |
| 480 | Löschung | `hätte` | `*(nicht da)*` | …infusionen gemacht dann [___] ich nochmal schmerzmittel… |
| 481 | Löschung | `ich` | `*(nicht da)*` | …gemacht dann hätte [___] nochmal schmerzmittel mitgehabt… |
| 482 | Löschung | `nochmal` | `*(nicht da)*` | …dann hätte ich [___] schmerzmittel mitgehabt für… |
| 483 | Löschung | `schmerzmittel` | `*(nicht da)*` | …hätte ich nochmal [___] mitgehabt für daheim… |
| 484 | Löschung | `mitgehabt` | `*(nicht da)*` | …ich nochmal schmerzmittel [___] für daheim aber… |
| 485 | Löschung | `für` | `*(nicht da)*` | …nochmal schmerzmittel mitgehabt [___] daheim aber die… |
| 486 | Löschung | `daheim` | `*(nicht da)*` | …schmerzmittel mitgehabt für [___] aber die habe… |
| 487 | Löschung | `aber` | `*(nicht da)*` | …mitgehabt für daheim [___] die habe ich… |
| 488 | Löschung | `die` | `*(nicht da)*` | …für daheim aber [___] habe ich dann… |
| 489 | Löschung | `habe` | `*(nicht da)*` | …daheim aber die [___] ich dann eigentlich… |
| 490 | Löschung | `ich` | `*(nicht da)*` | …aber die habe [___] dann eigentlich nicht… |
| 491 | Löschung | `dann` | `*(nicht da)*` | …die habe ich [___] eigentlich nicht mehr… |
| 492 | Löschung | `eigentlich` | `*(nicht da)*` | …habe ich dann [___] nicht mehr braucht… |
| 493 | Löschung | `nicht` | `*(nicht da)*` | …ich dann eigentlich [___] mehr braucht also… |
| 494 | Löschung | `mehr` | `*(nicht da)*` | …dann eigentlich nicht [___] braucht also haben… |
| 495 | Löschung | `braucht` | `*(nicht da)*` | …eigentlich nicht mehr [___] also haben sie… |
| 496 | Löschung | `also` | `*(nicht da)*` | …nicht mehr braucht [___] haben sie das… |
| 497 | Löschung | `haben` | `*(nicht da)*` | …mehr braucht also [___] sie das benötigt… |
| 498 | Löschung | `sie` | `*(nicht da)*` | …braucht also haben [___] das benötigt mit… |
| 499 | Löschung | `das` | `*(nicht da)*` | …also haben sie [___] benötigt mit dem… |
| 500 | Löschung | `benötigt` | `*(nicht da)*` | …haben sie das [___] mit dem hometrainer… |
| 501 | Löschung | `mit` | `*(nicht da)*` | …sie das benötigt [___] dem hometrainer haben… |
| 502 | Löschung | `dem` | `*(nicht da)*` | …das benötigt mit [___] hometrainer haben sie… |
| 503 | Löschung | `hometrainer` | `*(nicht da)*` | …benötigt mit dem [___] haben sie erwähnt… |
| 504 | Löschung | `haben` | `*(nicht da)*` | …mit dem hometrainer [___] sie erwähnt was… |
| 505 | Löschung | `sie` | `*(nicht da)*` | …dem hometrainer haben [___] erwähnt was haben… |
| 506 | Löschung | `erwähnt` | `*(nicht da)*` | …hometrainer haben sie [___] was haben sie… |
| 507 | Löschung | `was` | `*(nicht da)*` | …haben sie erwähnt [___] haben sie da… |
| 508 | Löschung | `haben` | `*(nicht da)*` | …sie erwähnt was [___] sie da genau… |
| 509 | Löschung | `sie` | `*(nicht da)*` | …erwähnt was haben [___] da genau gemacht… |
| 510 | Löschung | `da` | `*(nicht da)*` | …was haben sie [___] genau gemacht für… |
| 511 | Löschung | `genau` | `*(nicht da)*` | …haben sie da [___] gemacht für übungen… |
| 512 | Löschung | `gemacht` | `*(nicht da)*` | …sie da genau [___] für übungen nein… |
| 513 | Löschung | `für` | `*(nicht da)*` | …da genau gemacht [___] übungen nein eigentlich… |
| 514 | Löschung | `übungen` | `*(nicht da)*` | …genau gemacht für [___] nein eigentlich nur… |
| 515 | Löschung | `nein` | `*(nicht da)*` | …gemacht für übungen [___] eigentlich nur versucht… |
| 516 | Löschung | `eigentlich` | `*(nicht da)*` | …für übungen nein [___] nur versucht weil… |
| 517 | Löschung | `nur` | `*(nicht da)*` | …übungen nein eigentlich [___] versucht weil ich… |
| 518 | Löschung | `versucht` | `*(nicht da)*` | …nein eigentlich nur [___] weil ich eben… |
| 519 | Löschung | `weil` | `*(nicht da)*` | …eigentlich nur versucht [___] ich eben schon… |
| 520 | Löschung | `ich` | `*(nicht da)*` | …nur versucht weil [___] eben schon beugen… |
| 521 | Löschung | `eben` | `*(nicht da)*` | …versucht weil ich [___] schon beugen und… |
| 522 | Löschung | `schon` | `*(nicht da)*` | …weil ich eben [___] beugen und strecken… |
| 523 | Löschung | `beugen` | `*(nicht da)*` | …ich eben schon [___] und strecken kann… |
| 524 | Löschung | `und` | `*(nicht da)*` | …eben schon beugen [___] strecken kann also… |
| 525 | Löschung | `strecken` | `*(nicht da)*` | …schon beugen und [___] kann also dass… |
| 526 | Löschung | `kann` | `*(nicht da)*` | …beugen und strecken [___] also dass ich… |
| 527 | Löschung | `also` | `*(nicht da)*` | …und strecken kann [___] dass ich ein… |
| 528 | Löschung | `dass` | `*(nicht da)*` | …strecken kann also [___] ich ein bisschen… |
| 529 | Löschung | `ich` | `*(nicht da)*` | …kann also dass [___] ein bisschen bewegung… |
| 530 | Löschung | `ein` | `*(nicht da)*` | …also dass ich [___] bisschen bewegung habe… |
| 531 | Löschung | `bisschen` | `*(nicht da)*` | …dass ich ein [___] bewegung habe drinnen… |
| 532 | Löschung | `bewegung` | `*(nicht da)*` | …ich ein bisschen [___] habe drinnen halt… |
| 533 | Löschung | `habe` | `*(nicht da)*` | …ein bisschen bewegung [___] drinnen halt und… |
| 534 | Löschung | `drinnen` | `*(nicht da)*` | …bisschen bewegung habe [___] halt und da… |
| 535 | Löschung | `halt` | `*(nicht da)*` | …bewegung habe drinnen [___] und da ist… |
| 536 | Löschung | `und` | `*(nicht da)*` | …habe drinnen halt [___] da ist schon… |
| 537 | Löschung | `da` | `*(nicht da)*` | …drinnen halt und [___] ist schon ist… |
| 538 | Löschung | `ist` | `*(nicht da)*` | …halt und da [___] schon ist ihnen… |
| 539 | Löschung | `schon` | `*(nicht da)*` | …und da ist [___] ist ihnen da… |
| 540 | Löschung | `ist` | `*(nicht da)*` | …da ist schon [___] ihnen da auch… |
| 541 | Löschung | `ihnen` | `*(nicht da)*` | …ist schon ist [___] da auch aufgefallen… |
| 542 | Löschung | `da` | `*(nicht da)*` | …schon ist ihnen [___] auch aufgefallen dass… |
| 543 | Löschung | `auch` | `*(nicht da)*` | …ist ihnen da [___] aufgefallen dass es… |
| 544 | Löschung | `aufgefallen` | `*(nicht da)*` | …ihnen da auch [___] dass es einfach… |
| 545 | Löschung | `dass` | `*(nicht da)*` | …da auch aufgefallen [___] es einfach schon… |
| 546 | Löschung | `es` | `*(nicht da)*` | …auch aufgefallen dass [___] einfach schon ein… |
| 547 | Löschung | `einfach` | `*(nicht da)*` | …aufgefallen dass es [___] schon ein bisschen… |
| 548 | Löschung | `schon` | `*(nicht da)*` | …dass es einfach [___] ein bisschen weitergegangen… |
| 549 | Löschung | `ein` | `*(nicht da)*` | …es einfach schon [___] bisschen weitergegangen ist… |
| 550 | Löschung | `bisschen` | `*(nicht da)*` | …einfach schon ein [___] weitergegangen ist ja… |
| 551 | Löschung | `weitergegangen` | `*(nicht da)*` | …schon ein bisschen [___] ist ja die… |
| 552 | Löschung | `ist` | `*(nicht da)*` | …ein bisschen weitergegangen [___] ja die bewegung… |
| 553 | Löschung | `ja` | `*(nicht da)*` | …bisschen weitergegangen ist [___] die bewegung auf… |
| 554 | Löschung | `die` | `*(nicht da)*` | …weitergegangen ist ja [___] bewegung auf jeden… |
| 555 | Löschung | `bewegung` | `*(nicht da)*` | …ist ja die [___] auf jeden fall… |
| 556 | Löschung | `auf` | `*(nicht da)*` | …ja die bewegung [___] jeden fall besser… |
| 557 | Löschung | `jeden` | `*(nicht da)*` | …die bewegung auf [___] fall besser als… |
| 558 | Löschung | `fall` | `*(nicht da)*` | …bewegung auf jeden [___] besser als am… |
| 559 | Löschung | `besser` | `*(nicht da)*` | …auf jeden fall [___] als am anfang… |
| 560 | Löschung | `als` | `*(nicht da)*` | …jeden fall besser [___] am anfang also… |
| 561 | Löschung | `am` | `*(nicht da)*` | …fall besser als [___] anfang also sie… |
| 562 | Löschung | `anfang` | `*(nicht da)*` | …besser als am [___] also sie haben… |
| 563 | Löschung | `also` | `*(nicht da)*` | …als am anfang [___] sie haben auch… |
| 564 | Löschung | `sie` | `*(nicht da)*` | …am anfang also [___] haben auch fortschritte… |
| 565 | Löschung | `haben` | `*(nicht da)*` | …anfang also sie [___] auch fortschritte bemerkt… |
| 566 | Löschung | `auch` | `*(nicht da)*` | …also sie haben [___] fortschritte bemerkt ja… |
| 567 | Löschung | `fortschritte` | `*(nicht da)*` | …sie haben auch [___] bemerkt ja nur… |
| 568 | Löschung | `bemerkt` | `*(nicht da)*` | …haben auch fortschritte [___] ja nur zu… |
| 569 | Substitution | `ja` | `pause` | …auch fortschritte bemerkt [___] nur zu ihrer… |
| 570 | Substitution | `nur` | `pause` | …fortschritte bemerkt ja [___] zu ihrer wohnsituation… |
| 571 | Substitution | `zu` | `pause` | …bemerkt ja nur [___] ihrer wohnsituation sie… |
| 572 | Substitution | `ihrer` | `pause` | …ja nur zu [___] wohnsituation sie wohnen… |
| 573 | Substitution | `wohnsituation` | `pause` | …nur zu ihrer [___] sie wohnen in… |
| 574 | Substitution | `sie` | `pause` | …zu ihrer wohnsituation [___] wohnen in einem… |
| 575 | Substitution | `wohnen` | `pause` | …ihrer wohnsituation sie [___] in einem haus… |
| 576 | Substitution | `in` | `pause` | …wohnsituation sie wohnen [___] einem haus in… |
| 577 | Substitution | `einem` | `pause` | …sie wohnen in [___] haus in einer… |
| 578 | Substitution | `haus` | `pause` | …wohnen in einem [___] in einer wohnung… |
| 579 | Substitution | `in` | `pause` | …in einem haus [___] einer wohnung in… |
| 580 | Substitution | `einer` | `pause` | …einem haus in [___] wohnung in einer… |
| 581 | Substitution | `wohnung` | `pause` | …haus in einer [___] in einer wohnung… |
| 582 | Substitution | `in` | `pause` | …in einer wohnung [___] einer wohnung in… |
| 583 | Substitution | `einer` | `pause` | …einer wohnung in [___] wohnung in einer… |
| 584 | Substitution | `wohnung` | `pause` | …wohnung in einer [___] in einer wohnung… |
| 585 | Substitution | `in` | `pause` | …in einer wohnung [___] einer wohnung haben… |
| 586 | Substitution | `einer` | `pause` | …einer wohnung in [___] wohnung haben sie… |
| 587 | Substitution | `wohnung` | `pause` | …wohnung in einer [___] haben sie da… |
| 588 | Substitution | `haben` | `pause` | …in einer wohnung [___] sie da treppen… |
| 589 | Substitution | `sie` | `pause` | …einer wohnung haben [___] da treppen ja… |
| 590 | Substitution | `da` | `pause` | …wohnung haben sie [___] treppen ja im… |
| 591 | Substitution | `treppen` | `pause` | …haben sie da [___] ja im zweiten… |
| 592 | Substitution | `ja` | `pause` | …sie da treppen [___] im zweiten stock… |
| 593 | Substitution | `im` | `pause` | …da treppen ja [___] zweiten stock also… |
| 594 | Substitution | `zweiten` | `pause` | …treppen ja im [___] stock also sie… |
| 595 | Substitution | `stock` | `pause` | …ja im zweiten [___] also sie sind… |
| 596 | Substitution | `also` | `pause` | …im zweiten stock [___] sie sind im… |
| 597 | Substitution | `sie` | `pause` | …zweiten stock also [___] sind im zweiten… |
| 598 | Substitution | `sind` | `pause` | …stock also sie [___] im zweiten stock… |
| 599 | Substitution | `im` | `pause` | …also sie sind [___] zweiten stock das… |
| 600 | Substitution | `zweiten` | `pause` | …sie sind im [___] stock das heißt… |
| 601 | Substitution | `stock` | `pause` | …sind im zweiten [___] das heißt wie… |
| 602 | Substitution | `das` | `pause` | …im zweiten stock [___] heißt wie viele… |
| 603 | Substitution | `heißt` | `pause` | …zweiten stock das [___] wie viele treppen… |
| 604 | Substitution | `wie` | `pause` | …stock das heißt [___] viele treppen werden… |
| 605 | Substitution | `viele` | `pause` | …das heißt wie [___] treppen werden das… |
| 606 | Substitution | `treppen` | `pause` | …heißt wie viele [___] werden das ungefähr… |
| 607 | Substitution | `werden` | `pause` | …wie viele treppen [___] das ungefähr sein… |
| 608 | Substitution | `das` | `pause` | …viele treppen werden [___] ungefähr sein 20… |
| 609 | Substitution | `ungefähr` | `pause` | …treppen werden das [___] sein 20 bis… |
| 610 | Substitution | `sein` | `pause` | …werden das ungefähr [___] 20 bis 30… |
| 611 | Substitution | `20` | `pause` | …das ungefähr sein [___] bis 30 20… |
| 612 | Substitution | `bis` | `pause` | …ungefähr sein 20 [___] 30 20 bis… |
| 613 | Substitution | `30` | `pause` | …sein 20 bis [___] 20 bis 30… |
| 614 | Substitution | `20` | `pause` | …20 bis 30 [___] bis 30 treppen… |
| 615 | Substitution | `bis` | `pause` | …bis 30 20 [___] 30 treppen und… |
| 616 | Substitution | `30` | `pause` | …30 20 bis [___] treppen und das… |
| 617 | Substitution | `treppen` | `pause` | …20 bis 30 [___] und das hat… |
| 618 | Substitution | `und` | `pause` | …bis 30 treppen [___] das hat bis… |
| 619 | Substitution | `das` | `pause` | …30 treppen und [___] hat bis jetzt… |
| 620 | Substitution | `hat` | `pause` | …treppen und das [___] bis jetzt noch… |
| 621 | Substitution | `bis` | `pause` | …und das hat [___] jetzt noch nie… |
| 622 | Substitution | `jetzt` | `pause` | …das hat bis [___] noch nie so… |
| 623 | Substitution | `noch` | `pause` | …hat bis jetzt [___] nie so funktioniert… |
| 624 | Substitution | `nie` | `pause` | …bis jetzt noch [___] so funktioniert haben… |
| 625 | Substitution | `so` | `pause` | …jetzt noch nie [___] funktioniert haben sie… |
| 626 | Substitution | `funktioniert` | `pause` | …noch nie so [___] haben sie gesagt… |
| 627 | Substitution | `haben` | `pause` | …nie so funktioniert [___] sie gesagt ja… |
| 628 | Substitution | `sie` | `pause` | …so funktioniert haben [___] gesagt ja es… |
| 629 | Substitution | `gesagt` | `pause` | …funktioniert haben sie [___] ja es ist… |
| 630 | Substitution | `ja` | `pause` | …haben sie gesagt [___] es ist halt… |
| 631 | Substitution | `es` | `pause` | …sie gesagt ja [___] ist halt ich… |
| 632 | Substitution | `ist` | `pause` | …gesagt ja es [___] halt ich meine… |
| 633 | Substitution | `halt` | `pause` | …ja es ist [___] ich meine sicher… |
| 634 | Substitution | `ich` | `pause` | …es ist halt [___] meine sicher funktioniert… |
| 635 | Substitution | `meine` | `pause` | …ist halt ich [___] sicher funktioniert es… |
| 636 | Substitution | `sicher` | `pause` | …halt ich meine [___] funktioniert es aber… |
| 637 | Substitution | `funktioniert` | `pause` | …ich meine sicher [___] es aber ich… |
| 638 | Substitution | `es` | `pause` | …meine sicher funktioniert [___] aber ich überlege… |
| 639 | Substitution | `aber` | `pause` | …sicher funktioniert es [___] ich überlege mir… |
| 640 | Substitution | `ich` | `pause` | …funktioniert es aber [___] überlege mir halt… |
| 641 | Substitution | `überlege` | `pause` | …es aber ich [___] mir halt ob… |
| 642 | Substitution | `mir` | `pause` | …aber ich überlege [___] halt ob ich… |
| 643 | Substitution | `halt` | `pause` | …ich überlege mir [___] ob ich jetzt… |
| 644 | Substitution | `ob` | `pause` | …überlege mir halt [___] ich jetzt wirklich… |
| 645 | Substitution | `ich` | `pause` | …mir halt ob [___] jetzt wirklich runter… |
| 646 | Substitution | `jetzt` | `pause` | …halt ob ich [___] wirklich runter muss… |
| 647 | Substitution | `wirklich` | `pause` | …ob ich jetzt [___] runter muss ja… |
| 648 | Substitution | `runter` | `pause` | …ich jetzt wirklich [___] muss ja haben… |
| 649 | Substitution | `muss` | `pause` | …jetzt wirklich runter [___] ja haben sie… |
| 650 | Substitution | `ja` | `pause` | …wirklich runter muss [___] haben sie irgendeine… |
| 651 | Substitution | `haben` | `pause` | …runter muss ja [___] sie irgendeine unterstützung… |
| 652 | Substitution | `sie` | `pause` | …muss ja haben [___] irgendeine unterstützung die… |
| 653 | Substitution | `irgendeine` | `pause` | …ja haben sie [___] unterstützung die ihnen… |
| 654 | Substitution | `unterstützung` | `pause` | …haben sie irgendeine [___] die ihnen hilft… |
| 655 | Substitution | `die` | `pause` | …sie irgendeine unterstützung [___] ihnen hilft ja… |
| 656 | Substitution | `ihnen` | `pause` | …irgendeine unterstützung die [___] hilft ja also… |
| 657 | Substitution | `hilft` | `pause` | …unterstützung die ihnen [___] ja also familie… |
| 658 | Substitution | `ja` | `pause` | …die ihnen hilft [___] also familie freunde… |
| 659 | Substitution | `also` | `pause` | …ihnen hilft ja [___] familie freunde verwandte… |
| 660 | Substitution | `familie` | `pause` | …hilft ja also [___] freunde verwandte okay… |
| 661 | Substitution | `freunde` | `pause` | …ja also familie [___] verwandte okay haben… |
| 662 | Substitution | `verwandte` | `pause` | …also familie freunde [___] okay haben sie… |
| 663 | Substitution | `okay` | `pause` | …familie freunde verwandte [___] haben sie irgendwelche… |
| 664 | Substitution | `haben` | `pause` | …freunde verwandte okay [___] sie irgendwelche nebendiagnosen… |
| 665 | Substitution | `sie` | `pause` | …verwandte okay haben [___] irgendwelche nebendiagnosen wie… |
| 666 | Substitution | `irgendwelche` | `pause` | …okay haben sie [___] nebendiagnosen wie ein… |
| 667 | Substitution | `nebendiagnosen` | `pause` | …haben sie irgendwelche [___] wie ein beispiel… |
| 668 | Substitution | `wie` | `pause` | …sie irgendwelche nebendiagnosen [___] ein beispiel diabetes… |
| 669 | Substitution | `ein` | `pause` | …irgendwelche nebendiagnosen wie [___] beispiel diabetes oder… |
| 670 | Substitution | `beispiel` | `pause` | …nebendiagnosen wie ein [___] diabetes oder dass… |
| 671 | Substitution | `diabetes` | `pause` | …wie ein beispiel [___] oder dass sie… |
| 672 | Substitution | `oder` | `pause` | …ein beispiel diabetes [___] dass sie wüssten… |
| 673 | Substitution | `dass` | `pause` | …beispiel diabetes oder [___] sie wüssten auch… |
| 674 | Substitution | `sie` | `pause` | …diabetes oder dass [___] wüssten auch keine… |
| 675 | Substitution | `wüssten` | `pause` | …oder dass sie [___] auch keine auffälligkeiten… |
| 676 | Substitution | `auch` | `pause` | …dass sie wüssten [___] keine auffälligkeiten und… |
| 677 | Substitution | `keine` | `pause` | …sie wüssten auch [___] auffälligkeiten und sie… |
| 678 | Substitution | `auffälligkeiten` | `pause` | …wüssten auch keine [___] und sie stängern… |
| 679 | Substitution | `und` | `pause` | …auch keine auffälligkeiten [___] sie stängern auch… |
| 680 | Substitution | `sie` | `pause` | …keine auffälligkeiten und [___] stängern auch sonst… |
| 681 | Substitution | `stängern` | `pause` | …auffälligkeiten und sie [___] auch sonst nicht… |
| 682 | Substitution | `auch` | `pause` | …und sie stängern [___] sonst nicht unter… |
| 683 | Substitution | `sonst` | `pause` | …sie stängern auch [___] nicht unter medikamente… |
| 684 | Substitution | `nicht` | `pause` | …stängern auch sonst [___] unter medikamente also… |
| 685 | Substitution | `unter` | `pause` | …auch sonst nicht [___] medikamente also abgesehen… |
| 686 | Substitution | `medikamente` | `pause` | …sonst nicht unter [___] also abgesehen von… |
| 687 | Substitution | `also` | `pause` | …nicht unter medikamente [___] abgesehen von der… |
| 688 | Substitution | `abgesehen` | `pause` | …unter medikamente also [___] von der schmerztabletten… |
| 689 | Substitution | `von` | `pause` | …medikamente also abgesehen [___] der schmerztabletten nein… |
| 690 | Substitution | `der` | `pause` | …also abgesehen von [___] schmerztabletten nein nein… |
| 691 | Substitution | `schmerztabletten` | `pause` | …abgesehen von der [___] nein nein die… |
| 692 | Substitution | `nein` | `pause` | …von der schmerztabletten [___] nein die nicht… |
| 693 | Substitution | `nein` | `pause` | …der schmerztabletten nein [___] die nicht mehr… |
| 694 | Substitution | `die` | `pause` | …schmerztabletten nein nein [___] nicht mehr okay… |
| 695 | Substitution | `nicht` | `pause` | …nein nein die [___] mehr okay und… |
| 696 | Substitution | `mehr` | `pause` | …nein die nicht [___] okay und dann… |
| 697 | Substitution | `okay` | `pause` | …die nicht mehr [___] und dann nur… |
| 698 | Substitution | `und` | `pause` | …nicht mehr okay [___] dann nur zum… |
| 699 | Substitution | `dann` | `pause` | …mehr okay und [___] nur zum abschluss… |
| 700 | Substitution | `nur` | `pause` | …okay und dann [___] zum abschluss was… |
| 701 | Substitution | `zum` | `pause` | …und dann nur [___] abschluss was ist… |
| 702 | Substitution | `abschluss` | `pause` | …dann nur zum [___] was ist denn… |
| 703 | Substitution | `was` | `pause` | …nur zum abschluss [___] ist denn ihr… |
| 704 | Substitution | `ist` | `pause` | …zum abschluss was [___] denn ihr persönliches… |
| 705 | Substitution | `denn` | `pause` | …abschluss was ist [___] ihr persönliches hauptziel… |
| 706 | Substitution | `ihr` | `pause` | …was ist denn [___] persönliches hauptziel für… |
| 707 | Substitution | `persönliches` | `pause` | …ist denn ihr [___] hauptziel für die… |
| 708 | Substitution | `hauptziel` | `pause` | …denn ihr persönliches [___] für die therapie… |
| 709 | Substitution | `für` | `pause` | …ihr persönliches hauptziel [___] die therapie was… |
| 710 | Substitution | `die` | `pause` | …persönliches hauptziel für [___] therapie was würden… |
| 711 | Substitution | `therapie` | `pause` | …hauptziel für die [___] was würden sie… |
| 712 | Substitution | `was` | `pause` | …für die therapie [___] würden sie sie… |
| 713 | Substitution | `würden` | `pause` | …die therapie was [___] sie sie wünschen… |
| 714 | Substitution | `sie` | `pause` | …therapie was würden [___] sie wünschen was… |
| 715 | Substitution | `sie` | `pause` | …was würden sie [___] wünschen was erwarten… |
| 716 | Substitution | `wünschen` | `pause` | …würden sie sie [___] was erwarten sie… |
| 717 | Substitution | `was` | `pause` | …sie sie wünschen [___] erwarten sie sie… |
| 718 | Substitution | `erwarten` | `pause` | …sie wünschen was [___] sie sie dass… |
| 719 | Substitution | `sie` | `pause` | …wünschen was erwarten [___] sie dass ich… |
| 720 | Substitution | `sie` | `pause` | …was erwarten sie [___] dass ich halt… |
| 721 | Substitution | `dass` | `pause` | …erwarten sie sie [___] ich halt alles… |
| 722 | Substitution | `ich` | `pause` | …sie sie dass [___] halt alles wieder… |
| 723 | Substitution | `halt` | `pause` | …sie dass ich [___] alles wieder normal… |
| 724 | Substitution | `alles` | `pause` | …dass ich halt [___] wieder normal machen… |
| 725 | Substitution | `wieder` | `pause` | …ich halt alles [___] normal machen kann… |
| 726 | Substitution | `normal` | `pause` | …halt alles wieder [___] machen kann auch… |
| 727 | Substitution | `machen` | `pause` | …alles wieder normal [___] kann auch dass… |
| 728 | Substitution | `kann` | `pause` | …wieder normal machen [___] auch dass ich… |
| 729 | Substitution | `auch` | `pause` | …normal machen kann [___] dass ich eben… |
| 730 | Substitution | `dass` | `pause` | …machen kann auch [___] ich eben wieder… |
| 731 | Substitution | `ich` | `pause` | …kann auch dass [___] eben wieder volleyball… |
| 732 | Substitution | `eben` | `pause` | …auch dass ich [___] wieder volleyball spielen… |
| 733 | Substitution | `wieder` | `pause` | …dass ich eben [___] volleyball spielen kann… |
| 734 | Substitution | `volleyball` | `pause` | …ich eben wieder [___] spielen kann dass… |
| 735 | Substitution | `spielen` | `pause` | …eben wieder volleyball [___] kann dass ich… |
| 736 | Substitution | `kann` | `pause` | …wieder volleyball spielen [___] dass ich normal… |
| 737 | Substitution | `dass` | `pause` | …volleyball spielen kann [___] ich normal sportlen… |
| 738 | Substitution | `ich` | `pause` | …spielen kann dass [___] normal sportlen kann… |
| 739 | Substitution | `normal` | `pause` | …kann dass ich [___] sportlen kann also… |
| 740 | Substitution | `sportlen` | `pause` | …dass ich normal [___] kann also der… |
| 741 | Substitution | `kann` | `pause` | …ich normal sportlen [___] also der sport… |
| 742 | Substitution | `also` | `pause` | …normal sportlen kann [___] der sport steht… |
| 743 | Substitution | `der` | `pause` | …sportlen kann also [___] sport steht da… |
| 744 | Substitution | `sport` | `pause` | …kann also der [___] steht da auch… |
| 745 | Substitution | `steht` | `pause` | …also der sport [___] da auch im… |
| 746 | Substitution | `da` | `pause` | …der sport steht [___] auch im vordergrund… |
| 747 | Substitution | `auch` | `pause` | …sport steht da [___] im vordergrund ja… |
| 748 | Substitution | `im` | `pause` | …steht da auch [___] vordergrund ja ja… |
| 749 | Substitution | `vordergrund` | `pause` | …da auch im [___] ja ja in… |
| 750 | Substitution | `ja` | `pause` | …auch im vordergrund [___] ja in weiterer… |
| 751 | Substitution | `ja` | `pause` | …im vordergrund ja [___] in weiterer folge… |
| 752 | Substitution | `in` | `pause` | …vordergrund ja ja [___] weiterer folge natürlich… |
| 753 | Substitution | `weiterer` | `pause` | …ja ja in [___] folge natürlich vorher… |
| 754 | Substitution | `folge` | `pause` | …ja in weiterer [___] natürlich vorher möchte… |
| 755 | Substitution | `natürlich` | `pause` | …in weiterer folge [___] vorher möchte ich… |
| 756 | Substitution | `vorher` | `pause` | …weiterer folge natürlich [___] möchte ich mir… |
| 757 | Substitution | `möchte` | `pause` | …folge natürlich vorher [___] ich mir normal… |
| 758 | Substitution | `ich` | `pause` | …natürlich vorher möchte [___] mir normal gehen… |
| 759 | Substitution | `mir` | `pause` | …vorher möchte ich [___] normal gehen können… |
| 760 | Substitution | `normal` | `pause` | …möchte ich mir [___] gehen können keine… |
| 761 | Substitution | `gehen` | `pause` | …ich mir normal [___] können keine lüge… |
| 762 | Substitution | `können` | `pause` | …mir normal gehen [___] keine lüge und… |
| 763 | Substitution | `keine` | `pause` | …normal gehen können [___] lüge und ja… |
| 764 | Substitution | `lüge` | `pause` | …gehen können keine [___] und ja okay… |
| 765 | Substitution | `und` | `pause` | …können keine lüge [___] ja okay vielen… |
| 766 | Substitution | `ja` | `pause` | …keine lüge und [___] okay vielen dank… |
| 767 | Substitution | `okay` | `pause` | …lüge und ja [___] vielen dank frau… |
| 768 | Substitution | `vielen` | `pause` | …und ja okay [___] dank frau krebspartner… |
| 769 | Substitution | `dank` | `pause` | …ja okay vielen [___] frau krebspartner und… |
| 770 | Substitution | `frau` | `pause` | …okay vielen dank [___] krebspartner und wir… |
| 771 | Substitution | `krebspartner` | `pause` | …vielen dank frau [___] und wir treffen… |
| 772 | Substitution | `und` | `pause` | …dank frau krebspartner [___] wir treffen uns… |
| 773 | Substitution | `wir` | `pause` | …frau krebspartner und [___] treffen uns dann… |
| 774 | Substitution | `treffen` | `pause` | …krebspartner und wir [___] uns dann zur… |
| 775 | Substitution | `uns` | `pause` | …und wir treffen [___] dann zur nächsten… |
| 776 | Substitution | `dann` | `pause` | …wir treffen uns [___] zur nächsten behandlung… |
| 777 | Substitution | `zur` | `pause` | …treffen uns dann [___] nächsten behandlung danke… |
| 778 | Substitution | `nächsten` | `pause` | …uns dann zur [___] behandlung danke ich… |
| 779 | Substitution | `behandlung` | `pause` | …dann zur nächsten [___] danke ich hoffe… |
| 780 | Substitution | `danke` | `pause` | …zur nächsten behandlung [___] ich hoffe dass… |
| 781 | Substitution | `ich` | `pause` | …nächsten behandlung danke [___] hoffe dass sie… |
| 782 | Substitution | `hoffe` | `pause` | …behandlung danke ich [___] dass sie das… |
| 783 | Substitution | `dass` | `pause` | …danke ich hoffe [___] sie das passen… |
| 784 | Substitution | `sie` | `pause` | …ich hoffe dass [___] das passen ich… |
| 785 | Substitution | `das` | `pause` | …hoffe dass sie [___] passen ich hoffe… |
| 786 | Substitution | `passen` | `pause` | …dass sie das [___] ich hoffe dass… |
| 787 | Substitution | `ich` | `pause` | …sie das passen [___] hoffe dass sie… |
| 788 | Substitution | `hoffe` | `pause` | …das passen ich [___] dass sie das… |
| 789 | Substitution | `dass` | `pause` | …passen ich hoffe [___] sie das passen… |
| 790 | Substitution | `sie` | `pause` | …ich hoffe dass [___] das passen danke… |
| 791 | Substitution | `das` | `pause` | …hoffe dass sie [___] passen danke danke… |
| 792 | Substitution | `passen` | `pause` | …dass sie das [___] danke danke danke… |
| 793 | Substitution | `danke` | `pause` | …sie das passen [___] danke danke danke… |
| 794 | Substitution | `danke` | `pause` | …das passen danke [___] danke danke… |
| 795 | Substitution | `danke` | `pause` | …passen danke danke [___] danke… |
| 796 | Substitution | `danke` | `pause` | …danke danke danke [___]… |
