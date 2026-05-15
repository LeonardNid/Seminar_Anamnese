# LLM-Fehleranalyse: Whisper + SauerkrautLM 8b

> RAW STT → Formatted — Satzzeichen und Groß-/Kleinschreibung ignoriert. <br>
> Speaker-Label-Änderungen sind bereits aus der JSON entfernt.<br>
> **S** = Substitution | **D** = Löschung (im RAW, fehlt im FMT) | **I** = Einfügung (im FMT, nicht im RAW)<br>

---

## Modell-Informationen

| Komponente | Exakte Bezeichnung | Kontextfenster |
|---|---|---|
| STT | `faster-whisper large-v3-turbo` (lokal, CPU, int8) + pyannote/speaker-diarization-3.1 | — |
| LLM | `hf.co/QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF:Q4_K_M` (Ollama) | 131.072 Tokens (128k) |

---

## Übersicht

| Szenario | RAW-Wörter | FMT-Wörter | S | D | I | Fehler | Fehlerrate |
|---|---|---|---|---|---|---|---|
| OriginalDC | 237 | 247 | 0 | 0 | 10 | 10 | 4.2% |
| OriginalDC+Noise | 210 | 210 | 0 | 0 | 0 | 0 | 0.0% |
| LapInMitte | 226 | 226 | 0 | 0 | 0 | 0 | 0.0% |
| LapBeiArzt | 229 | 230 | 1 | 0 | 1 | 2 | 0.9% |
| Selbstkorrekturen | 210 | 278 | 2 | 3 | 71 | 76 | 36.2% |
| Unterbrechungen | 143 | 145 | 0 | 7 | 9 | 16 | 11.2% |
| Gedankensprünge | 190 | 190 | 0 | 0 | 0 | 0 | 0.0% |
| Meinungswechsel | 185 | 186 | 2 | 0 | 1 | 3 | 1.6% |
| Chaos | 252 | 301 | 0 | 0 | 49 | 49 | 19.4% |
| Anamnesegespräch | 2269 | 152 | 87 | 2117 | 0 | 2204 | 97.1% |
| PWC | 1512 | 1005 | 20 | 595 | 88 | 703 | 46.5% |

---

## OriginalDC

**Fehlerrate: 4.2%** — RAW: 237 Wörter | FMT: 247 Wörter | S=0 D=0 I=10 | Fehler=10

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `kein` | (FMT) …auskultation der pulmonis [___] gesprächsbeitrag mehr da… |
| 2 | Einfügung | `*(nicht da)*` | `gesprächsbeitrag` | (FMT) …der pulmonis kein [___] mehr da es… |
| 3 | Einfügung | `*(nicht da)*` | `mehr` | (FMT) …pulmonis kein gesprächsbeitrag [___] da es sich… |
| 4 | Einfügung | `*(nicht da)*` | `da` | (FMT) …kein gesprächsbeitrag mehr [___] es sich um… |
| 5 | Einfügung | `*(nicht da)*` | `es` | (FMT) …gesprächsbeitrag mehr da [___] sich um eine… |
| 6 | Einfügung | `*(nicht da)*` | `sich` | (FMT) …mehr da es [___] um eine fernsehsendung… |
| 7 | Einfügung | `*(nicht da)*` | `um` | (FMT) …da es sich [___] eine fernsehsendung handeltet… |
| 8 | Einfügung | `*(nicht da)*` | `eine` | (FMT) …es sich um [___] fernsehsendung handeltet auskultation… |
| 9 | Einfügung | `*(nicht da)*` | `fernsehsendung` | (FMT) …sich um eine [___] handeltet auskultation der… |
| 10 | Einfügung | `*(nicht da)*` | `handeltet` | (FMT) …um eine fernsehsendung [___] auskultation der pulmonisse… |

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

**Fehlerrate: 0.9%** — RAW: 229 Wörter | FMT: 230 Wörter | S=1 D=0 I=1 | Fehler=2

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `viel` | (FMT) …brügerie und trinke [___] wasser alles klar… |
| 2 | Substitution | `vielfamilien` | `wasser` | …brügerie und trinke [___] alles klar notiz… |

---

## Selbstkorrekturen

**Fehlerrate: 36.2%** — RAW: 210 Wörter | FMT: 278 Wörter | S=2 D=3 I=71 | Fehler=76

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Substitution | `inャ` | `inя` | …lichtempfindlich dass meine [___] nil r sigt… |
| 2 | Löschung | `nochинاء` | `*(nicht da)*` | …r sigt und [___] sarah hätte hierzu… |
| 3 | Löschung | `sarah` | `*(nicht da)*` | …sigt und nochинاء [___] hätte hierzu für… |
| 4 | Löschung | `hätte` | `*(nicht da)*` | …und nochинاء sarah [___] hierzu für mich… |
| 5 | Substitution | `hierzu` | `noch` | …nochинاء sarah hätte [___] für mich auch… |
| 6 | Einfügung | `*(nicht da)*` | `ich` | (FMT) …an knight besserhemen [___] habe den namen… |
| 7 | Einfügung | `*(nicht da)*` | `habe` | (FMT) …knight besserhemen ich [___] den namen des… |
| 8 | Einfügung | `*(nicht da)*` | `den` | (FMT) …besserhemen ich habe [___] namen des patienten… |
| 9 | Einfügung | `*(nicht da)*` | `namen` | (FMT) …ich habe den [___] des patienten als… |
| 10 | Einfügung | `*(nicht da)*` | `des` | (FMT) …habe den namen [___] patienten als herr… |
| 11 | Einfügung | `*(nicht da)*` | `patienten` | (FMT) …den namen des [___] als herr berger… |
| 12 | Einfügung | `*(nicht da)*` | `als` | (FMT) …namen des patienten [___] herr berger identifiziert… |
| 13 | Einfügung | `*(nicht da)*` | `herr` | (FMT) …des patienten als [___] berger identifiziert aber… |
| 14 | Einfügung | `*(nicht da)*` | `berger` | (FMT) …patienten als herr [___] identifiziert aber in… |
| 15 | Einfügung | `*(nicht da)*` | `identifiziert` | (FMT) …als herr berger [___] aber in der… |
| 16 | Einfügung | `*(nicht da)*` | `aber` | (FMT) …herr berger identifiziert [___] in der letzten… |
| 17 | Einfügung | `*(nicht da)*` | `in` | (FMT) …berger identifiziert aber [___] der letzten zeile… |
| 18 | Einfügung | `*(nicht da)*` | `der` | (FMT) …identifiziert aber in [___] letzten zeile ist… |
| 19 | Einfügung | `*(nicht da)*` | `letzten` | (FMT) …aber in der [___] zeile ist ein… |
| 20 | Einfügung | `*(nicht da)*` | `zeile` | (FMT) …in der letzten [___] ist ein teil… |
| 21 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …der letzten zeile [___] ein teil des… |
| 22 | Einfügung | `*(nicht da)*` | `ein` | (FMT) …letzten zeile ist [___] teil des satzes… |
| 23 | Einfügung | `*(nicht da)*` | `teil` | (FMT) …zeile ist ein [___] des satzes unleserlich… |
| 24 | Einfügung | `*(nicht da)*` | `des` | (FMT) …ist ein teil [___] satzes unleserlich und… |
| 25 | Einfügung | `*(nicht da)*` | `satzes` | (FMT) …ein teil des [___] unleserlich und daher… |
| 26 | Einfügung | `*(nicht da)*` | `unleserlich` | (FMT) …teil des satzes [___] und daher nicht… |
| 27 | Einfügung | `*(nicht da)*` | `und` | (FMT) …des satzes unleserlich [___] daher nicht zu… |
| 28 | Einfügung | `*(nicht da)*` | `daher` | (FMT) …satzes unleserlich und [___] nicht zu verstehen… |
| 29 | Einfügung | `*(nicht da)*` | `nicht` | (FMT) …unleserlich und daher [___] zu verstehen es… |
| 30 | Einfügung | `*(nicht da)*` | `zu` | (FMT) …und daher nicht [___] verstehen es scheint… |
| 31 | Einfügung | `*(nicht da)*` | `verstehen` | (FMT) …daher nicht zu [___] es scheint jedoch… |
| 32 | Einfügung | `*(nicht da)*` | `es` | (FMT) …nicht zu verstehen [___] scheint jedoch dass… |
| 33 | Einfügung | `*(nicht da)*` | `scheint` | (FMT) …zu verstehen es [___] jedoch dass es… |
| 34 | Einfügung | `*(nicht da)*` | `jedoch` | (FMT) …verstehen es scheint [___] dass es sich… |
| 35 | Einfügung | `*(nicht da)*` | `dass` | (FMT) …es scheint jedoch [___] es sich um… |
| 36 | Einfügung | `*(nicht da)*` | `es` | (FMT) …scheint jedoch dass [___] sich um einen… |
| 37 | Einfügung | `*(nicht da)*` | `sich` | (FMT) …jedoch dass es [___] um einen fehler… |
| 38 | Einfügung | `*(nicht da)*` | `um` | (FMT) …dass es sich [___] einen fehler handeltet… |
| 39 | Einfügung | `*(nicht da)*` | `einen` | (FMT) …es sich um [___] fehler handeltet da… |
| 40 | Einfügung | `*(nicht da)*` | `fehler` | (FMT) …sich um einen [___] handeltet da die… |
| 41 | Einfügung | `*(nicht da)*` | `handeltet` | (FMT) …um einen fehler [___] da die sprecher… |
| 42 | Einfügung | `*(nicht da)*` | `da` | (FMT) …einen fehler handeltet [___] die sprecher nummer… |
| 43 | Einfügung | `*(nicht da)*` | `die` | (FMT) …fehler handeltet da [___] sprecher nummer 00… |
| 44 | Einfügung | `*(nicht da)*` | `sprecher` | (FMT) …handeltet da die [___] nummer 00 wieder… |
| 45 | Einfügung | `*(nicht da)*` | `nummer` | (FMT) …da die sprecher [___] 00 wieder verwendet… |
| 46 | Einfügung | `*(nicht da)*` | `00` | (FMT) …die sprecher nummer [___] wieder verwendet wird… |
| 47 | Einfügung | `*(nicht da)*` | `wieder` | (FMT) …sprecher nummer 00 [___] verwendet wird obwohl… |
| 48 | Einfügung | `*(nicht da)*` | `verwendet` | (FMT) …nummer 00 wieder [___] wird obwohl bereits… |
| 49 | Einfügung | `*(nicht da)*` | `wird` | (FMT) …00 wieder verwendet [___] obwohl bereits herr… |
| 50 | Einfügung | `*(nicht da)*` | `obwohl` | (FMT) …wieder verwendet wird [___] bereits herr berger… |
| 51 | Einfügung | `*(nicht da)*` | `bereits` | (FMT) …verwendet wird obwohl [___] herr berger als… |
| 52 | Einfügung | `*(nicht da)*` | `herr` | (FMT) …wird obwohl bereits [___] berger als patient… |
| 53 | Einfügung | `*(nicht da)*` | `berger` | (FMT) …obwohl bereits herr [___] als patient identifiziert… |
| 54 | Einfügung | `*(nicht da)*` | `als` | (FMT) …bereits herr berger [___] patient identifiziert wurde… |
| 55 | Einfügung | `*(nicht da)*` | `patient` | (FMT) …herr berger als [___] identifiziert wurde ich… |
| 56 | Einfügung | `*(nicht da)*` | `identifiziert` | (FMT) …berger als patient [___] wurde ich habe… |
| 57 | Einfügung | `*(nicht da)*` | `wurde` | (FMT) …als patient identifiziert [___] ich habe den… |
| 58 | Einfügung | `*(nicht da)*` | `ich` | (FMT) …patient identifiziert wurde [___] habe den text… |
| 59 | Einfügung | `*(nicht da)*` | `habe` | (FMT) …identifiziert wurde ich [___] den text entsprechend… |
| 60 | Einfügung | `*(nicht da)*` | `den` | (FMT) …wurde ich habe [___] text entsprechend formatiert… |
| 61 | Einfügung | `*(nicht da)*` | `text` | (FMT) …ich habe den [___] entsprechend formatiert aber… |
| 62 | Einfügung | `*(nicht da)*` | `entsprechend` | (FMT) …habe den text [___] formatiert aber ich… |
| 63 | Einfügung | `*(nicht da)*` | `formatiert` | (FMT) …den text entsprechend [___] aber ich möchte… |
| 64 | Einfügung | `*(nicht da)*` | `aber` | (FMT) …text entsprechend formatiert [___] ich möchte darauf… |
| 65 | Einfügung | `*(nicht da)*` | `ich` | (FMT) …entsprechend formatiert aber [___] möchte darauf hinweisen… |
| 66 | Einfügung | `*(nicht da)*` | `möchte` | (FMT) …formatiert aber ich [___] darauf hinweisen dass… |
| 67 | Einfügung | `*(nicht da)*` | `darauf` | (FMT) …aber ich möchte [___] hinweisen dass die… |
| 68 | Einfügung | `*(nicht da)*` | `hinweisen` | (FMT) …ich möchte darauf [___] dass die letzte… |
| 69 | Einfügung | `*(nicht da)*` | `dass` | (FMT) …möchte darauf hinweisen [___] die letzte zeile… |
| 70 | Einfügung | `*(nicht da)*` | `die` | (FMT) …darauf hinweisen dass [___] letzte zeile möglicherweise… |
| 71 | Einfügung | `*(nicht da)*` | `letzte` | (FMT) …hinweisen dass die [___] zeile möglicherweise nicht… |
| 72 | Einfügung | `*(nicht da)*` | `zeile` | (FMT) …dass die letzte [___] möglicherweise nicht korrekt… |
| 73 | Einfügung | `*(nicht da)*` | `möglicherweise` | (FMT) …die letzte zeile [___] nicht korrekt ist… |
| 74 | Einfügung | `*(nicht da)*` | `nicht` | (FMT) …letzte zeile möglicherweise [___] korrekt ist… |
| 75 | Einfügung | `*(nicht da)*` | `korrekt` | (FMT) …zeile möglicherweise nicht [___] ist… |
| 76 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …möglicherweise nicht korrekt [___]… |

---

## Unterbrechungen

**Fehlerrate: 11.2%** — RAW: 143 Wörter | FMT: 145 Wörter | S=0 D=7 I=9 | Fehler=16

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `sehr` | (FMT) …seit wann genau [___] gut seit heute… |
| 2 | Einfügung | `*(nicht da)*` | `gut` | (FMT) …wann genau sehr [___] seit heute nacht… |
| 3 | Löschung | `38` | `*(nicht da)*` | …war es bei [___] 38 was komma… |
| 4 | Löschung | `was` | `*(nicht da)*` | …bei 38 38 [___] komma 2 komma… |
| 5 | Löschung | `komma` | `*(nicht da)*` | …38 38 was [___] 2 komma 5… |
| 6 | Löschung | `2` | `*(nicht da)*` | …38 was komma [___] komma 5 komma… |
| 7 | Löschung | `komma` | `*(nicht da)*` | …was komma 2 [___] 5 komma 3… |
| 8 | Löschung | `5` | `*(nicht da)*` | …komma 2 komma [___] komma 3 und… |
| 9 | Löschung | `komma` | `*(nicht da)*` | …2 komma 5 [___] 3 und ich… |
| 10 | Einfügung | `*(nicht da)*` | `frau` | (FMT) …ich es sagen [___] klein ist der… |
| 11 | Einfügung | `*(nicht da)*` | `klein` | (FMT) …es sagen frau [___] ist der name… |
| 12 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …sagen frau klein [___] der name des… |
| 13 | Einfügung | `*(nicht da)*` | `der` | (FMT) …frau klein ist [___] name des patienten… |
| 14 | Einfügung | `*(nicht da)*` | `name` | (FMT) …klein ist der [___] des patienten… |
| 15 | Einfügung | `*(nicht da)*` | `des` | (FMT) …ist der name [___] patienten… |
| 16 | Einfügung | `*(nicht da)*` | `patienten` | (FMT) …der name des [___]… |

---

## Gedankensprünge

**Fehlerrate: 0.0%** — RAW: 190 Wörter | FMT: 190 Wörter | S=0 D=0 I=0 | Fehler=0

*Keine Fehler gefunden.*

---

## Meinungswechsel

**Fehlerrate: 1.6%** — RAW: 185 Wörter | FMT: 186 Wörter | S=2 D=0 I=1 | Fehler=3

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `kein` | (FMT) …das war s [___] name genannt… |
| 2 | Substitution | `grund` | `name` | …das war s [___] 了吧… |
| 3 | Substitution | `了吧` | `genannt` | …war s grund [___]… |

---

## Chaos

**Fehlerrate: 19.4%** — RAW: 252 Wörter | FMT: 301 Wörter | S=0 D=0 I=49 | Fehler=49

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `1` | (FMT) …[___] der arzt ist… |
| 2 | Einfügung | `*(nicht da)*` | `der` | (FMT) …1 [___] arzt ist der… |
| 3 | Einfügung | `*(nicht da)*` | `arzt` | (FMT) …1 der [___] ist der sprecher… |
| 4 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …1 der arzt [___] der sprecher der… |
| 5 | Einfügung | `*(nicht da)*` | `der` | (FMT) …der arzt ist [___] sprecher der sich… |
| 6 | Einfügung | `*(nicht da)*` | `sprecher` | (FMT) …arzt ist der [___] der sich als… |
| 7 | Einfügung | `*(nicht da)*` | `der` | (FMT) …ist der sprecher [___] sich als bezeichnet… |
| 8 | Einfügung | `*(nicht da)*` | `sich` | (FMT) …der sprecher der [___] als bezeichnet und… |
| 9 | Einfügung | `*(nicht da)*` | `als` | (FMT) …sprecher der sich [___] bezeichnet und der… |
| 10 | Einfügung | `*(nicht da)*` | `bezeichnet` | (FMT) …der sich als [___] und der patient… |
| 11 | Einfügung | `*(nicht da)*` | `und` | (FMT) …sich als bezeichnet [___] der patient ist… |
| 12 | Einfügung | `*(nicht da)*` | `der` | (FMT) …als bezeichnet und [___] patient ist herr… |
| 13 | Einfügung | `*(nicht da)*` | `patient` | (FMT) …bezeichnet und der [___] ist herr schuster… |
| 14 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …und der patient [___] herr schuster 2… |
| 15 | Einfügung | `*(nicht da)*` | `herr` | (FMT) …der patient ist [___] schuster 2 der… |
| 16 | Einfügung | `*(nicht da)*` | `schuster` | (FMT) …patient ist herr [___] 2 der name… |
| 17 | Einfügung | `*(nicht da)*` | `2` | (FMT) …ist herr schuster [___] der name des… |
| 18 | Einfügung | `*(nicht da)*` | `der` | (FMT) …herr schuster 2 [___] name des patienten… |
| 19 | Einfügung | `*(nicht da)*` | `name` | (FMT) …schuster 2 der [___] des patienten ist… |
| 20 | Einfügung | `*(nicht da)*` | `des` | (FMT) …2 der name [___] patienten ist herr… |
| 21 | Einfügung | `*(nicht da)*` | `patienten` | (FMT) …der name des [___] ist herr schuster… |
| 22 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …name des patienten [___] herr schuster 3… |
| 23 | Einfügung | `*(nicht da)*` | `herr` | (FMT) …des patienten ist [___] schuster 3 hier… |
| 24 | Einfügung | `*(nicht da)*` | `schuster` | (FMT) …patienten ist herr [___] 3 hier ist… |
| 25 | Einfügung | `*(nicht da)*` | `3` | (FMT) …ist herr schuster [___] hier ist das… |
| 26 | Einfügung | `*(nicht da)*` | `hier` | (FMT) …herr schuster 3 [___] ist das transkript… |
| 27 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …schuster 3 hier [___] das transkript mit… |
| 28 | Einfügung | `*(nicht da)*` | `das` | (FMT) …3 hier ist [___] transkript mit den… |
| 29 | Einfügung | `*(nicht da)*` | `transkript` | (FMT) …hier ist das [___] mit den generischen… |
| 30 | Einfügung | `*(nicht da)*` | `mit` | (FMT) …ist das transkript [___] den generischen sprecher… |
| 31 | Einfügung | `*(nicht da)*` | `den` | (FMT) …das transkript mit [___] generischen sprecher labels… |
| 32 | Einfügung | `*(nicht da)*` | `generischen` | (FMT) …transkript mit den [___] sprecher labels ersetzt… |
| 33 | Einfügung | `*(nicht da)*` | `sprecher` | (FMT) …mit den generischen [___] labels ersetzt durch… |
| 34 | Einfügung | `*(nicht da)*` | `labels` | (FMT) …den generischen sprecher [___] ersetzt durch arzt… |
| 35 | Einfügung | `*(nicht da)*` | `ersetzt` | (FMT) …generischen sprecher labels [___] durch arzt und… |
| 36 | Einfügung | `*(nicht da)*` | `durch` | (FMT) …sprecher labels ersetzt [___] arzt und name… |
| 37 | Einfügung | `*(nicht da)*` | `arzt` | (FMT) …labels ersetzt durch [___] und name des… |
| 38 | Einfügung | `*(nicht da)*` | `und` | (FMT) …ersetzt durch arzt [___] name des patienten… |
| 39 | Einfügung | `*(nicht da)*` | `name` | (FMT) …durch arzt und [___] des patienten oder… |
| 40 | Einfügung | `*(nicht da)*` | `des` | (FMT) …arzt und name [___] patienten oder patient… |
| 41 | Einfügung | `*(nicht da)*` | `patienten` | (FMT) …und name des [___] oder patient in… |
| 42 | Einfügung | `*(nicht da)*` | `oder` | (FMT) …name des patienten [___] patient in falls… |
| 43 | Einfügung | `*(nicht da)*` | `patient` | (FMT) …des patienten oder [___] in falls kein… |
| 44 | Einfügung | `*(nicht da)*` | `in` | (FMT) …patienten oder patient [___] falls kein name… |
| 45 | Einfügung | `*(nicht da)*` | `falls` | (FMT) …oder patient in [___] kein name genannt… |
| 46 | Einfügung | `*(nicht da)*` | `kein` | (FMT) …patient in falls [___] name genannt wird… |
| 47 | Einfügung | `*(nicht da)*` | `name` | (FMT) …in falls kein [___] genannt wird herr… |
| 48 | Einfügung | `*(nicht da)*` | `genannt` | (FMT) …falls kein name [___] wird herr schuster… |
| 49 | Einfügung | `*(nicht da)*` | `wird` | (FMT) …kein name genannt [___] herr schuster kommen… |

---

## Anamnesegespräch

**Fehlerrate: 97.1%** — RAW: 2269 Wörter | FMT: 152 Wörter | S=87 D=2117 I=0 | Fehler=2204

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
| 60 | Löschung | `ja` | `*(nicht da)*` | …langsam buchstabieren gerne [___] becken westfalen west… |
| 61 | Löschung | `becken` | `*(nicht da)*` | …buchstabieren gerne ja [___] westfalen west palen… |
| 62 | Löschung | `westfalen` | `*(nicht da)*` | …gerne ja becken [___] west palen alles… |
| 63 | Löschung | `west` | `*(nicht da)*` | …ja becken westfalen [___] palen alles klar… |
| 64 | Löschung | `palen` | `*(nicht da)*` | …becken westfalen west [___] alles klar dankeschön… |
| 65 | Löschung | `alles` | `*(nicht da)*` | …westfalen west palen [___] klar dankeschön frau… |
| 66 | Löschung | `klar` | `*(nicht da)*` | …west palen alles [___] dankeschön frau becken… |
| 67 | Löschung | `dankeschön` | `*(nicht da)*` | …palen alles klar [___] frau becken westfalen… |
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
| 128 | Löschung | `das` | `*(nicht da)*` | …hausarztes verraten ja [___] ist der herr… |
| 129 | Löschung | `ist` | `*(nicht da)*` | …verraten ja das [___] der herr dr… |
| 130 | Löschung | `der` | `*(nicht da)*` | …ja das ist [___] herr dr becker… |
| 131 | Löschung | `herr` | `*(nicht da)*` | …das ist der [___] dr becker der… |
| 132 | Löschung | `dr` | `*(nicht da)*` | …ist der herr [___] becker der herr… |
| 133 | Löschung | `becker` | `*(nicht da)*` | …der herr dr [___] der herr dr… |
| 134 | Löschung | `der` | `*(nicht da)*` | …herr dr becker [___] herr dr becker… |
| 135 | Löschung | `herr` | `*(nicht da)*` | …dr becker der [___] dr becker wie… |
| 136 | Löschung | `dr` | `*(nicht da)*` | …becker der herr [___] becker wie der… |
| 137 | Löschung | `becker` | `*(nicht da)*` | …der herr dr [___] wie der beruf… |
| 138 | Löschung | `wie` | `*(nicht da)*` | …herr dr becker [___] der beruf oder… |
| 139 | Löschung | `der` | `*(nicht da)*` | …dr becker wie [___] beruf oder mit… |
| 140 | Löschung | `beruf` | `*(nicht da)*` | …becker wie der [___] oder mit e… |
| 141 | Löschung | `oder` | `*(nicht da)*` | …wie der beruf [___] mit e mit… |
| 142 | Löschung | `mit` | `*(nicht da)*` | …der beruf oder [___] e mit e… |
| 143 | Löschung | `e` | `*(nicht da)*` | …beruf oder mit [___] mit e mit… |
| 144 | Löschung | `mit` | `*(nicht da)*` | …oder mit e [___] e mit e… |
| 145 | Löschung | `e` | `*(nicht da)*` | …mit e mit [___] mit e alles… |
| 146 | Löschung | `mit` | `*(nicht da)*` | …e mit e [___] e alles klar… |
| 147 | Löschung | `e` | `*(nicht da)*` | …mit e mit [___] alles klar gut… |
| 148 | Löschung | `alles` | `*(nicht da)*` | …e mit e [___] klar gut frau… |
| 149 | Löschung | `klar` | `*(nicht da)*` | …mit e alles [___] gut frau becken… |
| 150 | Substitution | `gut` | `die` | …e alles klar [___] frau becken westfalen… |
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
| 164 | Löschung | `denn` | `*(nicht da)*` | …gebracht was ist [___] passiert ja ich… |
| 165 | Löschung | `passiert` | `*(nicht da)*` | …was ist denn [___] ja ich bin… |
| 166 | Löschung | `ja` | `*(nicht da)*` | …ist denn passiert [___] ich bin unvorsichtig… |
| 167 | Löschung | `ich` | `*(nicht da)*` | …denn passiert ja [___] bin unvorsichtig mit… |
| 168 | Löschung | `bin` | `*(nicht da)*` | …passiert ja ich [___] unvorsichtig mit meinem… |
| 169 | Löschung | `unvorsichtig` | `*(nicht da)*` | …ja ich bin [___] mit meinem fahrrad… |
| 170 | Löschung | `mit` | `*(nicht da)*` | …ich bin unvorsichtig [___] meinem fahrrad nach… |
| 171 | Löschung | `meinem` | `*(nicht da)*` | …bin unvorsichtig mit [___] fahrrad nach hause… |
| 172 | Löschung | `fahrrad` | `*(nicht da)*` | …unvorsichtig mit meinem [___] nach hause gefahren… |
| 173 | Löschung | `hause` | `*(nicht da)*` | …meinem fahrrad nach [___] gefahren von der… |
| 174 | Löschung | `gefahren` | `*(nicht da)*` | …fahrrad nach hause [___] von der arbeit… |
| 175 | Löschung | `von` | `*(nicht da)*` | …nach hause gefahren [___] der arbeit und… |
| 176 | Löschung | `der` | `*(nicht da)*` | …hause gefahren von [___] arbeit und hatte… |
| 177 | Löschung | `arbeit` | `*(nicht da)*` | …gefahren von der [___] und hatte leider… |
| 178 | Löschung | `und` | `*(nicht da)*` | …von der arbeit [___] hatte leider einen… |
| 179 | Löschung | `hatte` | `*(nicht da)*` | …der arbeit und [___] leider einen unfall… |
| 180 | Löschung | `leider` | `*(nicht da)*` | …arbeit und hatte [___] einen unfall okay… |
| 181 | Löschung | `einen` | `*(nicht da)*` | …und hatte leider [___] unfall okay dabei… |
| 182 | Löschung | `unfall` | `*(nicht da)*` | …hatte leider einen [___] okay dabei habe… |
| 183 | Löschung | `okay` | `*(nicht da)*` | …leider einen unfall [___] dabei habe ich… |
| 184 | Löschung | `dabei` | `*(nicht da)*` | …einen unfall okay [___] habe ich mich… |
| 185 | Löschung | `habe` | `*(nicht da)*` | …unfall okay dabei [___] ich mich verletzt… |
| 186 | Löschung | `ich` | `*(nicht da)*` | …okay dabei habe [___] mich verletzt den… |
| 187 | Löschung | `mich` | `*(nicht da)*` | …dabei habe ich [___] verletzt den krankenwagen… |
| 188 | Löschung | `verletzt` | `*(nicht da)*` | …habe ich mich [___] den krankenwagen gerufen… |
| 189 | Löschung | `den` | `*(nicht da)*` | …ich mich verletzt [___] krankenwagen gerufen und… |
| 190 | Löschung | `krankenwagen` | `*(nicht da)*` | …mich verletzt den [___] gerufen und da… |
| 191 | Löschung | `gerufen` | `*(nicht da)*` | …verletzt den krankenwagen [___] und da bin… |
| 192 | Löschung | `und` | `*(nicht da)*` | …den krankenwagen gerufen [___] da bin ich… |
| 193 | Löschung | `da` | `*(nicht da)*` | …krankenwagen gerufen und [___] bin ich jetzt… |
| 194 | Löschung | `bin` | `*(nicht da)*` | …gerufen und da [___] ich jetzt da… |
| 195 | Löschung | `ich` | `*(nicht da)*` | …und da bin [___] jetzt da sind… |
| 196 | Löschung | `jetzt` | `*(nicht da)*` | …da bin ich [___] da sind sie… |
| 197 | Löschung | `da` | `*(nicht da)*` | …bin ich jetzt [___] sind sie jetzt… |
| 198 | Löschung | `sind` | `*(nicht da)*` | …ich jetzt da [___] sie jetzt was… |
| 199 | Löschung | `sie` | `*(nicht da)*` | …jetzt da sind [___] jetzt was haben… |
| 200 | Löschung | `jetzt` | `*(nicht da)*` | …da sind sie [___] was haben sie… |
| 201 | Löschung | `was` | `*(nicht da)*` | …sind sie jetzt [___] haben sie denn… |
| 202 | Löschung | `haben` | `*(nicht da)*` | …sie jetzt was [___] sie denn jetzt… |
| 203 | Löschung | `sie` | `*(nicht da)*` | …jetzt was haben [___] denn jetzt für… |
| 204 | Löschung | `denn` | `*(nicht da)*` | …was haben sie [___] jetzt für beschwerden… |
| 205 | Löschung | `jetzt` | `*(nicht da)*` | …haben sie denn [___] für beschwerden beschwerden… |
| 206 | Löschung | `für` | `*(nicht da)*` | …sie denn jetzt [___] beschwerden beschwerden entschuldigung… |
| 207 | Löschung | `beschwerden` | `*(nicht da)*` | …denn jetzt für [___] beschwerden entschuldigung haben… |
| 208 | Löschung | `beschwerden` | `*(nicht da)*` | …jetzt für beschwerden [___] entschuldigung haben sie… |
| 209 | Substitution | `entschuldigung` | `einem` | …für beschwerden beschwerden [___] haben sie schmerzen… |
| 210 | Substitution | `haben` | `fahrradunfall` | …beschwerden beschwerden entschuldigung [___] sie schmerzen am… |
| 211 | Substitution | `sie` | `mit` | …beschwerden entschuldigung haben [___] schmerzen am kopf… |
| 212 | Löschung | `am` | `*(nicht da)*` | …haben sie schmerzen [___] kopf im oberkörper… |
| 213 | Löschung | `kopf` | `*(nicht da)*` | …sie schmerzen am [___] im oberkörper an… |
| 214 | Löschung | `oberkörper` | `*(nicht da)*` | …am kopf im [___] an den beinen… |
| 215 | Löschung | `an` | `*(nicht da)*` | …kopf im oberkörper [___] den beinen ja… |
| 216 | Löschung | `den` | `*(nicht da)*` | …im oberkörper an [___] beinen ja ich… |
| 217 | Löschung | `beinen` | `*(nicht da)*` | …oberkörper an den [___] ja ich bin… |
| 218 | Löschung | `ja` | `*(nicht da)*` | …an den beinen [___] ich bin auf… |
| 219 | Löschung | `ich` | `*(nicht da)*` | …den beinen ja [___] bin auf meine… |
| 220 | Löschung | `bin` | `*(nicht da)*` | …beinen ja ich [___] auf meine linke… |
| 221 | Löschung | `auf` | `*(nicht da)*` | …ja ich bin [___] meine linke seite… |
| 222 | Löschung | `meine` | `*(nicht da)*` | …ich bin auf [___] linke seite gefallen… |
| 223 | Löschung | `linke` | `*(nicht da)*` | …bin auf meine [___] seite gefallen und… |
| 224 | Löschung | `seite` | `*(nicht da)*` | …auf meine linke [___] gefallen und habe… |
| 225 | Löschung | `gefallen` | `*(nicht da)*` | …meine linke seite [___] und habe mir… |
| 226 | Löschung | `und` | `*(nicht da)*` | …linke seite gefallen [___] habe mir dabei… |
| 227 | Löschung | `habe` | `*(nicht da)*` | …seite gefallen und [___] mir dabei auch… |
| 228 | Löschung | `mir` | `*(nicht da)*` | …gefallen und habe [___] dabei auch tatsächlich… |
| 229 | Löschung | `dabei` | `*(nicht da)*` | …und habe mir [___] auch tatsächlich den… |
| 230 | Löschung | `auch` | `*(nicht da)*` | …habe mir dabei [___] tatsächlich den kopf… |
| 231 | Löschung | `tatsächlich` | `*(nicht da)*` | …mir dabei auch [___] den kopf leicht… |
| 232 | Löschung | `den` | `*(nicht da)*` | …dabei auch tatsächlich [___] kopf leicht gestoßen… |
| 233 | Löschung | `leicht` | `*(nicht da)*` | …tatsächlich den kopf [___] gestoßen ich habe… |
| 234 | Löschung | `gestoßen` | `*(nicht da)*` | …den kopf leicht [___] ich habe leichte… |
| 235 | Löschung | `ich` | `*(nicht da)*` | …kopf leicht gestoßen [___] habe leichte schmerzen… |
| 236 | Löschung | `habe` | `*(nicht da)*` | …leicht gestoßen ich [___] leichte schmerzen am… |
| 237 | Löschung | `leichte` | `*(nicht da)*` | …gestoßen ich habe [___] schmerzen am hinterkopf… |
| 238 | Löschung | `schmerzen` | `*(nicht da)*` | …ich habe leichte [___] am hinterkopf auf… |
| 239 | Löschung | `am` | `*(nicht da)*` | …habe leichte schmerzen [___] hinterkopf auf der… |
| 240 | Löschung | `hinterkopf` | `*(nicht da)*` | …leichte schmerzen am [___] auf der linken… |
| 241 | Löschung | `auf` | `*(nicht da)*` | …schmerzen am hinterkopf [___] der linken seite… |
| 242 | Löschung | `der` | `*(nicht da)*` | …am hinterkopf auf [___] linken seite ich… |
| 243 | Löschung | `linken` | `*(nicht da)*` | …hinterkopf auf der [___] seite ich kann… |
| 244 | Löschung | `seite` | `*(nicht da)*` | …auf der linken [___] ich kann außerdem… |
| 245 | Löschung | `ich` | `*(nicht da)*` | …der linken seite [___] kann außerdem meinen… |
| 246 | Löschung | `kann` | `*(nicht da)*` | …linken seite ich [___] außerdem meinen linken… |
| 247 | Löschung | `außerdem` | `*(nicht da)*` | …seite ich kann [___] meinen linken daumen… |
| 248 | Löschung | `meinen` | `*(nicht da)*` | …ich kann außerdem [___] linken daumen überhaupt… |
| 249 | Löschung | `linken` | `*(nicht da)*` | …kann außerdem meinen [___] daumen überhaupt nicht… |
| 250 | Löschung | `daumen` | `*(nicht da)*` | …außerdem meinen linken [___] überhaupt nicht bewegen… |
| 251 | Löschung | `überhaupt` | `*(nicht da)*` | …meinen linken daumen [___] nicht bewegen weil… |
| 252 | Löschung | `nicht` | `*(nicht da)*` | …linken daumen überhaupt [___] bewegen weil ich… |
| 253 | Löschung | `bewegen` | `*(nicht da)*` | …daumen überhaupt nicht [___] weil ich wirklich… |
| 254 | Löschung | `weil` | `*(nicht da)*` | …überhaupt nicht bewegen [___] ich wirklich starke… |
| 255 | Löschung | `ich` | `*(nicht da)*` | …nicht bewegen weil [___] wirklich starke schmerzen… |
| 256 | Löschung | `wirklich` | `*(nicht da)*` | …bewegen weil ich [___] starke schmerzen habe… |
| 257 | Löschung | `starke` | `*(nicht da)*` | …weil ich wirklich [___] schmerzen habe am… |
| 258 | Löschung | `schmerzen` | `*(nicht da)*` | …ich wirklich starke [___] habe am daumen… |
| 259 | Löschung | `habe` | `*(nicht da)*` | …wirklich starke schmerzen [___] am daumen und… |
| 260 | Löschung | `am` | `*(nicht da)*` | …starke schmerzen habe [___] daumen und er… |
| 261 | Löschung | `und` | `*(nicht da)*` | …habe am daumen [___] er ist auch… |
| 262 | Löschung | `er` | `*(nicht da)*` | …am daumen und [___] ist auch etwas… |
| 263 | Löschung | `ist` | `*(nicht da)*` | …daumen und er [___] auch etwas geschwollen… |
| 264 | Löschung | `auch` | `*(nicht da)*` | …und er ist [___] etwas geschwollen und… |
| 265 | Löschung | `etwas` | `*(nicht da)*` | …er ist auch [___] geschwollen und irgendwie… |
| 266 | Löschung | `geschwollen` | `*(nicht da)*` | …ist auch etwas [___] und irgendwie habe… |
| 267 | Löschung | `und` | `*(nicht da)*` | …auch etwas geschwollen [___] irgendwie habe ich… |
| 268 | Löschung | `irgendwie` | `*(nicht da)*` | …etwas geschwollen und [___] habe ich auch… |
| 269 | Löschung | `habe` | `*(nicht da)*` | …geschwollen und irgendwie [___] ich auch mein… |
| 270 | Löschung | `ich` | `*(nicht da)*` | …und irgendwie habe [___] auch mein knie… |
| 271 | Löschung | `auch` | `*(nicht da)*` | …irgendwie habe ich [___] mein knie richtig… |
| 272 | Löschung | `mein` | `*(nicht da)*` | …habe ich auch [___] knie richtig stark… |
| 273 | Löschung | `knie` | `*(nicht da)*` | …ich auch mein [___] richtig stark verletzt… |
| 274 | Löschung | `richtig` | `*(nicht da)*` | …auch mein knie [___] stark verletzt weil… |
| 275 | Löschung | `stark` | `*(nicht da)*` | …mein knie richtig [___] verletzt weil es… |
| 276 | Löschung | `verletzt` | `*(nicht da)*` | …knie richtig stark [___] weil es richtig… |
| 277 | Löschung | `weil` | `*(nicht da)*` | …richtig stark verletzt [___] es richtig geschwollen… |
| 278 | Löschung | `es` | `*(nicht da)*` | …stark verletzt weil [___] richtig geschwollen ist… |
| 279 | Löschung | `richtig` | `*(nicht da)*` | …verletzt weil es [___] geschwollen ist und… |
| 280 | Löschung | `geschwollen` | `*(nicht da)*` | …weil es richtig [___] ist und auch… |
| 281 | Löschung | `ist` | `*(nicht da)*` | …es richtig geschwollen [___] und auch sehr… |
| 282 | Löschung | `auch` | `*(nicht da)*` | …geschwollen ist und [___] sehr weh tut… |
| 283 | Löschung | `sehr` | `*(nicht da)*` | …ist und auch [___] weh tut okay… |
| 284 | Löschung | `weh` | `*(nicht da)*` | …und auch sehr [___] tut okay knie… |
| 285 | Löschung | `tut` | `*(nicht da)*` | …auch sehr weh [___] okay knie ist… |
| 286 | Löschung | `okay` | `*(nicht da)*` | …sehr weh tut [___] knie ist auch… |
| 287 | Löschung | `ist` | `*(nicht da)*` | …tut okay knie [___] auch geschwollen und… |
| 288 | Löschung | `auch` | `*(nicht da)*` | …okay knie ist [___] geschwollen und starke… |
| 289 | Löschung | `geschwollen` | `*(nicht da)*` | …knie ist auch [___] und starke schmerzen… |
| 290 | Löschung | `und` | `*(nicht da)*` | …ist auch geschwollen [___] starke schmerzen sagen… |
| 291 | Löschung | `starke` | `*(nicht da)*` | …auch geschwollen und [___] schmerzen sagen sie… |
| 292 | Löschung | `schmerzen` | `*(nicht da)*` | …geschwollen und starke [___] sagen sie genau… |
| 293 | Löschung | `sagen` | `*(nicht da)*` | …und starke schmerzen [___] sie genau frau… |
| 294 | Löschung | `sie` | `*(nicht da)*` | …starke schmerzen sagen [___] genau frau becken… |
| 295 | Löschung | `genau` | `*(nicht da)*` | …schmerzen sagen sie [___] frau becken westfalen… |
| 296 | Löschung | `frau` | `*(nicht da)*` | …sagen sie genau [___] becken westfalen haben… |
| 297 | Löschung | `becken` | `*(nicht da)*` | …sie genau frau [___] westfalen haben sie… |
| 298 | Löschung | `westfalen` | `*(nicht da)*` | …genau frau becken [___] haben sie denn… |
| 299 | Löschung | `haben` | `*(nicht da)*` | …frau becken westfalen [___] sie denn einen… |
| 300 | Löschung | `sie` | `*(nicht da)*` | …becken westfalen haben [___] denn einen fahrradhelm… |
| 301 | Löschung | `denn` | `*(nicht da)*` | …westfalen haben sie [___] einen fahrradhelm getragen… |
| 302 | Löschung | `einen` | `*(nicht da)*` | …haben sie denn [___] fahrradhelm getragen leider… |
| 303 | Löschung | `fahrradhelm` | `*(nicht da)*` | …sie denn einen [___] getragen leider nein… |
| 304 | Löschung | `getragen` | `*(nicht da)*` | …denn einen fahrradhelm [___] leider nein ich… |
| 305 | Löschung | `leider` | `*(nicht da)*` | …einen fahrradhelm getragen [___] nein ich muss… |
| 306 | Löschung | `nein` | `*(nicht da)*` | …fahrradhelm getragen leider [___] ich muss auch… |
| 307 | Löschung | `ich` | `*(nicht da)*` | …getragen leider nein [___] muss auch zugeben… |
| 308 | Löschung | `muss` | `*(nicht da)*` | …leider nein ich [___] auch zugeben dass… |
| 309 | Löschung | `auch` | `*(nicht da)*` | …nein ich muss [___] zugeben dass ich… |
| 310 | Löschung | `zugeben` | `*(nicht da)*` | …ich muss auch [___] dass ich sehr… |
| 311 | Löschung | `dass` | `*(nicht da)*` | …muss auch zugeben [___] ich sehr ungern… |
| 312 | Löschung | `ich` | `*(nicht da)*` | …auch zugeben dass [___] sehr ungern einen… |
| 313 | Löschung | `sehr` | `*(nicht da)*` | …zugeben dass ich [___] ungern einen fahrradhelm… |
| 314 | Löschung | `ungern` | `*(nicht da)*` | …dass ich sehr [___] einen fahrradhelm trage… |
| 315 | Löschung | `einen` | `*(nicht da)*` | …ich sehr ungern [___] fahrradhelm trage weil… |
| 316 | Löschung | `fahrradhelm` | `*(nicht da)*` | …sehr ungern einen [___] trage weil sie… |
| 317 | Löschung | `trage` | `*(nicht da)*` | …ungern einen fahrradhelm [___] weil sie mir… |
| 318 | Löschung | `weil` | `*(nicht da)*` | …einen fahrradhelm trage [___] sie mir so… |
| 319 | Löschung | `sie` | `*(nicht da)*` | …fahrradhelm trage weil [___] mir so unbequem… |
| 320 | Löschung | `mir` | `*(nicht da)*` | …trage weil sie [___] so unbequem sind… |
| 321 | Löschung | `so` | `*(nicht da)*` | …weil sie mir [___] unbequem sind und… |
| 322 | Löschung | `unbequem` | `*(nicht da)*` | …sie mir so [___] sind und es… |
| 323 | Löschung | `sind` | `*(nicht da)*` | …mir so unbequem [___] und es sieht… |
| 324 | Löschung | `und` | `*(nicht da)*` | …so unbequem sind [___] es sieht auch… |
| 325 | Löschung | `es` | `*(nicht da)*` | …unbequem sind und [___] sieht auch so… |
| 326 | Löschung | `sieht` | `*(nicht da)*` | …sind und es [___] auch so bescheuert… |
| 327 | Löschung | `auch` | `*(nicht da)*` | …und es sieht [___] so bescheuert aus… |
| 328 | Löschung | `so` | `*(nicht da)*` | …es sieht auch [___] bescheuert aus sie… |
| 329 | Löschung | `bescheuert` | `*(nicht da)*` | …sieht auch so [___] aus sie als… |
| 330 | Löschung | `aus` | `*(nicht da)*` | …auch so bescheuert [___] sie als frau… |
| 331 | Löschung | `sie` | `*(nicht da)*` | …so bescheuert aus [___] als frau würden… |
| 332 | Löschung | `als` | `*(nicht da)*` | …bescheuert aus sie [___] frau würden mich… |
| 333 | Löschung | `frau` | `*(nicht da)*` | …aus sie als [___] würden mich sicherlich… |
| 334 | Löschung | `würden` | `*(nicht da)*` | …sie als frau [___] mich sicherlich verstehen… |
| 335 | Löschung | `mich` | `*(nicht da)*` | …als frau würden [___] sicherlich verstehen ich… |
| 336 | Löschung | `sicherlich` | `*(nicht da)*` | …frau würden mich [___] verstehen ich verstehe… |
| 337 | Löschung | `verstehen` | `*(nicht da)*` | …würden mich sicherlich [___] ich verstehe sie… |
| 338 | Löschung | `ich` | `*(nicht da)*` | …mich sicherlich verstehen [___] verstehe sie voll… |
| 339 | Löschung | `verstehe` | `*(nicht da)*` | …sicherlich verstehen ich [___] sie voll und… |
| 340 | Löschung | `sie` | `*(nicht da)*` | …verstehen ich verstehe [___] voll und ganz… |
| 341 | Löschung | `voll` | `*(nicht da)*` | …ich verstehe sie [___] und ganz meiner… |
| 342 | Löschung | `und` | `*(nicht da)*` | …verstehe sie voll [___] ganz meiner frisur… |
| 343 | Löschung | `ganz` | `*(nicht da)*` | …sie voll und [___] meiner frisur tut… |
| 344 | Löschung | `meiner` | `*(nicht da)*` | …voll und ganz [___] frisur tut das… |
| 345 | Löschung | `frisur` | `*(nicht da)*` | …und ganz meiner [___] tut das auch… |
| 346 | Löschung | `tut` | `*(nicht da)*` | …ganz meiner frisur [___] das auch nicht… |
| 347 | Löschung | `das` | `*(nicht da)*` | …meiner frisur tut [___] auch nicht gut… |
| 348 | Löschung | `auch` | `*(nicht da)*` | …frisur tut das [___] nicht gut aber… |
| 349 | Löschung | `nicht` | `*(nicht da)*` | …tut das auch [___] gut aber da… |
| 350 | Löschung | `gut` | `*(nicht da)*` | …das auch nicht [___] aber da muss… |
| 351 | Löschung | `aber` | `*(nicht da)*` | …auch nicht gut [___] da muss ich… |
| 352 | Löschung | `da` | `*(nicht da)*` | …nicht gut aber [___] muss ich ihnen… |
| 353 | Löschung | `muss` | `*(nicht da)*` | …gut aber da [___] ich ihnen leider… |
| 354 | Löschung | `ich` | `*(nicht da)*` | …aber da muss [___] ihnen leider sagen… |
| 355 | Löschung | `ihnen` | `*(nicht da)*` | …da muss ich [___] leider sagen in… |
| 356 | Löschung | `leider` | `*(nicht da)*` | …muss ich ihnen [___] sagen in diesem… |
| 357 | Löschung | `sagen` | `*(nicht da)*` | …ich ihnen leider [___] in diesem fall… |
| 358 | Löschung | `diesem` | `*(nicht da)*` | …leider sagen in [___] fall gehen sicherheit… |
| 359 | Löschung | `fall` | `*(nicht da)*` | …sagen in diesem [___] gehen sicherheit und… |
| 360 | Löschung | `gehen` | `*(nicht da)*` | …in diesem fall [___] sicherheit und gesundheit… |
| 361 | Löschung | `sicherheit` | `*(nicht da)*` | …diesem fall gehen [___] und gesundheit definitiv… |
| 362 | Löschung | `und` | `*(nicht da)*` | …fall gehen sicherheit [___] gesundheit definitiv vor… |
| 363 | Löschung | `gesundheit` | `*(nicht da)*` | …gehen sicherheit und [___] definitiv vor aussehen… |
| 364 | Löschung | `definitiv` | `*(nicht da)*` | …sicherheit und gesundheit [___] vor aussehen frau… |
| 365 | Löschung | `vor` | `*(nicht da)*` | …und gesundheit definitiv [___] aussehen frau becken… |
| 366 | Löschung | `aussehen` | `*(nicht da)*` | …gesundheit definitiv vor [___] frau becken westfalen… |
| 367 | Löschung | `frau` | `*(nicht da)*` | …definitiv vor aussehen [___] becken westfalen bitte… |
| 368 | Löschung | `becken` | `*(nicht da)*` | …vor aussehen frau [___] westfalen bitte bitte… |
| 369 | Löschung | `westfalen` | `*(nicht da)*` | …aussehen frau becken [___] bitte bitte tragen… |
| 370 | Löschung | `bitte` | `*(nicht da)*` | …frau becken westfalen [___] bitte tragen sie… |
| 371 | Löschung | `bitte` | `*(nicht da)*` | …becken westfalen bitte [___] tragen sie beim… |
| 372 | Löschung | `tragen` | `*(nicht da)*` | …westfalen bitte bitte [___] sie beim nächsten… |
| 373 | Löschung | `sie` | `*(nicht da)*` | …bitte bitte tragen [___] beim nächsten mal… |
| 374 | Löschung | `beim` | `*(nicht da)*` | …bitte tragen sie [___] nächsten mal einen… |
| 375 | Löschung | `nächsten` | `*(nicht da)*` | …tragen sie beim [___] mal einen helm… |
| 376 | Löschung | `mal` | `*(nicht da)*` | …sie beim nächsten [___] einen helm da… |
| 377 | Löschung | `einen` | `*(nicht da)*` | …beim nächsten mal [___] helm da haben… |
| 378 | Löschung | `helm` | `*(nicht da)*` | …nächsten mal einen [___] da haben sie… |
| 379 | Löschung | `da` | `*(nicht da)*` | …mal einen helm [___] haben sie diesmal… |
| 380 | Löschung | `haben` | `*(nicht da)*` | …einen helm da [___] sie diesmal wirklich… |
| 381 | Löschung | `sie` | `*(nicht da)*` | …helm da haben [___] diesmal wirklich noch… |
| 382 | Löschung | `diesmal` | `*(nicht da)*` | …da haben sie [___] wirklich noch glück… |
| 383 | Löschung | `wirklich` | `*(nicht da)*` | …haben sie diesmal [___] noch glück gehabt… |
| 384 | Löschung | `noch` | `*(nicht da)*` | …sie diesmal wirklich [___] glück gehabt dass… |
| 385 | Löschung | `glück` | `*(nicht da)*` | …diesmal wirklich noch [___] gehabt dass nichts… |
| 386 | Löschung | `gehabt` | `*(nicht da)*` | …wirklich noch glück [___] dass nichts passiert… |
| 387 | Löschung | `dass` | `*(nicht da)*` | …noch glück gehabt [___] nichts passiert ist… |
| 388 | Löschung | `nichts` | `*(nicht da)*` | …glück gehabt dass [___] passiert ist da… |
| 389 | Löschung | `passiert` | `*(nicht da)*` | …gehabt dass nichts [___] ist da haben… |
| 390 | Löschung | `ist` | `*(nicht da)*` | …dass nichts passiert [___] da haben sie… |
| 391 | Löschung | `da` | `*(nicht da)*` | …nichts passiert ist [___] haben sie auf… |
| 392 | Löschung | `haben` | `*(nicht da)*` | …passiert ist da [___] sie auf jeden… |
| 393 | Löschung | `sie` | `*(nicht da)*` | …ist da haben [___] auf jeden fall… |
| 394 | Löschung | `auf` | `*(nicht da)*` | …da haben sie [___] jeden fall recht… |
| 395 | Löschung | `jeden` | `*(nicht da)*` | …haben sie auf [___] fall recht ich… |
| 396 | Löschung | `fall` | `*(nicht da)*` | …sie auf jeden [___] recht ich habe… |
| 397 | Löschung | `recht` | `*(nicht da)*` | …auf jeden fall [___] ich habe jetzt… |
| 398 | Löschung | `ich` | `*(nicht da)*` | …jeden fall recht [___] habe jetzt daraus… |
| 399 | Löschung | `habe` | `*(nicht da)*` | …fall recht ich [___] jetzt daraus gelernt… |
| 400 | Löschung | `jetzt` | `*(nicht da)*` | …recht ich habe [___] daraus gelernt und… |
| 401 | Löschung | `daraus` | `*(nicht da)*` | …ich habe jetzt [___] gelernt und werde… |
| 402 | Löschung | `gelernt` | `*(nicht da)*` | …habe jetzt daraus [___] und werde mir… |
| 403 | Löschung | `und` | `*(nicht da)*` | …jetzt daraus gelernt [___] werde mir auch… |
| 404 | Löschung | `werde` | `*(nicht da)*` | …daraus gelernt und [___] mir auch einen… |
| 405 | Löschung | `mir` | `*(nicht da)*` | …gelernt und werde [___] auch einen besorgen… |
| 406 | Löschung | `auch` | `*(nicht da)*` | …und werde mir [___] einen besorgen okay… |
| 407 | Löschung | `einen` | `*(nicht da)*` | …werde mir auch [___] besorgen okay sehr… |
| 408 | Löschung | `besorgen` | `*(nicht da)*` | …mir auch einen [___] okay sehr gut… |
| 409 | Löschung | `okay` | `*(nicht da)*` | …auch einen besorgen [___] sehr gut sie… |
| 410 | Löschung | `sehr` | `*(nicht da)*` | …einen besorgen okay [___] gut sie hatten… |
| 411 | Löschung | `gut` | `*(nicht da)*` | …besorgen okay sehr [___] sie hatten gesagt… |
| 412 | Löschung | `sie` | `*(nicht da)*` | …okay sehr gut [___] hatten gesagt sie… |
| 413 | Löschung | `hatten` | `*(nicht da)*` | …sehr gut sie [___] gesagt sie haben… |
| 414 | Löschung | `gesagt` | `*(nicht da)*` | …gut sie hatten [___] sie haben hinten… |
| 415 | Löschung | `sie` | `*(nicht da)*` | …sie hatten gesagt [___] haben hinten auf… |
| 416 | Löschung | `haben` | `*(nicht da)*` | …hatten gesagt sie [___] hinten auf der… |
| 417 | Löschung | `hinten` | `*(nicht da)*` | …gesagt sie haben [___] auf der linken… |
| 418 | Löschung | `auf` | `*(nicht da)*` | …sie haben hinten [___] der linken seite… |
| 419 | Löschung | `der` | `*(nicht da)*` | …haben hinten auf [___] linken seite des… |
| 420 | Löschung | `linken` | `*(nicht da)*` | …hinten auf der [___] seite des hinterkopfes… |
| 421 | Löschung | `seite` | `*(nicht da)*` | …auf der linken [___] des hinterkopfes eine… |
| 422 | Löschung | `des` | `*(nicht da)*` | …der linken seite [___] hinterkopfes eine kleine… |
| 423 | Löschung | `hinterkopfes` | `*(nicht da)*` | …linken seite des [___] eine kleine beule… |
| 424 | Löschung | `eine` | `*(nicht da)*` | …seite des hinterkopfes [___] kleine beule richtig… |
| 425 | Löschung | `kleine` | `*(nicht da)*` | …des hinterkopfes eine [___] beule richtig genau… |
| 426 | Löschung | `beule` | `*(nicht da)*` | …hinterkopfes eine kleine [___] richtig genau das… |
| 427 | Löschung | `richtig` | `*(nicht da)*` | …eine kleine beule [___] genau das ist… |
| 428 | Löschung | `genau` | `*(nicht da)*` | …kleine beule richtig [___] das ist richtig… |
| 429 | Löschung | `das` | `*(nicht da)*` | …beule richtig genau [___] ist richtig ja… |
| 430 | Löschung | `ist` | `*(nicht da)*` | …richtig genau das [___] richtig ja haben… |
| 431 | Löschung | `richtig` | `*(nicht da)*` | …genau das ist [___] ja haben sie… |
| 432 | Löschung | `ja` | `*(nicht da)*` | …das ist richtig [___] haben sie irgendeine… |
| 433 | Löschung | `haben` | `*(nicht da)*` | …ist richtig ja [___] sie irgendeine blutige… |
| 434 | Löschung | `sie` | `*(nicht da)*` | …richtig ja haben [___] irgendeine blutige verletzung… |
| 435 | Löschung | `irgendeine` | `*(nicht da)*` | …ja haben sie [___] blutige verletzung am… |
| 436 | Löschung | `blutige` | `*(nicht da)*` | …haben sie irgendeine [___] verletzung am kopf… |
| 437 | Löschung | `verletzung` | `*(nicht da)*` | …sie irgendeine blutige [___] am kopf oder… |
| 438 | Löschung | `am` | `*(nicht da)*` | …irgendeine blutige verletzung [___] kopf oder ist… |
| 439 | Löschung | `kopf` | `*(nicht da)*` | …blutige verletzung am [___] oder ist das… |
| 440 | Löschung | `oder` | `*(nicht da)*` | …verletzung am kopf [___] ist das alles… |
| 441 | Löschung | `ist` | `*(nicht da)*` | …am kopf oder [___] das alles es… |
| 442 | Löschung | `das` | `*(nicht da)*` | …kopf oder ist [___] alles es ist… |
| 443 | Löschung | `alles` | `*(nicht da)*` | …oder ist das [___] es ist mir… |
| 444 | Löschung | `es` | `*(nicht da)*` | …ist das alles [___] ist mir nichts… |
| 445 | Löschung | `ist` | `*(nicht da)*` | …das alles es [___] mir nichts anderes… |
| 446 | Löschung | `mir` | `*(nicht da)*` | …alles es ist [___] nichts anderes aufgefallen… |
| 447 | Löschung | `nichts` | `*(nicht da)*` | …es ist mir [___] anderes aufgefallen zum… |
| 448 | Löschung | `anderes` | `*(nicht da)*` | …ist mir nichts [___] aufgefallen zum glück… |
| 449 | Löschung | `aufgefallen` | `*(nicht da)*` | …mir nichts anderes [___] zum glück ist… |
| 450 | Löschung | `zum` | `*(nicht da)*` | …nichts anderes aufgefallen [___] glück ist es… |
| 451 | Löschung | `glück` | `*(nicht da)*` | …anderes aufgefallen zum [___] ist es glaube… |
| 452 | Löschung | `ist` | `*(nicht da)*` | …aufgefallen zum glück [___] es glaube ich… |
| 453 | Löschung | `es` | `*(nicht da)*` | …zum glück ist [___] glaube ich nur… |
| 454 | Löschung | `glaube` | `*(nicht da)*` | …glück ist es [___] ich nur die… |
| 455 | Löschung | `ich` | `*(nicht da)*` | …ist es glaube [___] nur die beule… |
| 456 | Löschung | `nur` | `*(nicht da)*` | …es glaube ich [___] die beule okay… |
| 457 | Löschung | `die` | `*(nicht da)*` | …glaube ich nur [___] beule okay sehr… |
| 458 | Löschung | `beule` | `*(nicht da)*` | …ich nur die [___] okay sehr gut… |
| 459 | Löschung | `okay` | `*(nicht da)*` | …nur die beule [___] sehr gut die… |
| 460 | Löschung | `sehr` | `*(nicht da)*` | …die beule okay [___] gut die schmerzen… |
| 461 | Löschung | `gut` | `*(nicht da)*` | …beule okay sehr [___] die schmerzen sind… |
| 462 | Löschung | `die` | `*(nicht da)*` | …okay sehr gut [___] schmerzen sind die… |
| 463 | Löschung | `schmerzen` | `*(nicht da)*` | …sehr gut die [___] sind die stark… |
| 464 | Löschung | `sind` | `*(nicht da)*` | …gut die schmerzen [___] die stark oder… |
| 465 | Löschung | `die` | `*(nicht da)*` | …die schmerzen sind [___] stark oder geht… |
| 466 | Löschung | `stark` | `*(nicht da)*` | …schmerzen sind die [___] oder geht es… |
| 467 | Löschung | `oder` | `*(nicht da)*` | …sind die stark [___] geht es die… |
| 468 | Löschung | `geht` | `*(nicht da)*` | …die stark oder [___] es die sind… |
| 469 | Löschung | `es` | `*(nicht da)*` | …stark oder geht [___] die sind nicht… |
| 470 | Löschung | `die` | `*(nicht da)*` | …oder geht es [___] sind nicht so… |
| 471 | Löschung | `sind` | `*(nicht da)*` | …geht es die [___] nicht so stark… |
| 472 | Löschung | `nicht` | `*(nicht da)*` | …es die sind [___] so stark das… |
| 473 | Löschung | `so` | `*(nicht da)*` | …die sind nicht [___] stark das geht… |
| 474 | Löschung | `stark` | `*(nicht da)*` | …sind nicht so [___] das geht tatsächlich… |
| 475 | Löschung | `das` | `*(nicht da)*` | …nicht so stark [___] geht tatsächlich am… |
| 476 | Löschung | `geht` | `*(nicht da)*` | …so stark das [___] tatsächlich am kopf… |
| 477 | Löschung | `tatsächlich` | `*(nicht da)*` | …stark das geht [___] am kopf sind… |
| 478 | Löschung | `am` | `*(nicht da)*` | …das geht tatsächlich [___] kopf sind die… |
| 479 | Löschung | `kopf` | `*(nicht da)*` | …geht tatsächlich am [___] sind die am… |
| 480 | Löschung | `sind` | `*(nicht da)*` | …tatsächlich am kopf [___] die am schwächsten… |
| 481 | Löschung | `die` | `*(nicht da)*` | …am kopf sind [___] am schwächsten okay… |
| 482 | Löschung | `am` | `*(nicht da)*` | …kopf sind die [___] schwächsten okay alles… |
| 483 | Löschung | `schwächsten` | `*(nicht da)*` | …sind die am [___] okay alles klar… |
| 484 | Löschung | `okay` | `*(nicht da)*` | …die am schwächsten [___] alles klar der… |
| 485 | Löschung | `alles` | `*(nicht da)*` | …am schwächsten okay [___] klar der daumen… |
| 486 | Löschung | `klar` | `*(nicht da)*` | …schwächsten okay alles [___] der daumen sie… |
| 487 | Löschung | `der` | `*(nicht da)*` | …okay alles klar [___] daumen sie haben… |
| 488 | Löschung | `daumen` | `*(nicht da)*` | …alles klar der [___] sie haben jetzt… |
| 489 | Löschung | `sie` | `*(nicht da)*` | …klar der daumen [___] haben jetzt gesagt… |
| 490 | Löschung | `haben` | `*(nicht da)*` | …der daumen sie [___] jetzt gesagt sie… |
| 491 | Löschung | `jetzt` | `*(nicht da)*` | …daumen sie haben [___] gesagt sie können… |
| 492 | Löschung | `gesagt` | `*(nicht da)*` | …sie haben jetzt [___] sie können den… |
| 493 | Löschung | `sie` | `*(nicht da)*` | …haben jetzt gesagt [___] können den daumen… |
| 494 | Löschung | `können` | `*(nicht da)*` | …jetzt gesagt sie [___] den daumen gar… |
| 495 | Löschung | `den` | `*(nicht da)*` | …gesagt sie können [___] daumen gar nicht… |
| 496 | Löschung | `daumen` | `*(nicht da)*` | …sie können den [___] gar nicht mehr… |
| 497 | Löschung | `gar` | `*(nicht da)*` | …können den daumen [___] nicht mehr recht… |
| 498 | Löschung | `nicht` | `*(nicht da)*` | …den daumen gar [___] mehr recht bewegen… |
| 499 | Löschung | `mehr` | `*(nicht da)*` | …daumen gar nicht [___] recht bewegen wenn… |
| 500 | Löschung | `recht` | `*(nicht da)*` | …gar nicht mehr [___] bewegen wenn wir… |
| 501 | Löschung | `bewegen` | `*(nicht da)*` | …nicht mehr recht [___] wenn wir jetzt… |
| 502 | Löschung | `wenn` | `*(nicht da)*` | …mehr recht bewegen [___] wir jetzt die… |
| 503 | Löschung | `wir` | `*(nicht da)*` | …recht bewegen wenn [___] jetzt die schmerzen… |
| 504 | Löschung | `jetzt` | `*(nicht da)*` | …bewegen wenn wir [___] die schmerzen einschätzen… |
| 505 | Löschung | `die` | `*(nicht da)*` | …wenn wir jetzt [___] schmerzen einschätzen an… |
| 506 | Löschung | `schmerzen` | `*(nicht da)*` | …wir jetzt die [___] einschätzen an einer… |
| 507 | Löschung | `einschätzen` | `*(nicht da)*` | …jetzt die schmerzen [___] an einer schmerzskala… |
| 508 | Löschung | `an` | `*(nicht da)*` | …die schmerzen einschätzen [___] einer schmerzskala wobei… |
| 509 | Löschung | `einer` | `*(nicht da)*` | …schmerzen einschätzen an [___] schmerzskala wobei 1… |
| 510 | Löschung | `schmerzskala` | `*(nicht da)*` | …einschätzen an einer [___] wobei 1 sehr… |
| 511 | Löschung | `wobei` | `*(nicht da)*` | …an einer schmerzskala [___] 1 sehr leichten… |
| 512 | Löschung | `1` | `*(nicht da)*` | …einer schmerzskala wobei [___] sehr leichten schmerzen… |
| 513 | Löschung | `sehr` | `*(nicht da)*` | …schmerzskala wobei 1 [___] leichten schmerzen entspricht… |
| 514 | Löschung | `leichten` | `*(nicht da)*` | …wobei 1 sehr [___] schmerzen entspricht und… |
| 515 | Löschung | `schmerzen` | `*(nicht da)*` | …1 sehr leichten [___] entspricht und 10… |
| 516 | Löschung | `entspricht` | `*(nicht da)*` | …sehr leichten schmerzen [___] und 10 sehr… |
| 517 | Löschung | `und` | `*(nicht da)*` | …leichten schmerzen entspricht [___] 10 sehr starken… |
| 518 | Löschung | `10` | `*(nicht da)*` | …schmerzen entspricht und [___] sehr starken schmerzen… |
| 519 | Löschung | `sehr` | `*(nicht da)*` | …entspricht und 10 [___] starken schmerzen wo… |
| 520 | Löschung | `starken` | `*(nicht da)*` | …und 10 sehr [___] schmerzen wo würden… |
| 521 | Löschung | `schmerzen` | `*(nicht da)*` | …10 sehr starken [___] wo würden sie… |
| 522 | Löschung | `wo` | `*(nicht da)*` | …sehr starken schmerzen [___] würden sie die… |
| 523 | Löschung | `würden` | `*(nicht da)*` | …starken schmerzen wo [___] sie die schmerzen… |
| 524 | Löschung | `sie` | `*(nicht da)*` | …schmerzen wo würden [___] die schmerzen des… |
| 525 | Löschung | `die` | `*(nicht da)*` | …wo würden sie [___] schmerzen des daumens… |
| 526 | Löschung | `schmerzen` | `*(nicht da)*` | …würden sie die [___] des daumens einstufen… |
| 527 | Löschung | `des` | `*(nicht da)*` | …sie die schmerzen [___] daumens einstufen beim… |
| 528 | Löschung | `daumens` | `*(nicht da)*` | …die schmerzen des [___] einstufen beim daumen… |
| 529 | Löschung | `einstufen` | `*(nicht da)*` | …schmerzen des daumens [___] beim daumen würde… |
| 530 | Löschung | `beim` | `*(nicht da)*` | …des daumens einstufen [___] daumen würde ich… |
| 531 | Löschung | `daumen` | `*(nicht da)*` | …daumens einstufen beim [___] würde ich schon… |
| 532 | Löschung | `würde` | `*(nicht da)*` | …einstufen beim daumen [___] ich schon sagen… |
| 533 | Löschung | `ich` | `*(nicht da)*` | …beim daumen würde [___] schon sagen geht… |
| 534 | Löschung | `schon` | `*(nicht da)*` | …daumen würde ich [___] sagen geht es… |
| 535 | Löschung | `sagen` | `*(nicht da)*` | …würde ich schon [___] geht es so… |
| 536 | Substitution | `geht` | `ein` | …ich schon sagen [___] es so auf… |
| 537 | Substitution | `es` | `krankenhaus` | …schon sagen geht [___] so auf die… |
| 538 | Substitution | `so` | `eingeliefert` | …sagen geht es [___] auf die 7… |
| 539 | Substitution | `auf` | `worden` | …geht es so [___] die 7 zu… |
| 540 | Löschung | `7` | `*(nicht da)*` | …so auf die [___] zu vor allem… |
| 541 | Löschung | `zu` | `*(nicht da)*` | …auf die 7 [___] vor allem wenn… |
| 542 | Substitution | `vor` | `untersuchung` | …die 7 zu [___] allem wenn ich… |
| 543 | Substitution | `allem` | `umfasst` | …7 zu vor [___] wenn ich versuche… |
| 544 | Substitution | `wenn` | `eine` | …zu vor allem [___] ich versuche ihn… |
| 545 | Substitution | `ich` | `reihe` | …vor allem wenn [___] versuche ihn zu… |
| 546 | Substitution | `versuche` | `von` | …allem wenn ich [___] ihn zu bewegen… |
| 547 | Substitution | `ihn` | `fragen` | …wenn ich versuche [___] zu bewegen okay… |
| 548 | Löschung | `bewegen` | `*(nicht da)*` | …versuche ihn zu [___] okay was ist… |
| 549 | Löschung | `okay` | `*(nicht da)*` | …ihn zu bewegen [___] was ist das… |
| 550 | Löschung | `was` | `*(nicht da)*` | …zu bewegen okay [___] ist das denn… |
| 551 | Löschung | `ist` | `*(nicht da)*` | …bewegen okay was [___] das denn für… |
| 552 | Löschung | `das` | `*(nicht da)*` | …okay was ist [___] denn für ein… |
| 553 | Löschung | `denn` | `*(nicht da)*` | …was ist das [___] für ein schmerz… |
| 554 | Löschung | `für` | `*(nicht da)*` | …ist das denn [___] ein schmerz ist… |
| 555 | Löschung | `ein` | `*(nicht da)*` | …das denn für [___] schmerz ist das… |
| 556 | Löschung | `schmerz` | `*(nicht da)*` | …denn für ein [___] ist das ein… |
| 557 | Löschung | `ist` | `*(nicht da)*` | …für ein schmerz [___] das ein stechender… |
| 558 | Löschung | `das` | `*(nicht da)*` | …ein schmerz ist [___] ein stechender schmerz… |
| 559 | Löschung | `ein` | `*(nicht da)*` | …schmerz ist das [___] stechender schmerz ein… |
| 560 | Löschung | `stechender` | `*(nicht da)*` | …ist das ein [___] schmerz ein ziehender… |
| 561 | Löschung | `schmerz` | `*(nicht da)*` | …das ein stechender [___] ein ziehender schmerz… |
| 562 | Löschung | `ein` | `*(nicht da)*` | …ein stechender schmerz [___] ziehender schmerz ein… |
| 563 | Löschung | `ziehender` | `*(nicht da)*` | …stechender schmerz ein [___] schmerz ein brennender… |
| 564 | Löschung | `schmerz` | `*(nicht da)*` | …schmerz ein ziehender [___] ein brennender schmerz… |
| 565 | Löschung | `ein` | `*(nicht da)*` | …ein ziehender schmerz [___] brennender schmerz das… |
| 566 | Löschung | `brennender` | `*(nicht da)*` | …ziehender schmerz ein [___] schmerz das ist… |
| 567 | Löschung | `schmerz` | `*(nicht da)*` | …schmerz ein brennender [___] das ist ein… |
| 568 | Löschung | `das` | `*(nicht da)*` | …ein brennender schmerz [___] ist ein stechender… |
| 569 | Löschung | `ist` | `*(nicht da)*` | …brennender schmerz das [___] ein stechender schmerz… |
| 570 | Löschung | `ein` | `*(nicht da)*` | …schmerz das ist [___] stechender schmerz würde… |
| 571 | Löschung | `stechender` | `*(nicht da)*` | …das ist ein [___] schmerz würde ich… |
| 572 | Löschung | `schmerz` | `*(nicht da)*` | …ist ein stechender [___] würde ich sagen… |
| 573 | Löschung | `würde` | `*(nicht da)*` | …ein stechender schmerz [___] ich sagen sehr… |
| 574 | Löschung | `ich` | `*(nicht da)*` | …stechender schmerz würde [___] sagen sehr stark… |
| 575 | Löschung | `sagen` | `*(nicht da)*` | …schmerz würde ich [___] sehr stark stechend… |
| 576 | Löschung | `sehr` | `*(nicht da)*` | …würde ich sagen [___] stark stechend wenn… |
| 577 | Löschung | `stark` | `*(nicht da)*` | …ich sagen sehr [___] stechend wenn ich… |
| 578 | Löschung | `stechend` | `*(nicht da)*` | …sagen sehr stark [___] wenn ich versuche… |
| 579 | Löschung | `wenn` | `*(nicht da)*` | …sehr stark stechend [___] ich versuche den… |
| 580 | Löschung | `ich` | `*(nicht da)*` | …stark stechend wenn [___] versuche den zu… |
| 581 | Löschung | `versuche` | `*(nicht da)*` | …stechend wenn ich [___] den zu bewegen… |
| 582 | Substitution | `den` | `ihrer` | …wenn ich versuche [___] zu bewegen okay… |
| 583 | Substitution | `zu` | `gesundheit` | …ich versuche den [___] bewegen okay und… |
| 584 | Substitution | `bewegen` | `vorerkrankungen` | …versuche den zu [___] okay und wie… |
| 585 | Substitution | `okay` | `medikamenteneinnahme` | …den zu bewegen [___] und wie sieht… |
| 586 | Löschung | `wie` | `*(nicht da)*` | …bewegen okay und [___] sieht es am… |
| 587 | Löschung | `sieht` | `*(nicht da)*` | …okay und wie [___] es am knie… |
| 588 | Löschung | `es` | `*(nicht da)*` | …und wie sieht [___] am knie aus… |
| 589 | Löschung | `am` | `*(nicht da)*` | …wie sieht es [___] knie aus können… |
| 590 | Löschung | `knie` | `*(nicht da)*` | …sieht es am [___] aus können sie… |
| 591 | Löschung | `aus` | `*(nicht da)*` | …es am knie [___] können sie das… |
| 592 | Löschung | `können` | `*(nicht da)*` | …am knie aus [___] sie das knie… |
| 593 | Löschung | `sie` | `*(nicht da)*` | …knie aus können [___] das knie bewegen… |
| 594 | Löschung | `das` | `*(nicht da)*` | …aus können sie [___] knie bewegen sehr… |
| 595 | Löschung | `knie` | `*(nicht da)*` | …können sie das [___] bewegen sehr sehr… |
| 596 | Löschung | `bewegen` | `*(nicht da)*` | …sie das knie [___] sehr sehr schwer… |
| 597 | Löschung | `sehr` | `*(nicht da)*` | …das knie bewegen [___] sehr schwer da… |
| 598 | Löschung | `sehr` | `*(nicht da)*` | …knie bewegen sehr [___] schwer da tut… |
| 599 | Löschung | `schwer` | `*(nicht da)*` | …bewegen sehr sehr [___] da tut es… |
| 600 | Löschung | `da` | `*(nicht da)*` | …sehr sehr schwer [___] tut es wirklich… |
| 601 | Löschung | `tut` | `*(nicht da)*` | …sehr schwer da [___] es wirklich sehr… |
| 602 | Löschung | `es` | `*(nicht da)*` | …schwer da tut [___] wirklich sehr stark… |
| 603 | Löschung | `wirklich` | `*(nicht da)*` | …da tut es [___] sehr stark weh… |
| 604 | Löschung | `sehr` | `*(nicht da)*` | …tut es wirklich [___] stark weh wenn… |
| 605 | Löschung | `stark` | `*(nicht da)*` | …es wirklich sehr [___] weh wenn ich… |
| 606 | Löschung | `weh` | `*(nicht da)*` | …wirklich sehr stark [___] wenn ich versuche… |
| 607 | Löschung | `wenn` | `*(nicht da)*` | …sehr stark weh [___] ich versuche mein… |
| 608 | Löschung | `ich` | `*(nicht da)*` | …stark weh wenn [___] versuche mein knie… |
| 609 | Löschung | `versuche` | `*(nicht da)*` | …weh wenn ich [___] mein knie zu… |
| 610 | Löschung | `mein` | `*(nicht da)*` | …wenn ich versuche [___] knie zu bewegen… |
| 611 | Löschung | `knie` | `*(nicht da)*` | …ich versuche mein [___] zu bewegen es… |
| 612 | Löschung | `zu` | `*(nicht da)*` | …versuche mein knie [___] bewegen es tut… |
| 613 | Löschung | `bewegen` | `*(nicht da)*` | …mein knie zu [___] es tut selbst… |
| 614 | Löschung | `es` | `*(nicht da)*` | …knie zu bewegen [___] tut selbst weh… |
| 615 | Löschung | `tut` | `*(nicht da)*` | …zu bewegen es [___] selbst weh wenn… |
| 616 | Löschung | `selbst` | `*(nicht da)*` | …bewegen es tut [___] weh wenn ich… |
| 617 | Löschung | `weh` | `*(nicht da)*` | …es tut selbst [___] wenn ich gerade… |
| 618 | Löschung | `wenn` | `*(nicht da)*` | …tut selbst weh [___] ich gerade einfach… |
| 619 | Löschung | `ich` | `*(nicht da)*` | …selbst weh wenn [___] gerade einfach so… |
| 620 | Löschung | `gerade` | `*(nicht da)*` | …weh wenn ich [___] einfach so hier… |
| 621 | Löschung | `einfach` | `*(nicht da)*` | …wenn ich gerade [___] so hier sitze… |
| 622 | Löschung | `so` | `*(nicht da)*` | …ich gerade einfach [___] hier sitze okay… |
| 623 | Löschung | `hier` | `*(nicht da)*` | …gerade einfach so [___] sitze okay sogar… |
| 624 | Löschung | `sitze` | `*(nicht da)*` | …einfach so hier [___] okay sogar im… |
| 625 | Löschung | `okay` | `*(nicht da)*` | …so hier sitze [___] sogar im ruhezustand… |
| 626 | Löschung | `sogar` | `*(nicht da)*` | …hier sitze okay [___] im ruhezustand ja… |
| 627 | Löschung | `im` | `*(nicht da)*` | …sitze okay sogar [___] ruhezustand ja wo… |
| 628 | Löschung | `ruhezustand` | `*(nicht da)*` | …okay sogar im [___] ja wo würden… |
| 629 | Löschung | `ja` | `*(nicht da)*` | …sogar im ruhezustand [___] wo würden sie… |
| 630 | Substitution | `wo` | `ihrem` | …im ruhezustand ja [___] würden sie die… |
| 631 | Substitution | `würden` | `sozialleben` | …ruhezustand ja wo [___] sie die schmerzen… |
| 632 | Löschung | `die` | `*(nicht da)*` | …wo würden sie [___] schmerzen hier einstufen… |
| 633 | Löschung | `schmerzen` | `*(nicht da)*` | …würden sie die [___] hier einstufen da… |
| 634 | Löschung | `hier` | `*(nicht da)*` | …sie die schmerzen [___] einstufen da würde… |
| 635 | Löschung | `einstufen` | `*(nicht da)*` | …die schmerzen hier [___] da würde ich… |
| 636 | Löschung | `da` | `*(nicht da)*` | …schmerzen hier einstufen [___] würde ich sagen… |
| 637 | Löschung | `würde` | `*(nicht da)*` | …hier einstufen da [___] ich sagen bei… |
| 638 | Löschung | `ich` | `*(nicht da)*` | …einstufen da würde [___] sagen bei 8… |
| 639 | Löschung | `sagen` | `*(nicht da)*` | …da würde ich [___] bei 8 wenn… |
| 640 | Löschung | `bei` | `*(nicht da)*` | …würde ich sagen [___] 8 wenn ich… |
| 641 | Löschung | `8` | `*(nicht da)*` | …ich sagen bei [___] wenn ich sitze… |
| 642 | Löschung | `wenn` | `*(nicht da)*` | …sagen bei 8 [___] ich sitze und… |
| 643 | Löschung | `ich` | `*(nicht da)*` | …bei 8 wenn [___] sitze und wenn… |
| 644 | Löschung | `sitze` | `*(nicht da)*` | …8 wenn ich [___] und wenn ich… |
| 645 | Löschung | `und` | `*(nicht da)*` | …wenn ich sitze [___] wenn ich versuche… |
| 646 | Löschung | `wenn` | `*(nicht da)*` | …ich sitze und [___] ich versuche mein… |
| 647 | Löschung | `ich` | `*(nicht da)*` | …sitze und wenn [___] versuche mein knie… |
| 648 | Löschung | `versuche` | `*(nicht da)*` | …und wenn ich [___] mein knie zu… |
| 649 | Löschung | `mein` | `*(nicht da)*` | …wenn ich versuche [___] knie zu bewegen… |
| 650 | Löschung | `knie` | `*(nicht da)*` | …ich versuche mein [___] zu bewegen ist… |
| 651 | Löschung | `zu` | `*(nicht da)*` | …versuche mein knie [___] bewegen ist es… |
| 652 | Löschung | `bewegen` | `*(nicht da)*` | …mein knie zu [___] ist es wirklich… |
| 653 | Löschung | `es` | `*(nicht da)*` | …zu bewegen ist [___] wirklich unerträglich okay… |
| 654 | Löschung | `wirklich` | `*(nicht da)*` | …bewegen ist es [___] unerträglich okay okay… |
| 655 | Löschung | `unerträglich` | `*(nicht da)*` | …ist es wirklich [___] okay okay gut… |
| 656 | Löschung | `okay` | `*(nicht da)*` | …es wirklich unerträglich [___] okay gut strahlen… |
| 657 | Löschung | `okay` | `*(nicht da)*` | …wirklich unerträglich okay [___] gut strahlen die… |
| 658 | Löschung | `gut` | `*(nicht da)*` | …unerträglich okay okay [___] strahlen die schmerzen… |
| 659 | Substitution | `strahlen` | `frisch` | …okay okay gut [___] die schmerzen noch… |
| 660 | Substitution | `die` | `verheiratet` | …okay gut strahlen [___] schmerzen noch in… |
| 661 | Substitution | `schmerzen` | `und` | …gut strahlen die [___] noch in andere… |
| 662 | Substitution | `noch` | `arbeitet` | …strahlen die schmerzen [___] in andere körperregionen… |
| 663 | Löschung | `andere` | `*(nicht da)*` | …schmerzen noch in [___] körperregionen aus nein… |
| 664 | Löschung | `körperregionen` | `*(nicht da)*` | …noch in andere [___] aus nein das… |
| 665 | Löschung | `aus` | `*(nicht da)*` | …in andere körperregionen [___] nein das zum… |
| 666 | Löschung | `nein` | `*(nicht da)*` | …andere körperregionen aus [___] das zum glück… |
| 667 | Löschung | `das` | `*(nicht da)*` | …körperregionen aus nein [___] zum glück nicht… |
| 668 | Löschung | `zum` | `*(nicht da)*` | …aus nein das [___] glück nicht okay… |
| 669 | Löschung | `glück` | `*(nicht da)*` | …nein das zum [___] nicht okay wie… |
| 670 | Löschung | `nicht` | `*(nicht da)*` | …das zum glück [___] okay wie sieht… |
| 671 | Löschung | `okay` | `*(nicht da)*` | …zum glück nicht [___] wie sieht es… |
| 672 | Löschung | `wie` | `*(nicht da)*` | …glück nicht okay [___] sieht es an… |
| 673 | Löschung | `sieht` | `*(nicht da)*` | …nicht okay wie [___] es an der… |
| 674 | Löschung | `es` | `*(nicht da)*` | …okay wie sieht [___] an der hand… |
| 675 | Löschung | `an` | `*(nicht da)*` | …wie sieht es [___] der hand aus… |
| 676 | Löschung | `der` | `*(nicht da)*` | …sieht es an [___] hand aus am… |
| 677 | Löschung | `hand` | `*(nicht da)*` | …es an der [___] aus am daumen… |
| 678 | Löschung | `aus` | `*(nicht da)*` | …an der hand [___] am daumen strahlen… |
| 679 | Löschung | `am` | `*(nicht da)*` | …der hand aus [___] daumen strahlen die… |
| 680 | Substitution | `daumen` | `einer` | …hand aus am [___] strahlen die schmerzen… |
| 681 | Substitution | `strahlen` | `marketingagentur` | …aus am daumen [___] die schmerzen da… |
| 682 | Löschung | `schmerzen` | `*(nicht da)*` | …daumen strahlen die [___] da irgendwie ins… |
| 683 | Löschung | `da` | `*(nicht da)*` | …strahlen die schmerzen [___] irgendwie ins handgelenk… |
| 684 | Löschung | `irgendwie` | `*(nicht da)*` | …die schmerzen da [___] ins handgelenk aus… |
| 685 | Löschung | `ins` | `*(nicht da)*` | …schmerzen da irgendwie [___] handgelenk aus oder… |
| 686 | Löschung | `handgelenk` | `*(nicht da)*` | …da irgendwie ins [___] aus oder in… |
| 687 | Löschung | `aus` | `*(nicht da)*` | …irgendwie ins handgelenk [___] oder in andere… |
| 688 | Löschung | `oder` | `*(nicht da)*` | …ins handgelenk aus [___] in andere finger… |
| 689 | Löschung | `in` | `*(nicht da)*` | …handgelenk aus oder [___] andere finger auch… |
| 690 | Löschung | `andere` | `*(nicht da)*` | …aus oder in [___] finger auch nicht… |
| 691 | Löschung | `finger` | `*(nicht da)*` | …oder in andere [___] auch nicht nein… |
| 692 | Löschung | `auch` | `*(nicht da)*` | …in andere finger [___] nicht nein okay… |
| 693 | Löschung | `nicht` | `*(nicht da)*` | …andere finger auch [___] nein okay sehr… |
| 694 | Löschung | `nein` | `*(nicht da)*` | …finger auch nicht [___] okay sehr sehr… |
| 695 | Löschung | `okay` | `*(nicht da)*` | …auch nicht nein [___] sehr sehr gut… |
| 696 | Löschung | `sehr` | `*(nicht da)*` | …nicht nein okay [___] sehr gut können… |
| 697 | Löschung | `sehr` | `*(nicht da)*` | …nein okay sehr [___] gut können sie… |
| 698 | Löschung | `gut` | `*(nicht da)*` | …okay sehr sehr [___] können sie sich… |
| 699 | Löschung | `können` | `*(nicht da)*` | …sehr sehr gut [___] sie sich an… |
| 700 | Löschung | `sie` | `*(nicht da)*` | …sehr gut können [___] sich an den… |
| 701 | Löschung | `sich` | `*(nicht da)*` | …gut können sie [___] an den unfall… |
| 702 | Löschung | `an` | `*(nicht da)*` | …können sie sich [___] den unfall erinnern… |
| 703 | Löschung | `den` | `*(nicht da)*` | …sie sich an [___] unfall erinnern frau… |
| 704 | Löschung | `unfall` | `*(nicht da)*` | …sich an den [___] erinnern frau beckenwestfalen… |
| 705 | Löschung | `erinnern` | `*(nicht da)*` | …an den unfall [___] frau beckenwestfalen ich… |
| 706 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …unfall erinnern frau [___] ich kann mich… |
| 707 | Löschung | `ich` | `*(nicht da)*` | …erinnern frau beckenwestfalen [___] kann mich gut… |
| 708 | Löschung | `kann` | `*(nicht da)*` | …frau beckenwestfalen ich [___] mich gut daran… |
| 709 | Löschung | `mich` | `*(nicht da)*` | …beckenwestfalen ich kann [___] gut daran erinnern… |
| 710 | Löschung | `gut` | `*(nicht da)*` | …ich kann mich [___] daran erinnern ja… |
| 711 | Löschung | `daran` | `*(nicht da)*` | …kann mich gut [___] erinnern ja ich… |
| 712 | Löschung | `erinnern` | `*(nicht da)*` | …mich gut daran [___] ja ich war… |
| 713 | Löschung | `ja` | `*(nicht da)*` | …gut daran erinnern [___] ich war am… |
| 714 | Löschung | `ich` | `*(nicht da)*` | …daran erinnern ja [___] war am anfang… |
| 715 | Löschung | `war` | `*(nicht da)*` | …erinnern ja ich [___] am anfang zwar… |
| 716 | Löschung | `am` | `*(nicht da)*` | …ja ich war [___] anfang zwar etwas… |
| 717 | Löschung | `anfang` | `*(nicht da)*` | …ich war am [___] zwar etwas benebelt… |
| 718 | Löschung | `zwar` | `*(nicht da)*` | …war am anfang [___] etwas benebelt und… |
| 719 | Löschung | `etwas` | `*(nicht da)*` | …am anfang zwar [___] benebelt und mir… |
| 720 | Löschung | `benebelt` | `*(nicht da)*` | …anfang zwar etwas [___] und mir war… |
| 721 | Löschung | `und` | `*(nicht da)*` | …zwar etwas benebelt [___] mir war es… |
| 722 | Löschung | `mir` | `*(nicht da)*` | …etwas benebelt und [___] war es ziemlich… |
| 723 | Löschung | `war` | `*(nicht da)*` | …benebelt und mir [___] es ziemlich schwindelig… |
| 724 | Löschung | `es` | `*(nicht da)*` | …und mir war [___] ziemlich schwindelig aber… |
| 725 | Löschung | `ziemlich` | `*(nicht da)*` | …mir war es [___] schwindelig aber ich… |
| 726 | Löschung | `schwindelig` | `*(nicht da)*` | …war es ziemlich [___] aber ich denke… |
| 727 | Löschung | `aber` | `*(nicht da)*` | …es ziemlich schwindelig [___] ich denke das… |
| 728 | Löschung | `ich` | `*(nicht da)*` | …ziemlich schwindelig aber [___] denke das lag… |
| 729 | Löschung | `denke` | `*(nicht da)*` | …schwindelig aber ich [___] das lag vielleicht… |
| 730 | Löschung | `das` | `*(nicht da)*` | …aber ich denke [___] lag vielleicht am… |
| 731 | Löschung | `lag` | `*(nicht da)*` | …ich denke das [___] vielleicht am schock… |
| 732 | Löschung | `vielleicht` | `*(nicht da)*` | …denke das lag [___] am schock im… |
| 733 | Löschung | `am` | `*(nicht da)*` | …das lag vielleicht [___] schock im ersten… |
| 734 | Löschung | `schock` | `*(nicht da)*` | …lag vielleicht am [___] im ersten moment… |
| 735 | Löschung | `im` | `*(nicht da)*` | …vielleicht am schock [___] ersten moment okay… |
| 736 | Löschung | `ersten` | `*(nicht da)*` | …am schock im [___] moment okay gibt… |
| 737 | Löschung | `moment` | `*(nicht da)*` | …schock im ersten [___] okay gibt es… |
| 738 | Löschung | `okay` | `*(nicht da)*` | …im ersten moment [___] gibt es sonst… |
| 739 | Löschung | `gibt` | `*(nicht da)*` | …ersten moment okay [___] es sonst etwas… |
| 740 | Löschung | `es` | `*(nicht da)*` | …moment okay gibt [___] sonst etwas was… |
| 741 | Löschung | `sonst` | `*(nicht da)*` | …okay gibt es [___] etwas was ihnen… |
| 742 | Löschung | `etwas` | `*(nicht da)*` | …gibt es sonst [___] was ihnen aufgefallen… |
| 743 | Löschung | `was` | `*(nicht da)*` | …es sonst etwas [___] ihnen aufgefallen ist… |
| 744 | Löschung | `ihnen` | `*(nicht da)*` | …sonst etwas was [___] aufgefallen ist seit… |
| 745 | Löschung | `aufgefallen` | `*(nicht da)*` | …etwas was ihnen [___] ist seit dem… |
| 746 | Löschung | `ist` | `*(nicht da)*` | …was ihnen aufgefallen [___] seit dem unfall… |
| 747 | Löschung | `seit` | `*(nicht da)*` | …ihnen aufgefallen ist [___] dem unfall was… |
| 748 | Löschung | `dem` | `*(nicht da)*` | …aufgefallen ist seit [___] unfall was ich… |
| 749 | Löschung | `unfall` | `*(nicht da)*` | …ist seit dem [___] was ich wissen… |
| 750 | Löschung | `was` | `*(nicht da)*` | …seit dem unfall [___] ich wissen sollte… |
| 751 | Löschung | `ich` | `*(nicht da)*` | …dem unfall was [___] wissen sollte ist… |
| 752 | Löschung | `wissen` | `*(nicht da)*` | …unfall was ich [___] sollte ist ihnen… |
| 753 | Löschung | `sollte` | `*(nicht da)*` | …was ich wissen [___] ist ihnen übel… |
| 754 | Löschung | `ist` | `*(nicht da)*` | …ich wissen sollte [___] ihnen übel geworden… |
| 755 | Löschung | `ihnen` | `*(nicht da)*` | …wissen sollte ist [___] übel geworden oder… |
| 756 | Löschung | `übel` | `*(nicht da)*` | …sollte ist ihnen [___] geworden oder vielleicht… |
| 757 | Löschung | `geworden` | `*(nicht da)*` | …ist ihnen übel [___] oder vielleicht doch… |
| 758 | Löschung | `oder` | `*(nicht da)*` | …ihnen übel geworden [___] vielleicht doch nochmal… |
| 759 | Löschung | `vielleicht` | `*(nicht da)*` | …übel geworden oder [___] doch nochmal schwarz… |
| 760 | Löschung | `doch` | `*(nicht da)*` | …geworden oder vielleicht [___] nochmal schwarz vor… |
| 761 | Löschung | `nochmal` | `*(nicht da)*` | …oder vielleicht doch [___] schwarz vor augen… |
| 762 | Löschung | `schwarz` | `*(nicht da)*` | …vielleicht doch nochmal [___] vor augen oder… |
| 763 | Löschung | `vor` | `*(nicht da)*` | …doch nochmal schwarz [___] augen oder fühlen… |
| 764 | Löschung | `augen` | `*(nicht da)*` | …nochmal schwarz vor [___] oder fühlen sie… |
| 765 | Löschung | `oder` | `*(nicht da)*` | …schwarz vor augen [___] fühlen sie sich… |
| 766 | Löschung | `fühlen` | `*(nicht da)*` | …vor augen oder [___] sie sich seltsam… |
| 767 | Löschung | `sie` | `*(nicht da)*` | …augen oder fühlen [___] sich seltsam seitdem… |
| 768 | Substitution | `sich` | `hat` | …oder fühlen sie [___] seltsam seitdem nein… |
| 769 | Substitution | `seltsam` | `keine` | …fühlen sie sich [___] seitdem nein außer… |
| 770 | Substitution | `seitdem` | `regelmäßige` | …sie sich seltsam [___] nein außer dass… |
| 771 | Substitution | `nein` | `medikamenteneinnahme` | …sich seltsam seitdem [___] außer dass ich… |
| 772 | Löschung | `dass` | `*(nicht da)*` | …seitdem nein außer [___] ich sehr starke… |
| 773 | Löschung | `ich` | `*(nicht da)*` | …nein außer dass [___] sehr starke schmerzen… |
| 774 | Löschung | `sehr` | `*(nicht da)*` | …außer dass ich [___] starke schmerzen habe… |
| 775 | Löschung | `starke` | `*(nicht da)*` | …dass ich sehr [___] schmerzen habe ist… |
| 776 | Löschung | `schmerzen` | `*(nicht da)*` | …ich sehr starke [___] habe ist mir… |
| 777 | Löschung | `habe` | `*(nicht da)*` | …sehr starke schmerzen [___] ist mir nichts… |
| 778 | Löschung | `ist` | `*(nicht da)*` | …starke schmerzen habe [___] mir nichts anderes… |
| 779 | Löschung | `mir` | `*(nicht da)*` | …schmerzen habe ist [___] nichts anderes aufgefallen… |
| 780 | Löschung | `nichts` | `*(nicht da)*` | …habe ist mir [___] anderes aufgefallen und… |
| 781 | Löschung | `anderes` | `*(nicht da)*` | …ist mir nichts [___] aufgefallen und dass… |
| 782 | Löschung | `aufgefallen` | `*(nicht da)*` | …mir nichts anderes [___] und dass ich… |
| 783 | Löschung | `und` | `*(nicht da)*` | …nichts anderes aufgefallen [___] dass ich am… |
| 784 | Löschung | `dass` | `*(nicht da)*` | …anderes aufgefallen und [___] ich am anfang… |
| 785 | Löschung | `ich` | `*(nicht da)*` | …aufgefallen und dass [___] am anfang nur… |
| 786 | Löschung | `am` | `*(nicht da)*` | …und dass ich [___] anfang nur etwas… |
| 787 | Löschung | `anfang` | `*(nicht da)*` | …dass ich am [___] nur etwas benebelt… |
| 788 | Löschung | `nur` | `*(nicht da)*` | …ich am anfang [___] etwas benebelt war… |
| 789 | Löschung | `etwas` | `*(nicht da)*` | …am anfang nur [___] benebelt war aber… |
| 790 | Substitution | `benebelt` | `der` | …anfang nur etwas [___] war aber jetzt… |
| 791 | Substitution | `war` | `pille` | …nur etwas benebelt [___] aber jetzt bin… |
| 792 | Löschung | `jetzt` | `*(nicht da)*` | …benebelt war aber [___] bin ich ganz… |
| 793 | Löschung | `bin` | `*(nicht da)*` | …war aber jetzt [___] ich ganz klar… |
| 794 | Löschung | `ich` | `*(nicht da)*` | …aber jetzt bin [___] ganz klar okay… |
| 795 | Löschung | `ganz` | `*(nicht da)*` | …jetzt bin ich [___] klar okay gut… |
| 796 | Löschung | `klar` | `*(nicht da)*` | …bin ich ganz [___] okay gut sehr… |
| 797 | Löschung | `okay` | `*(nicht da)*` | …ich ganz klar [___] gut sehr sehr… |
| 798 | Löschung | `gut` | `*(nicht da)*` | …ganz klar okay [___] sehr sehr gut… |
| 799 | Löschung | `sehr` | `*(nicht da)*` | …klar okay gut [___] sehr gut frau… |
| 800 | Löschung | `sehr` | `*(nicht da)*` | …okay gut sehr [___] gut frau beckenwestfalen… |
| 801 | Löschung | `gut` | `*(nicht da)*` | …gut sehr sehr [___] frau beckenwestfalen haben… |
| 802 | Löschung | `frau` | `*(nicht da)*` | …sehr sehr gut [___] beckenwestfalen haben sie… |
| 803 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …sehr gut frau [___] haben sie irgendwelche… |
| 804 | Löschung | `haben` | `*(nicht da)*` | …gut frau beckenwestfalen [___] sie irgendwelche vorerkrankungen… |
| 805 | Löschung | `irgendwelche` | `*(nicht da)*` | …beckenwestfalen haben sie [___] vorerkrankungen von denen… |
| 806 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …haben sie irgendwelche [___] von denen ich… |
| 807 | Löschung | `von` | `*(nicht da)*` | …sie irgendwelche vorerkrankungen [___] denen ich wissen… |
| 808 | Löschung | `denen` | `*(nicht da)*` | …irgendwelche vorerkrankungen von [___] ich wissen sollte… |
| 809 | Löschung | `ich` | `*(nicht da)*` | …vorerkrankungen von denen [___] wissen sollte wie… |
| 810 | Löschung | `wissen` | `*(nicht da)*` | …von denen ich [___] sollte wie zum… |
| 811 | Löschung | `sollte` | `*(nicht da)*` | …denen ich wissen [___] wie zum beispiel… |
| 812 | Löschung | `wie` | `*(nicht da)*` | …ich wissen sollte [___] zum beispiel erhöhten… |
| 813 | Löschung | `zum` | `*(nicht da)*` | …wissen sollte wie [___] beispiel erhöhten blutdruck… |
| 814 | Löschung | `beispiel` | `*(nicht da)*` | …sollte wie zum [___] erhöhten blutdruck oder… |
| 815 | Löschung | `erhöhten` | `*(nicht da)*` | …wie zum beispiel [___] blutdruck oder diabetes… |
| 816 | Löschung | `blutdruck` | `*(nicht da)*` | …zum beispiel erhöhten [___] oder diabetes oder… |
| 817 | Löschung | `oder` | `*(nicht da)*` | …beispiel erhöhten blutdruck [___] diabetes oder etwas… |
| 818 | Löschung | `diabetes` | `*(nicht da)*` | …erhöhten blutdruck oder [___] oder etwas anderes… |
| 819 | Löschung | `oder` | `*(nicht da)*` | …blutdruck oder diabetes [___] etwas anderes nichts… |
| 820 | Löschung | `etwas` | `*(nicht da)*` | …oder diabetes oder [___] anderes nichts ernsthaftes… |
| 821 | Löschung | `anderes` | `*(nicht da)*` | …diabetes oder etwas [___] nichts ernsthaftes ich… |
| 822 | Löschung | `nichts` | `*(nicht da)*` | …oder etwas anderes [___] ernsthaftes ich hatte… |
| 823 | Löschung | `ernsthaftes` | `*(nicht da)*` | …etwas anderes nichts [___] ich hatte eine… |
| 824 | Löschung | `ich` | `*(nicht da)*` | …anderes nichts ernsthaftes [___] hatte eine laktoseintoleranz… |
| 825 | Löschung | `eine` | `*(nicht da)*` | …ernsthaftes ich hatte [___] laktoseintoleranz vor einigen… |
| 826 | Löschung | `laktoseintoleranz` | `*(nicht da)*` | …ich hatte eine [___] vor einigen jahren… |
| 827 | Löschung | `vor` | `*(nicht da)*` | …hatte eine laktoseintoleranz [___] einigen jahren sie… |
| 828 | Löschung | `einigen` | `*(nicht da)*` | …eine laktoseintoleranz vor [___] jahren sie ist… |
| 829 | Löschung | `jahren` | `*(nicht da)*` | …laktoseintoleranz vor einigen [___] sie ist allerdings… |
| 830 | Löschung | `sie` | `*(nicht da)*` | …vor einigen jahren [___] ist allerdings schon… |
| 831 | Löschung | `ist` | `*(nicht da)*` | …einigen jahren sie [___] allerdings schon weg… |
| 832 | Löschung | `allerdings` | `*(nicht da)*` | …jahren sie ist [___] schon weg und… |
| 833 | Löschung | `schon` | `*(nicht da)*` | …sie ist allerdings [___] weg und jetzt… |
| 834 | Löschung | `weg` | `*(nicht da)*` | …ist allerdings schon [___] und jetzt wurde… |
| 835 | Löschung | `und` | `*(nicht da)*` | …allerdings schon weg [___] jetzt wurde bei… |
| 836 | Löschung | `jetzt` | `*(nicht da)*` | …schon weg und [___] wurde bei mir… |
| 837 | Löschung | `wurde` | `*(nicht da)*` | …weg und jetzt [___] bei mir vor… |
| 838 | Löschung | `bei` | `*(nicht da)*` | …und jetzt wurde [___] mir vor drei… |
| 839 | Löschung | `mir` | `*(nicht da)*` | …jetzt wurde bei [___] vor drei wochen… |
| 840 | Löschung | `vor` | `*(nicht da)*` | …wurde bei mir [___] drei wochen eine… |
| 841 | Löschung | `drei` | `*(nicht da)*` | …bei mir vor [___] wochen eine histaminunverträglichkeit… |
| 842 | Löschung | `wochen` | `*(nicht da)*` | …mir vor drei [___] eine histaminunverträglichkeit festgestellt… |
| 843 | Löschung | `histaminunverträglichkeit` | `*(nicht da)*` | …drei wochen eine [___] festgestellt wie äußert… |
| 844 | Löschung | `festgestellt` | `*(nicht da)*` | …wochen eine histaminunverträglichkeit [___] wie äußert sich… |
| 845 | Löschung | `wie` | `*(nicht da)*` | …eine histaminunverträglichkeit festgestellt [___] äußert sich die… |
| 846 | Löschung | `äußert` | `*(nicht da)*` | …histaminunverträglichkeit festgestellt wie [___] sich die unverträglichkeit… |
| 847 | Löschung | `sich` | `*(nicht da)*` | …festgestellt wie äußert [___] die unverträglichkeit wenn… |
| 848 | Löschung | `die` | `*(nicht da)*` | …wie äußert sich [___] unverträglichkeit wenn ich… |
| 849 | Löschung | `unverträglichkeit` | `*(nicht da)*` | …äußert sich die [___] wenn ich bestimmte… |
| 850 | Löschung | `wenn` | `*(nicht da)*` | …sich die unverträglichkeit [___] ich bestimmte sachen… |
| 851 | Löschung | `ich` | `*(nicht da)*` | …die unverträglichkeit wenn [___] bestimmte sachen esse… |
| 852 | Löschung | `bestimmte` | `*(nicht da)*` | …unverträglichkeit wenn ich [___] sachen esse oder… |
| 853 | Löschung | `sachen` | `*(nicht da)*` | …wenn ich bestimmte [___] esse oder trinke… |
| 854 | Löschung | `esse` | `*(nicht da)*` | …ich bestimmte sachen [___] oder trinke vor… |
| 855 | Löschung | `oder` | `*(nicht da)*` | …bestimmte sachen esse [___] trinke vor allem… |
| 856 | Löschung | `trinke` | `*(nicht da)*` | …sachen esse oder [___] vor allem in… |
| 857 | Löschung | `vor` | `*(nicht da)*` | …esse oder trinke [___] allem in kombination… |
| 858 | Löschung | `allem` | `*(nicht da)*` | …oder trinke vor [___] in kombination dann… |
| 859 | Löschung | `in` | `*(nicht da)*` | …trinke vor allem [___] kombination dann bekomme… |
| 860 | Löschung | `kombination` | `*(nicht da)*` | …vor allem in [___] dann bekomme ich… |
| 861 | Löschung | `dann` | `*(nicht da)*` | …allem in kombination [___] bekomme ich starke… |
| 862 | Löschung | `bekomme` | `*(nicht da)*` | …in kombination dann [___] ich starke bauchschmerzen… |
| 863 | Löschung | `ich` | `*(nicht da)*` | …kombination dann bekomme [___] starke bauchschmerzen übelkeit… |
| 864 | Löschung | `starke` | `*(nicht da)*` | …dann bekomme ich [___] bauchschmerzen übelkeit manchmal… |
| 865 | Löschung | `bauchschmerzen` | `*(nicht da)*` | …bekomme ich starke [___] übelkeit manchmal und… |
| 866 | Löschung | `übelkeit` | `*(nicht da)*` | …ich starke bauchschmerzen [___] manchmal und manchmal… |
| 867 | Löschung | `manchmal` | `*(nicht da)*` | …starke bauchschmerzen übelkeit [___] und manchmal auch… |
| 868 | Löschung | `und` | `*(nicht da)*` | …bauchschmerzen übelkeit manchmal [___] manchmal auch einen… |
| 869 | Löschung | `manchmal` | `*(nicht da)*` | …übelkeit manchmal und [___] auch einen ausschlag… |
| 870 | Löschung | `auch` | `*(nicht da)*` | …manchmal und manchmal [___] einen ausschlag hier… |
| 871 | Löschung | `einen` | `*(nicht da)*` | …und manchmal auch [___] ausschlag hier im… |
| 872 | Löschung | `ausschlag` | `*(nicht da)*` | …manchmal auch einen [___] hier im dekolleté… |
| 873 | Löschung | `hier` | `*(nicht da)*` | …auch einen ausschlag [___] im dekolleté bereich… |
| 874 | Löschung | `im` | `*(nicht da)*` | …einen ausschlag hier [___] dekolleté bereich okay… |
| 875 | Löschung | `dekolleté` | `*(nicht da)*` | …ausschlag hier im [___] bereich okay sonst… |
| 876 | Löschung | `bereich` | `*(nicht da)*` | …hier im dekolleté [___] okay sonst gibt… |
| 877 | Löschung | `okay` | `*(nicht da)*` | …im dekolleté bereich [___] sonst gibt es… |
| 878 | Löschung | `sonst` | `*(nicht da)*` | …dekolleté bereich okay [___] gibt es aber… |
| 879 | Löschung | `gibt` | `*(nicht da)*` | …bereich okay sonst [___] es aber keine… |
| 880 | Löschung | `es` | `*(nicht da)*` | …okay sonst gibt [___] aber keine vorerkrankungen… |
| 881 | Löschung | `aber` | `*(nicht da)*` | …sonst gibt es [___] keine vorerkrankungen nein… |
| 882 | Löschung | `keine` | `*(nicht da)*` | …gibt es aber [___] vorerkrankungen nein nein… |
| 883 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …es aber keine [___] nein nein okay… |
| 884 | Löschung | `nein` | `*(nicht da)*` | …aber keine vorerkrankungen [___] nein okay sehr… |
| 885 | Löschung | `nein` | `*(nicht da)*` | …keine vorerkrankungen nein [___] okay sehr gut… |
| 886 | Löschung | `okay` | `*(nicht da)*` | …vorerkrankungen nein nein [___] sehr gut frau… |
| 887 | Löschung | `sehr` | `*(nicht da)*` | …nein nein okay [___] gut frau beckenwestfalen… |
| 888 | Löschung | `gut` | `*(nicht da)*` | …nein okay sehr [___] frau beckenwestfalen sind… |
| 889 | Löschung | `frau` | `*(nicht da)*` | …okay sehr gut [___] beckenwestfalen sind sie… |
| 890 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …sehr gut frau [___] sind sie schon… |
| 891 | Löschung | `sind` | `*(nicht da)*` | …gut frau beckenwestfalen [___] sie schon einmal… |
| 892 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen sind [___] schon einmal operiert… |
| 893 | Löschung | `schon` | `*(nicht da)*` | …beckenwestfalen sind sie [___] einmal operiert worden… |
| 894 | Löschung | `einmal` | `*(nicht da)*` | …sind sie schon [___] operiert worden ja… |
| 895 | Löschung | `operiert` | `*(nicht da)*` | …sie schon einmal [___] worden ja ich… |
| 896 | Löschung | `worden` | `*(nicht da)*` | …schon einmal operiert [___] ja ich wurde… |
| 897 | Löschung | `ja` | `*(nicht da)*` | …einmal operiert worden [___] ich wurde vor… |
| 898 | Löschung | `ich` | `*(nicht da)*` | …operiert worden ja [___] wurde vor zwei… |
| 899 | Löschung | `wurde` | `*(nicht da)*` | …worden ja ich [___] vor zwei jahren… |
| 900 | Löschung | `vor` | `*(nicht da)*` | …ja ich wurde [___] zwei jahren am… |
| 901 | Löschung | `zwei` | `*(nicht da)*` | …ich wurde vor [___] jahren am fuß… |
| 902 | Substitution | `jahren` | `operation` | …wurde vor zwei [___] am fuß operiert… |
| 903 | Löschung | `fuß` | `*(nicht da)*` | …zwei jahren am [___] operiert mir wurde… |
| 904 | Löschung | `operiert` | `*(nicht da)*` | …jahren am fuß [___] mir wurde ein… |
| 905 | Löschung | `mir` | `*(nicht da)*` | …am fuß operiert [___] wurde ein halux… |
| 906 | Löschung | `wurde` | `*(nicht da)*` | …fuß operiert mir [___] ein halux valgus… |
| 907 | Löschung | `ein` | `*(nicht da)*` | …operiert mir wurde [___] halux valgus entfernt… |
| 908 | Löschung | `halux` | `*(nicht da)*` | …mir wurde ein [___] valgus entfernt ein… |
| 909 | Löschung | `valgus` | `*(nicht da)*` | …wurde ein halux [___] entfernt ein halux… |
| 910 | Löschung | `entfernt` | `*(nicht da)*` | …ein halux valgus [___] ein halux valgus… |
| 911 | Löschung | `ein` | `*(nicht da)*` | …halux valgus entfernt [___] halux valgus und… |
| 912 | Löschung | `halux` | `*(nicht da)*` | …valgus entfernt ein [___] valgus und welcher… |
| 913 | Löschung | `valgus` | `*(nicht da)*` | …entfernt ein halux [___] und welcher fuß… |
| 914 | Löschung | `und` | `*(nicht da)*` | …ein halux valgus [___] welcher fuß war… |
| 915 | Löschung | `welcher` | `*(nicht da)*` | …halux valgus und [___] fuß war das… |
| 916 | Löschung | `fuß` | `*(nicht da)*` | …valgus und welcher [___] war das der… |
| 917 | Löschung | `war` | `*(nicht da)*` | …und welcher fuß [___] das der rechte… |
| 918 | Löschung | `das` | `*(nicht da)*` | …welcher fuß war [___] der rechte fuß… |
| 919 | Löschung | `der` | `*(nicht da)*` | …fuß war das [___] rechte fuß der… |
| 920 | Löschung | `rechte` | `*(nicht da)*` | …war das der [___] fuß der rechte… |
| 921 | Löschung | `fuß` | `*(nicht da)*` | …das der rechte [___] der rechte fuß… |
| 922 | Löschung | `der` | `*(nicht da)*` | …der rechte fuß [___] rechte fuß sind… |
| 923 | Substitution | `rechte` | `rechten` | …rechte fuß der [___] fuß sind irgendwelche… |
| 924 | Löschung | `sind` | `*(nicht da)*` | …der rechte fuß [___] irgendwelche komplikationen während… |
| 925 | Löschung | `irgendwelche` | `*(nicht da)*` | …rechte fuß sind [___] komplikationen während oder… |
| 926 | Löschung | `komplikationen` | `*(nicht da)*` | …fuß sind irgendwelche [___] während oder nach… |
| 927 | Löschung | `während` | `*(nicht da)*` | …sind irgendwelche komplikationen [___] oder nach der… |
| 928 | Löschung | `oder` | `*(nicht da)*` | …irgendwelche komplikationen während [___] nach der operation… |
| 929 | Löschung | `nach` | `*(nicht da)*` | …komplikationen während oder [___] der operation aufgetreten… |
| 930 | Löschung | `der` | `*(nicht da)*` | …während oder nach [___] operation aufgetreten nein… |
| 931 | Löschung | `operation` | `*(nicht da)*` | …oder nach der [___] aufgetreten nein zum… |
| 932 | Löschung | `aufgetreten` | `*(nicht da)*` | …nach der operation [___] nein zum glück… |
| 933 | Löschung | `nein` | `*(nicht da)*` | …der operation aufgetreten [___] zum glück nicht… |
| 934 | Löschung | `zum` | `*(nicht da)*` | …operation aufgetreten nein [___] glück nicht nein… |
| 935 | Löschung | `glück` | `*(nicht da)*` | …aufgetreten nein zum [___] nicht nein sehr… |
| 936 | Löschung | `nicht` | `*(nicht da)*` | …nein zum glück [___] nein sehr gut… |
| 937 | Löschung | `nein` | `*(nicht da)*` | …zum glück nicht [___] sehr gut ich… |
| 938 | Löschung | `sehr` | `*(nicht da)*` | …glück nicht nein [___] gut ich konnte… |
| 939 | Löschung | `gut` | `*(nicht da)*` | …nicht nein sehr [___] ich konnte ganz… |
| 940 | Löschung | `ich` | `*(nicht da)*` | …nein sehr gut [___] konnte ganz bald… |
| 941 | Löschung | `konnte` | `*(nicht da)*` | …sehr gut ich [___] ganz bald wieder… |
| 942 | Löschung | `ganz` | `*(nicht da)*` | …gut ich konnte [___] bald wieder meine… |
| 943 | Löschung | `bald` | `*(nicht da)*` | …ich konnte ganz [___] wieder meine hohen… |
| 944 | Löschung | `wieder` | `*(nicht da)*` | …konnte ganz bald [___] meine hohen schuhe… |
| 945 | Löschung | `meine` | `*(nicht da)*` | …ganz bald wieder [___] hohen schuhe tragen… |
| 946 | Löschung | `hohen` | `*(nicht da)*` | …bald wieder meine [___] schuhe tragen perfekt… |
| 947 | Löschung | `schuhe` | `*(nicht da)*` | …wieder meine hohen [___] tragen perfekt dann… |
| 948 | Löschung | `tragen` | `*(nicht da)*` | …meine hohen schuhe [___] perfekt dann ist… |
| 949 | Löschung | `perfekt` | `*(nicht da)*` | …hohen schuhe tragen [___] dann ist wirklich… |
| 950 | Löschung | `dann` | `*(nicht da)*` | …schuhe tragen perfekt [___] ist wirklich alles… |
| 951 | Löschung | `ist` | `*(nicht da)*` | …tragen perfekt dann [___] wirklich alles gut… |
| 952 | Löschung | `wirklich` | `*(nicht da)*` | …perfekt dann ist [___] alles gut gelaufen… |
| 953 | Löschung | `alles` | `*(nicht da)*` | …dann ist wirklich [___] gut gelaufen frau… |
| 954 | Löschung | `gut` | `*(nicht da)*` | …ist wirklich alles [___] gelaufen frau beckenwestfalen… |
| 955 | Löschung | `gelaufen` | `*(nicht da)*` | …wirklich alles gut [___] frau beckenwestfalen nehmen… |
| 956 | Löschung | `frau` | `*(nicht da)*` | …alles gut gelaufen [___] beckenwestfalen nehmen sie… |
| 957 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …gut gelaufen frau [___] nehmen sie regelmäßig… |
| 958 | Löschung | `nehmen` | `*(nicht da)*` | …gelaufen frau beckenwestfalen [___] sie regelmäßig oder… |
| 959 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen nehmen [___] regelmäßig oder bei… |
| 960 | Löschung | `regelmäßig` | `*(nicht da)*` | …beckenwestfalen nehmen sie [___] oder bei bedarf… |
| 961 | Löschung | `oder` | `*(nicht da)*` | …nehmen sie regelmäßig [___] bei bedarf medikamente… |
| 962 | Löschung | `bei` | `*(nicht da)*` | …sie regelmäßig oder [___] bedarf medikamente ein… |
| 963 | Löschung | `bedarf` | `*(nicht da)*` | …regelmäßig oder bei [___] medikamente ein ich… |
| 964 | Löschung | `medikamente` | `*(nicht da)*` | …oder bei bedarf [___] ein ich nehme… |
| 965 | Löschung | `ein` | `*(nicht da)*` | …bei bedarf medikamente [___] ich nehme gelegentlich… |
| 966 | Löschung | `ich` | `*(nicht da)*` | …bedarf medikamente ein [___] nehme gelegentlich ein… |
| 967 | Löschung | `nehme` | `*(nicht da)*` | …medikamente ein ich [___] gelegentlich ein ibuprofen… |
| 968 | Löschung | `gelegentlich` | `*(nicht da)*` | …ein ich nehme [___] ein ibuprofen wenn… |
| 969 | Löschung | `ein` | `*(nicht da)*` | …ich nehme gelegentlich [___] ibuprofen wenn ich… |
| 970 | Löschung | `ibuprofen` | `*(nicht da)*` | …nehme gelegentlich ein [___] wenn ich kopfschmerzen… |
| 971 | Löschung | `wenn` | `*(nicht da)*` | …gelegentlich ein ibuprofen [___] ich kopfschmerzen habe… |
| 972 | Löschung | `ich` | `*(nicht da)*` | …ein ibuprofen wenn [___] kopfschmerzen habe und… |
| 973 | Löschung | `kopfschmerzen` | `*(nicht da)*` | …ibuprofen wenn ich [___] habe und ansonsten… |
| 974 | Löschung | `habe` | `*(nicht da)*` | …wenn ich kopfschmerzen [___] und ansonsten nehme… |
| 975 | Löschung | `und` | `*(nicht da)*` | …ich kopfschmerzen habe [___] ansonsten nehme ich… |
| 976 | Löschung | `ansonsten` | `*(nicht da)*` | …kopfschmerzen habe und [___] nehme ich die… |
| 977 | Löschung | `nehme` | `*(nicht da)*` | …habe und ansonsten [___] ich die pille… |
| 978 | Löschung | `ich` | `*(nicht da)*` | …und ansonsten nehme [___] die pille die… |
| 979 | Löschung | `die` | `*(nicht da)*` | …ansonsten nehme ich [___] pille die pille… |
| 980 | Löschung | `pille` | `*(nicht da)*` | …nehme ich die [___] die pille seit… |
| 981 | Löschung | `die` | `*(nicht da)*` | …ich die pille [___] pille seit wann… |
| 982 | Löschung | `pille` | `*(nicht da)*` | …die pille die [___] seit wann nehmen… |
| 983 | Löschung | `seit` | `*(nicht da)*` | …pille die pille [___] wann nehmen sie… |
| 984 | Löschung | `wann` | `*(nicht da)*` | …die pille seit [___] nehmen sie die… |
| 985 | Löschung | `nehmen` | `*(nicht da)*` | …pille seit wann [___] sie die pille… |
| 986 | Löschung | `sie` | `*(nicht da)*` | …seit wann nehmen [___] die pille sieben… |
| 987 | Löschung | `die` | `*(nicht da)*` | …wann nehmen sie [___] pille sieben oder… |
| 988 | Löschung | `pille` | `*(nicht da)*` | …nehmen sie die [___] sieben oder acht… |
| 989 | Löschung | `sieben` | `*(nicht da)*` | …sie die pille [___] oder acht jahren… |
| 990 | Substitution | `oder` | `vor` | …die pille sieben [___] acht jahren okay… |
| 991 | Substitution | `acht` | `zwei` | …pille sieben oder [___] jahren okay die… |
| 992 | Löschung | `okay` | `*(nicht da)*` | …oder acht jahren [___] die ibuprofen wenn… |
| 993 | Löschung | `die` | `*(nicht da)*` | …acht jahren okay [___] ibuprofen wenn sie… |
| 994 | Löschung | `ibuprofen` | `*(nicht da)*` | …jahren okay die [___] wenn sie kopfschmerzen… |
| 995 | Löschung | `wenn` | `*(nicht da)*` | …okay die ibuprofen [___] sie kopfschmerzen haben… |
| 996 | Löschung | `sie` | `*(nicht da)*` | …die ibuprofen wenn [___] kopfschmerzen haben wie… |
| 997 | Löschung | `kopfschmerzen` | `*(nicht da)*` | …ibuprofen wenn sie [___] haben wie viele… |
| 998 | Löschung | `haben` | `*(nicht da)*` | …wenn sie kopfschmerzen [___] wie viele milligramm… |
| 999 | Löschung | `wie` | `*(nicht da)*` | …sie kopfschmerzen haben [___] viele milligramm sind… |
| 1000 | Löschung | `viele` | `*(nicht da)*` | …kopfschmerzen haben wie [___] milligramm sind das… |
| 1001 | Löschung | `milligramm` | `*(nicht da)*` | …haben wie viele [___] sind das 400… |
| 1002 | Löschung | `sind` | `*(nicht da)*` | …wie viele milligramm [___] das 400 600… |
| 1003 | Löschung | `das` | `*(nicht da)*` | …viele milligramm sind [___] 400 600 800… |
| 1004 | Löschung | `400` | `*(nicht da)*` | …milligramm sind das [___] 600 800 also… |
| 1005 | Löschung | `600` | `*(nicht da)*` | …sind das 400 [___] 800 also meistens… |
| 1006 | Löschung | `800` | `*(nicht da)*` | …das 400 600 [___] also meistens das… |
| 1007 | Löschung | `also` | `*(nicht da)*` | …400 600 800 [___] meistens das was… |
| 1008 | Löschung | `meistens` | `*(nicht da)*` | …600 800 also [___] das was ich… |
| 1009 | Löschung | `das` | `*(nicht da)*` | …800 also meistens [___] was ich gerade… |
| 1010 | Löschung | `was` | `*(nicht da)*` | …also meistens das [___] ich gerade zu… |
| 1011 | Löschung | `ich` | `*(nicht da)*` | …meistens das was [___] gerade zu hause… |
| 1012 | Löschung | `gerade` | `*(nicht da)*` | …das was ich [___] zu hause habe… |
| 1013 | Löschung | `zu` | `*(nicht da)*` | …was ich gerade [___] hause habe aber… |
| 1014 | Löschung | `hause` | `*(nicht da)*` | …ich gerade zu [___] habe aber ich… |
| 1015 | Löschung | `habe` | `*(nicht da)*` | …gerade zu hause [___] aber ich glaube… |
| 1016 | Löschung | `aber` | `*(nicht da)*` | …zu hause habe [___] ich glaube 600… |
| 1017 | Löschung | `ich` | `*(nicht da)*` | …hause habe aber [___] glaube 600 600… |
| 1018 | Löschung | `glaube` | `*(nicht da)*` | …habe aber ich [___] 600 600 ja… |
| 1019 | Löschung | `600` | `*(nicht da)*` | …aber ich glaube [___] 600 ja alles… |
| 1020 | Löschung | `600` | `*(nicht da)*` | …ich glaube 600 [___] ja alles klar… |
| 1021 | Löschung | `ja` | `*(nicht da)*` | …glaube 600 600 [___] alles klar sind… |
| 1022 | Löschung | `alles` | `*(nicht da)*` | …600 600 ja [___] klar sind sie… |
| 1023 | Löschung | `klar` | `*(nicht da)*` | …600 ja alles [___] sind sie geimpft… |
| 1024 | Löschung | `sind` | `*(nicht da)*` | …ja alles klar [___] sie geimpft ich… |
| 1025 | Löschung | `sie` | `*(nicht da)*` | …alles klar sind [___] geimpft ich bin… |
| 1026 | Löschung | `geimpft` | `*(nicht da)*` | …klar sind sie [___] ich bin geimpft… |
| 1027 | Löschung | `ich` | `*(nicht da)*` | …sind sie geimpft [___] bin geimpft ja… |
| 1028 | Löschung | `bin` | `*(nicht da)*` | …sie geimpft ich [___] geimpft ja haben… |
| 1029 | Löschung | `geimpft` | `*(nicht da)*` | …geimpft ich bin [___] ja haben sie… |
| 1030 | Löschung | `ja` | `*(nicht da)*` | …ich bin geimpft [___] haben sie ganz… |
| 1031 | Löschung | `haben` | `*(nicht da)*` | …bin geimpft ja [___] sie ganz zufällig… |
| 1032 | Löschung | `sie` | `*(nicht da)*` | …geimpft ja haben [___] ganz zufällig ihren… |
| 1033 | Löschung | `ganz` | `*(nicht da)*` | …ja haben sie [___] zufällig ihren impfpass… |
| 1034 | Löschung | `zufällig` | `*(nicht da)*` | …haben sie ganz [___] ihren impfpass dabei… |
| 1035 | Löschung | `ihren` | `*(nicht da)*` | …sie ganz zufällig [___] impfpass dabei oh… |
| 1036 | Löschung | `impfpass` | `*(nicht da)*` | …ganz zufällig ihren [___] dabei oh leider… |
| 1037 | Löschung | `dabei` | `*(nicht da)*` | …zufällig ihren impfpass [___] oh leider nein… |
| 1038 | Löschung | `oh` | `*(nicht da)*` | …ihren impfpass dabei [___] leider nein eher… |
| 1039 | Löschung | `leider` | `*(nicht da)*` | …impfpass dabei oh [___] nein eher nicht… |
| 1040 | Löschung | `nein` | `*(nicht da)*` | …dabei oh leider [___] eher nicht hätte… |
| 1041 | Löschung | `eher` | `*(nicht da)*` | …oh leider nein [___] nicht hätte ich… |
| 1042 | Löschung | `nicht` | `*(nicht da)*` | …leider nein eher [___] hätte ich gewusst… |
| 1043 | Löschung | `hätte` | `*(nicht da)*` | …nein eher nicht [___] ich gewusst dass… |
| 1044 | Löschung | `ich` | `*(nicht da)*` | …eher nicht hätte [___] gewusst dass ich… |
| 1045 | Löschung | `gewusst` | `*(nicht da)*` | …nicht hätte ich [___] dass ich ins… |
| 1046 | Löschung | `dass` | `*(nicht da)*` | …hätte ich gewusst [___] ich ins krankenhaus… |
| 1047 | Löschung | `ich` | `*(nicht da)*` | …ich gewusst dass [___] ins krankenhaus muss… |
| 1048 | Löschung | `ins` | `*(nicht da)*` | …gewusst dass ich [___] krankenhaus muss ja… |
| 1049 | Löschung | `krankenhaus` | `*(nicht da)*` | …dass ich ins [___] muss ja ich… |
| 1050 | Löschung | `muss` | `*(nicht da)*` | …ich ins krankenhaus [___] ja ich muss… |
| 1051 | Löschung | `ja` | `*(nicht da)*` | …ins krankenhaus muss [___] ich muss auch… |
| 1052 | Löschung | `ich` | `*(nicht da)*` | …krankenhaus muss ja [___] muss auch wissen… |
| 1053 | Löschung | `muss` | `*(nicht da)*` | …muss ja ich [___] auch wissen dass… |
| 1054 | Löschung | `auch` | `*(nicht da)*` | …ja ich muss [___] wissen dass ich… |
| 1055 | Löschung | `wissen` | `*(nicht da)*` | …ich muss auch [___] dass ich den… |
| 1056 | Löschung | `dass` | `*(nicht da)*` | …muss auch wissen [___] ich den nicht… |
| 1057 | Löschung | `ich` | `*(nicht da)*` | …auch wissen dass [___] den nicht bei… |
| 1058 | Löschung | `den` | `*(nicht da)*` | …wissen dass ich [___] nicht bei mir… |
| 1059 | Löschung | `nicht` | `*(nicht da)*` | …dass ich den [___] bei mir trage… |
| 1060 | Löschung | `mir` | `*(nicht da)*` | …den nicht bei [___] trage sehr gut… |
| 1061 | Löschung | `trage` | `*(nicht da)*` | …nicht bei mir [___] sehr gut okay… |
| 1062 | Löschung | `sehr` | `*(nicht da)*` | …bei mir trage [___] gut okay frau… |
| 1063 | Löschung | `gut` | `*(nicht da)*` | …mir trage sehr [___] okay frau beckenwestfalen… |
| 1064 | Löschung | `okay` | `*(nicht da)*` | …trage sehr gut [___] frau beckenwestfalen wie… |
| 1065 | Löschung | `frau` | `*(nicht da)*` | …sehr gut okay [___] beckenwestfalen wie geht… |
| 1066 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …gut okay frau [___] wie geht es… |
| 1067 | Löschung | `wie` | `*(nicht da)*` | …okay frau beckenwestfalen [___] geht es ihnen… |
| 1068 | Löschung | `geht` | `*(nicht da)*` | …frau beckenwestfalen wie [___] es ihnen denn… |
| 1069 | Löschung | `es` | `*(nicht da)*` | …beckenwestfalen wie geht [___] ihnen denn sonst… |
| 1070 | Löschung | `ihnen` | `*(nicht da)*` | …wie geht es [___] denn sonst körperlich… |
| 1071 | Löschung | `denn` | `*(nicht da)*` | …geht es ihnen [___] sonst körperlich haben… |
| 1072 | Löschung | `sonst` | `*(nicht da)*` | …es ihnen denn [___] körperlich haben sie… |
| 1073 | Löschung | `körperlich` | `*(nicht da)*` | …ihnen denn sonst [___] haben sie in… |
| 1074 | Löschung | `haben` | `*(nicht da)*` | …denn sonst körperlich [___] sie in letzter… |
| 1075 | Löschung | `sie` | `*(nicht da)*` | …sonst körperlich haben [___] in letzter zeit… |
| 1076 | Löschung | `in` | `*(nicht da)*` | …körperlich haben sie [___] letzter zeit fieber… |
| 1077 | Löschung | `letzter` | `*(nicht da)*` | …haben sie in [___] zeit fieber gehabt… |
| 1078 | Löschung | `zeit` | `*(nicht da)*` | …sie in letzter [___] fieber gehabt oder… |
| 1079 | Löschung | `fieber` | `*(nicht da)*` | …in letzter zeit [___] gehabt oder schüttelfrost… |
| 1080 | Löschung | `gehabt` | `*(nicht da)*` | …letzter zeit fieber [___] oder schüttelfrost oder… |
| 1081 | Löschung | `oder` | `*(nicht da)*` | …zeit fieber gehabt [___] schüttelfrost oder nachtschweiß… |
| 1082 | Löschung | `schüttelfrost` | `*(nicht da)*` | …fieber gehabt oder [___] oder nachtschweiß oder… |
| 1083 | Löschung | `oder` | `*(nicht da)*` | …gehabt oder schüttelfrost [___] nachtschweiß oder fühlen… |
| 1084 | Löschung | `nachtschweiß` | `*(nicht da)*` | …oder schüttelfrost oder [___] oder fühlen sie… |
| 1085 | Löschung | `oder` | `*(nicht da)*` | …schüttelfrost oder nachtschweiß [___] fühlen sie sich… |
| 1086 | Löschung | `fühlen` | `*(nicht da)*` | …oder nachtschweiß oder [___] sie sich irgendwie… |
| 1087 | Löschung | `sie` | `*(nicht da)*` | …nachtschweiß oder fühlen [___] sich irgendwie ungut… |
| 1088 | Löschung | `sich` | `*(nicht da)*` | …oder fühlen sie [___] irgendwie ungut in… |
| 1089 | Löschung | `irgendwie` | `*(nicht da)*` | …fühlen sie sich [___] ungut in letzter… |
| 1090 | Löschung | `ungut` | `*(nicht da)*` | …sie sich irgendwie [___] in letzter zeit… |
| 1091 | Löschung | `in` | `*(nicht da)*` | …sich irgendwie ungut [___] letzter zeit nein… |
| 1092 | Löschung | `letzter` | `*(nicht da)*` | …irgendwie ungut in [___] zeit nein ich… |
| 1093 | Löschung | `zeit` | `*(nicht da)*` | …ungut in letzter [___] nein ich habe… |
| 1094 | Löschung | `nein` | `*(nicht da)*` | …in letzter zeit [___] ich habe gar… |
| 1095 | Löschung | `ich` | `*(nicht da)*` | …letzter zeit nein [___] habe gar keine… |
| 1096 | Löschung | `habe` | `*(nicht da)*` | …zeit nein ich [___] gar keine sonstigen… |
| 1097 | Löschung | `gar` | `*(nicht da)*` | …nein ich habe [___] keine sonstigen gesundheitlichen… |
| 1098 | Löschung | `keine` | `*(nicht da)*` | …ich habe gar [___] sonstigen gesundheitlichen probleme… |
| 1099 | Löschung | `sonstigen` | `*(nicht da)*` | …habe gar keine [___] gesundheitlichen probleme ich… |
| 1100 | Löschung | `gesundheitlichen` | `*(nicht da)*` | …gar keine sonstigen [___] probleme ich habe… |
| 1101 | Löschung | `probleme` | `*(nicht da)*` | …keine sonstigen gesundheitlichen [___] ich habe manchmal… |
| 1102 | Löschung | `ich` | `*(nicht da)*` | …sonstigen gesundheitlichen probleme [___] habe manchmal schwierigkeiten… |
| 1103 | Löschung | `habe` | `*(nicht da)*` | …gesundheitlichen probleme ich [___] manchmal schwierigkeiten beim… |
| 1104 | Löschung | `manchmal` | `*(nicht da)*` | …probleme ich habe [___] schwierigkeiten beim einschlafen… |
| 1105 | Löschung | `schwierigkeiten` | `*(nicht da)*` | …ich habe manchmal [___] beim einschlafen aber… |
| 1106 | Löschung | `beim` | `*(nicht da)*` | …habe manchmal schwierigkeiten [___] einschlafen aber das… |
| 1107 | Löschung | `einschlafen` | `*(nicht da)*` | …manchmal schwierigkeiten beim [___] aber das ist… |
| 1108 | Löschung | `aber` | `*(nicht da)*` | …schwierigkeiten beim einschlafen [___] das ist oft… |
| 1109 | Löschung | `das` | `*(nicht da)*` | …beim einschlafen aber [___] ist oft der… |
| 1110 | Löschung | `ist` | `*(nicht da)*` | …einschlafen aber das [___] oft der fall… |
| 1111 | Löschung | `oft` | `*(nicht da)*` | …aber das ist [___] der fall wenn… |
| 1112 | Löschung | `der` | `*(nicht da)*` | …das ist oft [___] fall wenn ich… |
| 1113 | Löschung | `fall` | `*(nicht da)*` | …ist oft der [___] wenn ich auf… |
| 1114 | Löschung | `wenn` | `*(nicht da)*` | …oft der fall [___] ich auf der… |
| 1115 | Löschung | `ich` | `*(nicht da)*` | …der fall wenn [___] auf der arbeit… |
| 1116 | Löschung | `auf` | `*(nicht da)*` | …fall wenn ich [___] der arbeit viel… |
| 1117 | Löschung | `der` | `*(nicht da)*` | …wenn ich auf [___] arbeit viel zu… |
| 1118 | Löschung | `arbeit` | `*(nicht da)*` | …ich auf der [___] viel zu tun… |
| 1119 | Löschung | `viel` | `*(nicht da)*` | …auf der arbeit [___] zu tun habe… |
| 1120 | Löschung | `zu` | `*(nicht da)*` | …der arbeit viel [___] tun habe oder… |
| 1121 | Löschung | `tun` | `*(nicht da)*` | …arbeit viel zu [___] habe oder zu… |
| 1122 | Löschung | `habe` | `*(nicht da)*` | …viel zu tun [___] oder zu viel… |
| 1123 | Löschung | `oder` | `*(nicht da)*` | …zu tun habe [___] zu viel nachdenke… |
| 1124 | Löschung | `zu` | `*(nicht da)*` | …tun habe oder [___] viel nachdenke also… |
| 1125 | Löschung | `viel` | `*(nicht da)*` | …habe oder zu [___] nachdenke also nichts… |
| 1126 | Löschung | `nachdenke` | `*(nicht da)*` | …oder zu viel [___] also nichts worüber… |
| 1127 | Löschung | `also` | `*(nicht da)*` | …zu viel nachdenke [___] nichts worüber ich… |
| 1128 | Löschung | `nichts` | `*(nicht da)*` | …viel nachdenke also [___] worüber ich mir… |
| 1129 | Löschung | `worüber` | `*(nicht da)*` | …nachdenke also nichts [___] ich mir bis… |
| 1130 | Löschung | `ich` | `*(nicht da)*` | …also nichts worüber [___] mir bis jetzt… |
| 1131 | Löschung | `mir` | `*(nicht da)*` | …nichts worüber ich [___] bis jetzt sorgen… |
| 1132 | Löschung | `bis` | `*(nicht da)*` | …worüber ich mir [___] jetzt sorgen gemacht… |
| 1133 | Löschung | `jetzt` | `*(nicht da)*` | …ich mir bis [___] sorgen gemacht habe… |
| 1134 | Löschung | `sorgen` | `*(nicht da)*` | …mir bis jetzt [___] gemacht habe okay… |
| 1135 | Löschung | `gemacht` | `*(nicht da)*` | …bis jetzt sorgen [___] habe okay prima… |
| 1136 | Löschung | `habe` | `*(nicht da)*` | …jetzt sorgen gemacht [___] okay prima ich… |
| 1137 | Löschung | `okay` | `*(nicht da)*` | …sorgen gemacht habe [___] prima ich glaube… |
| 1138 | Löschung | `prima` | `*(nicht da)*` | …gemacht habe okay [___] ich glaube das… |
| 1139 | Löschung | `ich` | `*(nicht da)*` | …habe okay prima [___] glaube das kennen… |
| 1140 | Löschung | `glaube` | `*(nicht da)*` | …okay prima ich [___] das kennen wir… |
| 1141 | Löschung | `das` | `*(nicht da)*` | …prima ich glaube [___] kennen wir auch… |
| 1142 | Löschung | `kennen` | `*(nicht da)*` | …ich glaube das [___] wir auch wirklich… |
| 1143 | Löschung | `wir` | `*(nicht da)*` | …glaube das kennen [___] auch wirklich alle… |
| 1144 | Löschung | `auch` | `*(nicht da)*` | …das kennen wir [___] wirklich alle ja… |
| 1145 | Löschung | `wirklich` | `*(nicht da)*` | …kennen wir auch [___] alle ja wie… |
| 1146 | Löschung | `alle` | `*(nicht da)*` | …wir auch wirklich [___] ja wie sieht… |
| 1147 | Löschung | `ja` | `*(nicht da)*` | …auch wirklich alle [___] wie sieht es… |
| 1148 | Löschung | `wie` | `*(nicht da)*` | …wirklich alle ja [___] sieht es denn… |
| 1149 | Löschung | `sieht` | `*(nicht da)*` | …alle ja wie [___] es denn aus… |
| 1150 | Löschung | `es` | `*(nicht da)*` | …ja wie sieht [___] denn aus mit… |
| 1151 | Löschung | `denn` | `*(nicht da)*` | …wie sieht es [___] aus mit ihrer… |
| 1152 | Löschung | `aus` | `*(nicht da)*` | …sieht es denn [___] mit ihrer periode… |
| 1153 | Löschung | `mit` | `*(nicht da)*` | …es denn aus [___] ihrer periode bekommen… |
| 1154 | Löschung | `ihrer` | `*(nicht da)*` | …denn aus mit [___] periode bekommen sie… |
| 1155 | Löschung | `periode` | `*(nicht da)*` | …aus mit ihrer [___] bekommen sie die… |
| 1156 | Löschung | `bekommen` | `*(nicht da)*` | …mit ihrer periode [___] sie die regelmäßig… |
| 1157 | Löschung | `sie` | `*(nicht da)*` | …ihrer periode bekommen [___] die regelmäßig ich… |
| 1158 | Löschung | `die` | `*(nicht da)*` | …periode bekommen sie [___] regelmäßig ich bekomme… |
| 1159 | Löschung | `regelmäßig` | `*(nicht da)*` | …bekommen sie die [___] ich bekomme sie… |
| 1160 | Löschung | `ich` | `*(nicht da)*` | …sie die regelmäßig [___] bekomme sie regelmäßig… |
| 1161 | Löschung | `bekomme` | `*(nicht da)*` | …die regelmäßig ich [___] sie regelmäßig ja… |
| 1162 | Löschung | `sie` | `*(nicht da)*` | …regelmäßig ich bekomme [___] regelmäßig ja seitdem… |
| 1163 | Löschung | `regelmäßig` | `*(nicht da)*` | …ich bekomme sie [___] ja seitdem ich… |
| 1164 | Löschung | `ja` | `*(nicht da)*` | …bekomme sie regelmäßig [___] seitdem ich die… |
| 1165 | Löschung | `seitdem` | `*(nicht da)*` | …sie regelmäßig ja [___] ich die pille… |
| 1166 | Löschung | `ich` | `*(nicht da)*` | …regelmäßig ja seitdem [___] die pille nehme… |
| 1167 | Löschung | `die` | `*(nicht da)*` | …ja seitdem ich [___] pille nehme bekomme… |
| 1168 | Löschung | `pille` | `*(nicht da)*` | …seitdem ich die [___] nehme bekomme ich… |
| 1169 | Löschung | `nehme` | `*(nicht da)*` | …ich die pille [___] bekomme ich sie… |
| 1170 | Löschung | `bekomme` | `*(nicht da)*` | …die pille nehme [___] ich sie ganz… |
| 1171 | Löschung | `ich` | `*(nicht da)*` | …pille nehme bekomme [___] sie ganz regelmäßig… |
| 1172 | Löschung | `sie` | `*(nicht da)*` | …nehme bekomme ich [___] ganz regelmäßig okay… |
| 1173 | Löschung | `ganz` | `*(nicht da)*` | …bekomme ich sie [___] regelmäßig okay wunderbar… |
| 1174 | Löschung | `regelmäßig` | `*(nicht da)*` | …ich sie ganz [___] okay wunderbar frau… |
| 1175 | Löschung | `okay` | `*(nicht da)*` | …sie ganz regelmäßig [___] wunderbar frau beckenwestfalen… |
| 1176 | Löschung | `wunderbar` | `*(nicht da)*` | …ganz regelmäßig okay [___] frau beckenwestfalen rauchen… |
| 1177 | Löschung | `frau` | `*(nicht da)*` | …regelmäßig okay wunderbar [___] beckenwestfalen rauchen sie… |
| 1178 | Löschung | `beckenwestfalen` | `*(nicht da)*` | …okay wunderbar frau [___] rauchen sie nein… |
| 1179 | Löschung | `rauchen` | `*(nicht da)*` | …wunderbar frau beckenwestfalen [___] sie nein ich… |
| 1180 | Löschung | `sie` | `*(nicht da)*` | …frau beckenwestfalen rauchen [___] nein ich habe… |
| 1181 | Löschung | `nein` | `*(nicht da)*` | …beckenwestfalen rauchen sie [___] ich habe früher… |
| 1182 | Löschung | `ich` | `*(nicht da)*` | …rauchen sie nein [___] habe früher geraucht… |
| 1183 | Löschung | `habe` | `*(nicht da)*` | …sie nein ich [___] früher geraucht falls… |
| 1184 | Löschung | `früher` | `*(nicht da)*` | …nein ich habe [___] geraucht falls das… |
| 1185 | Löschung | `geraucht` | `*(nicht da)*` | …ich habe früher [___] falls das relevant… |
| 1186 | Löschung | `falls` | `*(nicht da)*` | …habe früher geraucht [___] das relevant ist… |
| 1187 | Löschung | `das` | `*(nicht da)*` | …früher geraucht falls [___] relevant ist ja… |
| 1188 | Löschung | `relevant` | `*(nicht da)*` | …geraucht falls das [___] ist ja wie… |
| 1189 | Löschung | `ist` | `*(nicht da)*` | …falls das relevant [___] ja wie lange… |
| 1190 | Löschung | `ja` | `*(nicht da)*` | …das relevant ist [___] wie lange haben… |
| 1191 | Löschung | `wie` | `*(nicht da)*` | …relevant ist ja [___] lange haben sie… |
| 1192 | Löschung | `lange` | `*(nicht da)*` | …ist ja wie [___] haben sie aufgehört… |
| 1193 | Löschung | `haben` | `*(nicht da)*` | …ja wie lange [___] sie aufgehört ach… |
| 1194 | Löschung | `sie` | `*(nicht da)*` | …wie lange haben [___] aufgehört ach das… |
| 1195 | Löschung | `aufgehört` | `*(nicht da)*` | …lange haben sie [___] ach das müssten… |
| 1196 | Löschung | `ach` | `*(nicht da)*` | …haben sie aufgehört [___] das müssten jetzt… |
| 1197 | Löschung | `das` | `*(nicht da)*` | …sie aufgehört ach [___] müssten jetzt schon… |
| 1198 | Löschung | `müssten` | `*(nicht da)*` | …aufgehört ach das [___] jetzt schon acht… |
| 1199 | Löschung | `jetzt` | `*(nicht da)*` | …ach das müssten [___] schon acht jahre… |
| 1200 | Löschung | `schon` | `*(nicht da)*` | …das müssten jetzt [___] acht jahre sein… |
| 1201 | Löschung | `acht` | `*(nicht da)*` | …müssten jetzt schon [___] jahre sein seitdem… |
| 1202 | Löschung | `jahre` | `*(nicht da)*` | …jetzt schon acht [___] sein seitdem ich… |
| 1203 | Löschung | `sein` | `*(nicht da)*` | …schon acht jahre [___] seitdem ich aufgehört… |
| 1204 | Löschung | `seitdem` | `*(nicht da)*` | …acht jahre sein [___] ich aufgehört habe… |
| 1205 | Löschung | `ich` | `*(nicht da)*` | …jahre sein seitdem [___] aufgehört habe zum… |
| 1206 | Löschung | `aufgehört` | `*(nicht da)*` | …sein seitdem ich [___] habe zum glück… |
| 1207 | Löschung | `habe` | `*(nicht da)*` | …seitdem ich aufgehört [___] zum glück und… |
| 1208 | Löschung | `zum` | `*(nicht da)*` | …ich aufgehört habe [___] glück und wie… |
| 1209 | Löschung | `glück` | `*(nicht da)*` | …aufgehört habe zum [___] und wie lange… |
| 1210 | Löschung | `und` | `*(nicht da)*` | …habe zum glück [___] wie lange haben… |
| 1211 | Löschung | `wie` | `*(nicht da)*` | …zum glück und [___] lange haben sie… |
| 1212 | Löschung | `lange` | `*(nicht da)*` | …glück und wie [___] haben sie geraucht… |
| 1213 | Löschung | `haben` | `*(nicht da)*` | …und wie lange [___] sie geraucht damals… |
| 1214 | Löschung | `sie` | `*(nicht da)*` | …wie lange haben [___] geraucht damals sechs… |
| 1215 | Löschung | `geraucht` | `*(nicht da)*` | …lange haben sie [___] damals sechs sieben… |
| 1216 | Löschung | `damals` | `*(nicht da)*` | …haben sie geraucht [___] sechs sieben jahre… |
| 1217 | Löschung | `sechs` | `*(nicht da)*` | …sie geraucht damals [___] sieben jahre sechs… |
| 1218 | Löschung | `sieben` | `*(nicht da)*` | …geraucht damals sechs [___] jahre sechs sieben… |
| 1219 | Löschung | `jahre` | `*(nicht da)*` | …damals sechs sieben [___] sechs sieben jahre… |
| 1220 | Löschung | `sechs` | `*(nicht da)*` | …sechs sieben jahre [___] sieben jahre okay… |
| 1221 | Löschung | `sieben` | `*(nicht da)*` | …sieben jahre sechs [___] jahre okay gut… |
| 1222 | Löschung | `jahre` | `*(nicht da)*` | …jahre sechs sieben [___] okay gut trinken… |
| 1223 | Löschung | `okay` | `*(nicht da)*` | …sechs sieben jahre [___] gut trinken sie… |
| 1224 | Löschung | `gut` | `*(nicht da)*` | …sieben jahre okay [___] trinken sie alkohol… |
| 1225 | Löschung | `trinken` | `*(nicht da)*` | …jahre okay gut [___] sie alkohol ja… |
| 1226 | Löschung | `sie` | `*(nicht da)*` | …okay gut trinken [___] alkohol ja nicht… |
| 1227 | Löschung | `alkohol` | `*(nicht da)*` | …gut trinken sie [___] ja nicht viel… |
| 1228 | Löschung | `ja` | `*(nicht da)*` | …trinken sie alkohol [___] nicht viel aber… |
| 1229 | Löschung | `nicht` | `*(nicht da)*` | …sie alkohol ja [___] viel aber schon… |
| 1230 | Löschung | `viel` | `*(nicht da)*` | …alkohol ja nicht [___] aber schon abends… |
| 1231 | Löschung | `aber` | `*(nicht da)*` | …ja nicht viel [___] schon abends nach… |
| 1232 | Löschung | `schon` | `*(nicht da)*` | …nicht viel aber [___] abends nach der… |
| 1233 | Löschung | `abends` | `*(nicht da)*` | …viel aber schon [___] nach der arbeit… |
| 1234 | Löschung | `nach` | `*(nicht da)*` | …aber schon abends [___] der arbeit gerne… |
| 1235 | Löschung | `der` | `*(nicht da)*` | …schon abends nach [___] arbeit gerne ein… |
| 1236 | Löschung | `arbeit` | `*(nicht da)*` | …abends nach der [___] gerne ein glas… |
| 1237 | Löschung | `gerne` | `*(nicht da)*` | …nach der arbeit [___] ein glas wein… |
| 1238 | Löschung | `ein` | `*(nicht da)*` | …der arbeit gerne [___] glas wein und… |
| 1239 | Löschung | `glas` | `*(nicht da)*` | …arbeit gerne ein [___] wein und am… |
| 1240 | Löschung | `wein` | `*(nicht da)*` | …gerne ein glas [___] und am wochenende… |
| 1241 | Löschung | `und` | `*(nicht da)*` | …ein glas wein [___] am wochenende wenn… |
| 1242 | Löschung | `am` | `*(nicht da)*` | …glas wein und [___] wochenende wenn wir… |
| 1243 | Löschung | `wochenende` | `*(nicht da)*` | …wein und am [___] wenn wir mit… |
| 1244 | Löschung | `wenn` | `*(nicht da)*` | …und am wochenende [___] wir mit freunden… |
| 1245 | Löschung | `wir` | `*(nicht da)*` | …am wochenende wenn [___] mit freunden unterwegs… |
| 1246 | Löschung | `mit` | `*(nicht da)*` | …wochenende wenn wir [___] freunden unterwegs sind… |
| 1247 | Löschung | `freunden` | `*(nicht da)*` | …wenn wir mit [___] unterwegs sind dann… |
| 1248 | Löschung | `unterwegs` | `*(nicht da)*` | …wir mit freunden [___] sind dann gerne… |
| 1249 | Löschung | `sind` | `*(nicht da)*` | …mit freunden unterwegs [___] dann gerne auch… |
| 1250 | Löschung | `dann` | `*(nicht da)*` | …freunden unterwegs sind [___] gerne auch zwei… |
| 1251 | Löschung | `gerne` | `*(nicht da)*` | …unterwegs sind dann [___] auch zwei oder… |
| 1252 | Löschung | `auch` | `*(nicht da)*` | …sind dann gerne [___] zwei oder drei… |
| 1253 | Löschung | `zwei` | `*(nicht da)*` | …dann gerne auch [___] oder drei gäser… |
| 1254 | Löschung | `oder` | `*(nicht da)*` | …gerne auch zwei [___] drei gäser okay… |
| 1255 | Löschung | `drei` | `*(nicht da)*` | …auch zwei oder [___] gäser okay dieses… |
| 1256 | Löschung | `gäser` | `*(nicht da)*` | …zwei oder drei [___] okay dieses gläschen… |
| 1257 | Löschung | `okay` | `*(nicht da)*` | …oder drei gäser [___] dieses gläschen wein… |
| 1258 | Löschung | `dieses` | `*(nicht da)*` | …drei gäser okay [___] gläschen wein nach… |
| 1259 | Löschung | `gläschen` | `*(nicht da)*` | …gäser okay dieses [___] wein nach der… |
| 1260 | Löschung | `wein` | `*(nicht da)*` | …okay dieses gläschen [___] nach der arbeit… |
| 1261 | Löschung | `nach` | `*(nicht da)*` | …dieses gläschen wein [___] der arbeit ist… |
| 1262 | Löschung | `arbeit` | `*(nicht da)*` | …wein nach der [___] ist das so… |
| 1263 | Löschung | `ist` | `*(nicht da)*` | …nach der arbeit [___] das so einmal… |
| 1264 | Löschung | `das` | `*(nicht da)*` | …der arbeit ist [___] so einmal die… |
| 1265 | Löschung | `so` | `*(nicht da)*` | …arbeit ist das [___] einmal die woche… |
| 1266 | Löschung | `einmal` | `*(nicht da)*` | …ist das so [___] die woche zweimal… |
| 1267 | Löschung | `die` | `*(nicht da)*` | …das so einmal [___] woche zweimal oder… |
| 1268 | Löschung | `woche` | `*(nicht da)*` | …so einmal die [___] zweimal oder doch… |
| 1269 | Löschung | `zweimal` | `*(nicht da)*` | …einmal die woche [___] oder doch öfter… |
| 1270 | Löschung | `oder` | `*(nicht da)*` | …die woche zweimal [___] doch öfter ach… |
| 1271 | Löschung | `doch` | `*(nicht da)*` | …woche zweimal oder [___] öfter ach das… |
| 1272 | Löschung | `öfter` | `*(nicht da)*` | …zweimal oder doch [___] ach das ist… |
| 1273 | Löschung | `ach` | `*(nicht da)*` | …oder doch öfter [___] das ist schon… |
| 1274 | Löschung | `das` | `*(nicht da)*` | …doch öfter ach [___] ist schon fast… |
| 1275 | Löschung | `ist` | `*(nicht da)*` | …öfter ach das [___] schon fast jeden… |
| 1276 | Löschung | `schon` | `*(nicht da)*` | …ach das ist [___] fast jeden abend… |
| 1277 | Löschung | `fast` | `*(nicht da)*` | …das ist schon [___] jeden abend aber… |
| 1278 | Löschung | `jeden` | `*(nicht da)*` | …ist schon fast [___] abend aber ein… |
| 1279 | Löschung | `abend` | `*(nicht da)*` | …schon fast jeden [___] aber ein kleines… |
| 1280 | Löschung | `aber` | `*(nicht da)*` | …fast jeden abend [___] ein kleines gläschen… |
| 1281 | Löschung | `kleines` | `*(nicht da)*` | …abend aber ein [___] gläschen okay wunderbar… |
| 1282 | Löschung | `gläschen` | `*(nicht da)*` | …aber ein kleines [___] okay wunderbar frau… |
| 1283 | Löschung | `okay` | `*(nicht da)*` | …ein kleines gläschen [___] wunderbar frau beckenwestfalen… |
| 1284 | Substitution | `wunderbar` | `halux` | …kleines gläschen okay [___] frau beckenwestfalen nehmen… |
| 1285 | Substitution | `frau` | `valgus` | …gläschen okay wunderbar [___] beckenwestfalen nehmen sie… |
| 1286 | Substitution | `beckenwestfalen` | `operiert` | …okay wunderbar frau [___] nehmen sie das… |
| 1287 | Substitution | `nehmen` | `wurde` | …wunderbar frau beckenwestfalen [___] sie das jetzt… |
| 1288 | Löschung | `das` | `*(nicht da)*` | …beckenwestfalen nehmen sie [___] jetzt bitte nicht… |
| 1289 | Löschung | `jetzt` | `*(nicht da)*` | …nehmen sie das [___] bitte nicht persönlich… |
| 1290 | Löschung | `bitte` | `*(nicht da)*` | …sie das jetzt [___] nicht persönlich das… |
| 1291 | Löschung | `nicht` | `*(nicht da)*` | …das jetzt bitte [___] persönlich das ist… |
| 1292 | Löschung | `persönlich` | `*(nicht da)*` | …jetzt bitte nicht [___] das ist eine… |
| 1293 | Löschung | `das` | `*(nicht da)*` | …bitte nicht persönlich [___] ist eine reine… |
| 1294 | Löschung | `eine` | `*(nicht da)*` | …persönlich das ist [___] reine routinefrage die… |
| 1295 | Löschung | `reine` | `*(nicht da)*` | …das ist eine [___] routinefrage die ich… |
| 1296 | Löschung | `routinefrage` | `*(nicht da)*` | …ist eine reine [___] die ich aber… |
| 1297 | Löschung | `die` | `*(nicht da)*` | …eine reine routinefrage [___] ich aber natürlich… |
| 1298 | Löschung | `ich` | `*(nicht da)*` | …reine routinefrage die [___] aber natürlich auch… |
| 1299 | Löschung | `aber` | `*(nicht da)*` | …routinefrage die ich [___] natürlich auch ihnen… |
| 1300 | Löschung | `natürlich` | `*(nicht da)*` | …die ich aber [___] auch ihnen stellen… |
| 1301 | Löschung | `ihnen` | `*(nicht da)*` | …aber natürlich auch [___] stellen muss und… |
| 1302 | Löschung | `stellen` | `*(nicht da)*` | …natürlich auch ihnen [___] muss und zwar… |
| 1303 | Löschung | `muss` | `*(nicht da)*` | …auch ihnen stellen [___] und zwar nehmen… |
| 1304 | Löschung | `und` | `*(nicht da)*` | …ihnen stellen muss [___] zwar nehmen sie… |
| 1305 | Löschung | `zwar` | `*(nicht da)*` | …stellen muss und [___] nehmen sie drogen… |
| 1306 | Löschung | `nehmen` | `*(nicht da)*` | …muss und zwar [___] sie drogen nein… |
| 1307 | Löschung | `sie` | `*(nicht da)*` | …und zwar nehmen [___] drogen nein ich… |
| 1308 | Löschung | `drogen` | `*(nicht da)*` | …zwar nehmen sie [___] nein ich nehme… |
| 1309 | Löschung | `nein` | `*(nicht da)*` | …nehmen sie drogen [___] ich nehme keine… |
| 1310 | Löschung | `ich` | `*(nicht da)*` | …sie drogen nein [___] nehme keine drogen… |
| 1311 | Löschung | `nehme` | `*(nicht da)*` | …drogen nein ich [___] keine drogen wobei… |
| 1312 | Löschung | `keine` | `*(nicht da)*` | …nein ich nehme [___] drogen wobei ich… |
| 1313 | Löschung | `drogen` | `*(nicht da)*` | …ich nehme keine [___] wobei ich zugeben… |
| 1314 | Löschung | `wobei` | `*(nicht da)*` | …nehme keine drogen [___] ich zugeben muss… |
| 1315 | Löschung | `ich` | `*(nicht da)*` | …keine drogen wobei [___] zugeben muss dass… |
| 1316 | Löschung | `zugeben` | `*(nicht da)*` | …drogen wobei ich [___] muss dass ich… |
| 1317 | Löschung | `muss` | `*(nicht da)*` | …wobei ich zugeben [___] dass ich vor… |
| 1318 | Löschung | `dass` | `*(nicht da)*` | …ich zugeben muss [___] ich vor einiger… |
| 1319 | Löschung | `ich` | `*(nicht da)*` | …zugeben muss dass [___] vor einiger zeit… |
| 1320 | Löschung | `vor` | `*(nicht da)*` | …muss dass ich [___] einiger zeit ab… |
| 1321 | Löschung | `einiger` | `*(nicht da)*` | …dass ich vor [___] zeit ab und… |
| 1322 | Löschung | `zeit` | `*(nicht da)*` | …ich vor einiger [___] ab und zu… |
| 1323 | Löschung | `ab` | `*(nicht da)*` | …vor einiger zeit [___] und zu mal… |
| 1324 | Löschung | `und` | `*(nicht da)*` | …einiger zeit ab [___] zu mal ritalin… |
| 1325 | Löschung | `zu` | `*(nicht da)*` | …zeit ab und [___] mal ritalin genommen… |
| 1326 | Löschung | `mal` | `*(nicht da)*` | …ab und zu [___] ritalin genommen habe… |
| 1327 | Löschung | `ritalin` | `*(nicht da)*` | …und zu mal [___] genommen habe einfach… |
| 1328 | Löschung | `genommen` | `*(nicht da)*` | …zu mal ritalin [___] habe einfach weil… |
| 1329 | Löschung | `habe` | `*(nicht da)*` | …mal ritalin genommen [___] einfach weil wir… |
| 1330 | Löschung | `einfach` | `*(nicht da)*` | …ritalin genommen habe [___] weil wir ein… |
| 1331 | Löschung | `weil` | `*(nicht da)*` | …genommen habe einfach [___] wir ein paar… |
| 1332 | Löschung | `wir` | `*(nicht da)*` | …habe einfach weil [___] ein paar wirklich… |
| 1333 | Löschung | `ein` | `*(nicht da)*` | …einfach weil wir [___] paar wirklich große… |
| 1334 | Löschung | `paar` | `*(nicht da)*` | …weil wir ein [___] wirklich große projekte… |
| 1335 | Substitution | `wirklich` | `ansonsten` | …wir ein paar [___] große projekte auf… |
| 1336 | Substitution | `große` | `gesund` | …ein paar wirklich [___] projekte auf der… |
| 1337 | Substitution | `projekte` | `bis` | …paar wirklich große [___] auf der arbeit… |
| 1338 | Löschung | `der` | `*(nicht da)*` | …große projekte auf [___] arbeit hatten für… |
| 1339 | Löschung | `arbeit` | `*(nicht da)*` | …projekte auf der [___] hatten für die… |
| 1340 | Löschung | `hatten` | `*(nicht da)*` | …auf der arbeit [___] für die ich… |
| 1341 | Löschung | `für` | `*(nicht da)*` | …der arbeit hatten [___] die ich zuständig… |
| 1342 | Löschung | `ich` | `*(nicht da)*` | …hatten für die [___] zuständig war und… |
| 1343 | Löschung | `zuständig` | `*(nicht da)*` | …für die ich [___] war und ich… |
| 1344 | Löschung | `war` | `*(nicht da)*` | …die ich zuständig [___] und ich musste… |
| 1345 | Löschung | `und` | `*(nicht da)*` | …ich zuständig war [___] ich musste wirklich… |
| 1346 | Löschung | `ich` | `*(nicht da)*` | …zuständig war und [___] musste wirklich sehr… |
| 1347 | Löschung | `musste` | `*(nicht da)*` | …war und ich [___] wirklich sehr lange… |
| 1348 | Löschung | `wirklich` | `*(nicht da)*` | …und ich musste [___] sehr lange arbeiten… |
| 1349 | Löschung | `sehr` | `*(nicht da)*` | …ich musste wirklich [___] lange arbeiten und… |
| 1350 | Löschung | `lange` | `*(nicht da)*` | …musste wirklich sehr [___] arbeiten und ja… |
| 1351 | Löschung | `arbeiten` | `*(nicht da)*` | …wirklich sehr lange [___] und ja habe… |
| 1352 | Löschung | `und` | `*(nicht da)*` | …sehr lange arbeiten [___] ja habe zwei… |
| 1353 | Löschung | `ja` | `*(nicht da)*` | …lange arbeiten und [___] habe zwei dreimal… |
| 1354 | Löschung | `habe` | `*(nicht da)*` | …arbeiten und ja [___] zwei dreimal ritalin… |
| 1355 | Löschung | `zwei` | `*(nicht da)*` | …und ja habe [___] dreimal ritalin genommen… |
| 1356 | Löschung | `dreimal` | `*(nicht da)*` | …ja habe zwei [___] ritalin genommen okay… |
| 1357 | Löschung | `ritalin` | `*(nicht da)*` | …habe zwei dreimal [___] genommen okay das… |
| 1358 | Löschung | `genommen` | `*(nicht da)*` | …zwei dreimal ritalin [___] okay das war… |
| 1359 | Löschung | `okay` | `*(nicht da)*` | …dreimal ritalin genommen [___] das war es… |
| 1360 | Löschung | `das` | `*(nicht da)*` | …ritalin genommen okay [___] war es aber… |
| 1361 | Löschung | `war` | `*(nicht da)*` | …genommen okay das [___] es aber ja… |
| 1362 | Löschung | `es` | `*(nicht da)*` | …okay das war [___] aber ja ja… |
| 1363 | Löschung | `aber` | `*(nicht da)*` | …das war es [___] ja ja sehr… |
| 1364 | Löschung | `ja` | `*(nicht da)*` | …war es aber [___] ja sehr gut… |
| 1365 | Löschung | `ja` | `*(nicht da)*` | …es aber ja [___] sehr gut okay… |
| 1366 | Löschung | `sehr` | `*(nicht da)*` | …aber ja ja [___] gut okay prima… |
| 1367 | Löschung | `gut` | `*(nicht da)*` | …ja ja sehr [___] okay prima gut… |
| 1368 | Löschung | `okay` | `*(nicht da)*` | …ja sehr gut [___] prima gut kurz… |
| 1369 | Löschung | `prima` | `*(nicht da)*` | …sehr gut okay [___] gut kurz zu… |
| 1370 | Löschung | `gut` | `*(nicht da)*` | …gut okay prima [___] kurz zu ihrer… |
| 1371 | Löschung | `kurz` | `*(nicht da)*` | …okay prima gut [___] zu ihrer familie… |
| 1372 | Löschung | `zu` | `*(nicht da)*` | …prima gut kurz [___] ihrer familie gibt… |
| 1373 | Löschung | `ihrer` | `*(nicht da)*` | …gut kurz zu [___] familie gibt es… |
| 1374 | Löschung | `familie` | `*(nicht da)*` | …kurz zu ihrer [___] gibt es in… |
| 1375 | Löschung | `gibt` | `*(nicht da)*` | …zu ihrer familie [___] es in ihrer… |
| 1376 | Löschung | `es` | `*(nicht da)*` | …ihrer familie gibt [___] in ihrer familie… |
| 1377 | Löschung | `in` | `*(nicht da)*` | …familie gibt es [___] ihrer familie eltern… |
| 1378 | Löschung | `ihrer` | `*(nicht da)*` | …gibt es in [___] familie eltern großeltern… |
| 1379 | Löschung | `familie` | `*(nicht da)*` | …es in ihrer [___] eltern großeltern geschwister… |
| 1380 | Löschung | `eltern` | `*(nicht da)*` | …in ihrer familie [___] großeltern geschwister irgendwelche… |
| 1381 | Löschung | `großeltern` | `*(nicht da)*` | …ihrer familie eltern [___] geschwister irgendwelche vorerkrankungen… |
| 1382 | Löschung | `geschwister` | `*(nicht da)*` | …familie eltern großeltern [___] irgendwelche vorerkrankungen oder… |
| 1383 | Löschung | `irgendwelche` | `*(nicht da)*` | …eltern großeltern geschwister [___] vorerkrankungen oder chronische… |
| 1384 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …großeltern geschwister irgendwelche [___] oder chronische erkrankungen… |
| 1385 | Löschung | `oder` | `*(nicht da)*` | …geschwister irgendwelche vorerkrankungen [___] chronische erkrankungen wie… |
| 1386 | Löschung | `chronische` | `*(nicht da)*` | …irgendwelche vorerkrankungen oder [___] erkrankungen wie zum… |
| 1387 | Löschung | `erkrankungen` | `*(nicht da)*` | …vorerkrankungen oder chronische [___] wie zum beispiel… |
| 1388 | Löschung | `wie` | `*(nicht da)*` | …oder chronische erkrankungen [___] zum beispiel krebs… |
| 1389 | Löschung | `zum` | `*(nicht da)*` | …chronische erkrankungen wie [___] beispiel krebs oder… |
| 1390 | Löschung | `beispiel` | `*(nicht da)*` | …erkrankungen wie zum [___] krebs oder diabetes… |
| 1391 | Löschung | `krebs` | `*(nicht da)*` | …wie zum beispiel [___] oder diabetes oder… |
| 1392 | Löschung | `oder` | `*(nicht da)*` | …zum beispiel krebs [___] diabetes oder einen… |
| 1393 | Löschung | `diabetes` | `*(nicht da)*` | …beispiel krebs oder [___] oder einen herzinfarkt… |
| 1394 | Löschung | `oder` | `*(nicht da)*` | …krebs oder diabetes [___] einen herzinfarkt irgendetwas… |
| 1395 | Löschung | `einen` | `*(nicht da)*` | …oder diabetes oder [___] herzinfarkt irgendetwas was… |
| 1396 | Löschung | `herzinfarkt` | `*(nicht da)*` | …diabetes oder einen [___] irgendetwas was ihnen… |
| 1397 | Löschung | `irgendetwas` | `*(nicht da)*` | …oder einen herzinfarkt [___] was ihnen bekannt… |
| 1398 | Löschung | `was` | `*(nicht da)*` | …einen herzinfarkt irgendetwas [___] ihnen bekannt ist… |
| 1399 | Löschung | `ihnen` | `*(nicht da)*` | …herzinfarkt irgendetwas was [___] bekannt ist das… |
| 1400 | Löschung | `bekannt` | `*(nicht da)*` | …irgendetwas was ihnen [___] ist das gibt… |
| 1401 | Löschung | `ist` | `*(nicht da)*` | …was ihnen bekannt [___] das gibt es… |
| 1402 | Löschung | `das` | `*(nicht da)*` | …ihnen bekannt ist [___] gibt es ja… |
| 1403 | Löschung | `gibt` | `*(nicht da)*` | …bekannt ist das [___] es ja großeltern… |
| 1404 | Löschung | `es` | `*(nicht da)*` | …ist das gibt [___] ja großeltern auch… |
| 1405 | Löschung | `ja` | `*(nicht da)*` | …das gibt es [___] großeltern auch ja… |
| 1406 | Löschung | `großeltern` | `*(nicht da)*` | …gibt es ja [___] auch ja klar… |
| 1407 | Löschung | `auch` | `*(nicht da)*` | …es ja großeltern [___] ja klar mein… |
| 1408 | Löschung | `ja` | `*(nicht da)*` | …ja großeltern auch [___] klar mein großvater… |
| 1409 | Löschung | `klar` | `*(nicht da)*` | …großeltern auch ja [___] mein großvater hatte… |
| 1410 | Löschung | `mein` | `*(nicht da)*` | …auch ja klar [___] großvater hatte leberzirrhose… |
| 1411 | Löschung | `großvater` | `*(nicht da)*` | …ja klar mein [___] hatte leberzirrhose und… |
| 1412 | Löschung | `hatte` | `*(nicht da)*` | …klar mein großvater [___] leberzirrhose und ist… |
| 1413 | Löschung | `leberzirrhose` | `*(nicht da)*` | …mein großvater hatte [___] und ist leider… |
| 1414 | Löschung | `und` | `*(nicht da)*` | …großvater hatte leberzirrhose [___] ist leider auch… |
| 1415 | Löschung | `ist` | `*(nicht da)*` | …hatte leberzirrhose und [___] leider auch daran… |
| 1416 | Löschung | `leider` | `*(nicht da)*` | …leberzirrhose und ist [___] auch daran geschrauben… |
| 1417 | Löschung | `auch` | `*(nicht da)*` | …und ist leider [___] daran geschrauben oh… |
| 1418 | Löschung | `daran` | `*(nicht da)*` | …ist leider auch [___] geschrauben oh das… |
| 1419 | Löschung | `geschrauben` | `*(nicht da)*` | …leider auch daran [___] oh das tut… |
| 1420 | Löschung | `oh` | `*(nicht da)*` | …auch daran geschrauben [___] das tut mir… |
| 1421 | Löschung | `das` | `*(nicht da)*` | …daran geschrauben oh [___] tut mir leid… |
| 1422 | Löschung | `tut` | `*(nicht da)*` | …geschrauben oh das [___] mir leid danke… |
| 1423 | Löschung | `mir` | `*(nicht da)*` | …oh das tut [___] leid danke ist… |
| 1424 | Löschung | `leid` | `*(nicht da)*` | …das tut mir [___] danke ist schon… |
| 1425 | Löschung | `danke` | `*(nicht da)*` | …tut mir leid [___] ist schon lange… |
| 1426 | Löschung | `ist` | `*(nicht da)*` | …mir leid danke [___] schon lange her… |
| 1427 | Löschung | `schon` | `*(nicht da)*` | …leid danke ist [___] lange her und… |
| 1428 | Löschung | `lange` | `*(nicht da)*` | …danke ist schon [___] her und meine… |
| 1429 | Löschung | `her` | `*(nicht da)*` | …ist schon lange [___] und meine großmutter… |
| 1430 | Löschung | `und` | `*(nicht da)*` | …schon lange her [___] meine großmutter hatte… |
| 1431 | Löschung | `meine` | `*(nicht da)*` | …lange her und [___] großmutter hatte großkrebs… |
| 1432 | Löschung | `großmutter` | `*(nicht da)*` | …her und meine [___] hatte großkrebs aber… |
| 1433 | Löschung | `hatte` | `*(nicht da)*` | …und meine großmutter [___] großkrebs aber sie… |
| 1434 | Löschung | `großkrebs` | `*(nicht da)*` | …meine großmutter hatte [___] aber sie lebt… |
| 1435 | Löschung | `aber` | `*(nicht da)*` | …großmutter hatte großkrebs [___] sie lebt noch… |
| 1436 | Löschung | `sie` | `*(nicht da)*` | …hatte großkrebs aber [___] lebt noch okay… |
| 1437 | Löschung | `lebt` | `*(nicht da)*` | …großkrebs aber sie [___] noch okay sehr… |
| 1438 | Löschung | `noch` | `*(nicht da)*` | …aber sie lebt [___] okay sehr gut… |
| 1439 | Löschung | `okay` | `*(nicht da)*` | …sie lebt noch [___] sehr gut ihre… |
| 1440 | Löschung | `sehr` | `*(nicht da)*` | …lebt noch okay [___] gut ihre eltern… |
| 1441 | Löschung | `gut` | `*(nicht da)*` | …noch okay sehr [___] ihre eltern sind… |
| 1442 | Löschung | `ihre` | `*(nicht da)*` | …okay sehr gut [___] eltern sind gesund… |
| 1443 | Löschung | `eltern` | `*(nicht da)*` | …sehr gut ihre [___] sind gesund meine… |
| 1444 | Löschung | `sind` | `*(nicht da)*` | …gut ihre eltern [___] gesund meine eltern… |
| 1445 | Löschung | `gesund` | `*(nicht da)*` | …ihre eltern sind [___] meine eltern sind… |
| 1446 | Löschung | `meine` | `*(nicht da)*` | …eltern sind gesund [___] eltern sind zum… |
| 1447 | Löschung | `eltern` | `*(nicht da)*` | …sind gesund meine [___] sind zum glück… |
| 1448 | Löschung | `sind` | `*(nicht da)*` | …gesund meine eltern [___] zum glück gesund… |
| 1449 | Löschung | `zum` | `*(nicht da)*` | …meine eltern sind [___] glück gesund ja… |
| 1450 | Löschung | `glück` | `*(nicht da)*` | …eltern sind zum [___] gesund ja sehr… |
| 1451 | Löschung | `gesund` | `*(nicht da)*` | …sind zum glück [___] ja sehr schön… |
| 1452 | Löschung | `ja` | `*(nicht da)*` | …zum glück gesund [___] sehr schön haben… |
| 1453 | Löschung | `sehr` | `*(nicht da)*` | …glück gesund ja [___] schön haben sie… |
| 1454 | Löschung | `schön` | `*(nicht da)*` | …gesund ja sehr [___] haben sie geschwister… |
| 1455 | Löschung | `haben` | `*(nicht da)*` | …ja sehr schön [___] sie geschwister frau… |
| 1456 | Substitution | `sie` | `kistaminunverträglichkeit` | …sehr schön haben [___] geschwister frau böcken… |
| 1457 | Substitution | `geschwister` | `die` | …schön haben sie [___] frau böcken westfalen… |
| 1458 | Löschung | `böcken` | `*(nicht da)*` | …sie geschwister frau [___] westfalen ich habe… |
| 1459 | Löschung | `westfalen` | `*(nicht da)*` | …geschwister frau böcken [___] ich habe eine… |
| 1460 | Löschung | `ich` | `*(nicht da)*` | …frau böcken westfalen [___] habe eine schwester… |
| 1461 | Löschung | `habe` | `*(nicht da)*` | …böcken westfalen ich [___] eine schwester und… |
| 1462 | Löschung | `eine` | `*(nicht da)*` | …westfalen ich habe [___] schwester und sie… |
| 1463 | Löschung | `schwester` | `*(nicht da)*` | …ich habe eine [___] und sie hat… |
| 1464 | Löschung | `und` | `*(nicht da)*` | …habe eine schwester [___] sie hat auch… |
| 1465 | Löschung | `sie` | `*(nicht da)*` | …eine schwester und [___] hat auch ein… |
| 1466 | Löschung | `hat` | `*(nicht da)*` | …schwester und sie [___] auch ein paar… |
| 1467 | Löschung | `auch` | `*(nicht da)*` | …und sie hat [___] ein paar problemchen… |
| 1468 | Löschung | `ein` | `*(nicht da)*` | …sie hat auch [___] paar problemchen und… |
| 1469 | Löschung | `paar` | `*(nicht da)*` | …hat auch ein [___] problemchen und zwar… |
| 1470 | Löschung | `problemchen` | `*(nicht da)*` | …auch ein paar [___] und zwar hat… |
| 1471 | Löschung | `und` | `*(nicht da)*` | …ein paar problemchen [___] zwar hat sie… |
| 1472 | Löschung | `zwar` | `*(nicht da)*` | …paar problemchen und [___] hat sie asthma… |
| 1473 | Löschung | `sie` | `*(nicht da)*` | …und zwar hat [___] asthma und neurodermitis… |
| 1474 | Löschung | `asthma` | `*(nicht da)*` | …zwar hat sie [___] und neurodermitis asthma… |
| 1475 | Löschung | `und` | `*(nicht da)*` | …hat sie asthma [___] neurodermitis asthma und… |
| 1476 | Löschung | `neurodermitis` | `*(nicht da)*` | …sie asthma und [___] asthma und neurodermitis… |
| 1477 | Löschung | `asthma` | `*(nicht da)*` | …asthma und neurodermitis [___] und neurodermitis okay… |
| 1478 | Löschung | `und` | `*(nicht da)*` | …und neurodermitis asthma [___] neurodermitis okay aber… |
| 1479 | Löschung | `neurodermitis` | `*(nicht da)*` | …neurodermitis asthma und [___] okay aber sonst… |
| 1480 | Löschung | `okay` | `*(nicht da)*` | …asthma und neurodermitis [___] aber sonst ist… |
| 1481 | Löschung | `aber` | `*(nicht da)*` | …und neurodermitis okay [___] sonst ist auch… |
| 1482 | Löschung | `sonst` | `*(nicht da)*` | …neurodermitis okay aber [___] ist auch sie… |
| 1483 | Löschung | `ist` | `*(nicht da)*` | …okay aber sonst [___] auch sie gesund… |
| 1484 | Löschung | `auch` | `*(nicht da)*` | …aber sonst ist [___] sie gesund ja… |
| 1485 | Löschung | `sie` | `*(nicht da)*` | …sonst ist auch [___] gesund ja sonst… |
| 1486 | Löschung | `gesund` | `*(nicht da)*` | …ist auch sie [___] ja sonst geht… |
| 1487 | Löschung | `ja` | `*(nicht da)*` | …auch sie gesund [___] sonst geht sie… |
| 1488 | Löschung | `sonst` | `*(nicht da)*` | …sie gesund ja [___] geht sie gut… |
| 1489 | Löschung | `geht` | `*(nicht da)*` | …gesund ja sonst [___] sie gut sehr… |
| 1490 | Löschung | `sie` | `*(nicht da)*` | …ja sonst geht [___] gut sehr gut… |
| 1491 | Löschung | `gut` | `*(nicht da)*` | …sonst geht sie [___] sehr gut haben… |
| 1492 | Löschung | `sehr` | `*(nicht da)*` | …geht sie gut [___] gut haben sie… |
| 1493 | Löschung | `gut` | `*(nicht da)*` | …sie gut sehr [___] haben sie kinder… |
| 1494 | Löschung | `haben` | `*(nicht da)*` | …gut sehr gut [___] sie kinder frau… |
| 1495 | Löschung | `sie` | `*(nicht da)*` | …sehr gut haben [___] kinder frau böcken… |
| 1496 | Löschung | `kinder` | `*(nicht da)*` | …gut haben sie [___] frau böcken westfalen… |
| 1497 | Löschung | `frau` | `*(nicht da)*` | …haben sie kinder [___] böcken westfalen nein… |
| 1498 | Löschung | `böcken` | `*(nicht da)*` | …sie kinder frau [___] westfalen nein ich… |
| 1499 | Löschung | `westfalen` | `*(nicht da)*` | …kinder frau böcken [___] nein ich habe… |
| 1500 | Löschung | `nein` | `*(nicht da)*` | …frau böcken westfalen [___] ich habe keine… |
| 1501 | Löschung | `ich` | `*(nicht da)*` | …böcken westfalen nein [___] habe keine kinder… |
| 1502 | Löschung | `habe` | `*(nicht da)*` | …westfalen nein ich [___] keine kinder okay… |
| 1503 | Löschung | `kinder` | `*(nicht da)*` | …ich habe keine [___] okay wie sieht… |
| 1504 | Löschung | `okay` | `*(nicht da)*` | …habe keine kinder [___] wie sieht es… |
| 1505 | Löschung | `wie` | `*(nicht da)*` | …keine kinder okay [___] sieht es denn… |
| 1506 | Löschung | `sieht` | `*(nicht da)*` | …kinder okay wie [___] es denn in… |
| 1507 | Löschung | `es` | `*(nicht da)*` | …okay wie sieht [___] denn in ihrem… |
| 1508 | Substitution | `denn` | `vorerkrankungen` | …wie sieht es [___] in ihrem sozialleben… |
| 1509 | Löschung | `ihrem` | `*(nicht da)*` | …es denn in [___] sozialleben aus sind… |
| 1510 | Löschung | `sozialleben` | `*(nicht da)*` | …denn in ihrem [___] aus sind sie… |
| 1511 | Löschung | `aus` | `*(nicht da)*` | …in ihrem sozialleben [___] sind sie verheiratet… |
| 1512 | Löschung | `sind` | `*(nicht da)*` | …ihrem sozialleben aus [___] sie verheiratet ich… |
| 1513 | Löschung | `sie` | `*(nicht da)*` | …sozialleben aus sind [___] verheiratet ich bin… |
| 1514 | Löschung | `verheiratet` | `*(nicht da)*` | …aus sind sie [___] ich bin frisch… |
| 1515 | Löschung | `ich` | `*(nicht da)*` | …sind sie verheiratet [___] bin frisch verheiratet… |
| 1516 | Löschung | `bin` | `*(nicht da)*` | …sie verheiratet ich [___] frisch verheiratet ja… |
| 1517 | Löschung | `frisch` | `*(nicht da)*` | …verheiratet ich bin [___] verheiratet ja seit… |
| 1518 | Löschung | `verheiratet` | `*(nicht da)*` | …ich bin frisch [___] ja seit fünf… |
| 1519 | Löschung | `ja` | `*(nicht da)*` | …bin frisch verheiratet [___] seit fünf monaten… |
| 1520 | Löschung | `seit` | `*(nicht da)*` | …frisch verheiratet ja [___] fünf monaten wie… |
| 1521 | Löschung | `fünf` | `*(nicht da)*` | …verheiratet ja seit [___] monaten wie schön… |
| 1522 | Löschung | `monaten` | `*(nicht da)*` | …ja seit fünf [___] wie schön herzlichen… |
| 1523 | Löschung | `wie` | `*(nicht da)*` | …seit fünf monaten [___] schön herzlichen glückwunsch… |
| 1524 | Löschung | `schön` | `*(nicht da)*` | …fünf monaten wie [___] herzlichen glückwunsch auch… |
| 1525 | Löschung | `herzlichen` | `*(nicht da)*` | …monaten wie schön [___] glückwunsch auch dazu… |
| 1526 | Löschung | `glückwunsch` | `*(nicht da)*` | …wie schön herzlichen [___] auch dazu herzlichen… |
| 1527 | Löschung | `auch` | `*(nicht da)*` | …schön herzlichen glückwunsch [___] dazu herzlichen dank… |
| 1528 | Löschung | `dazu` | `*(nicht da)*` | …herzlichen glückwunsch auch [___] herzlichen dank sehr… |
| 1529 | Löschung | `herzlichen` | `*(nicht da)*` | …glückwunsch auch dazu [___] dank sehr schön… |
| 1530 | Löschung | `dank` | `*(nicht da)*` | …auch dazu herzlichen [___] sehr schön dann… |
| 1531 | Löschung | `sehr` | `*(nicht da)*` | …dazu herzlichen dank [___] schön dann gehe… |
| 1532 | Löschung | `schön` | `*(nicht da)*` | …herzlichen dank sehr [___] dann gehe ich… |
| 1533 | Löschung | `dann` | `*(nicht da)*` | …dank sehr schön [___] gehe ich davon… |
| 1534 | Löschung | `gehe` | `*(nicht da)*` | …sehr schön dann [___] ich davon aus… |
| 1535 | Löschung | `ich` | `*(nicht da)*` | …schön dann gehe [___] davon aus sie… |
| 1536 | Löschung | `davon` | `*(nicht da)*` | …dann gehe ich [___] aus sie leben… |
| 1537 | Löschung | `aus` | `*(nicht da)*` | …gehe ich davon [___] sie leben auch… |
| 1538 | Löschung | `sie` | `*(nicht da)*` | …ich davon aus [___] leben auch mit… |
| 1539 | Löschung | `leben` | `*(nicht da)*` | …davon aus sie [___] auch mit ihrem… |
| 1540 | Löschung | `auch` | `*(nicht da)*` | …aus sie leben [___] mit ihrem ehemann… |
| 1541 | Löschung | `mit` | `*(nicht da)*` | …sie leben auch [___] ihrem ehemann zusammen… |
| 1542 | Löschung | `ihrem` | `*(nicht da)*` | …leben auch mit [___] ehemann zusammen das… |
| 1543 | Löschung | `ehemann` | `*(nicht da)*` | …auch mit ihrem [___] zusammen das ist… |
| 1544 | Löschung | `zusammen` | `*(nicht da)*` | …mit ihrem ehemann [___] das ist richtig… |
| 1545 | Löschung | `das` | `*(nicht da)*` | …ihrem ehemann zusammen [___] ist richtig ja… |
| 1546 | Löschung | `ist` | `*(nicht da)*` | …ehemann zusammen das [___] richtig ja okay… |
| 1547 | Löschung | `richtig` | `*(nicht da)*` | …zusammen das ist [___] ja okay prima… |
| 1548 | Löschung | `ja` | `*(nicht da)*` | …das ist richtig [___] okay prima wir… |
| 1549 | Löschung | `okay` | `*(nicht da)*` | …ist richtig ja [___] prima wir hatten… |
| 1550 | Löschung | `prima` | `*(nicht da)*` | …richtig ja okay [___] wir hatten zwar… |
| 1551 | Löschung | `wir` | `*(nicht da)*` | …ja okay prima [___] hatten zwar eben… |
| 1552 | Löschung | `hatten` | `*(nicht da)*` | …okay prima wir [___] zwar eben schon… |
| 1553 | Löschung | `zwar` | `*(nicht da)*` | …prima wir hatten [___] eben schon mal… |
| 1554 | Löschung | `eben` | `*(nicht da)*` | …wir hatten zwar [___] schon mal kurz… |
| 1555 | Löschung | `schon` | `*(nicht da)*` | …hatten zwar eben [___] mal kurz über… |
| 1556 | Löschung | `mal` | `*(nicht da)*` | …zwar eben schon [___] kurz über ihre… |
| 1557 | Löschung | `kurz` | `*(nicht da)*` | …eben schon mal [___] über ihre arbeit… |
| 1558 | Löschung | `über` | `*(nicht da)*` | …schon mal kurz [___] ihre arbeit gesprochen… |
| 1559 | Löschung | `ihre` | `*(nicht da)*` | …mal kurz über [___] arbeit gesprochen aber… |
| 1560 | Substitution | `arbeit` | `ihrer` | …kurz über ihre [___] gesprochen aber ich… |
| 1561 | Substitution | `gesprochen` | `person` | …über ihre arbeit [___] aber ich habe… |
| 1562 | Löschung | `ich` | `*(nicht da)*` | …arbeit gesprochen aber [___] habe es nicht… |
| 1563 | Löschung | `habe` | `*(nicht da)*` | …gesprochen aber ich [___] es nicht ganz… |
| 1564 | Löschung | `nicht` | `*(nicht da)*` | …ich habe es [___] ganz auf dem… |
| 1565 | Löschung | `ganz` | `*(nicht da)*` | …habe es nicht [___] auf dem schirm… |
| 1566 | Löschung | `auf` | `*(nicht da)*` | …es nicht ganz [___] dem schirm ob… |
| 1567 | Löschung | `dem` | `*(nicht da)*` | …nicht ganz auf [___] schirm ob ich… |
| 1568 | Löschung | `schirm` | `*(nicht da)*` | …ganz auf dem [___] ob ich sie… |
| 1569 | Löschung | `ob` | `*(nicht da)*` | …auf dem schirm [___] ich sie schon… |
| 1570 | Löschung | `ich` | `*(nicht da)*` | …dem schirm ob [___] sie schon gefragt… |
| 1571 | Löschung | `sie` | `*(nicht da)*` | …schirm ob ich [___] schon gefragt habe… |
| 1572 | Löschung | `schon` | `*(nicht da)*` | …ob ich sie [___] gefragt habe was… |
| 1573 | Löschung | `gefragt` | `*(nicht da)*` | …ich sie schon [___] habe was sie… |
| 1574 | Löschung | `habe` | `*(nicht da)*` | …sie schon gefragt [___] was sie denn… |
| 1575 | Löschung | `was` | `*(nicht da)*` | …schon gefragt habe [___] sie denn beruflich… |
| 1576 | Löschung | `sie` | `*(nicht da)*` | …gefragt habe was [___] denn beruflich machen… |
| 1577 | Löschung | `denn` | `*(nicht da)*` | …habe was sie [___] beruflich machen ich… |
| 1578 | Löschung | `beruflich` | `*(nicht da)*` | …was sie denn [___] machen ich arbeite… |
| 1579 | Substitution | `machen` | `gibt` | …sie denn beruflich [___] ich arbeite in… |
| 1580 | Substitution | `ich` | `einige` | …denn beruflich machen [___] arbeite in einer… |
| 1581 | Substitution | `arbeite` | `vorerkrankungen` | …beruflich machen ich [___] in einer marketingagentur… |
| 1582 | Löschung | `einer` | `*(nicht da)*` | …ich arbeite in [___] marketingagentur wenn da… |
| 1583 | Löschung | `marketingagentur` | `*(nicht da)*` | …arbeite in einer [___] wenn da für… |
| 1584 | Löschung | `wenn` | `*(nicht da)*` | …in einer marketingagentur [___] da für größere… |
| 1585 | Löschung | `da` | `*(nicht da)*` | …einer marketingagentur wenn [___] für größere kunden… |
| 1586 | Löschung | `für` | `*(nicht da)*` | …marketingagentur wenn da [___] größere kunden und… |
| 1587 | Löschung | `größere` | `*(nicht da)*` | …wenn da für [___] kunden und für… |
| 1588 | Löschung | `kunden` | `*(nicht da)*` | …da für größere [___] und für größere… |
| 1589 | Löschung | `und` | `*(nicht da)*` | …für größere kunden [___] für größere firmenkunden… |
| 1590 | Löschung | `für` | `*(nicht da)*` | …größere kunden und [___] größere firmenkunden und… |
| 1591 | Löschung | `größere` | `*(nicht da)*` | …kunden und für [___] firmenkunden und marketingprojekte… |
| 1592 | Löschung | `firmenkunden` | `*(nicht da)*` | …und für größere [___] und marketingprojekte zuständig… |
| 1593 | Löschung | `und` | `*(nicht da)*` | …für größere firmenkunden [___] marketingprojekte zuständig sind… |
| 1594 | Löschung | `marketingprojekte` | `*(nicht da)*` | …größere firmenkunden und [___] zuständig sind okay… |
| 1595 | Löschung | `zuständig` | `*(nicht da)*` | …firmenkunden und marketingprojekte [___] sind okay sehr… |
| 1596 | Löschung | `sind` | `*(nicht da)*` | …und marketingprojekte zuständig [___] okay sehr gut… |
| 1597 | Löschung | `okay` | `*(nicht da)*` | …marketingprojekte zuständig sind [___] sehr gut eine… |
| 1598 | Löschung | `sehr` | `*(nicht da)*` | …zuständig sind okay [___] gut eine letzte… |
| 1599 | Löschung | `gut` | `*(nicht da)*` | …sind okay sehr [___] eine letzte frage… |
| 1600 | Löschung | `eine` | `*(nicht da)*` | …okay sehr gut [___] letzte frage noch… |
| 1601 | Löschung | `letzte` | `*(nicht da)*` | …sehr gut eine [___] frage noch frau… |
| 1602 | Löschung | `frage` | `*(nicht da)*` | …gut eine letzte [___] noch frau böcken… |
| 1603 | Löschung | `noch` | `*(nicht da)*` | …eine letzte frage [___] frau böcken westfalen… |
| 1604 | Löschung | `frau` | `*(nicht da)*` | …letzte frage noch [___] böcken westfalen waren… |
| 1605 | Löschung | `böcken` | `*(nicht da)*` | …frage noch frau [___] westfalen waren sie… |
| 1606 | Substitution | `westfalen` | `ihrer` | …noch frau böcken [___] waren sie in… |
| 1607 | Substitution | `waren` | `familiengeschichte` | …frau böcken westfalen [___] sie in der… |
| 1608 | Substitution | `sie` | `nach` | …böcken westfalen waren [___] in der letzten… |
| 1609 | Substitution | `in` | `abschluss` | …westfalen waren sie [___] der letzten zeit… |
| 1610 | Löschung | `letzten` | `*(nicht da)*` | …sie in der [___] zeit im ausland… |
| 1611 | Löschung | `zeit` | `*(nicht da)*` | …in der letzten [___] im ausland ja… |
| 1612 | Löschung | `im` | `*(nicht da)*` | …der letzten zeit [___] ausland ja ich… |
| 1613 | Löschung | `ausland` | `*(nicht da)*` | …letzten zeit im [___] ja ich war… |
| 1614 | Löschung | `ja` | `*(nicht da)*` | …zeit im ausland [___] ich war vor… |
| 1615 | Löschung | `ich` | `*(nicht da)*` | …im ausland ja [___] war vor zwei… |
| 1616 | Löschung | `war` | `*(nicht da)*` | …ausland ja ich [___] vor zwei monaten… |
| 1617 | Löschung | `vor` | `*(nicht da)*` | …ja ich war [___] zwei monaten geschäftlich… |
| 1618 | Löschung | `zwei` | `*(nicht da)*` | …ich war vor [___] monaten geschäftlich in… |
| 1619 | Löschung | `monaten` | `*(nicht da)*` | …war vor zwei [___] geschäftlich in singapur… |
| 1620 | Löschung | `geschäftlich` | `*(nicht da)*` | …vor zwei monaten [___] in singapur okay… |
| 1621 | Löschung | `in` | `*(nicht da)*` | …zwei monaten geschäftlich [___] singapur okay und… |
| 1622 | Löschung | `singapur` | `*(nicht da)*` | …monaten geschäftlich in [___] okay und wie… |
| 1623 | Löschung | `okay` | `*(nicht da)*` | …geschäftlich in singapur [___] und wie lange… |
| 1624 | Löschung | `und` | `*(nicht da)*` | …in singapur okay [___] wie lange waren… |
| 1625 | Löschung | `wie` | `*(nicht da)*` | …singapur okay und [___] lange waren sie… |
| 1626 | Löschung | `lange` | `*(nicht da)*` | …okay und wie [___] waren sie da… |
| 1627 | Löschung | `waren` | `*(nicht da)*` | …und wie lange [___] sie da zwei… |
| 1628 | Löschung | `sie` | `*(nicht da)*` | …wie lange waren [___] da zwei wochen… |
| 1629 | Löschung | `da` | `*(nicht da)*` | …lange waren sie [___] zwei wochen insgesamt… |
| 1630 | Löschung | `zwei` | `*(nicht da)*` | …waren sie da [___] wochen insgesamt zwei… |
| 1631 | Löschung | `wochen` | `*(nicht da)*` | …sie da zwei [___] insgesamt zwei wochen… |
| 1632 | Löschung | `insgesamt` | `*(nicht da)*` | …da zwei wochen [___] zwei wochen insgesamt… |
| 1633 | Löschung | `zwei` | `*(nicht da)*` | …zwei wochen insgesamt [___] wochen insgesamt okay… |
| 1634 | Löschung | `wochen` | `*(nicht da)*` | …wochen insgesamt zwei [___] insgesamt okay gut… |
| 1635 | Löschung | `insgesamt` | `*(nicht da)*` | …insgesamt zwei wochen [___] okay gut frau… |
| 1636 | Löschung | `okay` | `*(nicht da)*` | …zwei wochen insgesamt [___] gut frau böcken… |
| 1637 | Löschung | `gut` | `*(nicht da)*` | …wochen insgesamt okay [___] frau böcken westfalen… |
| 1638 | Löschung | `frau` | `*(nicht da)*` | …insgesamt okay gut [___] böcken westfalen von… |
| 1639 | Löschung | `böcken` | `*(nicht da)*` | …okay gut frau [___] westfalen von meiner… |
| 1640 | Löschung | `westfalen` | `*(nicht da)*` | …gut frau böcken [___] von meiner seite… |
| 1641 | Löschung | `von` | `*(nicht da)*` | …frau böcken westfalen [___] meiner seite war… |
| 1642 | Löschung | `meiner` | `*(nicht da)*` | …böcken westfalen von [___] seite war es… |
| 1643 | Löschung | `seite` | `*(nicht da)*` | …westfalen von meiner [___] war es das… |
| 1644 | Löschung | `war` | `*(nicht da)*` | …von meiner seite [___] es das ich… |
| 1645 | Löschung | `es` | `*(nicht da)*` | …meiner seite war [___] das ich habe… |
| 1646 | Löschung | `das` | `*(nicht da)*` | …seite war es [___] ich habe alle… |
| 1647 | Löschung | `ich` | `*(nicht da)*` | …war es das [___] habe alle fragen… |
| 1648 | Löschung | `habe` | `*(nicht da)*` | …es das ich [___] alle fragen gestellt… |
| 1649 | Löschung | `alle` | `*(nicht da)*` | …das ich habe [___] fragen gestellt ich… |
| 1650 | Löschung | `fragen` | `*(nicht da)*` | …ich habe alle [___] gestellt ich würde… |
| 1651 | Löschung | `gestellt` | `*(nicht da)*` | …habe alle fragen [___] ich würde das… |
| 1652 | Löschung | `ich` | `*(nicht da)*` | …alle fragen gestellt [___] würde das gleich… |
| 1653 | Löschung | `würde` | `*(nicht da)*` | …fragen gestellt ich [___] das gleich nochmal… |
| 1654 | Löschung | `das` | `*(nicht da)*` | …gestellt ich würde [___] gleich nochmal mit… |
| 1655 | Löschung | `gleich` | `*(nicht da)*` | …ich würde das [___] nochmal mit ihnen… |
| 1656 | Löschung | `nochmal` | `*(nicht da)*` | …würde das gleich [___] mit ihnen durchgehen… |
| 1657 | Löschung | `mit` | `*(nicht da)*` | …das gleich nochmal [___] ihnen durchgehen um… |
| 1658 | Löschung | `ihnen` | `*(nicht da)*` | …gleich nochmal mit [___] durchgehen um abzugleichen… |
| 1659 | Löschung | `durchgehen` | `*(nicht da)*` | …nochmal mit ihnen [___] um abzugleichen dass… |
| 1660 | Löschung | `um` | `*(nicht da)*` | …mit ihnen durchgehen [___] abzugleichen dass ich… |
| 1661 | Löschung | `abzugleichen` | `*(nicht da)*` | …ihnen durchgehen um [___] dass ich auch… |
| 1662 | Löschung | `dass` | `*(nicht da)*` | …durchgehen um abzugleichen [___] ich auch wirklich… |
| 1663 | Löschung | `ich` | `*(nicht da)*` | …um abzugleichen dass [___] auch wirklich alles… |
| 1664 | Löschung | `auch` | `*(nicht da)*` | …abzugleichen dass ich [___] wirklich alles richtig… |
| 1665 | Löschung | `wirklich` | `*(nicht da)*` | …dass ich auch [___] alles richtig notiert… |
| 1666 | Löschung | `alles` | `*(nicht da)*` | …ich auch wirklich [___] richtig notiert habe… |
| 1667 | Löschung | `richtig` | `*(nicht da)*` | …auch wirklich alles [___] notiert habe vorher… |
| 1668 | Löschung | `notiert` | `*(nicht da)*` | …wirklich alles richtig [___] habe vorher möchte… |
| 1669 | Löschung | `habe` | `*(nicht da)*` | …alles richtig notiert [___] vorher möchte ich… |
| 1670 | Löschung | `vorher` | `*(nicht da)*` | …richtig notiert habe [___] möchte ich sie… |
| 1671 | Löschung | `möchte` | `*(nicht da)*` | …notiert habe vorher [___] ich sie aber… |
| 1672 | Löschung | `ich` | `*(nicht da)*` | …habe vorher möchte [___] sie aber noch… |
| 1673 | Löschung | `sie` | `*(nicht da)*` | …vorher möchte ich [___] aber noch kurz… |
| 1674 | Löschung | `aber` | `*(nicht da)*` | …möchte ich sie [___] noch kurz fragen… |
| 1675 | Löschung | `noch` | `*(nicht da)*` | …ich sie aber [___] kurz fragen haben… |
| 1676 | Löschung | `kurz` | `*(nicht da)*` | …sie aber noch [___] fragen haben sie… |
| 1677 | Löschung | `fragen` | `*(nicht da)*` | …aber noch kurz [___] haben sie fragen… |
| 1678 | Löschung | `haben` | `*(nicht da)*` | …noch kurz fragen [___] sie fragen an… |
| 1679 | Löschung | `sie` | `*(nicht da)*` | …kurz fragen haben [___] fragen an mich… |
| 1680 | Löschung | `fragen` | `*(nicht da)*` | …fragen haben sie [___] an mich ja… |
| 1681 | Löschung | `an` | `*(nicht da)*` | …haben sie fragen [___] mich ja eine… |
| 1682 | Löschung | `mich` | `*(nicht da)*` | …sie fragen an [___] ja eine meinen… |
| 1683 | Löschung | `ja` | `*(nicht da)*` | …fragen an mich [___] eine meinen sie… |
| 1684 | Löschung | `eine` | `*(nicht da)*` | …an mich ja [___] meinen sie dass… |
| 1685 | Löschung | `meinen` | `*(nicht da)*` | …mich ja eine [___] sie dass es… |
| 1686 | Löschung | `sie` | `*(nicht da)*` | …ja eine meinen [___] dass es so… |
| 1687 | Löschung | `dass` | `*(nicht da)*` | …eine meinen sie [___] es so sehr… |
| 1688 | Löschung | `es` | `*(nicht da)*` | …meinen sie dass [___] so sehr schlimm… |
| 1689 | Löschung | `so` | `*(nicht da)*` | …sie dass es [___] sehr schlimm wird… |
| 1690 | Löschung | `sehr` | `*(nicht da)*` | …dass es so [___] schlimm wird oder… |
| 1691 | Substitution | `schlimm` | `untersuchung` | …es so sehr [___] wird oder meinen… |
| 1692 | Löschung | `oder` | `*(nicht da)*` | …sehr schlimm wird [___] meinen sie dass… |
| 1693 | Löschung | `meinen` | `*(nicht da)*` | …schlimm wird oder [___] sie dass ich… |
| 1694 | Löschung | `dass` | `*(nicht da)*` | …oder meinen sie [___] ich morgen wieder… |
| 1695 | Löschung | `ich` | `*(nicht da)*` | …meinen sie dass [___] morgen wieder arbeiten… |
| 1696 | Löschung | `morgen` | `*(nicht da)*` | …sie dass ich [___] wieder arbeiten gehen… |
| 1697 | Löschung | `wieder` | `*(nicht da)*` | …dass ich morgen [___] arbeiten gehen kann… |
| 1698 | Löschung | `arbeiten` | `*(nicht da)*` | …ich morgen wieder [___] gehen kann weil… |
| 1699 | Löschung | `gehen` | `*(nicht da)*` | …morgen wieder arbeiten [___] kann weil ich… |
| 1700 | Löschung | `kann` | `*(nicht da)*` | …wieder arbeiten gehen [___] weil ich ein… |
| 1701 | Löschung | `weil` | `*(nicht da)*` | …arbeiten gehen kann [___] ich ein sehr… |
| 1702 | Löschung | `ich` | `*(nicht da)*` | …gehen kann weil [___] ein sehr wichtiges… |
| 1703 | Löschung | `ein` | `*(nicht da)*` | …kann weil ich [___] sehr wichtiges meeting… |
| 1704 | Löschung | `sehr` | `*(nicht da)*` | …weil ich ein [___] wichtiges meeting habe… |
| 1705 | Substitution | `wichtiges` | `von` | …ich ein sehr [___] meeting habe morgen… |
| 1706 | Substitution | `meeting` | `einem` | …ein sehr wichtiges [___] habe morgen und… |
| 1707 | Substitution | `habe` | `arzt` | …sehr wichtiges meeting [___] morgen und wenn… |
| 1708 | Substitution | `morgen` | `untersucht` | …wichtiges meeting habe [___] und wenn ich… |
| 1709 | Löschung | `wenn` | `*(nicht da)*` | …habe morgen und [___] ich nicht selbst… |
| 1710 | Löschung | `ich` | `*(nicht da)*` | …morgen und wenn [___] nicht selbst hingehen… |
| 1711 | Löschung | `nicht` | `*(nicht da)*` | …und wenn ich [___] selbst hingehen kann… |
| 1712 | Löschung | `selbst` | `*(nicht da)*` | …wenn ich nicht [___] hingehen kann dann… |
| 1713 | Löschung | `hingehen` | `*(nicht da)*` | …ich nicht selbst [___] kann dann müsste… |
| 1714 | Löschung | `kann` | `*(nicht da)*` | …nicht selbst hingehen [___] dann müsste ich… |
| 1715 | Löschung | `müsste` | `*(nicht da)*` | …hingehen kann dann [___] ich mich darum… |
| 1716 | Löschung | `ich` | `*(nicht da)*` | …kann dann müsste [___] mich darum kümmern… |
| 1717 | Löschung | `mich` | `*(nicht da)*` | …dann müsste ich [___] darum kümmern und… |
| 1718 | Löschung | `darum` | `*(nicht da)*` | …müsste ich mich [___] kümmern und eine… |
| 1719 | Löschung | `kümmern` | `*(nicht da)*` | …ich mich darum [___] und eine vertretung… |
| 1720 | Löschung | `und` | `*(nicht da)*` | …mich darum kümmern [___] eine vertretung finden… |
| 1721 | Löschung | `eine` | `*(nicht da)*` | …darum kümmern und [___] vertretung finden ich… |
| 1722 | Löschung | `vertretung` | `*(nicht da)*` | …kümmern und eine [___] finden ich werde… |
| 1723 | Löschung | `finden` | `*(nicht da)*` | …und eine vertretung [___] ich werde ihnen… |
| 1724 | Löschung | `ich` | `*(nicht da)*` | …eine vertretung finden [___] werde ihnen jetzt… |
| 1725 | Löschung | `werde` | `*(nicht da)*` | …vertretung finden ich [___] ihnen jetzt etwas… |
| 1726 | Löschung | `ihnen` | `*(nicht da)*` | …finden ich werde [___] jetzt etwas sagen… |
| 1727 | Löschung | `jetzt` | `*(nicht da)*` | …ich werde ihnen [___] etwas sagen was… |
| 1728 | Löschung | `etwas` | `*(nicht da)*` | …werde ihnen jetzt [___] sagen was sie… |
| 1729 | Löschung | `sagen` | `*(nicht da)*` | …ihnen jetzt etwas [___] was sie wahrscheinlich… |
| 1730 | Löschung | `was` | `*(nicht da)*` | …jetzt etwas sagen [___] sie wahrscheinlich nicht… |
| 1731 | Löschung | `sie` | `*(nicht da)*` | …etwas sagen was [___] wahrscheinlich nicht so… |
| 1732 | Löschung | `wahrscheinlich` | `*(nicht da)*` | …sagen was sie [___] nicht so gerne… |
| 1733 | Löschung | `nicht` | `*(nicht da)*` | …was sie wahrscheinlich [___] so gerne hören… |
| 1734 | Löschung | `so` | `*(nicht da)*` | …sie wahrscheinlich nicht [___] gerne hören möchten… |
| 1735 | Löschung | `gerne` | `*(nicht da)*` | …wahrscheinlich nicht so [___] hören möchten aber… |
| 1736 | Löschung | `hören` | `*(nicht da)*` | …nicht so gerne [___] möchten aber da… |
| 1737 | Löschung | `möchten` | `*(nicht da)*` | …so gerne hören [___] aber da kann… |
| 1738 | Löschung | `aber` | `*(nicht da)*` | …gerne hören möchten [___] da kann ich… |
| 1739 | Löschung | `da` | `*(nicht da)*` | …hören möchten aber [___] kann ich ihnen… |
| 1740 | Löschung | `kann` | `*(nicht da)*` | …möchten aber da [___] ich ihnen leider… |
| 1741 | Löschung | `ich` | `*(nicht da)*` | …aber da kann [___] ihnen leider gerade… |
| 1742 | Löschung | `ihnen` | `*(nicht da)*` | …da kann ich [___] leider gerade noch… |
| 1743 | Löschung | `leider` | `*(nicht da)*` | …kann ich ihnen [___] gerade noch keinerlei… |
| 1744 | Löschung | `gerade` | `*(nicht da)*` | …ich ihnen leider [___] noch keinerlei positive… |
| 1745 | Löschung | `noch` | `*(nicht da)*` | …ihnen leider gerade [___] keinerlei positive auskunft… |
| 1746 | Löschung | `keinerlei` | `*(nicht da)*` | …leider gerade noch [___] positive auskunft drüber… |
| 1747 | Löschung | `positive` | `*(nicht da)*` | …gerade noch keinerlei [___] auskunft drüber geben… |
| 1748 | Löschung | `auskunft` | `*(nicht da)*` | …noch keinerlei positive [___] drüber geben das… |
| 1749 | Löschung | `drüber` | `*(nicht da)*` | …keinerlei positive auskunft [___] geben das was… |
| 1750 | Löschung | `geben` | `*(nicht da)*` | …positive auskunft drüber [___] das was sie… |
| 1751 | Löschung | `das` | `*(nicht da)*` | …auskunft drüber geben [___] was sie beschreiben… |
| 1752 | Löschung | `was` | `*(nicht da)*` | …drüber geben das [___] sie beschreiben bezüglich… |
| 1753 | Löschung | `sie` | `*(nicht da)*` | …geben das was [___] beschreiben bezüglich ihres… |
| 1754 | Löschung | `beschreiben` | `*(nicht da)*` | …das was sie [___] bezüglich ihres knies… |
| 1755 | Löschung | `bezüglich` | `*(nicht da)*` | …was sie beschreiben [___] ihres knies und… |
| 1756 | Löschung | `ihres` | `*(nicht da)*` | …sie beschreiben bezüglich [___] knies und auch… |
| 1757 | Löschung | `knies` | `*(nicht da)*` | …beschreiben bezüglich ihres [___] und auch ihres… |
| 1758 | Löschung | `und` | `*(nicht da)*` | …bezüglich ihres knies [___] auch ihres daumens… |
| 1759 | Löschung | `auch` | `*(nicht da)*` | …ihres knies und [___] ihres daumens da… |
| 1760 | Löschung | `ihres` | `*(nicht da)*` | …knies und auch [___] daumens da müssen… |
| 1761 | Löschung | `daumens` | `*(nicht da)*` | …und auch ihres [___] da müssen wir… |
| 1762 | Löschung | `da` | `*(nicht da)*` | …auch ihres daumens [___] müssen wir wirklich… |
| 1763 | Löschung | `müssen` | `*(nicht da)*` | …ihres daumens da [___] wir wirklich erstmal… |
| 1764 | Löschung | `wir` | `*(nicht da)*` | …daumens da müssen [___] wirklich erstmal mrt… |
| 1765 | Löschung | `wirklich` | `*(nicht da)*` | …da müssen wir [___] erstmal mrt bilder… |
| 1766 | Substitution | `erstmal` | `werden` | …müssen wir wirklich [___] mrt bilder von… |
| 1767 | Löschung | `von` | `*(nicht da)*` | …erstmal mrt bilder [___] machen und auch… |
| 1768 | Löschung | `machen` | `*(nicht da)*` | …mrt bilder von [___] und auch röntgenbilder… |
| 1769 | Löschung | `auch` | `*(nicht da)*` | …von machen und [___] röntgenbilder von machen… |
| 1770 | Löschung | `von` | `*(nicht da)*` | …und auch röntgenbilder [___] machen um wirklich… |
| 1771 | Substitution | `machen` | `aufgenommen` | …auch röntgenbilder von [___] um wirklich zu… |
| 1772 | Löschung | `wirklich` | `*(nicht da)*` | …von machen um [___] zu sehen was… |
| 1773 | Löschung | `zu` | `*(nicht da)*` | …machen um wirklich [___] sehen was da… |
| 1774 | Löschung | `sehen` | `*(nicht da)*` | …um wirklich zu [___] was da los… |
| 1775 | Löschung | `was` | `*(nicht da)*` | …wirklich zu sehen [___] da los ist… |
| 1776 | Löschung | `da` | `*(nicht da)*` | …zu sehen was [___] los ist nicht… |
| 1777 | Löschung | `los` | `*(nicht da)*` | …sehen was da [___] ist nicht dass… |
| 1778 | Löschung | `ist` | `*(nicht da)*` | …was da los [___] nicht dass sie… |
| 1779 | Löschung | `nicht` | `*(nicht da)*` | …da los ist [___] dass sie sich… |
| 1780 | Löschung | `dass` | `*(nicht da)*` | …los ist nicht [___] sie sich etwas… |
| 1781 | Löschung | `sie` | `*(nicht da)*` | …ist nicht dass [___] sich etwas gebrochen… |
| 1782 | Substitution | `sich` | `die` | …nicht dass sie [___] etwas gebrochen oder… |
| 1783 | Substitution | `etwas` | `schmerzen` | …dass sie sich [___] gebrochen oder gerissen… |
| 1784 | Substitution | `gebrochen` | `im` | …sie sich etwas [___] oder gerissen haben… |
| 1785 | Substitution | `oder` | `daumen` | …sich etwas gebrochen [___] gerissen haben sollte… |
| 1786 | Substitution | `gerissen` | `und` | …etwas gebrochen oder [___] haben sollte das… |
| 1787 | Substitution | `haben` | `knie` | …gebrochen oder gerissen [___] sollte das der… |
| 1788 | Substitution | `sollte` | `zu` | …oder gerissen haben [___] das der fall… |
| 1789 | Substitution | `das` | `beurteilen` | …gerissen haben sollte [___] der fall sein… |
| 1790 | Löschung | `fall` | `*(nicht da)*` | …sollte das der [___] sein muss man… |
| 1791 | Löschung | `sein` | `*(nicht da)*` | …das der fall [___] muss man abwägen… |
| 1792 | Löschung | `muss` | `*(nicht da)*` | …der fall sein [___] man abwägen ob… |
| 1793 | Löschung | `man` | `*(nicht da)*` | …fall sein muss [___] abwägen ob wir… |
| 1794 | Löschung | `abwägen` | `*(nicht da)*` | …sein muss man [___] ob wir sie… |
| 1795 | Löschung | `ob` | `*(nicht da)*` | …muss man abwägen [___] wir sie operieren… |
| 1796 | Löschung | `wir` | `*(nicht da)*` | …man abwägen ob [___] sie operieren oder… |
| 1797 | Löschung | `sie` | `*(nicht da)*` | …abwägen ob wir [___] operieren oder nicht… |
| 1798 | Löschung | `operieren` | `*(nicht da)*` | …ob wir sie [___] oder nicht das… |
| 1799 | Löschung | `oder` | `*(nicht da)*` | …wir sie operieren [___] nicht das können… |
| 1800 | Löschung | `nicht` | `*(nicht da)*` | …sie operieren oder [___] das können wir… |
| 1801 | Löschung | `das` | `*(nicht da)*` | …operieren oder nicht [___] können wir aber… |
| 1802 | Löschung | `können` | `*(nicht da)*` | …oder nicht das [___] wir aber alles… |
| 1803 | Löschung | `wir` | `*(nicht da)*` | …nicht das können [___] aber alles erst… |
| 1804 | Löschung | `aber` | `*(nicht da)*` | …das können wir [___] alles erst sagen… |
| 1805 | Löschung | `alles` | `*(nicht da)*` | …können wir aber [___] erst sagen wenn… |
| 1806 | Löschung | `erst` | `*(nicht da)*` | …wir aber alles [___] sagen wenn wir… |
| 1807 | Löschung | `sagen` | `*(nicht da)*` | …aber alles erst [___] wenn wir die… |
| 1808 | Löschung | `wenn` | `*(nicht da)*` | …alles erst sagen [___] wir die befunde… |
| 1809 | Löschung | `wir` | `*(nicht da)*` | …erst sagen wenn [___] die befunde da… |
| 1810 | Löschung | `die` | `*(nicht da)*` | …sagen wenn wir [___] befunde da haben… |
| 1811 | Löschung | `befunde` | `*(nicht da)*` | …wenn wir die [___] da haben ich… |
| 1812 | Löschung | `da` | `*(nicht da)*` | …wir die befunde [___] haben ich würde… |
| 1813 | Substitution | `haben` | `arzt` | …die befunde da [___] ich würde auch… |
| 1814 | Substitution | `ich` | `empfiehlt` | …befunde da haben [___] würde auch gerne… |
| 1815 | Substitution | `würde` | `ihr` | …da haben ich [___] auch gerne nochmal… |
| 1816 | Löschung | `gerne` | `*(nicht da)*` | …ich würde auch [___] nochmal eine untersuchung… |
| 1817 | Substitution | `nochmal` | `eine` | …würde auch gerne [___] eine untersuchung mit… |
| 1818 | Substitution | `eine` | `weitere` | …auch gerne nochmal [___] untersuchung mit ihrem… |
| 1819 | Löschung | `durchführen` | `*(nicht da)*` | …mit ihrem kopf [___] um dort auch… |
| 1820 | Löschung | `um` | `*(nicht da)*` | …ihrem kopf durchführen [___] dort auch zu… |
| 1821 | Löschung | `dort` | `*(nicht da)*` | …kopf durchführen um [___] auch zu checken… |
| 1822 | Löschung | `auch` | `*(nicht da)*` | …durchführen um dort [___] zu checken dass… |
| 1823 | Löschung | `zu` | `*(nicht da)*` | …um dort auch [___] checken dass da… |
| 1824 | Löschung | `checken` | `*(nicht da)*` | …dort auch zu [___] dass da eben… |
| 1825 | Substitution | `dass` | `durchzuführen` | …auch zu checken [___] da eben alles… |
| 1826 | Substitution | `da` | `die` | …zu checken dass [___] eben alles in… |
| 1827 | Substitution | `eben` | `frau` | …checken dass da [___] alles in ordnung… |
| 1828 | Substitution | `alles` | `wird` | …dass da eben [___] in ordnung ist… |
| 1829 | Substitution | `in` | `gebeten` | …da eben alles [___] ordnung ist und… |
| 1830 | Substitution | `ordnung` | `sich` | …eben alles in [___] ist und ja… |
| 1831 | Substitution | `ist` | `schonen` | …alles in ordnung [___] und ja wenn… |
| 1832 | Löschung | `ja` | `*(nicht da)*` | …ordnung ist und [___] wenn alles in… |
| 1833 | Löschung | `wenn` | `*(nicht da)*` | …ist und ja [___] alles in ordnung… |
| 1834 | Löschung | `alles` | `*(nicht da)*` | …und ja wenn [___] in ordnung ist… |
| 1835 | Löschung | `in` | `*(nicht da)*` | …ja wenn alles [___] ordnung ist würde… |
| 1836 | Löschung | `ordnung` | `*(nicht da)*` | …wenn alles in [___] ist würde ich… |
| 1837 | Löschung | `ist` | `*(nicht da)*` | …alles in ordnung [___] würde ich ihnen… |
| 1838 | Löschung | `würde` | `*(nicht da)*` | …in ordnung ist [___] ich ihnen trotzdem… |
| 1839 | Löschung | `ich` | `*(nicht da)*` | …ordnung ist würde [___] ihnen trotzdem raten… |
| 1840 | Löschung | `ihnen` | `*(nicht da)*` | …ist würde ich [___] trotzdem raten das… |
| 1841 | Löschung | `trotzdem` | `*(nicht da)*` | …würde ich ihnen [___] raten das meeting… |
| 1842 | Löschung | `raten` | `*(nicht da)*` | …ich ihnen trotzdem [___] das meeting vielleicht… |
| 1843 | Löschung | `vielleicht` | `*(nicht da)*` | …raten das meeting [___] morgen einmal online… |
| 1844 | Löschung | `morgen` | `*(nicht da)*` | …das meeting vielleicht [___] einmal online durchzuführen… |
| 1845 | Löschung | `einmal` | `*(nicht da)*` | …meeting vielleicht morgen [___] online durchzuführen damit… |
| 1846 | Löschung | `damit` | `*(nicht da)*` | …einmal online durchzuführen [___] sie sich schonen… |
| 1847 | Löschung | `sie` | `*(nicht da)*` | …online durchzuführen damit [___] sich schonen können… |
| 1848 | Löschung | `sich` | `*(nicht da)*` | …durchzuführen damit sie [___] schonen können aber… |
| 1849 | Löschung | `schonen` | `*(nicht da)*` | …damit sie sich [___] können aber genauere… |
| 1850 | Löschung | `können` | `*(nicht da)*` | …sie sich schonen [___] aber genauere auskunft… |
| 1851 | Löschung | `aber` | `*(nicht da)*` | …sich schonen können [___] genauere auskunft wie… |
| 1852 | Löschung | `genauere` | `*(nicht da)*` | …schonen können aber [___] auskunft wie bereits… |
| 1853 | Löschung | `auskunft` | `*(nicht da)*` | …können aber genauere [___] wie bereits gesagt… |
| 1854 | Löschung | `wie` | `*(nicht da)*` | …aber genauere auskunft [___] bereits gesagt kann… |
| 1855 | Löschung | `bereits` | `*(nicht da)*` | …genauere auskunft wie [___] gesagt kann ich… |
| 1856 | Löschung | `gesagt` | `*(nicht da)*` | …auskunft wie bereits [___] kann ich ihnen… |
| 1857 | Löschung | `kann` | `*(nicht da)*` | …wie bereits gesagt [___] ich ihnen erst… |
| 1858 | Löschung | `ich` | `*(nicht da)*` | …bereits gesagt kann [___] ihnen erst geben… |
| 1859 | Löschung | `ihnen` | `*(nicht da)*` | …gesagt kann ich [___] erst geben wenn… |
| 1860 | Löschung | `erst` | `*(nicht da)*` | …kann ich ihnen [___] geben wenn wir… |
| 1861 | Löschung | `geben` | `*(nicht da)*` | …ich ihnen erst [___] wenn wir alle… |
| 1862 | Löschung | `wenn` | `*(nicht da)*` | …ihnen erst geben [___] wir alle befunde… |
| 1863 | Löschung | `wir` | `*(nicht da)*` | …erst geben wenn [___] alle befunde da… |
| 1864 | Löschung | `alle` | `*(nicht da)*` | …geben wenn wir [___] befunde da haben… |
| 1865 | Löschung | `befunde` | `*(nicht da)*` | …wenn wir alle [___] da haben okay… |
| 1866 | Löschung | `da` | `*(nicht da)*` | …wir alle befunde [___] haben okay ich… |
| 1867 | Löschung | `haben` | `*(nicht da)*` | …alle befunde da [___] okay ich danke… |
| 1868 | Löschung | `okay` | `*(nicht da)*` | …befunde da haben [___] ich danke ihnen… |
| 1869 | Löschung | `ich` | `*(nicht da)*` | …da haben okay [___] danke ihnen online… |
| 1870 | Löschung | `danke` | `*(nicht da)*` | …haben okay ich [___] ihnen online wäre… |
| 1871 | Löschung | `ihnen` | `*(nicht da)*` | …okay ich danke [___] online wäre schwierig… |
| 1872 | Löschung | `online` | `*(nicht da)*` | …ich danke ihnen [___] wäre schwierig aber… |
| 1873 | Löschung | `wäre` | `*(nicht da)*` | …danke ihnen online [___] schwierig aber dann… |
| 1874 | Löschung | `schwierig` | `*(nicht da)*` | …ihnen online wäre [___] aber dann werde… |
| 1875 | Löschung | `aber` | `*(nicht da)*` | …online wäre schwierig [___] dann werde ich… |
| 1876 | Löschung | `dann` | `*(nicht da)*` | …wäre schwierig aber [___] werde ich jetzt… |
| 1877 | Löschung | `werde` | `*(nicht da)*` | …schwierig aber dann [___] ich jetzt gleich… |
| 1878 | Löschung | `ich` | `*(nicht da)*` | …aber dann werde [___] jetzt gleich dafür… |
| 1879 | Löschung | `jetzt` | `*(nicht da)*` | …dann werde ich [___] gleich dafür sorgen… |
| 1880 | Löschung | `gleich` | `*(nicht da)*` | …werde ich jetzt [___] dafür sorgen dass… |
| 1881 | Löschung | `dafür` | `*(nicht da)*` | …ich jetzt gleich [___] sorgen dass mich… |
| 1882 | Löschung | `sorgen` | `*(nicht da)*` | …jetzt gleich dafür [___] dass mich jemand… |
| 1883 | Löschung | `dass` | `*(nicht da)*` | …gleich dafür sorgen [___] mich jemand vertritt… |
| 1884 | Löschung | `mich` | `*(nicht da)*` | …dafür sorgen dass [___] jemand vertritt okay… |
| 1885 | Löschung | `jemand` | `*(nicht da)*` | …sorgen dass mich [___] vertritt okay sehr… |
| 1886 | Löschung | `vertritt` | `*(nicht da)*` | …dass mich jemand [___] okay sehr gut… |
| 1887 | Löschung | `okay` | `*(nicht da)*` | …mich jemand vertritt [___] sehr gut wir… |
| 1888 | Löschung | `sehr` | `*(nicht da)*` | …jemand vertritt okay [___] gut wir werden… |
| 1889 | Löschung | `gut` | `*(nicht da)*` | …vertritt okay sehr [___] wir werden auch… |
| 1890 | Löschung | `wir` | `*(nicht da)*` | …okay sehr gut [___] werden auch gleich… |
| 1891 | Löschung | `werden` | `*(nicht da)*` | …sehr gut wir [___] auch gleich die… |
| 1892 | Löschung | `auch` | `*(nicht da)*` | …gut wir werden [___] gleich die untersuchungen… |
| 1893 | Löschung | `gleich` | `*(nicht da)*` | …wir werden auch [___] die untersuchungen direkt… |
| 1894 | Löschung | `die` | `*(nicht da)*` | …werden auch gleich [___] untersuchungen direkt durchführen… |
| 1895 | Löschung | `untersuchungen` | `*(nicht da)*` | …auch gleich die [___] direkt durchführen wenn… |
| 1896 | Löschung | `direkt` | `*(nicht da)*` | …gleich die untersuchungen [___] durchführen wenn wir… |
| 1897 | Löschung | `durchführen` | `*(nicht da)*` | …die untersuchungen direkt [___] wenn wir mit… |
| 1898 | Löschung | `wenn` | `*(nicht da)*` | …untersuchungen direkt durchführen [___] wir mit der… |
| 1899 | Löschung | `wir` | `*(nicht da)*` | …direkt durchführen wenn [___] mit der aufnahme… |
| 1900 | Löschung | `mit` | `*(nicht da)*` | …durchführen wenn wir [___] der aufnahme fertig… |
| 1901 | Löschung | `der` | `*(nicht da)*` | …wenn wir mit [___] aufnahme fertig sind… |
| 1902 | Löschung | `aufnahme` | `*(nicht da)*` | …wir mit der [___] fertig sind dann… |
| 1903 | Löschung | `fertig` | `*(nicht da)*` | …mit der aufnahme [___] sind dann würde… |
| 1904 | Löschung | `sind` | `*(nicht da)*` | …der aufnahme fertig [___] dann würde ich… |
| 1905 | Löschung | `dann` | `*(nicht da)*` | …aufnahme fertig sind [___] würde ich sie… |
| 1906 | Löschung | `würde` | `*(nicht da)*` | …fertig sind dann [___] ich sie bitten… |
| 1907 | Löschung | `ich` | `*(nicht da)*` | …sind dann würde [___] sie bitten schon… |
| 1908 | Löschung | `sie` | `*(nicht da)*` | …dann würde ich [___] bitten schon mal… |
| 1909 | Löschung | `bitten` | `*(nicht da)*` | …würde ich sie [___] schon mal rüber… |
| 1910 | Löschung | `schon` | `*(nicht da)*` | …ich sie bitten [___] mal rüber ins… |
| 1911 | Löschung | `mal` | `*(nicht da)*` | …sie bitten schon [___] rüber ins untersuchungszimmer… |
| 1912 | Löschung | `rüber` | `*(nicht da)*` | …bitten schon mal [___] ins untersuchungszimmer zu… |
| 1913 | Löschung | `ins` | `*(nicht da)*` | …schon mal rüber [___] untersuchungszimmer zu gehen… |
| 1914 | Löschung | `untersuchungszimmer` | `*(nicht da)*` | …mal rüber ins [___] zu gehen und… |
| 1915 | Löschung | `zu` | `*(nicht da)*` | …rüber ins untersuchungszimmer [___] gehen und dann… |
| 1916 | Löschung | `gehen` | `*(nicht da)*` | …ins untersuchungszimmer zu [___] und dann geht… |
| 1917 | Löschung | `und` | `*(nicht da)*` | …untersuchungszimmer zu gehen [___] dann geht es… |
| 1918 | Löschung | `dann` | `*(nicht da)*` | …zu gehen und [___] geht es dort… |
| 1919 | Substitution | `geht` | `falls` | …gehen und dann [___] es dort auch… |
| 1920 | Löschung | `dort` | `*(nicht da)*` | …dann geht es [___] auch gleich los… |
| 1921 | Löschung | `auch` | `*(nicht da)*` | …geht es dort [___] gleich los alles… |
| 1922 | Löschung | `gleich` | `*(nicht da)*` | …es dort auch [___] los alles klar… |
| 1923 | Löschung | `los` | `*(nicht da)*` | …dort auch gleich [___] alles klar noch… |
| 1924 | Löschung | `alles` | `*(nicht da)*` | …auch gleich los [___] klar noch mal… |
| 1925 | Löschung | `klar` | `*(nicht da)*` | …gleich los alles [___] noch mal kurz… |
| 1926 | Löschung | `noch` | `*(nicht da)*` | …los alles klar [___] mal kurz zum… |
| 1927 | Löschung | `mal` | `*(nicht da)*` | …alles klar noch [___] kurz zum abgleich… |
| 1928 | Löschung | `kurz` | `*(nicht da)*` | …klar noch mal [___] zum abgleich sie… |
| 1929 | Löschung | `zum` | `*(nicht da)*` | …noch mal kurz [___] abgleich sie hatten… |
| 1930 | Löschung | `abgleich` | `*(nicht da)*` | …mal kurz zum [___] sie hatten einen… |
| 1931 | Löschung | `sie` | `*(nicht da)*` | …kurz zum abgleich [___] hatten einen fahrradunfall… |
| 1932 | Löschung | `hatten` | `*(nicht da)*` | …zum abgleich sie [___] einen fahrradunfall sind… |
| 1933 | Löschung | `einen` | `*(nicht da)*` | …abgleich sie hatten [___] fahrradunfall sind auf… |
| 1934 | Löschung | `fahrradunfall` | `*(nicht da)*` | …sie hatten einen [___] sind auf die… |
| 1935 | Löschung | `sind` | `*(nicht da)*` | …hatten einen fahrradunfall [___] auf die linke… |
| 1936 | Löschung | `auf` | `*(nicht da)*` | …einen fahrradunfall sind [___] die linke seite… |
| 1937 | Löschung | `die` | `*(nicht da)*` | …fahrradunfall sind auf [___] linke seite gestürzt… |
| 1938 | Löschung | `linke` | `*(nicht da)*` | …sind auf die [___] seite gestürzt und… |
| 1939 | Löschung | `seite` | `*(nicht da)*` | …auf die linke [___] gestürzt und haben… |
| 1940 | Löschung | `gestürzt` | `*(nicht da)*` | …die linke seite [___] und haben seitdem… |
| 1941 | Löschung | `und` | `*(nicht da)*` | …linke seite gestürzt [___] haben seitdem schmerzen… |
| 1942 | Löschung | `haben` | `*(nicht da)*` | …seite gestürzt und [___] seitdem schmerzen auf… |
| 1943 | Löschung | `seitdem` | `*(nicht da)*` | …gestürzt und haben [___] schmerzen auf der… |
| 1944 | Löschung | `schmerzen` | `*(nicht da)*` | …und haben seitdem [___] auf der linken… |
| 1945 | Löschung | `auf` | `*(nicht da)*` | …haben seitdem schmerzen [___] der linken kopfseite… |
| 1946 | Löschung | `der` | `*(nicht da)*` | …seitdem schmerzen auf [___] linken kopfseite im… |
| 1947 | Löschung | `linken` | `*(nicht da)*` | …schmerzen auf der [___] kopfseite im linken… |
| 1948 | Löschung | `kopfseite` | `*(nicht da)*` | …auf der linken [___] im linken daumen… |
| 1949 | Löschung | `im` | `*(nicht da)*` | …der linken kopfseite [___] linken daumen sowie… |
| 1950 | Löschung | `linken` | `*(nicht da)*` | …linken kopfseite im [___] daumen sowie im… |
| 1951 | Löschung | `daumen` | `*(nicht da)*` | …kopfseite im linken [___] sowie im linken… |
| 1952 | Löschung | `sowie` | `*(nicht da)*` | …im linken daumen [___] im linken knie… |
| 1953 | Löschung | `im` | `*(nicht da)*` | …linken daumen sowie [___] linken knie die… |
| 1954 | Löschung | `linken` | `*(nicht da)*` | …daumen sowie im [___] knie die schmerzen… |
| 1955 | Löschung | `knie` | `*(nicht da)*` | …sowie im linken [___] die schmerzen im… |
| 1956 | Löschung | `die` | `*(nicht da)*` | …im linken knie [___] schmerzen im kopf… |
| 1957 | Löschung | `schmerzen` | `*(nicht da)*` | …linken knie die [___] im kopf sind… |
| 1958 | Löschung | `im` | `*(nicht da)*` | …knie die schmerzen [___] kopf sind sehr… |
| 1959 | Löschung | `kopf` | `*(nicht da)*` | …die schmerzen im [___] sind sehr leicht… |
| 1960 | Löschung | `sind` | `*(nicht da)*` | …schmerzen im kopf [___] sehr leicht die… |
| 1961 | Löschung | `sehr` | `*(nicht da)*` | …im kopf sind [___] leicht die schmerzen… |
| 1962 | Löschung | `leicht` | `*(nicht da)*` | …kopf sind sehr [___] die schmerzen im… |
| 1963 | Löschung | `die` | `*(nicht da)*` | …sind sehr leicht [___] schmerzen im daumen… |
| 1964 | Löschung | `schmerzen` | `*(nicht da)*` | …sehr leicht die [___] im daumen dagegen… |
| 1965 | Löschung | `im` | `*(nicht da)*` | …leicht die schmerzen [___] daumen dagegen schon… |
| 1966 | Löschung | `daumen` | `*(nicht da)*` | …die schmerzen im [___] dagegen schon wesentlich… |
| 1967 | Löschung | `dagegen` | `*(nicht da)*` | …schmerzen im daumen [___] schon wesentlich stärker… |
| 1968 | Löschung | `schon` | `*(nicht da)*` | …im daumen dagegen [___] wesentlich stärker sie… |
| 1969 | Löschung | `wesentlich` | `*(nicht da)*` | …daumen dagegen schon [___] stärker sie haben… |
| 1970 | Löschung | `stärker` | `*(nicht da)*` | …dagegen schon wesentlich [___] sie haben die… |
| 1971 | Löschung | `sie` | `*(nicht da)*` | …schon wesentlich stärker [___] haben die schmerzintensität… |
| 1972 | Löschung | `haben` | `*(nicht da)*` | …wesentlich stärker sie [___] die schmerzintensität dort… |
| 1973 | Löschung | `die` | `*(nicht da)*` | …stärker sie haben [___] schmerzintensität dort mit… |
| 1974 | Löschung | `schmerzintensität` | `*(nicht da)*` | …sie haben die [___] dort mit einer… |
| 1975 | Löschung | `dort` | `*(nicht da)*` | …haben die schmerzintensität [___] mit einer 7… |
| 1976 | Löschung | `mit` | `*(nicht da)*` | …die schmerzintensität dort [___] einer 7 beschrieben… |
| 1977 | Löschung | `einer` | `*(nicht da)*` | …schmerzintensität dort mit [___] 7 beschrieben und… |
| 1978 | Löschung | `7` | `*(nicht da)*` | …dort mit einer [___] beschrieben und haben… |
| 1979 | Löschung | `beschrieben` | `*(nicht da)*` | …mit einer 7 [___] und haben gesagt… |
| 1980 | Löschung | `und` | `*(nicht da)*` | …einer 7 beschrieben [___] haben gesagt dass… |
| 1981 | Löschung | `haben` | `*(nicht da)*` | …7 beschrieben und [___] gesagt dass sie… |
| 1982 | Löschung | `gesagt` | `*(nicht da)*` | …beschrieben und haben [___] dass sie den… |
| 1983 | Löschung | `dass` | `*(nicht da)*` | …und haben gesagt [___] sie den daumen… |
| 1984 | Löschung | `sie` | `*(nicht da)*` | …haben gesagt dass [___] den daumen auch… |
| 1985 | Löschung | `den` | `*(nicht da)*` | …gesagt dass sie [___] daumen auch nicht… |
| 1986 | Löschung | `daumen` | `*(nicht da)*` | …dass sie den [___] auch nicht mehr… |
| 1987 | Löschung | `auch` | `*(nicht da)*` | …sie den daumen [___] nicht mehr bewegen… |
| 1988 | Löschung | `nicht` | `*(nicht da)*` | …den daumen auch [___] mehr bewegen können… |
| 1989 | Löschung | `mehr` | `*(nicht da)*` | …daumen auch nicht [___] bewegen können der… |
| 1990 | Löschung | `bewegen` | `*(nicht da)*` | …auch nicht mehr [___] können der schmerz… |
| 1991 | Löschung | `können` | `*(nicht da)*` | …nicht mehr bewegen [___] der schmerz wurde… |
| 1992 | Löschung | `der` | `*(nicht da)*` | …mehr bewegen können [___] schmerz wurde stechend… |
| 1993 | Löschung | `schmerz` | `*(nicht da)*` | …bewegen können der [___] wurde stechend beschrieben… |
| 1994 | Löschung | `wurde` | `*(nicht da)*` | …können der schmerz [___] stechend beschrieben und… |
| 1995 | Löschung | `stechend` | `*(nicht da)*` | …der schmerz wurde [___] beschrieben und gleiches… |
| 1996 | Löschung | `beschrieben` | `*(nicht da)*` | …schmerz wurde stechend [___] und gleiches gilt… |
| 1997 | Löschung | `und` | `*(nicht da)*` | …wurde stechend beschrieben [___] gleiches gilt für… |
| 1998 | Löschung | `gleiches` | `*(nicht da)*` | …stechend beschrieben und [___] gilt für das… |
| 1999 | Löschung | `gilt` | `*(nicht da)*` | …beschrieben und gleiches [___] für das knie… |
| 2000 | Löschung | `für` | `*(nicht da)*` | …und gleiches gilt [___] das knie auch… |
| 2001 | Löschung | `das` | `*(nicht da)*` | …gleiches gilt für [___] knie auch das… |
| 2002 | Löschung | `knie` | `*(nicht da)*` | …gilt für das [___] auch das knie… |
| 2003 | Löschung | `auch` | `*(nicht da)*` | …für das knie [___] das knie können… |
| 2004 | Löschung | `das` | `*(nicht da)*` | …das knie auch [___] knie können sie… |
| 2005 | Löschung | `knie` | `*(nicht da)*` | …knie auch das [___] können sie nicht… |
| 2006 | Löschung | `können` | `*(nicht da)*` | …auch das knie [___] sie nicht mehr… |
| 2007 | Löschung | `sie` | `*(nicht da)*` | …das knie können [___] nicht mehr bewegen… |
| 2008 | Löschung | `nicht` | `*(nicht da)*` | …knie können sie [___] mehr bewegen im… |
| 2009 | Löschung | `mehr` | `*(nicht da)*` | …können sie nicht [___] bewegen im ruhezustand… |
| 2010 | Löschung | `bewegen` | `*(nicht da)*` | …sie nicht mehr [___] im ruhezustand wurde… |
| 2011 | Löschung | `im` | `*(nicht da)*` | …nicht mehr bewegen [___] ruhezustand wurde die… |
| 2012 | Löschung | `ruhezustand` | `*(nicht da)*` | …mehr bewegen im [___] wurde die schmerzintensität… |
| 2013 | Löschung | `wurde` | `*(nicht da)*` | …bewegen im ruhezustand [___] die schmerzintensität mit… |
| 2014 | Löschung | `die` | `*(nicht da)*` | …im ruhezustand wurde [___] schmerzintensität mit einer… |
| 2015 | Löschung | `schmerzintensität` | `*(nicht da)*` | …ruhezustand wurde die [___] mit einer 8… |
| 2016 | Löschung | `mit` | `*(nicht da)*` | …wurde die schmerzintensität [___] einer 8 beschrieben… |
| 2017 | Löschung | `einer` | `*(nicht da)*` | …die schmerzintensität mit [___] 8 beschrieben bei… |
| 2018 | Löschung | `8` | `*(nicht da)*` | …schmerzintensität mit einer [___] beschrieben bei bewegung… |
| 2019 | Löschung | `beschrieben` | `*(nicht da)*` | …mit einer 8 [___] bei bewegung unerträglich… |
| 2020 | Löschung | `bei` | `*(nicht da)*` | …einer 8 beschrieben [___] bewegung unerträglich also… |
| 2021 | Löschung | `bewegung` | `*(nicht da)*` | …8 beschrieben bei [___] unerträglich also 10… |
| 2022 | Löschung | `unerträglich` | `*(nicht da)*` | …beschrieben bei bewegung [___] also 10 oder… |
| 2023 | Löschung | `also` | `*(nicht da)*` | …bei bewegung unerträglich [___] 10 oder mehr… |
| 2024 | Löschung | `10` | `*(nicht da)*` | …bewegung unerträglich also [___] oder mehr als… |
| 2025 | Löschung | `oder` | `*(nicht da)*` | …unerträglich also 10 [___] mehr als 10… |
| 2026 | Löschung | `mehr` | `*(nicht da)*` | …also 10 oder [___] als 10 auch… |
| 2027 | Löschung | `als` | `*(nicht da)*` | …10 oder mehr [___] 10 auch dieser… |
| 2028 | Löschung | `10` | `*(nicht da)*` | …oder mehr als [___] auch dieser schmerz… |
| 2029 | Löschung | `auch` | `*(nicht da)*` | …mehr als 10 [___] dieser schmerz ist… |
| 2030 | Löschung | `dieser` | `*(nicht da)*` | …als 10 auch [___] schmerz ist stechend… |
| 2031 | Löschung | `schmerz` | `*(nicht da)*` | …10 auch dieser [___] ist stechend ansonsten… |
| 2032 | Löschung | `ist` | `*(nicht da)*` | …auch dieser schmerz [___] stechend ansonsten sind… |
| 2033 | Löschung | `stechend` | `*(nicht da)*` | …dieser schmerz ist [___] ansonsten sind daumen… |
| 2034 | Löschung | `ansonsten` | `*(nicht da)*` | …schmerz ist stechend [___] sind daumen wie… |
| 2035 | Löschung | `sind` | `*(nicht da)*` | …ist stechend ansonsten [___] daumen wie auch… |
| 2036 | Löschung | `daumen` | `*(nicht da)*` | …stechend ansonsten sind [___] wie auch knie… |
| 2037 | Löschung | `wie` | `*(nicht da)*` | …ansonsten sind daumen [___] auch knie geschwollen… |
| 2038 | Löschung | `auch` | `*(nicht da)*` | …sind daumen wie [___] knie geschwollen richtig… |
| 2039 | Löschung | `knie` | `*(nicht da)*` | …daumen wie auch [___] geschwollen richtig sie… |
| 2040 | Löschung | `geschwollen` | `*(nicht da)*` | …wie auch knie [___] richtig sie haben… |
| 2041 | Löschung | `richtig` | `*(nicht da)*` | …auch knie geschwollen [___] sie haben gesagt… |
| 2042 | Löschung | `sie` | `*(nicht da)*` | …knie geschwollen richtig [___] haben gesagt dass… |
| 2043 | Löschung | `haben` | `*(nicht da)*` | …geschwollen richtig sie [___] gesagt dass sie… |
| 2044 | Löschung | `gesagt` | `*(nicht da)*` | …richtig sie haben [___] dass sie das… |
| 2045 | Löschung | `dass` | `*(nicht da)*` | …sie haben gesagt [___] sie das bewusstsein… |
| 2046 | Löschung | `sie` | `*(nicht da)*` | …haben gesagt dass [___] das bewusstsein nicht… |
| 2047 | Löschung | `das` | `*(nicht da)*` | …gesagt dass sie [___] bewusstsein nicht verloren… |
| 2048 | Löschung | `bewusstsein` | `*(nicht da)*` | …dass sie das [___] nicht verloren haben… |
| 2049 | Löschung | `verloren` | `*(nicht da)*` | …das bewusstsein nicht [___] haben bei dem… |
| 2050 | Löschung | `haben` | `*(nicht da)*` | …bewusstsein nicht verloren [___] bei dem unfall… |
| 2051 | Löschung | `bei` | `*(nicht da)*` | …nicht verloren haben [___] dem unfall dass… |
| 2052 | Löschung | `dem` | `*(nicht da)*` | …verloren haben bei [___] unfall dass sie… |
| 2053 | Löschung | `unfall` | `*(nicht da)*` | …haben bei dem [___] dass sie nur… |
| 2054 | Löschung | `dass` | `*(nicht da)*` | …bei dem unfall [___] sie nur kurz… |
| 2055 | Löschung | `sie` | `*(nicht da)*` | …dem unfall dass [___] nur kurz danach… |
| 2056 | Löschung | `nur` | `*(nicht da)*` | …unfall dass sie [___] kurz danach recht… |
| 2057 | Löschung | `kurz` | `*(nicht da)*` | …dass sie nur [___] danach recht schwindelig… |
| 2058 | Löschung | `danach` | `*(nicht da)*` | …sie nur kurz [___] recht schwindelig waren… |
| 2059 | Löschung | `recht` | `*(nicht da)*` | …nur kurz danach [___] schwindelig waren das… |
| 2060 | Löschung | `schwindelig` | `*(nicht da)*` | …kurz danach recht [___] waren das sei… |
| 2061 | Löschung | `waren` | `*(nicht da)*` | …danach recht schwindelig [___] das sei aber… |
| 2062 | Löschung | `das` | `*(nicht da)*` | …recht schwindelig waren [___] sei aber schon… |
| 2063 | Löschung | `sei` | `*(nicht da)*` | …schwindelig waren das [___] aber schon wieder… |
| 2064 | Löschung | `aber` | `*(nicht da)*` | …waren das sei [___] schon wieder vorbei… |
| 2065 | Löschung | `schon` | `*(nicht da)*` | …das sei aber [___] wieder vorbei genau… |
| 2066 | Löschung | `wieder` | `*(nicht da)*` | …sei aber schon [___] vorbei genau vorerkrankungen… |
| 2067 | Löschung | `vorbei` | `*(nicht da)*` | …aber schon wieder [___] genau vorerkrankungen haben… |
| 2068 | Löschung | `genau` | `*(nicht da)*` | …schon wieder vorbei [___] vorerkrankungen haben sie… |
| 2069 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …wieder vorbei genau [___] haben sie keine… |
| 2070 | Löschung | `haben` | `*(nicht da)*` | …vorbei genau vorerkrankungen [___] sie keine medikamente… |
| 2071 | Löschung | `sie` | `*(nicht da)*` | …genau vorerkrankungen haben [___] keine medikamente nehmen… |
| 2072 | Löschung | `keine` | `*(nicht da)*` | …vorerkrankungen haben sie [___] medikamente nehmen sie… |
| 2073 | Löschung | `medikamente` | `*(nicht da)*` | …haben sie keine [___] nehmen sie auch… |
| 2074 | Löschung | `nehmen` | `*(nicht da)*` | …sie keine medikamente [___] sie auch keine… |
| 2075 | Löschung | `sie` | `*(nicht da)*` | …keine medikamente nehmen [___] auch keine regelmäßig… |
| 2076 | Löschung | `auch` | `*(nicht da)*` | …medikamente nehmen sie [___] keine regelmäßig ein… |
| 2077 | Löschung | `keine` | `*(nicht da)*` | …nehmen sie auch [___] regelmäßig ein außer… |
| 2078 | Löschung | `regelmäßig` | `*(nicht da)*` | …sie auch keine [___] ein außer der… |
| 2079 | Löschung | `ein` | `*(nicht da)*` | …auch keine regelmäßig [___] außer der pille… |
| 2080 | Löschung | `außer` | `*(nicht da)*` | …keine regelmäßig ein [___] der pille sie… |
| 2081 | Löschung | `der` | `*(nicht da)*` | …regelmäßig ein außer [___] pille sie hatten… |
| 2082 | Löschung | `pille` | `*(nicht da)*` | …ein außer der [___] sie hatten eine… |
| 2083 | Löschung | `sie` | `*(nicht da)*` | …außer der pille [___] hatten eine operation… |
| 2084 | Löschung | `hatten` | `*(nicht da)*` | …der pille sie [___] eine operation am… |
| 2085 | Löschung | `eine` | `*(nicht da)*` | …pille sie hatten [___] operation am rechten… |
| 2086 | Löschung | `operation` | `*(nicht da)*` | …sie hatten eine [___] am rechten fuß… |
| 2087 | Löschung | `am` | `*(nicht da)*` | …hatten eine operation [___] rechten fuß vor… |
| 2088 | Löschung | `rechten` | `*(nicht da)*` | …eine operation am [___] fuß vor zwei… |
| 2089 | Löschung | `fuß` | `*(nicht da)*` | …operation am rechten [___] vor zwei jahren… |
| 2090 | Löschung | `vor` | `*(nicht da)*` | …am rechten fuß [___] zwei jahren da… |
| 2091 | Löschung | `zwei` | `*(nicht da)*` | …rechten fuß vor [___] jahren da wurde… |
| 2092 | Löschung | `jahren` | `*(nicht da)*` | …fuß vor zwei [___] da wurde der… |
| 2093 | Löschung | `da` | `*(nicht da)*` | …vor zwei jahren [___] wurde der halux… |
| 2094 | Löschung | `wurde` | `*(nicht da)*` | …zwei jahren da [___] der halux valgus… |
| 2095 | Löschung | `der` | `*(nicht da)*` | …jahren da wurde [___] halux valgus operiert… |
| 2096 | Löschung | `halux` | `*(nicht da)*` | …da wurde der [___] valgus operiert ansonsten… |
| 2097 | Löschung | `valgus` | `*(nicht da)*` | …wurde der halux [___] operiert ansonsten körperliche… |
| 2098 | Löschung | `operiert` | `*(nicht da)*` | …der halux valgus [___] ansonsten körperliche beschwerden… |
| 2099 | Löschung | `ansonsten` | `*(nicht da)*` | …halux valgus operiert [___] körperliche beschwerden gibt… |
| 2100 | Löschung | `körperliche` | `*(nicht da)*` | …valgus operiert ansonsten [___] beschwerden gibt es… |
| 2101 | Löschung | `beschwerden` | `*(nicht da)*` | …operiert ansonsten körperliche [___] gibt es keine… |
| 2102 | Löschung | `gibt` | `*(nicht da)*` | …ansonsten körperliche beschwerden [___] es keine sie… |
| 2103 | Löschung | `es` | `*(nicht da)*` | …körperliche beschwerden gibt [___] keine sie sind… |
| 2104 | Löschung | `keine` | `*(nicht da)*` | …beschwerden gibt es [___] sie sind ansonsten… |
| 2105 | Löschung | `sie` | `*(nicht da)*` | …gibt es keine [___] sind ansonsten gesund… |
| 2106 | Löschung | `sind` | `*(nicht da)*` | …es keine sie [___] ansonsten gesund gott… |
| 2107 | Löschung | `ansonsten` | `*(nicht da)*` | …keine sie sind [___] gesund gott sei… |
| 2108 | Löschung | `gesund` | `*(nicht da)*` | …sie sind ansonsten [___] gott sei dank… |
| 2109 | Löschung | `gott` | `*(nicht da)*` | …sind ansonsten gesund [___] sei dank bis… |
| 2110 | Löschung | `sei` | `*(nicht da)*` | …ansonsten gesund gott [___] dank bis auf… |
| 2111 | Löschung | `dank` | `*(nicht da)*` | …gesund gott sei [___] bis auf die… |
| 2112 | Löschung | `bis` | `*(nicht da)*` | …gott sei dank [___] auf die kistaminunverträglichkeit… |
| 2113 | Löschung | `auf` | `*(nicht da)*` | …sei dank bis [___] die kistaminunverträglichkeit genau… |
| 2114 | Löschung | `die` | `*(nicht da)*` | …dank bis auf [___] kistaminunverträglichkeit genau das… |
| 2115 | Löschung | `kistaminunverträglichkeit` | `*(nicht da)*` | …bis auf die [___] genau das hätte… |
| 2116 | Löschung | `genau` | `*(nicht da)*` | …auf die kistaminunverträglichkeit [___] das hätte ich… |
| 2117 | Löschung | `das` | `*(nicht da)*` | …die kistaminunverträglichkeit genau [___] hätte ich jetzt… |
| 2118 | Löschung | `hätte` | `*(nicht da)*` | …kistaminunverträglichkeit genau das [___] ich jetzt auch… |
| 2119 | Löschung | `ich` | `*(nicht da)*` | …genau das hätte [___] jetzt auch noch… |
| 2120 | Löschung | `jetzt` | `*(nicht da)*` | …das hätte ich [___] auch noch mit… |
| 2121 | Löschung | `auch` | `*(nicht da)*` | …hätte ich jetzt [___] noch mit eingebracht… |
| 2122 | Löschung | `noch` | `*(nicht da)*` | …ich jetzt auch [___] mit eingebracht vielen… |
| 2123 | Löschung | `mit` | `*(nicht da)*` | …jetzt auch noch [___] eingebracht vielen dank… |
| 2124 | Löschung | `eingebracht` | `*(nicht da)*` | …auch noch mit [___] vielen dank nochmal… |
| 2125 | Löschung | `vielen` | `*(nicht da)*` | …noch mit eingebracht [___] dank nochmal dafür… |
| 2126 | Löschung | `dank` | `*(nicht da)*` | …mit eingebracht vielen [___] nochmal dafür habe… |
| 2127 | Löschung | `nochmal` | `*(nicht da)*` | …eingebracht vielen dank [___] dafür habe ich… |
| 2128 | Löschung | `dafür` | `*(nicht da)*` | …vielen dank nochmal [___] habe ich mir… |
| 2129 | Löschung | `habe` | `*(nicht da)*` | …dank nochmal dafür [___] ich mir notiert… |
| 2130 | Löschung | `ich` | `*(nicht da)*` | …nochmal dafür habe [___] mir notiert genau… |
| 2131 | Löschung | `mir` | `*(nicht da)*` | …dafür habe ich [___] notiert genau es… |
| 2132 | Löschung | `notiert` | `*(nicht da)*` | …habe ich mir [___] genau es gibt… |
| 2133 | Löschung | `genau` | `*(nicht da)*` | …ich mir notiert [___] es gibt ein… |
| 2134 | Löschung | `es` | `*(nicht da)*` | …mir notiert genau [___] gibt ein paar… |
| 2135 | Löschung | `gibt` | `*(nicht da)*` | …notiert genau es [___] ein paar vorerkrankungen… |
| 2136 | Löschung | `ein` | `*(nicht da)*` | …genau es gibt [___] paar vorerkrankungen in… |
| 2137 | Löschung | `paar` | `*(nicht da)*` | …es gibt ein [___] vorerkrankungen in der… |
| 2138 | Löschung | `vorerkrankungen` | `*(nicht da)*` | …gibt ein paar [___] in der familiengeschichte… |
| 2139 | Löschung | `in` | `*(nicht da)*` | …ein paar vorerkrankungen [___] der familiengeschichte sie… |
| 2140 | Löschung | `der` | `*(nicht da)*` | …paar vorerkrankungen in [___] familiengeschichte sie sind… |
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
| 2152 | Substitution | `das` | `möglich` | …einer marketingagentur ja [___] ist alles richtig… |
| 2153 | Löschung | `alles` | `*(nicht da)*` | …ja das ist [___] richtig ja perfekt… |
| 2154 | Löschung | `richtig` | `*(nicht da)*` | …das ist alles [___] ja perfekt sehr… |
| 2155 | Löschung | `ja` | `*(nicht da)*` | …ist alles richtig [___] perfekt sehr gut… |
| 2156 | Löschung | `perfekt` | `*(nicht da)*` | …alles richtig ja [___] sehr gut frau… |
| 2157 | Löschung | `sehr` | `*(nicht da)*` | …richtig ja perfekt [___] gut frau becken… |
| 2158 | Löschung | `gut` | `*(nicht da)*` | …ja perfekt sehr [___] frau becken westfalen… |
| 2159 | Löschung | `frau` | `*(nicht da)*` | …perfekt sehr gut [___] becken westfalen dann… |
| 2160 | Löschung | `becken` | `*(nicht da)*` | …sehr gut frau [___] westfalen dann war… |
| 2161 | Löschung | `westfalen` | `*(nicht da)*` | …gut frau becken [___] dann war es… |
| 2162 | Löschung | `dann` | `*(nicht da)*` | …frau becken westfalen [___] war es das… |
| 2163 | Löschung | `war` | `*(nicht da)*` | …becken westfalen dann [___] es das jetzt… |
| 2164 | Löschung | `es` | `*(nicht da)*` | …westfalen dann war [___] das jetzt erstmal… |
| 2165 | Löschung | `das` | `*(nicht da)*` | …dann war es [___] jetzt erstmal von… |
| 2166 | Löschung | `jetzt` | `*(nicht da)*` | …war es das [___] erstmal von meiner… |
| 2167 | Löschung | `erstmal` | `*(nicht da)*` | …es das jetzt [___] von meiner seite… |
| 2168 | Löschung | `von` | `*(nicht da)*` | …das jetzt erstmal [___] meiner seite wir… |
| 2169 | Löschung | `meiner` | `*(nicht da)*` | …jetzt erstmal von [___] seite wir machen… |
| 2170 | Löschung | `seite` | `*(nicht da)*` | …erstmal von meiner [___] wir machen jetzt… |
| 2171 | Löschung | `wir` | `*(nicht da)*` | …von meiner seite [___] machen jetzt mit… |
| 2172 | Löschung | `machen` | `*(nicht da)*` | …meiner seite wir [___] jetzt mit den… |
| 2173 | Löschung | `jetzt` | `*(nicht da)*` | …seite wir machen [___] mit den untersuchungen… |
| 2174 | Löschung | `mit` | `*(nicht da)*` | …wir machen jetzt [___] den untersuchungen weiter… |
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
| 2202 | Substitution | `gleich` | `jemanden` | …dank und bis [___] bis gleich… |
| 2203 | Substitution | `bis` | `zu` | …und bis gleich [___] gleich… |
| 2204 | Substitution | `gleich` | `vertritten` | …bis gleich bis [___]… |

---

## PWC

**Fehlerrate: 46.5%** — RAW: 1512 Wörter | FMT: 1005 Wörter | S=20 D=595 I=88 | Fehler=703

| # | Typ | RAW | FORMATTED | Kontext |
|---|-----|-----|-----------|---------|
| 1 | Einfügung | `*(nicht da)*` | `das` | (FMT) …[___] transkript ist ein… |
| 2 | Einfügung | `*(nicht da)*` | `transkript` | (FMT) …das [___] ist ein arzt… |
| 3 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …das transkript [___] ein arzt patienten… |
| 4 | Einfügung | `*(nicht da)*` | `ein` | (FMT) …das transkript ist [___] arzt patienten gespräch… |
| 5 | Einfügung | `*(nicht da)*` | `arzt` | (FMT) …transkript ist ein [___] patienten gespräch zwischen… |
| 6 | Einfügung | `*(nicht da)*` | `patienten` | (FMT) …ist ein arzt [___] gespräch zwischen der… |
| 7 | Einfügung | `*(nicht da)*` | `gespräch` | (FMT) …ein arzt patienten [___] zwischen der physiotherapeutin… |
| 8 | Einfügung | `*(nicht da)*` | `zwischen` | (FMT) …arzt patienten gespräch [___] der physiotherapeutin eisley… |
| 9 | Einfügung | `*(nicht da)*` | `der` | (FMT) …patienten gespräch zwischen [___] physiotherapeutin eisley teresa… |
| 10 | Einfügung | `*(nicht da)*` | `physiotherapeutin` | (FMT) …gespräch zwischen der [___] eisley teresa und… |
| 11 | Einfügung | `*(nicht da)*` | `eisley` | (FMT) …zwischen der physiotherapeutin [___] teresa und frau… |
| 12 | Einfügung | `*(nicht da)*` | `teresa` | (FMT) …der physiotherapeutin eisley [___] und frau grasbeutner… |
| 13 | Einfügung | `*(nicht da)*` | `und` | (FMT) …physiotherapeutin eisley teresa [___] frau grasbeutner oder… |
| 14 | Einfügung | `*(nicht da)*` | `frau` | (FMT) …eisley teresa und [___] grasbeutner oder krebspartner… |
| 15 | Einfügung | `*(nicht da)*` | `grasbeutner` | (FMT) …teresa und frau [___] oder krebspartner was… |
| 16 | Einfügung | `*(nicht da)*` | `oder` | (FMT) …und frau grasbeutner [___] krebspartner was wahrscheinlich… |
| 17 | Einfügung | `*(nicht da)*` | `krebspartner` | (FMT) …frau grasbeutner oder [___] was wahrscheinlich ein… |
| 18 | Einfügung | `*(nicht da)*` | `was` | (FMT) …grasbeutner oder krebspartner [___] wahrscheinlich ein missverständnis… |
| 19 | Einfügung | `*(nicht da)*` | `wahrscheinlich` | (FMT) …oder krebspartner was [___] ein missverständnis ist… |
| 20 | Einfügung | `*(nicht da)*` | `ein` | (FMT) …krebspartner was wahrscheinlich [___] missverständnis ist frau… |
| 21 | Einfügung | `*(nicht da)*` | `missverständnis` | (FMT) …was wahrscheinlich ein [___] ist frau grasbeutner… |
| 22 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …wahrscheinlich ein missverständnis [___] frau grasbeutner ist… |
| 23 | Einfügung | `*(nicht da)*` | `frau` | (FMT) …ein missverständnis ist [___] grasbeutner ist 27… |
| 24 | Einfügung | `*(nicht da)*` | `grasbeutner` | (FMT) …missverständnis ist frau [___] ist 27 jahre… |
| 25 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …ist frau grasbeutner [___] 27 jahre alt… |
| 26 | Einfügung | `*(nicht da)*` | `27` | (FMT) …frau grasbeutner ist [___] jahre alt und… |
| 27 | Einfügung | `*(nicht da)*` | `jahre` | (FMT) …grasbeutner ist 27 [___] alt und arbeitet… |
| 28 | Einfügung | `*(nicht da)*` | `alt` | (FMT) …ist 27 jahre [___] und arbeitet als… |
| 29 | Einfügung | `*(nicht da)*` | `und` | (FMT) …27 jahre alt [___] arbeitet als büroangestellte… |
| 30 | Einfügung | `*(nicht da)*` | `arbeitet` | (FMT) …jahre alt und [___] als büroangestellte sie… |
| 31 | Einfügung | `*(nicht da)*` | `als` | (FMT) …alt und arbeitet [___] büroangestellte sie hat… |
| 32 | Einfügung | `*(nicht da)*` | `büroangestellte` | (FMT) …und arbeitet als [___] sie hat sich… |
| 33 | Einfügung | `*(nicht da)*` | `sie` | (FMT) …arbeitet als büroangestellte [___] hat sich vor… |
| 34 | Einfügung | `*(nicht da)*` | `hat` | (FMT) …als büroangestellte sie [___] sich vor einem… |
| 35 | Einfügung | `*(nicht da)*` | `sich` | (FMT) …büroangestellte sie hat [___] vor einem monat… |
| 36 | Einfügung | `*(nicht da)*` | `vor` | (FMT) …sie hat sich [___] einem monat einen… |
| 37 | Einfügung | `*(nicht da)*` | `einem` | (FMT) …hat sich vor [___] monat einen kreuzkontress… |
| 38 | Einfügung | `*(nicht da)*` | `monat` | (FMT) …sich vor einem [___] einen kreuzkontress zugezogen… |
| 39 | Einfügung | `*(nicht da)*` | `einen` | (FMT) …vor einem monat [___] kreuzkontress zugezogen als… |
| 40 | Einfügung | `*(nicht da)*` | `kreuzkontress` | (FMT) …einem monat einen [___] zugezogen als sie… |
| 41 | Einfügung | `*(nicht da)*` | `zugezogen` | (FMT) …monat einen kreuzkontress [___] als sie beim… |
| 42 | Einfügung | `*(nicht da)*` | `als` | (FMT) …einen kreuzkontress zugezogen [___] sie beim volleyballtraining… |
| 43 | Einfügung | `*(nicht da)*` | `sie` | (FMT) …kreuzkontress zugezogen als [___] beim volleyballtraining sprang… |
| 44 | Einfügung | `*(nicht da)*` | `beim` | (FMT) …zugezogen als sie [___] volleyballtraining sprang und… |
| 45 | Einfügung | `*(nicht da)*` | `volleyballtraining` | (FMT) …als sie beim [___] sprang und schief… |
| 46 | Einfügung | `*(nicht da)*` | `sprang` | (FMT) …sie beim volleyballtraining [___] und schief landete… |
| 47 | Einfügung | `*(nicht da)*` | `und` | (FMT) …beim volleyballtraining sprang [___] schief landete um… |
| 48 | Einfügung | `*(nicht da)*` | `schief` | (FMT) …volleyballtraining sprang und [___] landete um das… |
| 49 | Einfügung | `*(nicht da)*` | `landete` | (FMT) …sprang und schief [___] um das transkript… |
| 50 | Einfügung | `*(nicht da)*` | `um` | (FMT) …und schief landete [___] das transkript zu… |
| 51 | Einfügung | `*(nicht da)*` | `das` | (FMT) …schief landete um [___] transkript zu formatieren… |
| 52 | Einfügung | `*(nicht da)*` | `transkript` | (FMT) …landete um das [___] zu formatieren werde… |
| 53 | Einfügung | `*(nicht da)*` | `zu` | (FMT) …um das transkript [___] formatieren werde ich… |
| 54 | Einfügung | `*(nicht da)*` | `formatieren` | (FMT) …das transkript zu [___] werde ich die… |
| 55 | Einfügung | `*(nicht da)*` | `werde` | (FMT) …transkript zu formatieren [___] ich die generischen… |
| 56 | Einfügung | `*(nicht da)*` | `ich` | (FMT) …zu formatieren werde [___] die generischen sprecher… |
| 57 | Einfügung | `*(nicht da)*` | `die` | (FMT) …formatieren werde ich [___] generischen sprecher labels… |
| 58 | Einfügung | `*(nicht da)*` | `generischen` | (FMT) …werde ich die [___] sprecher labels ersetzen… |
| 59 | Einfügung | `*(nicht da)*` | `sprecher` | (FMT) …ich die generischen [___] labels ersetzen durch… |
| 60 | Einfügung | `*(nicht da)*` | `labels` | (FMT) …die generischen sprecher [___] ersetzen durch arzt… |
| 61 | Einfügung | `*(nicht da)*` | `ersetzen` | (FMT) …generischen sprecher labels [___] durch arzt und… |
| 62 | Einfügung | `*(nicht da)*` | `durch` | (FMT) …sprecher labels ersetzen [___] arzt und name… |
| 63 | Einfügung | `*(nicht da)*` | `arzt` | (FMT) …labels ersetzen durch [___] und name des… |
| 64 | Einfügung | `*(nicht da)*` | `und` | (FMT) …ersetzen durch arzt [___] name des patienten… |
| 65 | Einfügung | `*(nicht da)*` | `name` | (FMT) …durch arzt und [___] des patienten da… |
| 66 | Einfügung | `*(nicht da)*` | `des` | (FMT) …arzt und name [___] patienten da der… |
| 67 | Einfügung | `*(nicht da)*` | `patienten` | (FMT) …und name des [___] da der name… |
| 68 | Einfügung | `*(nicht da)*` | `da` | (FMT) …name des patienten [___] der name des… |
| 69 | Einfügung | `*(nicht da)*` | `der` | (FMT) …des patienten da [___] name des patienten… |
| 70 | Einfügung | `*(nicht da)*` | `name` | (FMT) …patienten da der [___] des patienten im… |
| 71 | Einfügung | `*(nicht da)*` | `des` | (FMT) …da der name [___] patienten im gespräch… |
| 72 | Einfügung | `*(nicht da)*` | `patienten` | (FMT) …der name des [___] im gespräch mehrfach… |
| 73 | Einfügung | `*(nicht da)*` | `im` | (FMT) …name des patienten [___] gespräch mehrfach genannt… |
| 74 | Einfügung | `*(nicht da)*` | `gespräch` | (FMT) …des patienten im [___] mehrfach genannt wird… |
| 75 | Einfügung | `*(nicht da)*` | `mehrfach` | (FMT) …patienten im gespräch [___] genannt wird ist… |
| 76 | Einfügung | `*(nicht da)*` | `genannt` | (FMT) …im gespräch mehrfach [___] wird ist es… |
| 77 | Einfügung | `*(nicht da)*` | `wird` | (FMT) …gespräch mehrfach genannt [___] ist es wahrscheinlich… |
| 78 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …mehrfach genannt wird [___] es wahrscheinlich frau… |
| 79 | Einfügung | `*(nicht da)*` | `es` | (FMT) …genannt wird ist [___] wahrscheinlich frau grasbeutner… |
| 80 | Einfügung | `*(nicht da)*` | `wahrscheinlich` | (FMT) …wird ist es [___] frau grasbeutner hier… |
| 81 | Einfügung | `*(nicht da)*` | `frau` | (FMT) …ist es wahrscheinlich [___] grasbeutner hier ist… |
| 82 | Einfügung | `*(nicht da)*` | `grasbeutner` | (FMT) …es wahrscheinlich frau [___] hier ist das… |
| 83 | Einfügung | `*(nicht da)*` | `hier` | (FMT) …wahrscheinlich frau grasbeutner [___] ist das formatierte… |
| 84 | Einfügung | `*(nicht da)*` | `ist` | (FMT) …frau grasbeutner hier [___] das formatierte transkript… |
| 85 | Einfügung | `*(nicht da)*` | `das` | (FMT) …grasbeutner hier ist [___] formatierte transkript grüß… |
| 86 | Einfügung | `*(nicht da)*` | `formatierte` | (FMT) …hier ist das [___] transkript grüß gott… |
| 87 | Einfügung | `*(nicht da)*` | `transkript` | (FMT) …ist das formatierte [___] grüß gott frau… |
| 88 | Löschung | `seit` | `*(nicht da)*` | …sie schon lange [___] wann machen sie… |
| 89 | Löschung | `wann` | `*(nicht da)*` | …schon lange seit [___] machen sie das… |
| 90 | Löschung | `machen` | `*(nicht da)*` | …lange seit wann [___] sie das ja… |
| 91 | Löschung | `sie` | `*(nicht da)*` | …seit wann machen [___] das ja ich… |
| 92 | Löschung | `das` | `*(nicht da)*` | …wann machen sie [___] ja ich glaube… |
| 93 | Substitution | `haben` | `habe` | …ins krankenhaus dann [___] wir nach der… |
| 94 | Substitution | `wir` | `ich` | …krankenhaus dann haben [___] nach der sicherheit… |
| 95 | Löschung | `und` | `*(nicht da)*` | …15 november immer [___] die operation ist… |
| 96 | Löschung | `die` | `*(nicht da)*` | …november immer und [___] operation ist ihrer… |
| 97 | Löschung | `operation` | `*(nicht da)*` | …immer und die [___] ist ihrer meinung… |
| 98 | Löschung | `ist` | `*(nicht da)*` | …und die operation [___] ihrer meinung nach… |
| 99 | Löschung | `ihrer` | `*(nicht da)*` | …die operation ist [___] meinung nach normal… |
| 100 | Löschung | `meinung` | `*(nicht da)*` | …operation ist ihrer [___] nach normal verlaufen… |
| 101 | Löschung | `nach` | `*(nicht da)*` | …ist ihrer meinung [___] normal verlaufen wie… |
| 102 | Löschung | `normal` | `*(nicht da)*` | …ihrer meinung nach [___] verlaufen wie haben… |
| 103 | Löschung | `verlaufen` | `*(nicht da)*` | …meinung nach normal [___] wie haben sie… |
| 104 | Löschung | `wie` | `*(nicht da)*` | …nach normal verlaufen [___] haben sie das… |
| 105 | Löschung | `haben` | `*(nicht da)*` | …normal verlaufen wie [___] sie das mitbekommen… |
| 106 | Löschung | `sie` | `*(nicht da)*` | …verlaufen wie haben [___] das mitbekommen war… |
| 107 | Löschung | `das` | `*(nicht da)*` | …wie haben sie [___] mitbekommen war der… |
| 108 | Löschung | `mitbekommen` | `*(nicht da)*` | …haben sie das [___] war der heilungsprozess… |
| 109 | Löschung | `war` | `*(nicht da)*` | …sie das mitbekommen [___] der heilungsprozess also… |
| 110 | Löschung | `also` | `*(nicht da)*` | …war der heilungsprozess [___] es ist mir… |
| 111 | Löschung | `es` | `*(nicht da)*` | …der heilungsprozess also [___] ist mir gesagt… |
| 112 | Substitution | `mir` | `ihnen` | …also es ist [___] gesagt worden dass… |
| 113 | Löschung | `und` | `*(nicht da)*` | …mit dem… ja [___] was haben sie… |
| 114 | Löschung | `gemacht` | `*(nicht da)*` | …sie da genau [___] was haben sie… |
| 115 | Löschung | `was` | `*(nicht da)*` | …da genau gemacht [___] haben sie da… |
| 116 | Löschung | `haben` | `*(nicht da)*` | …genau gemacht was [___] sie da schon… |
| 117 | Löschung | `sie` | `*(nicht da)*` | …gemacht was haben [___] da schon gemacht… |
| 118 | Löschung | `da` | `*(nicht da)*` | …was haben sie [___] schon gemacht ja… |
| 119 | Löschung | `schon` | `*(nicht da)*` | …haben sie da [___] gemacht ja hauptsächlich… |
| 120 | Löschung | `ja` | `*(nicht da)*` | …da schon gemacht [___] hauptsächlich mit… ja… |
| 121 | Löschung | `halt` | `*(nicht da)*` | …dann hat man [___] gesagt wie ich… |
| 122 | Löschung | `gesagt` | `*(nicht da)*` | …hat man halt [___] wie ich mit… |
| 123 | Löschung | `wie` | `*(nicht da)*` | …man halt gesagt [___] ich mit den… |
| 124 | Löschung | `ich` | `*(nicht da)*` | …halt gesagt wie [___] mit den krücken… |
| 125 | Löschung | `mit` | `*(nicht da)*` | …gesagt wie ich [___] den krücken gehen… |
| 126 | Löschung | `den` | `*(nicht da)*` | …wie ich mit [___] krücken gehen soll… |
| 127 | Löschung | `krücken` | `*(nicht da)*` | …ich mit den [___] gehen soll wie… |
| 128 | Löschung | `gehen` | `*(nicht da)*` | …mit den krücken [___] soll wie das… |
| 129 | Löschung | `soll` | `*(nicht da)*` | …den krücken gehen [___] wie das aufsteigen… |
| 130 | Löschung | `wie` | `*(nicht da)*` | …krücken gehen soll [___] das aufsteigen und… |
| 131 | Löschung | `das` | `*(nicht da)*` | …gehen soll wie [___] aufsteigen und das… |
| 132 | Löschung | `aufsteigen` | `*(nicht da)*` | …soll wie das [___] und das abrollen… |
| 133 | Löschung | `und` | `*(nicht da)*` | …wie das aufsteigen [___] das abrollen mit… |
| 134 | Löschung | `das` | `*(nicht da)*` | …das aufsteigen und [___] abrollen mit dem… |
| 135 | Löschung | `abrollen` | `*(nicht da)*` | …aufsteigen und das [___] mit dem furs… |
| 136 | Löschung | `mit` | `*(nicht da)*` | …und das abrollen [___] dem furs auch… |
| 137 | Löschung | `dem` | `*(nicht da)*` | …das abrollen mit [___] furs auch richtig… |
| 138 | Löschung | `furs` | `*(nicht da)*` | …abrollen mit dem [___] auch richtig einlernen… |
| 139 | Löschung | `auch` | `*(nicht da)*` | …mit dem furs [___] richtig einlernen wieder… |
| 140 | Löschung | `richtig` | `*(nicht da)*` | …dem furs auch [___] einlernen wieder ich… |
| 141 | Löschung | `einlernen` | `*(nicht da)*` | …furs auch richtig [___] wieder ich glaube… |
| 142 | Löschung | `wieder` | `*(nicht da)*` | …auch richtig einlernen [___] ich glaube am… |
| 143 | Löschung | `ich` | `*(nicht da)*` | …richtig einlernen wieder [___] glaube am schluss… |
| 144 | Löschung | `glaube` | `*(nicht da)*` | …einlernen wieder ich [___] am schluss sogar… |
| 145 | Löschung | `am` | `*(nicht da)*` | …wieder ich glaube [___] schluss sogar haben… |
| 146 | Löschung | `schluss` | `*(nicht da)*` | …ich glaube am [___] sogar haben wir… |
| 147 | Löschung | `sogar` | `*(nicht da)*` | …glaube am schluss [___] haben wir treppensteigen… |
| 148 | Löschung | `haben` | `*(nicht da)*` | …am schluss sogar [___] wir treppensteigen dann… |
| 149 | Löschung | `wir` | `*(nicht da)*` | …schluss sogar haben [___] treppensteigen dann also… |
| 150 | Löschung | `treppensteigen` | `*(nicht da)*` | …sogar haben wir [___] dann also ja… |
| 151 | Löschung | `dann` | `*(nicht da)*` | …haben wir treppensteigen [___] also ja und… |
| 152 | Löschung | `also` | `*(nicht da)*` | …wir treppensteigen dann [___] ja und ein… |
| 153 | Löschung | `ja` | `*(nicht da)*` | …treppensteigen dann also [___] und ein bisschen… |
| 154 | Löschung | `und` | `*(nicht da)*` | …dann also ja [___] ein bisschen so… |
| 155 | Löschung | `ein` | `*(nicht da)*` | …also ja und [___] bisschen so beugen… |
| 156 | Löschung | `bisschen` | `*(nicht da)*` | …ja und ein [___] so beugen üben… |
| 157 | Löschung | `so` | `*(nicht da)*` | …und ein bisschen [___] beugen üben so… |
| 158 | Löschung | `beugen` | `*(nicht da)*` | …ein bisschen so [___] üben so war… |
| 159 | Löschung | `üben` | `*(nicht da)*` | …bisschen so beugen [___] so war es… |
| 160 | Löschung | `so` | `*(nicht da)*` | …so beugen üben [___] war es halt… |
| 161 | Löschung | `war` | `*(nicht da)*` | …beugen üben so [___] es halt gegangen… |
| 162 | Löschung | `es` | `*(nicht da)*` | …üben so war [___] halt gegangen und… |
| 163 | Löschung | `gegangen` | `*(nicht da)*` | …war es halt [___] und sie haben… |
| 164 | Löschung | `und` | `*(nicht da)*` | …es halt gegangen [___] sie haben das… |
| 165 | Löschung | `sie` | `*(nicht da)*` | …halt gegangen und [___] haben das da… |
| 166 | Löschung | `haben` | `*(nicht da)*` | …gegangen und sie [___] das da schon… |
| 167 | Löschung | `das` | `*(nicht da)*` | …und sie haben [___] da schon eben… |
| 168 | Löschung | `da` | `*(nicht da)*` | …sie haben das [___] schon eben gesagt… |
| 169 | Löschung | `schon` | `*(nicht da)*` | …haben das da [___] eben gesagt sie… |
| 170 | Löschung | `eben` | `*(nicht da)*` | …das da schon [___] gesagt sie haben… |
| 171 | Löschung | `sie` | `*(nicht da)*` | …schon eben gesagt [___] haben eben mit… |
| 172 | Löschung | `haben` | `*(nicht da)*` | …eben gesagt sie [___] eben mit den… |
| 173 | Löschung | `eben` | `*(nicht da)*` | …gesagt sie haben [___] mit den stützen… |
| 174 | Löschung | `mit` | `*(nicht da)*` | …sie haben eben [___] den stützen das… |
| 175 | Löschung | `den` | `*(nicht da)*` | …haben eben mit [___] stützen das gelernt… |
| 176 | Löschung | `stützen` | `*(nicht da)*` | …eben mit den [___] das gelernt zum… |
| 177 | Löschung | `das` | `*(nicht da)*` | …mit den stützen [___] gelernt zum gehen… |
| 178 | Löschung | `gelernt` | `*(nicht da)*` | …den stützen das [___] zum gehen sie… |
| 179 | Löschung | `zum` | `*(nicht da)*` | …stützen das gelernt [___] gehen sie haben… |
| 180 | Löschung | `gehen` | `*(nicht da)*` | …das gelernt zum [___] sie haben das… |
| 181 | Löschung | `sie` | `*(nicht da)*` | …gelernt zum gehen [___] haben das auch… |
| 182 | Löschung | `haben` | `*(nicht da)*` | …zum gehen sie [___] das auch freuen… |
| 183 | Löschung | `das` | `*(nicht da)*` | …gehen sie haben [___] auch freuen gelernt… |
| 184 | Löschung | `auch` | `*(nicht da)*` | …sie haben das [___] freuen gelernt können… |
| 185 | Löschung | `freuen` | `*(nicht da)*` | …haben das auch [___] gelernt können sie… |
| 186 | Löschung | `gelernt` | `*(nicht da)*` | …das auch freuen [___] können sie das… |
| 187 | Löschung | `können` | `*(nicht da)*` | …auch freuen gelernt [___] sie das für… |
| 188 | Löschung | `sie` | `*(nicht da)*` | …freuen gelernt können [___] das für sie… |
| 189 | Löschung | `das` | `*(nicht da)*` | …gelernt können sie [___] für sie sagen… |
| 190 | Löschung | `für` | `*(nicht da)*` | …können sie das [___] sie sagen dass… |
| 191 | Löschung | `sie` | `*(nicht da)*` | …sie das für [___] sagen dass sie… |
| 192 | Löschung | `sagen` | `*(nicht da)*` | …das für sie [___] dass sie das… |
| 193 | Löschung | `dass` | `*(nicht da)*` | …für sie sagen [___] sie das dann… |
| 194 | Löschung | `sie` | `*(nicht da)*` | …sie sagen dass [___] das dann jetzt… |
| 195 | Löschung | `das` | `*(nicht da)*` | …sagen dass sie [___] dann jetzt in… |
| 196 | Löschung | `dann` | `*(nicht da)*` | …dass sie das [___] jetzt in den… |
| 197 | Löschung | `jetzt` | `*(nicht da)*` | …sie das dann [___] in den alltag… |
| 198 | Löschung | `in` | `*(nicht da)*` | …das dann jetzt [___] den alltag den… |
| 199 | Löschung | `den` | `*(nicht da)*` | …dann jetzt in [___] alltag den sie… |
| 200 | Löschung | `alltag` | `*(nicht da)*` | …jetzt in den [___] den sie jetzt… |
| 201 | Löschung | `den` | `*(nicht da)*` | …in den alltag [___] sie jetzt dann… |
| 202 | Löschung | `sie` | `*(nicht da)*` | …den alltag den [___] jetzt dann wieder… |
| 203 | Löschung | `jetzt` | `*(nicht da)*` | …alltag den sie [___] dann wieder haben… |
| 204 | Löschung | `dann` | `*(nicht da)*` | …den sie jetzt [___] wieder haben integriert… |
| 205 | Löschung | `wieder` | `*(nicht da)*` | …sie jetzt dann [___] haben integriert haben… |
| 206 | Löschung | `haben` | `*(nicht da)*` | …jetzt dann wieder [___] integriert haben ja… |
| 207 | Löschung | `integriert` | `*(nicht da)*` | …dann wieder haben [___] haben ja dass… |
| 208 | Löschung | `haben` | `*(nicht da)*` | …wieder haben integriert [___] ja dass das… |
| 209 | Löschung | `ja` | `*(nicht da)*` | …haben integriert haben [___] dass das auch… |
| 210 | Löschung | `dass` | `*(nicht da)*` | …integriert haben ja [___] das auch gut… |
| 211 | Löschung | `das` | `*(nicht da)*` | …haben ja dass [___] auch gut beherrschen… |
| 212 | Löschung | `auch` | `*(nicht da)*` | …ja dass das [___] gut beherrschen ja… |
| 213 | Löschung | `gut` | `*(nicht da)*` | …dass das auch [___] beherrschen ja schon… |
| 214 | Löschung | `beherrschen` | `*(nicht da)*` | …das auch gut [___] ja schon also… |
| 215 | Löschung | `ja` | `*(nicht da)*` | …auch gut beherrschen [___] schon also das… |
| 216 | Löschung | `schon` | `*(nicht da)*` | …gut beherrschen ja [___] also das auf… |
| 217 | Löschung | `also` | `*(nicht da)*` | …beherrschen ja schon [___] das auf jeden… |
| 218 | Löschung | `das` | `*(nicht da)*` | …ja schon also [___] auf jeden fall… |
| 219 | Löschung | `auf` | `*(nicht da)*` | …schon also das [___] jeden fall ja… |
| 220 | Löschung | `jeden` | `*(nicht da)*` | …also das auf [___] fall ja dass… |
| 221 | Löschung | `fall` | `*(nicht da)*` | …das auf jeden [___] ja dass sie… |
| 222 | Löschung | `ja` | `*(nicht da)*` | …auf jeden fall [___] dass sie darauf… |
| 223 | Löschung | `dass` | `*(nicht da)*` | …jeden fall ja [___] sie darauf achten… |
| 224 | Löschung | `sie` | `*(nicht da)*` | …fall ja dass [___] darauf achten einfach… |
| 225 | Löschung | `darauf` | `*(nicht da)*` | …ja dass sie [___] achten einfach wie… |
| 226 | Löschung | `achten` | `*(nicht da)*` | …dass sie darauf [___] einfach wie aufsteigen… |
| 227 | Löschung | `einfach` | `*(nicht da)*` | …sie darauf achten [___] wie aufsteigen genau… |
| 228 | Löschung | `wie` | `*(nicht da)*` | …darauf achten einfach [___] aufsteigen genau und… |
| 229 | Löschung | `aufsteigen` | `*(nicht da)*` | …achten einfach wie [___] genau und das… |
| 230 | Löschung | `genau` | `*(nicht da)*` | …einfach wie aufsteigen [___] und das hat… |
| 231 | Löschung | `und` | `*(nicht da)*` | …wie aufsteigen genau [___] das hat ihnen… |
| 232 | Löschung | `das` | `*(nicht da)*` | …aufsteigen genau und [___] hat ihnen zum… |
| 233 | Löschung | `hat` | `*(nicht da)*` | …genau und das [___] ihnen zum beispiel… |
| 234 | Löschung | `ihnen` | `*(nicht da)*` | …und das hat [___] zum beispiel schon… |
| 235 | Löschung | `zum` | `*(nicht da)*` | …das hat ihnen [___] beispiel schon geholfen… |
| 236 | Löschung | `beispiel` | `*(nicht da)*` | …hat ihnen zum [___] schon geholfen also… |
| 237 | Löschung | `schon` | `*(nicht da)*` | …ihnen zum beispiel [___] geholfen also sie… |
| 238 | Löschung | `geholfen` | `*(nicht da)*` | …zum beispiel schon [___] also sie haben… |
| 239 | Löschung | `also` | `*(nicht da)*` | …beispiel schon geholfen [___] sie haben da… |
| 240 | Löschung | `sie` | `*(nicht da)*` | …schon geholfen also [___] haben da gerne… |
| 241 | Löschung | `haben` | `*(nicht da)*` | …geholfen also sie [___] da gerne mitgemacht… |
| 242 | Löschung | `da` | `*(nicht da)*` | …also sie haben [___] gerne mitgemacht in… |
| 243 | Löschung | `gerne` | `*(nicht da)*` | …sie haben da [___] mitgemacht in der… |
| 244 | Löschung | `mitgemacht` | `*(nicht da)*` | …haben da gerne [___] in der therapie… |
| 245 | Löschung | `in` | `*(nicht da)*` | …da gerne mitgemacht [___] der therapie und…okay… |
| 246 | Löschung | `der` | `*(nicht da)*` | …gerne mitgemacht in [___] therapie und…okay ja… |
| 247 | Löschung | `therapie` | `*(nicht da)*` | …mitgemacht in der [___] und…okay ja das… |
| 248 | Löschung | `und…okay` | `*(nicht da)*` | …in der therapie [___] ja das war… |
| 249 | Löschung | `ja` | `*(nicht da)*` | …der therapie und…okay [___] das war halt… |
| 250 | Löschung | `das` | `*(nicht da)*` | …therapie und…okay ja [___] war halt der… |
| 251 | Löschung | `war` | `*(nicht da)*` | …und…okay ja das [___] halt der verlauf… |
| 252 | Löschung | `halt` | `*(nicht da)*` | …ja das war [___] der verlauf wie… |
| 253 | Löschung | `der` | `*(nicht da)*` | …das war halt [___] verlauf wie war… |
| 254 | Löschung | `verlauf` | `*(nicht da)*` | …war halt der [___] wie war das… |
| 255 | Löschung | `wie` | `*(nicht da)*` | …halt der verlauf [___] war das dann… |
| 256 | Löschung | `war` | `*(nicht da)*` | …der verlauf wie [___] das dann nach… |
| 257 | Löschung | `das` | `*(nicht da)*` | …verlauf wie war [___] dann nach der… |
| 258 | Löschung | `dann` | `*(nicht da)*` | …wie war das [___] nach der operation… |
| 259 | Löschung | `nach` | `*(nicht da)*` | …war das dann [___] der operation die… |
| 260 | Löschung | `der` | `*(nicht da)*` | …das dann nach [___] operation die woche… |
| 261 | Löschung | `operation` | `*(nicht da)*` | …dann nach der [___] die woche nachher… |
| 262 | Löschung | `die` | `*(nicht da)*` | …nach der operation [___] woche nachher und… |
| 263 | Löschung | `woche` | `*(nicht da)*` | …der operation die [___] nachher und wie… |
| 264 | Löschung | `nachher` | `*(nicht da)*` | …operation die woche [___] und wie ist… |
| 265 | Löschung | `und` | `*(nicht da)*` | …die woche nachher [___] wie ist ihnen… |
| 266 | Löschung | `ist` | `*(nicht da)*` | …nachher und wie [___] ihnen dann da… |
| 267 | Löschung | `ihnen` | `*(nicht da)*` | …und wie ist [___] dann da gegangen… |
| 268 | Löschung | `dann` | `*(nicht da)*` | …wie ist ihnen [___] da gegangen mit… |
| 269 | Löschung | `da` | `*(nicht da)*` | …ist ihnen dann [___] gegangen mit den… |
| 270 | Löschung | `gegangen` | `*(nicht da)*` | …ihnen dann da [___] mit den schmerzen… |
| 271 | Löschung | `mit` | `*(nicht da)*` | …dann da gegangen [___] den schmerzen ja… |
| 272 | Löschung | `den` | `*(nicht da)*` | …da gegangen mit [___] schmerzen ja schmerzen… |
| 273 | Löschung | `schmerzen` | `*(nicht da)*` | …gegangen mit den [___] ja schmerzen war… |
| 274 | Löschung | `ja` | `*(nicht da)*` | …mit den schmerzen [___] schmerzen war ja… |
| 275 | Löschung | `schmerzen` | `*(nicht da)*` | …den schmerzen ja [___] war ja war… |
| 276 | Löschung | `war` | `*(nicht da)*` | …schmerzen ja schmerzen [___] ja war okay… |
| 277 | Löschung | `ja` | `*(nicht da)*` | …ja schmerzen war [___] war okay sag… |
| 278 | Löschung | `war` | `*(nicht da)*` | …schmerzen war ja [___] okay sag ich… |
| 279 | Löschung | `okay` | `*(nicht da)*` | …war ja war [___] sag ich mal… |
| 280 | Löschung | `sag` | `*(nicht da)*` | …ja war okay [___] ich mal war… |
| 281 | Löschung | `ich` | `*(nicht da)*` | …war okay sag [___] mal war okay… |
| 282 | Löschung | `mal` | `*(nicht da)*` | …okay sag ich [___] war okay je… |
| 283 | Löschung | `war` | `*(nicht da)*` | …sag ich mal [___] okay je nachdem… |
| 284 | Löschung | `okay` | `*(nicht da)*` | …ich mal war [___] je nachdem je… |
| 285 | Löschung | `je` | `*(nicht da)*` | …mal war okay [___] nachdem je nach… |
| 286 | Löschung | `nachdem` | `*(nicht da)*` | …war okay je [___] je nach belastung… |
| 287 | Löschung | `je` | `*(nicht da)*` | …okay je nachdem [___] nach belastung je… |
| 288 | Löschung | `nach` | `*(nicht da)*` | …je nachdem je [___] belastung je nach… |
| 289 | Löschung | `belastung` | `*(nicht da)*` | …nachdem je nach [___] je nach belastung… |
| 290 | Löschung | `je` | `*(nicht da)*` | …je nach belastung [___] nach belastung es… |
| 291 | Löschung | `nach` | `*(nicht da)*` | …nach belastung je [___] belastung es war… |
| 292 | Löschung | `belastung` | `*(nicht da)*` | …belastung je nach [___] es war halt…… |
| 293 | Löschung | `es` | `*(nicht da)*` | …je nach belastung [___] war halt… ich… |
| 294 | Löschung | `war` | `*(nicht da)*` | …nach belastung es [___] halt… ich habe… |
| 295 | Löschung | `halt…` | `*(nicht da)*` | …belastung es war [___] ich habe mich… |
| 296 | Löschung | `habe` | `*(nicht da)*` | …war halt… ich [___] mich halt nicht… |
| 297 | Löschung | `halt` | `*(nicht da)*` | …ich habe mich [___] nicht viel bewegen… |
| 298 | Löschung | `nicht` | `*(nicht da)*` | …habe mich halt [___] viel bewegen können… |
| 299 | Substitution | `viel` | `wieder` | …mich halt nicht [___] bewegen können ich… |
| 300 | Löschung | `können` | `*(nicht da)*` | …nicht viel bewegen [___] ich bin ja… |
| 301 | Löschung | `ich` | `*(nicht da)*` | …viel bewegen können [___] bin ja eigentlich… |
| 302 | Löschung | `bin` | `*(nicht da)*` | …bewegen können ich [___] ja eigentlich nur… |
| 303 | Löschung | `ja` | `*(nicht da)*` | …können ich bin [___] eigentlich nur gelegen… |
| 304 | Löschung | `eigentlich` | `*(nicht da)*` | …ich bin ja [___] nur gelegen okay… |
| 305 | Löschung | `nur` | `*(nicht da)*` | …bin ja eigentlich [___] gelegen okay die… |
| 306 | Löschung | `gelegen` | `*(nicht da)*` | …ja eigentlich nur [___] okay die erste… |
| 307 | Löschung | `okay` | `*(nicht da)*` | …eigentlich nur gelegen [___] die erste woche… |
| 308 | Löschung | `die` | `*(nicht da)*` | …nur gelegen okay [___] erste woche deine… |
| 309 | Löschung | `erste` | `*(nicht da)*` | …gelegen okay die [___] woche deine letzte… |
| 310 | Löschung | `woche` | `*(nicht da)*` | …okay die erste [___] deine letzte zeit… |
| 311 | Löschung | `deine` | `*(nicht da)*` | …die erste woche [___] letzte zeit und… |
| 312 | Löschung | `letzte` | `*(nicht da)*` | …erste woche deine [___] zeit und dann… |
| 313 | Löschung | `zeit` | `*(nicht da)*` | …woche deine letzte [___] und dann ja… |
| 314 | Löschung | `und` | `*(nicht da)*` | …deine letzte zeit [___] dann ja mit… |
| 315 | Löschung | `dann` | `*(nicht da)*` | …letzte zeit und [___] ja mit den… |
| 316 | Löschung | `ja` | `*(nicht da)*` | …zeit und dann [___] mit den grücken… |
| 317 | Löschung | `mit` | `*(nicht da)*` | …und dann ja [___] den grücken halt… |
| 318 | Löschung | `den` | `*(nicht da)*` | …dann ja mit [___] grücken halt herumgehen… |
| 319 | Löschung | `grücken` | `*(nicht da)*` | …ja mit den [___] halt herumgehen ein… |
| 320 | Löschung | `halt` | `*(nicht da)*` | …mit den grücken [___] herumgehen ein bisschen… |
| 321 | Löschung | `herumgehen` | `*(nicht da)*` | …den grücken halt [___] ein bisschen aber… |
| 322 | Löschung | `ein` | `*(nicht da)*` | …grücken halt herumgehen [___] bisschen aber halt… |
| 323 | Löschung | `bisschen` | `*(nicht da)*` | …halt herumgehen ein [___] aber halt auch… |
| 324 | Löschung | `aber` | `*(nicht da)*` | …herumgehen ein bisschen [___] halt auch minimal… |
| 325 | Löschung | `halt` | `*(nicht da)*` | …ein bisschen aber [___] auch minimal okay… |
| 326 | Löschung | `auch` | `*(nicht da)*` | …bisschen aber halt [___] minimal okay dann… |
| 327 | Löschung | `minimal` | `*(nicht da)*` | …aber halt auch [___] okay dann sind… |
| 328 | Löschung | `okay` | `*(nicht da)*` | …halt auch minimal [___] dann sind wir… |
| 329 | Löschung | `dann` | `*(nicht da)*` | …auch minimal okay [___] sind wir jetzt… |
| 330 | Löschung | `sind` | `*(nicht da)*` | …minimal okay dann [___] wir jetzt schon… |
| 331 | Löschung | `wir` | `*(nicht da)*` | …okay dann sind [___] jetzt schon so… |
| 332 | Löschung | `jetzt` | `*(nicht da)*` | …dann sind wir [___] schon so weit… |
| 333 | Löschung | `schon` | `*(nicht da)*` | …sind wir jetzt [___] so weit dass… |
| 334 | Löschung | `so` | `*(nicht da)*` | …wir jetzt schon [___] weit dass wir… |
| 335 | Löschung | `weit` | `*(nicht da)*` | …jetzt schon so [___] dass wir darüber… |
| 336 | Löschung | `dass` | `*(nicht da)*` | …schon so weit [___] wir darüber reden… |
| 337 | Löschung | `wir` | `*(nicht da)*` | …so weit dass [___] darüber reden wie… |
| 338 | Löschung | `darüber` | `*(nicht da)*` | …weit dass wir [___] reden wie es… |
| 339 | Löschung | `reden` | `*(nicht da)*` | …dass wir darüber [___] wie es ihnen… |
| 340 | Löschung | `wie` | `*(nicht da)*` | …wir darüber reden [___] es ihnen jetzt… |
| 341 | Löschung | `es` | `*(nicht da)*` | …darüber reden wie [___] ihnen jetzt geht… |
| 342 | Löschung | `ihnen` | `*(nicht da)*` | …reden wie es [___] jetzt geht wie… |
| 343 | Löschung | `jetzt` | `*(nicht da)*` | …wie es ihnen [___] geht wie geht… |
| 344 | Löschung | `geht` | `*(nicht da)*` | …es ihnen jetzt [___] wie geht es… |
| 345 | Löschung | `wie` | `*(nicht da)*` | …ihnen jetzt geht [___] geht es ihnen… |
| 346 | Löschung | `geht` | `*(nicht da)*` | …jetzt geht wie [___] es ihnen wenn… |
| 347 | Löschung | `es` | `*(nicht da)*` | …geht wie geht [___] ihnen wenn sie… |
| 348 | Löschung | `ihnen` | `*(nicht da)*` | …wie geht es [___] wenn sie an… |
| 349 | Löschung | `wenn` | `*(nicht da)*` | …geht es ihnen [___] sie an die… |
| 350 | Löschung | `sie` | `*(nicht da)*` | …es ihnen wenn [___] an die schmerzen… |
| 351 | Löschung | `an` | `*(nicht da)*` | …ihnen wenn sie [___] die schmerzen denken… |
| 352 | Löschung | `die` | `*(nicht da)*` | …wenn sie an [___] schmerzen denken auf… |
| 353 | Löschung | `schmerzen` | `*(nicht da)*` | …sie an die [___] denken auf einer… |
| 354 | Löschung | `denken` | `*(nicht da)*` | …an die schmerzen [___] auf einer skala… |
| 355 | Löschung | `auf` | `*(nicht da)*` | …die schmerzen denken [___] einer skala von… |
| 356 | Löschung | `einer` | `*(nicht da)*` | …schmerzen denken auf [___] skala von 0… |
| 357 | Löschung | `skala` | `*(nicht da)*` | …denken auf einer [___] von 0 bis… |
| 358 | Löschung | `von` | `*(nicht da)*` | …auf einer skala [___] 0 bis 10… |
| 359 | Löschung | `0` | `*(nicht da)*` | …einer skala von [___] bis 10 und… |
| 360 | Löschung | `bis` | `*(nicht da)*` | …skala von 0 [___] 10 und 10… |
| 361 | Löschung | `10` | `*(nicht da)*` | …von 0 bis [___] und 10 sind… |
| 362 | Löschung | `und` | `*(nicht da)*` | …0 bis 10 [___] 10 sind die… |
| 363 | Löschung | `10` | `*(nicht da)*` | …bis 10 und [___] sind die schlimmsten… |
| 364 | Löschung | `sind` | `*(nicht da)*` | …10 und 10 [___] die schlimmsten schmerzen… |
| 365 | Löschung | `die` | `*(nicht da)*` | …und 10 sind [___] schlimmsten schmerzen die… |
| 366 | Löschung | `schlimmsten` | `*(nicht da)*` | …10 sind die [___] schmerzen die ich… |
| 367 | Löschung | `schmerzen` | `*(nicht da)*` | …sind die schlimmsten [___] die ich sich… |
| 368 | Löschung | `die` | `*(nicht da)*` | …die schlimmsten schmerzen [___] ich sich vorstellen… |
| 369 | Löschung | `ich` | `*(nicht da)*` | …schlimmsten schmerzen die [___] sich vorstellen könnte… |
| 370 | Löschung | `sich` | `*(nicht da)*` | …schmerzen die ich [___] vorstellen könnte und… |
| 371 | Löschung | `vorstellen` | `*(nicht da)*` | …die ich sich [___] könnte und 0… |
| 372 | Löschung | `könnte` | `*(nicht da)*` | …ich sich vorstellen [___] und 0 ist… |
| 373 | Löschung | `und` | `*(nicht da)*` | …sich vorstellen könnte [___] 0 ist schmerzfrei… |
| 374 | Löschung | `0` | `*(nicht da)*` | …vorstellen könnte und [___] ist schmerzfrei wo… |
| 375 | Löschung | `ist` | `*(nicht da)*` | …könnte und 0 [___] schmerzfrei wo würden… |
| 376 | Löschung | `schmerzfrei` | `*(nicht da)*` | …und 0 ist [___] wo würden sie… |
| 377 | Löschung | `wo` | `*(nicht da)*` | …0 ist schmerzfrei [___] würden sie sich… |
| 378 | Löschung | `würden` | `*(nicht da)*` | …ist schmerzfrei wo [___] sie sich da… |
| 379 | Löschung | `sie` | `*(nicht da)*` | …schmerzfrei wo würden [___] sich da eingliedern… |
| 380 | Löschung | `sich` | `*(nicht da)*` | …wo würden sie [___] da eingliedern ja… |
| 381 | Löschung | `da` | `*(nicht da)*` | …würden sie sich [___] eingliedern ja wie… |
| 382 | Löschung | `eingliedern` | `*(nicht da)*` | …sie sich da [___] ja wie gesagt… |
| 383 | Löschung | `ja` | `*(nicht da)*` | …sich da eingliedern [___] wie gesagt es… |
| 384 | Löschung | `wie` | `*(nicht da)*` | …da eingliedern ja [___] gesagt es kommt… |
| 385 | Löschung | `gesagt` | `*(nicht da)*` | …eingliedern ja wie [___] es kommt eigentlich… |
| 386 | Löschung | `es` | `*(nicht da)*` | …ja wie gesagt [___] kommt eigentlich auf… |
| 387 | Löschung | `kommt` | `*(nicht da)*` | …wie gesagt es [___] eigentlich auf die… |
| 388 | Löschung | `eigentlich` | `*(nicht da)*` | …gesagt es kommt [___] auf die belastung… |
| 389 | Löschung | `auf` | `*(nicht da)*` | …es kommt eigentlich [___] die belastung darauf… |
| 390 | Löschung | `die` | `*(nicht da)*` | …kommt eigentlich auf [___] belastung darauf an… |
| 391 | Löschung | `belastung` | `*(nicht da)*` | …eigentlich auf die [___] darauf an wenn… |
| 392 | Löschung | `darauf` | `*(nicht da)*` | …auf die belastung [___] an wenn ich… |
| 393 | Löschung | `an` | `*(nicht da)*` | …die belastung darauf [___] wenn ich jetzt… |
| 394 | Löschung | `wenn` | `*(nicht da)*` | …belastung darauf an [___] ich jetzt im… |
| 395 | Löschung | `ich` | `*(nicht da)*` | …darauf an wenn [___] jetzt im ruhezustand… |
| 396 | Löschung | `jetzt` | `*(nicht da)*` | …an wenn ich [___] im ruhezustand bin… |
| 397 | Löschung | `im` | `*(nicht da)*` | …wenn ich jetzt [___] ruhezustand bin und… |
| 398 | Löschung | `ruhezustand` | `*(nicht da)*` | …ich jetzt im [___] bin und mich… |
| 399 | Löschung | `bin` | `*(nicht da)*` | …jetzt im ruhezustand [___] und mich nicht… |
| 400 | Löschung | `und` | `*(nicht da)*` | …im ruhezustand bin [___] mich nicht bewege… |
| 401 | Löschung | `mich` | `*(nicht da)*` | …ruhezustand bin und [___] nicht bewege dann… |
| 402 | Löschung | `nicht` | `*(nicht da)*` | …bin und mich [___] bewege dann sage… |
| 403 | Löschung | `bewege` | `*(nicht da)*` | …und mich nicht [___] dann sage ich… |
| 404 | Löschung | `dann` | `*(nicht da)*` | …mich nicht bewege [___] sage ich vielleicht… |
| 405 | Löschung | `sage` | `*(nicht da)*` | …nicht bewege dann [___] ich vielleicht 1… |
| 406 | Löschung | `ich` | `*(nicht da)*` | …bewege dann sage [___] vielleicht 1 aber… |
| 407 | Löschung | `vielleicht` | `*(nicht da)*` | …dann sage ich [___] 1 aber wenn… |
| 408 | Löschung | `1` | `*(nicht da)*` | …sage ich vielleicht [___] aber wenn ich… |
| 409 | Löschung | `aber` | `*(nicht da)*` | …ich vielleicht 1 [___] wenn ich jetzt… |
| 410 | Löschung | `wenn` | `*(nicht da)*` | …vielleicht 1 aber [___] ich jetzt mit… |
| 411 | Löschung | `ich` | `*(nicht da)*` | …1 aber wenn [___] jetzt mit den… |
| 412 | Löschung | `jetzt` | `*(nicht da)*` | …aber wenn ich [___] mit den grücken… |
| 413 | Löschung | `mit` | `*(nicht da)*` | …wenn ich jetzt [___] den grücken gehe… |
| 414 | Löschung | `den` | `*(nicht da)*` | …ich jetzt mit [___] grücken gehe dann… |
| 415 | Löschung | `grücken` | `*(nicht da)*` | …jetzt mit den [___] gehe dann keine… |
| 416 | Löschung | `gehe` | `*(nicht da)*` | …mit den grücken [___] dann keine ahnung… |
| 417 | Löschung | `dann` | `*(nicht da)*` | …den grücken gehe [___] keine ahnung 3… |
| 418 | Löschung | `keine` | `*(nicht da)*` | …grücken gehe dann [___] ahnung 3 und… |
| 419 | Löschung | `ahnung` | `*(nicht da)*` | …gehe dann keine [___] 3 und wenn… |
| 420 | Löschung | `3` | `*(nicht da)*` | …dann keine ahnung [___] und wenn ich… |
| 421 | Löschung | `und` | `*(nicht da)*` | …keine ahnung 3 [___] wenn ich wirklich… |
| 422 | Löschung | `wenn` | `*(nicht da)*` | …ahnung 3 und [___] ich wirklich ohne… |
| 423 | Löschung | `ich` | `*(nicht da)*` | …3 und wenn [___] wirklich ohne stützen… |
| 424 | Löschung | `wirklich` | `*(nicht da)*` | …und wenn ich [___] ohne stützen probiere… |
| 425 | Löschung | `ohne` | `*(nicht da)*` | …wenn ich wirklich [___] stützen probiere dann… |
| 426 | Löschung | `stützen` | `*(nicht da)*` | …ich wirklich ohne [___] probiere dann bin… |
| 427 | Löschung | `probiere` | `*(nicht da)*` | …wirklich ohne stützen [___] dann bin ich… |
| 428 | Löschung | `dann` | `*(nicht da)*` | …ohne stützen probiere [___] bin ich sicher… |
| 429 | Löschung | `bin` | `*(nicht da)*` | …stützen probiere dann [___] ich sicher bei… |
| 430 | Löschung | `ich` | `*(nicht da)*` | …probiere dann bin [___] sicher bei 6… |
| 431 | Löschung | `sicher` | `*(nicht da)*` | …dann bin ich [___] bei 6 oder… |
| 432 | Löschung | `bei` | `*(nicht da)*` | …bin ich sicher [___] 6 oder 7… |
| 433 | Löschung | `6` | `*(nicht da)*` | …ich sicher bei [___] oder 7 bei… |
| 434 | Löschung | `oder` | `*(nicht da)*` | …sicher bei 6 [___] 7 bei 6… |
| 435 | Löschung | `7` | `*(nicht da)*` | …bei 6 oder [___] bei 6 oder… |
| 436 | Löschung | `bei` | `*(nicht da)*` | …6 oder 7 [___] 6 oder 7… |
| 437 | Löschung | `6` | `*(nicht da)*` | …oder 7 bei [___] oder 7 aber… |
| 438 | Löschung | `oder` | `*(nicht da)*` | …7 bei 6 [___] 7 aber es… |
| 439 | Löschung | `7` | `*(nicht da)*` | …bei 6 oder [___] aber es ist… |
| 440 | Löschung | `aber` | `*(nicht da)*` | …6 oder 7 [___] es ist je… |
| 441 | Löschung | `es` | `*(nicht da)*` | …oder 7 aber [___] ist je nach… |
| 442 | Löschung | `ist` | `*(nicht da)*` | …7 aber es [___] je nach belastung… |
| 443 | Löschung | `je` | `*(nicht da)*` | …aber es ist [___] nach belastung halt… |
| 444 | Löschung | `nach` | `*(nicht da)*` | …es ist je [___] belastung halt und… |
| 445 | Löschung | `belastung` | `*(nicht da)*` | …ist je nach [___] halt und das… |
| 446 | Löschung | `halt` | `*(nicht da)*` | …je nach belastung [___] und das ist… |
| 447 | Löschung | `und` | `*(nicht da)*` | …nach belastung halt [___] das ist ja… |
| 448 | Löschung | `das` | `*(nicht da)*` | …belastung halt und [___] ist ja der… |
| 449 | Löschung | `ist` | `*(nicht da)*` | …halt und das [___] ja der einzige… |
| 450 | Löschung | `ja` | `*(nicht da)*` | …und das ist [___] der einzige faktor… |
| 451 | Löschung | `der` | `*(nicht da)*` | …das ist ja [___] einzige faktor die… |
| 452 | Löschung | `einzige` | `*(nicht da)*` | …ist ja der [___] faktor die belastung… |
| 453 | Löschung | `faktor` | `*(nicht da)*` | …ja der einzige [___] die belastung der… |
| 454 | Löschung | `die` | `*(nicht da)*` | …der einzige faktor [___] belastung der einem… |
| 455 | Löschung | `belastung` | `*(nicht da)*` | …einzige faktor die [___] der einem da… |
| 456 | Löschung | `der` | `*(nicht da)*` | …faktor die belastung [___] einem da einfällt… |
| 457 | Löschung | `einem` | `*(nicht da)*` | …die belastung der [___] da einfällt wenn… |
| 458 | Löschung | `da` | `*(nicht da)*` | …belastung der einem [___] einfällt wenn sie… |
| 459 | Löschung | `einfällt` | `*(nicht da)*` | …der einem da [___] wenn sie an… |
| 460 | Löschung | `wenn` | `*(nicht da)*` | …einem da einfällt [___] sie an den… |
| 461 | Löschung | `sie` | `*(nicht da)*` | …da einfällt wenn [___] an den schmerz… |
| 462 | Löschung | `an` | `*(nicht da)*` | …einfällt wenn sie [___] den schmerz denken… |
| 463 | Löschung | `den` | `*(nicht da)*` | …wenn sie an [___] schmerz denken dass… |
| 464 | Löschung | `schmerz` | `*(nicht da)*` | …sie an den [___] denken dass sich… |
| 465 | Löschung | `denken` | `*(nicht da)*` | …an den schmerz [___] dass sich der… |
| 466 | Löschung | `dass` | `*(nicht da)*` | …den schmerz denken [___] sich der da… |
| 467 | Löschung | `sich` | `*(nicht da)*` | …schmerz denken dass [___] der da verändert… |
| 468 | Löschung | `der` | `*(nicht da)*` | …denken dass sich [___] da verändert ja… |
| 469 | Löschung | `da` | `*(nicht da)*` | …dass sich der [___] verändert ja eigentlich… |
| 470 | Löschung | `verändert` | `*(nicht da)*` | …sich der da [___] ja eigentlich ja… |
| 471 | Löschung | `ja` | `*(nicht da)*` | …der da verändert [___] eigentlich ja also… |
| 472 | Löschung | `eigentlich` | `*(nicht da)*` | …da verändert ja [___] ja also ich… |
| 473 | Löschung | `ja` | `*(nicht da)*` | …verändert ja eigentlich [___] also ich weiß… |
| 474 | Löschung | `also` | `*(nicht da)*` | …ja eigentlich ja [___] ich weiß ja… |
| 475 | Löschung | `ich` | `*(nicht da)*` | …eigentlich ja also [___] weiß ja das… |
| 476 | Löschung | `weiß` | `*(nicht da)*` | …ja also ich [___] ja das nicht… |
| 477 | Löschung | `ja` | `*(nicht da)*` | …also ich weiß [___] das nicht an… |
| 478 | Löschung | `das` | `*(nicht da)*` | …ich weiß ja [___] nicht an was… |
| 479 | Löschung | `nicht` | `*(nicht da)*` | …weiß ja das [___] an was sonst… |
| 480 | Löschung | `an` | `*(nicht da)*` | …ja das nicht [___] was sonst noch… |
| 481 | Löschung | `was` | `*(nicht da)*` | …das nicht an [___] sonst noch okay… |
| 482 | Löschung | `sonst` | `*(nicht da)*` | …nicht an was [___] noch okay und… |
| 483 | Löschung | `noch` | `*(nicht da)*` | …an was sonst [___] okay und sie… |
| 484 | Löschung | `okay` | `*(nicht da)*` | …was sonst noch [___] und sie haben… |
| 485 | Löschung | `und` | `*(nicht da)*` | …sonst noch okay [___] sie haben gesagt… |
| 486 | Löschung | `sie` | `*(nicht da)*` | …noch okay und [___] haben gesagt sie… |
| 487 | Löschung | `haben` | `*(nicht da)*` | …okay und sie [___] gesagt sie haben… |
| 488 | Löschung | `gesagt` | `*(nicht da)*` | …und sie haben [___] sie haben eben… |
| 489 | Löschung | `sie` | `*(nicht da)*` | …sie haben gesagt [___] haben eben mit… |
| 490 | Löschung | `haben` | `*(nicht da)*` | …haben gesagt sie [___] eben mit dem… |
| 491 | Löschung | `eben` | `*(nicht da)*` | …gesagt sie haben [___] mit dem gehen… |
| 492 | Löschung | `mit` | `*(nicht da)*` | …sie haben eben [___] dem gehen mit… |
| 493 | Löschung | `dem` | `*(nicht da)*` | …haben eben mit [___] gehen mit den… |
| 494 | Löschung | `gehen` | `*(nicht da)*` | …eben mit dem [___] mit den stützen… |
| 495 | Löschung | `mit` | `*(nicht da)*` | …mit dem gehen [___] den stützen das… |
| 496 | Löschung | `den` | `*(nicht da)*` | …dem gehen mit [___] stützen das funktioniert… |
| 497 | Löschung | `stützen` | `*(nicht da)*` | …gehen mit den [___] das funktioniert nur… |
| 498 | Löschung | `das` | `*(nicht da)*` | …mit den stützen [___] funktioniert nur kurz… |
| 499 | Löschung | `funktioniert` | `*(nicht da)*` | …den stützen das [___] nur kurz was… |
| 500 | Löschung | `nur` | `*(nicht da)*` | …stützen das funktioniert [___] kurz was können… |
| 501 | Löschung | `kurz` | `*(nicht da)*` | …das funktioniert nur [___] was können wir… |
| 502 | Löschung | `was` | `*(nicht da)*` | …funktioniert nur kurz [___] können wir da… |
| 503 | Löschung | `können` | `*(nicht da)*` | …nur kurz was [___] wir da forschen… |
| 504 | Löschung | `wir` | `*(nicht da)*` | …kurz was können [___] da forschen also… |
| 505 | Löschung | `da` | `*(nicht da)*` | …was können wir [___] forschen also sind… |
| 506 | Löschung | `forschen` | `*(nicht da)*` | …können wir da [___] also sind sie… |
| 507 | Löschung | `also` | `*(nicht da)*` | …wir da forschen [___] sind sie auf… |
| 508 | Löschung | `sind` | `*(nicht da)*` | …da forschen also [___] sie auf und… |
| 509 | Löschung | `sie` | `*(nicht da)*` | …forschen also sind [___] auf und zu… |
| 510 | Substitution | `auf` | `kann` | …also sind sie [___] und zu rausgegangen… |
| 511 | Löschung | `zu` | `*(nicht da)*` | …sie auf und [___] rausgegangen nein jetzt… |
| 512 | Löschung | `rausgegangen` | `*(nicht da)*` | …auf und zu [___] nein jetzt nicht… |
| 513 | Löschung | `nein` | `*(nicht da)*` | …und zu rausgegangen [___] jetzt nicht also… |
| 514 | Löschung | `jetzt` | `*(nicht da)*` | …zu rausgegangen nein [___] nicht also das… |
| 515 | Löschung | `nicht` | `*(nicht da)*` | …rausgegangen nein jetzt [___] also das ist… |
| 516 | Löschung | `also` | `*(nicht da)*` | …nein jetzt nicht [___] das ist jetzt… |
| 517 | Löschung | `ist` | `*(nicht da)*` | …nicht also das [___] jetzt auch ein… |
| 518 | Löschung | `jetzt` | `*(nicht da)*` | …also das ist [___] auch ein monat… |
| 519 | Löschung | `auch` | `*(nicht da)*` | …das ist jetzt [___] ein monat her… |
| 520 | Löschung | `ein` | `*(nicht da)*` | …ist jetzt auch [___] monat her also… |
| 521 | Löschung | `monat` | `*(nicht da)*` | …jetzt auch ein [___] her also nicht… |
| 522 | Löschung | `her` | `*(nicht da)*` | …auch ein monat [___] also nicht wirklich… |
| 523 | Löschung | `also` | `*(nicht da)*` | …ein monat her [___] nicht wirklich ich… |
| 524 | Löschung | `nicht` | `*(nicht da)*` | …monat her also [___] wirklich ich meine… |
| 525 | Löschung | `wirklich` | `*(nicht da)*` | …her also nicht [___] ich meine minimal… |
| 526 | Löschung | `ich` | `*(nicht da)*` | …also nicht wirklich [___] meine minimal einfach… |
| 527 | Löschung | `meine` | `*(nicht da)*` | …nicht wirklich ich [___] minimal einfach aber… |
| 528 | Löschung | `minimal` | `*(nicht da)*` | …wirklich ich meine [___] einfach aber ich… |
| 529 | Löschung | `einfach` | `*(nicht da)*` | …ich meine minimal [___] aber ich kann… |
| 530 | Löschung | `aber` | `*(nicht da)*` | …meine minimal einfach [___] ich kann nicht… |
| 531 | Löschung | `ich` | `*(nicht da)*` | …minimal einfach aber [___] kann nicht wirklich… |
| 532 | Löschung | `kann` | `*(nicht da)*` | …einfach aber ich [___] nicht wirklich zusammenkriegen… |
| 533 | Löschung | `nicht` | `*(nicht da)*` | …aber ich kann [___] wirklich zusammenkriegen jetzt… |
| 534 | Löschung | `wirklich` | `*(nicht da)*` | …ich kann nicht [___] zusammenkriegen jetzt spazieren… |
| 535 | Löschung | `zusammenkriegen` | `*(nicht da)*` | …kann nicht wirklich [___] jetzt spazieren oder… |
| 536 | Löschung | `jetzt` | `*(nicht da)*` | …nicht wirklich zusammenkriegen [___] spazieren oder so… |
| 537 | Löschung | `spazieren` | `*(nicht da)*` | …wirklich zusammenkriegen jetzt [___] oder so also… |
| 538 | Substitution | `oder` | `ging` | …zusammenkriegen jetzt spazieren [___] so also ich… |
| 539 | Substitution | `also` | `dass` | …spazieren oder so [___] ich bewege mich… |
| 540 | Löschung | `bewege` | `*(nicht da)*` | …so also ich [___] mich halt in… |
| 541 | Löschung | `mich` | `*(nicht da)*` | …also ich bewege [___] halt in der… |
| 542 | Löschung | `halt` | `*(nicht da)*` | …ich bewege mich [___] in der wohnung… |
| 543 | Löschung | `in` | `*(nicht da)*` | …bewege mich halt [___] der wohnung was… |
| 544 | Löschung | `der` | `*(nicht da)*` | …mich halt in [___] wohnung was das… |
| 545 | Löschung | `wohnung` | `*(nicht da)*` | …halt in der [___] was das nötigste… |
| 546 | Löschung | `was` | `*(nicht da)*` | …in der wohnung [___] das nötigste und… |
| 547 | Löschung | `das` | `*(nicht da)*` | …der wohnung was [___] nötigste und ja… |
| 548 | Löschung | `nötigste` | `*(nicht da)*` | …wohnung was das [___] und ja versuche… |
| 549 | Löschung | `und` | `*(nicht da)*` | …was das nötigste [___] ja versuche halt… |
| 550 | Löschung | `ja` | `*(nicht da)*` | …das nötigste und [___] versuche halt am… |
| 551 | Löschung | `versuche` | `*(nicht da)*` | …nötigste und ja [___] halt am heimtrainer… |
| 552 | Löschung | `halt` | `*(nicht da)*` | …und ja versuche [___] am heimtrainer ab… |
| 553 | Löschung | `am` | `*(nicht da)*` | …ja versuche halt [___] heimtrainer ab und… |
| 554 | Löschung | `heimtrainer` | `*(nicht da)*` | …versuche halt am [___] ab und zu… |
| 555 | Löschung | `ab` | `*(nicht da)*` | …halt am heimtrainer [___] und zu so… |
| 556 | Löschung | `und` | `*(nicht da)*` | …am heimtrainer ab [___] zu so weit… |
| 557 | Löschung | `zu` | `*(nicht da)*` | …heimtrainer ab und [___] so weit wie… |
| 558 | Löschung | `so` | `*(nicht da)*` | …ab und zu [___] weit wie möglich… |
| 559 | Löschung | `weit` | `*(nicht da)*` | …und zu so [___] wie möglich zu… |
| 560 | Löschung | `wie` | `*(nicht da)*` | …zu so weit [___] möglich zu beugen… |
| 561 | Löschung | `möglich` | `*(nicht da)*` | …so weit wie [___] zu beugen und… |
| 562 | Löschung | `zu` | `*(nicht da)*` | …weit wie möglich [___] beugen und das… |
| 563 | Löschung | `beugen` | `*(nicht da)*` | …wie möglich zu [___] und das eigentlich… |
| 564 | Löschung | `und` | `*(nicht da)*` | …möglich zu beugen [___] das eigentlich immer… |
| 565 | Löschung | `das` | `*(nicht da)*` | …zu beugen und [___] eigentlich immer unter… |
| 566 | Löschung | `eigentlich` | `*(nicht da)*` | …beugen und das [___] immer unter schmerzen… |
| 567 | Löschung | `unter` | `*(nicht da)*` | …das eigentlich immer [___] schmerzen dann wenn… |
| 568 | Löschung | `schmerzen` | `*(nicht da)*` | …eigentlich immer unter [___] dann wenn man… |
| 569 | Löschung | `dann` | `*(nicht da)*` | …immer unter schmerzen [___] wenn man sagt… |
| 570 | Löschung | `wenn` | `*(nicht da)*` | …unter schmerzen dann [___] man sagt mit… |
| 571 | Löschung | `man` | `*(nicht da)*` | …schmerzen dann wenn [___] sagt mit der… |
| 572 | Löschung | `sagt` | `*(nicht da)*` | …dann wenn man [___] mit der belastung… |
| 573 | Löschung | `mit` | `*(nicht da)*` | …wenn man sagt [___] der belastung variiert… |
| 574 | Löschung | `der` | `*(nicht da)*` | …man sagt mit [___] belastung variiert aber… |
| 575 | Löschung | `belastung` | `*(nicht da)*` | …sagt mit der [___] variiert aber ist… |
| 576 | Löschung | `variiert` | `*(nicht da)*` | …mit der belastung [___] aber ist noch… |
| 577 | Löschung | `aber` | `*(nicht da)*` | …der belastung variiert [___] ist noch nicht… |
| 578 | Löschung | `ist` | `*(nicht da)*` | …belastung variiert aber [___] noch nicht richtig… |
| 579 | Löschung | `noch` | `*(nicht da)*` | …variiert aber ist [___] nicht richtig schmerzfrei… |
| 580 | Löschung | `nicht` | `*(nicht da)*` | …aber ist noch [___] richtig schmerzfrei möglich… |
| 581 | Löschung | `richtig` | `*(nicht da)*` | …ist noch nicht [___] schmerzfrei möglich nein… |
| 582 | Löschung | `schmerzfrei` | `*(nicht da)*` | …noch nicht richtig [___] möglich nein nehmen… |
| 583 | Löschung | `möglich` | `*(nicht da)*` | …nicht richtig schmerzfrei [___] nein nehmen sie… |
| 584 | Löschung | `nein` | `*(nicht da)*` | …richtig schmerzfrei möglich [___] nehmen sie irgendwelche… |
| 585 | Löschung | `nehmen` | `*(nicht da)*` | …schmerzfrei möglich nein [___] sie irgendwelche medikamente… |
| 586 | Löschung | `sie` | `*(nicht da)*` | …möglich nein nehmen [___] irgendwelche medikamente nein… |
| 587 | Löschung | `irgendwelche` | `*(nicht da)*` | …nein nehmen sie [___] medikamente nein nehmen… |
| 588 | Löschung | `medikamente` | `*(nicht da)*` | …nehmen sie irgendwelche [___] nein nehmen sie… |
| 589 | Löschung | `nein` | `*(nicht da)*` | …sie irgendwelche medikamente [___] nehmen sie nichts… |
| 590 | Löschung | `nehmen` | `*(nicht da)*` | …irgendwelche medikamente nein [___] sie nichts nehmen… |
| 591 | Löschung | `sie` | `*(nicht da)*` | …medikamente nein nehmen [___] nichts nehmen sie… |
| 592 | Löschung | `nichts` | `*(nicht da)*` | …nein nehmen sie [___] nehmen sie nichts… |
| 593 | Löschung | `nehmen` | `*(nicht da)*` | …nehmen sie nichts [___] sie nichts haben… |
| 594 | Löschung | `sie` | `*(nicht da)*` | …sie nichts nehmen [___] nichts haben sie… |
| 595 | Löschung | `nichts` | `*(nicht da)*` | …nichts nehmen sie [___] haben sie anfangs… |
| 596 | Löschung | `haben` | `*(nicht da)*` | …nehmen sie nichts [___] sie anfangs aber… |
| 597 | Löschung | `sie` | `*(nicht da)*` | …sie nichts haben [___] anfangs aber wahrscheinlich… |
| 598 | Löschung | `anfangs` | `*(nicht da)*` | …nichts haben sie [___] aber wahrscheinlich eine… |
| 599 | Löschung | `aber` | `*(nicht da)*` | …haben sie anfangs [___] wahrscheinlich eine behandlung… |
| 600 | Löschung | `wahrscheinlich` | `*(nicht da)*` | …sie anfangs aber [___] eine behandlung ja… |
| 601 | Löschung | `eine` | `*(nicht da)*` | …anfangs aber wahrscheinlich [___] behandlung ja ich… |
| 602 | Löschung | `behandlung` | `*(nicht da)*` | …aber wahrscheinlich eine [___] ja ich habe… |
| 603 | Löschung | `ja` | `*(nicht da)*` | …wahrscheinlich eine behandlung [___] ich habe manchmal… |
| 604 | Löschung | `ich` | `*(nicht da)*` | …eine behandlung ja [___] habe manchmal schmerzmittel… |
| 605 | Löschung | `habe` | `*(nicht da)*` | …behandlung ja ich [___] manchmal schmerzmittel gekriegt… |
| 606 | Löschung | `manchmal` | `*(nicht da)*` | …ja ich habe [___] schmerzmittel gekriegt am… |
| 607 | Löschung | `schmerzmittel` | `*(nicht da)*` | …ich habe manchmal [___] gekriegt am anfang… |
| 608 | Löschung | `gekriegt` | `*(nicht da)*` | …habe manchmal schmerzmittel [___] am anfang sowieso… |
| 609 | Löschung | `am` | `*(nicht da)*` | …manchmal schmerzmittel gekriegt [___] anfang sowieso infusionen… |
| 610 | Löschung | `anfang` | `*(nicht da)*` | …schmerzmittel gekriegt am [___] sowieso infusionen gemacht… |
| 611 | Löschung | `sowieso` | `*(nicht da)*` | …gekriegt am anfang [___] infusionen gemacht dann… |
| 612 | Löschung | `infusionen` | `*(nicht da)*` | …am anfang sowieso [___] gemacht dann hätte… |
| 613 | Löschung | `gemacht` | `*(nicht da)*` | …anfang sowieso infusionen [___] dann hätte ich… |
| 614 | Löschung | `dann` | `*(nicht da)*` | …sowieso infusionen gemacht [___] hätte ich nochmal… |
| 615 | Löschung | `hätte` | `*(nicht da)*` | …infusionen gemacht dann [___] ich nochmal schmerzmittel… |
| 616 | Löschung | `ich` | `*(nicht da)*` | …gemacht dann hätte [___] nochmal schmerzmittel mitgehabt… |
| 617 | Löschung | `nochmal` | `*(nicht da)*` | …dann hätte ich [___] schmerzmittel mitgehabt für… |
| 618 | Löschung | `schmerzmittel` | `*(nicht da)*` | …hätte ich nochmal [___] mitgehabt für daheim… |
| 619 | Löschung | `mitgehabt` | `*(nicht da)*` | …ich nochmal schmerzmittel [___] für daheim aber… |
| 620 | Löschung | `für` | `*(nicht da)*` | …nochmal schmerzmittel mitgehabt [___] daheim aber die… |
| 621 | Löschung | `daheim` | `*(nicht da)*` | …schmerzmittel mitgehabt für [___] aber die habe… |
| 622 | Löschung | `aber` | `*(nicht da)*` | …mitgehabt für daheim [___] die habe ich… |
| 623 | Löschung | `die` | `*(nicht da)*` | …für daheim aber [___] habe ich dann… |
| 624 | Löschung | `habe` | `*(nicht da)*` | …daheim aber die [___] ich dann eigentlich… |
| 625 | Löschung | `ich` | `*(nicht da)*` | …aber die habe [___] dann eigentlich nicht… |
| 626 | Löschung | `dann` | `*(nicht da)*` | …die habe ich [___] eigentlich nicht mehr… |
| 627 | Löschung | `eigentlich` | `*(nicht da)*` | …habe ich dann [___] nicht mehr braucht… |
| 628 | Löschung | `nicht` | `*(nicht da)*` | …ich dann eigentlich [___] mehr braucht also… |
| 629 | Löschung | `mehr` | `*(nicht da)*` | …dann eigentlich nicht [___] braucht also haben… |
| 630 | Löschung | `braucht` | `*(nicht da)*` | …eigentlich nicht mehr [___] also haben sie… |
| 631 | Löschung | `also` | `*(nicht da)*` | …nicht mehr braucht [___] haben sie das… |
| 632 | Löschung | `haben` | `*(nicht da)*` | …mehr braucht also [___] sie das benötigt… |
| 633 | Löschung | `sie` | `*(nicht da)*` | …braucht also haben [___] das benötigt mit… |
| 634 | Löschung | `das` | `*(nicht da)*` | …also haben sie [___] benötigt mit dem… |
| 635 | Löschung | `benötigt` | `*(nicht da)*` | …haben sie das [___] mit dem hometrainer… |
| 636 | Löschung | `mit` | `*(nicht da)*` | …sie das benötigt [___] dem hometrainer haben… |
| 637 | Löschung | `dem` | `*(nicht da)*` | …das benötigt mit [___] hometrainer haben sie… |
| 638 | Löschung | `hometrainer` | `*(nicht da)*` | …benötigt mit dem [___] haben sie erwähnt… |
| 639 | Löschung | `haben` | `*(nicht da)*` | …mit dem hometrainer [___] sie erwähnt was… |
| 640 | Löschung | `sie` | `*(nicht da)*` | …dem hometrainer haben [___] erwähnt was haben… |
| 641 | Löschung | `erwähnt` | `*(nicht da)*` | …hometrainer haben sie [___] was haben sie… |
| 642 | Löschung | `was` | `*(nicht da)*` | …haben sie erwähnt [___] haben sie da… |
| 643 | Löschung | `haben` | `*(nicht da)*` | …sie erwähnt was [___] sie da genau… |
| 644 | Löschung | `sie` | `*(nicht da)*` | …erwähnt was haben [___] da genau gemacht… |
| 645 | Löschung | `da` | `*(nicht da)*` | …was haben sie [___] genau gemacht für… |
| 646 | Löschung | `genau` | `*(nicht da)*` | …haben sie da [___] gemacht für übungen… |
| 647 | Löschung | `gemacht` | `*(nicht da)*` | …sie da genau [___] für übungen nein… |
| 648 | Löschung | `für` | `*(nicht da)*` | …da genau gemacht [___] übungen nein eigentlich… |
| 649 | Löschung | `übungen` | `*(nicht da)*` | …genau gemacht für [___] nein eigentlich nur… |
| 650 | Löschung | `nein` | `*(nicht da)*` | …gemacht für übungen [___] eigentlich nur versucht… |
| 651 | Löschung | `eigentlich` | `*(nicht da)*` | …für übungen nein [___] nur versucht weil… |
| 652 | Löschung | `versucht` | `*(nicht da)*` | …nein eigentlich nur [___] weil ich eben… |
| 653 | Löschung | `weil` | `*(nicht da)*` | …eigentlich nur versucht [___] ich eben schon… |
| 654 | Löschung | `ich` | `*(nicht da)*` | …nur versucht weil [___] eben schon beugen… |
| 655 | Löschung | `eben` | `*(nicht da)*` | …versucht weil ich [___] schon beugen und… |
| 656 | Löschung | `schon` | `*(nicht da)*` | …weil ich eben [___] beugen und strecken… |
| 657 | Löschung | `beugen` | `*(nicht da)*` | …ich eben schon [___] und strecken kann… |
| 658 | Löschung | `und` | `*(nicht da)*` | …eben schon beugen [___] strecken kann also… |
| 659 | Löschung | `strecken` | `*(nicht da)*` | …schon beugen und [___] kann also dass… |
| 660 | Löschung | `kann` | `*(nicht da)*` | …beugen und strecken [___] also dass ich… |
| 661 | Löschung | `also` | `*(nicht da)*` | …und strecken kann [___] dass ich ein… |
| 662 | Löschung | `dass` | `*(nicht da)*` | …strecken kann also [___] ich ein bisschen… |
| 663 | Löschung | `ich` | `*(nicht da)*` | …kann also dass [___] ein bisschen bewegung… |
| 664 | Löschung | `bewegung` | `*(nicht da)*` | …ich ein bisschen [___] habe drinnen halt… |
| 665 | Löschung | `habe` | `*(nicht da)*` | …ein bisschen bewegung [___] drinnen halt und… |
| 666 | Löschung | `drinnen` | `*(nicht da)*` | …bisschen bewegung habe [___] halt und da… |
| 667 | Löschung | `halt` | `*(nicht da)*` | …bewegung habe drinnen [___] und da ist… |
| 668 | Löschung | `und` | `*(nicht da)*` | …habe drinnen halt [___] da ist schon… |
| 669 | Löschung | `da` | `*(nicht da)*` | …drinnen halt und [___] ist schon ist… |
| 670 | Löschung | `ist` | `*(nicht da)*` | …halt und da [___] schon ist ihnen… |
| 671 | Substitution | `schon` | `mehr` | …und da ist [___] ist ihnen da… |
| 672 | Substitution | `ist` | `konnte` | …da ist schon [___] ihnen da auch… |
| 673 | Substitution | `ihnen` | `als` | …ist schon ist [___] da auch aufgefallen… |
| 674 | Substitution | `da` | `am` | …schon ist ihnen [___] auch aufgefallen dass… |
| 675 | Substitution | `auch` | `tag` | …ist ihnen da [___] aufgefallen dass es… |
| 676 | Substitution | `aufgefallen` | `zuvor` | …ihnen da auch [___] dass es einfach… |
| 677 | Substitution | `dass` | `also` | …da auch aufgefallen [___] es einfach schon… |
| 678 | Löschung | `einfach` | `*(nicht da)*` | …aufgefallen dass es [___] schon ein bisschen… |
| 679 | Löschung | `schon` | `*(nicht da)*` | …dass es einfach [___] ein bisschen weitergegangen… |
| 680 | Löschung | `ein` | `*(nicht da)*` | …es einfach schon [___] bisschen weitergegangen ist… |
| 681 | Löschung | `bisschen` | `*(nicht da)*` | …einfach schon ein [___] weitergegangen ist ja… |
| 682 | Löschung | `weitergegangen` | `*(nicht da)*` | …schon ein bisschen [___] ist ja die… |
| 683 | Löschung | `ist` | `*(nicht da)*` | …ein bisschen weitergegangen [___] ja die bewegung… |
| 684 | Löschung | `ja` | `*(nicht da)*` | …bisschen weitergegangen ist [___] die bewegung auf… |
| 685 | Löschung | `die` | `*(nicht da)*` | …weitergegangen ist ja [___] bewegung auf jeden… |
| 686 | Löschung | `bewegung` | `*(nicht da)*` | …ist ja die [___] auf jeden fall… |
| 687 | Löschung | `auf` | `*(nicht da)*` | …ja die bewegung [___] jeden fall besser… |
| 688 | Löschung | `jeden` | `*(nicht da)*` | …die bewegung auf [___] fall besser als… |
| 689 | Löschung | `fall` | `*(nicht da)*` | …bewegung auf jeden [___] besser als am… |
| 690 | Löschung | `besser` | `*(nicht da)*` | …auf jeden fall [___] als am anfang… |
| 691 | Substitution | `als` | `war` | …jeden fall besser [___] am anfang also… |
| 692 | Substitution | `am` | `langsam` | …fall besser als [___] anfang also sie… |
| 693 | Substitution | `anfang` | `fortschritte` | …besser als am [___] also sie haben… |
| 694 | Substitution | `also` | `aber` | …als am anfang [___] sie haben auch… |
| 695 | Löschung | `in` | `*(nicht da)*` | …in einem haus [___] einer wohnung in… |
| 696 | Löschung | `einer` | `*(nicht da)*` | …einem haus in [___] wohnung in einer… |
| 697 | Löschung | `wohnung` | `*(nicht da)*` | …haus in einer [___] in einer wohnung… |
| 698 | Löschung | `ja` | `*(nicht da)*` | …wirklich runter muss [___] haben sie irgendeine… |
| 699 | Substitution | `sie` | `sich` | …was würden sie [___] wünschen was erwarten… |
| 700 | Löschung | `sie` | `*(nicht da)*` | …wünschen was erwarten [___] sie dass ich… |
| 701 | Löschung | `okay` | `*(nicht da)*` | …lüge und ja [___] vielen dank frau… |
| 702 | Substitution | `krebspartner` | `grasbeutner` | …vielen dank frau [___] und wir treffen… |
| 703 | Einfügung | `*(nicht da)*` | `danke` | (FMT) …sie das passen [___] ich hoffe dass… |
