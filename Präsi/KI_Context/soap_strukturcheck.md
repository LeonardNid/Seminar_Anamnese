# SOAP Strukturcheck

> Prüft ob alle 4 SOAP-Abschnitte (S/O/A/P) vorhanden und nicht leer sind.
> ✓ (Nw) = vorhanden mit N Wörtern | ✗ = fehlt | ⚠ leer = Header da, aber kein Inhalt

---

## Whisper+llama3.2

| Szenario | S | O | A | P | Score | Gesamt |
|---|---|---|---|---|---|---|
| OriginalDC | ✓ (57w) | ✓ (21w) | ✓ (34w) | ✓ (30w) | **4/4** | 145w |
| OriginalDC+Noise | ✓ (53w) | ✓ (22w) | ✓ (11w) | ✓ (19w) | **4/4** | 111w |
| LapInMitte | ✓ (51w) | ✓ (29w) | ✓ (26w) | ✓ (52w) | **4/4** | 162w |
| LapBeiArzt | ✓ (52w) | ✓ (23w) | ✓ (45w) | ✓ (30w) | **4/4** | 153w |
| Selbstkorrekturen | ✓ (57w) | ✓ (20w) | ✓ (31w) | ✓ (34w) | **4/4** | 150w |
| Unterbrechungen | ✓ (55w) | ✓ (34w) | ✓ (41w) | ✓ (52w) | **4/4** | 186w |
| Gedankensprünge | ✓ (34w) | ✓ (17w) | ✓ (28w) | ✓ (30w) | **4/4** | 113w |
| Meinungswechsel | ✓ (52w) | ✓ (12w) | ✓ (23w) | ✓ (38w) | **4/4** | 132w |
| Chaos | ✓ (47w) | ✓ (29w) | ✓ (39w) | ✓ (34w) | **4/4** | 157w |
| Anamnesegespräch | ✓ (96w) | ✓ (48w) | ✓ (36w) | ✓ (73w) | **4/4** | 261w |
| PWC | ✓ (84w) | ✓ (66w) | ✓ (20w) | ✓ (37w) | **4/4** | 214w |

*Vollständig (4/4): 11/11 Szenarien*

## Speechmatics+GPT4o

| Szenario | S | O | A | P | Score | Gesamt |
|---|---|---|---|---|---|---|
| OriginalDC | ✓ (42w) | ✓ (18w) | ✓ (20w) | ✓ (50w) | **4/4** | 138w |
| OriginalDC+Noise | ✓ (40w) | ✓ (33w) | ✓ (25w) | ✓ (54w) | **4/4** | 160w |
| LapInMitte | ✓ (45w) | ✓ (33w) | ✓ (27w) | ✓ (62w) | **4/4** | 175w |
| LapBeiArzt | ✓ (38w) | ✓ (10w) | ✓ (21w) | ✓ (44w) | **4/4** | 121w |
| Selbstkorrekturen | ✓ (46w) | ✓ (7w) | ✓ (27w) | ✓ (51w) | **4/4** | 139w |
| Unterbrechungen | ✓ (62w) | ✓ (11w) | ✓ (17w) | ✓ (50w) | **4/4** | 148w |
| Gedankensprünge | ✓ (46w) | ✓ (13w) | ✓ (8w) | ✓ (35w) | **4/4** | 106w |
| Meinungswechsel | ✓ (49w) | ✓ (6w) | ✓ (13w) | ✓ (37w) | **4/4** | 109w |
| Chaos | ✓ (43w) | ✓ (21w) | ✓ (19w) | ✓ (30w) | **4/4** | 121w |
| Anamnesegespräch | ✓ (112w) | ✓ (35w) | ✓ (26w) | ✓ (54w) | **4/4** | 235w |
| PWC | ✓ (110w) | ✓ (48w) | ✓ (25w) | ✓ (47w) | **4/4** | 238w |

*Vollständig (4/4): 11/11 Szenarien*

## AssemblyAI+GPT4o

| Szenario | S | O | A | P | Score | Gesamt |
|---|---|---|---|---|---|---|
| OriginalDC | ✓ (35w) | ✓ (17w) | ✓ (23w) | ✓ (56w) | **4/4** | 139w |
| OriginalDC+Noise | ✓ (38w) | ✓ (19w) | ✓ (15w) | ✓ (42w) | **4/4** | 122w |
| LapInMitte | ✓ (39w) | ✓ (17w) | ✓ (26w) | ✓ (61w) | **4/4** | 151w |
| LapBeiArzt | ✓ (34w) | ✓ (20w) | ✓ (19w) | ✓ (41w) | **4/4** | 122w |
| Selbstkorrekturen | ✓ (38w) | ✓ (5w) | ✓ (16w) | ✓ (44w) | **4/4** | 111w |
| Unterbrechungen | ✓ (67w) | ✓ (12w) | ✓ (18w) | ✓ (33w) | **4/4** | 138w |
| Gedankensprünge | ✓ (76w) | ✓ (22w) | ✓ (7w) | ✓ (56w) | **4/4** | 169w |
| Meinungswechsel | ✓ (47w) | ✓ (17w) | ✓ (18w) | ✓ (41w) | **4/4** | 131w |
| Chaos | ✓ (36w) | ✓ (22w) | ✓ (24w) | ✓ (26w) | **4/4** | 116w |
| Anamnesegespräch | ✓ (113w) | ✓ (24w) | ✓ (28w) | ✓ (49w) | **4/4** | 222w |
| PWC | ✓ (101w) | ✓ (39w) | ✓ (30w) | ✓ (39w) | **4/4** | 217w |

*Vollständig (4/4): 11/11 Szenarien*

## Whisper+Sauerkraut

| Szenario | S | O | A | P | Score | Gesamt |
|---|---|---|---|---|---|---|
| OriginalDC | ✓ (45w) | ✓ (27w) | ✓ (36w) | ✓ (45w) | **4/4** | 157w |
| OriginalDC+Noise | ✓ (29w) | ✓ (20w) | ✓ (25w) | ✓ (33w) | **4/4** | 111w |
| LapInMitte | ✓ (9w) | ✓ (22w) | ✓ (23w) | ✓ (70w) | **4/4** | 128w |
| LapBeiArzt | ✓ (64w) | ✓ (24w) | ✓ (28w) | ✓ (64w) | **4/4** | 188w |
| Selbstkorrekturen | ✓ (33w) | ✓ (31w) | ✓ (36w) | ✓ (66w) | **4/4** | 170w |
| Unterbrechungen | ✓ (31w) | ✓ (13w) | ✓ (27w) | ✓ (74w) | **4/4** | 149w |
| Gedankensprünge | ✓ (36w) | ✓ (31w) | ✓ (30w) | ✓ (96w) | **4/4** | 197w |
| Meinungswechsel | ✓ (33w) | ✓ (22w) | ✓ (20w) | ✓ (57w) | **4/4** | 134w |
| Chaos | ✓ (39w) | ✓ (34w) | ✓ (27w) | ✓ (13w) | **4/4** | 121w |
| Anamnesegespräch | ✓ (35w) | ✓ (46w) | ✓ (39w) | ✓ (104w) | **4/4** | 228w |
| PWC | ✓ (70w) | ✓ (25w) | ✓ (56w) | ✓ (81w) | **4/4** | 239w |

*Vollständig (4/4): 11/11 Szenarien*

## Whisper+gemma4

| Szenario | S | O | A | P | Score | Gesamt |
|---|---|---|---|---|---|---|
| OriginalDC | ✓ (77w) | ✓ (40w) | ✓ (24w) | ✓ (60w) | **4/4** | 209w |
| OriginalDC+Noise | ✓ (82w) | ✓ (43w) | ✓ (40w) | ✓ (56w) | **4/4** | 229w |
| LapInMitte | ✓ (79w) | ✓ (43w) | ✓ (30w) | ✓ (51w) | **4/4** | 211w |
| LapBeiArzt | ✓ (65w) | ✓ (34w) | ✓ (39w) | ✓ (68w) | **4/4** | 214w |
| Selbstkorrekturen | ✓ (71w) | ✓ (11w) | ✓ (19w) | ✓ (58w) | **4/4** | 167w |
| Unterbrechungen | ✓ (71w) | ✓ (11w) | ✓ (26w) | ✓ (72w) | **4/4** | 188w |
| Gedankensprünge | ✓ (65w) | ✓ (28w) | ✓ (14w) | ✓ (45w) | **4/4** | 160w |
| Meinungswechsel | ✓ (55w) | ✓ (32w) | ✓ (42w) | ✓ (50w) | **4/4** | 187w |
| Chaos | ✓ (58w) | ✓ (38w) | ✓ (45w) | ✓ (50w) | **4/4** | 199w |
| Anamnesegespräch | ✓ (88w) | ✓ (65w) | ✓ (35w) | ✓ (48w) | **4/4** | 244w |
| PWC | ✗ | ✗ | ✗ | ✗ | **0/4** ⚠ | 0w |

*Vollständig (4/4): 10/11 Szenarien*

---

## Zusammenfassung

| Modell | 4/4 ✓ | 3/4 | 2/4 | 1/4 | 0/4 | Ø Wörter |
|---|---|---|---|---|---|---|
| Whisper+llama3.2 | 11 | 0 | 0 | 0 | 0 | 162w |
| Speechmatics+GPT4o | 11 | 0 | 0 | 0 | 0 | 154w |
| AssemblyAI+GPT4o | 11 | 0 | 0 | 0 | 0 | 149w |
| Whisper+Sauerkraut | 11 | 0 | 0 | 0 | 0 | 166w |
| Whisper+gemma4 | 10 | 0 | 0 | 0 | 1 | 183w |
