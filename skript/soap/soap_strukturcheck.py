"""
SOAP Strukturcheck für alle 55 Durchläufe.
Prüft: Sind alle 4 Abschnitte (S/O/A/P) vorhanden? Sind sie nicht leer?
Wie viele Wörter hat jeder Abschnitt?
"""
import json, re
from pathlib import Path

REPO      = Path(__file__).parent.parent.parent
DATA_FILE = REPO / "results" / "history_no_speaker.json"
OUT_FILE  = REPO / "docs" / "soap_strukturcheck.md"

PUNCT = re.compile(r'[.,!?;:\-–—()\[\]"\'„"«»/]')

# ── Szenario-Mapping ───────────────────────────────────────────────────────
AUDIO_MAP = {
    'OriginalDC.m4a':                'OriginalDC',
    'OriginalDCWhiteNoise.m4a':      'OriginalDC+Noise',
    'OriginalLapInMitte.wav':        'LapInMitte',
    'OriginalLapBeiArzt.wav':        'LapBeiArzt',
    'SelbstkorrekturLapInMitte.wav': 'Selbstkorrekturen',
    'UnterbrechungLapInMitte.wav':   'Unterbrechungen',
    'GedankenprüngeLapInMitte.wav':  'Gedankensprünge',
    'MeinungswechselLapinMitte.wav': 'Meinungswechsel',
    'ChaosLapInMitte.wav':           'Chaos',
}

def scenario(audio_file):
    b = audio_file.split('/')[-1]
    if b in AUDIO_MAP: return AUDIO_MAP[b]
    if 'PWC' in b: return 'PWC'
    if 'Anamnesegesr' in b or 'Anamnesegespräch' in b: return 'Anamnesegespräch'
    return b

def model_label(e):
    stt = e.get('stt_model', '')
    llm = e.get('llm_model', '')
    if 'Speechmatics' in stt:   return 'Speechmatics+GPT4o'
    if 'AssemblyAI'  in stt:    return 'AssemblyAI+GPT4o'
    if 'gemma4'      in llm:    return 'Whisper+gemma4'
    if 'Sauerkraut'  in llm:    return 'Whisper+Sauerkraut'
    if 'llama3.2'    in llm:    return 'Whisper+llama3.2'
    return f'{stt} + {llm}'

MODEL_ORDER = [
    'Whisper+llama3.2',
    'Speechmatics+GPT4o',
    'AssemblyAI+GPT4o',
    'Whisper+Sauerkraut',
    'Whisper+gemma4',
]

SCENARIO_ORDER = [
    'OriginalDC', 'OriginalDC+Noise', 'LapInMitte', 'LapBeiArzt',
    'Selbstkorrekturen', 'Unterbrechungen', 'Gedankensprünge',
    'Meinungswechsel', 'Chaos', 'Anamnesegespräch', 'PWC',
]

# ── SOAP-Abschnitte erkennen ───────────────────────────────────────────────
# Passt auf: S:, S (Subjektiv):, **S (Subjektiv):**, S (Subjective): usw.
# Kein (?<!\w) — Abschnittsbuchstabe kann direkt am Ende des Vorgänger-Inhalts kleben
# (z.B. "...SchmerzenA (Assessment):"). Spezifität kommt durch das Folgemuster.
SECTION_SPLIT = re.compile(
    r'\*{0,2}'           # optionale Markdown-Sterne
    r'([SOAP])'          # Abschnitts-Buchstabe
    r'\s*'
    r'(?:\([^)]*\))?'    # optionaler Klammerinhalt: (Subjektiv) etc.
    r'\s*[:\*]',         # Doppelpunkt oder Stern
    re.MULTILINE
)

def word_count(text):
    norm = PUNCT.sub(' ', text).lower()
    return len([w for w in norm.split() if w])

def check_soap(soap_text):
    """Gibt dict mit Ergebnis pro Abschnitt zurück."""
    result = {sec: {'present': False, 'words': 0} for sec in 'SOAP'}

    if not soap_text or not soap_text.strip():
        return result

    # Alle Treffer finden und nach Position sortieren
    matches = list(SECTION_SPLIT.finditer(soap_text))

    if not matches:
        # Kein strukturiertes Format gefunden — SOAP existiert aber ist unstrukturiert
        return result

    # Text zwischen aufeinanderfolgenden Headern als Sektionsinhalt nehmen
    for i, m in enumerate(matches):
        sec = m.group(1).upper()
        if sec not in result:
            continue
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(soap_text)
        content = soap_text[start:end].strip()
        result[sec]['present'] = True
        result[sec]['words']   = word_count(content)

    return result

def score(r):
    """0–4: Anzahl vorhandener nicht-leerer Abschnitte."""
    return sum(1 for s in 'SOAP' if r[s]['present'] and r[s]['words'] > 0)

def fmt_cell(r, sec):
    if not r[sec]['present']:
        return '✗'
    w = r[sec]['words']
    if w == 0:
        return '⚠ leer'
    return f'✓ ({w}w)'

# ── Hauptprogramm ──────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    data = json.load(f)

print(f'Einträge gesamt: {len(data)}')

# Analyse
rows = []
for e in data:
    scen  = scenario(e['audio_file'])
    model = model_label(e)
    soap  = e.get('soap', '') or ''
    r     = check_soap(soap)
    sc    = score(r)
    total = word_count(soap)
    rows.append(dict(scenario=scen, model=model, result=r,
                     score=sc, total_words=total, soap_empty=(total == 0)))
    status = '✓✓✓✓' if sc == 4 else f'{sc}/4 ⚠'
    print(f'  {model:22s} {scen:20s} {status}  total={total}w')

# ── Markdown ───────────────────────────────────────────────────────────────
out = []
out.append('# SOAP Strukturcheck')
out.append('')
out.append('> Prüft ob alle 4 SOAP-Abschnitte (S/O/A/P) vorhanden und nicht leer sind.')
out.append('> ✓ (Nw) = vorhanden mit N Wörtern | ✗ = fehlt | ⚠ leer = Header da, aber kein Inhalt')
out.append('')
out.append('---')
out.append('')

# ── Übersichtstabelle pro Modell ───────────────────────────────────────────
for model in MODEL_ORDER:
    model_rows = [r for r in rows if r['model'] == model]
    if not model_rows: continue

    out.append(f'## {model}')
    out.append('')
    out.append('| Szenario | S | O | A | P | Score | Gesamt |')
    out.append('|---|---|---|---|---|---|---|')

    perfect = 0
    for scen in SCENARIO_ORDER:
        row = next((r for r in model_rows if r['scenario'] == scen), None)
        if row is None: continue
        r  = row['result']
        sc = row['score']
        perfect += (sc == 4)
        flag = '' if sc == 4 else ' ⚠'
        out.append(
            f"| {scen} "
            f"| {fmt_cell(r,'S')} "
            f"| {fmt_cell(r,'O')} "
            f"| {fmt_cell(r,'A')} "
            f"| {fmt_cell(r,'P')} "
            f"| **{sc}/4**{flag} "
            f"| {row['total_words']}w |"
        )

    out.append('')
    out.append(f'*Vollständig (4/4): {perfect}/{len(model_rows)} Szenarien*')
    out.append('')

# ── Zusammenfassung ────────────────────────────────────────────────────────
out.append('---')
out.append('')
out.append('## Zusammenfassung')
out.append('')
out.append('| Modell | 4/4 ✓ | 3/4 | 2/4 | 1/4 | 0/4 | Ø Wörter |')
out.append('|---|---|---|---|---|---|---|')

for model in MODEL_ORDER:
    model_rows = [r for r in rows if r['model'] == model]
    if not model_rows: continue
    counts = {i: sum(1 for r in model_rows if r['score'] == i) for i in range(5)}
    avg_w  = sum(r['total_words'] for r in model_rows) / len(model_rows)
    out.append(
        f"| {model} "
        f"| {counts[4]} "
        f"| {counts[3]} "
        f"| {counts[2]} "
        f"| {counts[1]} "
        f"| {counts[0]} "
        f"| {avg_w:.0f}w |"
    )

OUT_FILE.write_text('\n'.join(out) + '\n')
print(f'\nGeschrieben: {OUT_FILE}')
