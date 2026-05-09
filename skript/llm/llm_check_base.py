"""
Gemeinsame Logik für alle LLM-Fehleranalyse-Scripts.
Vergleicht RAW STT-Output mit LLM-formatiertem Output (aus history_no_speaker.json).
Speaker-Label-Änderungen sind bereits aus der JSON entfernt.
"""
import json, re
from pathlib import Path

REPO      = Path(__file__).parent.parent.parent
DATA_FILE = REPO / "results" / "history_no_speaker.json"

PUNCT = re.compile(r'[.,!?;:\-–—()\[\]"\'„"«»/]')

ORDER = [
    'OriginalDC', 'OriginalDC+Noise', 'LapInMitte', 'LapBeiArzt',
    'Selbstkorrekturen', 'Unterbrechungen', 'Gedankensprünge',
    'Meinungswechsel', 'Chaos', 'Anamnesegespräch', 'PWC',
]

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

def to_words(text):
    norm = PUNCT.sub(' ', text).lower()
    return [w for w in norm.split() if w]

def align(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    ops, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            ops.append(('OK', ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('S',  ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('D',  ref[i-1], None));      i -= 1
        else:
            ops.append(('I',  None, hyp[j-1]));      j -= 1
    ops.reverse()
    return ops

def ctx(words, pos, w=3):
    lo, hi = max(0, pos - w), min(len(words), pos + w + 1)
    parts = list(words[lo:hi])
    rel = pos - lo
    parts[rel:rel+1] = ['[___]']
    return '…' + ' '.join(parts) + '…'

def run(stt_filter, llm_filter, title, out_file):
    with open(DATA_FILE) as f:
        data = json.load(f)

    entries = [
        e for e in data
        if stt_filter.lower() in e.get('stt_model', '').lower()
        and llm_filter.lower() in e.get('llm_model', '').lower()
    ]
    print(f'{title}: {len(entries)} Einträge gefunden')

    results = {}
    for e in entries:
        name  = scenario(e['audio_file'])
        raw_w = to_words(e['raw'])
        fmt_w = to_words(e['formatted'])
        ops   = align(raw_w, fmt_w)

        errors, S, D, I = [], 0, 0, 0
        rp = fp = 0
        for op, r, h in ops:
            if op == 'S':
                errors.append(('Substitution', r, h, ctx(raw_w, rp))); S += 1; rp += 1; fp += 1
            elif op == 'D':
                errors.append(('Löschung', r, '*(nicht da)*', ctx(raw_w, rp))); D += 1; rp += 1
            elif op == 'I':
                errors.append(('Einfügung', '*(nicht da)*', h, '(FMT) ' + ctx(fmt_w, fp))); I += 1; fp += 1
            else:
                rp += 1; fp += 1

        total = S + D + I
        rate  = total / len(raw_w) * 100 if raw_w else 0
        results[name] = dict(raw=len(raw_w), fmt=len(fmt_w), S=S, D=D, I=I,
                             total=total, rate=rate, errors=errors)
        print(f'  {name:25s} raw={len(raw_w):4d} fmt={len(fmt_w):4d}  S={S} D={D} I={I}  {rate:.1f}%')

    out = []
    out.append(f'# LLM-Fehleranalyse: {title}')
    out.append('')
    out.append('> RAW STT → Formatted — Satzzeichen und Groß-/Kleinschreibung ignoriert.')
    out.append('> Speaker-Label-Änderungen sind bereits aus der JSON entfernt.')
    out.append('> **S** = Substitution | **D** = Löschung (im RAW, fehlt im FMT) | **I** = Einfügung (im FMT, nicht im RAW)')
    out.append('')
    out.append('---')
    out.append('')
    out.append('## Übersicht')
    out.append('')
    out.append('| Szenario | RAW-Wörter | FMT-Wörter | S | D | I | Fehler | Fehlerrate |')
    out.append('|---|---|---|---|---|---|---|---|')
    for name in ORDER:
        if name not in results: continue
        r = results[name]
        out.append(f"| {name} | {r['raw']} | {r['fmt']} | {r['S']} | {r['D']} | {r['I']} | {r['total']} | {r['rate']:.1f}% |")

    for name in ORDER:
        if name not in results: continue
        r = results[name]
        out.append('')
        out.append('---')
        out.append('')
        out.append(f'## {name}')
        out.append('')
        out.append(f"**Fehlerrate: {r['rate']:.1f}%** — RAW: {r['raw']} Wörter | FMT: {r['fmt']} Wörter "
                   f"| S={r['S']} D={r['D']} I={r['I']} | Fehler={r['total']}")
        out.append('')
        if not r['errors']:
            out.append('*Keine Fehler gefunden.*')
            continue
        out.append('| # | Typ | RAW | FORMATTED | Kontext |')
        out.append('|---|-----|-----|-----------|---------|')
        for i, (typ, rw, fw, c) in enumerate(r['errors'], 1):
            out.append(f'| {i} | {typ} | `{rw}` | `{fw}` | {c} |')

    out_path = REPO / "docs" / out_file
    out_path.write_text('\n'.join(out) + '\n')
    print(f'\nGeschrieben: {out_path}')
