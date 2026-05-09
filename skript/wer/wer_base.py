"""
Gemeinsame Logik für alle WER-Analyse-Scripts.
Vergleicht RAW STT-Output (aus history_no_speaker.json) mit Ground Truth.
"""
import json, re
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
GT_FILE = REPO / "docs" / "Seminar Ground Truth Texte.md"
DATA_FILE = REPO / "results" / "history_no_speaker.json"

PUNCT = re.compile(r'[.,!?;:\-–—()\[\]"\'„"«»/]')

ORDER = [
    'OriginalDC', 'OriginalDC+Noise', 'LapInMitte', 'LapBeiArzt',
    'Selbstkorrekturen', 'Unterbrechungen', 'Gedankensprünge',
    'Meinungswechsel', 'Chaos', 'Anamnesegespräch', 'PWC',
]

# ── Ground Truth parsen ────────────────────────────────────────────────────

GT_NAME_MAP = {
    'original dc white noise':                          'OriginalDC+Noise',
    'original dc':                                      'OriginalDC',
    'original lap in mitte':                            'LapInMitte',
    'original lap bei arzt':                            'LapBeiArzt',
    'selbstkorrekturen':                                'Selbstkorrekturen',
    'unterbrechungen':                                  'Unterbrechungen',
    'gedankensprünge':                                  'Gedankensprünge',
    'meinungswechsel':                                  'Meinungswechsel',
    'chaos':                                            'Chaos',
    'das anamnesegespräch':                             'Anamnesegespräch',
    'anamnesegesrpäch pwc':                             'PWC',
    'anamnesegespräch pwc':                             'PWC',
}

def parse_gt(path=GT_FILE):
    text = path.read_text()
    sections = re.split(r'^#{1,4}\s+Ground Truth:\s*', text, flags=re.MULTILINE)
    gt = {}
    for sec in sections[1:]:
        lines = sec.split('\n')
        raw_name = lines[0].strip().rstrip()
        key = None
        for k, v in GT_NAME_MAP.items():
            if k in raw_name.lower():
                key = v
                break
        if key is None:
            continue
        # Sprecherpräfixe entfernen und Text zusammenführen
        body = '\n'.join(lines[1:])
        body = re.sub(r'\*\*(?:Arzt|Patient|Patientin)\*\*\s*:\s*', '', body)
        gt[key] = to_words(body)
    return gt

# ── Audio-Datei → Szenario-Name ─────────────────────────────────────────────

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
    if b in AUDIO_MAP:
        return AUDIO_MAP[b]
    if 'PWC' in b:
        return 'PWC'
    if 'Anamnesegesr' in b or 'Anamnesegespräch' in b:
        return 'Anamnesegespräch'
    return b

# ── Normalisierung ──────────────────────────────────────────────────────────

def to_words(text):
    norm = PUNCT.sub(' ', text).lower()
    return [w for w in norm.split() if w]

# ── Levenshtein-Alignment ──────────────────────────────────────────────────

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

# ── Hauptfunktion ──────────────────────────────────────────────────────────

def run(stt_filter, llm_filter, title, out_file):
    gt = parse_gt()
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
        name = scenario(e['audio_file'])
        if name not in gt:
            print(f'  WARNUNG: kein Ground Truth für {name}')
            continue

        ref = gt[name]
        hyp = to_words(e['raw'])
        ops = align(ref, hyp)

        errors, S, D, I = [], 0, 0, 0
        rp = hp = 0
        for op, r, h in ops:
            if op == 'S':
                errors.append(('Substitution', r, h, ctx(ref, rp))); S += 1; rp += 1; hp += 1
            elif op == 'D':
                errors.append(('Löschung', r, '*(fehlt)*', ctx(ref, rp))); D += 1; rp += 1
            elif op == 'I':
                errors.append(('Einfügung', '*(nicht da)*', h, '(STT) ' + ctx(hyp, hp))); I += 1; hp += 1
            else:
                rp += 1; hp += 1

        total = S + D + I
        wer = total / len(ref) * 100 if ref else 0
        results[name] = dict(ref=len(ref), hyp=len(hyp), S=S, D=D, I=I,
                             total=total, wer=wer, errors=errors)
        print(f'  {name:25s} ref={len(ref):4d} hyp={len(hyp):4d}  S={S} D={D} I={I}  WER={wer:.1f}%')

    # ── Markdown schreiben ─────────────────────────────────────────────────
    out = []
    out.append(f'# WER-Analyse: {title}')
    out.append('')
    out.append('> Ground Truth → STT RAW — Satzzeichen und Groß-/Kleinschreibung ignoriert.')
    out.append('> **S** = Substitution | **D** = Löschung (in GT, fehlt im STT) | **I** = Einfügung (im STT, nicht in GT)')
    out.append('> WER = (S + D + I) / Ref-Wörter × 100%')
    out.append('')
    out.append('---')
    out.append('')
    out.append('## Übersicht')
    out.append('')
    out.append('| Szenario | Ref-Wörter | STT-Wörter | S | D | I | Edit-Dist | WER |')
    out.append('|---|---|---|---|---|---|---|---|')
    for name in ORDER:
        if name not in results:
            continue
        r = results[name]
        out.append(f"| {name} | {r['ref']} | {r['hyp']} | {r['S']} | {r['D']} | {r['I']} | {r['total']} | **{r['wer']:.1f}%** |")

    for name in ORDER:
        if name not in results:
            continue
        r = results[name]
        out.append('')
        out.append('---')
        out.append('')
        out.append(f'## {name}')
        out.append('')
        out.append(f"**WER: {r['wer']:.1f}%** — Ref: {r['ref']} Wörter | STT: {r['hyp']} Wörter "
                   f"| S={r['S']} D={r['D']} I={r['I']} | Edit-Dist={r['total']}")
        out.append('')
        if not r['errors']:
            out.append('*Keine Fehler gefunden.*')
            continue
        out.append('| # | Typ | Ground Truth | STT-Output | Kontext (GT) |')
        out.append('|---|-----|-------------|-----------|--------------|')
        for i, (typ, gw, sw, c) in enumerate(r['errors'], 1):
            out.append(f'| {i} | {typ} | `{gw}` | `{sw}` | {c} |')

    out_path = REPO / "docs" / out_file
    out_path.write_text('\n'.join(out) + '\n')
    print(f'\nGeschrieben: {out_path}')
