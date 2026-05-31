"""
Erstellt docs/soap_eval_auswertung.md aus results/soap_eval_results.json.
Übersichtstabellen pro Modell + Detailtabellen pro Szenario.
"""
import json
from pathlib import Path

REPO     = Path(__file__).parent.parent.parent
IN_FILE  = REPO / "results" / "v1_praesentation" / "soap_eval_results.json"
OUT_FILE = REPO / "docs" / "v1_praesentation" / "soap_eval_auswertung.md"

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
URTEIL_SYMBOL = {
    'akzeptabel':           '✓',
    'ueberarbeitung_noetig':'⚠',
    'nicht_verwendbar':     '✗',
}

with open(IN_FILE) as f:
    data = json.load(f)

# Index: (model, scenario) → entry
idx = {(e['model'], e['scenario']): e for e in data}

def cell(e):
    """Kurze Zellen-Darstellung für Übersichtstabelle."""
    if not e:
        return '—'
    if e.get('auto_skip'):
        return f'✗ auto'
    if 'fehler' in e:
        return '⚡ Fehler'
    gs = e.get('gesamt_score', 0)
    u  = e.get('urteil', '')
    sym = URTEIL_SYMBOL.get(u, '?')
    return f'{sym} {gs}/8'

def score_cell(e, sec):
    if not e or e.get('auto_skip') or 'fehler' in e:
        return '—'
    return str(e.get(sec, {}).get('score', '—'))

# ── Markdown aufbauen ──────────────────────────────────────────────────────
out = []
out.append('# SOAP Evaluation: Auswertung')
out.append('')
out.append('> Bewertet wird die Informationsübertragung vom Transkript in die SOAP-Notiz.')
out.append('> Score pro Sektion: 0–2 | Gesamt: 0–8')
out.append('> ✓ = akzeptabel (7–8) | ⚠ = Überarbeitung nötig (4–6) | ✗ = nicht verwendbar (0–3)')
out.append('> `auto` = automatisch nicht_verwendbar (leeres SOAP oder >90% LLM-Fehlerrate)')
out.append('')
out.append('---')
out.append('')

# ── 1. Übersicht: alle Modelle × alle Szenarien ────────────────────────────
out.append('## Übersicht')
out.append('')
header = '| Szenario | ' + ' | '.join(MODEL_ORDER) + ' |'
sep    = '|---' * (len(MODEL_ORDER) + 1) + '|'
out.append(header)
out.append(sep)
for scen in SCENARIO_ORDER:
    cells = [cell(idx.get((m, scen))) for m in MODEL_ORDER]
    out.append(f'| {scen} | ' + ' | '.join(cells) + ' |')
out.append('')

# ── 2. Zusammenfassung pro Modell ──────────────────────────────────────────
out.append('## Zusammenfassung pro Modell')
out.append('')
out.append('| Modell | ✓ akzeptabel | ⚠ Überarb. | ✗ nicht verwendbar | Ø Score | Ø S | Ø O | Ø A | Ø P |')
out.append('|---|---|---|---|---|---|---|---|---|')

for model in MODEL_ORDER:
    entries = [idx.get((model, s)) for s in SCENARIO_ORDER if idx.get((model, s))]
    bewertete = [e for e in entries if not e.get('auto_skip') and 'fehler' not in e]

    counts = {'akzeptabel': 0, 'ueberarbeitung_noetig': 0, 'nicht_verwendbar': 0}
    for e in entries:
        u = e.get('urteil', 'nicht_verwendbar')
        counts[u] = counts.get(u, 0) + 1

    def avg(sec):
        vals = [e.get(sec, {}).get('score') for e in bewertete if isinstance(e.get(sec, {}).get('score'), int)]
        return f'{sum(vals)/len(vals):.1f}' if vals else '—'

    scores = [e.get('gesamt_score', 0) for e in bewertete]
    avg_gs = f'{sum(scores)/len(scores):.1f}' if scores else '—'

    out.append(
        f'| {model} '
        f'| {counts["akzeptabel"]} '
        f'| {counts["ueberarbeitung_noetig"]} '
        f'| {counts["nicht_verwendbar"]} '
        f'| {avg_gs} '
        f'| {avg("S")} '
        f'| {avg("O")} '
        f'| {avg("A")} '
        f'| {avg("P")} |'
    )
out.append('')

# ── 3. Details pro Modell ──────────────────────────────────────────────────
out.append('---')
out.append('')
out.append('## Details pro Modell')
out.append('')

for model in MODEL_ORDER:
    out.append(f'### {model}')
    out.append('')
    out.append('| Szenario | S | O | A | P | Gesamt | Urteil |')
    out.append('|---|---|---|---|---|---|---|')

    for scen in SCENARIO_ORDER:
        e = idx.get((model, scen))
        if not e:
            continue
        if e.get('auto_skip'):
            reason = e.get('auto_skip_grund', '')
            out.append(f'| {scen} | — | — | — | — | **auto** | ✗ nicht_verwendbar (`{reason}`) |')
            continue
        if 'fehler' in e:
            out.append(f'| {scen} | — | — | — | — | **Fehler** | ⚡ |')
            continue
        gs  = e.get('gesamt_score', 0)
        u   = e.get('urteil', '')
        sym = URTEIL_SYMBOL.get(u, '?')
        out.append(
            f'| {scen} '
            f'| {score_cell(e,"S")} '
            f'| {score_cell(e,"O")} '
            f'| {score_cell(e,"A")} '
            f'| {score_cell(e,"P")} '
            f'| **{gs}/8** '
            f'| {sym} {u} |'
        )
    out.append('')

    # Halluzinationen & Auslassungen zusammenfassen
    all_hall = []
    all_aus  = []
    for scen in SCENARIO_ORDER:
        e = idx.get((model, scen))
        if not e or e.get('auto_skip') or 'fehler' in e:
            continue
        for sec in 'SOAP':
            for h in e.get(sec, {}).get('halluzinationen', []):
                all_hall.append((scen, sec, h))
            for a in e.get(sec, {}).get('auslassungen', []):
                all_aus.append((scen, sec, a))

    if all_hall:
        out.append('**Halluzinationen:**')
        out.append('')
        out.append('| Szenario | Sektion | Beschreibung |')
        out.append('|---|---|---|')
        for scen, sec, h in all_hall:
            out.append(f'| {scen} | {sec} | {h} |')
        out.append('')

    if all_aus:
        out.append('**Auslassungen:**')
        out.append('')
        out.append('| Szenario | Sektion | Beschreibung |')
        out.append('|---|---|---|')
        for scen, sec, a in all_aus:
            out.append(f'| {scen} | {sec} | {a} |')
        out.append('')

    out.append('---')
    out.append('')

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text('\n'.join(out) + '\n')
print(f'Geschrieben: {OUT_FILE}')

# Kurze Konsolzusammenfassung
print()
print('Urteil-Übersicht:')
for model in MODEL_ORDER:
    entries = [idx.get((model, s)) for s in SCENARIO_ORDER if idx.get((model, s))]
    counts = {}
    for e in entries:
        u = e.get('urteil', '?')
        counts[u] = counts.get(u, 0) + 1
    bewertete = [e for e in entries if not e.get('auto_skip') and 'fehler' not in e]
    scores = [e.get('gesamt_score', 0) for e in bewertete]
    avg = sum(scores)/len(scores) if scores else 0
    print(f'  {model:22s}  ✓={counts.get("akzeptabel",0)}  ⚠={counts.get("ueberarbeitung_noetig",0)}  ✗={counts.get("nicht_verwendbar",0)}  Ø={avg:.1f}')
