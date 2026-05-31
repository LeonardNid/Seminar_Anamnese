"""
SOAP Evaluator — läuft sequenziell durch alle 55 Einträge.
Ruft für jeden Eintrag 'claude -p' mit dem Prompt auf und speichert
alle Ergebnisse in results/soap_eval_results.json.

Einträge mit >90% LLM-Fehlerrate oder leerem SOAP werden automatisch
als "nicht_verwendbar" markiert ohne Claude-Aufruf.
"""
import json, re, subprocess, sys, time
from pathlib import Path

REPO      = Path(__file__).parent.parent.parent
DATA_FILE = REPO / "results" / "v1_praesentation" / "history_no_speaker.json"
OUT_FILE  = REPO / "results" / "v1_praesentation" / "soap_eval_results.json"
PROMPT_TPL= (REPO / "skript" / "soap" / "soap_eval_prompt.md").read_text()

PUNCT = re.compile(r'[.,!?;:\-–—()\[\]"\'„"«»/]')

# ── Urteil deterministisch berechnen ──────────────────────────────────────
def urteil(score):
    if score >= 7: return "akzeptabel"
    if score >= 4: return "ueberarbeitung_noetig"
    return "nicht_verwendbar"

# ── Hilfsfunktionen ────────────────────────────────────────────────────────
def to_words(text):
    norm = PUNCT.sub(' ', text).lower()
    return [w for w in norm.split() if w]

def model_label(e):
    stt = e.get('stt_model', '')
    llm = e.get('llm_model', '')
    if 'Speechmatics' in stt: return 'Speechmatics+GPT4o'
    if 'AssemblyAI'   in stt: return 'AssemblyAI+GPT4o'
    if 'gemma4'       in llm: return 'Whisper+gemma4'
    if 'Sauerkraut'   in llm: return 'Whisper+Sauerkraut'
    if 'llama3.2'     in llm: return 'Whisper+llama3.2'
    return f'{stt}+{llm}'

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

# ── LLM-Fehlerrate schätzen (wortbasiert, kein volles Alignment) ───────────
def llm_fehlerrate(raw_text, fmt_text):
    raw_w = to_words(raw_text)
    fmt_w = to_words(fmt_text)
    if not raw_w:
        return 100.0
    # Heuristik: wenn FMT viel kürzer als RAW → viele Löschungen → hohe Fehlerrate
    # Exakte Untergrenze: min. (len(raw_w) - len(fmt_w)) / len(raw_w) deletions
    min_deletions = max(0, len(raw_w) - len(fmt_w))
    return min_deletions / len(raw_w) * 100

# ── Claude aufrufen ────────────────────────────────────────────────────────
def call_claude(prompt_text, retries=2):
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ['claude', '-p', prompt_text],
                capture_output=True,
                text=True,
                timeout=120
            )
            output = result.stdout.strip()
            if not output and result.stderr:
                print(f'    STDERR: {result.stderr[:200]}')
            return output
        except subprocess.TimeoutExpired:
            print(f'    Timeout (Versuch {attempt+1}/{retries})')
            time.sleep(5)
        except Exception as ex:
            print(f'    Fehler: {ex}')
            time.sleep(5)
    return None

# ── JSON aus Claude-Antwort extrahieren ───────────────────────────────────
def extract_json(text):
    if not text:
        return None
    # Codeblöcke entfernen falls Claude sie trotzdem hinzufügt
    text = re.sub(r'```(?:json)?\s*', '', text).strip('`').strip()
    # Erstes vollständiges JSON-Objekt extrahieren
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None

# ── Nicht-verwendbar-Eintrag erstellen ────────────────────────────────────
def auto_nicht_verwendbar(reason):
    return {
        'S': {'score': 0, 'halluzinationen': [], 'auslassungen': []},
        'O': {'score': 0, 'halluzinationen': [], 'auslassungen': []},
        'A': {'score': 0, 'halluzinationen': [], 'auslassungen': []},
        'P': {'score': 0, 'halluzinationen': [], 'auslassungen': []},
        'gesamt_score': 0,
        'urteil': 'nicht_verwendbar',
        'auto_skip': True,
        'auto_skip_grund': reason,
    }

# ── Hauptprogramm ──────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    data = json.load(f)

# Bereits vorhandene Ergebnisse laden (für Fortsetzung nach Abbruch)
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        all_results = json.load(f)
    # Fehler-Einträge (kein gültiges JSON, Usage-Limit etc.) herausfiltern und wiederholen
    results  = [r for r in all_results if 'fehler' not in r]
    failed   = [r for r in all_results if 'fehler' in r]
    done_keys = {(r['model'], r['scenario']) for r in results}
    print(f'Vorhandene Ergebnisse: {len(all_results)} gesamt, '
          f'{len(results)} OK, {len(failed)} Fehler werden wiederholt.')
else:
    results = []
    done_keys = set()
    print('Keine vorhandenen Ergebnisse — starte neu.')

total = len(data)
for idx, e in enumerate(data, 1):
    scen  = scenario(e['audio_file'])
    model = model_label(e)
    key   = (model, scen)

    if key in done_keys:
        print(f'[{idx:2d}/{total}] {model:22s} | {scen:20s} — übersprungen (bereits vorhanden)')
        continue

    print(f'[{idx:2d}/{total}] {model:22s} | {scen:20s}', end=' ', flush=True)

    raw_text = e.get('raw', '')       or ''
    fmt_text = e.get('formatted', '') or ''
    soap     = e.get('soap', '')      or ''

    base = dict(
        model=model,
        scenario=scen,
        stt_model=e.get('stt_model', ''),
        llm_model=e.get('llm_model', ''),
        audio_file=e.get('audio_file', ''),
    )

    # Auto-skip: leeres SOAP
    if not soap.strip():
        print('→ AUTO-SKIP (leeres SOAP)')
        results.append({**base, **auto_nicht_verwendbar('SOAP ist leer')})
        with open(OUT_FILE, 'w') as f: json.dump(results, f, ensure_ascii=False, indent=2)
        continue

    # Auto-skip: LLM-Fehlerrate >90%
    rate = llm_fehlerrate(raw_text, fmt_text)
    if rate > 90:
        print(f'→ AUTO-SKIP (LLM-Fehlerrate {rate:.0f}%)')
        results.append({**base, **auto_nicht_verwendbar(f'LLM-Fehlerrate {rate:.0f}% > 90%'),
                        'llm_fehlerrate': round(rate, 1)})
        with open(OUT_FILE, 'w') as f: json.dump(results, f, ensure_ascii=False, indent=2)
        continue

    # Claude aufrufen
    prompt = PROMPT_TPL.replace('{formatted}', fmt_text).replace('{soap}', soap)
    raw_response = call_claude(prompt)
    parsed = extract_json(raw_response)

    if parsed is None:
        print(f'→ FEHLER (kein gültiges JSON)')
        preview = (raw_response or '')[:200]
        print(f'   Antwort: {preview}')
        results.append({**base,
                        'llm_fehlerrate': round(rate, 1),
                        'fehler': 'kein_gueltiges_json',
                        'raw_response_preview': preview})
        with open(OUT_FILE, 'w') as f: json.dump(results, f, ensure_ascii=False, indent=2)
        continue

    # Urteil deterministisch hinzufügen
    gs = parsed.get('gesamt_score', 0)
    entry = {
        **base,
        'llm_fehlerrate': round(rate, 1),
        'auto_skip': False,
        'S': parsed.get('S', {}),
        'O': parsed.get('O', {}),
        'A': parsed.get('A', {}),
        'P': parsed.get('P', {}),
        'gesamt_score': gs,
        'urteil': urteil(gs),
    }
    results.append(entry)
    with open(OUT_FILE, 'w') as f: json.dump(results, f, ensure_ascii=False, indent=2)

    scores = f"S={parsed.get('S',{}).get('score','?')} O={parsed.get('O',{}).get('score','?')} A={parsed.get('A',{}).get('score','?')} P={parsed.get('P',{}).get('score','?')}"
    print(f'→ {scores} gesamt={gs} ({urteil(gs)})')

print(f'\nFertig. {len(results)} Ergebnisse in {OUT_FILE}')
