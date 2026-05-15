"""
Testlauf mit einem einzelnen Eintrag — zeigt Prompt und rohe Claude-Antwort.
"""
import json, subprocess, re
from pathlib import Path

REPO      = Path(__file__).parent.parent.parent
DATA_FILE = REPO / "results" / "history_no_speaker.json"
PROMPT_TPL= (REPO / "skript" / "soap" / "soap_eval_prompt.md").read_text()

with open(DATA_FILE) as f:
    data = json.load(f)

# Ersten Eintrag mit nicht-leerem SOAP nehmen
e = next(x for x in data if x.get('soap','').strip())

print(f'Szenario : {e["audio_file"].split("/")[-1]}')
print(f'Modell   : {e["stt_model"]} + {e["llm_model"]}')
print(f'SOAP len : {len(e["soap"])} Zeichen')
print()

prompt = PROMPT_TPL.replace('{formatted}', e['formatted']).replace('{soap}', e['soap'])

print('=== PROMPT (gekürzt) ===')
print(prompt[:400], '...')
print()

print('=== CLAUDE ANTWORT ===')
result = subprocess.run(
    ['claude', '-p', prompt],
    capture_output=True,
    text=True,
    timeout=120
)
output = result.stdout.strip()
print(output)

print()
print('=== JSON PARSE ===')
text = re.sub(r'```(?:json)?\s*', '', output).strip('`').strip()
match = re.search(r'\{.*\}', text, re.DOTALL)
if match:
    try:
        parsed = json.loads(match.group())
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except json.JSONDecodeError as ex:
        print(f'Parse-Fehler: {ex}')
else:
    print('Kein JSON gefunden in der Antwort.')
