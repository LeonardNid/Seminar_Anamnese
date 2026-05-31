"""
Deterministische Sprecher-Formatierung für alle drei STT-Label-Formate:
  - Whisper+pyannote (inline):  SPEAKER_00: text, SPEAKER_??: text
  - AssemblyAI (inline):        Speaker A: text, Speaker B: text
  - Speechmatics (eigene Zeile): SPEAKER: S1 \n text-zeilen \n SPEAKER: S2 \n ...

Ablauf:
  1. parse_turns(raw)          → geordnete [(label, text), ...]
  2. distinct_labels(turns)    → eindeutige Labels in Auftrittsreihenfolge
  3. identify_roles(raw, labels, client, model)  → {label: "Arzt"/"Name"/"Patient(in)"}
  4. apply_mapping(turns, mapping) → formatiertes Transkript (deterministisch)
"""

import re
import json

# ── Label-Muster ─────────────────────────────────────────────────────────────

# Speechmatics: Label steht allein auf einer Zeile, z.B. "SPEAKER: S1"
_RE_SPEECHMATICS_SOLO = re.compile(r'^\s*SPEAKER:\s*(S\w+)\s*$')

# Inline: SPEAKER_00: / SPEAKER_??: / Speaker A: / Speaker 0:
_RE_INLINE = re.compile(r'^\s*(SPEAKER_[\w?]+|Speaker\s+\w+):\s*(.*)')


def parse_turns(raw: str) -> list[tuple[str, str]]:
    """Zerlegt raw STT-Text in geordnete (label, text)-Turns.

    Gibt [] zurück wenn keine Labels erkannt werden (Rohtext bleibt unverändert).
    """
    if not raw or not raw.strip():
        return []

    lines = raw.splitlines()

    # Erkennt das Format anhand der ersten gefundenen Label-Zeile
    is_speechmatics = any(_RE_SPEECHMATICS_SOLO.match(l) for l in lines)

    turns: list[tuple[str, str]] = []

    if is_speechmatics:
        # Speechmatics: Label auf eigener Zeile, Text in nachfolgenden Zeilen
        current_label = None
        current_parts: list[str] = []
        for line in lines:
            m = _RE_SPEECHMATICS_SOLO.match(line)
            if m:
                if current_label is not None and current_parts:
                    turns.append((current_label, ' '.join(current_parts).strip()))
                current_label = f"SPEAKER: {m.group(1)}"
                current_parts = []
            else:
                stripped = line.strip()
                if stripped:
                    current_parts.append(stripped)
        if current_label is not None and current_parts:
            turns.append((current_label, ' '.join(current_parts).strip()))
    else:
        # Inline-Format (Whisper/AssemblyAI)
        current_label = None
        current_parts: list[str] = []
        for line in lines:
            m = _RE_INLINE.match(line)
            if m:
                if current_label is not None and current_parts:
                    turns.append((current_label, ' '.join(current_parts).strip()))
                current_label = m.group(1)
                text = m.group(2).strip()
                current_parts = [text] if text else []
            else:
                stripped = line.strip()
                if stripped:
                    if current_label is None:
                        # Text vor dem ersten Label — als anonymen Turn behandeln
                        current_label = "__UNKNOWN__"
                        current_parts = [stripped]
                    else:
                        current_parts.append(stripped)
        if current_label is not None and current_parts:
            turns.append((current_label, ' '.join(current_parts).strip()))

    # Leerzeilen-Turns verwerfen
    turns = [(l, t) for l, t in turns if t.strip()]
    return turns


def distinct_labels(turns: list[tuple[str, str]]) -> list[str]:
    """Eindeutige Labels in Reihenfolge ihres ersten Auftretens."""
    seen: list[str] = []
    for label, _ in turns:
        if label not in seen:
            seen.append(label)
    return seen


def _extract_json(text: str) -> dict | None:
    """Extrahiert erstes JSON-Objekt aus LLM-Antwort (robust gegen Codeblöcke)."""
    if not text:
        return None
    text = re.sub(r'```(?:json)?\s*', '', text).strip('`').strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def identify_roles(raw: str, labels: list[str], client, model: str) -> dict[str, str]:
    """Fragt die LLM, welches Label Arzt / Patient ist.

    Gibt ein Mapping {label → "Arzt"/"Name"/"Patient(in)"} zurück.
    Fehlende Labels werden im apply_mapping-Fallback abgefangen.
    """
    if not labels:
        return {}

    labels_str = '\n'.join(f'- {l}' for l in labels)
    system = (
        "Du erhältst ein Roh-Transkript eines Arzt-Patienten-Gesprächs und eine Liste von "
        "Sprecher-Labels. Bestimme NUR, welches Label der Arzt und welches der Patient ist, "
        "und finde den Patientennamen, falls er im Gespräch genannt wird.\n"
        "Antworte AUSSCHLIESSLICH als JSON-Objekt, das jedes Label auf genau einen der "
        "folgenden Werte abbildet:\n"
        '  - "Arzt" (für den Arzt)\n'
        '  - den tatsächlichen Patientennamen z.B. "Frau Weber" (wenn genannt)\n'
        '  - "Patient(in)" (wenn kein Name genannt wird)\n'
        "Kein weiterer Text, kein Markdown, kein Kommentar."
    )
    user = f"Sprecher-Labels:\n{labels_str}\n\nRoh-Transkript:\n{raw}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json(content)
        if parsed and isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass

    return {}


def apply_mapping(turns: list[tuple[str, str]], mapping: dict[str, str]) -> str:
    """Ersetzt Labels deterministisch durch die gemappten Rollen.

    - Ein Turn = eine Zeile: "Rolle: Text"
    - Zeilenumbruch nach jedem Sprecherwechsel
    - Text wörtlich unverändert
    - Unbekannte Labels: Original-Label beibehalten
    """
    lines = []
    for label, text in turns:
        rolle = mapping.get(label, label)
        lines.append(f"{rolle}: {text}")
    return '\n'.join(lines)


def format_transcript_deterministic(raw: str, client, model: str) -> str:
    """Kompletter Formatierungs-Ablauf in einem Aufruf (kein Streaming).

    Gibt den Rohtext zurück wenn keine Labels erkannt werden.
    """
    turns = parse_turns(raw)
    if not turns:
        return raw
    labels = distinct_labels(turns)
    mapping = identify_roles(raw, labels, client, model)
    return apply_mapping(turns, mapping)
