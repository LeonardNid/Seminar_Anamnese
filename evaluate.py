import json
import re
import difflib
from pathlib import Path
from collections import defaultdict

HISTORY_FILE = Path("history.json")
GT_FILE = Path("docs/Seminar Texte.md")
OUT_FILE = Path("evaluation_report.md")

MODEL_LABELS = {
    ("Whisper Large-v3-turbo (Lokal)", "Llama-3.1-SauerkrautLM-8b-Instruct"): "Whisper+Sauerkraut",
    ("Whisper Large-v3-turbo (Lokal)", "llama3.2"): "Whisper+llama3.2",
    ("Speechmatics (Cloud)", "OpenAI GPT-4o"): "Speechmatics+GPT-4o",
}
MODEL_ORDER = ["Whisper+Sauerkraut", "Whisper+llama3.2", "Speechmatics+GPT-4o"]

AUDIO_TO_SCENARIO = {
    "OriginalLapInMitte.wav": "Original",
    "OriginalLapBeiArzt.wav": "Original",
    "OriginalDC.m4a": "Original",
    "OriginalDCWhiteNoise.m4a": "Original",
    "SelbstkorrekturLapInMitte.wav": "Selbstkorrekturen",
    "UnterbrechungLapInMitte.wav": "Unterbrechungen",
    "GedankenprüngeLapInMitte.wav": "Gedankensprünge",
    "MeinungswechselLapinMitte.wav": "Meinungswechsel",
    "ChaosLapInMitte.wav": "Chaos",
}

AUDIO_DISPLAY = {
    "OriginalLapInMitte.wav": "Original (Laptop Mitte)",
    "OriginalLapBeiArzt.wav": "Original (Laptop Arzt)",
    "OriginalDC.m4a": "Original (DC-Mikro)",
    "OriginalDCWhiteNoise.m4a": "Original (DC + Rauschen)",
    "SelbstkorrekturLapInMitte.wav": "Selbstkorrekturen",
    "UnterbrechungLapInMitte.wav": "Unterbrechungen",
    "GedankenprüngeLapInMitte.wav": "Gedankensprünge",
    "MeinungswechselLapinMitte.wav": "Meinungswechsel",
    "ChaosLapInMitte.wav": "Chaos",
    "Das Anamnesegespräch Teil 1 medizinische Fachsprachprüfung Fall Unfall - ärztesprech (128k).wav": "Anamnesegespräch (ext.)",
}

# ── Ground-truth parsing ─────────────────────────────────────────────────────

def parse_ground_truth(md: str) -> dict:
    sections = {}
    # Split on top-level H1 headers
    parts = re.split(r"^# (.+)$", md, flags=re.MULTILINE)
    # parts = ["", title1, body1, title2, body2, ...]
    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        body = parts[i + 1]
        # Skip "Original Englisch" (no matching audio in history)
        if "Englisch" not in title:
            sections[title] = clean_ground_truth(body)
        i += 2
    return sections

def clean_ground_truth(text: str) -> str:
    # Remove sub-headers (####)
    text = re.sub(r"^#{1,6} .+$", "", text, flags=re.MULTILINE)
    # Remove meta lines (Testziel, Regieanweisung, Szenario)
    text = re.sub(r"^\*\*(?:Testziel|Regieanweisung|Szenario):\*\*.+$", "", text, flags=re.MULTILINE)
    # Remove stage directions in square brackets and their wrapper (_[..._], [...]_)
    text = re.sub(r"_?\[.*?\]_?", "", text)
    # Remove markdown bold/italic markers
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"(?<!\w)[*_](?!\w)|(?<!\w)[*_]|[*_](?!\w)", "", text)
    # Remove speaker prefixes like **Arzt:**, **Patientin:**
    text = re.sub(r"^(?:Arzt|Patientin|Patient)\s*:", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    # Remove emoji lines
    text = re.sub(r"^.*🇩🇪.*$|^.*🇬🇧.*$", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# ── Text normalization ────────────────────────────────────────────────────────

def strip_speaker_labels(text: str) -> str:
    # Whisper diarization format: SPEAKER_00:, SPEAKER_01:, SPEAKER_??:
    text = re.sub(r"SPEAKER_\w+\s*:", "", text)
    # Speechmatics format: SPEAKER: S1\n
    text = re.sub(r"SPEAKER\s*:\s*S\d+\s*", "", text)
    # Formatted labels: Arzt:, Patientin:, Patient:, name:
    text = re.sub(r"^(?:Arzt|Patientin|Patient|Frau \w+|Herr \w+|Transkript)\s*:", "", text, flags=re.MULTILINE)
    return text.strip()

def normalize_words(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()

# ── WER computation ───────────────────────────────────────────────────────────

def edit_distance_ops(ref: list, hyp: list) -> tuple:
    sm = difflib.SequenceMatcher(None, ref, hyp)
    insertions = deletions = substitutions = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "insert":
            insertions += j2 - j1
    return insertions, deletions, substitutions

def word_error_rate(reference: str, hypothesis: str) -> tuple:
    ref_words = normalize_words(strip_speaker_labels(reference))
    hyp_words = normalize_words(strip_speaker_labels(hypothesis))
    if not ref_words:
        return 0.0, {}
    ins, dels, subs = edit_distance_ops(ref_words, hyp_words)
    total_errors = ins + dels + subs
    wer = total_errors / len(ref_words)
    return round(wer, 3), {"insertions": ins, "deletions": dels, "substitutions": subs}

def char_similarity(a: str, b: str) -> float:
    a = strip_speaker_labels(a)
    b = strip_speaker_labels(b)
    sm = difflib.SequenceMatcher(None, a, b)
    return round(sm.ratio(), 3)

# ── Error examples ────────────────────────────────────────────────────────────

def get_error_examples(reference: str, hypothesis: str, n: int = 8) -> list:
    ref_words = normalize_words(strip_speaker_labels(reference))
    hyp_words = normalize_words(strip_speaker_labels(hypothesis))
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
    errors = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            errors.append(f"'{' '.join(ref_words[i1:i2])}' → '{' '.join(hyp_words[j1:j2])}'")
        elif tag == "delete":
            errors.append(f"[fehlt] '{' '.join(ref_words[i1:i2])}'")
        elif tag == "insert":
            errors.append(f"[extra] '{' '.join(hyp_words[j1:j2])}'")
        if len(errors) >= n:
            break
    return errors

# ── Formatting evaluation ─────────────────────────────────────────────────────

def eval_formatting(entry: dict) -> dict:
    raw = entry["raw"]
    fmt = entry["formatted"]
    speaker_still_present = bool(re.search(r"SPEAKER_\w+\s*:|SPEAKER\s*:\s*S\d+", fmt))
    arzt_label_present = bool(re.search(r"\bArzt\s*:", fmt))
    raw_stripped = strip_speaker_labels(raw)
    fmt_stripped = strip_speaker_labels(fmt)
    preservation = char_similarity(raw_stripped, fmt_stripped)
    # Check if LLM added substantial content beyond labeling
    raw_len = len(raw_stripped.split())
    fmt_len = len(fmt_stripped.split())
    word_diff = fmt_len - raw_len
    return {
        "speaker_replaced": not speaker_still_present,
        "arzt_label": arzt_label_present,
        "text_preserved": preservation,
        "word_diff": word_diff,
    }

# ── SOAP evaluation ───────────────────────────────────────────────────────────

SOAP_PATTERNS = {
    "S": re.compile(r"\bS[\s\(:)]|Subjektiv", re.IGNORECASE),
    "O": re.compile(r"\bO[\s\(:)]|Objektiv", re.IGNORECASE),
    "A": re.compile(r"\bA[\s\(:)]|Assessment|Beurteilung", re.IGNORECASE),
    "P": re.compile(r"\bP[\s\(:)]|Plan\b", re.IGNORECASE),
}

# Medical key terms to check per scenario
SCENARIO_KEY_TERMS = {
    "Original": ["38,8", "husten", "fieber", "sputum", "dyspnoe", "kamillentee", "drogerie"],
    "Selbstkorrekturen": ["mittwoch", "kopfschmerz", "rechte", "paracetamol", "ibuprofen", "600", "lichtempfindlich"],
    "Unterbrechungen": ["bauchschmerz", "erbrochen", "38,3", "hähnchen", "bauchnabel"],
    "Gedankensprünge": ["schulter", "knackt", "ellenbogen", "rotator", "tennis"],
    "Meinungswechsel": ["reflux", "sodbrennen", "angina", "kein", "brustbein"],
    "Chaos": ["ohrenschmalz", "schwindel", "tinnitus", "ibuprofen", "800", "rechts"],
}

def eval_soap(entry: dict, scenario: str) -> dict:
    soap = entry["soap"]
    formatted = entry["formatted"]
    sections_found = {k: bool(p.search(soap)) for k, p in SOAP_PATTERNS.items()}
    structure_complete = all(sections_found.values())
    key_terms = SCENARIO_KEY_TERMS.get(scenario, [])
    soap_lower = soap.lower()
    fmt_lower = formatted.lower()
    terms_in_soap = [t for t in key_terms if t in soap_lower]
    term_coverage = len(terms_in_soap) / len(key_terms) if key_terms else 1.0
    # Simple hallucination check: numbers in SOAP not in formatted
    soap_numbers = set(re.findall(r"\b\d+[,.]?\d*\b", soap))
    fmt_numbers = set(re.findall(r"\b\d+[,.]?\d*\b", formatted))
    extra_numbers = soap_numbers - fmt_numbers
    # filter common noise (1, 2, 3, 4 etc.)
    hallucinated_numbers = [n for n in extra_numbers if float(n.replace(",", ".")) > 4]
    return {
        "sections": sections_found,
        "structure_complete": structure_complete,
        "term_coverage": round(term_coverage, 2),
        "terms_found": terms_in_soap,
        "terms_total": key_terms,
        "hallucinated_numbers": sorted(hallucinated_numbers),
    }

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_pct(v: float) -> str:
    return f"{v*100:.0f}%"

def fmt_wer(v: float) -> str:
    return f"{v*100:.1f}%"

def tick(b: bool) -> str:
    return "✓" if b else "✗"

# ── Main report generation ────────────────────────────────────────────────────

def main():
    history = json.loads(HISTORY_FILE.read_text())
    gt_md = GT_FILE.read_text()
    ground_truth = parse_ground_truth(gt_md)

    # Group entries: audio_file → model_label → entry
    grouped = defaultdict(dict)
    for entry in history:
        label = MODEL_LABELS.get((entry["stt_model"], entry["llm_model"]), "Unknown")
        grouped[entry["audio_file"]][label] = entry

    audio_files = [
        "OriginalLapInMitte.wav",
        "OriginalLapBeiArzt.wav",
        "OriginalDC.m4a",
        "OriginalDCWhiteNoise.m4a",
        "SelbstkorrekturLapInMitte.wav",
        "UnterbrechungLapInMitte.wav",
        "GedankenprüngeLapInMitte.wav",
        "MeinungswechselLapinMitte.wav",
        "ChaosLapInMitte.wav",
        "Das Anamnesegespräch Teil 1 medizinische Fachsprachprüfung Fall Unfall - ärztesprech (128k).wav",
    ]

    lines = []
    W = lines.append

    W("# Evaluation Report: AI Medical Documentation Pipeline\n")
    W(f"**Dateien:** `history.json` · `docs/Seminar Texte.md`  ")
    W(f"**Einträge:** {len(history)} (10 Audiodateien × 3 Modellkombinationen)\n")
    W("**Modelle:**\n")
    W("| Kürzel | STT | LLM |")
    W("|--------|-----|-----|")
    W("| Whisper+Sauerkraut | Whisper Large-v3-turbo (Lokal) | Llama-3.1-SauerkrautLM-8b-Instruct |")
    W("| Whisper+llama3.2 | Whisper Large-v3-turbo (Lokal) | llama3.2 |")
    W("| Speechmatics+GPT-4o | Speechmatics (Cloud) | OpenAI GPT-4o |")
    W("")

    # ── Übersichtstabelle WER ────────────────────────────────────────────────
    W("---\n")
    W("## 1. Übersicht: WER-Scores (STT vs. Original)\n")
    W("> WER = Word Error Rate. Niedriger ist besser. "
      "Nur für Dateien mit Ground-Truth (9 von 10). "
      "\"Das Anamnesegespräch\" hat keinen Referenztext.\n")
    W("| Audiodatei | Szenario | Whisper+Sauerkraut | Whisper+llama3.2 | Speechmatics+GPT-4o |")
    W("|-----------|----------|-------------------|-----------------|---------------------|")

    wer_table = {}
    for af in audio_files:
        scenario = AUDIO_TO_SCENARIO.get(af)
        gt_text = ground_truth.get(scenario, "") if scenario else ""
        display = AUDIO_DISPLAY.get(af, af)
        row_wers = {}
        for ml in MODEL_ORDER:
            entry = grouped[af].get(ml)
            if entry and gt_text:
                wer_val, _ = word_error_rate(gt_text, entry["raw"])
                row_wers[ml] = wer_val
            else:
                row_wers[ml] = None
        wer_table[af] = row_wers
        cells = []
        for ml in MODEL_ORDER:
            v = row_wers[ml]
            cells.append(fmt_wer(v) if v is not None else "—")
        W(f"| {display} | {scenario or '—'} | {cells[0]} | {cells[1]} | {cells[2]} |")
    W("")

    # ── Abschnitt 2: STT-Qualität ────────────────────────────────────────────
    W("---\n")
    W("## 2. STT-Qualität: Original vs. Raw\n")

    for af in audio_files:
        scenario = AUDIO_TO_SCENARIO.get(af)
        gt_text = ground_truth.get(scenario, "") if scenario else ""
        display = AUDIO_DISPLAY.get(af, af)
        W(f"### {display}\n")

        if not gt_text:
            W("*Kein Ground-Truth-Text vorhanden — kein WER-Vergleich möglich.*\n")
            for ml in MODEL_ORDER:
                entry = grouped[af].get(ml)
                if entry:
                    rc = entry["stats"]["raw_char_count"]
                    W(f"**{ml}:** {rc} Zeichen im Raw-Output  ")
            W("")
            continue

        W("**Metriken:**\n")
        W("| Modell | WER | Ins | Del | Sub | Char-Sim. |")
        W("|--------|-----|-----|-----|-----|-----------|")
        error_examples = {}
        for ml in MODEL_ORDER:
            entry = grouped[af].get(ml)
            if not entry:
                W(f"| {ml} | — | — | — | — | — |")
                continue
            wer_val, ops = word_error_rate(gt_text, entry["raw"])
            csim = char_similarity(gt_text, entry["raw"])
            W(f"| {ml} | {fmt_wer(wer_val)} | {ops['insertions']} | {ops['deletions']} | {ops['substitutions']} | {fmt_pct(csim)} |")
            error_examples[ml] = get_error_examples(gt_text, entry["raw"])
        W("")

        W("**Beispielfehler (STT vs. Ground Truth):**\n")
        for ml in MODEL_ORDER:
            examples = error_examples.get(ml, [])
            if examples:
                W(f"*{ml}:*")
                for ex in examples:
                    W(f"- {ex}")
                W("")
            else:
                W(f"*{ml}: keine Fehler gefunden*\n")

    # ── Abschnitt 3: Formatierung ────────────────────────────────────────────
    W("---\n")
    W("## 3. Formatierung: Raw vs. Formatted\n")
    W("> Prüft ob Speaker-Labels korrekt ersetzt wurden und wie viel Text verändert wurde.\n")
    W("| Audiodatei | Modell | Speaker ersetzt | Arzt-Label | Text-Sim. | Wortdiff |")
    W("|-----------|--------|----------------|-----------|-----------|----------|")

    for af in audio_files:
        display = AUDIO_DISPLAY.get(af, af)
        for ml in MODEL_ORDER:
            entry = grouped[af].get(ml)
            if not entry:
                continue
            r = eval_formatting(entry)
            W(f"| {display} | {ml} | {tick(r['speaker_replaced'])} | {tick(r['arzt_label'])} | {fmt_pct(r['text_preserved'])} | {r['word_diff']:+d} |")
    W("")

    W("**Auffälligkeiten:**\n")
    for af in audio_files:
        display = AUDIO_DISPLAY.get(af, af)
        for ml in MODEL_ORDER:
            entry = grouped[af].get(ml)
            if not entry:
                continue
            r = eval_formatting(entry)
            notes = []
            if not r["speaker_replaced"]:
                notes.append("Speaker-Labels nicht vollständig ersetzt")
            if not r["arzt_label"]:
                notes.append("Kein 'Arzt:'-Label im Formatted")
            if r["text_preserved"] < 0.85:
                notes.append(f"Geringe Text-Ähnlichkeit ({fmt_pct(r['text_preserved'])})")
            if abs(r["word_diff"]) > 20:
                notes.append(f"Großer Wortunterschied ({r['word_diff']:+d} Wörter)")
            if notes:
                W(f"- **{display} / {ml}:** {'; '.join(notes)}")
    W("")

    # ── Abschnitt 4: SOAP ────────────────────────────────────────────────────
    W("---\n")
    W("## 4. SOAP-Überprüfung\n")
    W("| Audiodatei | Modell | S | O | A | P | Struktur | Schlüsselbegriffe | Halluz.? |")
    W("|-----------|--------|---|---|---|---|----------|------------------|---------|")

    soap_results = {}
    for af in audio_files:
        scenario = AUDIO_TO_SCENARIO.get(af)
        display = AUDIO_DISPLAY.get(af, af)
        soap_results[af] = {}
        for ml in MODEL_ORDER:
            entry = grouped[af].get(ml)
            if not entry:
                continue
            r = eval_soap(entry, scenario or "")
            soap_results[af][ml] = r
            secs = r["sections"]
            halluz = "Ja" if r["hallucinated_numbers"] else "Nein"
            coverage = f"{len(r['terms_found'])}/{len(r['terms_total'])}" if r["terms_total"] else "—"
            W(f"| {display} | {ml} | {tick(secs['S'])} | {tick(secs['O'])} | {tick(secs['A'])} | {tick(secs['P'])} | {tick(r['structure_complete'])} | {coverage} | {halluz} |")
    W("")

    W("**SOAP-Details:**\n")
    for af in audio_files:
        scenario = AUDIO_TO_SCENARIO.get(af)
        display = AUDIO_DISPLAY.get(af, af)
        for ml in MODEL_ORDER:
            r = soap_results[af].get(ml)
            if not r:
                continue
            issues = []
            missing = [k for k, v in r["sections"].items() if not v]
            if missing:
                issues.append(f"Fehlende Sektionen: {', '.join(missing)}")
            missed_terms = [t for t in r["terms_total"] if t not in r["terms_found"]]
            if missed_terms:
                issues.append(f"Schlüsselbegriffe nicht erwähnt: {', '.join(missed_terms)}")
            if r["hallucinated_numbers"]:
                issues.append(f"Zahlen im SOAP nicht im Transkript: {', '.join(r['hallucinated_numbers'])}")
            if issues:
                W(f"- **{display} / {ml}:** {'; '.join(issues)}")
    W("")

    # ── Abschnitt 5: Modellvergleich ─────────────────────────────────────────
    W("---\n")
    W("## 5. Modellvergleich\n")

    W("### 5.1 STT-Qualität: Whisper vs. Speechmatics\n")
    W("> Ø WER über alle Dateien mit Ground-Truth\n")
    avg_wer = {ml: [] for ml in MODEL_ORDER}
    for af, row in wer_table.items():
        for ml in MODEL_ORDER:
            if row[ml] is not None:
                avg_wer[ml].append(row[ml])
    W("| Modell | Ø WER | Beste Datei | Schlechteste Datei |")
    W("|--------|-------|-------------|-------------------|")
    for ml in MODEL_ORDER:
        vals = avg_wer[ml]
        if not vals:
            W(f"| {ml} | — | — | — |")
            continue
        avg = sum(vals) / len(vals)
        files_with_gt = [af for af in audio_files if wer_table[af][ml] is not None]
        best_af = min(files_with_gt, key=lambda af: wer_table[af][ml])
        worst_af = max(files_with_gt, key=lambda af: wer_table[af][ml])
        W(f"| {ml} | {fmt_wer(avg)} | {AUDIO_DISPLAY[best_af]} ({fmt_wer(wer_table[best_af][ml])}) | {AUDIO_DISPLAY[worst_af]} ({fmt_wer(wer_table[worst_af][ml])}) |")
    W("")

    W("### 5.2 Formatierungsqualität\n")
    W("| Modell | Ø Text-Sim. | Speaker-Fehler | Wortdiff > 20 |")
    W("|--------|-------------|---------------|--------------|")
    fmt_stats = {ml: {"sims": [], "speaker_err": 0, "worddiff": 0} for ml in MODEL_ORDER}
    for af in audio_files:
        for ml in MODEL_ORDER:
            entry = grouped[af].get(ml)
            if not entry:
                continue
            r = eval_formatting(entry)
            fmt_stats[ml]["sims"].append(r["text_preserved"])
            if not r["speaker_replaced"]:
                fmt_stats[ml]["speaker_err"] += 1
            if abs(r["word_diff"]) > 20:
                fmt_stats[ml]["worddiff"] += 1
    for ml in MODEL_ORDER:
        s = fmt_stats[ml]
        avg_sim = sum(s["sims"]) / len(s["sims"]) if s["sims"] else 0
        W(f"| {ml} | {fmt_pct(avg_sim)} | {s['speaker_err']}/10 | {s['worddiff']}/10 |")
    W("")

    W("### 5.3 SOAP-Qualität\n")
    W("| Modell | Struktur vollst. | Ø Schlüsselbegriff-Abdeckung | Halluz.-Fälle |")
    W("|--------|-----------------|------------------------------|--------------|")
    soap_stats = {ml: {"complete": 0, "coverage": [], "halluz": 0} for ml in MODEL_ORDER}
    for af in audio_files:
        for ml in MODEL_ORDER:
            r = soap_results[af].get(ml)
            if not r:
                continue
            if r["structure_complete"]:
                soap_stats[ml]["complete"] += 1
            if r["terms_total"]:
                soap_stats[ml]["coverage"].append(r["term_coverage"])
            if r["hallucinated_numbers"]:
                soap_stats[ml]["halluz"] += 1
    for ml in MODEL_ORDER:
        s = soap_stats[ml]
        avg_cov = (sum(s["coverage"]) / len(s["coverage"]) if s["coverage"] else 0)
        W(f"| {ml} | {s['complete']}/10 | {fmt_pct(avg_cov)} | {s['halluz']}/10 |")
    W("")

    W("### 5.4 Gesamtfazit\n")
    W("| Dimension | Bestes Modell | Schwächstes Modell |")
    W("|-----------|--------------|-------------------|")
    # STT winner: lowest avg WER
    stt_avgs = {ml: sum(avg_wer[ml])/len(avg_wer[ml]) for ml in MODEL_ORDER if avg_wer[ml]}
    stt_best = min(stt_avgs, key=stt_avgs.get)
    stt_worst = max(stt_avgs, key=stt_avgs.get)
    W(f"| STT (WER) | {stt_best} ({fmt_wer(stt_avgs[stt_best])}) | {stt_worst} ({fmt_wer(stt_avgs[stt_worst])}) |")
    # Formatting winner: highest avg text sim
    fmt_avgs = {ml: sum(fmt_stats[ml]["sims"])/len(fmt_stats[ml]["sims"]) for ml in MODEL_ORDER if fmt_stats[ml]["sims"]}
    fmt_best = max(fmt_avgs, key=fmt_avgs.get)
    fmt_worst = min(fmt_avgs, key=fmt_avgs.get)
    W(f"| Formatierung (Text-Sim.) | {fmt_best} ({fmt_pct(fmt_avgs[fmt_best])}) | {fmt_worst} ({fmt_pct(fmt_avgs[fmt_worst])}) |")
    # SOAP winner: completeness, then key term coverage as tiebreaker
    soap_coverage = {ml: (sum(soap_stats[ml]["coverage"]) / len(soap_stats[ml]["coverage"]) if soap_stats[ml]["coverage"] else 0) for ml in MODEL_ORDER}
    soap_best = max(MODEL_ORDER, key=lambda ml: (soap_stats[ml]["complete"], soap_coverage[ml]))
    soap_worst = min(MODEL_ORDER, key=lambda ml: (soap_stats[ml]["complete"], soap_coverage[ml]))
    if soap_stats[soap_best]["complete"] == soap_stats[soap_worst]["complete"]:
        W(f"| SOAP (Schlüsselbegriffe) | {soap_best} ({fmt_pct(soap_coverage[soap_best])}) | {soap_worst} ({fmt_pct(soap_coverage[soap_worst])}) |")
    else:
        W(f"| SOAP (Struktur) | {soap_best} ({soap_stats[soap_best]['complete']}/10) | {soap_worst} ({soap_stats[soap_worst]['complete']}/10) |")
    W("")
    W("**Stärken und Schwächen:**\n")
    W("- **Speechmatics+GPT-4o**: Stärkste Formatierung und SOAP-Qualität durch GPT-4o; "
      "Speechmatics-STT produziert manchmal fehlende Diarizerungsinfo (nur ein Speaker-Block).")
    W("- **Whisper+Sauerkraut**: Gute STT-Qualität durch Whisper-Diarisierung (SPEAKER_00/01); "
      "SauerkrautLM kompakter in SOAP, aber strukturell oft vollständig.")
    W("- **Whisper+llama3.2**: Gleiche STT-Qualität wie Sauerkraut; llama3.2 neigt zu "
      "mehr ausschweifenden SOAP-Texten und gelegentlichen inhaltlichen Abweichungen.")
    W("")

    result = "\n".join(lines)
    OUT_FILE.write_text(result)
    print(f"Report geschrieben: {OUT_FILE} ({len(result)} Zeichen, {len(lines)} Zeilen)")
    print(f"Verarbeitete Einträge: {len(history)}")

if __name__ == "__main__":
    main()
