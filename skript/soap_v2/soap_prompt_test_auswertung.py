"""
Erstellt docs/v2/soap_prompt_test_auswertung.md aus results/v2/soap_prompt_test_eval.json.

Zeigt: Baseline vs. Kandidat je Szenario/Modell mit Δ gesamt_score.
"""
import json
from pathlib import Path

REPO     = Path(__file__).parent.parent.parent
IN_FILE  = REPO / "results" / "v2" / "soap_prompt_test_eval.json"
OUT_FILE = REPO / "docs" / "v2" / "soap_prompt_test_auswertung.md"

SZENARIEN   = ["OriginalDC", "Chaos", "Anamnesegespräch"]
MODELLE     = ["gpt-4o", "llama3.2"]
VARIANTEN   = ["baseline", "kandidat"]


def urteil(score: int) -> str:
    if score >= 7: return "akzeptabel"
    if score >= 4: return "überarb. nötig"
    return "nicht verwendbar"


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(IN_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Index: (scenario, model, prompt_variante) → entry
    idx: dict[tuple, dict] = {}
    for e in data:
        if "fehler" in e:
            continue
        k = (e["scenario"], e["model"], e["prompt_variante"])
        idx[k] = e

    out = [
        "# SOAP-Prompt-Engineering: Baseline vs. Kandidat",
        "",
        "Eingabe: Ground-Truth-Transkripte (3 Szenarien × 2 Modelle × 2 Prompts = 12 Läufe)",
        "Evaluiert mit Claude (soap_eval_prompt.md, 0–8 Punkte).",
        "",
        "---",
        "",
    ]

    # ── Detailtabelle ────────────────────────────────────────────────────────
    out.append("## Ergebnisse im Detail")
    out.append("")
    out.append("| Szenario | Modell | Variante | S | O | A | P | Gesamt | Urteil |")
    out.append("|---|---|---|---|---|---|---|---|---|")

    for szenario in SZENARIEN:
        for model in MODELLE:
            for var in VARIANTEN:
                e = idx.get((szenario, model, var))
                if e is None:
                    out.append(f"| {szenario} | {model} | {var} | — | — | — | — | — | — |")
                    continue
                s = e.get("S", {}).get("score", "?")
                o = e.get("O", {}).get("score", "?")
                a = e.get("A", {}).get("score", "?")
                p = e.get("P", {}).get("score", "?")
                gs = e.get("gesamt_score", 0)
                out.append(f"| {szenario} | {model} | {var} | {s} | {o} | {a} | {p} | **{gs}** | {urteil(gs)} |")

    # ── Vergleichstabelle Δ ──────────────────────────────────────────────────
    out += [
        "",
        "---",
        "",
        "## Vergleich: Δ gesamt_score (Kandidat − Baseline)",
        "",
        "| Szenario | Modell | Baseline | Kandidat | Δ |",
        "|---|---|---|---|---|",
    ]

    for szenario in SZENARIEN:
        for model in MODELLE:
            base = idx.get((szenario, model, "baseline"))
            kand = idx.get((szenario, model, "kandidat"))
            gs_b = base.get("gesamt_score", 0) if base else 0
            gs_k = kand.get("gesamt_score", 0) if kand else 0
            delta = gs_k - gs_b
            sign = "+" if delta > 0 else ""
            out.append(f"| {szenario} | {model} | {gs_b} | {gs_k} | **{sign}{delta}** |")

    # ── Mittelwerte je Variante ──────────────────────────────────────────────
    def avg_score(variante: str) -> float:
        scores = [e.get("gesamt_score", 0) for e in idx.values()
                  if e.get("prompt_variante") == variante]
        return sum(scores) / len(scores) if scores else 0.0

    avg_b = avg_score("baseline")
    avg_k = avg_score("kandidat")
    delta_avg = avg_k - avg_b
    sign = "+" if delta_avg > 0 else ""

    out += [
        "",
        "---",
        "",
        "## Mittelwerte je Prompt-Variante (alle Szenarien & Modelle)",
        "",
        "| Variante | Ø gesamt_score (0–8) |",
        "|---|---|",
        f"| Baseline  | {avg_b:.2f} |",
        f"| Kandidat  | {avg_k:.2f} |",
        f"| **Δ**     | **{sign}{delta_avg:.2f}** |",
        "",
    ]

    OUT_FILE.write_text('\n'.join(out) + '\n', encoding="utf-8")
    print(f"Geschrieben: {OUT_FILE}")


if __name__ == "__main__":
    main()
