#!/usr/bin/env python3
"""
Compare your human grades with the LLM judge's grades in answers_*.csv files.

Reports, overall and by question type:
  - your accuracy (the number to report), with a bootstrap 95% confidence interval
  - the judge's accuracy on the same questions
  - how often the judge agreed with you (exact match and Cohen's kappa)
  - whether the judge's errors are lenient (graded higher than you) or harsh
  - every question where you and the judge disagreed

Usage (from the rag_pipeline folder):
    python judge_agreement.py answers_dev_qwen2.5-7b_k10_gf.csv

Compare several judges on the same answers (human grades taken from --human-from
wherever the other files' human_grade column is empty; matched by question and
identical answer text):
    python judge_agreement.py answers_dev_qwen2.5-7b_k10_gf.csv \
        answers_dev_qwen2.5-7b_k10_gf_judge-qwen2.5-14b.csv \
        --human-from answers_dev_qwen2.5-7b_k10_gf.csv

Grades accepted in human_grade (any case): correct / c / 1, partial / p / 0.5,
incorrect / i / wrong / 0.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

LEVELS = ["incorrect", "partial", "correct"]
SCORE = {"incorrect": 0.0, "partial": 0.5, "correct": 1.0}
ALIASES = {
    "correct": "correct", "c": "correct", "1": "correct", "1.0": "correct", "yes": "correct", "y": "correct",
    "partial": "partial", "p": "partial", "0.5": "partial", ".5": "partial", "partly": "partial",
    "incorrect": "incorrect", "i": "incorrect", "wrong": "incorrect", "0": "incorrect", "0.0": "incorrect",
    "no": "incorrect", "n": "incorrect", "x": "incorrect",
}
BOOTSTRAP_ITERS = 5000
SEED = 7


def norm(grade: str | None) -> str | None:
    return ALIASES.get((grade or "").strip().lower())


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def accuracy(grades: list[str]) -> float:
    return sum(SCORE[g] for g in grades) / len(grades) if grades else float("nan")


def bootstrap_ci(grades: list[str]) -> tuple[float, float]:
    if len(grades) < 2:
        return float("nan"), float("nan")
    rng = random.Random(SEED)
    stats = sorted(accuracy([rng.choice(grades) for _ in grades]) for _ in range(BOOTSTRAP_ITERS))
    return stats[int(0.025 * BOOTSTRAP_ITERS)], stats[int(0.975 * BOOTSTRAP_ITERS) - 1]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[l] * cb[l] for l in LEVELS) / (n * n)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def pct(v: float) -> str:
    return "–" if v != v else f"{100 * v:.0f}%"  # v != v catches NaN


def analyze(path: Path, human_lookup: dict) -> tuple[list[str], dict]:
    rows = read(path)
    pairs, skipped, unparsed_judge, borrowed = [], [], [], 0
    for r in rows:
        h = norm(r.get("human_grade"))
        if h is None and human_lookup:
            h = norm(human_lookup.get((r["qid"], r["model_answer"], r.get("gold_answer", ""))))
            borrowed += h is not None
        j = norm(r.get("judge_correctness"))
        if h is None:
            skipped.append(r["qid"])
            continue
        if j is None:
            unparsed_judge.append(r["qid"])
            j = "incorrect"  # an unusable judge verdict counts as a miss
        pairs.append({**r, "human": h, "judge": j})

    if not pairs:
        sys.exit(f"{path}: no rows with a usable human_grade.")

    types = sorted({p["type"] for p in pairs})
    lines = [f"# Judge agreement: {path.name}", ""]
    notes = [f"Graded questions: {len(pairs)} of {len(rows)}"]
    if borrowed:
        notes.append(f"{borrowed} human grades taken from --human-from (identical answers)")
    if skipped:
        notes.append(f"no human grade (skipped): {', '.join(skipped)}")
    if unparsed_judge:
        notes.append(f"judge verdict unreadable, counted as incorrect: {', '.join(unparsed_judge)}")
    lines += [" | ".join(notes), "",
              "Accuracy: correct = 1, partial = 0.5, incorrect = 0. CI = bootstrap 95% interval "
              "over questions. Kappa: 0 = chance-level agreement, 1 = perfect.", "",
              "| type | n | your accuracy | 95% CI | judge accuracy | agreement | kappa | judge too lenient | judge too harsh |",
              "|---|---|---|---|---|---|---|---|---|"]

    summary = {}
    for t in types + ["ALL"]:
        sel = [p for p in pairs if t == "ALL" or p["type"] == t]
        h, j = [p["human"] for p in sel], [p["judge"] for p in sel]
        lo, hi = bootstrap_ci(h)
        lenient = sum(SCORE[b] > SCORE[a] for a, b in zip(h, j))
        harsh = sum(SCORE[b] < SCORE[a] for a, b in zip(h, j))
        agree = sum(a == b for a, b in zip(h, j)) / len(sel)
        kappa = cohen_kappa(h, j)
        name = f"**{t}**" if t == "ALL" else t
        ci = "–" if lo != lo else f"{pct(lo)}–{pct(hi)}"
        lines.append(f"| {name} | {len(sel)} | {pct(accuracy(h))} | {ci} | {pct(accuracy(j))} | "
                     f"{pct(agree)} | {'–' if kappa != kappa else f'{kappa:.2f}'} | {lenient} | {harsh} |")
        if t == "ALL":
            summary = {"n": len(sel), "human_acc": accuracy(h), "ci": (lo, hi), "judge_acc": accuracy(j),
                       "agree": agree, "kappa": kappa, "lenient": lenient, "harsh": harsh}

    # confusion matrix
    cm = defaultdict(int)
    for p in pairs:
        cm[(p["human"], p["judge"])] += 1
    lines += ["", "## Confusion matrix (rows = your grade, columns = judge)", "",
              "| you \\ judge | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for h in LEVELS:
        lines.append(f"| {h} | " + " | ".join(str(cm[(h, j)]) for j in LEVELS) + " |")

    # disagreements
    dis = [p for p in pairs if p["human"] != p["judge"]]
    lines += ["", f"## Disagreements ({len(dis)})", ""]
    if dis:
        cell = lambda s: str(s).replace("|", "/").replace("\n", " ")
        lines += ["| qid | type | you | judge | direction | question | model answer | gold answer | judge reason |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in dis:
            direction = "lenient" if SCORE[p["judge"]] > SCORE[p["human"]] else "harsh"
            lines.append(f"| {p['qid']} | {p['type']} | {p['human']} | {p['judge']} | {direction} | "
                         f"{cell(p['question'])} | {cell(p['model_answer'])} | {cell(p['gold_answer'])} | "
                         f"{cell(p.get('judge_reason', ''))} |")
    else:
        lines.append("None.")
    return lines, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", type=Path, nargs="+", help="one or more answers_*.csv files")
    ap.add_argument("--human-from", type=Path,
                    help="file whose human_grade column fills in grades missing from the others")
    args = ap.parse_args()

    lookup = {}
    if args.human_from:
        lookup = {(r["qid"], r["model_answer"], r.get("gold_answer", "")): r.get("human_grade", "")
                  for r in read(args.human_from)}

    summaries = []
    for path in args.answers:
        if not path.exists():
            sys.exit(f"Not found: {path}")
        lines, summary = analyze(path, lookup)
        out = path.with_name(f"judge_agreement_{path.stem.removeprefix('answers_')}.md")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines[:lines.index("## Confusion matrix (rows = your grade, columns = judge)")]))
        print(f"Full report with disagreements: {out}\n")
        summaries.append((path.name, summary))

    if len(summaries) > 1:
        print("# Judge comparison (same answers, your grades as ground truth)\n")
        print("| file | n | your accuracy | judge accuracy | agreement | kappa | lenient | harsh |")
        print("|---|---|---|---|---|---|---|---|")
        for name, s in summaries:
            k = "–" if s["kappa"] != s["kappa"] else f"{s['kappa']:.2f}"
            print(f"| {name} | {s['n']} | {pct(s['human_acc'])} | {pct(s['judge_acc'])} | {pct(s['agree'])} | "
                  f"{k} | {s['lenient']} | {s['harsh']} |")


if __name__ == "__main__":
    main()
