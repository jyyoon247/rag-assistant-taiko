#!/usr/bin/env python3
"""
Paired comparison of two or more answer runs, using YOUR human grades.

Every run answered the same questions, so instead of comparing two averages with
wide, overlapping confidence intervals, this compares them question by question:

  - wins / losses / ties: on how many questions the new setup scored higher,
    lower or the same as the baseline (correct 1, partial 0.5, incorrect 0)
  - difference in accuracy with a PAIRED bootstrap 95% interval (resamples
    questions, keeping each question's pair of grades together)
  - sign test p-value: if the two setups were equally good, how likely is a
    win/loss split at least this lopsided? (ties ignored)
  - the list of questions that changed, with both answers

Usage (from the rag_pipeline folder):
    python compare_runs.py answers_test_qwen2.5-7b_k10_gf.csv \
        answers_test_qwen2.5-7b_k10_gf_router-pg.csv

    # baseline vs several setups at once:
    python compare_runs.py BASELINE.csv SETUP1.csv SETUP2.csv SETUP3.csv

Only questions with a human grade in both files are compared.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from math import comb
from pathlib import Path

from judge_agreement import SCORE, norm

BOOTSTRAP_ITERS = 10000
SEED = 11


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = {r["qid"]: r for r in csv.DictReader(f)}
    for r in rows.values():
        r["_grade"] = norm(r.get("human_grade"))
    return rows


def paired_ci(diffs: list[float]) -> tuple[float, float]:
    if len(diffs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(SEED)
    n = len(diffs)
    stats = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(BOOTSTRAP_ITERS))
    return stats[int(0.025 * BOOTSTRAP_ITERS)], stats[int(0.975 * BOOTSTRAP_ITERS) - 1]


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def pts(v: float) -> str:
    return "–" if v != v else f"{100 * v:+.0f}"


def pct(v: float) -> str:
    return "–" if v != v else f"{100 * v:.0f}%"


def compare(base_path: Path, new_path: Path, types_filter: set[str] | None) -> tuple[list[str], dict]:
    base, new = load(base_path), load(new_path)
    common = [q for q in base if q in new and base[q]["_grade"] and new[q]["_grade"]
              and (not types_filter or base[q]["type"] in types_filter)]
    skipped = [q for q in base if q in new and q not in common
               and (not types_filter or base[q]["type"] in types_filter)]
    if not common:
        sys.exit(f"No questions with human grades in both {base_path} and {new_path}.")

    types = sorted({base[q]["type"] for q in common})
    lines = [f"# {new_path.name}", f"vs baseline {base_path.name}", "",
             f"Questions compared: {len(common)}"
             + (f" | skipped (missing a human grade): {', '.join(skipped)}" if skipped else ""), "",
             "Difference = new minus baseline, in accuracy points. Interval = paired bootstrap 95%. "
             "p = two-sided sign test on wins vs losses (ties ignored); small samples rarely reach p < 0.05.", "",
             "| type | n | baseline | new | difference | 95% interval | wins | losses | ties | sign test p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    overall = {}
    for t in types + ["ALL"]:
        qs = [q for q in common if t == "ALL" or base[q]["type"] == t]
        a = [SCORE[base[q]["_grade"]] for q in qs]
        b = [SCORE[new[q]["_grade"]] for q in qs]
        diffs = [y - x for x, y in zip(a, b)]
        wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        ties = len(diffs) - wins - losses
        lo, hi = paired_ci(diffs)
        p = sign_test(wins, losses)
        name = f"**{t}**" if t == "ALL" else t
        interval = "–" if lo != lo else f"{pts(lo)} to {pts(hi)}"
        lines.append(f"| {name} | {len(qs)} | {pct(sum(a) / len(a))} | {pct(sum(b) / len(b))} | "
                     f"{pts(sum(diffs) / len(diffs))} | {interval} | {wins} | {losses} | {ties} | "
                     f"{'–' if p != p else f'{p:.2f}'} |")
        if t == "ALL":
            overall = {"n": len(qs), "base": sum(a) / len(a), "new": sum(b) / len(b),
                       "diff": sum(diffs) / len(diffs), "lo": lo, "hi": hi,
                       "wins": wins, "losses": losses, "ties": ties, "p": p}

    changed = [q for q in common if base[q]["_grade"] != new[q]["_grade"]]
    lines += ["", f"## Questions whose grade changed ({len(changed)})", ""]
    if changed:
        cell = lambda s: str(s).replace("|", "/").replace("\n", " ")
        lines += ["| qid | type | baseline grade | new grade | route (new) | question | baseline answer | new answer |",
                  "|---|---|---|---|---|---|---|---|"]
        for q in sorted(changed, key=lambda q: (SCORE[new[q]["_grade"]] - SCORE[base[q]["_grade"]], q)):
            lines.append(f"| {q} | {base[q]['type']} | {base[q]['_grade']} | {new[q]['_grade']} | "
                         f"{new[q].get('route', '')} | {cell(base[q]['question'])} | "
                         f"{cell(base[q]['model_answer'])} | {cell(new[q]['model_answer'])} |")
    else:
        lines.append("None.")
    return lines, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("others", type=Path, nargs="+")
    ap.add_argument("--types", help="only these question types, comma-separated (e.g. aggregate,multi_doc)")
    args = ap.parse_args()
    types_filter = {t.strip() for t in args.types.split(",")} if args.types else None

    summaries = []
    for path in [args.baseline] + args.others:
        if not path.exists():
            sys.exit(f"Not found: {path}")
    for path in args.others:
        lines, overall = compare(args.baseline, path, types_filter)
        out = path.with_name(f"compare_{args.baseline.stem.removeprefix('answers_')}"
                             f"__vs__{path.stem.removeprefix('answers_')}.md")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines[:lines.index(next(l for l in lines if l.startswith("## Questions")))]))
        print(f"Full list of changed questions: {out}\n")
        summaries.append((path.name, overall))

    if len(summaries) > 1:
        print(f"# Summary vs {args.baseline.name}\n")
        print("| setup | n | baseline | new | difference | 95% interval | wins / losses / ties | sign test p |")
        print("|---|---|---|---|---|---|---|---|")
        for name, s in summaries:
            interval = "–" if s["lo"] != s["lo"] else f"{pts(s['lo'])} to {pts(s['hi'])}"
            p_str = "–" if s["p"] != s["p"] else f"{s['p']:.2f}"
            print(f"| {name} | {s['n']} | {pct(s['base'])} | {pct(s['new'])} | {pts(s['diff'])} | {interval} | "
                  f"{s['wins']} / {s['losses']} / {s['ties']} | {p_str} |")


if __name__ == "__main__":
    main()
