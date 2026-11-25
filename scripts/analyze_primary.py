"""Results on the primary evaluation (matched vignette pairs).

The first model is the baseline. For each model, prints recommendation
accuracy and path concordance on all, original, and perturbed vignettes, the
perturbation gap (perturbed minus original), and the difference from the
baseline with a 95% CI from bootstrap resampling of source articles. Then
prints pair-level outcomes, recommendations corrected and broken relative to
the baseline, and the perturbation gap by the number of sentences changed.

  python scripts/analyze_primary.py \
      base=outputs/qwen_base/primary/generated_predictions.jsonl \
      decision_path=outputs/qwen_decision_path/primary/generated_predictions.jsonl
"""
import argparse
import difflib
import json
import re
from collections import Counter

from common import pair_key, read_jsonl, vignette_from_prompt
from stats import cluster_bootstrap, load_scores, mean

ACCURACY, CONCORDANCE = 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/primary.jsonl")
    ap.add_argument("--pairs", default="data/primary_pairs.json")
    ap.add_argument("--overlap", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                    help="count perturbed vignettes that both models get wrong")
    ap.add_argument("models", nargs="+", help="name=path/to/generated_predictions.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.eval)
    meta = [r["meta"] for r in rows]
    keep = set(json.load(open(args.pairs)))
    kept = [i for i, m in enumerate(meta) if pair_key(m) in keep]
    pairs = {}
    for i in kept:
        pairs.setdefault(pair_key(meta[i]), {})[meta[i]["variant"]] = i
    models = [(s.split("=", 1)[0], load_scores(s.split("=", 1)[1])) for s in args.models]
    base_name, base = models[0]
    print(f"{len(pairs)} pairs, {len(kept)} vignettes, "
          f"{len({(meta[i]['pmcid'], meta[i]['fig_id']) for i in kept})} trees, "
          f"{len({meta[i]['pmcid'] for i in kept})} articles")

    for metric, label in ((ACCURACY, "recommendation accuracy"), (CONCORDANCE, "path concordance")):
        print(f"\n{label} (%)")
        print(f"{'model':<16}{'all':>7}{'orig':>7}{'pert':>7}{'gap':>7}{'diff':>8}  95% CI")
        for name, s in models:
            v = {"original": [], "perturbation": []}
            for i in kept:
                if i in s and s[i][metric] >= 0:
                    v[meta[i]["variant"]].append(s[i][metric])
            orig, pert = mean(v["original"]), mean(v["perturbation"])
            line = (f"{name:<16}{100 * mean(v['original'] + v['perturbation']):>7.1f}"
                    f"{100 * orig:>7.1f}{100 * pert:>7.1f}{100 * (pert - orig):>+7.1f}")
            if s is not base:
                diffs, articles = [], []
                for i in kept:
                    if i in s and i in base and s[i][metric] >= 0 and base[i][metric] >= 0:
                        diffs.append(s[i][metric] - base[i][metric])
                        articles.append(meta[i]["pmcid"])
                lo, hi = cluster_bootstrap(diffs, articles)
                line += f"{100 * mean(diffs):>+8.1f}  [{100 * lo:.1f}, {100 * hi:.1f}]"
            print(line)

    def correct(s, i):
        return None if s.get(i, (-1, -1))[ACCURACY] < 0 else s[i][ACCURACY] >= 1

    print("\npair outcomes: both correct / original only / perturbed only / neither")
    states = {}
    for name, s in models:
        states[name] = {k: (correct(s, v["original"]), correct(s, v["perturbation"]))
                        for k, v in pairs.items()}
        states[name] = {k: st for k, st in states[name].items() if None not in st}
        c = Counter({(True, True): "both", (True, False): "orig", (False, True): "pert",
                     (False, False): "neither"}[st] for st in states[name].values())
        print(f"{name:<16}{c['both']:>5}{c['orig']:>5}{c['pert']:>5}{c['neither']:>5}")

    # Changes on perturbed vignettes relative to the baseline
    base_wrong = {k for k, (o, p) in states[base_name].items() if o and not p}
    base_right = {k for k, (o, p) in states[base_name].items() if p}
    print(f"\nperturbed vignettes, relative to {base_name}")
    print(f"pairs with only the original correct under {base_name}: {len(base_wrong)}")
    for name, s in models[1:]:
        fixed = sum(1 for k in base_wrong if k in states[name] and states[name][k][1])
        broken = sum(1 for k in base_right if k in states[name] and not states[name][k][1])
        gained = sum(1 for v in pairs.values() if correct(s, v["perturbation"]) is True
                     and correct(base, v["perturbation"]) is False)
        lost = sum(1 for v in pairs.values() if correct(base, v["perturbation"]) is True
                   and correct(s, v["perturbation"]) is False)
        print(f"  {name}: corrects {fixed} of {len(base_wrong)}, breaks {broken} "
              f"of {len(base_right)}; correct only under {name}: {gained}, only under {base_name}: {lost}")

    # Perturbation gap by the number of sentences that differ within a pair
    def sentences(t):
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]

    sizes = {}
    for k, v in pairs.items():
        a = sentences(vignette_from_prompt(rows[v["original"]]["messages"][0]["content"]))
        b = sentences(vignette_from_prompt(rows[v["perturbation"]]["messages"][0]["content"]))
        ops = difflib.SequenceMatcher(None, a, b).get_opcodes()
        sizes[k] = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    print(f"\nperturbation gap in accuracy by sentences changed "
          f"(median {sorted(sizes.values())[len(sizes) // 2]})")
    for cap in (1, 2, 3, None):
        ks = [k for k, n in sizes.items() if n >= 1 and (cap is None or n <= cap)]
        label = "all" if cap is None else "1" if cap == 1 else f"1 to {cap}"
        print(f"  {label} (n={len(ks)}):", end="")
        for name, s in models:
            o = [s[pairs[k]["original"]][ACCURACY] for k in ks
                 if s.get(pairs[k]["original"], (-1, -1))[ACCURACY] >= 0]
            p = [s[pairs[k]["perturbation"]][ACCURACY] for k in ks
                 if s.get(pairs[k]["perturbation"], (-1, -1))[ACCURACY] >= 0]
            print(f"  {name} {100 * (mean(p) - mean(o)):+.1f}", end="")
        print()

    if args.overlap:
        a, b = (dict(models)[n] for n in args.overlap)
        both = sum(1 for v in pairs.values()
                   if correct(a, v["perturbation"]) is False and correct(b, v["perturbation"]) is False)
        print(f"\nperturbed vignettes wrong under both {args.overlap[0]} and {args.overlap[1]}: {both}")


if __name__ == "__main__":
    main()
