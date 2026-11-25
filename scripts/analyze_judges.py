"""Agreement between two judges on the same responses in the primary evaluation.

Prints, for recommendation accuracy, the number of responses on which the two
judges agree, percentage agreement, and Cohen's kappa, each with a 95% CI from
bootstrap resampling of source articles. Also prints path concordance under
each judge.

  python scripts/analyze_judges.py \
      outputs/qwen_base/primary/generated_predictions.jsonl.judge.jsonl \
      outputs/second_judge/qwen_base_primary.judge.jsonl
"""
import argparse
import json

from common import pair_key, read_jsonl
from stats import cluster_bootstrap, kappa, load_scores, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judge_a")
    ap.add_argument("judge_b")
    ap.add_argument("--eval", default="data/primary.jsonl")
    ap.add_argument("--pairs", default="data/primary_pairs.json")
    args = ap.parse_args()

    meta = [r["meta"] for r in read_jsonl(args.eval)]
    keep = set(json.load(open(args.pairs)))
    a, b = load_scores(args.judge_a), load_scores(args.judge_b)
    items = [i for i, m in enumerate(meta)
             if pair_key(m) in keep and a[i][1] >= 0 and b[i][1] >= 0]
    ratings = [(a[i][1], b[i][1]) for i in items]
    articles = [meta[i]["pmcid"] for i in items]

    agree = lambda pairs: mean([x == y for x, y in pairs])
    lo, hi = cluster_bootstrap(ratings, articles, stat=agree)
    k_lo, k_hi = cluster_bootstrap(ratings, articles, stat=kappa)
    n_agree = sum(x == y for x, y in ratings)
    print(f"responses: {len(ratings)}, agreed: {n_agree} "
          f"({sum(x == y == 1 for x, y in ratings)} correct, {sum(x == y == 0 for x, y in ratings)} incorrect)")
    print(f"agreement: {100 * n_agree / len(ratings):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
    print(f"kappa: {kappa(ratings):.3f} [{k_lo:.3f}, {k_hi:.3f}]")
    for name, s in (("judge A", a), ("judge B", b)):
        print(f"{name}: accuracy {100 * mean([s[i][1] for i in items]):.1f}, "
              f"concordance {100 * mean([s[i][0] for i in items if s[i][0] >= 0]):.1f}")


if __name__ == "__main__":
    main()
