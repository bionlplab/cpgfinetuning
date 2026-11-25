"""Results on the robustness benchmark.

The first model is the baseline. Prints recommendation accuracy and path
concordance for each model, and the difference from the baseline with a 95%
CI from bootstrap resampling of source articles.

  python scripts/analyze_robustness.py \
      base=outputs/qwen_base/robustness/generated_predictions.jsonl \
      decision_path=outputs/qwen_decision_path/robustness/generated_predictions.jsonl
"""
import argparse

from stats import cluster_bootstrap, load_meta, load_scores, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/robustness.jsonl")
    ap.add_argument("models", nargs="+", help="name=path/to/generated_predictions.jsonl")
    args = ap.parse_args()

    meta = load_meta(args.eval)
    models = [(s.split("=", 1)[0], load_scores(s.split("=", 1)[1])) for s in args.models]
    base = models[0][1]
    for metric, label in ((1, "recommendation accuracy"), (0, "path concordance")):
        print(f"\n{label} (%)")
        print(f"{'model':<16}{'value':>7}{'n':>6}{'diff':>8}  95% CI")
        for name, s in models:
            v = [x[metric] for x in s.values() if x[metric] >= 0]
            line = f"{name:<16}{100 * mean(v):>7.1f}{len(v):>6}"
            if s is not base:
                diffs, articles = [], []
                for i, m in enumerate(meta):
                    if i in s and i in base and s[i][metric] >= 0 and base[i][metric] >= 0:
                        diffs.append(s[i][metric] - base[i][metric])
                        articles.append(m["pmcid"])
                if diffs:
                    lo, hi = cluster_bootstrap(diffs, articles)
                    line += f"{100 * mean(diffs):>+8.1f}  [{100 * lo:.1f}, {100 * hi:.1f}]"
            print(line)


if __name__ == "__main__":
    main()
