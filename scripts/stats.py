"""Loading judge scores and computing bootstrap confidence intervals."""
import json
import os
import random
import sys


def load_meta(path):
    with open(path) as f:
        return [json.loads(line)["meta"] for line in f if line.strip()]


def load_scores(predictions):
    """{row index: (concordance, accuracy)} from <predictions>.judge.jsonl."""
    path = predictions if predictions.endswith(".judge.jsonl") else predictions + ".judge.jsonl"
    if not os.path.exists(path):
        sys.exit(f"missing {path}, run judge.py first")
    scores = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                scores[r["i"]] = (r.get("p", -1), r.get("o", -1))
    return scores


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def cluster_bootstrap(values, clusters, stat=mean, n_rep=10000, seed=0):
    """95% CI of stat(values), resampling whole clusters with replacement.

    With clusters=None, single items are resampled.
    """
    rng = random.Random(seed)
    if clusters is None:
        groups = [[v] for v in values]
    else:
        by = {}
        for v, c in zip(values, clusters):
            by.setdefault(c, []).append(v)
        groups = list(by.values())
    stats = []
    for _ in range(n_rep):
        sample = []
        for _ in range(len(groups)):
            sample.extend(groups[rng.randrange(len(groups))])
        s = stat(sample)
        if s == s:
            stats.append(s)
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def kappa(pairs):
    """Cohen's kappa for a list of (rating_a, rating_b) with binary ratings."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0
