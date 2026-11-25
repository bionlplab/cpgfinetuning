"""Verify every decision of every vignette in a pair file.

For each decision on a row's target path, Claude Opus 4.6 is shown the
criterion, its options, and the vignette, and must name the option on the
target path. A pair fails if any decision of either vignette fails. Writes
the failing pair keys to a JSON file.

  python scripts/verify_pairs.py train_pairs_generated.jsonl --out failed_pairs.json
"""
import argparse
import json
from collections import defaultdict

from common import CHECK_PROMPT, read_jsonl, strict_match, tree_from_prompt, vignette_from_prompt, walk
from llm import CachedLLM


def decisions(row):
    """(criterion, options, answer) for each decision with two or more real options."""
    tree = tree_from_prompt(row["messages"][0]["content"])
    traversal = json.loads(row["messages"][1]["content"])["traversal"]
    out = []
    for node, step in walk(tree, traversal):
        if node == "fail":
            return None
        if node == "end":
            break
        options = [e["label"] for e in node.get("edges", [])
                   if e.get("label") and e["label"].lower() != "next"]
        if len(options) >= 2:
            out.append((node["text"], options, step.get("answer")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--out", default="failed_pairs.json")
    ap.add_argument("--cache", default="cache/verify.jsonl")
    args = ap.parse_args()

    llm = CachedLLM("claude-opus-4.6", args.cache)
    failed = defaultdict(list)
    n_checked = 0
    for row in read_jsonl(args.pairs):
        m = row["meta"]
        pair = "|".join(str(m[k]) for k in ("pmcid", "fig_id", "node", "seed") if k in m)
        member = m.get("branch") or m.get("variant")
        decs = decisions(row)
        if decs is None:
            failed[pair].append(f"{member}: target path does not follow the tree")
            continue
        vignette = vignette_from_prompt(row["messages"][0]["content"])
        for i, (criterion, options, wanted) in enumerate(decs):
            prompt = CHECK_PROMPT.format(criterion=criterion, options=" | ".join(options),
                                         vignette=vignette)
            reply = llm.ask(f"{pair}|{member}|d{i}", prompt, 200, temperature=0)
            n_checked += 1
            if not reply or not strict_match(reply, options, wanted):
                failed[pair].append(f"{member}: wanted {wanted!r}, got {(reply or '')[:60]!r}")

    json.dump({"failed_verification": sorted(failed)}, open(args.out, "w"), indent=1)
    print(f"decisions checked: {n_checked}, pairs failing: {len(failed)}")
    for k, reasons in sorted(failed.items()):
        print(" ", k, reasons[0])


if __name__ == "__main__":
    main()
