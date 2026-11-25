"""Structural checks on the matched-pair training set.

Counts, for the training pairs:
  identical vignette texts within a pair
  pairs whose two target paths take the same branch at the changed node
  target steps that do not correspond to an edge of the tree
  steps whose branch label appears on two edges leading to different nodes
  traversals that stop before a final node, and recommendations that differ
  from the final node text
  trees and articles shared with the two evaluation sets
Every count except the pair, tree, and article totals is expected to be zero.

  python scripts/check_train_pairs.py
"""
import argparse
import json
import re
from collections import defaultdict

from common import read_jsonl, tree_from_prompt, walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/train_matched_pair.jsonl")
    ap.add_argument("--eval", nargs="+", default=["data/primary.jsonl", "data/robustness.jsonl"])
    args = ap.parse_args()

    rows = read_jsonl(args.pairs)
    pairs = defaultdict(dict)
    for i, r in enumerate(rows):
        m = r["meta"]
        pairs[(m["pmcid"], m["fig_id"], m["node"])][m["branch"]] = i
    target = lambda r: json.loads(r["messages"][1]["content"])
    space = lambda s: re.sub(r"\s+", " ", s or "").strip()

    print(f"pairs: {len(pairs)}, rows: {len(rows)}, "
          f"incomplete pairs: {sum(len(v) != 2 for v in pairs.values())}")

    identical = sum(rows[v["A"]]["messages"][0]["content"] == rows[v["B"]]["messages"][0]["content"]
                    for v in pairs.values())
    print("identical vignette texts:", identical)

    same_branch = 0
    for (_, _, nid), v in pairs.items():
        tree = tree_from_prompt(rows[v["A"]]["messages"][0]["content"])
        text = next((n["text"] for n in tree["nodes"] if n["id"] == nid), None)
        answers = []
        for b in ("A", "B"):
            steps = [s for s in target(rows[v[b]])["traversal"] if s.get("decision_point") == text]
            answers.append(steps[0].get("answer") if steps else None)
        same_branch += answers[0] is not None and answers[0] == answers[1]
    print("pairs with the same branch at the changed node:", same_branch)

    off_tree = ambiguous = stopped = rec_mismatch = 0
    for r in rows:
        tree = tree_from_prompt(r["messages"][0]["content"])
        nodes = {n["id"]: n for n in tree["nodes"]}
        by_text = defaultdict(list)
        for n in tree["nodes"]:
            by_text[n["text"]].append(n)
        t = target(r)
        for s in t["traversal"]:
            if not any(e.get("label") == s.get("answer") and nodes[e["target"]]["text"] == s.get("next_step")
                       for n in by_text.get(s.get("decision_point"), []) for e in n.get("edges", [])):
                off_tree += 1
        for node, step in walk(tree, t["traversal"]):
            if node == "fail":
                off_tree += 1
                break
            if node == "end":
                stopped += bool(step.get("edges"))
                rec_mismatch += space(t.get("recommendation")) != space(step["text"])
                break
            same = [e for e in node.get("edges", []) if e.get("label") == step.get("answer")]
            if step.get("answer") and len({nodes[e["target"]]["text"] for e in same}) > 1:
                ambiguous += 1
    print("target steps not on a tree edge:", off_tree)
    print("steps with an ambiguous branch label:", ambiguous)
    print("traversals that stop early:", stopped, "| recommendation mismatches:", rec_mismatch)

    train_trees = {k[:2] for k in pairs}
    train_articles = {k[0] for k in pairs}
    for path in args.eval:
        metas = [r["meta"] for r in read_jsonl(path)]
        trees = {(m["pmcid"], m["fig_id"]) for m in metas}
        articles = {m["pmcid"] for m in metas}
        print(f"{path}: shared trees {len(trees & train_trees)}, "
              f"shared articles {len(articles & train_articles)}")
    print(f"trees: {len(train_trees)}, articles: {len(train_articles)}")


if __name__ == "__main__":
    main()
