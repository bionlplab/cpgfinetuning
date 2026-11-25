"""Select the matched pairs used in the primary evaluation.

Pairs are excluded, in this order, when:
  1. the two vignettes have identical text
  2. either vignette failed strict verification (data/primary_failed_verification.json)
  3. both vignettes lead to the same recommendation
  4. the changed decision node has two edges with the same label
  5. a target step answers "next" at a node whose edges all have real labels,
     or the pair is the BMI pair checked below
  6. a target path passes a node with two edges that have the same label

  python scripts/select_primary_pairs.py --out data/primary_pairs.json
"""
import argparse
import json
from collections import defaultdict

from common import pair_key, read_jsonl, tree_from_prompt, vignette_from_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/primary.jsonl")
    ap.add_argument("--failed", default="data/primary_failed_verification.json")
    ap.add_argument("--out", default="data/primary_pairs.json")
    args = ap.parse_args()

    rows = read_jsonl(args.eval)
    pairs = defaultdict(dict)
    for i, r in enumerate(rows):
        pairs[pair_key(r["meta"])][r["meta"]["variant"]] = i
    print("generated pairs:", len(pairs))

    text = lambda i: vignette_from_prompt(rows[i]["messages"][0]["content"])
    target = lambda i: json.loads(rows[i]["messages"][-1]["content"])

    identical = {k for k, v in pairs.items() if text(v["original"]) == text(v["perturbation"])}
    failed = set(json.load(open(args.failed))["failed_verification"])
    remaining = {k: v for k, v in pairs.items() if k not in identical | failed}

    same_rec, dup_at_node, label_error, dup_on_path = set(), set(), set(), set()
    for k, v in remaining.items():
        tree = tree_from_prompt(rows[v["original"]]["messages"][0]["content"])
        nodes = {n["id"]: n for n in tree["nodes"]}
        by_text = defaultdict(list)
        for n in tree["nodes"]:
            by_text[n["text"]].append(n)
        if target(v["original"]).get("recommendation") == target(v["perturbation"]).get("recommendation"):
            same_rec.add(k)
        labels = [e.get("label", "") for e in nodes[rows[v["original"]]["meta"]["node"]].get("edges", [])]
        if len(labels) != len(set(labels)):
            dup_at_node.add(k)
        for i in v.values():
            for step in target(i)["traversal"]:
                for n in by_text.get(step.get("decision_point"), []):
                    labs = [e.get("label", "") for e in n.get("edges", [])]
                    if len(labs) != len(set(labs)):
                        dup_on_path.add(k)
                    if (str(step.get("answer", "")).lower() == "next" and labs
                            and all(l and l.lower() != "next" for l in labs)):
                        label_error.add(k)

    # This perturbed vignette states a BMI of 24.8, but its target path takes the ">= 30" branch.
    bmi = "PMC7532757|fig0005|n7|1"
    if bmi in remaining:
        i = remaining[bmi]["perturbation"]
        assert "24.8" in rows[i]["messages"][0]["content"]
        assert any(s.get("decision_point") == "BMI" and "30" in str(s.get("answer", ""))
                   for s in target(i)["traversal"])
        label_error.add(bmi)

    label_error -= same_rec | dup_at_node
    dup_on_path -= same_rec | dup_at_node | label_error
    print("1. identical text:", len(identical))
    print("2. failed verification:", len(failed))
    print("3. same recommendation:", len(same_rec))
    print("4. duplicate labels at the changed node:", len(dup_at_node - same_rec),
          f"(plus {len(dup_at_node & same_rec)} already excluded by rule 3)")
    print("5. target label errors:", len(label_error))
    print("6. duplicate labels on the target path:", len(dup_on_path))

    keep = sorted(set(pairs) - identical - failed - same_rec - dup_at_node - label_error - dup_on_path)
    trees = {tuple(k.split("|")[:2]) for k in keep}
    articles = {k.split("|")[0] for k in keep}
    print(f"kept: {len(keep)} pairs, {len(trees)} trees, {len(articles)} articles")
    json.dump(keep, open(args.out, "w"), indent=0)


if __name__ == "__main__":
    main()
