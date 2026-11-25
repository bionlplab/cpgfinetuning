"""Generate matched vignette pairs for matched-pair fine-tuning.

For each decision node of each training tree, Claude Opus 4.6 writes one
vignette for the first branch and a minimally edited vignette for the second
branch. Target paths and recommendations come from the tree. A pair is kept
only if both vignettes pass verification at every decision on their paths.
The pairs are then checked with verify_pairs.py and check_train_pairs.py.

  python scripts/generate_train_pairs.py --trees data/trees.jsonl --out train_pairs_generated.jsonl
"""
import argparse
import json

from common import (CHECK_PROMPT, build_graph, edges_of, is_decision, make_example,
                    path_to, path_to_leaf, read_jsonl, steps_for, strict_match)
from llm import CachedLLM

ORIGINAL_PROMPT = """Write a realistic clinical vignette of 120-180 words for a patient walking through a clinical guideline.

The vignette's facts must clearly determine these decisions, in this order:
{constraints}

The final action for this patient will be: {rec}

Rules: include age and sex, give concrete values for every deciding fact, do not mention any flowchart, guideline, decision, or the final action. Output the vignette text only."""

PERTURB_PROMPT = """Below is a clinical vignette.

{vignette}

Rewrite it, changing as little as possible, so that this decision now goes the other way:
At "{criterion}" the facts must now clearly select "{newlabel}" instead of "{oldlabel}".
{extra}
Keep every other fact identical. Do not mention any flowchart, guideline, or final action. Output the rewritten vignette only."""


def constraint_lines(decisions):
    return "\n".join(f'- At "{c}", the facts select "{w}".' for c, _, w in decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", default="data/trees.jsonl")
    ap.add_argument("--out", default="train_pairs_generated.jsonl")
    ap.add_argument("--cache", default="cache/train_pairs.jsonl")
    args = ap.parse_args()

    llm = CachedLLM("claude-opus-4.6", args.cache)

    def verified(key, vignette, decisions):
        for i, (criterion, options, wanted) in enumerate(decisions):
            prompt = CHECK_PROMPT.format(criterion=criterion, options=" | ".join(options),
                                         vignette=vignette)
            reply = llm.ask(f"{key}|v{i}", prompt, 120)
            if not reply or not strict_match(reply, options, wanted):
                return False
        return True

    trees = {(r["pmcid"], r["fig_id"]): r["tree"] for r in read_jsonl(args.trees)
             if r["split"] == "train"}
    out = open(args.out, "w")
    kept = failed = 0
    for (pmcid, fig_id), tree in trees.items():
        nodes, root = build_graph(tree)
        for nid, node in nodes.items():
            if not is_decision(node):
                continue
            branches = [e for e in edges_of(node) if e.get("label")]
            prefix = path_to(nodes, root, nid)
            if not prefix:
                continue
            leaf_a = path_to_leaf(nodes, branches[0]["target"])
            leaf_b = path_to_leaf(nodes, branches[1]["target"])
            if not leaf_a or not leaf_b:
                continue
            a = steps_for(nodes, prefix + leaf_a)
            b = steps_for(nodes, prefix + leaf_b)
            if not a or not b:
                continue
            steps_a, dec_a, rec_a = a
            steps_b, dec_b, rec_b = b
            key = f"{pmcid}|{fig_id}|{nid}"

            vig_a = llm.ask(f"{key}|genA", ORIGINAL_PROMPT.format(
                constraints=constraint_lines(dec_a), rec=rec_a), 600)
            if not vig_a or len(vig_a) < 80:
                failed += 1
                continue

            # Decisions after the changed node that the second vignette must also satisfy
            new_dec = [(c, o, w) for c, o, w in dec_b
                       if not any(c == c2 and w == w2 for c2, _, w2 in dec_a)]
            extra = ""
            if len(new_dec) > 1:
                extra = "Also make sure:\n" + constraint_lines(
                    [d for d in new_dec if d[0] != node.get("text", "")])
            vig_b = llm.ask(f"{key}|genB", PERTURB_PROMPT.format(
                vignette=vig_a, criterion=node.get("text", ""),
                newlabel=branches[1]["label"], oldlabel=branches[0]["label"],
                extra=extra), 600)
            if not vig_b or len(vig_b) < 80:
                failed += 1
                continue

            if not (verified(f"{key}|A", vig_a, dec_a) and verified(f"{key}|B", vig_b, dec_b)):
                failed += 1
                continue
            for branch, vig, steps, rec in (("A", vig_a, steps_a, rec_a),
                                            ("B", vig_b, steps_b, rec_b)):
                meta = {"pmcid": pmcid, "fig_id": fig_id, "node": nid, "branch": branch}
                out.write(json.dumps(make_example(tree, vig, steps, rec, meta)) + "\n")
            kept += 1
        print(f"{pmcid} {fig_id}: {kept} pairs kept, {failed} failed")
    out.close()


if __name__ == "__main__":
    main()
