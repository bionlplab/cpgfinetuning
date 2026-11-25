"""Make the recommendation-only and prose versions of a data file.

Modes:
  recommendation-only-train  ask for the recommendation only, and reduce the
                             target to the recommendation and confidence
  recommendation-only        ask for the recommendation only (target unchanged,
                             so the judge still sees the full target path)
  prose                      ask for a plain-text explanation instead of JSON

  python scripts/make_prompt_variants.py recommendation-only-train data/train_decision_path.jsonl data/train_recommendation_only.jsonl
  python scripts/make_prompt_variants.py recommendation-only data/primary.jsonl data/primary_recommendation_only.jsonl
  python scripts/make_prompt_variants.py prose data/primary.jsonl data/primary_prose.jsonl
"""
import json
import sys

MARKER = "Respond with ONLY a JSON object:"

RECOMMENDATION_ONLY = (
    "Respond with ONLY a JSON object:\n"
    "{\n"
    '  "recommendation":"The final action/recommendation you reached",\n'
    '  "confidence":"high/medium/low"\n'
    "}"
)

PROSE = (
    "Work through the tree for this patient and explain your reasoning in "
    "plain text. State each decision you make and the branch you take, and "
    "end with one line of the form:\n"
    "FINAL RECOMMENDATION: <the action or recommendation you reached>"
)


def main():
    mode, src, dst = sys.argv[1:4]
    if mode not in ("recommendation-only-train", "recommendation-only", "prose"):
        sys.exit(f"unknown mode: {mode}")
    instruction = PROSE if mode == "prose" else RECOMMENDATION_ONLY
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            user = r["messages"][0]["content"]
            i = user.find(MARKER)
            if i >= 0:
                r["messages"][0]["content"] = user[:i] + instruction
                if mode == "recommendation-only-train":
                    t = json.loads(r["messages"][1]["content"])
                    r["messages"][1]["content"] = json.dumps({
                        "recommendation": t.get("recommendation", ""),
                        "confidence": t.get("confidence", "high")})
                n += 1
            fout.write(json.dumps(r) + "\n")
    print(f"{mode}: changed {n} rows, wrote {dst}")


if __name__ == "__main__":
    main()
