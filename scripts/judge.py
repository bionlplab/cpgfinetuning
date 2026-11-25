"""Score model responses with an LLM judge.

For each row of a LLaMA-Factory generated_predictions.jsonl file, the judge
compares the response with the target path and recommendation in the label
and returns a recommendation accuracy score (0 or 1) and a path concordance
score (0 to 1). Scores are written next to the predictions as
<predictions>.judge.jsonl with one row {"i", "p", "o"} per response
(p = concordance, o = accuracy, -1 if the response could not be scored).
Rows already in that file are skipped.

  python scripts/judge.py outputs/qwen_base/primary/generated_predictions.jsonl
  python scripts/judge.py --judge gpt-5.4 --out outputs/second_judge/qwen_base_primary.judge.jsonl \
      outputs/qwen_base/primary/generated_predictions.jsonl
"""
import argparse
import json
import os
import re
import time

from common import first_json_object
from llm import LLM

JUDGE_PROMPT = """You are evaluating a clinical AI model's response against ground truth from a verified clinical decision tree.

GROUND TRUTH:
  Correct recommendation: {correct_action}
  Vignette type: concordant (original)
  {path_info}

MODEL'S RESPONSE:
{model_response}

Evaluate TWO things:

1. OUTCOME SCORE (0 or 1):
   Does the model's final recommendation match the ground truth?
   Score 1 if the recommendation is clinically equivalent (exact wording not required, but the core action must be the same).
   Score 0 if the recommendation is different, missing, or contradicts the ground truth.

2. PROCESS SCORE (0.0 to 1.0):
   How well does the model's reasoning follow the correct clinical logic?
   {process_criteria}

Respond with ONLY a JSON object:
{{
  "outcome_score": 0 or 1,
  "process_score": 0.0,
  "outcome_rationale": "Why the recommendation matches or doesn't",
  "process_rationale": "How well the reasoning follows the guideline logic",
  "reasoning_gaps": ["any decision points the model missed or got wrong"]
}}"""

PROCESS_CRITERIA = """For PATH-LEVEL (concordant) vignettes, evaluate the model's traversal against the correct path.

   The correct path has N = {num_steps} steps (decision nodes and process steps, excluding the final action):
{step_descriptions}

   Score the model's trace by checking each step in sequence:
   - For DECISION steps: did the model (1) identify the correct criterion AND (2) take the correct branch/edge? Both must be correct.
   - For PROCESS steps: did the model mention, acknowledge, or perform this step in its reasoning?
   - Score += 1/N for each step the model gets correct
   - IMPORTANT: Once the model makes an error (wrong branch at a decision, or skips a process step), all subsequent steps are automatically wrong — do not give credit after a divergence point
   - Score 0.0 if the model skips the traversal entirely or does not describe its reasoning path

   Process score = (number of consecutive correct steps from the start) / N"""


def build_prompt(target, response):
    traversal = target.get("traversal", []) or []
    recommendation = target.get("recommendation", "")
    path = [s.get("decision_point", "") for s in traversal] + [recommendation]
    lines = []
    for n, s in enumerate(traversal, 1):
        answer = str(s.get("answer", "")).strip()
        text = s.get("decision_point", "")[:120]
        if answer.lower() == "next":
            lines.append(f'     {n}. [PROCESS]  "{text}"')
        else:
            lines.append(f'     {n}. [DECISION] "{text}" → take edge "{answer}"')
    if not lines:
        lines = ["     Path: " + " → ".join(path)]
    return JUDGE_PROMPT.format(
        correct_action=recommendation,
        path_info="Correct path: " + " → ".join(str(p) for p in path),
        model_response=response[:32000],
        process_criteria=PROCESS_CRITERIA.format(step_descriptions="\n".join(lines),
                                                 num_steps=len(traversal) or len(path)))


def parse_scores(raw):
    for text in (raw, re.sub(r"^```json\s*", "", re.sub(r"\s*```$", "", raw))):
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass
    result = first_json_object(raw)
    if result is None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            result = json.loads(m.group()) if m else None
        except (json.JSONDecodeError, ValueError):
            result = None
    return result or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--judge", default="claude-opus-4.6", choices=["claude-opus-4.6", "gpt-5.4"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or args.predictions + ".judge.jsonl"

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["i"] for line in f if line.strip()}
    llm = LLM(args.judge)
    rows = [json.loads(line) for line in open(args.predictions) if line.strip()]
    with open(out_path, "a") as out:
        for i, r in enumerate(rows):
            if i in done:
                continue
            target = first_json_object(r.get("label") or "")
            scores = {}
            if target:
                prompt = build_prompt(target, r.get("predict") or "")
                for attempt in range(3):
                    try:
                        scores = parse_scores(llm(prompt, 1500, temperature=0))
                        break
                    except Exception:
                        time.sleep(2 * (attempt + 1))
            out.write(json.dumps({"i": i, "p": scores.get("process_score", -1),
                                  "o": scores.get("outcome_score", -1)}) + "\n")
            out.flush()

    scores = [json.loads(line) for line in open(out_path) if line.strip()]
    o = [s["o"] for s in scores if s["o"] >= 0]
    p = [s["p"] for s in scores if s["p"] >= 0]
    print(f"{args.predictions}: accuracy {100 * sum(o) / max(len(o), 1):.1f} (n={len(o)}), "
          f"concordance {100 * sum(p) / max(len(p), 1):.1f} (n={len(p)})")


if __name__ == "__main__":
    main()
