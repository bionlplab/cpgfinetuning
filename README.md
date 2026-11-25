# Fine-Tuning Small Language Models to Execute Clinical Guidelines

This repository contains the data, model outputs, configurations, and code for the paper "Fine-Tuning Small Language Models to Execute Clinical Guidelines".

We fine-tuned Qwen3-VL-8B-Instruct and Mistral-7B-Instruct-v0.3 to apply clinical guideline decision trees to patient vignettes. Each prompt contains a decision tree in JSON and a vignette. The model returns the decision path through the tree and a final recommendation. An LLM judge scores the path (path concordance) and the recommendation (recommendation accuracy).

## Contents

```
data/       decision trees, fine-tuning sets, and evaluation sets
configs/    LLaMA-Factory configurations for fine-tuning and inference
outputs/    model responses and judge scores for every model in the paper
scripts/    vignette generation, verification, scoring, and analysis
```

## Data

The data files use the LLaMA-Factory sharegpt format. Each row has `messages` (the prompt and the target response), `images` (empty), and `meta` (the source article, figure, and vignette). `data/dataset_info.json` registers the files with LLaMA-Factory.

| File | Rows | Content |
|---|---|---|
| `trees.jsonl` | 260 | Decision trees: 190 for fine-tuning, 30 for the primary evaluation, and 40 for the robustness benchmark |
| `train_decision_path.jsonl` | 569 | Decision-path fine-tuning set (190 trees) |
| `train_matched_pair.jsonl` | 402 | Matched-pair fine-tuning set: 201 vignette pairs from 114 trees |
| `train_recommendation_only.jsonl` | 569 | Recommendation-only fine-tuning set: the decision-path set with the recommendation as the only target |
| `primary.jsonl` | 394 | All 197 generated vignette pairs for the primary evaluation |
| `primary_pairs.json` | 116 | Pairs in the primary evaluation (232 vignettes, 22 trees, 20 articles) |
| `primary_failed_verification.json` | 14 | Generated pairs that failed strict verification |
| `robustness.jsonl` | 120 | Robustness benchmark: 120 vignettes from 40 trees |
| `*_recommendation_only.jsonl`, `*_prose.jsonl` | | The evaluation sets with the recommendation-only and prose instructions |

Each tree comes from a guideline flowchart in an open-access PubMed Central article, identified by `pmcid` and `fig_id`. Every tree in the evaluation sets passed manual review. In the fine-tuning sets, `meta.cohort` is `gold` for trees that passed manual review and `silver` for trees that passed only the automated checks.

In `primary.jsonl`, `meta.variant` is `original` or `perturbation`, `meta.node` is the decision node that differs between the two vignettes, and `meta.seed` is the generation seed. A pair key is `pmcid|fig_id|node|seed`.

## Fine-tune and run the models

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and run each command from the repository root.

```
llamafactory-cli train configs/train/qwen_decision_path.yaml
llamafactory-cli train configs/predict/qwen_decision_path_primary.yaml
```

Fine-tuning writes a LoRA adapter to `saves/`. Inference writes `generated_predictions.jsonl` to `outputs/<model>/<evaluation set>/`. Every run uses seed 0, bf16 precision, and greedy decoding. The `qwen_base`, `qwen_prose`, and `mistral_base` configurations run the models without fine-tuning.

## Score the responses

`scripts/judge.py` sends each response to Claude Opus 4.6 (the primary judge) or GPT-5.4 (the second judge) through Azure. The judge returns a recommendation accuracy score (0 or 1) and a path concordance score (0 to 1). Set these environment variables before you run it:

- `AZURE_AI_API_KEY` and `CLAUDE_BASE_URL` for Claude Opus 4.6 on Azure AI Foundry
- `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` for GPT-5.4 on Azure OpenAI

```
python scripts/judge.py outputs/qwen_decision_path/primary/generated_predictions.jsonl
python scripts/judge.py --judge gpt-5.4 --out outputs/second_judge/qwen_decision_path_primary.judge.jsonl \
    outputs/qwen_decision_path/primary/generated_predictions.jsonl
```

The scores go to `generated_predictions.jsonl.judge.jsonl`. This repository includes the scores for every model, so the analysis does not need API access.

## Reproduce the results

The analysis scripts use only the Python standard library. Confidence intervals come from 10,000 bootstrap resamples of source articles.

Primary evaluation, Qwen3-VL-8B:

```
P=generated_predictions.jsonl
python scripts/analyze_primary.py \
    base=outputs/qwen_base/primary/$P \
    decision_path=outputs/qwen_decision_path/primary/$P \
    matched_pair=outputs/qwen_matched_pair/primary/$P \
    recommendation_only=outputs/qwen_recommendation_only/primary/$P \
    prose=outputs/qwen_prose/primary/$P
```

Primary evaluation, Mistral-7B:

```
python scripts/analyze_primary.py \
    base=outputs/mistral_base/primary/$P \
    decision_path=outputs/mistral_decision_path/primary/$P \
    matched_pair=outputs/mistral_matched_pair/primary/$P
```

Robustness benchmark: run `scripts/analyze_robustness.py` with the same arguments, with `robustness` in place of `primary`.

Perturbed vignettes that both base models answer incorrectly:

```
python scripts/analyze_primary.py --overlap qwen mistral \
    qwen=outputs/qwen_base/primary/$P mistral=outputs/mistral_base/primary/$P
```

Agreement between the two judges:

```
python scripts/analyze_judges.py \
    outputs/qwen_base/primary/$P.judge.jsonl \
    outputs/second_judge/qwen_base_primary.judge.jsonl
```

Path concordance under the second judge:

```
python scripts/analyze_primary.py \
    base=outputs/second_judge/qwen_base_primary.judge.jsonl \
    decision_path=outputs/second_judge/qwen_decision_path_primary.judge.jsonl
```

Selection of the 116 primary pairs and checks of the matched-pair fine-tuning set:

```
python scripts/select_primary_pairs.py --out primary_pairs_check.json
python scripts/check_train_pairs.py
```

## Generate new vignette pairs

The generation and verification scripts call Claude Opus 4.6 and store each reply in `cache/`. A new run produces different vignettes.

1. `scripts/generate_primary_pairs.py` writes matched pairs for the evaluation trees in `data/trees.jsonl`.
2. `scripts/select_primary_pairs.py` applies the exclusion rules and writes the list of pairs to keep.
3. `scripts/generate_train_pairs.py` writes matched pairs for the fine-tuning trees.
4. `scripts/verify_pairs.py` checks every decision of every vignette in a pair file.
5. `scripts/check_train_pairs.py` checks the structure of the matched-pair fine-tuning set.
6. `scripts/make_prompt_variants.py` makes the recommendation-only and prose versions of a data file.

## Requirements

- Python 3.9 or later
- `anthropic` and `openai` (`pip install -r requirements.txt`) for generation, verification, and scoring
- LLaMA-Factory for fine-tuning and inference
