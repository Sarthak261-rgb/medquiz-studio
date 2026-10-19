# Results

Run `gen-v1`, data fingerprint `a8be3c5ed22b01d3`. Recomputed by `medquiz report` from the files in `runs/` and `data/reviews/`. Nothing here is typed in by hand, and CI fails if a rerun gives a different file.

| | |
|---|---|
| Generator | `qwen2.5:7b-instruct-q4_K_M` (digest 845dbda0ea48), temperature 0.7 |
| Judge | `llama3.1:8b-instruct-q4_K_M` (digest 46e0c10c039e), temperature 0.0 |
| NLI | `cross-encoder/nli-deberta-v3-base` |
| Embeddings | `BAAI/bge-small-en-v1.5` + FAISS |
| Source | `openstax/osbooks-anatomy-physiology@5ae32b3f4bc2`, 151 chunks |
| Hardware | NVIDIA GeForce RTX 4050 Laptop GPU |

## Human review

Not reviewed yet (0 of 151). Run `medquiz review` to start.

## Automatic checks

Auto-rejected (at least one hard check failed): **84.8% (128/151), 95% CI 78.2 to 89.6%** (128 fail, 23 pass).

| Primary failure | Share of generated questions |
|---|---|
| Distractors obviously wrong | 70.2% (106/151) |
| Marked answer is wrong | 7.3% (11/151) |
| More than one correct option | 4.6% (7/151) |
| Answer not supported by the source | 2.6% (4/151) |

| Check | Kind | Failed | Applied | Fail rate |
|---|---|---|---|---|
| `absolute_terms` | soft | 7 | 151 | 4.6% |
| `book_overlap` | soft | 2 | 151 | 1.3% |
| `citation_retrieval` | soft | 13 | 151 | 8.6% |
| `closed_book_solve` | info | 37 | 151 | 24.5% |
| `duplicate` | hard | 0 | 151 | 0.0% |
| `evidence_in_source` | hard | 0 | 151 | 0.0% |
| `judge_answer_grounded` | hard | 4 | 151 | 2.6% |
| `judge_blind_solve` | hard | 14 | 151 | 9.3% |
| `judge_distractor_quality` | hard | 119 | 151 | 78.8% |
| `judge_single_correct` | hard | 13 | 151 | 8.6% |
| `judge_stem_clear` | soft | 0 | 151 | 0.0% |
| `key_terms_in_source` | soft | 10 | 140 | 7.1% |
| `length_cue` | soft | 5 | 151 | 3.3% |
| `negative_stem` | soft | 4 | 151 | 2.6% |
| `nli_answer_entailed` | soft | 58 | 151 | 38.4% |
| `nli_distractors_not_entailed` | soft | 43 | 116 | 37.1% |
| `no_all_none_of_above` | hard | 1 | 151 | 0.7% |
| `numeric_grounding` | soft | 2 | 22 | 9.1% |
| `options_distinct` | hard | 0 | 151 | 0.0% |
| `options_only_solve` | info | 70 | 151 | 46.4% |
| `self_answer_grounded` | info | 12 | 151 | 8.0% |
| `self_blind_solve` | info | 13 | 151 | 8.6% |
| `self_distractor_quality` | info | 131 | 151 | 86.8% |
| `self_single_correct` | info | 18 | 151 | 11.9% |
| `self_stem_clear` | info | 1 | 151 | 0.7% |
| `statement_consistent` | soft | 12 | 151 | 8.0% |
| `stem_leak` | hard | 0 | 151 | 0.0% |
| `text_reference` | soft | 39 | 151 | 25.8% |

Closed-book solve (judge answers with no passage): 75.5% (114/151) correct. A high number means many questions test general knowledge rather than the passage.

Options only (judge sees the four options, no stem and no passage): 53.6% (81/151), 95% CI 45.7 to 61.4% correct, above the 25% a blind guess would get. Anything well above chance means the options themselves leak the answer.

## Judge vs judge: does the generator grade itself more kindly?

Same prompts, passages and decision rule; only the judge model changes: `llama3.1:8b-instruct-q4_K_M` (independent) against `qwen2.5:7b-instruct-q4_K_M` (the generator itself).

| Check | Independent judge passes | Self-judge passes | Agreement | Kappa |
|---|---|---|---|---|
| blind solve | 90.7% (137/151) | 91.4% (138/151) | 0.93 | 0.55 |
| answer grounded | 97.4% (147/151) | 92.0% (139/151) | 0.92 | 0.22 |
| single correct | 91.4% (138/151) | 88.1% (133/151) | 0.90 | 0.46 |
| distractor quality | 21.2% (32/151) | 13.2% (20/151) | 0.75 | 0.13 |
| stem clear | 100.0% (151/151) | 99.3% (150/151) | 0.99 | 0.00 |

Auto-reject rate with the independent judge: 84.8% (128/151), 95% CI 78.2 to 89.6%. With the generator as its own judge: 89.4% (135/151), 95% CI 83.5 to 93.4%. Kappa between the two verdicts: 0.09.

## Judge vs human

Needs human review first.

## What each hard check costs

The configured gate was fixed before any human review. This table only shows what happens when one hard check is dropped, or when only the rules or only the judge are used.

| Gate | Auto-rejected |
|---|---|
| configured gate | 84.8% (128/151) |
| without options_distinct | 84.8% (128/151) |
| without no_all_none_of_above | 84.8% (128/151) |
| without stem_leak | 84.8% (128/151) |
| without evidence_in_source | 84.8% (128/151) |
| without duplicate | 84.8% (128/151) |
| without judge_blind_solve | 84.1% (127/151) |
| without judge_answer_grounded | 84.1% (127/151) |
| without judge_single_correct | 82.8% (125/151) |
| without judge_distractor_quality | 14.6% (22/151) |
| rules and retrieval only | 0.7% (1/151) |
| LLM judge only | 84.8% (128/151) |

### Is "implausible" just "not in the passage"?

Distractor plausibility labels from the independent judge, split by its verdict on the same option.

| Verdict | low | medium | high | Share low |
|---|---|---|---|---|
| contradicted | 42 | 10 | 27 | 53% |
| not_in_passage | 292 | 49 | 16 | 82% |
| supported | 0 | 2 | 13 | 0% |

## Judge stress test

Questions that passed every check, broken on purpose in one known way. The label is known by construction, so this measures the checks, not the generator.

| Mutation | n | Auto-rejected | Caught by |
|---|---|---|---|
| swap_key | 23 | 100.0% (23/23) | `judge_blind_solve` 100.0% (23/23); `judge_answer_grounded` 17.4% (4/23); `nli_answer_entailed` 61.9% (13/21) |
| wrong_passage | 23 | 100.0% (23/23) | `judge_answer_grounded` 69.6% (16/23); `nli_answer_entailed` 95.7% (22/23); `evidence_in_source` 100.0% (23/23) |
| near_miss_passage | 23 | 100.0% (23/23) | `judge_answer_grounded` 39.1% (9/23); `nli_answer_entailed` 91.3% (21/23); `evidence_in_source` 100.0% (23/23) |
| second_correct | 23 | 65.2% (15/23) | `judge_single_correct` 30.4% (7/23); `judge_blind_solve` 26.1% (6/23); `options_distinct` 4.3% (1/23); `nli_distractors_not_entailed` 47.6% (10/21) |
| absurd_distractors | 23 | 100.0% (23/23) | `judge_distractor_quality` 100.0% (23/23) |

Reversing the option order (nothing else changed) flipped the judge's blind answer in 0.0% (0/23) of questions and flipped the overall verdict in 52.2% (12/23).

## Control: the textbook's own questions

The 57 review MCQs written by the OpenStax authors went through the same checks, each with the best-matching passage from its own section.

- Auto-rejected: 89.5% (51/57), 95% CI 78.9 to 95.1%
- Judge blind-solve accuracy: 75.4% (43/57), 95% CI 62.9 to 84.8%
- Judge closed-book accuracy: 66.7% (38/57)
- Auto-rejected among NOT/EXCEPT stems: 33.3% (1/3)

| Primary failure | Share |
|---|---|
| Distractors obviously wrong | 61.4% (35/57) |
| Marked answer is wrong | 24.6% (14/57) |
| Answer not supported by the source | 3.5% (2/57) |

## Retrieval

Does a book question retrieve a chunk from its own section? n=57.

| Query | R@1 | R@3 | R@5 |
|---|---|---|---|
| stem only | 0.81 | 0.93 | 0.96 |
| stem plus answer | 0.91 | 0.96 | 0.98 |

## Recall vs reasoning

Generator was asked for one type; the judge (and the reviewer, if set) labelled what they saw.

| Asked -> judged | Count |
|---|---|
| reasoning -> reasoning | 74 |
| reasoning -> recall | 1 |
| recall -> reasoning | 41 |
| recall -> recall | 35 |

Key position as generated: {'A': 50, 'B': 54, 'C': 29, 'D': 18}. After the seeded shuffle: {'A': 32, 'B': 37, 'C': 46, 'D': 36}.

## Images

12 subjects x 4 seeds per arm, same prompts and seeds for both arms. The LoRA was trained on 56 OpenStax anatomical drawings (52 figures excluded: third-party credits, photos, charts, tables). Every image carries a visible SYNTHETIC banner and EXIF metadata.

| | SD 1.5 | SD 1.5 + OpenStax LoRA |
|---|---|---|
| Images | 48 | 48 |
| Human-rejected | not reviewed | not reviewed |
| VLM told the subject: auto-rejected | 0.0% (0/48) | 0.0% (0/48) |
| VLM told the subject: says it shows it | 100.0% (48/48) | 100.0% (48/48) |
| VLM told the subject: garbled text | 0.0% (0/48) | 0.0% (0/48) |
| VLM blind: auto-rejected | 43.8% (21/48) | 31.2% (15/48) |
| VLM blind: names another structure | 41.7% (20/48) | 31.2% (15/48) |
| VLM blind: image has labels | 41.7% (20/48) | 72.9% (35/48) |
| VLM blind: labels not textbook terms | 6.2% (3/48) | 10.4% (5/48) |
| CLIP score (mean) | 29.94 | 30.98 |
| Seconds per image | 7.4 | 10.6 |
| GPU Wh per image | 0.1127 | 0.1270 |
| GPU Wh per accepted image | n/a | n/a |

LoRA: rank 8, 1,594,368 trainable parameters, 1200 steps on 56 figures, 1838 s and 8.00 Wh on the laptop GPU, peak 2.05 GB VRAM. Loss 0.06114 -> 0.07254 (diffusion loss is noisy and says little about anatomy, which is why the review exists).

![contact sheet](figures/images_contact_sheet.jpg)

## Cost

Everything ran locally, so the cost is time and electricity. GPU energy is read from the driver's energy counter, not estimated.

| Stage | Items | Wall time (s) | GPU energy (Wh) | Counted in cost |
|---|---|---|---|---|
| generate | 151 | 1077 | 17.56 | yes |
| check_rules | 302 | 2 | 0.16 | yes |
| check_retrieval | 151 | 54 | 0.07 | yes |
| check_nli | 151 | 62 | 0.94 | yes |
| check_judge | 151 | 2073 | 34.74 | yes |
| check_selfjudge | 151 | 1177 | 26.04 | no (experiment) |

LLM tokens: 138,243 for generation and 254,944 for checks (22% of that is the closed-book and options-only diagnostics, which never block a question). The self-judge experiment is left out of every cost figure.

| Per | Seconds | Tokens | GPU Wh |
|---|---|---|---|
| generated question | 21.6 | 2604 | 0.354 |
| auto-passed question | 142.1 | | 2.325 |

