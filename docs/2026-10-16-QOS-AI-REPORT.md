# qos_ai: distilling a small FP-RISC / Sol / QOS model

Report date: 2026-10-16. Work ran from 2026-09-12 to 2026-10-16.

## Summary

The goal was a small model of our own, trained from scratch, that answers questions about FP-RISC, Sol and QOS. It
learns from a larger Qwen3.5 "teacher" model that reads the fprisc and qos repositories.

The best model is a 168M-parameter student. It was pretrained from scratch on 1.5 billion tokens of public
educational English (FineWeb-Edu), then distilled on 118,193 question/answer pairs written by Qwen3.5-4B. It is in
`qos_ai/pod_pt15/student.pt`, and is now the default for `uv run python qos_ai.py chat`.

Across the project, held-out documentation perplexity fell from 60.1 to 18.4. Since the first student that could be
measured this way, answers that finish on their own rose from 49% to 79%, and answers that loop badly fell from 86%
to 60%. The model writes fluent English and plausible-looking code, but its facts
about FP-RISC, Sol and QOS are still mostly wrong. Without help, 60% of its answers loop. With the no-repeat rule
that `chat` now applies by default, none loop exactly and 97.5% finish.

Two things helped every time: a stronger teacher, and more general English pretraining. Two ideas made generation
worse: training on the student's own answers (self-correction), and having the teacher repair those answers.

Total RunPod spend was $29.50: $27.37 of GPU time and $2.13 of storage.

## How it works

`qos_ai.py` runs a pipeline of steps, each resumable:

1. **corpus**: chunk the docs and source of the fprisc and qos repositories (1,820 chunks).
2. **ask**: the teacher reads each chunk and writes a question and an answer about it.
3. **teach**: the teacher scores every token of the docs and the Q&A, keeping its top 32 predictions per token.
4. **pretrain** (added later): train the student on general English with ordinary next-token loss.
5. **train**: the student learns the teacher's predictions (distillation) plus the real next token.
6. **chat** and **evalloops**: talk to the student, and measure how often its answers loop or finish.

Heavy steps ran on rented RunPod GPUs, mostly an RTX 4090, through the scripts in `runpod/`.

## Measurements used

- **Held-out doc perplexity**: how well a student predicts 32 windows of real documentation (16,384 tokens) that
  every run held out. Lower is better. It is comparable across all students.
- **Validation loss**: distillation loss on held-out docs and Q&A. Only roughly comparable between runs, because the
  teacher and the held-out Q&A changed.
- **Bad loops**: share of answers to 200 held-out questions in which a 4-token phrase occurs 4 or more times.
- **Finished**: share of those answers that end on their own within the token limit (400 tokens unless noted).

## Results by run

| Date | Run | Student | Doc perplexity | Bad loops | Finished | GPU cost |
|---|---|---|---|---|---|---|
| 09-25 | 0.8B teacher, 92,540 pairs | 69M | 60.1 | | | ~$1.24 |
| 09-25 | 0.8B teacher, 125,155 pairs, teacher embeddings | 193M | 42.0 | | | ~$0.87 |
| 09-25 | 4B teacher, 47,572 pairs | 201M | 39.7 | | | ~$1.26 |
| 09-26 | 4B teacher, 118,193 pairs | 209M | 26.7 | | | ~$4.20 |
| 10-14 | 500M tokens of English, then 4B pairs | 168M | 20.6 | 86% | 49% | ~$2.52 |
| 10-15 | Self-correction, 1 round | 168M | | 94.5% * | 0.5% * | ~$0.92 |
| 10-15 | Teacher repair, 2 rounds | 168M | 19.3 | 89% | 16.5% | ~$1.50 |
| 10-15 | **1.5B tokens of English, then 4B pairs** | **168M** | **18.4** | **60%** | **79%** | ~$3.80 |
| 10-16 | 3B tokens of English, then 4B pairs | 168M | 18.1 | 80% | 56.5% | ~$11.20 |

\* Self-correction was measured with a 200-token limit. On that limit its starting point was 82.5% bad loops and
29.5% finished.

Loop rates for the 1.5B and 3B students come from the same Mac with identical settings. The others were measured
on the GPU pod.

## What improved things

- **Teacher size (0.8B to 4B).** The 4B kept 87% of the pairs it wrote, against about 62% for the 0.8B on similar
  question styles. It also wrote real Sol syntax where the 0.8B described code in prose. With fewer than half the
  pairs, the 4B-taught student matched the 0.8B-taught one (39.7 against 42.0 perplexity).
- **More teacher data (47k to 118k 4B pairs).** Doc perplexity fell from 39.7 to 26.7.
- **Starting from the teacher's token embeddings.** The 193M student nearly matched the 69M student's final
  validation loss (5.62 against 5.50) after 1,000 of its 2,491 steps, and finished at 4.51. Size and data changed at the same time, so the effect is not isolated.
- **General English pretraining (the biggest single gain).** 500M tokens took perplexity from 26.7 to 20.6. Tripling
  to 1.5B tokens brought it to 18.4, cut bad loops from 86% to 60%, and raised finished answers from 49% to 79%.

## What made things worse

- **Self-correction.** The student answered 16,000 questions and trained on the teacher's scores of those answers.
  Bad loops went from 82.5% to 94.5%, and finished answers from 29.5% to 0.5%. The teacher copies loops it is shown:
  at positions where the student repeated itself, the teacher's top prediction was the repeat 94% of the time,
  against 49% elsewhere. Removing the repeated token from the teacher's choices did not help (95% and 0%).
- **Teacher repair.** Each student answer was cut before its first repeat and the 4B wrote the rest. Validation
  loss improved (3.55 to 3.44), but generation got worse: finished answers fell from 49% to 11.5% after one round
  and 16.5% after two. The repaired answers were longer than the original Q&A (median 159 tokens against 115), and
  the student learned to keep writing.
- **3B tokens of English instead of 1.5B.** Doc perplexity improved slightly (18.4 to 18.1), but bad loops rose
  from 60% to 80% and finished answers fell from 79% to 56.5%. With one run of each, run-to-run variation cannot be
  ruled out. The extra 1.5B tokens were all new text, with no earlier data mixed back in.

## No-repeat rule at answer time

`chat` now blocks any token that would repeat a 3-token phrase already in the same answer. This hides loops
rather than fixing the model.

Measured on the 1.5B-token student, 200 held-out questions, up to 400 tokens, on the M1 Mac mini:

| Rule | Bad loops | Finished | Average length |
|---|---|---|---|
| None | 60% | 79% | 210 tokens |
| No repeated 4-token phrase | 0% | 94.5% | 178 tokens |
| **No repeated 3-token phrase (now the default)** | **0%** | **97.5%** | **157 tokens** |

The no-rule result matched the earlier measurement on the M4 exactly, so the comparison is consistent.

The 3-token rule is the better setting: it finishes more answers and keeps them shorter. It cannot make the
answers correct, though. For example, asked how to convert a word to an integer in FP-RISC, the student invents a
`FP` type constructor. The rule also sometimes pushes the model into near-repeats instead, such as cycling through
"HART", "HARD", "hart", "harden", "heart". Turn it off with `--no-repeat 0`.

## Problems found and fixed along the way

- **Random tokens on Apple GPUs.** `torch.multinomial` on MPS returns zero-probability tokens in about 0.7% of
  draws. Before this was found, 39% of the Q&A written on the Mac contained random Chinese or Cyrillic text.
  Sampling now uses the Gumbel-max trick instead.
- **Generation never stopped.** Qwen3.5 ships no generation config, so it ignored its end-of-answer token until told.
- **Vendored code in the corpus.** 23% of the corpus was Espressif's ESP-IDF components. They are now excluded.
- **API key copied to a pod.** `push.sh` sent the whole `runpod/` folder, including the RunPod API key. It was
  deleted from the pod within minutes, and the script now sends only named files.
- **A 6-hour hang.** In the 3B run, a periodic sample reloaded the tokenizer, which made a network request to Hugging
  Face that never returned. The GPU sat idle for about 6 hours (about $5.30). The tokenizer is now loaded once, and
  progress checks now watch for a stalled log as well as for completion.
- **Unreliable community GPUs.** Two Community RTX 4090s were on a host whose GPU failed with "CUDA unknown
  error", and a Community RTX 5090 never booted. Later runs used Secure Cloud.

## Costs

| Item | Amount |
|---|---|
| RunPod GPU time | $27.37 |
| RunPod storage for stopped pods | $2.13 |
| Total | $29.50 |

About $6.20 of the total was avoidable: the 6-hour hang and about an hour of idle time after the 3B run finished.
All pods are terminated, so nothing is billing.

## Where the files are

| Path | Contents |
|---|---|
| `qos_ai/student.pt` | Best student (same as `qos_ai/pod_pt15/student.pt`), default for `chat` |
| `qos_ai/pod_pt15/pretrained.pt` | 1.5B-token pretrained checkpoint, before distillation |
| `qos_ai/pod_pt30/pretrained.pt` | 3B-token pretrained checkpoint, the strongest base on English |
| `qos_ai/pod_pt/qa.jsonl` | The 118,193 4B Q&A pairs |
| `qos_ai/pod_pt/vocab.pt` | The fixed 65,536-token student vocabulary |
| `qos_ai/pod/`, `qos_ai/pod_4b*/`, `qos_ai/pod_repair/` | Earlier students and data |

## Suggested next steps

1. **Re-distill the 3B-token checkpoint with different settings.** It is the best base on English and on the docs,
   so a second distillation, for example one epoch or a shorter answer style, may beat the current student. About
   $0.50 to $1 of GPU time.
2. **More English pretraining, mixing earlier data back in.** The jump from 500M to 1.5B tokens was the largest
   improvement measured. Mixing old and new data would test whether the 3B regression came from using only new text.
3. **A larger student.** At 168M parameters the model may be near what this much data can teach it.
