# QOS AI project overview: what worked and what did not

Updated: 2026-10-28T13:30:49+11:00 (Australia/Sydney). Scope: recorded experiments through 2026-10-28.
This is a synthesis of saved reports and measurements, not a new training run or
an audit of current cloud resources. Historical reports and their raw data remain unchanged.

## Current position

The project has produced a working custom architecture, reproducible 168M and 557M
research lineages, compiler-verified Sol curricula, increasingly demanding evaluations,
and a useful pretrained reference model. The strongest established reference is
**Qwen3.5-2B round 4, running locally with MLX 8-bit**. Its retained comparison scores
are 341/360 on the original Sol suite, 111/111 executable programs and 18/18 on the
older multi-input suite. These do not establish broad coding reliability.

The latest harder diagnostic is the clearest warning: that reference completes
only 4/48 requests, despite supplying working programs for 21/36 code tasks. It
frequently omits or miscalculates output, makes an inconsistent repair decision,
or fails to return the complete unchanged program when asked.

A controlled repair/generation curriculum improved working programs to 24/36 in
both seeds. Complete answers reached 6/48 and 5/48, and the selected first seed
passed the original regression floors (342/360 Sol tests). The second seed missed
the predeclared complete-answer threshold. The adapter is retained as experimental;
round 4 remains the reference and the new final holdout is unused.

The user's priority remains a custom model. Qwen supplies a strong comparison for
what the curriculum can teach a pretrained model; the custom 557M remains the
platform for studying learning from random initialization. Those are distinct
research goals, and their results should be reported separately.

Sources: [latest controlled comparison](2026-10-28-QWEN-2B-CONTROLLED.md),
[Qwen 2B](2026-10-27-QWEN-2B.md), [Mac runtime comparison](2026-10-27-MAC-INFERENCE.md).

## How the project developed

| Phase | What changed | What was learned |
| --- | --- | --- |
| Early custom students, September to October 16 | Teacher-generated repository Q&A, distillation, general-English pretraining; multiple architecture sizes | Language fit and stopping improved, but fluency and lower perplexity did not ensure factual answers. |
| Verified 168M fine-tunes, October 16-19 | 65 examples, then 664; input-preserving examples, balanced repair/no-change data, broader operation coverage and two seeds | Verified teaching improved concrete behavior. Narrow follow-ups also caused forgetting and output regressions. |
| Custom 557M pilot and Stages 1-2, October 19-20 | 459M public pretraining tokens, domain adaptation, then roughly 0.90B cumulative public pretraining tokens | The pipeline worked from random weights. Domain training was necessary; extra pretraining improved losses more clearly than task scores. |
| Custom 557M Stages 3-5, October 20-21 | 60,000 executed program tasks, arithmetic checks, token-accounted replay, varied wording, verified self-samples and retrieval examples | Runnable code improved greatly. Multi-input testing exposed shortcuts; retention and specialization traded off. |
| Qwen reference, October 22-27 | Same Stage 5 mixture on pretrained 0.8B, followed by targeted data, steps-first targets, different closed-book questions and 2B | Pretraining produced the biggest practical capability jump. Answer order and retrieval-training design mattered. |
| MLX and harder 2B checks, October 27-28 | Efficient Mac inference, source-aware retrieval, complete-answer diagnostics, controlled short adapters and a second seed | Local experimentation became practical. Old high scores concealed important complete-answer failures; recent gains remain small. |

The custom architecture history includes teacher-embedding initialization in some
early experiments. The 556,526,080-parameter lineage was explicitly initialized from
random weights without pretrained embeddings. It should not be presented as having
received Qwen's foundation training.

Sources: [original project report](2026-10-16-QOS-AI-REPORT.md),
[verified 168M training](2026-10-16-FINETUNE.md), [balanced repairs](2026-10-18-BALANCED-REPAIR.md),
[557M pilot](2026-10-19-RUNPOD-557M.md), [Stage 1](2026-10-20-STAGE1-557M.md),
[Stage 2](2026-10-20-STAGE2-557M.md).

## Comparable capability milestones

These rows use the original 360-question Sol suite and related diagnostics. Training
recipes and runtimes differ, so the table is a progress record, not a controlled
parameter-scaling experiment. The 111-program score originally tests supplied
examples; multi-input scores and the newer diagnostic provide stronger checks.
The 360-question suite has informed development and is no longer an untouched test.

| Model / treatment | Sol answers /360 | Executable programs /111 | Older multi-input /18 |
| --- | ---: | ---: | ---: |
| Custom 557M Stage 3 specialist | 179 | 103 | 0 |
| Custom 557M Stage 5, with retention/wording/retrieval mixture | 142 | 98 | 8 |
| Qwen 0.8B, first Stage 5 fine-tune | 289 | 109 | 13 |
| Qwen 0.8B round 4, CUDA bf16 | 324 | 107 | 16 |
| Qwen 2B round 4, CUDA bf16 | 339 | 110 | 18 |
| Same 2B round-4 weights, MLX 8-bit | 341 | 111 | 18 |
| Latest experimental 2B mixed adapter, first seed | 342 | 111 | 18 |

CUDA-to-MLX differences are runtime/precision differences on the same round-4
weights, not additional learning. The older benchmark's multi-input suite has
correlated paraphrases and a finite set of probes. Passing it does not mean the
model can handle arbitrary programs or requirements.

Sources: [Stage 3](2026-10-20-STAGE3-SOL-PROGRAMS.md),
[Stage 5](2026-10-21-STAGE5.md), [first Qwen comparison](2026-10-22-QWEN-SFT-REFERENCE.md),
[round 4](2026-10-26-QWEN-ROUND4.md), [2B comparison](2026-10-27-QWEN-2B.md),
[MLX comparison](2026-10-27-MAC-INFERENCE.md),
[latest regression results](2026-10-28-QWEN-2B-CONTROLLED.md).

## What has worked

**1. Verified examples and balanced coverage.** The first 65-example 168M SFT
reduced raw severe looping from 49/90 to 7/90 and improved explicit output matches
from 1/33 to 7/33. Expanding to 664 examples removed those raw loops on the measured
set and reached 11/33 output matches. A balanced repair curriculum subsequently
improved guarded working generation from 14/24 to 21/24 and applied repairs from
18/48 to 27/48 on the input-focused branch. These are useful behavioral gains,
although the companion correctness measures did not all improve together.
[65 examples](2026-10-16-FINETUNE.md), [664 examples](2026-10-16-EXPANDED-FINETUNE.md),
[balanced curriculum](2026-10-18-BALANCED-REPAIR.md).

**2. Broad, executable Sol teaching.** For the custom 557M, moving from the
0.9B pilot-SFT checkpoint to Stage 3's 60,000 executed tasks increased original
Sol-suite answers from 12/360 to 179/360 and executable programs from 7/111 to
103/111. Python oracles agreed with the compiler on all 60,360 authored training
and evaluation fixtures. This is the clearest custom-lineage coding gain. The
later zero on the multi-input suite shows why the original execution score alone
was insufficient. [Stage 3](2026-10-20-STAGE3-SOL-PROGRAMS.md).

**3. Running code to check claimed outputs.** Execution augmentation raised the
Stage 3 specialist from 179/360 to 352/360 and the first tuned Qwen from 289/360 to
358/360 on the same original suite. It directly addresses the recurring pattern
of correct code plus wrong arithmetic. These are tool-assisted results, not gains
in the model's unassisted knowledge. Finite execution checks cannot establish all
input behavior. [Execution comparison](2026-10-21-NEXT-STEPS-PROGRESS.md),
[Qwen local use](2026-10-22-QWEN-LOCAL-USE.md).

**4. Better foundations and domain exposure.** On the same Stage 5 mixture,
pretrained Qwen 0.8B reached 289/360 versus the custom Stage 5 model's 142/360, with
13/18 versus 8/18 on multi-input tasks. The second 0.8B seed confirmed the large
advantage. This comparison changes foundation model, architecture and pretraining
history; it cannot attribute the difference to parameter count alone. Within the
custom lineage, giving the 557M its missing domain stage raised development lexical
matches from 9/90 to 34/90 under the reported raw settings, though that remains a
proxy. [Qwen reference](2026-10-22-QWEN-SFT-REFERENCE.md),
[second seed](2026-10-22-QWEN-LOCAL-USE.md), [domain stage](2026-10-20-STAGE1-557M.md).

**5. Answer structure and the later Qwen recipe.** Round 2 to round 3 increased
original Sol answers from 290/360 to 321/360; round 4 reached 324/360. These runs
put calculation steps before the final answer and changed closed-book replay.
The sequence supports that recipe, but round 3 changed two factors, so the exact
isolated effect of answer ordering is unmeasured. With the round-4 recipe, the
2B CUDA model reached 339/360 and solved 18/18 multi-input tasks, against the 0.8B's
324/360 and 16/18. [Round 3](2026-10-26-QWEN-ROUND3.md),
[round 4](2026-10-26-QWEN-ROUND4.md), [2B](2026-10-27-QWEN-2B.md).

**6. Retention-aware mixtures and more demanding evaluation.** Stage 4/5 replay
kept general-language loss far better than the Stage 3 specialist; Stage 5's unseen
English perplexity was 51.2 versus Stage 3's 160.7. Its multi-input score rose to
8/18 from Stage 3's zero, while its original Sol score fell to 142/360. This is a
useful balanced research variant, not a replacement for every specialist. Freezing
splits, checking several inputs and keeping output/decision/program scores separate
made those tradeoffs visible. [Stage 4 evaluation](2026-10-21-NEXT-STEPS-PROGRESS.md),
[Stage 5](2026-10-21-STAGE5.md).

**7. Targeted retrieval and a practical Mac runtime.** Long references improved
Qwen's held-out answer likelihood, but factual correctness still needed separate
review. The latest bounded source-index expansion retrieved the correct file in
the top three for 20/20 new queries and the specified evidence for 19/20. Those
queries cover five newly added libraries and do not establish full-repository
retrieval quality. MLX 8-bit made the 2B practical on the M1: the recorded decode
probe reached 26.1 tokens/s with 2.6 GB peak MLX memory, with small answer changes
relative to CUDA. [Retrieval context](2026-10-22-QWEN-LOCAL-USE.md),
[source audit](2026-10-28-QWEN-2B-CONTROLLED.md), [MLX](2026-10-27-MAC-INFERENCE.md).

## What failed or gave mixed results

| Approach / observation | Evidence | Lesson |
| --- | --- | --- |
| Unverified self-correction | Early 168M severe loops rose from 82.5% to 94.5% under the recorded 200-token comparison. Teacher repair also reduced finishing. | A teacher can reinforce errors and repetition; compiler-verified self-samples are a different treatment. |
| Assuming more pretraining is automatically better | The 168M 3B-token variant slightly improved documentation perplexity but looped more than the 1.5B-token variant after distillation. The 557M 459M→0.90B step improved loss, with mixed small task-score changes. | Track behavior, replay, SFT effects and seeds as well as next-token fit. |
| Narrow or aggressive SFT | Input-focused 168M training fixed two generation tasks but output matches fell 11/33→1/33. The 557M pilot SFT worsened documentation perplexity 23.8→37.9 after domain adaptation. | A local objective can erase another skill. |
| Treating every replay mixture as an upgrade | Stage 2 mixed SFT preserved text losses better but lost short-answer and no-change behavior. Stage 5 improved multi-input use while lowering original Sol scores. | Record target-token shares and preserve variants for different purposes. |
| Dense/hybrid retrieval as an assumed improvement | On 300 source-file queries, dense top-1 was 49.0%, BM25 56.3%; hybrid top-3 was 75.7% versus BM25's 74.0%. | Identifier-heavy source questions still favor lexical matching; the measured hybrid gain was small. |
| Same-question closed-book pairs | Round 3 restored closed-book scores while retrieval-assisted lexical matches on 51 questions fell from round 2's 28 to 16. Multiple changes were combined. | Identical targets with and without references can make references less useful; round 4 used different questions. |
| Repetition blocking as a quality metric | It removed exact loops but could corrupt legitimate repeated code and calculation steps. | Report raw decoding; stopping and truth are separate. |
| Trusting weak graders | Single-input execution accepted constants and bad unused branches; regexes accepted unsupported claims and missed valid paraphrases; trace names rejected correct renamed values. | Keep frozen historical scores, add explicit supplementary audits and test real behavior. |
| October 27 short 2B repair adapter | Corrected run fell from 341/360 to 253/360; complete fresh answers fell 4/48→3/48. A library loss-mask padding bug was found and corrected before the final run. | A valid training implementation and verified targets do not guarantee transfer. |
| October 28 controlled 2B update | Working programs improved 21/36→24/36 in both seeds, but complete answers were 6/48 and 5/48; the second seed missed the threshold. | There is modest coding headroom, but the complete-answer improvement is not established strongly enough for promotion. |

Sources: [early failures](2026-10-16-QOS-AI-REPORT.md),
[input-focused SFT](2026-10-16-INPUTS-FINETUNE.md), [Stage 1](2026-10-20-STAGE1-557M.md),
[Stage 2](2026-10-20-STAGE2-557M.md), [hybrid and round 2](2026-10-22-QWEN-ROUND2.md),
[round 3](2026-10-26-QWEN-ROUND3.md), [strict trace audit and decoding controls](2026-10-19-GENERALIZATION.md),
[October 27 pilot](2026-10-27-QWEN-2B-REPAIR.md),
[October 28 controlled runs](2026-10-28-QWEN-2B-CONTROLLED.md).

## What the scores can and cannot tell us

Perplexity, lexical patterns, runnable code, correct stdout, reusable behavior,
complete answers and supported documentation are different measurements. A falling
loss is evidence of better prediction on that probe, not proof that a generated
explanation is true. Base Qwen's zero on the original Sol suite partly reflected
syntax and answer-format mismatches, not absence of all underlying reasoning.

Benchmarks expanded over time. A score from two early generation questions cannot
be treated as equivalent to the later 24-program, 18-task multi-input or 48-question
complete-answer diagnostics. Guarded/raw decoding and CUDA/MLX precision changes
also matter. The early students retain unknown or imperfectly separable ancestral
exposure; later split fixes cannot retroactively certify old pretraining.

The historical 360-question suite and newer development sets have informed repeated
iterations. They are development evidence. The latest 24-question final holdout has
not been answered by the models. No claim of general coding competence or broad
factual accuracy follows from the high scores on older, narrower tasks.

## Where to focus next

1. Keep the custom 557M and Qwen reference as separate lines. Use the reference to
   identify curriculum/evaluation gaps and to propose examples; verify targets with
   source or executable oracles. Scaling the custom foundation is a research option,
   but these results do not establish a parameter ceiling or a guaranteed token budget.
2. Target consistent repair decisions and complete answers: unchanged valid programs,
   actual edits on broken programs, preserved supplied inputs and correct final stdout.
   Include training-only counterexamples whose wrong operation order passes one input.
3. Select curricula with token exposure and optimizer updates both visible. Repeat
   promising gains, preserve retention and reserve final evaluation until selection is
   complete. The latest 8K runs were small partial passes, not full-dataset training.
4. Keep execution and source retrieval available as measured assistance. Expand source
   coverage and factual review independently of model fitting; label tool-assisted
   output so it is not counted as unassisted model accuracy.
5. Preserve operational evidence: exact model/data/compiler/runtime identities,
   hash-verified off-machine backups and verified job shutdown. Past failures included
   idle time from a stalled tokenizer download and lost seed-17 weights despite saved
   answers. Saved answers can support an experiment comparison but cannot replace a
   checkpoint backup or prove that a cloud resource is currently stopped.

No new training, cloud lifecycle action, model promotion or final-holdout generation
was performed to prepare this overview. The pending controlled experiment was
committed as `a678748`; its earlier repair-pilot predecessor is `3abbaf6`.
