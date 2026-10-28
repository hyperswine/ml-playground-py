# Qwen3.5 internals and our remaining data/recipe gaps

Updated: 2026-10-28T13:46:54+11:00 (Australia/Sydney). Status: source/configuration audit and bounded M1 inference probe complete.
No training, model promotion, paid compute or final-holdout evaluation was performed.

## Finding

Qwen3.5 has meaningful architectural differences from an ordinary all-attention
GPT-style decoder, and our custom 557M is also simpler than many modern dense
transformers. The evidence nevertheless points first to **curriculum, response
contracts, adaptation coverage and inference protocol** for our current Sol failures.
It does not isolate an architectural defect or establish that data alone explains
the gap between Qwen and the custom model.

The inspected model is the actual round-4 MLX 8-bit checkpoint, configuration SHA-256
`ddfcb7c9c8e637f1bee327053edf61575f754fce61142c5125a3b65bb3acce8a`.
Installed runtime: mlx-lm 0.32.2 and MLX 0.32.4. Model configuration, actual adapter
tensor names/shapes, reconciled parameter counts, template parity and runtime source
hashes are saved in [inspection.json](data/qos_ai_benchmark/qwen_internals_20261028/inspection.json).
This is a forward-pass and training-interface audit, not causal circuit analysis.

## What is different inside this exact model

| Component | Our custom 557M | Qwen3.5-2B text backbone |
| --- | --- | --- |
| Blocks / hidden width | 24 / 1,280 | 24 / 2,048 |
| Sequence mixing | 24 causal full self-attention blocks | 18 Gated DeltaNet blocks and six gated full-attention blocks |
| Full-attention heads | 20 query/key/value heads, dimension 64 | Eight query heads, two shared KV heads, dimension 256 |
| Normalization | LayerNorm | RMSNorm, plus normalized queries/keys and gated normalization |
| Feed-forward | Two linear matrices with GELU, intermediate 5,120 | Three linear matrices with SiLU gating (SwiGLU), intermediate 6,144 |
| Position handling | Learned absolute embeddings, context 512 | Partial rotary position encoding; configuration supports native context 262,144 |
| Vocabulary / output | 65,536-token subset, tied input/output weights | 248,320 entries, tied input/output weights |
| Vision | None | Original checkpoint is multimodal; the inspected MLX conversion contains no vision tensors |
| Multi-token prediction | None | Present in the family/configuration; converted weights contain no MTP tensors |

Custom source: [qos_ai.py](../src/ml_playground/model/qos_ai.py),
[557M configuration](../configs/qos_ai_557m.json). Qwen sources:
[official pinned model card](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md),
[MLX Qwen implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/qwen3_5.py),
[full attention and MLP implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/qwen3_next.py).

The layout repeats one cycle six times:
`DeltaNet + MLP → DeltaNet + MLP → DeltaNet + MLP → full attention + MLP`.
The **2B checkpoint is dense**. Generic family descriptions mention mixture-of-experts,
but our configuration has no expert routing and its runtime constructs dense MLPs.
RMSNorm, SwiGLU, grouped-query attention and RoPE are common modern design choices;
the hybrid recurrent/full-attention stack is the more distinctive change here.

### Gated DeltaNet is learned state management

A conventional attention layer compares a query with cached prior keys and forms
a weighted sum of values. A DeltaNet layer instead maintains a matrix-valued state.
In simplified single-head notation, matching the installed recurrent implementation:

```text
D = decay * previous_state
error = value - D @ key
state = D + beta * outer(error, key)
output = state @ query
```

The learned gates control decay and writing; a short causal convolution mixes
nearby tokens before this update, and an output gate controls what leaves the block.
This changes transient inference state, not the model's trained weights. It is not
a built-in arithmetic interpreter or a verifier of generated code.

The recurrent state is fixed in size per DeltaNet head; the full-attention blocks
still keep token-dependent KV caches. Therefore the whole hybrid must not be called
fully linear-memory or fully linear-time at every layer. The hybrid retains explicit
attention periodically instead of requiring compressed state to do all retrieval.
The original paper discusses finite-state retrieval tradeoffs, but our failures on
short prompts do not establish memory collisions as their cause.
[Recurrent source](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/gated_delta.py),
[Gated DeltaNet paper](https://arxiv.org/abs/2412.06464).

### Parameter counts need interpretation

The inspected text-only model has **1,881,824,512 logical parameters** under the
runtime's quantization-aware count. The tied embedding/output matrix accounts for
**508,559,360 parameters (27.0%)**. The remaining transformer blocks account for
1,373,263,104, plus 2,048 final-norm parameters. These reconcile exactly.
The original multimodal checkpoint's total count includes components that our
text-only runtime drops. A nominal “2B versus 557M” comparison is not an allocation
of every additional parameter to arithmetic or reasoning.

MTP was a training feature advertised for this family and can support specialized
speculative serving. Our converted weights and ordinary token loop do not invoke
MTP or execute several candidate tokens as a hidden reasoning procedure. The
vision tower is likewise absent from this local conversion. Neither explains a
secret advantage in our current text-only loop.

## What our fine-tuning actually changes

The earlier round-4 reference was a full text-model fine-tune. The recent M1 pilots
are different: QLoRA on **only blocks 20–23**, one final hybrid cycle. Inspecting the
saved tensors confirms exactly **1,401,600 adapter parameters**, about **0.0745%**
of the text backbone. That fraction measures adapter size, not how many internal
features those updates can influence.

| Adapted block | Kind | Adapter parameters |
| --- | --- | ---: |
| 20 | Gated DeltaNet + dense MLP | 360,704 |
| 21 | Gated DeltaNet + dense MLP | 360,704 |
| 22 | Gated DeltaNet + dense MLP | 360,704 |
| 23 | Gated full attention + dense MLP | 319,488 |

The adapter includes packed QKV, gate projections and output projection in DeltaNet,
Q/K/V/output projections in full attention, and MLP gate/up/down projections.
It **does reach the hybrid modules**. The suspected mistake of adapting only familiar
attention names and missing DeltaNet does not describe these runs. Convolution,
base decay parameters, normalization and token embeddings remain frozen under this
adapter recipe.

Only 120 unique examples were visited by the selected first seed, with 8,208
supervised tokens and 60 updates; the repeat visited 116, with 8,192 tokens and 58
updates. They were partial passes through 528 prepared training examples. The
mixed schedule was half replay/new by rows but 71.6% new by target tokens. Those
facts make “the verified dataset did not work” too broad a conclusion: the experiment
tested a small, specific amount of a limited adapter recipe.
[Driver](../scripts/mac/mlx_repair.py),
[controlled-run evidence](2026-10-28-QWEN-2B-CONTROLLED.md),
[MLX adapter selection](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/tuner/utils.py).

## Thinking mode and the small inference check

The historical prefix explicitly inserts an empty closed thinking section. It
matches the installed official non-thinking template for all 48 fresh development
questions and all 120 rows in the selected training schedule.
[Template parity audit](data/qos_ai_benchmark/qwen_internals_20261028/template_parity.json). This is **a supported mode, not a malformed prompt**; the 2B model card
says non-thinking is its default. Enabling thinking uses a different open prefix.
Writing `/think` in plain text is not the documented switch for Qwen3.5.
[Official template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/chat_template.jinja),
[model guidance](https://huggingface.co/Qwen/Qwen3.5-2B#thinking-mode).

Four existing development questions were fixed before generation: two output
calculations, one generation task and one repair with an operation-order trap.
Two saved models were compared: public `Qwen3.5-2B` before our domain fine-tune,
and round 4. The first is an **untuned instruction checkpoint**, not the separate
`Qwen3.5-2B-Base` foundation checkpoint. No model download was needed.

All conditions used greedy decoding, with no training or benchmark changes:
original per-question budget (96/320 tokens), non-thinking with 512 tokens, and
thinking with 512 tokens. The official serving examples use sampling; this probe
controls decoding across conditions and is not a replication of vendor benchmarks.

| Model and condition | Complete final answers /4 | Finished /4 | Relevant observation |
| --- | ---: | ---: | --- |
| Untuned instruction, original budget | 0 | 2 | Limited Sol/response-contract performance |
| Untuned instruction, 512 non-thinking | 1 | 3 | More budget helped one item |
| Untuned instruction, 512 thinking | 0 | 0 | All four reached the cap without closing thinking |
| Round 4, original budget | 1 | 4 | All stopped naturally |
| Round 4, 512 non-thinking | 1 | 4 | Same generated answers/token count as the original budget |
| Round 4, 512 thinking | 0 | 4 | All stopped without closing the thinking section |

For round 4's thinking-prefix outputs, the raw text contains two correct final-value
claims, but no well-formed final section. Scoring raw text as a separate diagnostic
gives 2/4, while protocol-compliant final answers remain 0/4. Some displayed
intermediate calculations are still inconsistent; a correct final number is not
an audited explanation. The results suggest response-mode conditioning has changed
under the no-thinking-focused domain SFT. They do not prove that all native thinking
capability was lost.

The untuned model's cap hits also do not distinguish productive longer reasoning
from looping. The model card independently notes that this 2B variant can loop in
thinking mode. Four selected questions and a 512-token cap cannot estimate broad
accuracy, establish a model capacity limit, or prove that thinking never helps.
A proper mode ablation needs adequate separate reasoning/final budgets, protocol
validation and multiple seeds. [Saved probe and grades](data/qos_ai_benchmark/qwen_internals_20261028/probe_scores.json).

## Is this mostly a data issue?

**For the current task, data/recipe/protocol issues are better-supported immediate
suspects than a defect in the hybrid architecture.** “Just add more examples” is
not an adequate diagnosis either. Evidence and limits:

- Verified targets already demonstrate the required Sol operations, but the model
  still generates a repair and says “No change needed,” omits requested stdout, or
  generalizes the wrong operation order. These point to instruction-contract and
  composition transfer gaps; short prompts do not stress the advertised long context.
- Reordering verified targets and changing replay changed results materially while
  architecture stayed fixed. Those older runs combined changes, so they do not
  isolate a single causal factor.
- Correctly written code can have a wrong stated output. Running it addresses that
  failure directly; neither normal attention nor DeltaNet supplies exact execution.
- The recent conservative update reproduced 24/36 working programs versus 21/36,
  but complete answers were 6/48 and 5/48 versus 4/48. A small amount of training
  changed behavior without changing architecture; it did not establish reliability.
- The last-four-layer rank-8 recipe leaves most model weights frozen and visits
  only a minority of the prepared pool. It cannot test the best possible curriculum
  or the model's full adaptation capacity.
- Source/weight availability does not expose the complete original 2B corpus,
  ordering, filtering and post-training recipe in reproducible form in the official
  materials checked. We cannot apportion the custom-model/Qwen gap numerically to
  architecture, training data or optimizer from these experiments.

The comparison with our custom model is particularly confounded: it changes
architecture, text-backbone size, tokenizer, context training, foundation exposure
and post-training. Some modern architecture choices are worth testing in a new
custom lineage, but the current checkpoints are not an architecture ablation.

## Best next discriminating experiments

1. Keep the current reference and frozen scores. Improve complete-answer and repair
   decision teaching using training-only counterexamples and varied question wording.
2. Predeclare a mode/protocol ablation: no-thinking versus thinking, enough budget
   for each, correct final-section extraction and the same scoring. Do not silently
   change the historical benchmark settings or assume more tokens will help.
3. With the same verified rows and supervised-token budget, compare adapter coverage
   (last four blocks versus more blocks) and record update counts and actual examples
   visited. Layer coverage can be measured without first enlarging the base model.
4. To investigate the hybrid architecture, compare an appropriately sized conventional
   attention baseline under the same curriculum and budget. Different upstream
   pretraining still confounds that result; a true architecture experiment requires
   matched custom pretraining and careful parameter/compute accounting.
5. For the custom line, test RoPE, RMSNorm and gated MLP changes independently before
   copying the entire hybrid stack. They require a new controlled model lineage;
   they are not drop-in changes to preserved checkpoints.

## More internal visibility is available

Official Qwen-Scope releases include sparse autoencoders for the residual streams
of all 24 layers of **Qwen3.5-2B-Base**, with 32,768 learned features. They can support
activation inspection and intervention. They are trained against a different
checkpoint from our post-trained, full-SFT round-4 model. Feature interpretations
or steering directions must be checked for compatibility and causal effect; they
are not a turnkey explanation of our repair errors. No SAE weights were downloaded
or activation intervention performed in this audit.
[Official SAE model card](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100),
[Qwen-Scope paper](https://arxiv.org/abs/2605.11887).

## Reproduction and evidence

`scripts/mac/qwen_internals.py` is a read-only inspector/probe for the existing
M1 MLX environment. It checks parameter reconciliation and template parity and
records runtime source hashes. Its output path must be new. The scorer used the
existing pinned compiler and fresh-development grader, preserving raw answers and
separate final-output grades. The four-question probe is exploratory evidence only.

The M1 directory is `~/qos_ai_eval/experiments/qwen_internals_20261028`.
Saved evidence is under `docs/data/qos_ai_benchmark/qwen_internals_20261028/`.
The original inspection had a root-leaf counting issue in the runtime helper;
the corrected inspection explicitly counts the tied embedding/final norm and
asserts that all components sum to the runtime's total. The original remote record
is preserved as `inspection.json`, corrected there as `inspection_v2.json`; this
repository's `inspection.json` is the corrected record.

Validation: the read-only inspection ran successfully on the M1; parameter totals
reconcile, all 168 audited prompts match the official non-thinking template, the
24 bounded generations completed, and the existing executable grader scored the
four-question comparisons. Five repository-layout tests passed. No training was run.
