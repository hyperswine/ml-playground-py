# Precedents for training the custom 557M on a few billion tokens

Updated: 2026-10-24T20:34:50+11:00 (Australia/Sydney). Kind: source review and experiment planning.
No training, checkpoint download, model promotion or cloud operation was performed for this review.

## Finding

There are credible published precedents in this range. The closest usable references are
Pythia-410M intermediate checkpoints, Cerebras-GPT-256M/590M, and DataComp-LM's
400M research scale. They establish that this is a useful scale for controlled research;
they do not establish that our proposed run will match a pretrained Qwen assistant.

The current custom lineage has approximately 0.90B general pretraining exposures, before
later domain/SFT stages. The proposed US$50 run was estimated to add about 4-5B exposures,
so the intended comparison is approximately **4.9-5.9B cumulative general pretraining tokens**.
That estimate assumes the previously measured 512-token setup and is not a new throughput
measurement or a guarantee of future GPU pricing. See the
[Stage 2 record](2026-10-20-STAGE2-557M.md) for the existing exposure.

## Closest precedents

| Reference | Parameters | Training exposure | Why it matters |
| --- | ---: | ---: | --- |
| [Pythia-410M, step2000 / step3000](https://github.com/EleutherAI/pythia#using-pythia) | About 405M actual, named 410M | **4.194B / 6.291B**, calculated | Public snapshots bracket our proposed cumulative exposure. |
| [Cerebras-GPT-256M](https://arxiv.org/pdf/2304.03208#page=3) | 256M | **5.1B**, reported | Completed from-scratch run at almost the same token budget, with a smaller model. |
| [Cerebras-GPT-590M](https://arxiv.org/pdf/2304.03208#page=3) | 590M | **11.8B**, reported | Very close parameter scale, with roughly twice our proposed total exposure. |
| [DataComp-LM 400M-1x](https://arxiv.org/html/2406.11794v3#S3) | 412M | **8.2B**, reported | A standardized small-model track for measuring data-curation quality. |

### Pythia: the closest exposure-matched checkpoint comparison

EleutherAI documents 2,097,152 tokens per step and checkpoints every 1,000 steps.
Thus step2000 is 4,194,304,000 tokens and step3000 is 6,291,456,000 tokens.
These are early snapshots of a roughly 300B-token run, **not completed 4-6B-token
training schedules**. Learning-rate schedules and initialization/data differ from ours.
Use the current suite rather than the older `-v0` variants.
Sources: [official repository](https://github.com/EleutherAI/pythia),
[parameter-count table](https://huggingface.co/EleutherAI/pythia-410m).

### Cerebras-GPT: a completed run close to our model size

Table 1 of the paper directly reports the 256M/5.1B and 590M/11.8B combinations.
These are autoregressive base models trained on the Pile, with 2,048-token context;
our measured run used 512. They are useful architecture/training references, not
ready-made Sol assistants. The paper supplies a recipe and downstream evaluations.
Source: [Cerebras-GPT paper, Table 1 and Section 2](https://arxiv.org/pdf/2304.03208).

### DataComp-LM: evidence for focusing on the data

Table 1 specifies 412M parameters and 8.2B training tokens for 400M-1x. It is a
benchmark scale with a fixed training recipe, not one universal checkpoint score.
The study tests filtering and mixing, and finds strong correlation between data-method
rankings at small and larger scales. Its relevance is that useful data experiments do
not require trillion-token runs. Source: [DCLM, Section 3 and Table 1](https://arxiv.org/html/2406.11794v3#S3).

## How to interpret the token budget

At 556,526,080 parameters, 4.9-5.9B cumulative tokens corresponds to roughly
8.8-10.6 tokens per parameter. The often-used 20-token-per-parameter reference
would be approximately 11.1B tokens here. The Chinchilla study investigates allocation
of a fixed training-compute budget; this is **not a minimum for learning, a hard
quality ceiling, or proof that 11.1B is optimal for our corpus and architecture**.
Source: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

Inference for this project: a 4-5B-token continuation is a defensible research stage,
with room for further training if independent measures improve. A smaller matched-compute
control would help test whether this budget is better spent on model size or exposure,
but the requested priority remains developing the existing custom model.

## What to record in a future experiment

1. Keep the original pretraining checkpoint and optimizer state. Explicitly distinguish
   continued pretraining from a fresh random-initialization run and from SFT.
2. Prepare fresh, diverse, deduplicated English/code data. Record both processed token
   exposures and unique corpus tokens; replaying an old corpus is not new data.
3. Save checkpoints around 1B additional-token intervals, with corpus hashes, schedule,
   accepted updates, runtime and actual spend.
4. Compare general English/code loss on unchanged held-out text. Across tokenizers,
   use a normalization such as bits per UTF-8 byte rather than raw token perplexity.
5. Apply the same bounded Sol adaptation recipe to selected pretraining checkpoints.
   Report executable coding, input preservation, outputs and repair scores separately
   from prose/lexical matching. Keep a fresh reserved evaluation untouched by selection.
6. Optionally evaluate the two early Pythia checkpoints and Cerebras-GPT-590M on the
   same general-language tests. Pin exact revisions before downloading. They require
   matched Sol adaptation before Sol scores can support a fair capability comparison.

No benchmark scores for these external models were produced in this review. Their
published results should not be substituted for measurements on our own tasks.

## References excluded from direct budget comparisons

The fine-tuned Qwen reference inherits extensive pretraining; its small SFT token count
is not its total learning budget. Likewise, Pythia's final checkpoint is approximately
300B tokens, not 4-6B. TinyStories is useful evidence about narrow synthetic curricula,
but coherent children's stories are a different objective from general English, code
and Sol. See the [TinyStories paper](https://arxiv.org/abs/2305.07759) for that scope.
