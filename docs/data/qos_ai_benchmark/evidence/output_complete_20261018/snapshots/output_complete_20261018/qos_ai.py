"""Distill Qwen3.5 0.8B into qos_ai: a small from-scratch model that only knows FP-RISC, Sol and QOS.

The idea is a teaching loop.  Qwen (the teacher) already speaks English and code, and it can read
the fprisc and qos docs and source.  qos_ai (the student) starts as random weights, like myaistuff_text.py,
but instead of only learning "which character comes next" from raw text it learns from what the
teacher *would have predicted* at every token.  That is knowledge distillation: the student's
probabilities regress towards the teacher's probabilities (a KL loss), plus the usual
next-token cross-entropy so it still commits to a real answer.

Every step writes files that the next step reads, so each can be rerun on its own:

  uv run python qos_ai.py corpus                   # 1. fprisc + qos docs and source -> qos_ai/corpus.jsonl
  uv run python qos_ai.py ask --rounds 3           # 2. teacher writes Q&A pairs about each chunk -> qos_ai/qa.jsonl
  uv run python qos_ai.py splits                   # freeze source/example splits and benchmark exclusions
  uv run python qos_ai.py lessonplan              # inspect packing/rejections without loading a teacher
  uv run python qos_ai.py teach                    # 3. teacher scores every token of docs + Q&A -> qos_ai/lessons/*.pt
  uv run python qos_ai.py train                    # 4. student learns from the lessons -> qos_ai/student.pt
  uv run python qos_ai.py chat "What is the std tier?"   # 5. talk to the student
  uv run python qos_ai.py loop --rounds 3          # 2-4 repeated: new questions each round, student keeps training

`ask` and `teach` resume where they stopped: `ask` skips pairs already in qa.jsonl, and `teach`
writes shards and only scores what is new or changed (a new round of Q&A, a rebuilt corpus).

On a CUDA machine (a rented GPU) see runpod/: the same steps, with vLLM for `ask` when available.

Quick smoke run of the whole thing (a few minutes, tiny student):
  uv run python qos_ai.py corpus
  uv run python qos_ai.py ask --limit 16
  uv run python qos_ai.py teach --limit 40
  uv run python qos_ai.py train --steps 100 --embedding-size 256 --layers 4 --heads 4 --batch-size 8
  uv run python qos_ai.py chat "What is Sol?"

Two Apple-GPU (MPS) pitfalls this script works around:
  * torch.multinomial on MPS sometimes returns tokens whose probability is exactly zero (about 0.7%
    of draws), i.e. uniformly random tokens.  Sampling therefore uses the Gumbel-max trick
    (argmax of logits plus Gumbel noise), which is an exact sample and never calls multinomial.
    Before this fix a third of the teacher's answers contained random Chinese/Cyrillic tokens.
  * the model ships no generation config, so generation only stops at <|im_end|> if told to.

Making it more coherent (in order of expected payoff):
  * more teacher text: more `ask` rounds -- every token the teacher writes is in-domain training
    data with soft targets attached; the raw corpus alone is under 1M tokens.
  * `train --teacher-embeddings`: start the student's token embeddings (and its tied output head)
    from the teacher's own embedding rows.  Requires --embedding-size 1024.
  * a small student (`--embedding-size 512 --layers 8`, ~60M parameters) gets coherent sooner on
    little data than a big one.
  * watch the validation loss printed alongside the samples: when it stops falling while the
    training loss keeps falling, the student is memorizing and needs new questions, not more steps.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qos_ai_data import (SCHEMA as SPLIT_SCHEMA, digest, file_hash, make_manifest, pack_examples,
                         read_rows, row_split, rows_fingerprint, validate_manifest)

import torch
from torch import nn
from torch.nn import functional as F


# QOS_AI_TEACHER picks a bigger teacher, e.g. Qwen/Qwen3.5-4B; every Qwen3.5 size shares one tokenizer.
TEACHER_MODEL = os.environ.get("QOS_AI_TEACHER", "Qwen/Qwen3.5-0.8B")
# --teacher-embeddings copies rows from this model: 1024 wide, same tokenizer as every teacher size.
EMBEDDING_MODEL = "Qwen/Qwen3.5-0.8B"
# qos-fpr was split into fprisc (language, compiler, std, Sol) and qos (the OS); the corpus reads both.
DEFAULT_ROOTS = [Path.home() / "Documents/GitHub/fprisc", Path.home() / "Documents/GitHub/qos"]
WORK_DIR = Path(__file__).parent / os.environ.get("QOS_AI_DIR", "qos_ai")  # separate dirs keep experiments apart
CORPUS_FILE = WORK_DIR / "corpus.jsonl"
QA_FILE = WORK_DIR / "qa.jsonl"
LESSONS_DIR = WORK_DIR / "lessons_v2"  # old window-split lessons remain untouched
SPLIT_FILE = WORK_DIR / "splits.json"
BENCHMARK_FILE = Path(__file__).parent / "qos_ai_benchmark/benchmark.jsonl"
VERIFIED_FILE = Path(__file__).parent / "qos_ai_benchmark/verified_train.jsonl"
STUDENT_FILE = WORK_DIR / "student.pt"
PRETRAIN_TOKENS = WORK_DIR / "pretrain_tokens.bin"  # general English, as raw Qwen ids (uint32)
VOCAB_FILE = WORK_DIR / "vocab.pt"                  # the fixed student vocabulary chosen by `pretokenize`
PRETRAINED_FILE = WORK_DIR / "pretrained.pt"
SELF_FILE = WORK_DIR / "self.jsonl"                 # the student's own answers, for the teacher to score
REPAIR_FILE = WORK_DIR / "repair.jsonl"             # the student's start of an answer + the teacher's continuation
SELF_REPEAT_LESSONS = 3                             # loop points learned per self-play answer before the rest is skipped
QWEN_VOCAB = 248320                                 # rows in every Qwen3.5 embedding table

# The Haskell compiler source is left out: it is a third of the repo but it is about the compiler's
# internals, not about writing FP-RISC, Sol or QOS code.
CORPUS_EXTENSIONS = {".md", ".txt", ".fpr", ".sol", ".py", ".toml"}
SOURCE_EXTENSIONS = (".fpr", ".sol", ".py")
# managed_components holds vendored Espressif ESP-IDF code, not FP-RISC; CMakeLists.txt is build config, not prose.
SKIP_DIRECTORIES = {".git", "dist-newstyle", "build", ".fpr", "models", ".qos", "dist", "node_modules", "__pycache__", "managed_components"}
SKIP_FILES = {"CMakeLists.txt"}

SYSTEM_PROMPT = (
  "You are an expert on the FP-RISC functional language, its Sol scripting profile and the QOS operating "
  "system (the fprisc and qos projects). Answer precisely and concisely."
)

# The student reuses Qwen's chat format (thinking disabled) so the teacher scores Q&A text natively.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"
NO_THINKING = "<think>\n\n</think>\n\n"
SAMPLE_PROMPT = "What is FP-RISC?"


# ----------------------------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------------------------


def choose_device(requested: str) -> torch.device:
  if requested != "auto":
    return torch.device(requested)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  with path.open(encoding="utf-8") as handle:
    return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def hub_kwargs(model: str = TEACHER_MODEL) -> dict[str, Any]:
  """Use the cached model when it is there (no network), otherwise let Hugging Face download it."""

  from huggingface_hub import try_to_load_from_cache

  # the weights index, not config.json: a tokenizer download caches the config without any weights
  return {"local_files_only": isinstance(try_to_load_from_cache(model, "model.safetensors.index.json"), str)}


def load_tokenizer() -> Any:
  from transformers import AutoTokenizer

  return AutoTokenizer.from_pretrained(TEACHER_MODEL, **hub_kwargs())


def load_teacher(device: torch.device) -> Any:
  from transformers import Qwen3_5ForConditionalGeneration

  dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
  print(f"Loading teacher {TEACHER_MODEL} on {device} ({dtype})...")
  model = Qwen3_5ForConditionalGeneration.from_pretrained(TEACHER_MODEL, dtype=dtype, **hub_kwargs())
  return model.to(device).eval()


def chat_text(question: str, answer: str | None = None) -> str:
  """The single-turn chat format the student is trained on and prompted with (Qwen's, thinking off)."""

  text = f"{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n{NO_THINKING}"
  if answer is not None:
    text += f"{answer}{IM_END}"
  return text


def gumbel_sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float = 1.0) -> torch.Tensor:
  """Temperature, top-k and top-p, then one exact sample per row via the Gumbel-max trick.

  argmax(logits + Gumbel noise) is distributed exactly like softmax(logits), and unlike
  torch.multinomial it never picks a zero-probability token on MPS.
  """

  logits = logits.float() / temperature
  if 0 < top_k < logits.size(-1):
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    logits = logits.masked_fill(logits < kth, float("-inf"))
  if top_p < 1.0:
    ordered, order = logits.sort(dim=-1, descending=True)
    probabilities = ordered.softmax(-1)
    drop = probabilities.cumsum(-1) - probabilities > top_p  # keep the smallest prefix that reaches top_p
    logits = logits.scatter(-1, order, ordered.masked_fill(drop, float("-inf")))
  gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-10, 1.0)))
  return (logits + gumbel).argmax(dim=-1)


# ----------------------------------------------------------------------------------------------
# step 1: corpus -- copy the docs and source out of fprisc and qos into chunks
# ----------------------------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int) -> list[str]:
  """Split on blank lines, packing paragraphs into chunks of roughly chunk_size characters."""

  chunks: list[str] = []
  current = ""
  for paragraph in text.split("\n\n"):
    if current and len(current) + len(paragraph) > chunk_size:
      chunks.append(current.strip())
      current = ""
    current += paragraph + "\n\n"
    while len(current) > 2 * chunk_size:  # a single huge paragraph, e.g. a long code block
      chunks.append(current[:chunk_size].strip())
      current = current[chunk_size:]
  if current.strip():
    chunks.append(current.strip())
  return chunks


def build_corpus(args: argparse.Namespace) -> None:
  rows: list[dict[str, Any]] = []
  files = 0
  for root in args.roots:
    if not root.is_dir():
      raise ValueError(f"checkout not found at {root}")
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
      if path.suffix not in CORPUS_EXTENSIONS or path.name in SKIP_FILES or SKIP_DIRECTORIES.intersection(path.relative_to(root).parts):
        continue
      try:
        text = path.read_text(encoding="utf-8")
      except (OSError, UnicodeDecodeError):
        continue
      if not text.strip() or "\x00" in text:
        continue

      relative = f"{root.name}/{path.relative_to(root).as_posix()}"  # e.g. fprisc/docs/SEMANTICS.md
      files += 1
      for index, chunk in enumerate(chunk_text(text, args.chunk_size)):
        rows.append({"path": relative, "chunk": index, "text": chunk})
  if args.limit:
    rows = rows[: args.limit]

  CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
  CORPUS_FILE.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
  characters = sum(len(row["text"]) for row in rows)
  print(f"Wrote {len(rows)} chunks from {files} files ({characters:,} characters) to {CORPUS_FILE}")


# ----------------------------------------------------------------------------------------------
# step 2: ask -- the teacher reads each chunk and writes a question and its answer in one go
# ----------------------------------------------------------------------------------------------

QUESTION_STYLES = {
  "explain": "Write one specific question a developer might ask about FP-RISC, Sol or QOS.",
  "howto": "Write one 'How do I ...' question about FP-RISC, Sol or QOS.",
  "why": "Write one 'Why ...' question about a design decision or rule in FP-RISC, Sol or QOS.",
  "compare": "Write one question asking how two related FP-RISC, Sol or QOS concepts differ.",
  "troubleshoot": "Write one question about what goes wrong, or which error appears, when something in FP-RISC, Sol or QOS is used incorrectly.",
  "code": "Write one short request for a small code example, like 'Write a Sol function that ...' or 'Show FP-RISC code that ...'.",
  "walkthrough": "Write one question asking what a specific function or definition does and how it works.",
  # the "extended" set: different angles on the same material, so later rounds are not paraphrases of earlier ones
  "concept": "Write one beginner question asking what a specific term, type or keyword used here means.",
  "usage": "Write one question asking how to call or use a specific function, type, command or option shown here, including its arguments.",
  "predict": "Write one question that shows two to five lines of code from here and asks what they return, print or do.",
  "fix": "Write one question that shows a short piece of FP-RISC or Sol code with a single realistic mistake and asks how to fix it.",
  "program": "Write one request for a complete small program of 5 to 15 lines that uses what is shown here.",
  "summary": "Write one question that names this file by its path and asks what it is for, answered in a short paragraph.",
  "options": "Write one question that names a specific task from here and asks which functions, commands or settings are available for it.",
}
# Each round takes the next `--questions-per-chunk` styles from its pool, so rounds ask different kinds of question.
STYLE_SETS = {
  "original": {"doc": ["explain", "howto", "why", "compare", "troubleshoot"], "source": ["code", "walkthrough", "explain", "howto", "troubleshoot"]},
  "extended": {"doc": ["concept", "usage", "summary", "options", "compare", "why"], "source": ["predict", "fix", "program", "usage", "concept", "summary"]},
  "all": {
    "doc": ["explain", "howto", "concept", "why", "usage", "compare", "summary", "troubleshoot", "options"],
    "source": ["code", "walkthrough", "predict", "usage", "explain", "fix", "howto", "program", "concept", "troubleshoot", "summary"],
  },
}

PAIR_PATTERN = re.compile(r"^\W*Question\W*:(.+?)\n\s*\W*Answer\W*:(.+)$", re.S)
NEXT_QUESTION = re.compile(r"\n\s*\W*Question\W*:")
# The corpus is English and code; these scripts in an answer mean the teacher drifted.
FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ؀-ۿ฀-๿぀-ヿ㐀-鿿가-힯]")
MENTIONS_SOURCE = re.compile(
  r"\b(excerpts?|snippets?|reference material|provided (code|text)|(shown|described|used|given|listed) here|this (file|section|document|passage))\b", re.I
)
PYTHON_BLOCK = re.compile(r"```\s*python", re.I)
# The teacher sometimes repeats the instruction, or points at material the student will never see.
ECHOES_INSTRUCTION = re.compile(r"^\W*write (one|a single|a) (short )?(question|request)\b", re.I)
VAGUE_REFERENCE = re.compile(
  r"\bthe (provided|given|above|following) (code|example|definition|function|file|module|text|snippet|profile|program|document)", re.I
)


def styles_for(path: str, round_index: int, count: int, style_set: str = "original") -> list[str]:
  pool = STYLE_SETS[style_set]["source" if path.endswith(SOURCE_EXTENSIONS) else "doc"]
  start = ((round_index - 1) * count) % len(pool)
  return [pool[(start + offset) % len(pool)] for offset in range(min(count, len(pool)))]


def ask_prompt(chunk: dict[str, Any], style: str) -> str:
  return (
    f"Reference material from `{chunk['path']}`:\n\n{chunk['text']}\n\n"
    f"{QUESTION_STYLES[style]} It must be answerable from the reference material. Then answer it the way "
    "official documentation would: state the facts directly in a few sentences, with a short code example "
    "when it helps. Write any code in the same language as the reference material (FP-RISC, Sol or Python), "
    "never another one. Never refer to the reference material, an excerpt or a snippet.\n\n"
    "Reply in exactly this format:\nQuestion: <the question>\nAnswer: <the answer>"
  )


def trim_unfinished(answer: str) -> str:
  """An answer cut off by the token limit is trimmed back to its last complete sentence or code block."""

  if answer.count("```") % 2:  # cut inside a code block: drop the unfinished block
    answer = answer[: answer.rfind("```")]
  answer = answer.rstrip()
  if answer.endswith((".", "```")):
    return answer
  ends = [answer.rfind(". ") + 1, answer.rfind(".\n") + 1]
  fence = answer.rfind("```\n")
  if fence >= 0:
    ends.append(fence + 3)
  return answer[: max(ends)].rstrip()


def clean_pair(question: str, answer: str, path: str = "") -> str | None:
  """Why a pair is unusable, or None if it is fine.  Applied when pairs are written and when they are read."""

  if len(question) < 12 or len(answer) < 30:
    return "too short"
  if path and not path.endswith(".py") and PYTHON_BLOCK.search(question + answer):
    return "wrong language"
  if FOREIGN_SCRIPT.search(question + answer):
    return "foreign script"
  if MENTIONS_SOURCE.search(question + answer) or VAGUE_REFERENCE.search(question + " " + answer):
    return "mentions the excerpt"
  if ECHOES_INSTRUCTION.search(question):
    return "echoes the instruction"
  return None


def parse_pair(text: str, finished: bool, path: str = "") -> tuple[str, str] | str:
  """(question, answer) from the teacher's reply, or the reason it was rejected."""

  match = PAIR_PATTERN.match(NEXT_QUESTION.split(text.strip())[0])
  if not match:
    return "unparsable"
  question, answer = (part.strip().strip("*").strip() for part in match.groups())
  if not finished:
    answer = trim_unfinished(answer)
  return clean_pair(question, answer, path) or (question, answer)


def usable_pairs() -> list[dict[str, Any]]:
  """Q&A rows about chunks still in the corpus that pass the quality filter (old rows included)."""

  corpus_paths = {row["path"] for row in read_jsonl(CORPUS_FILE)}
  return [row for row in read_jsonl(QA_FILE) if row["path"] in corpus_paths and clean_pair(row["question"], row["answer"], row["path"]) is None]


class HfGenerator:
  """Batched generation with transformers; works on CPU, MPS and CUDA."""

  def __init__(self, device: torch.device, args: argparse.Namespace):
    from transformers import LogitsProcessor, LogitsProcessorList

    self.tokenizer = load_tokenizer()
    self.model = load_teacher(device)
    self.stop_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in (IM_END, END_OF_TEXT)]

    class GumbelSampler(LogitsProcessor):
      # generate() runs greedy (argmax) over these scores, which makes the argmax an exact sample
      def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        choice = gumbel_sample(scores, args.temperature, args.top_k, args.top_p)
        return torch.full_like(scores, float("-inf")).scatter(-1, choice[:, None], 0.0)

    self.sampler = LogitsProcessorList([GumbelSampler()])

  def __call__(self, prompts: list[str], max_new_tokens: int, seed: int) -> list[tuple[str, bool]]:
    torch.manual_seed(seed)
    texts = [
      self.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
      )
      for prompt in prompts
    ]
    self.tokenizer.padding_side = "left"
    batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
    with torch.inference_mode():
      output = self.model.generate(
        **batch, max_new_tokens=max_new_tokens, do_sample=False, logits_processor=self.sampler,
        eos_token_id=self.stop_ids, pad_token_id=self.stop_ids[1],
      )
    generated = output[:, batch["input_ids"].shape[1]:]
    finished = torch.isin(generated, torch.tensor(self.stop_ids, device=generated.device)).any(dim=1).tolist()
    return list(zip(self.tokenizer.batch_decode(generated, skip_special_tokens=True), finished))


class VllmGenerator:
  """Batched generation with vLLM (CUDA only); much faster than transformers for thousands of prompts."""

  def __init__(self, args: argparse.Namespace):
    from vllm import LLM

    self.args = args
    self.llm = LLM(
      model=TEACHER_MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.gpu_memory,
      limit_mm_per_prompt={"image": 0, "video": 0}, seed=args.seed,
    )

  def __call__(self, prompts: list[str], max_new_tokens: int, seed: int) -> list[tuple[str, bool]]:
    from vllm import SamplingParams

    params = [
      SamplingParams(
        temperature=self.args.temperature, top_p=self.args.top_p, top_k=self.args.top_k,
        max_tokens=max_new_tokens, stop=["\nQuestion:", "\n**Question"], seed=seed + index,
      )
      for index in range(len(prompts))
    ]
    conversations = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}] for prompt in prompts]
    outputs = self.llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    return [(output.outputs[0].text, output.outputs[0].finish_reason == "stop") for output in outputs]

  def continue_text(self, texts: list[str], max_new_tokens: int, seed: int) -> list[tuple[str, bool]]:
    """Continue fully formatted prompts (chat template already applied, assistant turn already begun)."""

    from vllm import SamplingParams

    params = [
      SamplingParams(temperature=self.args.temperature, top_p=self.args.top_p, top_k=self.args.top_k,
                     max_tokens=max_new_tokens, stop=[IM_END], seed=seed + index)
      for index in range(len(texts))
    ]
    outputs = self.llm.generate(texts, params, use_tqdm=False)
    return [(output.outputs[0].text, output.outputs[0].finish_reason == "stop") for output in outputs]


def make_generator(args: argparse.Namespace) -> Any:
  device = choose_device(args.device)
  engine = args.engine
  if engine == "auto":
    try:
      import vllm  # noqa: F401

      engine = "vllm" if device.type == "cuda" else "hf"
    except ImportError:
      engine = "hf"
  print(f"Generating with {engine} on {device}")
  return VllmGenerator(args) if engine == "vllm" else HfGenerator(device, args)


def ask_teacher(args: argparse.Namespace) -> None:
  random.seed(args.seed)
  chunks = read_jsonl(CORPUS_FILE)
  if not chunks:
    raise ValueError(f"Run `qos_ai.py corpus` first, {CORPUS_FILE} is empty")
  if args.limit:
    chunks = random.sample(chunks, min(args.limit, len(chunks)))  # spread across files, not the first N

  done = {(row["path"], row["chunk"], row["style"], row["round"]) for row in read_jsonl(QA_FILE)}
  rounds = range(args.round, args.round + args.rounds)
  jobs = [
    (round_index, chunk, style)
    for round_index in rounds
    for chunk in chunks
    for style in styles_for(chunk["path"], round_index, args.questions_per_chunk, args.style_set)
    if (chunk["path"], chunk["chunk"], style, round_index) not in done
  ]
  print(f"{len(jobs)} question/answer pairs to generate for rounds {rounds.start}-{rounds.stop - 1} ({len(done)} already in {QA_FILE})")
  if not jobs:
    return

  generate = make_generator(args)
  rejected: dict[str, int] = {}
  kept = 0
  answer_tokens = 0
  started = time.perf_counter()
  for round_index in rounds:
    round_jobs = [job for job in jobs if job[0] == round_index]
    round_jobs.sort(key=lambda job: len(job[1]["text"]))  # similar lengths per batch = less padding
    for start in range(0, len(round_jobs), args.batch_size):
      batch = round_jobs[start: start + args.batch_size]
      seed = args.seed + 1000 * round_index + start
      replies = generate([ask_prompt(chunk, style) for _, chunk, style in batch], args.max_answer_tokens + 64, seed)

      rows = []
      for (_, chunk, style), (reply, finished) in zip(batch, replies):
        pair = parse_pair(reply, finished, chunk["path"])
        if isinstance(pair, str):
          rejected[pair] = rejected.get(pair, 0) + 1
          continue
        question, answer = pair
        rows.append({"path": chunk["path"], "chunk": chunk["chunk"], "style": style, "round": round_index, "question": question, "answer": answer})
        answer_tokens += len(answer) // 4  # rough: ~4 characters per token
      append_jsonl(QA_FILE, rows)
      kept += len(rows)

      elapsed = time.perf_counter() - started
      finished_jobs = sum(1 for job in jobs if job[0] < round_index) + start + len(batch)
      print(
        f"[round {round_index}, {finished_jobs}/{len(jobs)} prompts, {elapsed / 60:.1f} min] kept {kept}, "
        f"~{answer_tokens / max(elapsed / 60, 1e-9):.0f} answer tokens/min, rejected {rejected}  "
        f"Q: {rows[0]['question'][:80] if rows else '-'}",
        flush=True,
      )


# ----------------------------------------------------------------------------------------------
# step 3: teach -- the teacher scores every token, keeping its top-k predictions as soft targets
# ----------------------------------------------------------------------------------------------


def to_windows(stream: list[tuple[int, bool]], context_length: int) -> list[tuple[list[int], list[bool]]]:
  """Cut a (token, learn) stream into full windows.  mask[t]: the prediction at t (of token t+1) counts."""

  windows = []
  for start in range(0, len(stream), context_length):
    window = stream[start: start + context_length]
    if len(window) >= 8:
      windows.append(([token for token, _ in window], [t + 1 < len(window) and window[t + 1][1] for t in range(len(window))]))
  return windows


def split_inputs() -> tuple[list, list, list, list]:
  verified = read_rows(VERIFIED_FILE)
  if verified:
    proof_path = VERIFIED_FILE.parent / "verification.json"
    if not proof_path.exists():
      raise ValueError("Run qos_ai_bench.py references and save verification.json before using verified lessons")
    proof = json.loads(proof_path.read_text())
    passed = {r["id"] for r in proof["results"] if r["passed"]}
    if proof.get("verified_data_hash") != rows_fingerprint(verified) or any(r.get("program") and r["id"] not in passed for r in verified):
      raise ValueError("Verified example content changed or lacks passing reference execution")
  return read_jsonl(CORPUS_FILE), usable_pairs(), verified, read_rows(BENCHMARK_FILE)


def prepare_splits(args: argparse.Namespace) -> None:
  corpus, qa, verified, benchmark = split_inputs()
  if not corpus or not benchmark:
    raise ValueError("Corpus and reviewed benchmark are required before preparing splits")
  manifest = make_manifest(corpus, qa, verified, benchmark)
  if SPLIT_FILE.exists():
    old = json.loads(SPLIT_FILE.read_text())
    if old != manifest:
      raise ValueError("Frozen splits differ; use a new QOS_AI_DIR rather than replacing an experiment's manifest")
  else:
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"manifest": str(SPLIT_FILE), "hash": manifest["hash"], **manifest["counts"]}, indent=2))


def current_splits() -> dict:
  if not SPLIT_FILE.exists():
    raise ValueError("Run qos_ai.py splits before teaching/training; legacy lesson splits are not accepted")
  manifest = json.loads(SPLIT_FILE.read_text())
  validate_manifest(manifest, *split_inputs())
  return manifest


def lesson_sources(tokenizer: Any, context_length: int) -> dict[str, tuple[str, list[tuple[list[int], list[bool]]]]]:
  """Split whole sources/examples first, then pack each partition independently."""
  manifest = current_splits()
  corpus, qa, verified, _ = split_inputs()
  groups: dict[str, list] = {}
  for split in ("train", "dev", "test"):
    documents = []
    for row in corpus:
      if manifest["files"][row["path"]] == split and digest([row["path"], row["chunk"]]) not in manifest.get("excluded_chunks", []):
        text = f"# File: {row['path']}\n{row['text']}\n"
        documents.extend((t, True) for t in tokenizer.encode(text, add_special_tokens=False))
    groups[f"docs_{split}"] = to_windows(documents, context_length)

  newline = tokenizer.encode("\n", add_special_tokens=False)
  examples: dict[str, list] = {}
  rejected = []
  for row in qa + verified:
    split = row_split(row, manifest)
    if split == "exclude":
      continue
    prompt = tokenizer.encode(chat_text(row["question"]), add_special_tokens=False)
    tokens = tokenizer.encode(chat_text(row["question"], row["answer"]), add_special_tokens=False)
    example = [(t, i >= len(prompt)) for i, t in enumerate(tokens)] + [(t, False) for t in newline]
    name = f"qa_{split}_r{row.get('round', 0):03d}"
    if len(example) > context_length:
      rejected.append({"source": name, "question": row["question"], "tokens": len(example), "reason": "oversized complete example"})
      continue
    examples.setdefault(name, []).append(example)

  for kind, path in (("self", SELF_FILE), ("repair", REPAIR_FILE)):
    for row in read_jsonl(path):
      if row_split(row, manifest) != "train":
        continue
      prefix = row.get("prefix", "") if kind == "repair" else ""
      body = row["continuation"] if kind == "repair" else row["answer"]
      prompt = tokenizer.encode(chat_text(row["question"]) + prefix, add_special_tokens=False)
      body = tokenizer.encode(body + (IM_END if row["finished"] else ""), add_special_tokens=False)
      example = [(t, False) for t in prompt] + [(t, True) for t in body] + [(t, False) for t in newline]
      if len(example) > context_length:
        rejected.append({"source": kind, "question": row["question"], "tokens": len(example), "reason": "oversized complete example"})
        continue
      examples.setdefault(f"{kind}_train_r{row['round']:03d}", []).append(example)
  for name, items in examples.items():
    groups[name] = pack_examples(items, context_length)
  WORK_DIR.mkdir(parents=True, exist_ok=True)
  (WORK_DIR / "packing_rejections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejected))
  print(f"Packing rejected {len(rejected)} oversized examples; details in {WORK_DIR / 'packing_rejections.jsonl'}", flush=True)
  # Hash exact tokens/masks, teacher identity and the frozen split contract.
  return {name: (digest({"schema": SPLIT_SCHEMA, "split_hash": manifest["hash"], "teacher": TEACHER_MODEL,
                        "name": name, "windows": windows}), windows) for name, windows in groups.items()}


def lesson_plan(args: argparse.Namespace) -> None:
  sources = lesson_sources(load_tokenizer(), args.context_length)
  print(json.dumps({name: {"signature": signature, "windows": len(windows)}
                    for name, (signature, windows) in sources.items()}, indent=2))


def teach(args: argparse.Namespace) -> None:
  device = choose_device(args.device)
  tokenizer = load_tokenizer()
  sources = lesson_sources(tokenizer, args.context_length)
  split_hash = current_splits()["hash"]
  pad_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
  LESSONS_DIR.mkdir(parents=True, exist_ok=True)

  # Plan: which shards are missing or stale.  Shards of sources that no longer exist are removed.
  plan: list[tuple[Path, str, str, list[tuple[list[int], list[bool]]]]] = []
  wanted: set[Path] = set()
  for name, (signature, windows) in sources.items():
    if args.limit:
      windows = windows[: args.limit]
    for shard_index, start in enumerate(range(0, len(windows), args.shard_size)):
      path = LESSONS_DIR / f"{name}_{shard_index:04d}.pt"
      wanted.add(path)
      if path.exists():
        header = torch.load(path, weights_only=True, mmap=True)
        if (header["signature"], header["context_length"], header["top_k"], header["limit"]) == (signature, args.context_length, args.top_k, args.limit):
          continue
      plan.append((path, name, signature, windows[start: start + args.shard_size]))
  for stale in set(LESSONS_DIR.glob("*.pt")) - wanted:
    stale.unlink()

  total = sum(len(windows) for _, (_, windows) in sources.items())
  todo = sum(len(windows) for *_, windows in plan)
  print(f"{len(sources)} sources ({', '.join(sources)}), {total} windows of {args.context_length} tokens; {todo} windows to score in {len(plan)} shards")
  if not plan:
    return

  model = load_teacher(device)
  started = time.perf_counter()
  scored = 0
  for path, name, signature, windows in plan:
    count = len(windows)
    tokens = torch.full((count, args.context_length), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((count, args.context_length), dtype=torch.bool)
    for index, (sequence, mask) in enumerate(windows):
      tokens[index, : len(sequence)] = torch.tensor(sequence)
      loss_mask[index, : len(mask)] = torch.tensor(mask)

    topk_ids = torch.zeros((count, args.context_length, args.top_k), dtype=torch.int32)
    topk_logits = torch.zeros((count, args.context_length, args.top_k), dtype=torch.float16)
    for start in range(0, count, args.batch_size):
      batch = tokens[start: start + args.batch_size].to(device)
      with torch.inference_mode():
        # logits are [batch, length, 248k], so keep the batch small and reduce row by row
        logits = model(input_ids=batch, attention_mask=(batch != pad_id).long()).logits
        for row in range(len(batch)):
          values, indices = torch.topk(logits[row].float(), args.top_k, dim=-1)
          topk_ids[start + row] = indices.to(torch.int32).cpu()
          topk_logits[start + row] = values.half().cpu()
        del logits

    torch.save(
      {
        "signature": signature, "context_length": args.context_length, "top_k": args.top_k, "limit": args.limit,
        "split_schema": SPLIT_SCHEMA, "split_hash": split_hash,
        "split": name.split("_")[1],
        "is_qa": name.startswith("qa_"), "is_self": name.startswith("self_"), "is_repair": name.startswith("repair_"), "tokens": tokens.to(torch.int32), "loss_mask": loss_mask,
        "topk_ids": topk_ids, "topk_logits": topk_logits,
      },
      path,
    )
    scored += count
    rate = scored * args.context_length / (time.perf_counter() - started)
    print(f"scored {path.name}: {scored}/{todo} windows ({rate:.0f} tokens/s)", flush=True)


# ----------------------------------------------------------------------------------------------
# the student: the same decoder-only transformer as myaistuff_text.py, scaled up
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentConfig:
  vocabulary_size: int
  context_length: int = 512
  embedding_size: int = 1024
  head_count: int = 16
  layer_count: int = 16


class CausalSelfAttention(nn.Module):
  def __init__(self, config: StudentConfig):
    super().__init__()
    if config.embedding_size % config.head_count != 0:
      raise ValueError("embedding-size must be divisible by heads")
    self.head_count = config.head_count
    self.query_key_value = nn.Linear(config.embedding_size, 3 * config.embedding_size, bias=False)
    self.output = nn.Linear(config.embedding_size, config.embedding_size, bias=False)

  def forward(self, inputs: torch.Tensor, cache: dict[str, torch.Tensor] | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    batch_size, token_count, embedding_size = inputs.shape
    query, key, value = self.query_key_value(inputs).view(batch_size, token_count, 3, self.head_count, -1).permute(2, 0, 3, 1, 4)
    if cache is None:
      # training: same maths as the hand-written version in myaistuff_text.py, but fused and without
      # materializing the full attention matrix for the backward pass
      attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    else:
      # generation: keys and values of earlier tokens are kept, so each new token costs one position
      if "key" in cache:
        key = torch.cat((cache["key"], key), dim=2)
        value = torch.cat((cache["value"], value), dim=2)
      cache["key"], cache["value"] = key, value
      attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    return self.output(attended.transpose(1, 2).reshape(batch_size, token_count, embedding_size))


class DecoderBlock(nn.Module):
  def __init__(self, config: StudentConfig):
    super().__init__()
    self.attention_norm = nn.LayerNorm(config.embedding_size)
    self.attention = CausalSelfAttention(config)
    self.feed_forward_norm = nn.LayerNorm(config.embedding_size)
    self.feed_forward = nn.Sequential(
      nn.Linear(config.embedding_size, 4 * config.embedding_size, bias=False),
      nn.GELU(),
      nn.Linear(4 * config.embedding_size, config.embedding_size, bias=False),
    )

  def forward(self, inputs: torch.Tensor, cache: dict[str, torch.Tensor] | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    inputs = inputs + self.attention(self.attention_norm(inputs), cache, mask)
    return inputs + self.feed_forward(self.feed_forward_norm(inputs))


class QosAi(nn.Module):
  def __init__(self, config: StudentConfig):
    super().__init__()
    self.config = config
    self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_size)
    self.position_embedding = nn.Embedding(config.context_length, config.embedding_size)
    self.blocks = nn.Sequential(*(DecoderBlock(config) for _ in range(config.layer_count)))
    self.final_norm = nn.LayerNorm(config.embedding_size)
    self.language_model_head = nn.Linear(config.embedding_size, config.vocabulary_size, bias=False)
    self.language_model_head.weight = self.token_embedding.weight  # tied, like Qwen
    self.apply(self._init_weights)

  @staticmethod
  def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    token_count = tokens.size(1)
    if token_count > self.config.context_length:
      raise ValueError(f"Input exceeds context length {self.config.context_length}")
    positions = torch.arange(token_count, device=tokens.device)
    hidden = self.token_embedding(tokens) + self.position_embedding(positions)
    return self.language_model_head(self.final_norm(self.blocks(hidden)))

  def step(self, tokens: torch.Tensor, positions: torch.Tensor, caches: list[dict[str, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
    """One cached forward pass for batched generation; positions are per row (prompts are left-padded)."""

    hidden = self.token_embedding(tokens) + self.position_embedding(positions)
    for block, cache in zip(self.blocks, caches):
      hidden = block(hidden, cache, mask)
    return self.language_model_head(self.final_norm(hidden))

  @torch.no_grad()
  def generate_batch(self, prompts: list[list[int]], new_token_count: int, stop_id: int, temperature: float = 0.8, top_k: int = 40, no_repeat: int = 0) -> list[tuple[list[int], bool]]:
    """Answers for many prompts at once: [(answer ids, finished with stop_id)].

    Prompts are left-padded to a common length.  Each row's positions start at 0 at its first real
    token, and padded keys are masked out (a padded query attends only to itself, which keeps its
    row finite and never reaches a real token).
    """

    self.eval()
    device = next(self.parameters()).device
    width = max(len(prompt) for prompt in prompts)
    if width + new_token_count > self.config.context_length:
      new_token_count = self.config.context_length - width
    pads = torch.tensor([width - len(prompt) for prompt in prompts], device=device)
    tokens = torch.tensor([[0] * (width - len(prompt)) + prompt for prompt in prompts], device=device)
    index = torch.arange(width, device=device)
    real = index[None, :] >= pads[:, None]                                    # [batch, keys]
    causal = torch.tril(torch.ones(width, width, dtype=torch.bool, device=device))
    mask = (causal[None] & real[:, None, :]) | torch.eye(width, dtype=torch.bool, device=device)[None]
    positions = (index[None, :] - pads[:, None]).clamp(min=0)
    caches: list[dict[str, torch.Tensor]] = [{} for _ in self.blocks]
    logits = self.step(tokens, positions, caches, mask[:, None])[:, -1].float()

    answers = torch.zeros((len(prompts), 0), dtype=torch.long, device=device)
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    guards = [NoRepeat(no_repeat) for _ in prompts]
    for offset in range(new_token_count):
      logits[:, 0] = float("-inf")  # never emit <unk>
      if no_repeat > 0:
        for row, guard in enumerate(guards):
          banned = guard.banned()
          if banned:
            logits[row, list(banned)] = float("-inf")
      next_token = logits.argmax(-1) if temperature <= 0 else gumbel_sample(logits, temperature, top_k)
      next_token = torch.where(done, torch.zeros_like(next_token), next_token)
      answers = torch.cat((answers, next_token[:, None]), dim=1)
      if no_repeat > 0:
        for guard, token in zip(guards, next_token.tolist()):
          guard.add(token)
      finished |= (next_token == stop_id) & ~done
      done |= next_token == stop_id
      if bool(done.all()):
        break
      real = torch.cat((real, torch.ones(len(prompts), 1, dtype=torch.bool, device=device)), dim=1)
      step_positions = (width + offset - pads)[:, None]
      logits = self.step(next_token[:, None], step_positions, caches, real[:, None, None, :])[:, -1].float()

    results = []
    for row, is_finished in zip(answers.tolist(), finished.tolist()):
      if stop_id in row:
        row = row[: row.index(stop_id)]
      results.append((row, is_finished))
    return results

  @torch.no_grad()
  def generate(self, tokens: torch.Tensor, new_token_count: int, stop_id: int, temperature: float = 0.7, top_k: int = 20, no_repeat: int = 0) -> torch.Tensor:
    self.eval()
    guard = NoRepeat(no_repeat)
    for _ in range(new_token_count):
      logits = self(tokens[:, -self.config.context_length:])[:, -1, :].float()
      logits[:, 0] = float("-inf")  # never emit <unk>
      for token in guard.banned():
        logits[:, token] = float("-inf")
      if temperature <= 0:
        next_token = logits.argmax(dim=-1, keepdim=True)
      else:
        next_token = gumbel_sample(logits, temperature, top_k)[:, None]
      tokens = torch.cat((tokens, next_token), dim=1)
      guard.add(int(next_token))
      if int(next_token) == stop_id:
        break
    return tokens


class NoRepeat:
  """Blocks any token that would complete an n-token phrase already written in this answer.

  Standard "no-repeat n-gram" decoding: it hides loops at answer time rather than curing the model,
  but for a small model it is what makes answers readable.  n=0 turns it off.
  """

  def __init__(self, n: int):
    self.n = n
    self.followers: dict[tuple[int, ...], set[int]] = {}
    self.tokens: list[int] = []

  def banned(self) -> set[int]:
    if self.n <= 0 or len(self.tokens) < self.n - 1:
      return set()
    return self.followers.get(tuple(self.tokens[len(self.tokens) - (self.n - 1):]), set())

  def add(self, token: int) -> None:
    self.tokens.append(token)
    if self.n > 0 and len(self.tokens) >= self.n:
      self.followers.setdefault(tuple(self.tokens[-self.n:-1]), set()).add(token)


def parameter_count(model: nn.Module) -> int:
  return sum(parameter.numel() for parameter in model.parameters())


def save_student(path: Path, model: QosAi, lessons: dict[str, Any], step: int) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "model_state": {key.removeprefix("_orig_mod."): value for key, value in model.state_dict().items()},
      "config": asdict(model.config),
      "qwen_ids": lessons["qwen_ids"],
      "pad_id": lessons["pad_id"],
      "im_end_id": lessons["im_end_id"],
      "step": step,
      "teacher": TEACHER_MODEL,
      "split_hash": lessons.get("split_hash", ""),
      "benchmark_ancestry": lessons.get("benchmark_ancestry", "unknown"),
    },
    path,
  )


def load_student(path: Path, device: torch.device) -> tuple[QosAi, dict[str, Any]]:
  checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  model = QosAi(StudentConfig(**checkpoint["config"]))
  model.load_state_dict(checkpoint["model_state"])
  return model.to(device), checkpoint


class StudentTokenizer:
  """Qwen's tokenizer squeezed to the student's vocabulary; unknown tokens become id 0."""

  def __init__(self, qwen_ids: torch.Tensor, qwen: Any = None):
    self.qwen = qwen if qwen is not None else load_tokenizer()
    self.qwen_ids = qwen_ids
    self.lookup = torch.zeros(max(len(self.qwen), int(qwen_ids.max()) + 1), dtype=torch.long)
    self.lookup[qwen_ids[1:]] = torch.arange(1, len(qwen_ids))

  def encode(self, text: str) -> list[int]:
    return self.lookup[torch.tensor(self.qwen.encode(text, add_special_tokens=False))].tolist()

  def decode(self, student_ids: list[int]) -> str:
    return self.qwen.decode([int(self.qwen_ids[i]) for i in student_ids if i != 0], skip_special_tokens=True)


def answer(model: QosAi, tokenizer: StudentTokenizer, question: str, checkpoint: dict[str, Any], args: argparse.Namespace) -> str:
  device = next(model.parameters()).device
  prompt = torch.tensor([tokenizer.encode(chat_text(question))], device=device)
  output = model.generate(prompt, args.max_new_tokens, checkpoint["im_end_id"], args.temperature, args.top_k, getattr(args, "no_repeat", 0))
  return tokenizer.decode(output[0, prompt.shape[1]:].tolist()).strip()


# ----------------------------------------------------------------------------------------------
# step 4: train -- the student regresses towards the teacher's predictions
# ----------------------------------------------------------------------------------------------

LESSON_KEYS = ("tokens", "loss_mask", "topk_ids", "topk_logits")


def load_lessons(qwen_ids: torch.Tensor | None = None) -> dict[str, Any]:
  """All shards, remapped from Qwen ids to the student's reduced vocabulary.

  The vocabulary is student id 0 = <unk>, then every Qwen id the lessons mention, sorted.  A
  continued student keeps its own vocabulary; ids it has never seen map to <unk>.
  """

  files = sorted(LESSONS_DIR.glob("*.pt"))
  if not files:
    raise ValueError(f"No lessons in {LESSONS_DIR}: run `qos_ai.py teach` first")
  shards = [torch.load(path, weights_only=True) for path in files]
  manifest = current_splits()
  for path, shard in zip(files, shards):
    if shard.get("split_schema") != SPLIT_SCHEMA or shard.get("split_hash") != manifest["hash"] or shard.get("split") not in ("train", "dev", "test"):
      raise ValueError(f"Legacy or stale lessons: {path}; rescore with teach")
  tokenizer = load_tokenizer()
  special = [tokenizer.convert_tokens_to_ids(token) for token in (IM_START, IM_END, END_OF_TEXT, "<think>", "</think>")]

  if qwen_ids is None:
    seen = torch.cat([torch.cat((shard["tokens"].flatten(), shard["topk_ids"].flatten())).unique() for shard in shards if shard["split"] == "train"] + [torch.tensor(special, dtype=torch.int32)])
    qwen_ids = torch.cat((torch.tensor([-1]), seen.unique().long()))
  lookup = torch.zeros(max(len(tokenizer), int(qwen_ids.max()) + 1, 248320), dtype=torch.int32)
  lookup[qwen_ids[1:]] = torch.arange(1, len(qwen_ids), dtype=torch.int32)

  def correct_self_play(shard: dict[str, Any]) -> None:
    """Stop the teacher from endorsing the student's loops (in place, on Qwen ids).

    Shown a repeated phrase, the teacher predicts the repeat again (measured: its top choice is the
    student's repeated token 94% of the time, against 49% elsewhere).  So wherever the student's next
    token continues a phrase it already wrote, that token is removed from the teacher's top-k and
    the student learns the teacher's best other continuation.  After SELF_REPEAT_LESSONS such points
    in one answer the rest of it is not learned: by then the context is mostly loop.
    """

    tokens, mask, ids, logits = shard["tokens"], shard["loss_mask"], shard["topk_ids"], shard["topk_logits"]
    for window in range(len(tokens)):
      sequence = tokens[window].tolist()
      seen: set[tuple[int, ...]] = set()
      repeats = 0
      for position in range(len(sequence) - 1):
        if not mask[window, position]:
          if position == 0 or mask[window, position - 1]:
            seen, repeats = set(), 0  # a new answer starts after this unlearned stretch
          continue
        gram = tuple(sequence[max(0, position - 3): position + 2])
        if len(gram) == 5 and gram in seen:
          repeats += 1
          if repeats > SELF_REPEAT_LESSONS:
            mask[window, position] = False
            continue
          logits[window, position][ids[window, position] == sequence[position + 1]] = -3e4
        seen.add(gram)

  lessons: dict[str, Any] = {key: [] for key in LESSON_KEYS}
  lessons["is_qa"] = []
  lessons["is_self"] = []
  lessons["is_repair"] = []
  lessons["partition"] = []
  for shard in shards:
    if shard.get("is_self", False):
      correct_self_play(shard)
    lessons["partition"].append(torch.full((len(shard["tokens"]),), ("train", "dev", "test").index(shard["split"])))
    lessons["tokens"].append(lookup[shard["tokens"].long()])
    topk_ids = lookup[shard["topk_ids"].long()]
    lessons["topk_ids"].append(topk_ids)
    lessons["loss_mask"].append(shard["loss_mask"])
    # a teacher prediction outside the student's vocabulary is dropped rather than taught as <unk>
    lessons["topk_logits"].append(shard["topk_logits"].masked_fill(topk_ids == 0, -3e4))
    lessons["is_qa"].append(torch.full((len(shard["tokens"]),), shard["is_qa"]))
    lessons["is_self"].append(torch.full((len(shard["tokens"]),), shard.get("is_self", False)))
    lessons["is_repair"].append(torch.full((len(shard["tokens"]),), shard.get("is_repair", False)))
  lessons = {key: torch.cat(value) for key, value in lessons.items()}

  unknown = float((lessons["tokens"] == 0).float().mean())
  if unknown > 0:
    print(f"{unknown:.2%} of lesson tokens are outside the student's vocabulary")
  lessons.update(qwen_ids=qwen_ids, pad_id=int(lookup[special[2]]), im_end_id=int(lookup[special[1]]), split_hash=manifest["hash"])
  return lessons


def distillation_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """alpha * KL(teacher top-k || student) * T^2  +  (1 - alpha) * cross-entropy on the real next token."""

  logits = logits[:, :-1].float()  # the prediction at position t is about token t+1
  targets = batch["tokens"][:, 1:].long()
  mask = batch["loss_mask"][:, :-1].float()
  count = mask.sum().clamp(min=1)

  hard = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(mask)
  # On the student's own answers the next token is whatever it wrote, mistakes included, so only the
  # teacher's distribution is learned there (hard_weight 0); everywhere else both terms count.
  hard_mask = mask * batch["hard_weight"][:, None] if "hard_weight" in batch else mask
  hard = (hard * hard_mask).sum() / hard_mask.sum().clamp(min=1)

  # Teacher's top-k renormalized to a distribution; student log-probs gathered at those same ids.
  temperature = args.distill_temperature
  teacher = F.softmax(batch["topk_logits"][:, :-1].float() / temperature, dim=-1)
  student_log = F.log_softmax(logits / temperature, dim=-1).gather(-1, batch["topk_ids"][:, :-1].long())
  soft = (teacher * (torch.log(teacher + 1e-9) - student_log)).sum(-1)
  soft = (soft * mask).sum() / count * temperature**2

  return args.alpha * soft + (1 - args.alpha) * hard, hard, soft


def load_teacher_embeddings(qwen_ids: torch.Tensor) -> torch.Tensor:
  """The teacher's embedding rows for the student's vocabulary, read straight from the safetensors file."""

  from safetensors import safe_open
  from transformers.utils import cached_file

  key = "model.language_model.embed_tokens.weight"
  index = json.loads(Path(cached_file(EMBEDDING_MODEL, "model.safetensors.index.json", **hub_kwargs(EMBEDDING_MODEL))).read_text())
  with safe_open(cached_file(EMBEDDING_MODEL, index["weight_map"][key], **hub_kwargs(EMBEDDING_MODEL)), "pt") as file:
    table = file.get_tensor(key)
  return table[qwen_ids[1:]].float()


def split_windows(lessons: dict[str, Any]) -> tuple[torch.Tensor, ...]:
  """Train partitions and development validation; test windows never select checkpoints."""
  indices = torch.arange(len(lessons["tokens"]))
  if "partition" not in lessons:
    raise ValueError("Lessons have no explicit partition; legacy window holdouts are refused")
  training = lessons["partition"] == 0
  own, repair = lessons["is_self"], lessons["is_repair"]
  docs = ~lessons["is_qa"] & ~own & ~repair
  validation = indices[lessons["partition"] == 1]
  return (indices[training & docs], indices[training & lessons["is_qa"]],
          indices[training & own], indices[training & repair], validation)


@torch.no_grad()
def evaluate(model: nn.Module, lessons: dict[str, Any], indices: torch.Tensor, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
  """Mean (loss, hard, soft) over held-out windows the student never trains on."""

  model.eval()
  totals = torch.zeros(3)
  batches = 0
  for start in range(0, len(indices), args.batch_size):
    rows = indices[start: start + args.batch_size]
    batch = {key: lessons[key][rows.to(lessons[key].device)].to(device) for key in LESSON_KEYS}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
      logits = model(batch["tokens"].long())
    totals += torch.tensor([value.item() for value in distillation_loss(logits, batch, args)])
    batches += 1
  return totals / max(batches, 1)


def train(args: argparse.Namespace) -> None:
  torch.manual_seed(args.seed)
  device = choose_device(args.device)
  if args.bf16 is None:
    args.bf16 = device.type == "cuda"
  if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

  checkpoint = torch.load(args.load, map_location="cpu", weights_only=True) if args.load else None
  lessons = load_lessons(checkpoint["qwen_ids"] if checkpoint else None)
  lessons["benchmark_ancestry"] = checkpoint.get("benchmark_ancestry", "legacy_or_unknown") if checkpoint else "split_isolated_from_scratch"
  window_count, context_length = lessons["tokens"].shape
  if device.type == "cuda":  # a few GB at most; saves a host-to-device copy every step
    for key in LESSON_KEYS:
      lessons[key] = lessons[key].to(device)

  docs, qa, own, repairs, validation = split_windows(lessons)
  self_per_batch = round(args.batch_size * args.self_fraction) if len(own) else 0
  repair_per_batch = round(args.batch_size * args.repair_fraction) if len(repairs) else 0
  qa_per_batch = round(args.batch_size * args.qa_fraction) if len(qa) else 0
  docs_per_batch = args.batch_size - self_per_batch - repair_per_batch - qa_per_batch
  if docs_per_batch < 0:
    raise ValueError("--qa-fraction, --self-fraction and --repair-fraction must not add up to more than 1")
  print(f"Using device {device} (bf16={args.bf16}); {len(docs)} doc + {len(qa)} Q&A + {len(own)} self-play + {len(repairs)} repair "
        f"training windows, {len(validation)} validation windows of {context_length} tokens; each batch: {docs_per_batch} doc, "
        f"{qa_per_batch} Q&A, {self_per_batch} self-play, {repair_per_batch} repair")

  if checkpoint is not None:
    model, _ = load_student(args.load, device)
    step = checkpoint["step"]
    print(f"Continuing student from {args.load} at step {step}")
  else:
    config = StudentConfig(
      vocabulary_size=len(lessons["qwen_ids"]),
      context_length=context_length,
      embedding_size=args.embedding_size,
      head_count=args.heads,
      layer_count=args.layers,
    )
    model = QosAi(config).to(device)
    step = 0
    if args.teacher_embeddings:
      table = load_teacher_embeddings(lessons["qwen_ids"])
      if table.shape[1] != config.embedding_size:
        raise ValueError(f"--teacher-embeddings needs --embedding-size {table.shape[1]}, the teacher's width")
      with torch.no_grad():
        model.token_embedding.weight[1:] = table.to(device)  # row 0 (<unk>) stays random; the output head is tied
      print("Token embeddings start from the teacher's")
  print(f"Student has {parameter_count(model):,} parameters")

  steps = args.steps
  if args.epochs:
    steps = math.ceil(args.epochs * (len(docs) + len(qa) + len(own) + len(repairs)) / (args.batch_size * args.grad_accumulation))
    print(f"{args.epochs} epochs = {steps} steps")
  tokenizer = StudentTokenizer(lessons["qwen_ids"])
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
  forward = torch.compile(model) if args.compile else model
  first_step = step
  final_step = step + steps

  def learning_rate(current: int) -> float:
    current -= first_step  # a continued run (e.g. from the pretrained model) gets its own warmup and decay
    if current < args.warmup_steps:
      return args.learning_rate * current / args.warmup_steps
    progress = (current - args.warmup_steps) / max(1, steps - args.warmup_steps)
    return args.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

  def draw(pool: torch.Tensor, count: int) -> torch.Tensor:
    return pool[torch.randint(len(pool), (count,))]

  started = time.perf_counter()
  while step < final_step:
    step += 1
    model.train()
    for group in optimizer.param_groups:
      group["lr"] = learning_rate(step)

    # gradient accumulation: several small batches add up their gradients before one optimizer step
    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.grad_accumulation):
      # Q&A is a small share of all tokens but it is the format the student is asked in, so each
      # batch draws a fixed share of Q&A windows
      parts = [draw(docs, docs_per_batch)] if docs_per_batch else []
      parts += [draw(qa, qa_per_batch)] if qa_per_batch else []
      parts += [draw(own, self_per_batch)] if self_per_batch else []
      parts += [draw(repairs, repair_per_batch)] if repair_per_batch else []
      rows = torch.cat(parts)
      batch = {key: lessons[key][rows.to(lessons[key].device)].to(device) for key in LESSON_KEYS}
      batch["hard_weight"] = (~lessons["is_self"][rows.cpu()]).float().to(device)
      with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
        logits = forward(batch["tokens"].long())
      loss, hard, soft = distillation_loss(logits, batch, args)
      (loss / args.grad_accumulation).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step == 1 or step % args.report_every == 0 or step == final_step:
      steps_done = step - (final_step - steps)
      rate = steps_done * args.grad_accumulation * args.batch_size * context_length / (time.perf_counter() - started)
      print(f"step {step:>5}/{final_step}: loss={loss.item():.3f} hard={hard.item():.3f} soft={soft.item():.3f} lr={learning_rate(step):.2e} ({rate:.0f} tokens/s)", flush=True)
    if step % args.sample_every == 0 or step == final_step:
      if len(validation):
        result = evaluate(model, lessons, validation, args, device)
        print(f"  validation: loss={result[0]:.3f} hard={result[1]:.3f} soft={result[2]:.3f}")
      print(f"  {SAMPLE_PROMPT} -> {answer(model, tokenizer, SAMPLE_PROMPT, lessons, args)!r}", flush=True)
    if step % args.save_every == 0 or step == final_step:
      save_student(args.output, model, lessons, step)

  print(f"Saved student to {args.output}")


# ----------------------------------------------------------------------------------------------
# general English first: pretokenize + pretrain
# ----------------------------------------------------------------------------------------------
# A model trained from scratch only on ~20M tokens of FP-RISC Q&A memorizes fragments and loops.
# Pretraining the same student on a few hundred million tokens of ordinary educational English
# (FineWeb-Edu) gives it grammar and general knowledge first; `train --load qos_ai/pretrained.pt`
# then specializes it with the teacher's lessons.  Still our own model, trained from scratch.

FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_FILES = [f"sample/10BT/{index:03d}_00000.parquet" for index in range(14)]  # ~2.15 GB each


def iter_texts(paths: list[Path]) -> Any:
  """Document texts from parquet files (a `text` column, as in FineWeb) or jsonl files ({"text": ...})."""

  for path in paths:
    if path.suffix == ".parquet":
      import pyarrow.parquet as pq

      for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["text"]):
        yield from batch.column(0).to_pylist()
    else:
      for row in read_jsonl(path):
        yield row["text"]


def domain_token_ids(tokenizer: Any) -> set[int]:
  """Every Qwen id that appears in the corpus or the usable Q&A, in the chat format the student sees."""

  ids: set[int] = set()
  for row in read_jsonl(CORPUS_FILE):
    ids.update(tokenizer.encode(f"# File: {row['path']}\n{row['text']}\n", add_special_tokens=False))
  for row in usable_pairs():
    ids.update(tokenizer.encode(chat_text(row["question"], row["answer"]), add_special_tokens=False))
  return ids


def pretokenize(args: argparse.Namespace) -> None:
  import numpy as np

  tokenizer = load_tokenizer()
  if args.data:
    paths = [Path(path) for path in args.data]
  else:
    from huggingface_hub import hf_hub_download

    paths = []  # downloaded one file at a time, only as many as the token budget needs
  end_of_text = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
  counts = np.zeros(QWEN_VOCAB, dtype=np.int64)
  PRETRAIN_TOKENS.parent.mkdir(parents=True, exist_ok=True)
  written = 0
  started = time.perf_counter()
  with PRETRAIN_TOKENS.open("ab" if args.append else "wb") as out:
    file_index = args.first_file
    while written < args.tokens:
      if args.data:
        if file_index >= len(paths):
          break
        path = paths[file_index]
      else:
        if file_index >= len(FINEWEB_FILES):
          break
        print(f"downloading {FINEWEB_REPO}/{FINEWEB_FILES[file_index]}...", flush=True)
        path = Path(hf_hub_download(FINEWEB_REPO, FINEWEB_FILES[file_index], repo_type="dataset"))
      file_index += 1

      def flush(texts: list[str]) -> int:
        count = 0
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
          ids.append(end_of_text)  # documents are separated by <|endoftext|>
          array = np.asarray(ids, dtype=np.uint32)
          array.tofile(out)
          counts[:] += np.bincount(array, minlength=QWEN_VOCAB)[:QWEN_VOCAB]
          count += len(array)
        return count

      texts: list[str] = []
      for text in iter_texts([path]):
        texts.append(text)
        if len(texts) < 2000:
          continue
        written += flush(texts)
        texts = []
        rate = written / (time.perf_counter() - started)
        print(f"tokenized {written / 1e6:.0f}M / {args.tokens / 1e6:.0f}M tokens ({rate / 1e3:.0f}k tokens/s)", flush=True)
        if written >= args.tokens:
          break
      if texts and written < args.tokens:
        written += flush(texts)
  print(f"Wrote {written:,} tokens to {PRETRAIN_TOKENS}")
  if args.append:
    print(f"{PRETRAIN_TOKENS} now holds {PRETRAIN_TOKENS.stat().st_size // 4:,} tokens; the vocabulary in {VOCAB_FILE} is unchanged")
    return  # the student's vocabulary is fixed once it has been trained

  # The fixed student vocabulary: every token the domain text uses, the chat-format specials, and
  # then the most frequent general-English tokens until the vocabulary reaches --vocab-size.
  special = {tokenizer.convert_tokens_to_ids(token) for token in (IM_START, IM_END, END_OF_TEXT, "<think>", "</think>")}
  keep = domain_token_ids(tokenizer) | special
  for qwen_id in np.argsort(-counts):
    if len(keep) >= args.vocab_size - 1 or counts[qwen_id] == 0:
      break
    keep.add(int(qwen_id))
  qwen_ids = torch.cat((torch.tensor([-1]), torch.tensor(sorted(keep))))
  covered = counts[sorted(keep)].sum() / max(counts.sum(), 1)
  torch.save({"qwen_ids": qwen_ids, "general_coverage": float(covered)}, VOCAB_FILE)
  print(f"Vocabulary of {len(qwen_ids):,} tokens covers {covered:.2%} of the general text; saved to {VOCAB_FILE}")


def student_lookup(qwen_ids: torch.Tensor) -> torch.Tensor:
  lookup = torch.zeros(max(QWEN_VOCAB, int(qwen_ids.max()) + 1), dtype=torch.long)
  lookup[qwen_ids[1:]] = torch.arange(1, len(qwen_ids))
  return lookup


def pretrain(args: argparse.Namespace) -> None:
  import numpy as np

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = choose_device(args.device)
  if args.bf16 is None:
    args.bf16 = device.type == "cuda"
  if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

  tokens = np.memmap(PRETRAIN_TOKENS, dtype=np.uint32, mode="r")
  qwen_ids = torch.load(VOCAB_FILE, weights_only=True)["qwen_ids"]
  lookup = student_lookup(qwen_ids).to(device)
  length = args.context_length
  validation_start = len(tokens) - args.validation_tokens  # the tail is held out
  tokens_per_step = args.batch_size * args.grad_accumulation * length
  total_steps = math.ceil(args.tokens / tokens_per_step) if args.tokens else math.ceil(validation_start / tokens_per_step)

  tokenizer = load_tokenizer()
  pad_id = int(lookup[tokenizer.convert_tokens_to_ids(END_OF_TEXT)])
  vocabulary = {"qwen_ids": qwen_ids, "pad_id": pad_id, "im_end_id": int(lookup[tokenizer.convert_tokens_to_ids(IM_END)])}

  if args.load:
    model, checkpoint = load_student(args.load, device)
    step = checkpoint["step"]
    print(f"Resuming pretraining from {args.load} at step {step}")
  else:
    config = StudentConfig(vocabulary_size=len(qwen_ids), context_length=length, embedding_size=args.embedding_size, head_count=args.heads, layer_count=args.layers)
    model = QosAi(config).to(device)
    step = 0
    if args.teacher_embeddings:
      table = load_teacher_embeddings(qwen_ids)
      if table.shape[1] != config.embedding_size:
        raise ValueError(f"--teacher-embeddings needs --embedding-size {table.shape[1]}")
      with torch.no_grad():
        model.token_embedding.weight[1:] = table.to(device)
  print(f"Pretraining {parameter_count(model):,} parameters on {len(tokens):,} tokens ({args.validation_tokens:,} held out); "
        f"{total_steps} steps of {tokens_per_step:,} tokens; device {device}, bf16={args.bf16}", flush=True)

  def draw(count: int, low: int, high: int) -> torch.Tensor:
    starts = np.random.randint(low, high - length - 1, count)
    batch = torch.from_numpy(np.stack([tokens[start: start + length + 1].astype(np.int64) for start in starts]))
    return lookup[batch.to(device)]

  def lm_loss(network: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
      logits = network(batch[:, :-1])
    return F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten(), ignore_index=0)  # <unk> targets are skipped

  forward = model
  if args.compile:
    try:
      forward = torch.compile(model)
      lm_loss(forward, draw(2, 0, validation_start)).backward()  # compile now, so a failure falls back right away
      model.zero_grad(set_to_none=True)
    except Exception as error:  # noqa: BLE001 - any compiler failure: train uncompiled instead of dying
      print(f"torch.compile failed ({type(error).__name__}: {str(error)[:200]}); continuing without it")
      forward = model

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

  resumed_at = step

  def learning_rate(current: int) -> float:
    if current < args.warmup_steps:
      return args.learning_rate * current / args.warmup_steps
    progress = min(1.0, (current - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
    rate = args.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
    # a resumed run starts with a fresh optimizer (and, when the token budget grew, a higher rate than
    # where it stopped), so it warms up again rather than jumping straight to that rate
    if resumed_at:
      rate *= min(1.0, (current - resumed_at) / args.warmup_steps)
    return rate

  @torch.no_grad()
  def validation_loss() -> float:
    model.eval()
    generator_state = np.random.get_state()
    np.random.seed(1234)  # the same held-out windows every time
    losses = [float(lm_loss(model, draw(args.batch_size, validation_start, len(tokens)))) for _ in range(4)]
    np.random.set_state(generator_state)
    model.train()
    return sum(losses) / len(losses)

  student = StudentTokenizer(qwen_ids, tokenizer)  # built once: reloading mid-run can block on a Hugging Face request

  def sample(prompt: str) -> str:
    prompt_ids = torch.tensor([student.encode(prompt)], device=device)
    output = model.generate(prompt_ids, 40, pad_id, 0.7, 20)
    model.train()
    return student.decode(output[0, prompt_ids.shape[1]:].tolist())

  started = time.perf_counter()
  first_step = step
  deadline = started + args.max_minutes * 60 if args.max_minutes else None
  model.train()
  while step < total_steps:
    step += 1
    for group in optimizer.param_groups:
      group["lr"] = learning_rate(step)
    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.grad_accumulation):
      loss = lm_loss(forward, draw(args.batch_size, 0, validation_start))
      (loss / args.grad_accumulation).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step == first_step + 1 or step % args.report_every == 0:
      rate = (step - first_step) * tokens_per_step / (time.perf_counter() - started)
      remaining = (total_steps - step) * tokens_per_step / max(rate, 1) / 60
      print(f"step {step:>6}/{total_steps}: loss={loss.item():.3f} lr={learning_rate(step):.2e} ({rate / 1e3:.0f}k tokens/s, ~{remaining:.0f} min left)", flush=True)
    finished = step == total_steps or (deadline is not None and time.perf_counter() > deadline)
    if step % args.sample_every == 0 or finished:
      print(f"  validation loss={validation_loss():.3f}  sample: The purpose of a compiler is{sample('The purpose of a compiler is')!r}", flush=True)
    if step % args.save_every == 0 or finished:
      save_student(args.output, model, vocabulary, step)
    if finished:
      break
  print(f"Saved pretrained student to {args.output} at step {step} ({step * tokens_per_step / 1e6:.0f}M tokens)")


# ----------------------------------------------------------------------------------------------
# self-correction: the student answers, the teacher scores those answers, the student learns
# ----------------------------------------------------------------------------------------------
# Trained only on the teacher's perfect text, the student never sees its own mistakes, so once
# it drifts (a repeated phrase) it has no idea how to recover and loops.  Here it writes its own
# answers; `teach` then has the teacher score every token of them, and `train` learns the
# teacher's predictions on that text (never the student's own tokens as targets).


def loops(ids: list[int], phrase: int = 3, times: int = 3) -> bool:
  """Whether any `phrase`-token sequence occurs at least `times` times."""

  seen: dict[tuple[int, ...], int] = {}
  for index in range(len(ids) - phrase + 1):
    key = tuple(ids[index: index + phrase])
    seen[key] = seen.get(key, 0) + 1
    if seen[key] >= times:
      return True
  return False


def unique_questions(rows: list[dict[str, Any]]) -> list[str]:
  return list(dict.fromkeys(row["question"] for row in rows))


def answer_many(model: QosAi, tokenizer: "StudentTokenizer", questions: list[str], im_end_id: int, args: argparse.Namespace) -> list[tuple[list[int], bool]]:
  device = next(model.parameters()).device
  results: list[tuple[list[int], bool]] = []
  for start in range(0, len(questions), args.batch_size):
    prompts = [tokenizer.encode(chat_text(question)) for question in questions[start: start + args.batch_size]]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
      results += model.generate_batch(prompts, args.max_new_tokens, im_end_id, args.temperature, args.top_k, getattr(args, "no_repeat", 0))
  return results


def selfplay(args: argparse.Namespace) -> None:
  random.seed(args.seed + args.round)
  torch.manual_seed(args.seed + args.round)
  device = choose_device(args.device)
  model, checkpoint = load_student(args.load, device)
  tokenizer = StudentTokenizer(checkpoint["qwen_ids"])
  manifest = current_splits()
  questions = unique_questions([row for row in usable_pairs() if row_split(row, manifest) == "train"])
  random.shuffle(questions)
  done = {row["question"] for row in read_jsonl(SELF_FILE) if row["round"] == args.round}
  questions = [question for question in questions[: args.questions] if question not in done]
  print(f"Round {args.round}: the student answers {len(questions)} questions ({len(done)} already done)", flush=True)

  started = time.perf_counter()
  looped = finished_count = 0
  for start in range(0, len(questions), args.save_every):
    chunk = questions[start: start + args.save_every]
    results = answer_many(model, tokenizer, chunk, checkpoint["im_end_id"], args)
    rows = []
    for question, (ids, finished) in zip(chunk, results):
      looped += loops(ids)
      finished_count += finished
      rows.append({"round": args.round, "question": question, "answer": tokenizer.decode(ids), "finished": finished})
    append_jsonl(SELF_FILE, rows)
    total = start + len(chunk)
    print(f"answered {total}/{len(questions)} ({total / (time.perf_counter() - started):.0f}/s); "
          f"looping {looped / total:.0%}, finished {finished_count / total:.0%}", flush=True)


def loop_onset(ids: list[int], phrase: int = 4) -> int | None:
  """Index where the first repeat of a `phrase`-token sequence begins, or None if nothing repeats."""

  seen: set[tuple[int, ...]] = set()
  for index in range(len(ids) - phrase + 1):
    key = tuple(ids[index: index + phrase])
    if key in seen:
      return index
    seen.add(key)
  return None


def repair(args: argparse.Namespace) -> None:
  """The teacher finishes the student's answers from just before their first repeat.

  For each self-play answer of `--source-round`: cut it where the first repeated 4-token phrase
  begins (an unfinished answer without a repeat is cut at its last sentence; a clean finished one
  needs no repair), then the teacher continues from that point with the question's reference
  material in view, and ends the answer.  The student later learns that continuation from its own
  imperfect start, which is what it never sees in the teacher's own answers.
  """

  random.seed(args.seed)
  tokenizer = load_tokenizer()
  references = {(row["path"], row["chunk"]): row for row in read_jsonl(CORPUS_FILE)}
  manifest = current_splits()
  source_of = {row["question"]: (row["path"], row["chunk"]) for row in usable_pairs() if row_split(row, manifest) == "train"}
  done = {row["question"] for row in read_jsonl(REPAIR_FILE) if row["round"] == args.round}

  jobs = []
  for row in read_jsonl(SELF_FILE):
    if row["round"] != args.source_round or row["question"] in done or row["question"] not in source_of:
      continue
    ids = tokenizer.encode(row["answer"], add_special_tokens=False)
    cut = loop_onset(ids)
    if cut is None:
      if row["finished"]:
        continue
      text = row["answer"]
      cut_text = text[: max(text.rfind(". "), text.rfind(".\n")) + 1]
    else:
      cut_text = tokenizer.decode(ids[:cut])
    jobs.append((row["question"], cut_text, references[source_of[row["question"]]]))
  random.shuffle(jobs)
  jobs = jobs[: args.questions]
  print(f"Round {args.round}: the teacher finishes {len(jobs)} answers from self-play round {args.source_round}", flush=True)
  if not jobs:
    return

  generate = VllmGenerator(args)
  started = time.perf_counter()
  kept = 0
  rejected: dict[str, int] = {}
  for start in range(0, len(jobs), args.batch_size):
    batch = jobs[start: start + args.batch_size]
    texts = []
    for question, prefix, reference in batch:
      user = (f"Reference material from `{reference['path']}`:\n\n{reference['text']}\n\nQuestion: {question}\n\n"
              "Answer the way official documentation would: state the facts directly in a few sentences, with a short "
              "code example when it helps. Write any code in the same language as the reference material. "
              "Never refer to the reference material, an excerpt or a snippet.")
      prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
      )
      texts.append(prompt + prefix)
    replies = generate.continue_text(texts, args.max_new_tokens, args.seed + start)
    rows = []
    for (question, prefix, _), (continuation, finished) in zip(batch, replies):
      problem = clean_pair(question, prefix + continuation)
      if problem and problem != "too short":
        rejected[problem] = rejected.get(problem, 0) + 1
        continue
      rows.append({"round": args.round, "question": question, "prefix": prefix, "continuation": continuation, "finished": finished})
    append_jsonl(REPAIR_FILE, rows)
    kept += len(rows)
    print(f"[{start + len(batch)}/{len(jobs)}, {(time.perf_counter() - started) / 60:.1f} min] kept {kept}, rejected {rejected}", flush=True)


def evalloops(args: argparse.Namespace) -> None:
  torch.manual_seed(args.seed)
  device = choose_device(args.device)
  model, checkpoint = load_student(args.load, device)
  tokenizer = StudentTokenizer(checkpoint["qwen_ids"])
  manifest = current_splits()
  questions = unique_questions([row for row in usable_pairs() if row_split(row, manifest) == "dev"])[: args.questions]
  if not questions:
    raise ValueError("No development questions in frozen splits")
  print("Exposure note: existing checkpoints may have seen these questions; use benchmark reports for ancestry metadata")
  results = answer_many(model, tokenizer, questions, checkpoint["im_end_id"], args)
  looped = sum(loops(ids) for ids, _ in results)
  severe = sum(loops(ids, phrase=4, times=4) for ids, _ in results)
  finished = sum(done for _, done in results)
  length = sum(len(ids) for ids, _ in results) / max(len(results), 1)
  print(f"{args.load}: {len(results)} held-out questions; repeats a 3-token phrase 3+ times {looped / len(results):.1%}, "
        f"a 4-token phrase 4+ times {severe / len(results):.1%}; finished {finished / len(results):.1%}; "
        f"mean answer {length:.0f} tokens", flush=True)
  for question, (ids, _) in list(zip(questions, results))[: args.show]:
    print(f"  Q: {question[:100]}\n  A: {tokenizer.decode(ids)[:300]!r}")


# ----------------------------------------------------------------------------------------------
# step 5: chat
# ----------------------------------------------------------------------------------------------


def chat(args: argparse.Namespace) -> None:
  torch.manual_seed(args.seed)
  device = choose_device(args.device)
  model, checkpoint = load_student(args.load, device)
  tokenizer = StudentTokenizer(checkpoint["qwen_ids"])
  print(f"qos_ai: {parameter_count(model):,} parameters, trained {checkpoint['step']} steps")

  if args.prompt:
    print(answer(model, tokenizer, args.prompt, checkpoint, args))
    return
  print("Ask about FP-RISC, Sol or QOS. Enter /quit to exit.")
  while True:
    try:
      question = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
      print()
      break
    if question.lower() in {"/quit", "/exit"}:
      break
    if question:
      print(f"\nqos_ai: {answer(model, tokenizer, question, checkpoint, args)}")


# ----------------------------------------------------------------------------------------------
# loop: ask -> teach -> train, repeated, so the student sees fresh questions every round
# ----------------------------------------------------------------------------------------------


def loop(args: argparse.Namespace) -> None:
  script = [sys.executable, str(Path(__file__))]
  for round_index in range(1, args.rounds + 1):
    print(f"\n===== round {round_index}/{args.rounds} =====")
    subprocess.run(script + ["ask", "--round", str(round_index), "--seed", str(args.seed)], check=True)
    subprocess.run(script + ["teach"], check=True)  # only the new round is scored
    train_args = ["train", "--steps", str(args.steps)]
    if round_index > 1 or STUDENT_FILE.exists():
      train_args += ["--load", str(STUDENT_FILE)]
    elif args.teacher_embeddings:
      train_args += ["--teacher-embeddings"]
    subprocess.run(script + train_args, check=True)


# ----------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  commands = parser.add_subparsers(dest="command", required=True)
  devices = ("auto", "cpu", "mps", "cuda")

  splits_parser = commands.add_parser("splits", help="freeze file-grouped data splits and benchmark exclusions before teaching")
  splits_parser.set_defaults(run=prepare_splits)

  corpus_parser = commands.add_parser("corpus", help="collect fprisc and qos docs and source into chunks")
  corpus_parser.add_argument("--roots", type=Path, nargs="+", default=DEFAULT_ROOTS, help="checkouts to read (fprisc and qos)")
  corpus_parser.add_argument("--chunk-size", type=int, default=1800, help="characters per chunk")
  corpus_parser.add_argument("--limit", type=int, default=0, help="stop after this many chunks (smoke runs)")
  corpus_parser.set_defaults(run=build_corpus)

  ask_parser = commands.add_parser("ask", help="teacher writes Q&A pairs about each chunk")
  ask_parser.add_argument("--limit", type=int, default=0, help="only use a random sample of N chunks")
  ask_parser.add_argument("--round", type=int, default=1, help="first round; pairs are keyed by round so later rounds add fresh ones")
  ask_parser.add_argument("--rounds", type=int, default=1, help="how many rounds to generate, with the teacher loaded once")
  ask_parser.add_argument("--questions-per-chunk", type=int, default=3, help="question styles per chunk per round")
  ask_parser.add_argument("--style-set", choices=sorted(STYLE_SETS), default="original", help="which question styles to draw from; extended = new angles for later rounds")
  ask_parser.add_argument("--max-answer-tokens", type=int, default=256)
  ask_parser.add_argument("--batch-size", type=int, default=8, help="prompts per generate call; use hundreds with vLLM")
  ask_parser.add_argument("--temperature", type=float, default=0.7)
  ask_parser.add_argument("--top-p", type=float, default=0.8)
  ask_parser.add_argument("--top-k", type=int, default=20)
  ask_parser.add_argument("--engine", choices=("auto", "hf", "vllm"), default="auto", help="auto = vLLM on CUDA when installed")
  ask_parser.add_argument("--gpu-memory", type=float, default=0.75, help="share of GPU memory vLLM may use; the rest is headroom for its 248k-vocab sampler")
  ask_parser.add_argument("--seed", type=int, default=7)
  ask_parser.add_argument("--device", choices=devices, default="auto")
  ask_parser.set_defaults(run=ask_teacher)

  lessonplan_parser = commands.add_parser("lessonplan", help="inspect split packing without loading a teacher or training")
  lessonplan_parser.add_argument("--context-length", type=int, default=512)
  lessonplan_parser.set_defaults(run=lesson_plan)

  teach_parser = commands.add_parser("teach", help="teacher scores every token of docs and Q&A")
  teach_parser.add_argument("--context-length", type=int, default=512)
  teach_parser.add_argument("--limit", type=int, default=0, help="only score the first N windows of each source (smoke runs)")
  teach_parser.add_argument("--top-k", type=int, default=32, help="teacher predictions kept per token")
  teach_parser.add_argument("--batch-size", type=int, default=4, help="each window costs ~500MB of teacher logits in fp32")
  teach_parser.add_argument("--shard-size", type=int, default=256, help="windows per lessons file")
  teach_parser.add_argument("--device", choices=devices, default="auto")
  teach_parser.set_defaults(run=teach)

  train_parser = commands.add_parser("train", help="train the student on the lessons")
  train_parser.add_argument("--steps", type=int, default=3000)
  train_parser.add_argument("--epochs", type=float, default=0, help="if set, overrides --steps: passes over the training windows")
  train_parser.add_argument("--batch-size", type=int, default=2, help="windows per forward pass; 2 fits the default student in 16GB")
  train_parser.add_argument("--grad-accumulation", type=int, default=4, help="forward passes per optimizer step")
  train_parser.add_argument("--qa-fraction", type=float, default=0.5, help="share of each batch drawn from Q&A windows")
  train_parser.add_argument("--self-fraction", type=float, default=0.0, help="share of each batch drawn from self-play windows (the student's own answers)")
  train_parser.add_argument("--repair-fraction", type=float, default=0.0, help="share of each batch drawn from repair windows (the teacher finishing the student's answers)")
  train_parser.add_argument("--embedding-size", type=int, default=1024)
  train_parser.add_argument("--heads", type=int, default=16)
  train_parser.add_argument("--layers", type=int, default=16)
  train_parser.add_argument("--teacher-embeddings", action="store_true", help="initialize token embeddings from the teacher (needs embedding size 1024)")
  train_parser.add_argument("--learning-rate", type=float, default=3e-4)
  train_parser.add_argument("--warmup-steps", type=int, default=100)
  train_parser.add_argument("--alpha", type=float, default=0.7, help="weight of the soft (teacher) loss vs the hard next-token loss")
  train_parser.add_argument("--distill-temperature", type=float, default=2.0)
  train_parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="bfloat16 autocast; default on for CUDA")
  train_parser.add_argument("--compile", action="store_true", help="torch.compile the student (CUDA)")
  train_parser.add_argument("--report-every", type=int, default=25)
  train_parser.add_argument("--sample-every", type=int, default=250)
  train_parser.add_argument("--save-every", type=int, default=250)
  train_parser.add_argument("--max-new-tokens", type=int, default=60)
  train_parser.add_argument("--temperature", type=float, default=0.7)
  train_parser.add_argument("--top-k", type=int, default=20)
  train_parser.add_argument("--seed", type=int, default=7)
  train_parser.add_argument("--device", choices=devices, default="auto")
  train_parser.add_argument("--output", type=Path, default=STUDENT_FILE)
  train_parser.add_argument("--load", type=Path, help="continue training this checkpoint")
  train_parser.set_defaults(run=train)

  chat_parser = commands.add_parser("chat", help="ask the student a question")
  chat_parser.add_argument("prompt", nargs="?", help="one-shot question; omit for interactive mode")
  chat_parser.add_argument("--load", type=Path, default=STUDENT_FILE)
  chat_parser.add_argument("--max-new-tokens", type=int, default=200)
  chat_parser.add_argument("--temperature", type=float, default=0.7, help="zero for greedy decoding")
  chat_parser.add_argument("--top-k", type=int, default=20)
  chat_parser.add_argument("--no-repeat", type=int, default=3, help="block repeating any phrase of this many tokens within an answer; 0 = off")
  chat_parser.add_argument("--seed", type=int, default=7)
  chat_parser.add_argument("--device", choices=devices, default="auto")
  chat_parser.set_defaults(run=chat)

  pretokenize_parser = commands.add_parser("pretokenize", help="download FineWeb-Edu, tokenize it, and choose the fixed student vocabulary")
  pretokenize_parser.add_argument("--tokens", type=int, default=500_000_000, help="general-English tokens to prepare")
  pretokenize_parser.add_argument("--vocab-size", type=int, default=65536)
  pretokenize_parser.add_argument("--data", type=Path, nargs="*", help="local .parquet/.jsonl files instead of downloading FineWeb-Edu")
  pretokenize_parser.add_argument("--append", action="store_true", help="add --tokens more tokens to the existing file and keep the vocabulary")
  pretokenize_parser.add_argument("--first-file", type=int, default=0, help="first FineWeb-Edu file (or --data file) to read, e.g. 1 to skip data already used")
  pretokenize_parser.set_defaults(run=pretokenize)

  pretrain_parser = commands.add_parser("pretrain", help="train the student on general English (next-token loss)")
  pretrain_parser.add_argument("--tokens", type=int, default=0, help="tokens to train on; 0 = one pass over the pretokenized data")
  pretrain_parser.add_argument("--validation-tokens", type=int, default=1_000_000)
  pretrain_parser.add_argument("--context-length", type=int, default=512)
  pretrain_parser.add_argument("--batch-size", type=int, default=16)
  pretrain_parser.add_argument("--grad-accumulation", type=int, default=2)
  pretrain_parser.add_argument("--embedding-size", type=int, default=1024)
  pretrain_parser.add_argument("--heads", type=int, default=16)
  pretrain_parser.add_argument("--layers", type=int, default=8)
  pretrain_parser.add_argument("--teacher-embeddings", action="store_true")
  pretrain_parser.add_argument("--learning-rate", type=float, default=6e-4)
  pretrain_parser.add_argument("--warmup-steps", type=int, default=500)
  pretrain_parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
  pretrain_parser.add_argument("--compile", action="store_true", help="torch.compile; falls back to eager if it fails")
  pretrain_parser.add_argument("--max-minutes", type=float, default=0, help="stop and save after this long (budget guard)")
  pretrain_parser.add_argument("--report-every", type=int, default=100)
  pretrain_parser.add_argument("--sample-every", type=int, default=2000)
  pretrain_parser.add_argument("--save-every", type=int, default=2000)
  pretrain_parser.add_argument("--seed", type=int, default=7)
  pretrain_parser.add_argument("--device", choices=devices, default="auto")
  pretrain_parser.add_argument("--output", type=Path, default=PRETRAINED_FILE)
  pretrain_parser.add_argument("--load", type=Path, help="resume this checkpoint")
  pretrain_parser.set_defaults(run=pretrain)

  for name, function, help_text in (("selfplay", selfplay, "the student answers questions itself, for the teacher to score"),
                                    ("evalloops", evalloops, "measure loops on development questions from the frozen split manifest")):
    sub = commands.add_parser(name, help=help_text)
    sub.add_argument("--load", type=Path, default=STUDENT_FILE)
    sub.add_argument("--questions", type=int, default=16000 if name == "selfplay" else 200)
    sub.add_argument("--round", type=int, default=1)
    sub.add_argument("--batch-size", type=int, default=256 if name == "selfplay" else 50)
    sub.add_argument("--save-every", type=int, default=2048)
    sub.add_argument("--max-new-tokens", type=int, default=200)
    sub.add_argument("--temperature", type=float, default=0.8 if name == "selfplay" else 0.4)
    sub.add_argument("--top-k", type=int, default=40 if name == "selfplay" else 20)
    sub.add_argument("--show", type=int, default=3)
    sub.add_argument("--no-repeat", type=int, default=0, help="block repeating any phrase of this many tokens within an answer; 0 = off")
    sub.add_argument("--seed", type=int, default=7)
    sub.add_argument("--device", choices=devices, default="auto")
    sub.set_defaults(run=function)

  repair_parser = commands.add_parser("repair", help="the teacher finishes self-play answers from just before their first repeat (vLLM)")
  repair_parser.add_argument("--round", type=int, default=1)
  repair_parser.add_argument("--source-round", type=int, default=1, help="which self-play round's answers to repair")
  repair_parser.add_argument("--questions", type=int, default=16000)
  repair_parser.add_argument("--batch-size", type=int, default=1024)
  repair_parser.add_argument("--max-new-tokens", type=int, default=256)
  repair_parser.add_argument("--temperature", type=float, default=0.7)
  repair_parser.add_argument("--top-p", type=float, default=0.8)
  repair_parser.add_argument("--top-k", type=int, default=20)
  repair_parser.add_argument("--gpu-memory", type=float, default=0.75)
  repair_parser.add_argument("--seed", type=int, default=7)
  repair_parser.set_defaults(run=repair)

  loop_parser = commands.add_parser("loop", help="repeat ask -> teach -> train")
  loop_parser.add_argument("--rounds", type=int, default=3)
  loop_parser.add_argument("--steps", type=int, default=1000, help="training steps per round")
  loop_parser.add_argument("--teacher-embeddings", action="store_true", help="passed to train when the student is created")
  loop_parser.add_argument("--seed", type=int, default=7)
  loop_parser.set_defaults(run=loop)

  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  args.run(args)
