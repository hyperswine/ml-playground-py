"""Run the official Qwen3.5 0.8B model locally with Hugging Face Transformers.

Examples:
  uv run python qwen08.py "Write a Python function that checks for primes."
  uv run python qwen08.py --interactive
  uv run python qwen08.py --device mps --thinking "Solve 17 * 23."
"""

import argparse
import time
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "Briefly explain why the sky is blue."


def choose_device(requested: str) -> torch.device:
  if requested != "auto":
    return torch.device(requested)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
  if device.type == "mps":
    return torch.float32
  if device.type == "cuda" and torch.cuda.is_bf16_supported():
    return torch.bfloat16
  if device.type == "cuda":
    return torch.float16
  return torch.float32


def synchronize(device: torch.device) -> None:
  if device.type == "cuda":
    torch.cuda.synchronize()
  elif device.type == "mps":
    torch.mps.synchronize()


def generate_response(
  model: Any,
  processor: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
) -> str:
  inputs = processor.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      return_dict=True,
      return_tensors="pt",
      enable_thinking=args.thinking,
  )
  inputs = inputs.to(device)
  input_token_count = inputs["input_ids"].shape[1]

  synchronize(device)
  started_at = time.perf_counter()

  with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

  synchronize(device)
  elapsed = time.perf_counter() - started_at
  generated_ids = output_ids[:, input_token_count:]
  generated_token_count = generated_ids.shape[1]
  response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

  print(f"[{generated_token_count} tokens in {elapsed:.2f}s, {generated_token_count / elapsed:.1f} tokens/s]")
  return response.strip()


def run(args: argparse.Namespace) -> None:
  if args.temperature <= 0:
    raise ValueError("temperature must be greater than zero")

  torch.manual_seed(args.seed)
  device = choose_device(args.device)
  dtype = choose_dtype(device)
  print(f"Loading {args.model} on {device} with {dtype}...")
  print("The first run downloads and caches the model from Hugging Face.")

  processor: Any = AutoProcessor.from_pretrained(
      args.model,
      cache_dir=args.cache_dir,
      local_files_only=args.local_files_only,
  )
  model: Any = Qwen3_5ForConditionalGeneration.from_pretrained(
      args.model,
      dtype=dtype,
      cache_dir=args.cache_dir,
      local_files_only=args.local_files_only,
  )
  model.to(device)
  model.eval()

  messages: list[dict[str, str]] = []
  if args.system:
    messages.append({"role": "system", "content": args.system})

  if args.prompt is not None or not args.interactive:
    prompt = args.prompt or DEFAULT_PROMPT
    messages.append({"role": "user", "content": prompt})
    response = generate_response(model, processor, messages, args, device)
    print(f"\n{response}\n")
    messages.append({"role": "assistant", "content": response})

  if args.interactive:
    print("Interactive mode. Enter /quit to exit.")
    while True:
      try:
        prompt = input("\nYou: ").strip()
      except (EOFError, KeyboardInterrupt):
        print()
        break

      if prompt.lower() in {"/quit", "/exit"}:
        break
      if not prompt:
        continue

      messages.append({"role": "user", "content": prompt})
      response = generate_response(model, processor, messages, args, device)
      print(f"\nQwen: {response}")
      messages.append({"role": "assistant", "content": response})


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("prompt", nargs="?", help="one-shot prompt")
  parser.add_argument("--interactive", action="store_true", help="keep chatting without reloading the model")
  parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local model path")
  parser.add_argument("--system", default="You are a concise, helpful assistant.")
  parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
  parser.add_argument("--max-new-tokens", type=int, default=256)
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top-p", type=float, default=1.0)
  parser.add_argument("--top-k", type=int, default=20)
  parser.add_argument("--repetition-penalty", type=float, default=1.0)
  parser.add_argument("--thinking", action="store_true", help="enable Qwen3.5 thinking mode")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--cache-dir", help="optional Hugging Face model cache directory")
  parser.add_argument("--local-files-only", action="store_true", help="do not download missing model files")
  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args())