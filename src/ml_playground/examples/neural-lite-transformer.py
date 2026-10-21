"""Train a tiny decoder-only transformer and generate a text continuation.

Examples:
  uv run python neural-lite-transformer.py --steps 1000 --generate-length 50 --device mps
  uv run python neural-lite-transformer.py --load text_transformer.pt --prompt "def "
  uv run python neural-lite-transformer.py --steps 1000 --generate-length 50 --prompt "from __future__ import " --load text_transformer.pt
"""

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def load_text_corpus(directory: Path) -> tuple[str, list[str]]:
  """Recursively combine non-empty UTF-8 text files, skipping binary files."""

  if not directory.is_dir():
    raise ValueError(f"Training directory does not exist: {directory}")

  documents: list[str] = []
  filenames: list[str] = []

  for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
    try:
      text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
      continue

    if not text.strip() or "\x00" in text:
      continue

    relative_name = path.relative_to(directory).as_posix()
    documents.append(f"\n\n# File: {relative_name}\n{text}")
    filenames.append(relative_name)

  if not documents:
    raise ValueError(f"No non-empty UTF-8 text files found under {directory}")

  return "".join(documents), filenames


class CharacterTokenizer:
  def __init__(self, characters: list[str]):
    self.characters = characters
    self.char_to_id = {character: index for index, character in enumerate(characters)} # this basically creates a dictionary that maps each character to its index in the list of characters

  @classmethod
  def from_text(cls, text: str) -> "CharacterTokenizer":
    return cls(sorted(set(text)))

  def encode(self, text: str) -> list[int]:
    unknown = sorted(set(text).difference(self.char_to_id))

    if unknown:
      raise ValueError(f"Prompt contains characters outside the training vocabulary: {unknown!r}")
    else:
      return [self.char_to_id[character] for character in text]

  def decode(self, token_ids: list[int]) -> str:
    return "".join(self.characters[token_id] for token_id in token_ids)

  def __len__(self) -> int:
    return len(self.characters)


@dataclass(frozen=True)
class ModelConfig:
  vocabulary_size: int
  context_length: int = 128
  embedding_size: int = 64 # size of token embeddings
  head_count: int = 4 # number of attention heads
  layer_count: int = 2


class CausalSelfAttention(nn.Module):
  def __init__(self, config: ModelConfig):

    super().__init__()

    if config.embedding_size % config.head_count != 0:
      raise ValueError("embedding-size must be divisible by heads")

    self.head_count = config.head_count
    self.head_size = config.embedding_size // config.head_count

    # basically, two linear layers, one for query/key/value and one for output
    self.query_key_value = nn.Linear(config.embedding_size, 3 * config.embedding_size)
    self.output = nn.Linear(config.embedding_size, config.embedding_size)

    self.register_buffer("causal_mask", torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    batch_size, token_count, embedding_size = inputs.shape

    # this just computes the query, key, and value matrices for the attention mechanism. It does this by applying a linear transformation to the input tensor
    query_key_value = self.query_key_value(inputs)
    query_key_value = query_key_value.view(batch_size, token_count, 3, self.head_count, self.head_size).permute(2, 0, 3, 1, 4)
    # Q, K, V matrices. Q is of dim (X, Y), K is of dim (Y, Z), and V is of dim (Z, W). The output of the attention mechanism is of dim (X, W)
    query, key, value = query_key_value.unbind(dim=0)

    # first, we compute the attention scores by taking the dot product of the query and key matrices, and then scaling by the square root of the head size.
    attention_scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
    # then, we apply a causal mask to the attention scores to prevent attending to future tokens. This is done by setting the attention scores for future tokens to negative infinity, which will result in a zero probability after applying the softmax function.
    attention_scores = attention_scores.masked_fill(~self.causal_mask[:token_count, :token_count], float("-inf")) # type: ignore
    # finally, we apply the softmax function to the attention scores to obtain the attention weights, and then compute the attended values by taking the weighted sum of the value matrix.
    attention_weights = F.softmax(attention_scores, dim=-1)

    # compute the attended values by taking the weighted sum of the value matrix
    attended = attention_weights @ value
    # note, contiguous() is used to ensure that the tensor is stored in a contiguous block of memory and view() is used to reshape the tensor to the original shape of the input tensor.
    attended = attended.transpose(1, 2).contiguous().view(batch_size, token_count, embedding_size)

    return self.output(attended)


class DecoderBlock(nn.Module):
  def __init__(self, config: ModelConfig):

    super().__init__()

    self.attention_norm = nn.LayerNorm(config.embedding_size)
    self.attention = CausalSelfAttention(config) # this is the big deal that allows us to attend to all previous tokens in the sequence
    self.feed_forward_norm = nn.LayerNorm(config.embedding_size)

    # main thinking of the transformer is in the feed forward network, which is a simple two-layer MLP
    # can be thought of as the Key-Value store, where the first layer is the key and the second layer is the value. The first layer maps the input to a higher dimensional space, and the second layer maps it back to the original space. The GELU activation function is used to introduce non-linearity.
    self.feed_forward = nn.Sequential(
        nn.Linear(config.embedding_size, 4 * config.embedding_size),
        nn.GELU(),
        nn.Linear(4 * config.embedding_size, config.embedding_size),
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    inputs = inputs + self.attention(self.attention_norm(inputs))

    return inputs + self.feed_forward(self.feed_forward_norm(inputs))


class TinyDecoderTransformer(nn.Module):
  def __init__(self, config: ModelConfig):

    super().__init__()

    self.config = config
    self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_size)
    self.position_embedding = nn.Embedding(config.context_length, config.embedding_size)

    # this creates a sequence of decoder blocks, each of which contains a self-attention mechanism and a feed-forward network. The number of decoder blocks is determined by the layer_count parameter in the ModelConfig.
    self.blocks = nn.Sequential(*(DecoderBlock(config) for _ in range(config.layer_count)))

    self.final_norm = nn.LayerNorm(config.embedding_size)
    self.language_model_head = nn.Linear(config.embedding_size, config.vocabulary_size)

  def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:

    token_count = tokens.size(1)

    if token_count > self.config.context_length:
      raise ValueError(f"Input exceeds context length {self.config.context_length}")

    positions = torch.arange(token_count, device=tokens.device)
    hidden = self.token_embedding(tokens) + self.position_embedding(positions)
    hidden = self.final_norm(self.blocks(hidden))
    logits = self.language_model_head(hidden)

    loss = None

    if targets is not None:
      # standard cross-entropy loss for language modeling, where the logits are flattened to a 2D tensor targets are flattened to a 1D tensor
      # loss is computed by comparing the predicted logits with the true target tokens.
      loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    return logits, loss

  @torch.no_grad()
  def generate(self, tokens: torch.Tensor, new_token_count: int, temperature: float = 0.8, top_k: int = 20,) -> torch.Tensor:

    self.eval()

    for _ in range(new_token_count):
      context = tokens[:, -self.config.context_length:]
      logits, _ = self(context)
      next_logits = logits[:, -1, :] / temperature

      if top_k > 0:
        top_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        next_logits[next_logits < top_values[:, [-1]]] = float("-inf")

      probabilities = F.softmax(next_logits, dim=-1)
      next_token = torch.multinomial(probabilities, num_samples=1)
      tokens = torch.cat((tokens, next_token), dim=1)

    return tokens


def choose_device(requested: str) -> torch.device:
  if requested != "auto":
    return torch.device(requested)
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def random_batch(data: torch.Tensor, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  maximum_start = data.numel() - context_length - 1

  if maximum_start < 0:
    raise ValueError(f"Corpus has {data.numel()} characters but context length is {context_length}")

  starts = torch.randint(maximum_start + 1, (batch_size,))
  inputs = torch.stack([data[start:start + context_length] for start in starts])
  targets = torch.stack([data[start + 1:start + context_length + 1] for start in starts])

  return inputs.to(device), targets.to(device)


def train_model( model: TinyDecoderTransformer, data: torch.Tensor, args: argparse.Namespace, device: torch.device,) -> None:
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

  for step in range(1, args.steps + 1):
    inputs, targets = random_batch(data, args.batch_size, model.config.context_length, device)
    _, loss = model(inputs, targets)
    assert loss is not None

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step == 1 or step % args.report_every == 0 or step == args.steps:
      print(f"step {step:>5}/{args.steps}: loss={loss.item():.4f}")


def save_checkpoint( path: Path, model: TinyDecoderTransformer, tokenizer: CharacterTokenizer, filenames: list[str],) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
      {
          "model_state": model.state_dict(),
          "config": asdict(model.config),
          "characters": tokenizer.characters,
          "filenames": filenames,
      },
      path,
  )


def load_checkpoint(path: Path, device: torch.device) -> tuple[TinyDecoderTransformer, CharacterTokenizer]:
  checkpoint = torch.load(path, map_location=device, weights_only=True)
  config = ModelConfig(**checkpoint["config"])
  tokenizer = CharacterTokenizer(checkpoint["characters"])
  model = TinyDecoderTransformer(config).to(device)
  model.load_state_dict(checkpoint["model_state"])

  return model, tokenizer


def run(args: argparse.Namespace) -> None:
  if args.temperature <= 0:
    raise ValueError("temperature must be greater than zero")

  torch.manual_seed(args.seed)
  device = choose_device(args.device)
  print(f"Using device: {device}")

  if args.load is not None:
    model, tokenizer = load_checkpoint(args.load, device)
    print(f"Loaded model from {args.load}")
  else:
    corpus, filenames = load_text_corpus(args.directory)

    # create a tokenizer
    tokenizer = CharacterTokenizer.from_text(corpus)

    encoded_corpus = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    config = ModelConfig(
        vocabulary_size=len(tokenizer),
        context_length=args.context_length,
        embedding_size=args.embedding_size,
        head_count=args.heads,
        layer_count=args.layers,
    )
    model = TinyDecoderTransformer(config).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(
        f"Training on {len(filenames)} files, {len(corpus):,} characters, "
        f"{len(tokenizer)} unique characters; model has {parameter_count:,} parameters"
    )

    train_model(model, encoded_corpus, args, device)
    save_checkpoint(args.output, model, tokenizer, filenames)

    print(f"Saved model to {args.output}")

  prompt_tokens = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

  generated_tokens = model.generate(
      prompt_tokens,
      args.generate_length,
      temperature=args.temperature,
      top_k=args.top_k,
  )
  print("\nGenerated text:\n" + tokenizer.decode(generated_tokens[0].tolist()))


def parse_args() -> argparse.Namespace:

  parser = argparse.ArgumentParser(description=__doc__)

  parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent / "py1")
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--context-length", type=int, default=128)
  parser.add_argument("--embedding-size", type=int, default=64)
  parser.add_argument("--heads", type=int, default=4)
  parser.add_argument("--layers", type=int, default=2)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--report-every", type=int, default=100)
  parser.add_argument("--prompt", default="def ")
  parser.add_argument("--generate-length", type=int, default=50)
  parser.add_argument("--temperature", type=float, default=0.8)
  parser.add_argument("--top-k", type=int, default=20, help="zero disables top-k sampling")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
  parser.add_argument("--output", type=Path, default=Path("text_transformer.pt"))
  parser.add_argument("--load", type=Path, help="skip training and load this checkpoint")

  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args())
