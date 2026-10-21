"""Train a small neural network to identify a file from a source-code span."""

# uv run python neural-lite.py --recursive --epochs 20 --samples-per-epoch 8192 --validation-samples 2048 --batch-size 128 --span-length 256 --embedding-size 48 --hidden-size 96 --learning-rate 1e-3 --seed 7 --output filename_classifier.pt

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

def load_source_files(directory: Path, recursive: bool = False) -> tuple[list[str], list[bytes]]:
  """Return filenames and non-empty Python source files from directory."""

  paths = directory.rglob("*.py") if recursive else directory.glob("*.py")
  paths = sorted(path for path in paths if path.is_file())

  # could use list comprehension like sources = [path.read_bytes() for path in paths if path.read_bytes()]
  names = []
  sources = []
  for path in paths:
    source = path.read_bytes()
    if source:
      names.append(str(path.relative_to(directory)))
      sources.append(source)

  if len(sources) < 2:
    raise ValueError("At least two non-empty .py files are needed for classification")

  return names, sources

def random_span(source: bytes, span_length: int) -> list[int]:
  """Choose a random span and convert its bytes to tokens, padding with token zero."""

  if len(source) >= span_length:
    start = random.randint(0, len(source) - span_length) # [0, |source| - |span_length|]
    span = source[start:start + span_length]
  else:
    span = source

  return [byte + 1 for byte in span] + [0] * (span_length - len(span))

class SourceSpanDataset(Dataset):
  def __init__(self, sources: list[bytes], span_length: int, sample_count: int):
    self.sources = sources
    self.span_length = span_length
    self.sample_count = sample_count

  def __len__(self) -> int:
    return self.sample_count

  def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
    label = random.randrange(len(self.sources))
    tokens = random_span(self.sources[label], self.span_length)
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(label)

class FilenameClassifier(nn.Module):
  def __init__(self, file_count: int, embedding_size: int = 48, hidden_size: int = 96):
    super().__init__()

    # embedding layers map each byte value (0-256) to a vector of size embedding_size
    self.embedding = nn.Embedding(257, embedding_size, padding_idx=0)

    # the classifier is a simple feedforward network with one hidden layer (2 linear layers with ReLU activation)
    self.layers = nn.Sequential(
        nn.Linear(embedding_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, file_count),
    )

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    # embed the tokens, mask out the padding tokens, and average the embeddings to get a single vector for the span
    embedded = self.embedding(tokens)
    mask = tokens.ne(0).unsqueeze(-1)
    pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return self.layers(pooled)

def choose_device() -> torch.device:
  if torch.backends.mps.is_available():
    print("Metal/MPS is available; using MPS.")
    return torch.device("mps")
  else:
    print("Metal/MPS is NOT available; using CPU instead.")
    return torch.device("cpu")

# need this to interface with PyTorch's DataLoader, which expects a Dataset object
def make_loader(sources: list[bytes], span_length: int, sample_count: int, batch_size: int) -> DataLoader:
  dataset = SourceSpanDataset(sources, span_length, sample_count)
  return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# returns the average loss and accuracy for a single epoch of training
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,) -> tuple[float, float]:
  model.train() # set training mode, subsequent things will be tracked for gradient computation
  loss_function = nn.CrossEntropyLoss() # use cross entropy loss for multi-class classification
  total_loss = 0.0
  correct = 0
  total = 0

  for tokens, labels in loader:
    tokens, labels = tokens.to(device), labels.to(device) # send the data to the device (CPU or GPU)
    optimizer.zero_grad() # clear the gradients from the previous step
    logits = model(tokens)

    loss = loss_function(logits, labels) # compute the loss for the current batch
    loss.backward() # backpropagate the loss to compute gradients for each parameter
    optimizer.step() # update the model parameters using the computed gradients

    # accumulate the loss and accuracy for reporting
    total_loss += loss.item() * labels.size(0)
    correct += (logits.argmax(dim=1) == labels).sum().item()
    total += labels.size(0)

  return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
  model.eval()

  correct = 0
  total = 0

  for tokens, labels in loader:
    tokens, labels = tokens.to(device), labels.to(device)
    predictions = model(tokens).argmax(dim=1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

  return correct / total

def train(args: argparse.Namespace) -> None:
  random.seed(args.seed)
  torch.manual_seed(args.seed)

  names, sources = load_source_files(args.directory, args.recursive)
  device = choose_device()

  model = FilenameClassifier(len(names), args.embedding_size, args.hidden_size).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  train_loader = make_loader(sources, args.span_length, args.samples_per_epoch, args.batch_size)
  validation_loader = make_loader(sources, args.span_length, args.validation_samples, args.batch_size)

  ## START TRAINING

  print(f"Training on {len(names)} files with device={device}")

  for epoch in range(1, args.epochs + 1):
    loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
    validation_accuracy = evaluate(model, validation_loader, device)

    print(f"epoch {epoch:>2}: loss={loss:.4f}, "f"train_accuracy={train_accuracy:.1%}, validation_accuracy={validation_accuracy:.1%}")

  ## END OF TRAINING

  # save the model state and metadata to a file
  torch.save({"model_state": model.state_dict(), "filenames": names, "span_length": args.span_length, "embedding_size": args.embedding_size, "hidden_size": args.hidden_size,}, args.output,)
  print(f"Saved model to {args.output}")

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent)
  parser.add_argument("--recursive", action="store_true", help="include Python files in subdirectories")
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--samples-per-epoch", type=int, default=4096)
  parser.add_argument("--validation-samples", type=int, default=1024)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--span-length", type=int, default=256)
  parser.add_argument("--embedding-size", type=int, default=48)
  parser.add_argument("--hidden-size", type=int, default=96)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--output", type=Path, default=Path("filename_classifier.pt"))
  return parser.parse_args()

if __name__ == "__main__":
  train(parse_args())
