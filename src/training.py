"""Triplet-loss fine-tuning for sentence-transformer bi-encoders."""

from dataclasses import asdict, dataclass
import random
import numpy as np


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    margin: float = 0.2
    seed: int = 42
    max_grad_norm: float = 1.0

    def validate(self):
        if min(self.epochs, self.batch_size) < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0 or self.margin < 0:
            raise ValueError("learning_rate must be positive and margin non-negative.")
        return self

    def to_dict(self):
        return asdict(self)


def train_triplet_model(model, triplets, config=TrainingConfig(), device=None):
    """Fine-tune in-place and return epoch-level loss history.

    Imports torch lazily so data/evaluation utilities remain lightweight.
    """
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader
    from .triplets import TripletDataset

    config.validate()
    if triplets.empty:
        raise ValueError("No triplets supplied.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model.to(device)
    loader = DataLoader(
        TripletDataset(triplets), batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed), num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def embed(texts):
        features = model.tokenize(list(texts))
        features = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in features.items()}
        return functional.normalize(model(features)["sentence_embedding"], p=2, dim=1)

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            query, positive, negative = embed(batch["query"]), embed(batch["positive"]), embed(batch["negative"])
            positive_score = (query * positive).sum(dim=1)
            negative_score = (query * negative).sum(dim=1)
            loss = functional.relu(config.margin + negative_score - positive_score).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                           error_if_nonfinite=True)
            optimizer.step()
            batch_size = len(batch["query"])
            total += loss.item() * batch_size
            seen += batch_size
        history.append({"epoch": epoch, "average_loss": total / seen})
    model.eval()
    return history
