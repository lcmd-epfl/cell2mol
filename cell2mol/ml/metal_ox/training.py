"""Training / evaluation / single-graph prediction (cell2mol-free)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from .constants import CLASS_TO_OX, OX_STATES


def compute_class_weights(graphs: list[HeteroData]) -> torch.Tensor:
    """Inverse-frequency class weights for `F.cross_entropy`; classes absent
    from `graphs` get weight 0."""
    counts = torch.zeros(len(OX_STATES))

    for g in graphs:
        counts[int(g["metal"].y.item())] += 1

    weights = torch.zeros(len(OX_STATES))
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])

    return weights


def mask_charges(
    data: HeteroData, charge_columns: dict[str, tuple[int, ...]], p: float
) -> HeteroData:
    """
    Charge dropout: in each graph of the batch, with probability `p`, zero
    every column in `charge_columns` (Variant.charge_columns) on all its
    nodes. That is exactly what a structure whose cell2mol charge assignment
    failed looks like. Training records only come from structures where it
    succeeded, so without this the model never sees has_charge=0 and is out
    of distribution on the failed structures it's most needed for. Whole
    graphs are masked because assignment fails per structure, not per atom.
    Modifies the collated batch in place (it holds copies, not the dataset's
    tensors).
    """
    num_graphs = data["metal"].x.shape[0]
    dropped = torch.rand(num_graphs, device=data["metal"].x.device) < p

    for node_type, cols in charge_columns.items():
        store = data[node_type]
        if store.num_nodes == 0:
            continue
        batch = (
            store.batch
            if "batch" in store
            else torch.zeros(store.num_nodes, dtype=torch.long, device=store.x.device)
        )
        rows = dropped[batch].nonzero(as_tuple=True)[0]
        store.x[rows.unsqueeze(1), torch.tensor(cols, device=store.x.device)] = 0.0

    return data


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    transform: Callable[[HeteroData], HeteroData] | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        if transform is not None:
            data = transform(data)

        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), data["metal"].y, weight=class_weights)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    transform: Callable[[HeteroData], HeteroData] | None = None,
) -> dict[str, Any]:
    """Aggregate metrics (loss, accuracy, macro-F1) plus per-sample
    true/predicted class, confidence and probabilities, in dataset order
    (requires shuffle=False)."""
    model.eval()

    total_loss = 0.0
    y_true, y_pred, confidence, probabilities = [], [], [], []

    for data in loader:
        data = data.to(device)
        if transform is not None:
            data = transform(data)

        logits = model(data)
        y = data["metal"].y

        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)

        total_loss += float(F.cross_entropy(logits, y).item())
        y_true.extend(y.tolist())
        y_pred.extend(pred.tolist())
        confidence.extend(probs.gather(1, pred.unsqueeze(1)).squeeze(1).tolist())
        probabilities.extend(probs.tolist())

    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    macro_f1 = (
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(OX_STATES))),
            average="macro",
            zero_division=0,
        )
        if total
        else 0.0
    )

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(macro_f1),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    checkpoint_path: str | None = None,
    checkpoint_meta: dict | None = None,
    class_weights: torch.Tensor | None = None,
    train_transform: Callable[[HeteroData], HeteroData] | None = None,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    early_stopping_patience: int = 15,
) -> nn.Module:
    """AdamW + ReduceLROnPlateau(val_loss); checkpoints on best val macro-F1,
    early-stops on val_loss. `train_transform` (e.g. charge dropout) applies
    to training batches only."""
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, class_weights, train_transform
        )
        val = evaluate_with_predictions(model, val_loader, device)

        scheduler.step(val["loss"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | "
            f"val_acc={val['accuracy']:.4f} | "
            f"val_f1={val['macro_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        best_val_acc = max(best_val_acc, val["accuracy"])

        if checkpoint_path and val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": best_val_f1,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    **(checkpoint_meta or {}),
                },
                checkpoint_path,
            )

        if val["loss"] < best_val_loss - 1e-4:
            best_val_loss = val["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val_loss improvement for {early_stopping_patience} epochs)"
            )
            break

    return model


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    """Rows = true class, columns = predicted class, indexed by OX_STATES order."""
    n = len(OX_STATES)
    matrix = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1

    return matrix


def print_confusion_matrix(y_true: list[int], y_pred: list[int]) -> None:
    matrix = confusion_matrix(y_true, y_pred)

    print("true\\pred".ljust(10) + "".join(f"{ox:>6d}" for ox in OX_STATES))
    for i, ox in enumerate(OX_STATES):
        print(
            f"{ox:<10d}" + "".join(f"{matrix[i, j]:>6d}" for j in range(len(OX_STATES)))
        )


@torch.no_grad()
def predict_oxidation_state(
    model: nn.Module,
    graph: HeteroData,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probs = torch.softmax(model(graph.to(device)), dim=-1)

    pred_class = int(probs.argmax(dim=-1).item())

    return {
        "oxidation_state": CLASS_TO_OX[pred_class],
        "class_index": pred_class,
        "confidence": float(probs[0, pred_class].item()),
        "probabilities": {
            CLASS_TO_OX[i]: float(probs[0, i].item()) for i in range(probs.shape[-1])
        },
    }
