import torch
from lbp_kikuchi.training.losses import combined_loss, strain_loss
from lbp_kikuchi.training.metrics import compute_all_metrics


def _to_device(targets: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}


def _cat_dicts(list_of_dicts: list[dict]) -> dict:
    return {k: torch.cat([d[k] for d in list_of_dicts]) for k in list_of_dicts[0]}


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    orientation_loss_weight: float = 1.0,
    loss_fn: str = "huber",
    huber_delta: float = 0.5,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, targets in loader:
        X = X.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        outputs = model(X)
        loss = combined_loss(
            outputs, targets,
            orientation_loss_weight=orientation_loss_weight,
            loss_fn=loss_fn,
            delta=huber_delta,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append({k: v.detach().cpu() for k, v in outputs.items()})
        all_targets.append({k: v.cpu() for k, v in targets.items()})

    metrics = compute_all_metrics(_cat_dicts(all_preds), _cat_dicts(all_targets))
    metrics["loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    orientation_loss_weight: float = 1.0,
    loss_fn: str = "huber",
    huber_delta: float = 0.5,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, targets in loader:
        X = X.to(device)
        targets = _to_device(targets, device)

        outputs = model(X)
        total_loss += combined_loss(
            outputs, targets,
            orientation_loss_weight=orientation_loss_weight,
            loss_fn=loss_fn,
            delta=huber_delta,
        ).item()
        all_preds.append({k: v.cpu() for k, v in outputs.items()})
        all_targets.append({k: v.cpu() for k, v in targets.items()})

    metrics = compute_all_metrics(_cat_dicts(all_preds), _cat_dicts(all_targets))
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Stage 3 pair engine (PairModel: grid_a, grid_b → Δε)
# ---------------------------------------------------------------------------


def train_one_epoch_pair(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: str = "huber",
    huber_delta: float = 0.5,
) -> dict[str, float]:
    """
    One training epoch for PairModel.

    Loader must yield (grid_a, grid_b, targets_dict) batches as produced by
    GridPairDataset. Only strain is predicted (no orientation term for pairs).
    """
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for grid_a, grid_b, targets in loader:
        grid_a = grid_a.to(device, non_blocking=True)
        grid_b = grid_b.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        outputs = model(grid_a, grid_b)
        loss = strain_loss(outputs["strain"], targets["strain"],
                           loss_fn=loss_fn, delta=huber_delta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append({k: v.detach().cpu() for k, v in outputs.items()})
        all_targets.append({k: v.cpu() for k, v in targets.items()})

    metrics = compute_all_metrics(_cat_dicts(all_preds), _cat_dicts(all_targets))
    metrics["loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def evaluate_pair(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: str = "huber",
    huber_delta: float = 0.5,
) -> dict[str, float]:
    """Validation / test evaluation for PairModel."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for grid_a, grid_b, targets in loader:
        grid_a = grid_a.to(device)
        grid_b = grid_b.to(device)
        targets = _to_device(targets, device)

        outputs = model(grid_a, grid_b)
        total_loss += strain_loss(
            outputs["strain"], targets["strain"],
            loss_fn=loss_fn, delta=huber_delta,
        ).item()
        all_preds.append({k: v.cpu() for k, v in outputs.items()})
        all_targets.append({k: v.cpu() for k, v in targets.items()})

    metrics = compute_all_metrics(_cat_dicts(all_preds), _cat_dicts(all_targets))
    metrics["loss"] = total_loss / len(loader)
    return metrics
