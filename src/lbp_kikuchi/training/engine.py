import torch
from tqdm import tqdm

from lbp_kikuchi.training.losses import combined_loss, pair_loss, strain_loss
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
    epoch: int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    bar = tqdm(loader, desc=f"Train E{epoch}", unit="batch", leave=False)
    for X, targets in bar:
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
        bar.set_postfix(loss=f"{loss.item():.4f}")
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
    epoch: int = 0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    bar = tqdm(loader, desc=f"Val   E{epoch}", unit="batch", leave=False)
    for X, targets in bar:
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
    sv_weight: float = 0.1,
    bounds_weight: float = 0.01,
    max_abs_strain: float = 0.05,
    epoch: int = 0,
) -> dict[str, float]:
    """
    One training epoch for PairModel.

    Loader must yield (grid_a, grid_b, targets_dict, pos_a, pos_b) batches
    as produced by GridPairDataset.  The loss is the sum of:
        - regression loss on Δε
        - Saint-Venant loop-consistency loss (weighted by sv_weight)
        - physical-bounds penalty       (weighted by bounds_weight)
    """
    model.train()
    total_loss = 0.0
    total_sv = 0.0
    total_bounds = 0.0
    all_preds, all_targets = [], []

    bar = tqdm(loader, desc=f"Train E{epoch}", unit="batch", leave=False)
    for grid_a, grid_b, targets, pos_a, pos_b in bar:
        grid_a  = grid_a.to(device, non_blocking=True)
        grid_b  = grid_b.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        pos_a   = pos_a.to(device, non_blocking=True)
        pos_b   = pos_b.to(device, non_blocking=True)

        outputs = model(grid_a, grid_b)
        loss, components = pair_loss(
            outputs["strain"], targets["strain"],
            pos_a, pos_b,
            loss_fn=loss_fn, delta=huber_delta,
            sv_weight=sv_weight,
            bounds_weight=bounds_weight,
            max_abs_strain=max_abs_strain,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        total_sv     += components["loss_sv"]
        total_bounds += components["loss_bounds"]
        bar.set_postfix(loss=f"{loss.item():.4f}", sv=f"{components['loss_sv']:.4f}")
        all_preds.append({k: v.detach().cpu() for k, v in outputs.items()})
        all_targets.append({k: v.cpu() for k, v in targets.items()})

    n = len(loader)
    metrics = compute_all_metrics(_cat_dicts(all_preds), _cat_dicts(all_targets))
    metrics["loss"]         = total_loss   / n
    metrics["loss_sv"]      = total_sv     / n
    metrics["loss_bounds"]  = total_bounds / n
    return metrics


@torch.no_grad()
def evaluate_pair(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: str = "huber",
    huber_delta: float = 0.5,
    sv_weight: float = 0.1,
    bounds_weight: float = 0.01,
    max_abs_strain: float = 0.05,
    epoch: int = 0,
) -> dict[str, float]:
    """Validation / test evaluation for PairModel."""
    model.eval()
    total_loss = 0.0
    total_sv = 0.0
    total_bounds = 0.0
    all_preds, all_targets = [], []

    bar = tqdm(loader, desc=f"Val   E{epoch}", unit="batch", leave=False)
    for grid_a, grid_b, targets, pos_a, pos_b in bar:
        grid_a  = grid_a.to(device)
        grid_b  = grid_b.to(device)
        targets = _to_device(targets, device)
        pos_a   = pos_a.to(device)
        pos_b   = pos_b.to(device)

        outputs = model(grid_a, grid_b)
        loss, components = pair_loss(
            outputs["strain"], targets["strain"],
            pos_a, pos_b,
            loss_fn=loss_fn, delta=huber_delta,
            sv_weight=sv_weight,
            bounds_weight=bounds_weight,
            max_abs_strain=max_abs_strain,
        )
        total_loss   += loss.item()
        total_sv     += components["loss_sv"]
        total_bounds += components["loss_bounds"]
        all_preds.append({k: v.cpu() for k, v in outputs.items()})
        all_targets.append({k: v.cpu() for k, v in targets.items()})

    n = len(loader)
    metrics = compute_all_metrics(_cat_dicts(all_preds), _cat_dicts(all_targets))
    metrics["loss"]         = total_loss   / n
    metrics["loss_sv"]      = total_sv     / n
    metrics["loss_bounds"]  = total_bounds / n
    return metrics
