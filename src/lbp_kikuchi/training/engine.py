import torch
from lbp_kikuchi.training.losses import combined_loss
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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, targets in loader:
        X = X.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        outputs = model(X)
        loss = combined_loss(outputs, targets, orientation_loss_weight)

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
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, targets in loader:
        X = X.to(device)
        targets = _to_device(targets, device)

        outputs = model(X)
        total_loss += combined_loss(outputs, targets, orientation_loss_weight).item()
        all_preds.append({k: v.cpu() for k, v in outputs.items()})
        all_targets.append({k: v.cpu() for k, v in targets.items()})

    metrics = compute_all_metrics(_cat_dicts(all_preds), _cat_dicts(all_targets))
    metrics["loss"] = total_loss / len(loader)
    return metrics
