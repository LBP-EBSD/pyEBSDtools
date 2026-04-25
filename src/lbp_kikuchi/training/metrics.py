import torch

VOIGT_NAMES = ["e11", "e22", "e33", "e23", "e13", "e12"]


def strain_mae_per_component(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """
    Per-component MAE for Voigt strain [ε11, ε22, ε33, ε23, ε13, ε12].

    Returns: {'mae_e11': ..., 'mae_e22': ..., ..., 'mae_e12': ...}
    """
    mae = (pred - target).abs().mean(dim=0)   # (6,)
    return {f"mae_{name}": mae[i].item() for i, name in enumerate(VOIGT_NAMES)}


def strain_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error across all Voigt components and samples."""
    return ((pred - target) ** 2).mean().sqrt().item()


def strain_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error across all Voigt components and samples."""
    return (pred - target).abs().mean().item()


def quaternion_angular_error_deg(
    q_pred: torch.Tensor, q_true: torch.Tensor
) -> float:
    """
    Mean angular error (in degrees) between predicted and true quaternions.
    Handles the q / -q double-cover.

    q_pred, q_true: (B, 4) unit quaternions
    """
    dot = (q_pred * q_true).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle_rad = 2.0 * torch.acos(dot)
    return torch.rad2deg(angle_rad).mean().item()


def compute_all_metrics(outputs: dict, targets: dict) -> dict[str, float]:
    """
    Compute all relevant metrics from accumulated predictions and targets.

    Args:
        outputs: dict with keys 'strain' (and optionally 'orientation'),
                 each containing a full-epoch tensor of predictions.
        targets: dict with the same keys, containing ground truth.

    Returns:
        Flat dict mapping metric name → scalar float.
    """
    metrics: dict[str, float] = {}

    if "strain" in outputs and "strain" in targets:
        pred_s, true_s = outputs["strain"], targets["strain"]
        metrics["strain_rmse"] = strain_rmse(pred_s, true_s)
        metrics["strain_mae"] = strain_mae(pred_s, true_s)
        metrics.update(strain_mae_per_component(pred_s, true_s))

    if "orientation" in outputs and "orientation" in targets:
        metrics["orientation_error_deg"] = quaternion_angular_error_deg(
            outputs["orientation"], targets["orientation"]
        )

    return metrics
