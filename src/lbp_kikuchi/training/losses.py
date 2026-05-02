import torch
import torch.nn.functional as F

_VALID_LOSS_FNS = ("huber", "mae", "mse")


def strain_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_fn: str = "huber",
    delta: float = 0.5,
) -> torch.Tensor:
    """
    Regression loss over all 6 Voigt strain components.

    Args:
        pred, target: (B, 6) tensors.
        loss_fn:      One of 'huber' (Smooth-L1), 'mae' (L1), 'mse' (L2).
        delta:        Huber transition point; only used when loss_fn='huber'.
    """
    if loss_fn == "huber":
        return F.huber_loss(pred, target, delta=delta)
    elif loss_fn == "mae":
        return F.l1_loss(pred, target)
    elif loss_fn == "mse":
        return F.mse_loss(pred, target)
    else:
        raise ValueError(
            f"Unknown loss_fn {loss_fn!r}. Choose from {_VALID_LOSS_FNS}."
        )


def quaternion_geodesic_loss(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    """
    Geodesic (angular) distance on the unit quaternion sphere S³.

    Handles the q / -q double-cover: we take |dot product| before acos,
    so both representations of the same rotation give zero loss.

    Args:
        q_pred, q_true: (B, 4) unit quaternions
    Returns:
        Scalar mean angular error in radians.
    """
    dot = (q_pred * q_true).sum(dim=-1).abs().clamp(0.0, 1.0)
    return (2.0 * torch.acos(dot)).mean()


def combined_loss(
    outputs: dict,
    targets: dict,
    orientation_loss_weight: float = 1.0,
    loss_fn: str = "huber",
    delta: float = 0.5,
) -> torch.Tensor:
    """
    Combined loss for models that predict strain and (optionally) orientation.

    Args:
        outputs:                 dict with keys 'strain' and optionally 'orientation'.
        targets:                 dict with matching keys.
        orientation_loss_weight: Scale factor for the orientation term.
        loss_fn:                 Strain loss type — 'huber', 'mae', or 'mse'.
        delta:                   Huber delta; ignored for mae/mse.
    """
    loss = strain_loss(outputs["strain"], targets["strain"], loss_fn=loss_fn, delta=delta)

    if "orientation" in outputs and "orientation" in targets:
        loss = loss + orientation_loss_weight * quaternion_geodesic_loss(
            outputs["orientation"], targets["orientation"]
        )

    return loss
