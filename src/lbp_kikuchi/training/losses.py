import torch
import torch.nn.functional as F


def strain_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over all 6 Voigt strain components."""
    return F.mse_loss(pred, target)


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
) -> torch.Tensor:
    """
    Combined loss for models that predict strain and (optionally) orientation.

    Args:
        outputs:                dict with keys 'strain' and optionally 'orientation'.
        targets:                dict with matching keys.
        orientation_loss_weight: Scale factor for the orientation term.

    Returns:
        Scalar total loss.
    """
    loss = strain_loss(outputs["strain"], targets["strain"])

    if "orientation" in outputs and "orientation" in targets:
        loss = loss + orientation_loss_weight * quaternion_geodesic_loss(
            outputs["orientation"], targets["orientation"]
        )

    return loss
