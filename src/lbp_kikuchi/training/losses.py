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


# ---------------------------------------------------------------------------
# Saint-Venant / physical-compatibility losses for Stage 3 (pair model)
# ---------------------------------------------------------------------------

def physical_bounds_loss(
    delta_pred: torch.Tensor,
    max_abs_strain: float = 0.05,
) -> torch.Tensor:
    """
    Penalise predicted Δε components that exceed physically plausible bounds.

    For elastic deformation of common structural metals the total strain rarely
    exceeds ~2–3 % (0.02–0.03) and the *change* between adjacent scan points
    is even smaller (<<1 %). Any individual Voigt component of Δε that exceeds
    ``max_abs_strain`` is penalised quadratically.

    This acts as a soft constraint that stops the network from cheating its way
    to low MSE by predicting large-magnitude spurious values.

    Args:
        delta_pred:     (B, 6) predicted Δε in **physical** strain units.
                        ``pair_loss`` denormalises network outputs before calling this;
                        do not pass z-scored tensors or ``max_abs_strain`` will be wrong.
        max_abs_strain: Soft threshold in absolute strain units. Default 0.05
                        (5%), which is already beyond the elastic limit for
                        most metals but gives the network some headroom.

    Returns:
        Scalar mean penalty ≥ 0.
    """
    excess = F.relu(delta_pred.abs() - max_abs_strain)
    return (excess ** 2).mean()


def loop_consistency_loss(
    delta_pred: torch.Tensor,
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
) -> torch.Tensor:
    """
    Saint-Venant integrability (loop-consistency) loss for pairwise Δε predictions.

    A strain field ε(r, c) is compatible (satisfies Saint-Venant conditions)
    if and only if it is derivable from a continuous displacement field u(x).
    In the discrete pairwise setting this translates to *path independence*:
    the accumulated Δε along any closed loop must be zero.

    For the simplest rectangular loop on the scan grid:

        (r, c) ──h──► (r, c+1)
          │                │
          v                v
        (r+1,c) ──h──► (r+1,c+1)

    The integrability condition is:

        Δε_h(r,c→r,c+1) + Δε_v(r,c+1→r+1,c+1)
        − Δε_h(r+1,c→r+1,c+1) − Δε_v(r,c→r+1,c)  =  0      [per component]

    This function finds all *complete* rectangles whose four constitutent pairs
    happen to be present in the current batch, then penalises their residuals.

    Horizontal pairs are detected by  pos_b[i] = pos_a[i] + [0, 1].
    Vertical   pairs are detected by  pos_b[i] = pos_a[i] + [1, 0].

    When position is unknown (pos_a == -1, i.e. the fallback from
    GridPairDataset when positions were not stored), the function returns 0.

    Args:
        delta_pred: (B, 6) predicted Δε in **physical** Voigt units (same linear space as
                    ground-truth Δε). Per-component z-scores must not be mixed here or loop
                    residuals are meaningless.
        pos_a:      (B, 2) int tensor — scan position [row, col] of grid A center.
        pos_b:      (B, 2) int tensor — scan position [row, col] of grid B center.

    Returns:
        Scalar mean squared loop residual ≥ 0 (0 when no loops are found).
    """
    if pos_a[0, 0].item() == -1:
        return delta_pred.new_tensor(0.0)

    diff = pos_b - pos_a   # (B, 2)

    # Separate horizontal (Δcol=+1) and vertical (Δrow=+1) pairs.
    h_mask = (diff[:, 0] == 0) & (diff[:, 1] == 1)
    v_mask = (diff[:, 0] == 1) & (diff[:, 1] == 0)

    h_idx = h_mask.nonzero(as_tuple=True)[0]
    v_idx = v_mask.nonzero(as_tuple=True)[0]

    if h_idx.numel() == 0 or v_idx.numel() == 0:
        return delta_pred.new_tensor(0.0)

    # Build lookup dictionaries: (r,c) → batch index, for each direction.
    # We use integer-encoded keys to avoid Python dict overhead on GPU tensors.
    # r * MAX_COLS + c — 10 000 cols is more than enough for any EBSD scan.
    MAX_COLS = 10_000
    h_keys = pos_a[h_idx, 0] * MAX_COLS + pos_a[h_idx, 1]  # (H_cnt,) key = (r,c) of A
    v_keys = pos_a[v_idx, 0] * MAX_COLS + pos_a[v_idx, 1]  # (V_cnt,) key = (r,c) of A

    # For a rectangle rooted at (r,c) we need:
    #   h_bot : horizontal pair (r,   c)   → (r,   c+1)   key = r*MC + c
    #   h_top : horizontal pair (r+1, c)   → (r+1, c+1)   key = (r+1)*MC + c
    #   v_left: vertical   pair (r,   c)   → (r+1, c)     key = r*MC + c
    #   v_rgt : vertical   pair (r,   c+1) → (r+1, c+1)   key = r*MC + (c+1)
    # Root key for h_bot and v_left is the same: r*MC+c.

    # Build Python dicts from tensors (done on CPU for indexing convenience;
    # tensor ops for matching are GPU-friendly if the dicts are small).
    h_dict = {int(k.item()): int(bi.item())
              for k, bi in zip(h_keys, h_idx)}
    v_dict = {int(k.item()): int(bi.item())
              for k, bi in zip(v_keys, v_idx)}

    residuals = []
    for key, bot_h_bi in h_dict.items():
        r = key // MAX_COLS
        c = key % MAX_COLS
        top_h_key = (r + 1) * MAX_COLS + c
        v_l_key   = r * MAX_COLS + c
        v_r_key   = r * MAX_COLS + (c + 1)

        if top_h_key not in h_dict or v_l_key not in v_dict or v_r_key not in v_dict:
            continue

        dh_bot = delta_pred[bot_h_bi]                # Δε_h(r,   c→c+1)
        dh_top = delta_pred[h_dict[top_h_key]]        # Δε_h(r+1, c→c+1)
        dv_l   = delta_pred[v_dict[v_l_key]]          # Δε_v(r,   c→r+1)
        dv_r   = delta_pred[v_dict[v_r_key]]          # Δε_v(r,   c+1→r+1)

        # Loop residual: go right-then-down minus down-then-right.
        # Both paths start at (r,c) and end at (r+1,c+1).
        residual = dh_bot + dv_r - dh_top - dv_l     # (6,)
        residuals.append((residual ** 2).sum())

    if not residuals:
        return delta_pred.new_tensor(0.0)

    return torch.stack(residuals).mean()


def pair_loss(
    delta_pred: torch.Tensor,
    delta_target: torch.Tensor,
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    loss_fn: str = "huber",
    delta: float = 0.5,
    sv_weight: float = 0.1,
    bounds_weight: float = 0.01,
    max_abs_strain: float = 0.05,
    delta_mean: torch.Tensor | None = None,
    delta_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Full Stage-3 loss: regression + Saint-Venant integrability + physical bounds.

    Total loss:
        L = L_regression
          + sv_weight     * L_loop_consistency   (Saint-Venant integrability)
          + bounds_weight * L_physical_bounds    (per-component magnitude clamp)

    Regression is computed on **normalized** Δε (what the network predicts).

    The Saint-Venant and physical-bounds terms must run in **physical** Δε units:
    ``Δε_phys = Δε_norm * std + mean`` (broadcast over batch). If this conversion
    were skipped and ``max_abs_strain`` (e.g. 0.05) were applied to normalized
    tensors, almost every correct prediction would be penalised and the model
    collapses to predicting ~0 in normalized space (flat scatter plots).

    Args:
        delta_pred:     (B, 6) predicted Δε in normalized training space.
        delta_target:   (B, 6) ground-truth Δε (normalized).
        pos_a, pos_b:   (B, 2) scan positions (int32).
        delta_mean:     (6,) tensor — train-split mean of Δε per Voigt component.
        delta_std:      (6,) tensor — train-split std of Δε per component.
                        Required whenever ``sv_weight`` or ``bounds_weight`` is non-zero.

    Returns:
        (total_loss, components_dict) where components_dict contains
        'loss_reg', 'loss_sv', 'loss_bounds' as scalar floats for logging.
    """
    loss_reg = strain_loss(delta_pred, delta_target, loss_fn=loss_fn, delta=delta)

    need_phys = (sv_weight != 0.0) or (bounds_weight != 0.0)
    if need_phys:
        if delta_mean is None or delta_std is None:
            raise ValueError(
                "pair_loss requires delta_mean and delta_std (train-split Δε mean/std, "
                "shape (6,)) whenever sv_weight or bounds_weight is non-zero — "
                "Saint-Venant and physical_bounds_loss operate in physical strain units."
            )
        dm = delta_mean.to(device=delta_pred.device, dtype=delta_pred.dtype)
        ds = delta_std.to(device=delta_pred.device, dtype=delta_pred.dtype)
        pred_phys = delta_pred * ds + dm
        loss_sv = loop_consistency_loss(pred_phys, pos_a, pos_b)
        loss_bounds = physical_bounds_loss(pred_phys, max_abs_strain=max_abs_strain)
    else:
        loss_sv = delta_pred.new_tensor(0.0)
        loss_bounds = delta_pred.new_tensor(0.0)

    total = loss_reg + sv_weight * loss_sv + bounds_weight * loss_bounds

    components = {
        "loss_reg":    loss_reg.item(),
        "loss_sv":     loss_sv.item(),
        "loss_bounds": loss_bounds.item(),
    }
    return total, components
