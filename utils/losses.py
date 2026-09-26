import torch
import torch.nn.functional as F


def rain_weighted_l1(pred, target, weight_factor=5.0, threshold=0.0):
    """L1 loss with extra weight on pixels where target is "rain" (target > threshold,
    in whatever normalized space pred/target already live in). Plain L1 lets a model
    minimize loss by staying close to the dominant dry background and barely engaging
    with the rare wet pixels; this makes getting those pixels wrong cost more."""
    weights = torch.where(target > threshold, weight_factor, 1.0)
    return (weights * (pred - target).abs()).mean()


def gradient_loss(pred, target):
    """L1 loss on horizontal/vertical finite-difference gradients -- rewards matching
    the target's spatial structure (edges, gradients), which plain per-pixel L1 doesn't
    penalize directly (a smoothed-out prediction can still have low pixel-wise L1)."""
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


def ssim_loss(pred, target, window_size=11, c1=0.01 ** 2, c2=0.03 ** 2):
    """1 - SSIM (structural similarity), via a uniform (box) window average pool --
    rewards correct local contrast/structure rather than just per-pixel magnitude.
    c1/c2 are the standard SSIM stabilizing constants, calibrated for a roughly [0, 1]
    or [0, 255] dynamic range; on this project's z-scored normalized precip values,
    they may need retuning (pass different --ssim_c1/--ssim_c2 values) since the
    data's actual dynamic range is different."""
    pad = window_size // 2
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    mu_pred_sq, mu_target_sq = mu_pred.pow(2), mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return 1.0 - ssim_map.mean()


# Registry used by train.py's --losses CLI flag to build the active extra loss terms.
LOSS_REGISTRY = {
    "rain_weighted": rain_weighted_l1,
    "gradient": gradient_loss,
    "ssim": ssim_loss,
}


def build_extra_losses(names, args):
    """--losses names -> [(name, weight, fn), ...] with each fn already bound to its
    CLI-configured hyperparameters (so generator_step only needs to call fn(pred, target))."""
    extra = []
    for name in names:
        if name == "rain_weighted":
            fn = lambda pred, target: rain_weighted_l1(
                pred, target, weight_factor=args.rain_weight_factor, threshold=args.rain_weight_threshold)
            extra.append((name, args.rain_weight_loss_weight, fn))
        elif name == "gradient":
            extra.append((name, args.gradient_loss_weight, gradient_loss))
        elif name == "ssim":
            fn = lambda pred, target: ssim_loss(pred, target, c1=args.ssim_c1, c2=args.ssim_c2)
            extra.append((name, args.ssim_loss_weight, fn))
        else:
            raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return extra
