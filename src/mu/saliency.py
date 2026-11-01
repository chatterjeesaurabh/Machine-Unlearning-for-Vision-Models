"""M4: Gradient based Weight Saliency.

Mask = top-`ratio` fraction (global, over all parameters) of |grad_theta CE(theta_0; D_f)|, gradient accumulated over the
whole forget set at the original weights, eval mode, no augmentation (Weight Saliency paper eq. 3-5;
sample_repos/Unlearn-Saliency-master/Classification/generate_mask.py). `ratio` = fraction of weights that are UPDATED.
"""
import time

import torch
import torch.nn.functional as F

from mu.data import GPULoader


def compute_saliency_mask(model, data, forget_idx, ratio, batch_size=256):
    t0 = time.time()
    model.eval()
    model.zero_grad(set_to_none=True)
    for x, y, _ in GPULoader(data, "train", forget_idx, batch_size):
        F.cross_entropy(model(x), y, reduction="sum").backward()
    grads = {n: p.grad.detach().abs() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    mask = _topk_mask(grads, ratio)
    return mask, time.time() - t0


def random_mask(model, ratio, seed=0):
    g = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    scores = {n: torch.rand(p.shape, device=p.device, generator=g) for n, p in model.named_parameters()}
    return _topk_mask(scores, ratio)


def _topk_mask(scores, ratio):
    flat = torch.cat([s.flatten() for s in scores.values()])
    k = max(1, int(round(ratio * flat.numel())))
    thresh = torch.topk(flat, k, sorted=True).values[-1]
    return {n: (s >= thresh).float() for n, s in scores.items()}


def mask_stats(mask):
    tot = sum(m.numel() for m in mask.values())
    on = sum(m.sum().item() for m in mask.values())
    per_layer = {n: round(m.mean().item(), 4) for n, m in mask.items()}
    return {"update_ratio_actual": on / tot, "n_params": tot, "per_param_ratio": per_layer}


class MaskedUpdate:
    """Applies a weight mask to SGD updates: grad *= mask before the step; after the step, masked-out weights are restored
    to theta_0 and their momentum zeroed (otherwise weight decay / momentum would still move them)."""

    def __init__(self, model, mask):
        self.mask = mask
        self.params = dict(model.named_parameters())
        self.theta0 = {n: p.detach().clone() for n, p in self.params.items()} if mask is not None else None

    def before_step(self):
        if self.mask is None:
            return
        for n, p in self.params.items():
            if p.grad is not None:
                p.grad.mul_(self.mask[n])

    @torch.no_grad()
    def after_step(self, optimizer):
        if self.mask is None:
            return
        for n, p in self.params.items():
            m = self.mask[n]
            p.mul_(m).add_(self.theta0[n] * (1 - m))
            buf = optimizer.state.get(p, {}).get("momentum_buffer")
            if buf is not None:
                buf.mul_(m)
