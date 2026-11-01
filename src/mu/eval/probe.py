"""E4: Linear probe and Nearest-Class-Centre (NCC) on frozen embeddings (Illusion-of-Unlearning protocol).

Probe: fresh linear head (512 -> 10) trained on embeddings of ALL train-pool samples (retain + forget), Adam lr 1e-3,
bs 1024, 100 epochs, no augmentation (paper does not specify the optimizer; this is our choice). Evaluated per class on test.
NCC: class means of train-pool embeddings; test sample -> nearest mean (Euclidean).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _per_class(pred, y, n=10):
    return [float((pred[y == c] == c).float().mean() * 100) for c in range(n)]


def linear_probe(f_tr, y_tr, f_te, y_te, epochs=100, lr=1e-3, bs=1024, seed=0):
    g = torch.Generator(device=f_tr.device).manual_seed(seed)
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-6
    xtr, xte = (f_tr - mu) / sd, (f_te - mu) / sd
    head = torch.nn.Linear(f_tr.shape[1], 10).to(f_tr.device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=xtr.device, generator=g)
        for s in range(0, len(xtr), bs):
            b = perm[s:s + bs]
            loss = F.cross_entropy(head(xtr[b]), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return head(xte).argmax(1), head(xtr).argmax(1)


def ncc(f_tr, y_tr, f_te):
    means = torch.stack([f_tr[y_tr == c].mean(0) for c in range(10)])
    return torch.cdist(f_te, means).argmin(1)


def probe_metrics(f_tr, y_tr, f_te, y_te, forget_classes, seed=0):
    out = {}
    pred_lp, _ = linear_probe(f_tr, y_tr, f_te, y_te, seed=seed)
    pred_ncc = ncc(f_tr, y_tr, f_te)
    fc = torch.isin(y_te, torch.as_tensor(forget_classes, device=y_te.device))
    for name, pred in (("probe", pred_lp), ("ncc", pred_ncc)):
        out[f"{name}_test_acc"] = float((pred == y_te).float().mean() * 100)
        out[f"{name}_test_forgetcls_acc"] = float((pred[fc] == y_te[fc]).float().mean() * 100)
        out[f"{name}_test_othercls_acc"] = float((pred[~fc] == y_te[~fc]).float().mean() * 100)
        out[f"{name}_test_per_class"] = _per_class(pred, y_te)
    return out
