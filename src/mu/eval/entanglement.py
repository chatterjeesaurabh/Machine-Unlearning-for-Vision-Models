"""E3: Embedding entanglement between forget (F) and retain (R) sets, measured on the ORIGINAL model's embeddings.

Global ES (RUM paper):
  ES = [1/|R| sum_R ||phi - mu_R||^2 + 1/|F| sum_F ||phi - mu_F||^2] / [0.5 (||mu_R - mu||^2 + ||mu_F - mu||^2)],  mu = mean over F u R.
Per-sample (project doc formula, restricted to the k nearest neighbours N_k(i) of forget sample i in F u R, self excluded):
  E_i = sum_{j in N_k(i) n R} exp(-d(i,j)^2 / sigma^2) / sum_{j in N_k(i)} exp(-d(i,j)^2 / sigma^2)
  sigma = median of all forget-sample kNN distances. Higher E_i = forget sample sits among retain samples = more entangled.
Also: mean distance to the k nearest RETAIN neighbours (slide 14), smaller = more entangled.
"""
import numpy as np
import torch


def global_es(phi_r, phi_f):
    mu_r, mu_f = phi_r.mean(0), phi_f.mean(0)
    mu = torch.cat([phi_r, phi_f]).mean(0)
    intra = ((phi_r - mu_r) ** 2).sum(1).mean() + ((phi_f - mu_f) ** 2).sum(1).mean()
    inter = 0.5 * (((mu_r - mu) ** 2).sum() + ((mu_f - mu) ** 2).sum())
    return float(intra / inter)


@torch.no_grad()
def per_sample_entanglement(phi_f, phi_r, k=50, chunk=512):
    """Returns dict of numpy arrays (len |F|): E (kernel retain share), retain_frac (unweighted share of retain among kNN),
    mean_dist_retain_knn, plus sigma."""
    all_phi = torch.cat([phi_f, phi_r])
    is_retain = torch.cat([torch.zeros(len(phi_f), dtype=torch.bool, device=phi_f.device),
                           torch.ones(len(phi_r), dtype=torch.bool, device=phi_f.device)])
    nn_d, nn_r, ret_d = [], [], []
    for s in range(0, len(phi_f), chunk):
        q = phi_f[s:s + chunk]
        d = torch.cdist(q, all_phi)
        d[torch.arange(len(q)), torch.arange(s, s + len(q))] = float("inf")  # exclude self
        vals, idx = torch.topk(d, k, largest=False)
        nn_d.append(vals)
        nn_r.append(is_retain[idx])
        ret_d.append(torch.topk(d[:, len(phi_f):], k, largest=False).values.mean(1))
    nn_d, nn_r = torch.cat(nn_d), torch.cat(nn_r)
    sigma = nn_d.median()
    w = torch.exp(-(nn_d ** 2) / sigma ** 2)
    e = (w * nn_r).sum(1) / w.sum(1).clamp_min(1e-30)
    return {"E": e.cpu().numpy(), "retain_frac": nn_r.float().mean(1).cpu().numpy(),
            "mean_dist_retain_knn": torch.cat(ret_d).cpu().numpy(), "sigma": float(sigma), "k": k}


def tertile_partition(score, tiebreak=None):
    """0 = low, 1 = medium, 2 = high: equal-size rank tertiles of `score`; ties broken by `tiebreak` (higher = more entangled).
    Needed for class forgetting, where most E_i are exactly 0 (all kNN are other forget samples); there we use
    tiebreak = -mean distance to the k nearest retain samples (slide 14: closer to retain = more entangled)."""
    order = np.lexsort((tiebreak if tiebreak is not None else np.zeros_like(score), score))
    part = np.empty(len(score), dtype=np.int8)
    part[order] = (np.arange(len(score)) * 3 // len(score)).astype(np.int8)
    return part
