"""Objective 3: post-unlearning quantization.

Simulated (fake) weight-only quantization: per-output-channel symmetric uniform quantization of every Conv2d / Linear weight
to b bits (scale = max|w| / (2^(b-1) - 1)), BatchNorm and biases kept in fp32. Evaluated: forget / retain / test acc, MIA.
Mechanism stats vs the Original model theta_o: relative weight change ||theta_u - theta_o|| / ||theta_o||, mean |delta| / quant step,
and fraction of quantized weights identical to the quantized Original (Q(theta_u) == Q(theta_o)) -> tests H1 ("quantization
maps small unlearning updates back onto the original grid").
Writes records/runs/<run_id>/quant_metrics.json.
Usage: python scripts/eval_quant.py --pattern 'S1c0__*__s0' [--bits 8 6 4]
"""
import argparse
import copy
import fnmatch
import glob
import json
import os

import numpy as np
import torch

from mu import CKPT_DIR, RECORDS_DIR
from mu.data import CIFAR10GPU
from mu.eval.accuracy import evaluate_splits
from mu.eval.mia import loss_mia
from mu.models import resnet18
from mu.splits import load_split
from mu.utils import load_ckpt, run_with_oom_retry, setup_device


def ckpt_for(run_id):
    return os.path.join(CKPT_DIR, "orig__original__s0.pt") if "__original__" in run_id else os.path.join(CKPT_DIR, f"{run_id}.pt")


def qweights(model):
    return {n: m.weight for n, m in model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))}


@torch.no_grad()
def quantize_tensor(w, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().flatten(1).amax(1).clamp_min(1e-12) / qmax
    shape = (-1,) + (1,) * (w.dim() - 1)
    q = torch.round(w / scale.view(shape)).clamp_(-qmax, qmax)
    return q, scale.view(shape)


@torch.no_grad()
def quantized_copy(model, bits):
    m = copy.deepcopy(model)
    for _, w in qweights(m).items():
        q, s = quantize_tensor(w, bits)
        w.copy_(q * s)
    return m


@torch.no_grad()
def mechanism(model, orig, bits_list):
    wu, wo = qweights(model), qweights(orig)
    num = sum(((wu[n] - wo[n]) ** 2).sum().item() for n in wu)
    den = sum((wo[n] ** 2).sum().item() for n in wo)
    out = {"rel_weight_change": float(np.sqrt(num / den)), "frac_weights_changed": float(sum((wu[n] != wo[n]).sum().item() for n in wu) / sum(w.numel() for w in wu.values()))}
    for b in bits_list:
        same, tot, ratio = 0, 0, []
        for n in wu:
            qu, su = quantize_tensor(wu[n], b)
            qo, so = quantize_tensor(wo[n], b)
            same += (qu * su == qo * so).sum().item()
            tot += qu.numel()
            ratio.append(((wu[n] - wo[n]).abs() / so).mean().item())
        out[f"int{b}"] = {"frac_same_as_quantized_original": same / tot, "mean_abs_delta_over_step": float(np.mean(ratio))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--pattern", default=None)
    p.add_argument("--bits", nargs="+", type=int, default=[8, 6, 4])
    p.add_argument("--skip_done", type=int, default=1)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--mem_frac", type=float, default=0.45)
    args = p.parse_args()
    runs = list(args.runs)
    if args.pattern:
        runs += [os.path.basename(d) for d in sorted(glob.glob(os.path.join(RECORDS_DIR, "runs", "*"))) if fnmatch.fnmatch(os.path.basename(d), args.pattern)]
    device = setup_device(args.gpu, args.mem_frac)
    data = CIFAR10GPU(device)
    orig = resnet18().to(device).to(memory_format=torch.channels_last)
    load_ckpt(orig, ckpt_for("x__original__s0"), device)
    for rid in runs:
        out = os.path.join(RECORDS_DIR, "runs", rid, "quant_metrics.json")
        if (args.skip_done and os.path.exists(out)) or not os.path.exists(ckpt_for(rid)):
            continue
        cfg = json.load(open(os.path.join(RECORDS_DIR, "runs", rid, "config.json")))
        if cfg["scenario"] == "orig":
            continue
        split = load_split(cfg["scenario"], cfg["seed"])
        sc = split["scenario"]
        fcls = [sc["class"]] if sc["type"] == "class" else sc["classes"]
        model = resnet18().to(device).to(memory_format=torch.channels_last)
        load_ckpt(model, ckpt_for(rid), device)
        res = {"protocol": "simulated per-channel symmetric weight-only quantization of Conv/Linear weights; BN fp32", "mechanism": mechanism(model, orig, args.bits)}
        for b in [32] + args.bits:
            mq = model if b == 32 else quantized_copy(model, b)
            m, o = evaluate_splits(mq, data, split, fcls)
            m["mia"] = loss_mia(o["forget"]["loss"], o["forget"]["label"], o["test"]["loss"], o["test"]["label"], seed=cfg["seed"])["mia_acc"]
            res[f"w{b}"] = {k: v for k, v in m.items() if not isinstance(v, list)}
        json.dump(res, open(out, "w"), indent=1)
        print(rid, {k: (round(v["acc_forget"], 2), round(v["acc_test"], 2)) for k, v in res.items() if k.startswith("w")})


if __name__ == "__main__":
    run_with_oom_retry(main)
