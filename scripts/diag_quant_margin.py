"""Why does INT4 lower the forget accuracy of unlearned models more than Retrain's (S2)? Margin diagnostic.
For each model: fp32 true-class confidence p_y of every forget / same-class test sample, which samples flip (correct -> wrong) at INT4,
and the fp32 confidence of flipped vs. kept samples. Writes records/results/quant_margin_diag.md.
Usage: python scripts/diag_quant_margin.py --scenario S2k359p30 --seed 0 [--ckpt_dirs dir1 dir2]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from mu import CKPT_DIR, RECORDS_DIR
from mu.best_configs import METHODS, run_id
from mu.data import CIFAR10GPU
from mu.models import resnet18
from mu.splits import load_split
from mu.utils import load_ckpt, run_with_oom_retry, setup_device

import sys
sys.path.insert(0, os.path.dirname(__file__))
from eval_quant import quantized_copy  # noqa: E402


@torch.no_grad()
def conf_pred(model, x, y, bs=2000):
    model.eval()
    ps, pr = [], []
    for i in range(0, len(x), bs):
        lg = model((x[i:i + bs].float() / 255).contiguous(memory_format=torch.channels_last)).float()
        ps.append(lg.softmax(1).gather(1, y[i:i + bs, None])[:, 0])
        pr.append(lg.argmax(1))
    return torch.cat(ps).cpu().numpy(), torch.cat(pr).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="S2k359p30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--ckpt_dirs", nargs="*", default=[CKPT_DIR, "/anchor/anchor7/sachatt/cs6886/project/outputs/checkpoints"])
    p.add_argument("--gpu", type=int, default=1)
    args = p.parse_args()

    def ckpt(rid):
        name = "orig__original__s0.pt" if "__original__" in rid else f"{rid}.pt"
        for d in args.ckpt_dirs:
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
        return None

    device = setup_device(args.gpu, 0.45)
    data = CIFAR10GPU(device)
    split = load_split(args.scenario, args.seed)
    sc = split["scenario"]
    fcls = [sc["class"]] if sc["type"] == "class" else sc["classes"]
    xf, yf = data.x_train[split["forget_idx"]], data.y_train[split["forget_idx"]]
    tmask = torch.isin(data.y_test, torch.tensor(fcls, device=device))
    xt, yt = data.x_test[tmask], data.y_test[tmask]
    models = [("Retrain", f"{args.scenario}__retrain__s{args.seed}")] + [(m, run_id(args.scenario, m, "none", args.seed)) for m in METHODS]
    L = [f"# INT{args.bits} margin diagnostic - {args.scenario} seed {args.seed}",
         "p_y = fp32 softmax probability of the true label. 'low margin' = correctly classified with p_y < 0.9. 'flipped' = correct in fp32, wrong at "
         f"INT{args.bits}. Same-class test = test images of the forget classes {fcls}.", "",
         "| model | set | acc fp32 | acc INT | flipped (% of set) | median p_y all | % low-margin among correct | median p_y of flipped | median p_y of kept-correct |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, rid in models:
        path = ckpt(rid)
        if path is None:
            L.append(f"| {name} | - | checkpoint missing ({rid}) | | | | | | |")
            continue
        m = resnet18().to(device).to(memory_format=torch.channels_last)
        load_ckpt(m, path, device)
        mq = quantized_copy(m, args.bits)
        for sname, x, y in (("forget", xf, yf), ("same-class test", xt, yt)):
            p32, pr32 = conf_pred(m, x, y)
            _, prq = conf_pred(mq, x, y)
            yn = y.cpu().numpy()
            c32, cq = pr32 == yn, prq == yn
            flipped, kept = c32 & ~cq, c32 & cq
            low = (p32[c32] < 0.9).mean() * 100 if c32.any() else float("nan")
            L.append(f"| {name} | {sname} | {c32.mean()*100:.2f} | {cq.mean()*100:.2f} | {flipped.mean()*100:.2f} | {np.median(p32):.3f} | {low:.1f} | "
                     f"{np.median(p32[flipped]) if flipped.any() else float('nan'):.3f} | {np.median(p32[kept]) if kept.any() else float('nan'):.3f} |")
        print(L[-2]); print(L[-1])
    out = os.path.join(RECORDS_DIR, "results", "quant_margin_diag.md")
    mode = "a" if os.path.exists(out) else "w"
    open(out, mode).write("\n".join(L) + "\n\n")
    print("->", out)


if __name__ == "__main__":
    run_with_oom_retry(main)
