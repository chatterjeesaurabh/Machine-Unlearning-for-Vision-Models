"""Train the Original model (scenario 'orig') or a Retrain-from-scratch reference (any other scenario: trains on retain set only).

Recipe (Weight Saliency paper App. C + its code defaults): SGD lr 0.1, momentum 0.9, wd 5e-4, batch 256, 182 epochs,
cosine LR, RandomCrop(32,4) + HFlip. Last epoch is kept.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mu import CKPT_DIR, ROOT
from mu.data import CIFAR10GPU, GPULoader
from mu.eval.accuracy import acc_of, evaluate_splits, predict
from mu.eval.mia import loss_mia
from mu.models import resnet18
from mu.records import RunRecorder
from mu.splits import load_split
from mu.utils import save_ckpt, seed_everything, setup_device, run_with_oom_retry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="orig")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=182)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--mem_frac", type=float, default=0.45, help="hard cap of the caching allocator (fraction of GPU memory)")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    method = "original" if args.scenario == "orig" else "retrain"
    run_id = f"{args.scenario}__{method}" + (f"__{args.tag}" if args.tag else "") + f"__s{args.seed}"
    device = setup_device(args.gpu, args.mem_frac)
    seed_everything(args.seed)
    split = load_split(args.scenario, args.seed)
    sc = split["scenario"]
    forget_classes = [sc["class"]] if sc["type"] == "class" else sc.get("classes")
    ckpt_path = os.path.join(CKPT_DIR, f"{run_id}.pt")

    rec = RunRecorder(run_id, {
        "kind": "reference", "method": method, "scenario": args.scenario, "scenario_meta": sc, "seed": args.seed,
        "split_files": ["data/splits/base.npz", f"data/splits/{args.scenario}__s{args.seed}.npz"],
        "n_train": int(len(split["retain_idx"])), "n_forget": int(len(split["forget_idx"])), "n_val": int(len(split["val_idx"])),
        "hparams": {"epochs": args.epochs, "lr": args.lr, "schedule": "cosine", "batch_size": args.batch_size,
                    "optimizer": "SGD", "momentum": args.momentum, "weight_decay": args.wd,
                    "augmentation": "RandomCrop(32,pad=4)+HFlip (GPU)", "precision": "bf16 autocast", "mem_frac": args.mem_frac},
        "checkpoint": os.path.relpath(ckpt_path, ROOT),
    })
    log = rec.log
    data = CIFAR10GPU(device)
    model = resnet18().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    loader = GPULoader(data, "train", split["retain_idx"], args.batch_size, shuffle=True, augment=True, seed=args.seed)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    log.info(f"run {run_id}: train on {len(split['retain_idx'])} samples, forget {len(split['forget_idx'])}, {len(loader)} it/epoch")

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        tot_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        n = 0
        for x, y, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.detach() * len(y)
            correct += (out.argmax(1) == y).sum()
            n += len(y)
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "train_loss": tot_loss.item() / n, "train_acc": correct.item() / n * 100,
               "epoch_time_s": time.time() - te}
        if ep % args.eval_every == 0 or ep == args.epochs:
            row["val_acc"] = acc_of(predict(model, data, "train", split["val_idx"]))
            row["test_acc"] = acc_of(predict(model, data, "test", np.arange(10000)))
            if len(split["forget_idx"]):
                row["forget_acc"] = acc_of(predict(model, data, "train", split["forget_idx"]))
            log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        rec.history(row)
    train_time = time.time() - t0
    save_ckpt(model, ckpt_path, run_id=run_id, epochs=args.epochs)

    metrics, outs = evaluate_splits(model, data, split, forget_classes)
    metrics.update(train_time_s=train_time, peak_mem_gb=torch.cuda.max_memory_allocated(device) / 1e9)
    if len(split["forget_idx"]):
        mia = loss_mia(outs["forget"]["loss"], outs["forget"]["label"], outs["test"]["loss"], outs["test"]["label"], seed=args.seed)
        metrics["mia"] = mia
        metrics["mia_val"] = loss_mia(outs["forget"]["loss"], outs["forget"]["label"], outs["val"]["loss"], outs["val"]["label"], seed=args.seed)
    rec.per_sample(forget_idx=split["forget_idx"], forget_loss=outs["forget"]["loss"], forget_pred=outs["forget"]["pred"],
                   test_loss=outs["test"]["loss"], test_pred=outs["test"]["pred"], test_label=outs["test"]["label"])
    log.info(f"FINAL { {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))} }")
    if "mia" in metrics:
        log.info(f"MIA acc={metrics['mia']['mia_acc']:.2f} auc={metrics['mia']['mia_auc']:.2f}")
    rec.finish(metrics)


if __name__ == "__main__":
    run_with_oom_retry(main)
