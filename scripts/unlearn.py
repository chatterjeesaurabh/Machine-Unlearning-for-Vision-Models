"""Run one unlearning job (or evaluate the Original model on a scenario with --method none).

Example:
  python scripts/unlearn.py --scenario S1c0 --seed 0 --method GAGD --mask saliency --mask_ratio 0.5 --hp lr=0.01 beta=0.99
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import yaml

from mu import CKPT_DIR, RECORDS_DIR, ROOT
from mu.data import CIFAR10GPU
from mu.eval.accuracy import evaluate_splits, predict
from mu.eval.mia import loss_mia
from mu.models import resnet18
from mu.records import RunRecorder
from mu.saliency import compute_saliency_mask, mask_stats, random_mask
from mu.splits import load_split
from mu.unlearn import METHODS
from mu.unlearn.engine import Engine
from mu.utils import load_ckpt, save_ckpt, seed_everything, setup_device, run_with_oom_retry

METHOD_NAMES = {"GA": "M1 Gradient Ascent", "GAGD": "M2 Gradient Ascent + Descent", "TS": "M3a Teacher-Student Unlearning",
                "RLrand": "M3b-R Random Relabeling", "RLbnd": "M3b-B Boundary Relabeling", "none": "Original (no unlearning)",
                "TSjoint": "M3a variant: Teacher-Student, joint loss (no alternation)"}


def load_hp(method, scenario_type, overrides):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "methods.yaml")))[method]
    hp = {**cfg.get("common", {}), **(cfg.get(scenario_type) or {})}
    for kv in overrides:
        k, v = kv.split("=", 1)
        v = yaml.safe_load(v)
        if isinstance(v, str):  # YAML 1.1 reads '1e-05' (no dot) as a string
            try:
                v = float(v)
            except ValueError:
                pass
        hp[k] = v
    return hp


def retrain_metrics(scenario, seed, data, split, device):
    """Retrain reference metrics; back-fills 'mia_val' (forget vs same-class VAL losses) for older retrain runs."""
    rid = f"{scenario}__retrain__s{seed}"
    p = os.path.join(RECORDS_DIR, "runs", rid, "metrics.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    if "mia_val" not in r:
        m = resnet18().to(device).to(memory_format=torch.channels_last)
        load_ckpt(m, os.path.join(CKPT_DIR, f"{rid}.pt"), device)
        fo, va = predict(m, data, "train", split["forget_idx"]), predict(m, data, "train", split["val_idx"])
        r["mia_val"] = loss_mia(fo["loss"], fo["label"], va["loss"], va["label"], seed=seed)
        json.dump(r, open(p, "w"), indent=1)
        del m
    return r


def avg_gap(m, r):
    """avg_gap (report, Weight Saliency paper style): mean |.| over forget acc, retain acc, test acc, MIA acc (test negatives).
    sel_gap (hyperparameter selection, no test data): mean |.| over forget acc, retain acc, val acc, MIA acc (val negatives)."""
    g = {f"gap_{k}": abs(m[k] - r[k]) for k in ("acc_forget", "acc_retain", "acc_test", "acc_val")}
    g["gap_mia_acc"] = abs(m["mia"]["mia_acc"] - r["mia"]["mia_acc"])
    g["gap_mia_val_acc"] = abs(m["mia_val"]["mia_acc"] - r["mia_val"]["mia_acc"])
    g["avg_gap"] = (g["gap_acc_forget"] + g["gap_acc_retain"] + g["gap_acc_test"] + g["gap_mia_acc"]) / 4
    g["sel_gap"] = (g["gap_acc_forget"] + g["gap_acc_retain"] + g["gap_acc_val"] + g["gap_mia_val_acc"]) / 4
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", required=True, choices=list(METHODS) + ["none"])
    p.add_argument("--mask", default="none", choices=["none", "saliency", "random"])
    p.add_argument("--mask_ratio", type=float, default=0.5, help="fraction of weights updated")
    p.add_argument("--hp", nargs="*", default=[], help="overrides key=value")
    p.add_argument("--tag", default="")
    p.add_argument("--orig_ckpt", default=os.path.join(CKPT_DIR, "orig__original__s0.pt"))
    p.add_argument("--save_ckpt", type=int, default=1)
    p.add_argument("--eval_label", default=None, help="run-id label for --method none (default 'original')")
    p.add_argument("--ref_scenario", default=None, help="scenario whose Retrain run is the reference (default: --scenario)")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--mem_frac", type=float, default=0.45)
    args = p.parse_args()

    device = setup_device(args.gpu, args.mem_frac)
    seed_everything(args.seed)
    split = load_split(args.scenario, args.seed)
    sc = split["scenario"]
    forget_classes = [sc["class"]] if sc["type"] == "class" else sc.get("classes")
    method = (args.eval_label or "original") if args.method == "none" else args.method
    mask_tag = {"none": "", "saliency": f"__sal{args.mask_ratio:g}", "random": f"__randmask{args.mask_ratio:g}"}[args.mask]
    run_id = f"{args.scenario}__{method}{mask_tag}" + (f"__{args.tag}" if args.tag else "") + f"__s{args.seed}"
    hp = load_hp(args.method, sc["type"], args.hp) if args.method != "none" else {}
    data = CIFAR10GPU(device)
    ref_sc = args.ref_scenario or args.scenario  # sequential steps use the base scenario's Retrain as reference
    retr = retrain_metrics(ref_sc, args.seed, data, split if ref_sc == args.scenario else load_split(ref_sc, args.seed), device)
    if hp.get("stop_forget_acc") == "retrain":
        hp["stop_forget_acc"] = max(retr["acc_forget"], hp.get("stop_floor", 0.0)) if retr else None

    model = resnet18().to(device).to(memory_format=torch.channels_last)
    load_ckpt(model, args.orig_ckpt, device)
    original = copy.deepcopy(model).eval()
    for q in original.parameters():
        q.requires_grad_(False)

    rec = RunRecorder(run_id, {
        "kind": "eval" if args.method == "none" else "unlearn", "method": method, "method_name": METHOD_NAMES[args.method],
        "mask": args.mask, "mask_ratio": args.mask_ratio if args.mask != "none" else None,
        "scenario": args.scenario, "scenario_meta": sc, "seed": args.seed, "hparams": hp, "tag": args.tag,
        "split_files": ["data/splits/base.npz", f"data/splits/{args.scenario}__s{args.seed}.npz"],
        "n_forget": int(len(split["forget_idx"])), "n_retain": int(len(split["retain_idx"])),
        "init_checkpoint": os.path.relpath(args.orig_ckpt, ROOT), "retrain_reference": f"{ref_sc}__retrain__s{args.seed}" if retr else None,
    })
    torch.cuda.reset_peak_memory_stats(device)

    mask, mask_time, extra = None, 0.0, {}
    if args.mask == "saliency":
        mask, mask_time = compute_saliency_mask(model, data, split["forget_idx"], args.mask_ratio)
    elif args.mask == "random":
        mask = random_mask(model, args.mask_ratio, seed=args.seed)
    if mask is not None:
        st = mask_stats(mask)
        extra.update(mask_update_ratio=st["update_ratio_actual"], mask_time_s=mask_time)
        with open(os.path.join(rec.dir, "mask_stats.json"), "w") as f:
            json.dump(st, f, indent=1)
        rec.log.info(f"mask {args.mask} ratio={st['update_ratio_actual']:.4f} time={mask_time:.1f}s")

    engine = Engine(model, original, data, split, hp, mask, device, args.seed, rec)
    engine.ref = retr
    t0 = time.time()
    if args.method != "none":
        model.train()
        METHODS[args.method](engine)
        if hp.get("bn_recalib"):
            engine.bn_recalibrate()
        if hp.get("select_best") and getattr(engine, "best_state", None) is not None:
            model.load_state_dict(engine.best_state)
            extra.update(best_epoch=engine.best_epoch, best_epoch_score=engine.best_score)
            rec.log.info(f"restored best epoch {engine.best_epoch} (score {engine.best_score:.3f})")
    torch.cuda.synchronize()
    unlearn_time = time.time() - t0 - engine.eval_time

    metrics, outs = evaluate_splits(model, data, split, forget_classes)
    metrics.update(unlearn_time_s=unlearn_time, total_time_s=unlearn_time + mask_time, peak_mem_gb=torch.cuda.max_memory_allocated(device) / 1e9,
                   stopped_early=engine.stopped_early, **extra)
    # NaN logits -> argmax 0 -> spurious accuracies; flag explicitly
    metrics["diverged"] = bool(not all(np.isfinite(outs[k]["loss"]).all() for k in ("forget", "retain", "test")))
    metrics["mia"] = loss_mia(outs["forget"]["loss"], outs["forget"]["label"], outs["test"]["loss"], outs["test"]["label"], seed=args.seed)
    metrics["mia_val"] = loss_mia(outs["forget"]["loss"], outs["forget"]["label"], outs["val"]["loss"], outs["val"]["label"], seed=args.seed)
    excl = np.setdiff1d(split["pool_idx"], np.concatenate([split["forget_idx"], split["retain_idx"]]))
    if len(excl):  # samples in neither forget nor retain set (e.g. the other entanglement partitions in <base>-Ex<part>)
        ex = predict(model, data, "train", excl)
        metrics["acc_excluded"] = float((ex["pred"] == ex["label"]).mean() * 100)
        metrics["n_excluded"] = int(len(excl))
    if retr:
        metrics.update(avg_gap(metrics, retr))
        metrics["retrain_ref"] = {k: retr[k] for k in ("acc_forget", "acc_retain", "acc_test", "acc_val")} | {
            "mia_acc": retr["mia"]["mia_acc"], "mia_val_acc": retr["mia_val"]["mia_acc"]}
    rec.per_sample(forget_idx=split["forget_idx"], forget_loss=outs["forget"]["loss"], forget_pred=outs["forget"]["pred"],
                   forget_label=outs["forget"]["label"], test_loss=outs["test"]["loss"], test_pred=outs["test"]["pred"], test_label=outs["test"]["label"])
    if args.save_ckpt and args.method != "none":
        save_ckpt(model, os.path.join(CKPT_DIR, f"{run_id}.pt"), run_id=run_id)
    short = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items() if not isinstance(v, (list, dict))}
    rec.log.info(f"FINAL {short} MIA acc={metrics['mia']['mia_acc']:.2f} auc={metrics['mia']['mia_auc']:.2f}")
    rec.finish(metrics)


if __name__ == "__main__":
    run_with_oom_retry(main)
