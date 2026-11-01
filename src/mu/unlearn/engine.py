"""Shared unlearning engine: optimizer, masked step (Weight Saliency), grad clipping, loaders, per-epoch logging, timing."""
import time

import numpy as np
import torch

from mu.data import GPULoader
from mu.eval.accuracy import acc_of, predict
from mu.saliency import MaskedUpdate


class Engine:
    def __init__(self, model, original, data, split, hp, mask, device, seed, recorder):
        self.model, self.original, self.data, self.split = model, original, data, split
        self.hp, self.device, self.seed, self.rec = hp, device, seed, recorder
        self.masked = MaskedUpdate(model, mask)
        self.eval_time = 0.0
        self.stopped_early = False
        rng = np.random.RandomState(seed)
        r = split["retain_idx"]
        self.retain_eval_idx = r[rng.choice(len(r), min(5000, len(r)), replace=False)]

    def train_mode(self):
        """model.train(); with hp bn_mode='frozen' BatchNorm layers stay in eval mode (running stats not updated).
        Default 'train' = reference repos' behaviour (BN stats drift towards the batches seen, e.g. forget-only batches)."""
        self.model.train()
        if self.hp.get("bn_mode", "train") == "frozen":
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

    def optimizer(self, lr=None):
        return torch.optim.SGD(self.model.parameters(), lr=lr if lr is not None else self.hp["lr"],
                               momentum=self.hp.get("momentum", 0.9), weight_decay=self.hp.get("weight_decay", 5e-4))

    def loader(self, which, batch_size=None, labels=None, augment=None, seed_offset=0):
        idx = self.split[f"{which}_idx"]
        return GPULoader(self.data, "train", idx, batch_size or self.hp["batch_size"], shuffle=True,
                         augment=self.hp.get("augment", True) if augment is None else augment,
                         seed=self.seed * 100 + seed_offset, labels=labels)

    def step(self, loss, opt):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        self.masked.before_step()
        if self.hp.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hp["grad_clip"])
        opt.step()
        self.masked.after_step(opt)

    def iter_should_stop(self):
        """Iteration-level early stop: every hp['stop_every_iters'] steps, check forget acc on a fixed 1000-sample forget subset."""
        k = self.hp.get("stop_every_iters")
        stop = self.hp.get("stop_forget_acc")
        if not k or stop is None:
            return False
        self._iters = getattr(self, "_iters", 0) + 1
        if self._iters % k:
            return False
        t0 = time.time()
        if not hasattr(self, "_forget_sub"):
            f = self.split["forget_idx"]
            self._forget_sub = f[np.random.RandomState(self.seed).choice(len(f), min(1000, len(f)), replace=False)]
        a = acc_of(predict(self.model, self.data, "train", self._forget_sub))
        self.train_mode()
        self.eval_time += time.time() - t0
        if a <= stop:
            self.stopped_early = True
            self.rec.log.info(f"iteration-level early stop at iter {self._iters}: forget acc (1k subset) {a:.2f} <= {stop}")
            return True
        return False

    @torch.no_grad()
    def bn_recalibrate(self, n_batches=40, batch_size=256):
        """Re-estimate BatchNorm running statistics on RETAIN data (no weight update). Removes running-stat drift caused by
        forget-only batches (diagnostic / improvement option, hp bn_recalib=1)."""
        bns = [m for m in self.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None  # cumulative average
        self.model.train()
        for i, (x, _, _) in enumerate(self.loader("retain", batch_size=batch_size, augment=False, seed_offset=777)):
            if i >= n_batches:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.model(x)
        for m in bns:
            m.momentum = 0.1
        self.model.eval()

    def quick_forget_acc(self):
        t0 = time.time()
        a = acc_of(predict(self.model, self.data, "train", self.split["forget_idx"]))
        self.train_mode()
        self.eval_time += time.time() - t0
        return a

    def end_epoch(self, epoch, **extra):
        """Per-epoch quick eval (forget full, retain 5k subset, val, test). Returns True if early-stop criterion met."""
        t0 = time.time()
        d, s = self.data, self.split
        row = {"epoch": epoch, **extra,
               "forget_acc": acc_of(predict(self.model, d, "train", s["forget_idx"])),
               "retain_acc_5k": acc_of(predict(self.model, d, "train", self.retain_eval_idx)),
               "val_acc": acc_of(predict(self.model, d, "train", s["val_idx"])),
               "test_acc": acc_of(predict(self.model, d, "test", np.arange(10000)))}
        self.train_mode()
        self.eval_time += time.time() - t0
        ref = getattr(self, "ref", None)
        if self.hp.get("select_best") and ref:
            # best-epoch checkpoint (no test data): mean |forget - R|, |retain(5k) - R|, |val - R| vs the Retrain reference
            row["epoch_sel_score"] = (abs(row["forget_acc"] - ref["acc_forget"]) + abs(row["retain_acc_5k"] - ref["acc_retain"])
                                      + abs(row["val_acc"] - ref["acc_val"])) / 3
            if row["epoch_sel_score"] < getattr(self, "best_score", float("inf")):
                self.best_score, self.best_epoch = row["epoch_sel_score"], epoch
                self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.rec.history(row)
        self.rec.log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        stop = self.hp.get("stop_forget_acc")
        if stop is not None and row["forget_acc"] <= stop:
            self.stopped_early = True
            self.rec.log.info(f"early stop: forget_acc {row['forget_acc']:.2f} <= {stop}")
            return True
        return False
