"""E1: accuracy and per-sample outputs (no augmentation)."""
import numpy as np
import torch
import torch.nn.functional as F

from mu.data import GPULoader


@torch.no_grad()
def predict(model, data, part, indices, batch_size=1000):
    """Returns dict of numpy arrays: logits-derived loss, pred, label, true-class prob."""
    model.eval()
    losses, preds, labels, conf = [], [], [], []
    if len(indices) == 0:
        return {k: np.array([]) for k in ("loss", "pred", "label", "conf")}
    for x, y, _ in GPULoader(data, part, indices, batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits = logits.float()
        losses.append(F.cross_entropy(logits, y, reduction="none"))
        preds.append(logits.argmax(1))
        labels.append(y)
        conf.append(logits.softmax(1).gather(1, y[:, None])[:, 0])
    cat = lambda t: torch.cat(t).cpu().numpy()
    return {"loss": cat(losses), "pred": cat(preds), "label": cat(labels), "conf": cat(conf)}


def acc_of(out):
    return float((out["pred"] == out["label"]).mean() * 100) if len(out["label"]) else float("nan")


def per_class_acc(out, num_classes=10):
    return [float((out["pred"][out["label"] == c] == c).mean() * 100) if (out["label"] == c).any() else float("nan")
            for c in range(num_classes)]


def evaluate_splits(model, data, split, test_classes_forget=None):
    """Accuracy on forget / retain / val / test (+ test split into forget-classes and other classes).

    test_classes_forget: classes whose test samples count as 'test_forget' (S1: [c]; S2: K)."""
    y_test = data.y_test.cpu().numpy()
    outs = {
        "forget": predict(model, data, "train", split["forget_idx"]),
        "retain": predict(model, data, "train", split["retain_idx"]),
        "val": predict(model, data, "train", split["val_idx"]),
        "test": predict(model, data, "test", np.arange(len(y_test))),
    }
    m = {f"acc_{k}": acc_of(v) for k, v in outs.items()}
    if test_classes_forget:
        tf = np.isin(y_test, test_classes_forget)
        m["acc_test_forgetcls"] = float((outs["test"]["pred"][tf] == y_test[tf]).mean() * 100)
        m["acc_test_othercls"] = float((outs["test"]["pred"][~tf] == y_test[~tf]).mean() * 100)
    m["test_per_class"] = per_class_acc(outs["test"])
    return m, outs
