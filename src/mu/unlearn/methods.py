"""Unlearning methods. Each takes an Engine `E` (model already initialised to the Original weights) and trains E.model in place.

M1  GA      Gradient Ascent                     loss = -CE(forget)                       [RUM-main/unlearn/GA.py]
M2  GAGD    Gradient Ascent + Descent           loss = b*CE(retain) - (1-b)*CE(forget)   [RUM-main/unlearn/neggrad.py]
M3a TS      Teacher-Student Unlearning          max-epochs on forget (-KL), min-epochs on retain (a*KL + g*CE)  [RUM-main/unlearn/scrub.py]
M3b RLrand  Random Relabeling                   forget pass with random wrong labels, then retain pass  [Unlearn-Saliency/unlearn/RL.py]
M3b RLbnd   Boundary Relabeling (nearest wrong) forget relabeled by the Original model's prediction on an FGSM neighbour  [RUM-main/unlearn/boundary_sh.py]
"""
from itertools import cycle

import torch
import torch.nn.functional as F

from mu.data import NUM_CLASSES

AC = dict(device_type="cuda", dtype=torch.bfloat16)


def forward_forget(E, x, r_iter):
    """Forward a forget batch. With hp bn_mix_retain=1 a retain batch is concatenated so that BatchNorm (train mode) normalises
    with, and updates running statistics from, a representative mixed batch; the returned logits are only the forget part."""
    if r_iter is None:
        return E.model(x)
    xr, _, _ = next(r_iter)
    return E.model(torch.cat([x, xr]))[: len(x)]


def retain_mixer(E, ep):
    if not E.hp.get("bn_mix_retain"):
        return None
    return cycle(E.loader("retain", batch_size=E.hp.get("mix_batch_size", E.hp["batch_size"]), seed_offset=5000 + ep))


def gradient_ascent(E):
    opt = E.optimizer()
    for ep in range(1, E.hp["epochs"] + 1):
        E.train_mode()
        stop = False
        r_iter = retain_mixer(E, ep)
        for x, y, _ in E.loader("forget", seed_offset=ep):
            with torch.autocast(**AC):
                loss = -F.cross_entropy(forward_forget(E, x, r_iter), y)
            E.step(loss, opt)
            if E.iter_should_stop():  # optional iteration-level early stop (hp stop_every_iters)
                stop = True
                break
        if E.end_epoch(ep, lr=opt.param_groups[0]["lr"], last_loss=loss.item()) or stop:
            break


def ga_gd(E):
    b = E.hp["beta"]
    opt = E.optimizer()
    for ep in range(1, E.hp["epochs"] + 1):
        E.train_mode()
        f_iter = cycle(E.loader("forget", batch_size=E.hp.get("forget_batch_size"), seed_offset=1000 + ep))
        for (xr, yr, _), (xf, yf, _) in zip(E.loader("retain", seed_offset=ep), f_iter):
            with torch.autocast(**AC):
                if E.hp.get("joint_forward"):  # one forward pass on [retain; forget] -> BN sees mixed batches
                    out = E.model(torch.cat([xr, xf]))
                    lr_, lf = F.cross_entropy(out[: len(xr)], yr), F.cross_entropy(out[len(xr):], yf)
                else:  # repo behaviour: separate forward passes (BN also updated on forget-only batches)
                    lr_, lf = F.cross_entropy(E.model(xr), yr), F.cross_entropy(E.model(xf), yf)
                loss = b * lr_ - (1 - b) * lf
            E.step(loss, opt)
        if E.end_epoch(ep, lr=opt.param_groups[0]["lr"], loss_retain=lr_.item(), loss_forget=lf.item()):
            break


def _kd(s_logits, t_logits, T):
    return F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1), reduction="batchmean") * T * T


def teacher_student(E):
    hp = E.hp
    T, alpha, gamma = hp["kd_T"], hp["alpha"], hp["gamma"]
    teacher = E.original.eval()
    opt = E.optimizer()
    decay = set(hp.get("lr_decay_epochs", []))
    for ep in range(1, hp["epochs"] + 1):
        if ep in decay:
            for g in opt.param_groups:
                g["lr"] *= 0.1
        E.train_mode()
        lmax, f_after_max = float("nan"), float("nan")
        if ep <= hp["msteps"]:
            for x, _, _ in E.loader("forget", batch_size=hp["forget_batch_size"], seed_offset=1000 + ep):
                with torch.autocast(**AC):
                    with torch.no_grad():
                        t = teacher(x)
                    loss = -_kd(E.model(x).float(), t.float(), T)
                E.step(loss, opt)
                lmax = loss.item()
            f_after_max = E.quick_forget_acc()
        for x, y, _ in E.loader("retain", batch_size=hp["retain_batch_size"], seed_offset=ep):
            with torch.autocast(**AC):
                with torch.no_grad():
                    t = teacher(x)
                s = E.model(x)
                loss = alpha * _kd(s.float(), t.float(), T) + gamma * F.cross_entropy(s, y)
            E.step(loss, opt)
        E.end_epoch(ep, lr=opt.param_groups[0]["lr"], max_loss=lmax, min_loss=loss.item(), did_max=int(ep <= hp["msteps"]),
                    forget_acc_after_max=f_after_max)


def teacher_student_joint(E):
    """Variant (not in the paper): optimise the full objective simultaneously at every step instead of alternating
    max/min epochs: loss = alpha*KL_T(retain) + gamma*CE(retain) - lam_f*KL_T(forget), one retain batch + one (cycled)
    forget batch per step."""
    hp = E.hp
    T, alpha, gamma, lam = hp["kd_T"], hp["alpha"], hp["gamma"], hp.get("lam_f", 1.0)
    teacher = E.original.eval()
    opt = E.optimizer()
    decay = set(hp.get("lr_decay_epochs", []))
    for ep in range(1, hp["epochs"] + 1):
        if ep in decay:
            for g in opt.param_groups:
                g["lr"] *= 0.1
        E.train_mode()
        f_iter = cycle(E.loader("forget", batch_size=hp["forget_batch_size"], seed_offset=1000 + ep))
        for (xr, yr, _), (xf, _, _) in zip(E.loader("retain", batch_size=hp["retain_batch_size"], seed_offset=ep), f_iter):
            with torch.autocast(**AC):
                with torch.no_grad():
                    tr, tf = teacher(xr), teacher(xf)
                sr, sf = E.model(xr), E.model(xf)
                l_r = alpha * _kd(sr.float(), tr.float(), T) + gamma * F.cross_entropy(sr, yr)
                l_f = _kd(sf.float(), tf.float(), T)
                loss = l_r - lam * l_f
            E.step(loss, opt)
        E.end_epoch(ep, lr=opt.param_groups[0]["lr"], loss_retain=l_r.item(), kl_forget=l_f.item())


def random_relabel(E):
    opt = E.optimizer()
    g = torch.Generator(device=E.device).manual_seed(E.seed)
    for ep in range(1, E.hp["epochs"] + 1):
        E.train_mode()
        for x, y, _ in E.loader("forget", seed_offset=1000 + ep):
            y_rand = (y + torch.randint(1, NUM_CLASSES, y.shape, device=y.device, generator=g)) % NUM_CLASSES
            with torch.autocast(**AC):
                loss_f = F.cross_entropy(E.model(x), y_rand)
            E.step(loss_f, opt)
        if E.hp.get("use_retain", True):
            for x, y, _ in E.loader("retain", seed_offset=ep):
                with torch.autocast(**AC):
                    loss_r = F.cross_entropy(E.model(x), y)
                E.step(loss_r, opt)
        if E.end_epoch(ep, lr=opt.param_groups[0]["lr"], loss_forget=loss_f.item()):
            break


def _wrong_argmax(logits, y):
    logits = logits.float().clone()
    logits.scatter_(1, y[:, None], float("-inf"))
    return logits.argmax(1)


def neighbour_labels(original, x, y, eps, search="fgsm", steps=10):
    """Boundary Relabeling neighbour search on the frozen Original model (pixel space [0,1]).
    fgsm  : paper / repo: ONE sign-gradient step of size eps; label = Original's prediction on x' (if still y: best other class).
    pgd   : iterative sign-gradient steps of size eps/steps; each sample takes the FIRST wrong class it crosses into
            (= nearest decision boundary along the attack path); samples that never cross get the best other class at the end.
    logit2: no attack; the Original's highest-scoring wrong class on the clean image (nearest class in logit space)."""
    if search == "logit2":
        with torch.no_grad(), torch.autocast(**AC):
            return _wrong_argmax(original(x), y)
    if search == "fgsm":
        x_adv = x.detach().clone().requires_grad_(True)
        with torch.autocast(**AC):
            loss = F.cross_entropy(original(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = torch.round(torch.clamp(x + eps * grad.sign(), 0, 1) * 255) / 255
        with torch.no_grad(), torch.autocast(**AC):
            return _wrong_argmax(original(x_adv), y)
    alpha = eps / steps
    x_adv = x.detach().clone()
    label = torch.full_like(y, -1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.autocast(**AC):
            logits = original(x_adv)
            loss = F.cross_entropy(logits, y)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = torch.min(torch.max(x_adv.detach() + alpha * grad.sign(), x - eps), x + eps).clamp_(0, 1)
        with torch.no_grad(), torch.autocast(**AC):
            pred = original(x_adv).argmax(1)
        newly = (label < 0) & (pred != y)
        label[newly] = pred[newly]
        if (label >= 0).all():
            break
    if (label < 0).any():
        with torch.no_grad(), torch.autocast(**AC):
            fb = _wrong_argmax(original(x_adv), y)
        label = torch.where(label < 0, fb, label)
    return label


def boundary_relabel(E):
    hp = E.hp
    opt = E.optimizer()
    orig = E.original.eval()
    for p in orig.parameters():
        p.requires_grad_(False)
    for ep in range(1, hp["epochs"] + 1):
        E.train_mode()
        r_iter = retain_mixer(E, ep)
        for x, y, _ in E.loader("forget", seed_offset=1000 + ep):
            y_nb = neighbour_labels(orig, x, y, hp["eps"], hp.get("nb_search", "fgsm"), hp.get("nb_steps", 10))
            with torch.autocast(**AC):
                loss = F.cross_entropy(forward_forget(E, x, r_iter), y_nb)
            E.step(loss, opt)
        if hp.get("use_retain", False):  # improvement: retain fine-tuning pass (as in Random Relabeling) to protect retain data
            for x, y, _ in E.loader("retain", seed_offset=ep):
                with torch.autocast(**AC):
                    loss_r = F.cross_entropy(E.model(x), y)
                E.step(loss_r, opt)
        if E.end_epoch(ep, lr=opt.param_groups[0]["lr"], loss=loss.item(),
                       nb_label_hist=torch.bincount(y_nb, minlength=NUM_CLASSES).tolist().__str__().replace(",", ";")):
            break


METHODS = {"GA": gradient_ascent, "GAGD": ga_gd, "TS": teacher_student, "TSjoint": teacher_student_joint,
           "RLrand": random_relabel, "RLbnd": boundary_relabel}
