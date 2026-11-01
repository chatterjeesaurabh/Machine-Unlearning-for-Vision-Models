"""CIFAR-10 kept as uint8 tensors on the GPU, with GPU-side augmentation (RandomCrop(32, pad=4) + HFlip)."""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from mu import DATA_DIR

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_cifar10_numpy(root=os.path.join(DATA_DIR, "CIFAR10", "cifar-10-batches-py")):
    def _load(name):
        with open(os.path.join(root, name), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        return d["data"].reshape(-1, 3, 32, 32), np.asarray(d["labels"], dtype=np.int64)

    xs, ys = zip(*[_load(f"data_batch_{i}") for i in range(1, 6)])
    x_test, y_test = _load("test_batch")
    return np.concatenate(xs), np.concatenate(ys), x_test, y_test


class CIFAR10GPU:
    """Holds train (50k, original order) and test (10k) as uint8 NCHW tensors on `device`."""

    def __init__(self, device):
        x_tr, y_tr, x_te, y_te = load_cifar10_numpy()
        self.device = device
        self.x_train = torch.from_numpy(x_tr).to(device)
        self.y_train = torch.from_numpy(y_tr).to(device)
        self.x_test = torch.from_numpy(x_te).to(device)
        self.y_test = torch.from_numpy(y_te).to(device)

    def get(self, part):
        return (self.x_train, self.y_train) if part == "train" else (self.x_test, self.y_test)


def gpu_augment(x, generator=None, pad=4):
    """x: float [N,C,H,W] in [0,1]. Zero-padded random crop + random horizontal flip (per sample)."""
    n, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (n,), device=dev, generator=generator)
    j = torch.randint(0, 2 * pad + 1, (n,), device=dev, generator=generator)
    rows = (i[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (j[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    out = xp[torch.arange(n, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None], rows, cols]
    flip = torch.rand(n, device=dev, generator=generator) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


class GPULoader:
    """Iterates (x float[0,1], y, idx) over `indices` of a CIFAR10GPU part.

    labels: optional tensor overriding labels (same length as indices), e.g. for relabeling methods.
    """

    def __init__(self, data, part, indices, batch_size, shuffle=False, augment=False, seed=0, labels=None, drop_last=False):
        self.x, self.y = data.get(part)
        self.indices = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=data.device)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long, device=data.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.drop_last = drop_last
        self.gen = torch.Generator(device=data.device)
        self.gen.manual_seed(seed)

    def __len__(self):
        n = len(self.indices)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.indices)
        order = torch.randperm(n, device=self.indices.device, generator=self.gen) if self.shuffle else torch.arange(n, device=self.indices.device)
        for b in range(len(self)):
            pos = order[b * self.batch_size:(b + 1) * self.batch_size]
            idx = self.indices[pos]
            x = self.x[idx].float().div_(255.0)
            if self.augment:
                x = gpu_augment(x, self.gen)
            y = self.y[idx] if self.labels is None else self.labels[pos]
            yield x.contiguous(memory_format=torch.channels_last), y, idx
