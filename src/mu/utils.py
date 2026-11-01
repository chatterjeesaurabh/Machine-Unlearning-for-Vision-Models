import fcntl
import os
import random
import sys

import numpy as np
import torch

from mu import ROOT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_LOCK_FH = None


def acquire_gpu_lock():
    """Project-wide exclusive lock so only ONE GPU job of this project runs at a time. Lives on LOCAL disk (/tmp): on NFS,
    blocked waiters only re-poll periodically and get starved by new requesters. Released automatically at process exit."""
    global _LOCK_FH
    if _LOCK_FH is not None or os.environ.get("MU_LOCK_HELD") == "1":  # parent holds it, e.g. scripts/gpu_batch.sh
        return
    path = "/tmp/mu_cs6886_gpu.lock"
    _LOCK_FH = open(path, "a+")
    try:
        fcntl.flock(_LOCK_FH, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[gpu lock] waiting for {path} ...", flush=True)
        fcntl.flock(_LOCK_FH, fcntl.LOCK_EX)
    _LOCK_FH.seek(0)
    _LOCK_FH.truncate()
    _LOCK_FH.write(f"pid {os.getpid()} {' '.join(sys.argv)}\n")
    _LOCK_FH.flush()


def setup_device(gpu, mem_frac):
    """Acquire the project GPU lock, select GPU and hard-cap this process's caching allocator to mem_frac of device memory."""
    acquire_gpu_lock()
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(mem_frac, device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device


def run_with_oom_retry(main, max_tries=120, wait_s=60):
    """GPUs are shared with other users: on CUDA OOM (caused by other processes), free memory, wait and re-run `main` from scratch."""
    import gc
    import time
    for attempt in range(1, max_tries + 1):
        try:
            return main()
        except torch.OutOfMemoryError as e:
            print(f"[oom retry] attempt {attempt}/{max_tries}: {str(e)[:160]} -> waiting {wait_s}s", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(wait_s)
    raise RuntimeError("giving up after repeated CUDA OOM")


def save_ckpt(model, path, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_ckpt(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    return ck
