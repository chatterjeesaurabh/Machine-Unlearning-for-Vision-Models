"""Run recorder: writes records/runs/<run_id>/{config.json, history.csv, metrics.json, per_sample.npz, stdout.log, summary.md}."""
import csv
import datetime
import json
import logging
import os
import socket
import sys

import numpy as np
import torch

from mu import RECORDS_DIR


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class RunRecorder:
    def __init__(self, run_id, config):
        self.run_id = run_id
        self.dir = os.path.join(RECORDS_DIR, "runs", run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.config = dict(config)
        self.config.update(run_id=run_id, status="running", start_time=_now(), host=socket.gethostname(),
                           cmd=" ".join(sys.argv), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
        self._write_json("config.json", self.config)
        self._hist_path = os.path.join(self.dir, "history.csv")
        if os.path.exists(self._hist_path):
            os.remove(self._hist_path)
        self._hist_fields = None
        self.log = logging.getLogger(run_id)
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(os.path.join(self.dir, "stdout.log"), mode="w"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            self.log.addHandler(h)

    def _write_json(self, name, obj):
        with open(os.path.join(self.dir, name), "w") as f:
            json.dump(obj, f, indent=1, default=_jsonable)

    def history(self, row):
        if self._hist_fields is None:
            self._hist_fields = list(row.keys())
            with open(self._hist_path, "w", newline="") as f:
                csv.DictWriter(f, self._hist_fields).writeheader()
        with open(self._hist_path, "a", newline="") as f:
            csv.DictWriter(f, self._hist_fields, extrasaction="ignore").writerow(row)

    def per_sample(self, **arrays):
        np.savez_compressed(os.path.join(self.dir, "per_sample.npz"), **arrays)

    def finish(self, metrics, status="done", notes=""):
        self.config.update(status=status, end_time=_now())
        self._write_json("config.json", self.config)
        self._write_json("metrics.json", metrics)
        lines = [f"# {self.run_id}", "", f"- status: {status}", f"- start: {self.config['start_time']}  end: {self.config['end_time']}",
                 "", "## Config", "```json", json.dumps(self.config, indent=1, default=_jsonable), "```", "", "## Metrics"]
        lines += [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in _flatten(metrics).items()]
        if notes:
            lines += ["", "## Notes", notes]
        with open(os.path.join(self.dir, "summary.md"), "w") as f:
            f.write("\n".join(lines) + "\n")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)) and len(v) > 12:
            out[key] = f"[list len {len(v)}]"
        else:
            out[key] = v
    return out
