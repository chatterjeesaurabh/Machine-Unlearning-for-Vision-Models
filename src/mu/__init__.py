import os

ROOT = os.environ.get("MU_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA_DIR = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
CKPT_DIR = os.path.join(ROOT, "outputs", "checkpoints")
RECORDS_DIR = os.path.join(ROOT, "records")
