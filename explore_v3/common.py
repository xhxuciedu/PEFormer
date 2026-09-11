"""V3 paths and immutable artifact helpers; never write to v2."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "explore_v3"
CACHE = OUT / "cache"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "explore_v2"))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2, default=lambda x: x.item())

def provenance(paths):
    return {str(Path(p).relative_to(ROOT)): sha256(p) for p in paths}
