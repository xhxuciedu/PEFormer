"""Exact E26 pairwise replication, isolated from historical E26 run outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import e26_adaptation_pilot as pilot


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[20260911, 20260912])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this declared replication; no silent CPU run")
    torch.set_num_threads(4)
    pilot.OUTDIR = pilot.C.CACHE / "adapt_followup_runs"
    pilot.OUTDIR.mkdir(parents=True, exist_ok=True)
    name = f"S_pairwise_s{args.seed}"
    for suffix in (".pt", ".json", ".provenance.json"):
        if (pilot.OUTDIR / (name + suffix)).exists():
            raise FileExistsError(f"Refusing to overwrite {name + suffix}")
    start = sorted((pilot.ROOT / "checkpoints").glob(f"{pilot.START_CKPT}_*"))[-1] / "best.pt"
    paths = [start, Path(pilot.__file__), Path(pilot.adapt.__file__),
             Path(pilot.AD.__file__), pilot.C.CACHE / "adaptation_partition_v2.parquet",
             pilot.C.OUT / "reserved_panel_v2.parquet"]
    provenance = {"seed": args.seed, "torch": torch.__version__,
                  "device": torch.cuda.get_device_name(),
                  "inputs_sha256": {str(p.relative_to(pilot.ROOT)): sha256(p) for p in paths},
                  "protocol": "Exact E26 pairwise hyperparameters; validation-only selection"}
    (pilot.OUTDIR / (name + ".provenance.json")).write_text(json.dumps(provenance, indent=2))
    sys.argv = [pilot.__file__, "--arm", "S", "--sel-loss", "pairwise", "--tag", "pairwise",
                "--seed", str(args.seed)]
    pilot.main()


if __name__ == "__main__":
    main()
